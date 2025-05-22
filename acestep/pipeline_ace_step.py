"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

import json
import math
import os
import random
import re
import time

import librosa
import torch
import torch.nn as nn
import torchaudio

from acestep.apg_guidance import (
    apg_forward,
    MomentumBuffer,
    cfg_forward,
    cfg_zero_star,
    cfg_double_condition_forward,
)
from acestep.cpu_offload import cpu_offload
from acestep.language_segmentation import LangSegment, language_filters
from acestep.models.ace_step_transformer import ACEStepTransformer2DModel
from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from acestep.schedulers.scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler
from acestep.schedulers.scheduling_flow_match_pingpong import FlowMatchPingPongScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download, snapshot_download
from loguru import logger
from tqdm import tqdm
from transformers import UMT5EncoderModel, AutoTokenizer


# from diffusers.pipelines.pipeline_utils import DiffusionPipeline


torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


SUPPORT_LANGUAGES = {
    "en": 259,
    "de": 260,
    "fr": 262,
    "es": 284,
    "it": 285,
    "pt": 286,
    "pl": 294,
    "tr": 295,
    "ru": 267,
    "cs": 293,
    "nl": 297,
    "ar": 5022,
    "zh": 5023,
    "ja": 5412,
    "hu": 5753,
    "ko": 6152,
    "hi": 6680,
}

structure_pattern = re.compile(r"\[.*?\]")


def ensure_directory_exists(directory):
    directory = str(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)


REPO_ID = "ACE-Step/ACE-Step-v1-3.5B"
REPO_ID_QUANT = REPO_ID + "-q4-K-M" # ??? update this i guess


# class ACEStepPipeline(DiffusionPipeline):
class ACEStepPipeline:
    def __init__(
        self,
        checkpoint_dir=None,
        device_id=0,
        dtype="bfloat16",
        text_encoder_checkpoint_path=None,
        persistent_storage_path=None,
        torch_compile=False,
        cpu_offload=False,
        quantized=False,
        overlapped_decode=False,
        **kwargs,
    ):
        if not checkpoint_dir:
            if persistent_storage_path is None:
                checkpoint_dir = os.path.join(os.path.expanduser("~"), ".cache/ace-step/checkpoints")
            else:
                checkpoint_dir = os.path.join(persistent_storage_path, "checkpoints")
        ensure_directory_exists(checkpoint_dir)

        self.checkpoint_dir = checkpoint_dir
        self.lora_path = "none"

        self.lora_weight = 1
        self.device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")
        if self.device.type == "cpu" and torch.backends.mps.is_available():
            self.device = torch.device("mps")

        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        if self.device.type == "mps":
            self.dtype = torch.float32
        if 'ACE_PIPELINE_DTYPE' in os.environ and len(os.environ['ACE_PIPELINE_DTYPE']):
            self.dtype = getattr(torch, os.environ['ACE_PIPELINE_DTYPE'])

        self.loaded = False
        self.torch_compile = torch_compile
        self.cpu_offload = cpu_offload
        self.quantized = quantized
        self.overlapped_decode = overlapped_decode

    def cleanup_memory(self):
        """Clean up GPU and CPU memory to prevent VRAM overflow during multiple generations."""
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            # Log memory usage if in verbose mode
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info("GPU Memory: {:.2f}GB allocated, {:.2f}GB reserved", allocated, reserved)

        # Collect Python garbage
        import gc
        gc.collect()

    def get_checkpoint_path(self, checkpoint_dir, repo):
        checkpoint_dir_models = None
        
        if checkpoint_dir is not None:
            required_dirs = ["music_dcae_f8c8", "music_vocoder", "ace_step_transformer", "umt5-base"]
            all_dirs_exist = True
            for dir_name in required_dirs:
                dir_path = os.path.join(checkpoint_dir, dir_name)
                if not os.path.exists(dir_path):
                    all_dirs_exist = False
                    break
            
            if all_dirs_exist:
                logger.info("Load models from: {}", checkpoint_dir)
                checkpoint_dir_models = checkpoint_dir
        
        if checkpoint_dir_models is None:
            if checkpoint_dir is None:
                logger.info("Download models from Hugging Face: {}", repo)
                checkpoint_dir_models = snapshot_download(repo)
            else:
                logger.info("Download models from Hugging Face: {}, cache to: {}", repo, checkpoint_dir)
                checkpoint_dir_models = snapshot_download(repo, cache_dir=checkpoint_dir)

        return checkpoint_dir_models

    def load_checkpoint(self, checkpoint_dir=None, export_quantized_weights=False):
        checkpoint_dir = self.get_checkpoint_path(checkpoint_dir, REPO_ID)
        dcae_checkpoint_path = os.path.join(checkpoint_dir, "music_dcae_f8c8")
        vocoder_checkpoint_path = os.path.join(checkpoint_dir, "music_vocoder")
        ace_step_checkpoint_path = os.path.join(checkpoint_dir, "ace_step_transformer")
        text_encoder_checkpoint_path = os.path.join(checkpoint_dir, "umt5-base")

        # ACE
        self.ace_step_transformer = ACEStepTransformer2DModel.from_pretrained(ace_step_checkpoint_path, torch_dtype=self.dtype)
        if self.cpu_offload:
            self.ace_step_transformer = self.ace_step_transformer.to("cpu").eval().to(self.dtype)
        else:
            self.ace_step_transformer = self.ace_step_transformer.to(self.device).eval().to(self.dtype)
        if self.torch_compile:
            self.ace_step_transformer = torch.compile(self.ace_step_transformer)

        # DCAE
        self.music_dcae = MusicDCAE(dcae_checkpoint_path, vocoder_checkpoint_path)
        if self.cpu_offload:  # might be redundant
            self.music_dcae = self.music_dcae.to("cpu").eval().to(self.dtype)
        else:
            self.music_dcae = self.music_dcae.to(self.device).eval().to(self.dtype)
        if self.torch_compile:
            self.music_dcae = torch.compile(self.music_dcae)

        # UMT5
        text_encoder_model = UMT5EncoderModel.from_pretrained(text_encoder_checkpoint_path, torch_dtype=self.dtype)
        if self.cpu_offload:
            text_encoder_model = text_encoder_model.to("cpu").eval().to(self.dtype)
        else:
            text_encoder_model = text_encoder_model.to(self.device).eval().to(self.dtype)
        text_encoder_model.requires_grad_(False)
        self.text_encoder_model = text_encoder_model
        if self.torch_compile:
            self.text_encoder_model = torch.compile(self.text_encoder_model)

        # Lyric Tokenizer
        lang_segment = LangSegment()
        lang_segment.setfilters(language_filters.default)
        self.lang_segment = lang_segment
        self.lyric_tokenizer = VoiceBpeTokenizer()

        # Text Tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_checkpoint_path)
        self.loaded = True

        # compile
        if self.torch_compile:
            if export_quantized_weights:
                from torchao.quantization import quantize_, Int4WeightOnlyConfig

                group_size = 128
                use_hqq = True
                quantize_(self.ace_step_transformer, Int4WeightOnlyConfig(group_size=group_size, use_hqq=use_hqq))
                quantize_(self.text_encoder_model, Int4WeightOnlyConfig(group_size=group_size, use_hqq=use_hqq))

                # save quantized weights
                pth = os.path.join(ace_step_checkpoint_path, "diffusion_pytorch_model_int4wo.bin")
                torch.save(self.ace_step_transformer.state_dict(), pth)
                print("Quantized Weights Saved to: ", pth)

                pth = os.path.join(text_encoder_checkpoint_path, "pytorch_model_int4wo.bin")
                torch.save(self.text_encoder_model.state_dict(), pth)
                print("Quantized Weights Saved to: ", pth)


    def load_quantized_checkpoint(self, checkpoint_dir=None):
        checkpoint_dir = self.get_checkpoint_path(checkpoint_dir, REPO_ID_QUANT)
        dcae_checkpoint_path = os.path.join(checkpoint_dir, "music_dcae_f8c8")
        vocoder_checkpoint_path = os.path.join(checkpoint_dir, "music_vocoder")
        ace_step_checkpoint_path = os.path.join(checkpoint_dir, "ace_step_transformer")
        text_encoder_checkpoint_path = os.path.join(checkpoint_dir, "umt5-base")

        self.music_dcae = MusicDCAE(dcae_checkpoint_path, vocoder_checkpoint_path)
        if self.cpu_offload:
            self.music_dcae.eval().to(self.dtype).to(self.device)
        else:
            self.music_dcae.eval().to(self.dtype).to('cpu')
        self.music_dcae = torch.compile(self.music_dcae)


        self.ace_step_transformer = ACEStepTransformer2DModel.from_pretrained(ace_step_checkpoint_path)
        self.ace_step_transformer.eval().to(self.dtype).to('cpu')
        self.ace_step_transformer = torch.compile(self.ace_step_transformer)
        pth = os.path.join(ace_step_checkpoint_path, "diffusion_pytorch_model_int4wo.bin")
        state_dict = torch.load(pth, map_location=self.device)
        self.ace_step_transformer.load_state_dict(state_dict, assign=True)
        self.ace_step_transformer.torchao_quantized = True

        self.text_encoder_model = UMT5EncoderModel.from_pretrained(text_encoder_checkpoint_path)
        self.text_encoder_model.eval().to(self.dtype).to('cpu')
        self.text_encoder_model = torch.compile(self.text_encoder_model)
        pth = os.path.join(text_encoder_checkpoint_path, "pytorch_model_int4wo.bin")
        state_dict = torch.load(pth, map_location=self.device)
        self.text_encoder_model.load_state_dict(state_dict, assign=True)
        self.text_encoder_model.torchao_quantized = True

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_checkpoint_path)

        lang_segment = LangSegment()
        lang_segment.setfilters(language_filters.default)
        self.lang_segment = lang_segment
        self.lyric_tokenizer = VoiceBpeTokenizer()

        self.loaded = True

    @cpu_offload("text_encoder_model")
    def get_text_embeddings(self, texts, text_max_length=256):
        inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=text_max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        if self.text_encoder_model.device != self.device:
            self.text_encoder_model.to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        return last_hidden_states, attention_mask

    @cpu_offload("text_encoder_model")
    def get_text_embeddings_null(self, texts, text_max_length=256, tau=0.01, l_min=8, l_max=10):
        inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=text_max_length)

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.text_encoder_model.device != self.device:
            self.text_encoder_model.to(self.device)

        def forward_with_temperature(inputs, tau=0.01, l_min=8, l_max=10):
            handlers = []

            def hook(module, input, output):
                output[:] *= tau
                return output

            for i in range(l_min, l_max):
                block = self.text_encoder_model.encoder.block[i]
                layer = block.layer[0]
                query = layer.SelfAttention.q
                handler = query.register_forward_hook(hook)
                handlers.append(handler)

            with torch.no_grad():
                outputs = self.text_encoder_model(**inputs)
                last_hidden_states = outputs.last_hidden_state

            for hook in handlers:
                hook.remove()

            return last_hidden_states

        last_hidden_states = forward_with_temperature(inputs, tau, l_min, l_max)

        return last_hidden_states

    def set_seeds(self, batch_size, manual_seeds=None):
        processed_input_seeds = None
        if manual_seeds is not None:
            if isinstance(manual_seeds, str):
                if "," in manual_seeds:
                    processed_input_seeds = list(map(int, manual_seeds.split(",")))
                elif manual_seeds.isdigit():
                    processed_input_seeds = [int(manual_seeds)]
            elif isinstance(manual_seeds, list) and all(isinstance(s, int) for s in manual_seeds):
                if len(manual_seeds) > 0:
                    processed_input_seeds = manual_seeds
            elif isinstance(manual_seeds, int):
                processed_input_seeds = [manual_seeds]

        random_generators = [torch.Generator(device=self.device) for _ in range(batch_size)]
        actual_seeds = []

        for i in range(batch_size):
            current_seed_for_generator = None
            if processed_input_seeds is None:
                current_seed_for_generator = torch.randint(0, 2**32, (1,)).item()
            else:
                if i < len(processed_input_seeds):
                    current_seed_for_generator = processed_input_seeds[i]
                else:
                    current_seed_for_generator = processed_input_seeds[-1]
            random_generators[i].manual_seed(current_seed_for_generator)
            actual_seeds.append(current_seed_for_generator)

        return random_generators, actual_seeds

    def get_lang(self, text):
        language = "en"
        try:
            _ = self.lang_segment.getTexts(text)
            langCounts = self.lang_segment.getCounts()
            language = langCounts[0][0]
            if len(langCounts) > 1 and language == "en":
                language = langCounts[1][0]
        except Exception as err:
            language = "en"

        return language

    def tokenize_lyrics(self, lyrics, debug=False):
        lines = lyrics.split("\n")
        lyric_token_idx = [261]
        for line in lines:
            line = line.strip()
            if not line:
                lyric_token_idx += [2]
                continue

            lang = self.get_lang(line)

            if lang not in SUPPORT_LANGUAGES:
                lang = "en"
            if "zh" in lang:
                lang = "zh"
            if "spa" in lang:
                lang = "es"

            try:
                if structure_pattern.match(line):
                    token_idx = self.lyric_tokenizer.encode(line, "en")
                else:
                    token_idx = self.lyric_tokenizer.encode(line, lang)
                if debug:
                    toks = self.lyric_tokenizer.batch_decode([[tok_id] for tok_id in token_idx])
                    logger.info("debbug {} --> {} --> {}", line, lang, toks)
                lyric_token_idx = lyric_token_idx + token_idx + [2]
            except Exception as e:
                print("tokenize error", e, "for line", line, "major_language", lang)

        return lyric_token_idx

    @cpu_offload("ace_step_transformer")
    def calc_v(
        self,
        zt_src,
        zt_tar,
        t,
        encoder_text_hidden_states,
        text_attention_mask,
        target_encoder_text_hidden_states,
        target_text_attention_mask,
        speaker_embds,
        target_speaker_embeds,
        lyric_token_ids,
        lyric_mask,
        target_lyric_token_ids,
        target_lyric_mask,
        do_classifier_free_guidance=False,
        guidance_scale=1.0,
        target_guidance_scale=1.0,
        attention_mask=None,
        momentum_buffer=None,
        momentum_buffer_tar=None,
        return_src_pred=True,
        cfg_type="apg", # ATTN: NEVER SET OR USED BY OTHER FUNCS, OLD?
    ):
        noise_pred_src = None
        if return_src_pred:
            src_latent_model_input = torch.cat([zt_src, zt_src]) if do_classifier_free_guidance else zt_src
            timestep = t.expand(src_latent_model_input.shape[0])

            # source
            noise_pred_src = self.ace_step_transformer(src_latent_model_input, attention_mask, encoder_text_hidden_states, text_attention_mask, speaker_embds, lyric_token_ids, lyric_mask, timestep).sample

            if do_classifier_free_guidance:
                noise_pred_with_cond_src, noise_pred_uncond_src = noise_pred_src.chunk(2)
                if cfg_type == "apg":
                    noise_pred_src = apg_forward(noise_pred_with_cond_src, noise_pred_uncond_src, guidance_scale, momentum_buffer)
                elif cfg_type == "cfg":
                    noise_pred_src = cfg_forward(noise_pred_with_cond_src, noise_pred_uncond_src, guidance_scale)

        tar_latent_model_input = torch.cat([zt_tar, zt_tar]) if do_classifier_free_guidance else zt_tar
        timestep = t.expand(tar_latent_model_input.shape[0])

        # target
        noise_pred_tar = self.ace_step_transformer(tar_latent_model_input, attention_mask, target_encoder_text_hidden_states, target_text_attention_mask, target_speaker_embeds, target_lyric_token_ids, target_lyric_mask, timestep).sample

        if do_classifier_free_guidance:
            noise_pred_with_cond_tar, noise_pred_uncond_tar = noise_pred_tar.chunk(2)
            if cfg_type == "apg":
                noise_pred_tar = apg_forward(noise_pred_with_cond_tar, noise_pred_uncond_tar, target_guidance_scale, momentum_buffer_tar)
            elif cfg_type == "cfg":
                noise_pred_tar = cfg_forward(noise_pred_with_cond_tar, noise_pred_uncond_tar, target_guidance_scale)

        return noise_pred_src, noise_pred_tar

    @torch.no_grad()
    def flowedit_diffusion_process(
        self,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        target_encoder_text_hidden_states,
        target_text_attention_mask,
        target_speaker_embeds,
        target_lyric_token_ids,
        target_lyric_mask,
        src_latents,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        n_min=0,
        n_max=1.0,
        n_avg=1,
        scheduler_type="euler",
    ):

        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        target_guidance_scale = guidance_scale
        bsz = encoder_text_hidden_states.shape[0]

        scheduler = FlowMatchEulerDiscreteScheduler(1000, 3.0)

        T_steps = infer_steps
        frame_length = src_latents.shape[-1]
        attention_mask = torch.ones(bsz, frame_length, device=self.device, dtype=self.dtype)

        timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, self.device)

        if do_classifier_free_guidance:
            attention_mask = torch.cat([attention_mask] * 2, dim=0)

            cat_zero_lambda = lambda x: torch.cat([x, torch.zeros_like(x)], 0)
            cat_repl_lambda = lambda x: torch.cat([x, x], 0)

            encoder_text_hidden_states = cat_zero_lambda(encoder_text_hidden_states)
            text_attention_mask = cat_repl_lambda(text_attention_mask)

            target_encoder_text_hidden_states = cat_zero_lambda(target_encoder_text_hidden_states)
            target_text_attention_mask = cat_repl_lambda(target_text_attention_mask)

            speaker_embds = cat_zero_lambda(speaker_embds)
            target_speaker_embeds = cat_zero_lambda(target_speaker_embeds)

            lyric_token_ids = cat_zero_lambda(lyric_token_ids)
            lyric_mask = cat_zero_lambda(lyric_mask)

            target_lyric_token_ids = cat_zero_lambda(target_lyric_token_ids)
            target_lyric_mask = cat_zero_lambda(target_lyric_mask)

        momentum_buffer = MomentumBuffer()
        momentum_buffer_tar = MomentumBuffer()
        x_src = src_latents
        zt_edit = x_src.clone()
        xt_tar = None
        n_min = int(infer_steps * n_min)
        n_max = int(infer_steps * n_max)

        logger.info("flowedit start from {} to {}", n_min, n_max)

        for i, t in tqdm(enumerate(timesteps), total=T_steps):
            if i < n_min:
                continue

            t_i = t / 1000

            if i + 1 < len(timesteps):
                t_im1 = (timesteps[i + 1]) / 1000
            else:
                t_im1 = torch.zeros_like(t_i).to(self.device)

            if i < n_max:
                # Calculate the average of the V predictions
                V_delta_avg = torch.zeros_like(x_src)
                for k in range(n_avg):
                    fwd_noise = randn_tensor(x_src.shape, random_generators, self.device, self.dtype)

                    zt_src = (1 - t_i) * x_src + (t_i) * fwd_noise

                    zt_tar = zt_edit + zt_src - x_src

                    Vt_src, Vt_tar = self.calc_v(zt_src, zt_tar, t, encoder_text_hidden_states, text_attention_mask, target_encoder_text_hidden_states, target_text_attention_mask, speaker_embds, target_speaker_embeds, lyric_token_ids, lyric_mask, target_lyric_token_ids, target_lyric_mask, do_classifier_free_guidance, guidance_scale, target_guidance_scale, attention_mask, momentum_buffer)
                    V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src) # - (hfg - 1) * (x_src)

                zt_edit = zt_edit.to(torch.float32) # arbitrary, should be settable for compatibility
                if scheduler_type != "pingpong":
                    # propagate direct ODE
                    zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
                    zt_edit = zt_edit.to(self.dtype)
                else:
                    # propagate pingpong SDE
                    zt_edit_denoised = zt_edit - t_i * V_delta_avg
                    noise = torch.empty_like(zt_edit).normal_(generator=random_generators[0] if random_generators else None)
                    prev_sample = (1 - t_im1) * zt_edit_denoised + t_im1 * noise

            else:  # i >= T_steps-n_min # regular sampling for last n_min steps
                if i == n_max:
                    fwd_noise = randn_tensor(x_src.shape, random_generators, self.device, self.dtype)
                    scheduler._init_step_index(t)
                    sigma = scheduler.sigmas[scheduler.step_index]
                    xt_src = sigma * fwd_noise + (1.0 - sigma) * x_src
                    xt_tar = zt_edit + xt_src - x_src

                _, Vt_tar = self.calc_v(None, xt_tar, t, encoder_text_hidden_states, text_attention_mask, target_encoder_text_hidden_states, target_text_attention_mask, speaker_embds, target_speaker_embeds, lyric_token_ids, lyric_mask, target_lyric_token_ids, target_lyric_mask, do_classifier_free_guidance, guidance_scale, target_guidance_scale, attention_mask, momentum_buffer_tar=momentum_buffer_tar, return_src_pred=False)

                xt_tar = xt_tar.to(torch.float32)
                if scheduler_type != "pingpong":
                    prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
                    prev_sample = prev_sample.to(self.dtype)
                    xt_tar = prev_sample
                else:
                    prev_sample = xt_tar - t_i * Vt_tar
                    noise = torch.empty_like(zt_edit).normal_(generator=random_generators[0] if random_generators else None)
                    prev_sample = (1 - t_im1) * prev_sample + t_im1 * noise
                    xt_tar = prev_sample

        target_latents = zt_edit if xt_tar is None else xt_tar

        return target_latents

    def add_latents_noise(self, gt_latents, sigma_max, noise, scheduler_type, infer_steps,):
        bsz = gt_latents.shape[0]
        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(1000, 3.0, sigma_max=sigma_max)
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(1000, 3.0, sigma_max=sigma_max)
        elif scheduler_type == "pingpong":
            scheduler = FlowMatchPingPongScheduler(1000, 3.0, sigma_max=sigma_max)

        infer_steps = int(sigma_max * infer_steps)
        timesteps, num_inference_steps = retrieve_timesteps(scheduler, infer_steps, self.device)

        noisy_image = gt_latents * (1 - scheduler.sigma_max) + noise * scheduler.sigma_max
        logger.info(f"{scheduler.sigma_min=} {scheduler.sigma_max=} {timesteps=} {num_inference_steps=}")

        return noisy_image, timesteps, scheduler, num_inference_steps

    @cpu_offload("ace_step_transformer")
    @torch.no_grad()
    def text2music_diffusion_process(
        self,
        duration,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
        scheduler_type="euler",
        cfg_type="apg",
        guidance_interval=0.5,
        guidance_interval_decay=1.0,
        min_guidance_scale=3.0,
        oss_steps=[],
        encoder_text_hidden_states_null=None,
        use_erg_lyric=False,
        use_erg_diffusion=False,
        retake_random_generators=None,
        retake_variance=0.5,
        add_retake_noise=False,
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
        repaint_start=0,
        repaint_end=0,
        src_latents=None,
        audio2audio_enable=False,
        ref_audio_strength=0.5,
        ref_latents=None,
        zero_steps=1,
        use_zero_init=True,
    ):

        logger.info("cfg_type: {}, guidance_scale: {}, omega_scale: {}", cfg_type, guidance_scale, omega_scale)
        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        do_double_condition_guidance = False
        if guidance_scale_text is not None and guidance_scale_text > 1.0 and guidance_scale_lyric is not None and guidance_scale_lyric > 1.0:
            do_double_condition_guidance = True
            logger.info("do_double_condition_guidance: {}, guidance_scale_text: {}, guidance_scale_lyric: {}", do_double_condition_guidance, guidance_scale_text, guidance_scale_lyric)

        bsz = encoder_text_hidden_states.shape[0]

        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(1000, 3.0)
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(1000, 3.0)
        elif scheduler_type == "pingpong":
            scheduler = FlowMatchPingPongScheduler(1000, 3.0)

        frame_length = int(duration * 44100 / 512 / 8)
        if src_latents is not None:
            frame_length = src_latents.shape[-1]
        
        if ref_latents is not None:
            frame_length = ref_latents.shape[-1]

        if len(oss_steps) > 0:
            infer_steps = max(oss_steps)
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, infer_steps, self.device)
            new_timesteps = torch.zeros(len(oss_steps), dtype=self.dtype, device=self.device)
            for idx in range(len(oss_steps)):
                new_timesteps[idx] = timesteps[oss_steps[idx] - 1]
            num_inference_steps = len(oss_steps)
            sigmas = (new_timesteps / 1000).float().cpu().numpy()
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, self.device, sigmas=sigmas)
            logger.info("oss_steps: {}, num_inference_steps: {} after remapping to timesteps {}", oss_steps, num_inference_steps, timesteps)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, infer_steps, self.device)

        target_latents = randn_tensor((bsz, 8, 16, frame_length), random_generators, self.device, self.dtype)

        is_repaint = False
        is_extend = False

        if add_retake_noise:
            n_min = int(infer_steps * (1 - retake_variance))
            retake_variance = torch.tensor(retake_variance * math.pi / 2).to(self.device).to(self.dtype)
            retake_latents = randn_tensor((bsz, 8, 16, frame_length), retake_random_generators, self.device, self.dtype)
            repaint_start_frame = int(repaint_start * 44100 / 512 / 8)
            repaint_end_frame = int(repaint_end * 44100 / 512 / 8)
            x0 = src_latents
            # retake
            is_repaint = repaint_end_frame - repaint_start_frame != frame_length

            is_extend = (repaint_start_frame < 0) or (repaint_end_frame > frame_length)
            if is_extend:
                is_repaint = True

            # TODO: train a mask aware repainting controlnet
            # to make sure mean = 0, std = 1
            if not is_repaint:
                target_latents = torch.cos(retake_variance) * target_latents + torch.sin(retake_variance) * retake_latents
            elif not is_extend:
                # if repaint_end_frame
                repaint_mask = torch.zeros((bsz, 8, 16, frame_length), device=self.device, dtype=self.dtype)
                repaint_mask[:, :, :, repaint_start_frame:repaint_end_frame] = 1.0
                repaint_noise = torch.cos(retake_variance) * target_latents + torch.sin(retake_variance) * retake_latents
                repaint_noise = torch.where(repaint_mask == 1.0, repaint_noise, target_latents)
                zt_edit = x0.clone()
                z0 = repaint_noise
            elif is_extend:
                to_right_pad_gt_latents = None
                to_left_pad_gt_latents = None
                gt_latents = src_latents
                src_latents_length = gt_latents.shape[-1]
                max_infer_fame_length = int(240 * 44100 / 512 / 8)
                left_pad_frame_length = 0
                right_pad_frame_length = 0
                right_trim_length = 0
                left_trim_length = 0

                if repaint_start_frame < 0:
                    left_pad_frame_length = abs(repaint_start_frame)
                    frame_length = left_pad_frame_length + gt_latents.shape[-1]
                    extend_gt_latents = torch.nn.functional.pad(gt_latents, (left_pad_frame_length, 0), "constant", 0)

                    if frame_length > max_infer_fame_length:
                        right_trim_length = frame_length - max_infer_fame_length
                        extend_gt_latents = extend_gt_latents[:, :, :, :max_infer_fame_length]
                        to_right_pad_gt_latents = extend_gt_latents[:, :, :, -right_trim_length:]
                        frame_length = max_infer_fame_length

                    repaint_start_frame = 0
                    gt_latents = extend_gt_latents

                if repaint_end_frame > src_latents_length:
                    right_pad_frame_length = repaint_end_frame - gt_latents.shape[-1]
                    frame_length = gt_latents.shape[-1] + right_pad_frame_length
                    extend_gt_latents = torch.nn.functional.pad(gt_latents, (0, right_pad_frame_length), "constant", 0)

                    if frame_length > max_infer_fame_length:
                        left_trim_length = frame_length - max_infer_fame_length
                        extend_gt_latents = extend_gt_latents[:, :, :, -max_infer_fame_length:]
                        to_left_pad_gt_latents = extend_gt_latents[:, :, :, :left_trim_length]
                        frame_length = max_infer_fame_length

                    repaint_end_frame = frame_length
                    gt_latents = extend_gt_latents

                repaint_mask = torch.zeros((bsz, 8, 16, frame_length), device=self.device, dtype=self.dtype)

                if left_pad_frame_length > 0:
                    repaint_mask[:, :, :, :left_pad_frame_length] = 1.0
                if right_pad_frame_length > 0:
                    repaint_mask[:, :, :, -right_pad_frame_length:] = 1.0

                x0 = gt_latents
                padd_list = []

                if left_pad_frame_length > 0:
                    padd_list.append(retake_latents[:, :, :, :left_pad_frame_length])

                padd_list.append(target_latents[:,:,:, left_trim_length : target_latents.shape[-1] - right_trim_length])

                if right_pad_frame_length > 0:
                    padd_list.append(retake_latents[:, :, :, -right_pad_frame_length:])

                target_latents = torch.cat(padd_list, dim=-1)
                assert target_latents.shape[-1] == x0.shape[-1], f"{target_latents.shape=} {x0.shape=}"
                zt_edit = x0.clone()
                z0 = target_latents

        if audio2audio_enable and ref_latents is not None:
            logger.info("audio2audio_enable: {}, ref_latents: {}", audio2audio_enable, ref_latents.shape)
            target_latents, timesteps, scheduler, num_inference_steps = self.add_latents_noise(ref_latents, (1-ref_audio_strength), target_latents, scheduler_type, infer_steps)

        attention_mask = torch.ones(bsz, frame_length, device=self.device, dtype=self.dtype)

        # guidance interval
        start_idx = int(num_inference_steps * ((1 - guidance_interval) / 2))
        end_idx = int(num_inference_steps * (guidance_interval / 2 + 0.5))
        logger.info("start_idx: {}, end_idx: {}, num_inference_steps: {}", start_idx, end_idx, num_inference_steps)

        momentum_buffer = MomentumBuffer()

        def forward_encoder_with_temperature(self, inputs, tau=0.01, l_min=4, l_max=6):
            handlers = []
            def hook(module, input, output):
                output[:] *= tau
                return output

            for i in range(l_min, l_max):
                encoder = self.ace_step_transformer.lyric_encoder.encoders[i]
                query = encoder.self_attn.linear_q
                handler = query.register_forward_hook(hook)
                handlers.append(handler)

            encoder_hidden_states, encoder_hidden_mask = self.ace_step_transformer.encode(**inputs)
            for hook in handlers:
                hook.remove()
            return encoder_hidden_states

        # P(speaker, text, lyric)
        encoder_hidden_states, encoder_hidden_mask = self.ace_step_transformer.encode(encoder_text_hidden_states, text_attention_mask, speaker_embds, lyric_token_ids, lyric_mask)

        if use_erg_lyric:
            # P(null_speaker, text_weaker, lyric_weaker)
            inputs = {
                "encoder_text_hidden_states": encoder_text_hidden_states_null if encoder_text_hidden_states_null is not None else torch.zeros_like(encoder_text_hidden_states),
                "text_attention_mask": text_attention_mask,
                "speaker_embeds": torch.zeros_like(speaker_embds),
                "lyric_token_idx": lyric_token_ids,
                "lyric_mask": lyric_mask,
            }
            encoder_hidden_states_null = forward_encoder_with_temperature(self, inputs)
        else:
            # P(null_speaker, null_text, null_lyric)
            encoder_hidden_states_null, _ = self.ace_step_transformer.encode(torch.zeros_like(encoder_text_hidden_states), text_attention_mask, torch.zeros_like(speaker_embds), torch.zeros_like(lyric_token_ids), lyric_mask)

        encoder_hidden_states_no_lyric = None
        if do_double_condition_guidance:
            if use_erg_lyric:
                # P(null_speaker, text, lyric_weaker)
                inputs = {
                    "encoder_text_hidden_states": encoder_text_hidden_states,
                    "text_attention_mask": text_attention_mask,
                    "speaker_embeds": torch.zeros_like(speaker_embds),
                    "lyric_token_idx": lyric_token_ids,
                    "lyric_mask": lyric_mask,
                }
                encoder_hidden_states_no_lyric = forward_encoder_with_temperature(self, inputs)
            else:
                # P(null_speaker, text, no_lyric)
                encoder_hidden_states_no_lyric, _ = self.ace_step_transformer.encode(encoder_text_hidden_states, text_attention_mask, torch.zeros_like(speaker_embds), torch.zeros_like(lyric_token_ids), lyric_mask)

        def forward_diffusion_with_temperature(self, hidden_states, timestep, inputs, tau=0.01, l_min=15, l_max=20):
            handlers = []
            def hook(module, input, output):
                output[:] *= tau
                return output

            for i in range(l_min, l_max):
                handler = self.ace_step_transformer.transformer_blocks[i].attn.to_q.register_forward_hook(hook)
                handlers.append(handler)
                handler = self.ace_step_transformer.transformer_blocks[i].cross_attn.to_q.register_forward_hook(hook)
                handlers.append(handler)

            sample = self.ace_step_transformer.decode(hidden_states=hidden_states, timestep=timestep, **inputs).sample
            for hook in handlers:
                hook.remove()
            return sample

        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):

            if is_repaint:
                if i < n_min:
                    continue
                elif i == n_min:
                    t_i = t / 1000
                    zt_src = (1 - t_i) * x0 + (t_i) * z0
                    target_latents = zt_edit + zt_src - x0
                    logger.info("repaint start from {} add {} level of noise", n_min, t_i)

            # expand the latents if we are doing classifier free guidance
            latents = target_latents

            is_in_guidance_interval = start_idx <= i < end_idx
            if is_in_guidance_interval and do_classifier_free_guidance:
                # compute current guidance scale
                if guidance_interval_decay > 0:
                    # Linearly interpolate to calculate the current guidance scale
                    progress = (i - start_idx) / (end_idx - start_idx - 1)  # 归一化到[0,1]
                    current_guidance_scale = guidance_scale - (guidance_scale - min_guidance_scale) * progress * guidance_interval_decay
                else:
                    current_guidance_scale = guidance_scale

                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                output_length = latent_model_input.shape[-1]
                # P(x|speaker, text, lyric)
                noise_pred_with_cond = self.ace_step_transformer.decode(latent_model_input, attention_mask, encoder_hidden_states, encoder_hidden_mask, timestep, output_length).sample

                noise_pred_with_only_text_cond = None
                if do_double_condition_guidance and encoder_hidden_states_no_lyric is not None:
                    noise_pred_with_only_text_cond = self.ace_step_transformer.decode(latent_model_input, attention_mask, encoder_hidden_states_no_lyric, encoder_hidden_mask, timestep, output_length).sample

                if use_erg_diffusion:
                    inputs = {
                        "encoder_hidden_states": encoder_hidden_states_null,
                        "encoder_hidden_mask": encoder_hidden_mask,
                        "output_length": output_length,
                        "attention_mask": attention_mask,
                    }
                    noise_pred_uncond = forward_diffusion_with_temperature(self, latent_model_input, timestep, inputs)
                else:
                    noise_pred_uncond = self.ace_step_transformer.decode(latent_model_input, attention_mask, encoder_hidden_states_null, encoder_hidden_mask, timestep, output_length).sample

                if do_double_condition_guidance and noise_pred_with_only_text_cond is not None:
                    noise_pred = cfg_double_condition_forward(noise_pred_with_cond, noise_pred_uncond, noise_pred_with_only_text_cond, guidance_scale_text, guidance_scale_lyric)

                elif cfg_type == "apg":
                    noise_pred = apg_forward(noise_pred_with_cond, noise_pred_uncond, current_guidance_scale, momentum_buffer)
                elif cfg_type == "cfg":
                    noise_pred = cfg_forward(noise_pred_with_cond, noise_pred_uncond, current_guidance_scale)
                elif cfg_type == "cfg_star":
                    noise_pred = cfg_zero_star(noise_pred_with_cond, noise_pred_uncond, current_guidance_scale, i, zero_steps, use_zero_init)
            else:
                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.ace_step_transformer.decode(latent_model_input, attention_mask, encoder_hidden_states, encoder_hidden_mask, timestep, latent_model_input.shape[-1]).sample

            if is_repaint and i >= n_min:
                t_i = t / 1000
                if i + 1 < len(timesteps):
                    t_im1 = (timesteps[i + 1]) / 1000
                else:
                    t_im1 = torch.zeros_like(t_i).to(self.device)
                target_latents = target_latents.to(torch.float32)
                prev_sample = target_latents + (t_im1 - t_i) * noise_pred
                prev_sample = prev_sample.to(self.dtype)
                target_latents = prev_sample
                zt_src = (1 - t_im1) * x0 + (t_im1) * z0
                target_latents = torch.where(repaint_mask == 1.0, target_latents, zt_src)
            else:
                target_latents = scheduler.step(noise_pred, t, target_latents, return_dict=False, omega=omega_scale, generator=random_generators[0])[0]

        if is_extend:
            if to_right_pad_gt_latents is not None:
                target_latents = torch.cat([target_latents, to_right_pad_gt_latents], dim=-1)
            if to_left_pad_gt_latents is not None:
                target_latents = torch.cat([to_right_pad_gt_latents, target_latents], dim=0)

        return target_latents

    @cpu_offload("music_dcae")
    def latents2audio(
        self,
        latents,
        target_wav_duration_second=30,
        save_path=None,
        format="wav",
        sample_rate=48000,
    ):
        output_audio_paths = []
        bs = latents.shape[0]
        pred_latents = latents
        with torch.no_grad():
            if self.overlapped_decode and target_wav_duration_second > 48:
                _, pred_wavs = self.music_dcae.decode_overlap(pred_latents, sr=sample_rate)
            else:
                _, pred_wavs = self.music_dcae.decode(pred_latents, sr=sample_rate)
        pred_wavs = [pred_wav.cpu().float() for pred_wav in pred_wavs]

        for i in tqdm(range(bs)):
            output_audio_path = self.save_wav_file(pred_wavs[i], i, save_path, sample_rate, format)
            output_audio_paths.append(output_audio_path)

        return output_audio_paths

    def save_wav_file(self, target_wav, idx, save_path=None, sample_rate=48000, format="wav"):
        if save_path is None:
            logger.warning("save_path is None, using default path ./outputs/")
            base_path = "./outputs"
            ensure_directory_exists(base_path)
            output_path_wav = f"{base_path}/output_{time.strftime('%Y%m%d%H%M%S')}_{idx}." + format
        else:
            ensure_directory_exists(os.path.dirname(save_path))
            if os.path.isdir(save_path):
                logger.info("Provided save_path '{}' is a directory. Appending timestamped filename.", save_path)
                output_path_wav = os.path.join(save_path, f"output_{time.strftime('%Y%m%d%H%M%S')}_{idx}."+format)
            else:
                output_path_wav = save_path

        target_wav = target_wav.float()
        logger.info("Saving audio to {}", output_path_wav)
        torchaudio.save(output_path_wav, target_wav, sample_rate=sample_rate, format=format)

        return output_path_wav

    @cpu_offload("music_dcae")
    def infer_latents(self, input_audio_path):
        if input_audio_path is None:
            return None
        input_audio, sr = self.music_dcae.load_audio(input_audio_path)
        input_audio = input_audio.unsqueeze(0)
        input_audio = input_audio.to(device=self.device, dtype=self.dtype)
        latents, _ = self.music_dcae.encode(input_audio, sr=sr)

        return latents

    def load_lora(self, lora_name_or_path, lora_weight):
        if (lora_name_or_path != self.lora_path or lora_weight != self.lora_weight) and lora_name_or_path != "none":
            if not os.path.exists(lora_name_or_path):
                lora_download_path = snapshot_download(lora_name_or_path, cache_dir=self.checkpoint_dir)
            else:
                lora_download_path = lora_name_or_path
            if self.lora_path != "none":
                self.ace_step_transformer.unload_lora()
            self.ace_step_transformer.load_lora_adapter(os.path.join(lora_download_path, "pytorch_lora_weights.safetensors"), adapter_name="ace_step_lora", with_alpha=True, prefix=None)

            logger.info("Loading lora weights from: {} download path is: {} weight: {}", lora_name_or_path, lora_download_path, lora_weight)
            set_weights_and_activate_adapters(self.ace_step_transformer, ["ace_step_lora"], [lora_weight])
            self.lora_path = lora_name_or_path
            self.lora_weight = lora_weight

        elif self.lora_path != "none" and lora_name_or_path == "none":
            logger.info("No lora weights to load.")
            self.ace_step_transformer.unload_lora()

    def __call__(
        self,
        format: str = "wav",
        audio_duration: float = 60.0,
        prompt: str = None,
        lyrics: str = None,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        manual_seeds: list = None,
        guidance_interval: float = 0.5,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        audio2audio_enable: bool = False,
        ref_audio_strength: float = 0.5,
        ref_audio_input: str = None,
        lora_name_or_path: str = "none",
        lora_weight: float = 1.0,
        retake_seeds: list = None,
        retake_variance: float = 0.5,
        task: str = "text2music",
        repaint_start: int = 0,
        repaint_end: int = 0,
        src_audio_path: str = None,
        edit_target_prompt: str = None,
        edit_target_lyrics: str = None,
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        edit_n_avg: int = 1,
        save_path: str = None,
        batch_size: int = 1,
        debug: bool = False,
    ):
        start_time = time.time()

        if audio2audio_enable and ref_audio_input is not None:
            task = "audio2audio"

        if not self.loaded:
            logger.warning("Checkpoint not loaded, loading checkpoint...")
            if self.quantized:
                self.load_quantized_checkpoint(self.checkpoint_dir)
            else:
                self.load_checkpoint(self.checkpoint_dir)

        self.load_lora(lora_name_or_path, lora_weight)
        load_model_cost = time.time() - start_time
        logger.info("Model loaded in {:.2f} seconds.", load_model_cost)

        start_time = time.time()

        random_generators, actual_seeds = self.set_seeds(batch_size, manual_seeds)
        retake_random_generators, actual_retake_seeds = self.set_seeds(batch_size, retake_seeds)

        if isinstance(oss_steps, str) and len(oss_steps) > 0:
            oss_steps = list(map(int, oss_steps.split(",")))
        else:
            oss_steps = []

        texts = [prompt]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(texts)
        encoder_text_hidden_states = encoder_text_hidden_states.repeat(batch_size, 1, 1)
        text_attention_mask = text_attention_mask.repeat(batch_size, 1)

        encoder_text_hidden_states_null = None
        if use_erg_tag:
            encoder_text_hidden_states_null = self.get_text_embeddings_null(texts)
            encoder_text_hidden_states_null = encoder_text_hidden_states_null.repeat(batch_size, 1, 1)

        # not support for released checkpoint
        speaker_embeds = torch.zeros(batch_size, 512).to(self.device).to(self.dtype)

        # 6 lyric
        lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        if len(lyrics) > 0:
            lyric_token_idx = self.tokenize_lyrics(lyrics, debug=debug)
            lyric_mask = [1] * len(lyric_token_idx)
            lyric_token_idx = torch.tensor(lyric_token_idx).unsqueeze(0).to(self.device).repeat(batch_size, 1)
            lyric_mask = torch.tensor(lyric_mask).unsqueeze(0).to(self.device).repeat(batch_size, 1)

        if audio_duration <= 0:
            audio_duration = random.uniform(30.0, 240.0)
            logger.info("random audio duration: {}", audio_duration)

        end_time = time.time()
        preprocess_time_cost = end_time - start_time
        start_time = end_time

        add_retake_noise = task in ("retake", "repaint", "extend")
        # retake equal to repaint
        if task == "retake":
            repaint_start = 0
            repaint_end = audio_duration

        src_latents = None
        if src_audio_path is not None:
            assert src_audio_path is not None and task in (
                "repaint", "edit", "extend",
            ), "src_audio_path is required for retake/repaint/extend task"
            assert os.path.exists(src_audio_path), f"src_audio_path {src_audio_path} does not exist"
            src_latents = self.infer_latents(src_audio_path)
        
        ref_latents = None
        if ref_audio_input is not None and audio2audio_enable:
            assert ref_audio_input is not None, "ref_audio_input is required for audio2audio task"
            assert os.path.exists(ref_audio_input), f"ref_audio_input {ref_audio_input} does not exist"
            ref_latents = self.infer_latents(ref_audio_input)

        if task == "edit":
            texts = [edit_target_prompt]
            target_encoder_text_hidden_states, target_text_attention_mask = self.get_text_embeddings(texts)
            target_encoder_text_hidden_states = target_encoder_text_hidden_states.repeat(batch_size, 1, 1)
            target_text_attention_mask = target_text_attention_mask.repeat(batch_size, 1)

            target_lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
            target_lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
            if len(edit_target_lyrics) > 0:
                target_lyric_token_idx = self.tokenize_lyrics(edit_target_lyrics, debug=True)
                target_lyric_mask = [1] * len(target_lyric_token_idx)
                target_lyric_token_idx = torch.tensor(target_lyric_token_idx).unsqueeze(0).to(self.device).repeat(batch_size, 1)
                target_lyric_mask = torch.tensor(target_lyric_mask).unsqueeze(0).to(self.device).repeat(batch_size, 1)

            target_speaker_embeds = speaker_embeds.clone()

            target_latents = self.flowedit_diffusion_process(encoder_text_hidden_states, text_attention_mask, speaker_embeds, lyric_token_idx, lyric_mask, target_encoder_text_hidden_states, target_text_attention_mask, target_speaker_embeds, target_lyric_token_idx, target_lyric_mask, src_latents, retake_random_generators, infer_step, guidance_scale, edit_n_min, edit_n_max, edit_n_avg, scheduler_type)
        else:
            target_latents = self.text2music_diffusion_process(audio_duration, encoder_text_hidden_states, text_attention_mask, speaker_embeds, lyric_token_idx, lyric_mask, random_generators, infer_step, guidance_scale, omega_scale, scheduler_type, cfg_type, guidance_interval, guidance_interval_decay, min_guidance_scale, oss_steps, encoder_text_hidden_states_null, use_erg_lyric, use_erg_diffusion, retake_random_generators, retake_variance, add_retake_noise, guidance_scale_text, guidance_scale_lyric, repaint_start, repaint_end, src_latents, audio2audio_enable, ref_audio_strength, ref_latents)

        end_time = time.time()
        diffusion_time_cost = end_time - start_time
        start_time = end_time

        output_paths = self.latents2audio(target_latents, audio_duration, save_path, format)

        # Clean up memory after generation
        self.cleanup_memory()

        end_time = time.time()
        latent2audio_time_cost = end_time - start_time
        timecosts = {
            "preprocess": preprocess_time_cost,
            "diffusion": diffusion_time_cost,
            "latent2audio": latent2audio_time_cost,
        }

        input_params_json = {
            "format": format,
            "lora_name_or_path": lora_name_or_path,
            "lora_weight": lora_weight,
            "task": task,
            "prompt": prompt if task != "edit" else edit_target_prompt,
            "lyrics": lyrics if task != "edit" else edit_target_lyrics,
            "audio_duration": audio_duration,
            "infer_step": infer_step,
            "guidance_scale": guidance_scale,
            "scheduler_type": scheduler_type,
            "cfg_type": cfg_type,
            "omega_scale": omega_scale,
            "guidance_interval": guidance_interval,
            "guidance_interval_decay": guidance_interval_decay,
            "min_guidance_scale": min_guidance_scale,
            "use_erg_tag": use_erg_tag,
            "use_erg_lyric": use_erg_lyric,
            "use_erg_diffusion": use_erg_diffusion,
            "oss_steps": oss_steps,
            "timecosts": timecosts,
            "actual_seeds": actual_seeds,
            "retake_seeds": actual_retake_seeds,
            "retake_variance": retake_variance,
            "guidance_scale_text": guidance_scale_text,
            "guidance_scale_lyric": guidance_scale_lyric,
            "repaint_start": repaint_start,
            "repaint_end": repaint_end,
            "edit_n_min": edit_n_min,
            "edit_n_max": edit_n_max,
            "edit_n_avg": edit_n_avg,
            "src_audio_path": src_audio_path,
            "edit_target_prompt": edit_target_prompt,
            "edit_target_lyrics": edit_target_lyrics,
            "audio2audio_enable": audio2audio_enable,
            "ref_audio_strength": ref_audio_strength,
            "ref_audio_input": ref_audio_input,
        }
        # save input_params_json
        for output_audio_path in output_paths:
            input_params_json_save_path = output_audio_path.replace(f".{format}", "_input_params.json")
            input_params_json["audio_path"] = output_audio_path
            with open(input_params_json_save_path, "w", encoding="utf-8") as f:
                json.dump(input_params_json, f, indent=4, ensure_ascii=False)

        return output_paths + [input_params_json]
