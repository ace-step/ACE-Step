# pipeline_ace_step_tamil.py
"""
ACE-Step Tamil Modified: Adapted for Tamil TTS with custom tokenizer
"""

import random
import time
import os
import re

import torch
from loguru import logger
from tqdm import tqdm
import json
import math
from huggingface_hub import snapshot_download

from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from acestep.schedulers.scheduling_flow_match_heun_discrete import (
    FlowMatchHeunDiscreteScheduler,
)
from acestep.schedulers.scheduling_flow_match_pingpong import (
    FlowMatchPingPongScheduler,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils.torch_32 import randn_tensor
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from transformers import UMT5EncoderModel, AutoTokenizer

from acestep.language_segmentation import LangSegment, language_filters
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
from acestep.models.ace_step_transformer import ACEStepTransformer2DModel
from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
from acestep.apg_guidance import (
    apg_forward,
    MomentumBuffer,
    cfg_forward,
    cfg_zero_star,
    cfg_double_condition_forward,
)
import torchaudio
from .cpu_offload import cpu_offload

# Import your Tamil tokenizer
from tamil_tokenizer_ace import AceStyleTamilTokenizer

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add Tamil to supported languages
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
    "ta": 7000,  # Added Tamil
}

structure_pattern = re.compile(r"\[.*?\]")

def ensure_directory_exists(directory):
    directory = str(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

REPO_ID = "ACE-Step/ACE-Step-v1-3.5B"
REPO_ID_QUANT = REPO_ID + "-q4-K-M"

class ACEStepTamilPipeline:
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
        tamil_tokenizer_path="chkpts/tokenizer_ace",  # Add Tamil tokenizer path
        **kwargs,
    ):
        if not checkpoint_dir:
            if persistent_storage_path is None:
                checkpoint_dir = os.path.join(
                    os.path.expanduser("~"), ".cache/ace-step/checkpoints"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
            else:
                checkpoint_dir = os.path.join(persistent_storage_path, "checkpoints")
        ensure_directory_exists(checkpoint_dir)

        self.checkpoint_dir = checkpoint_dir
        self.lora_path = "none"
        self.lora_weight = 1
        device = (
            torch.device(f"cuda:{device_id}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if device.type == "cpu" and torch.backends.mps.is_available():
            device = torch.device("mps")
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        if device.type == "mps" and self.dtype == torch.bfloat16:
            self.dtype = torch.float16
        if device.type == "mps":
            self.dtype = torch.float32
        if 'ACE_PIPELINE_DTYPE' in os.environ and len(os.environ['ACE_PIPELINE_DTYPE']):
            self.dtype = getattr(torch, os.environ['ACE_PIPELINE_DTYPE'])
        self.device = device
        self.loaded = False
        self.torch_compile = torch_compile
        self.cpu_offload = cpu_offload
        self.quantized = quantized
        self.overlapped_decode = overlapped_decode
        
        # Load Tamil tokenizer
        self.tamil_tokenizer_path = tamil_tokenizer_path
        self.tamil_tokenizer = None
        self._load_tamil_tokenizer()

    def _load_tamil_tokenizer(self):
        """Load your custom Tamil tokenizer"""
        try:
            self.tamil_tokenizer = AceStyleTamilTokenizer.from_pretrained(self.tamil_tokenizer_path)
            print(f"✅ Tamil tokenizer loaded with vocab size: {self.tamil_tokenizer.vocab_size}")
        except Exception as e:
            print(f"❌ Failed to load Tamil tokenizer: {e}")
            self.tamil_tokenizer = None

    def cleanup_memory(self):
        """Clean up GPU and CPU memory to prevent VRAM overflow during multiple generations."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

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
                logger.info(f"Load models from: {checkpoint_dir}")
                checkpoint_dir_models = checkpoint_dir
        
        if checkpoint_dir_models is None:
            if checkpoint_dir is None:
                logger.info(f"Download models from Hugging Face: {repo}")
                checkpoint_dir_models = snapshot_download(repo)
            else:
                logger.info(f"Download models from Hugging Face: {repo}, cache to: {checkpoint_dir}")
                checkpoint_dir_models = snapshot_download(repo, cache_dir=checkpoint_dir)
        return checkpoint_dir_models

    def load_checkpoint(self, checkpoint_dir=None, export_quantized_weights=False):
        checkpoint_dir = self.get_checkpoint_path(checkpoint_dir, REPO_ID)
        dcae_checkpoint_path = os.path.join(checkpoint_dir, "music_dcae_f8c8")
        vocoder_checkpoint_path = os.path.join(checkpoint_dir, "music_vocoder")
        ace_step_checkpoint_path = os.path.join(checkpoint_dir, "ace_step_transformer")
        text_encoder_checkpoint_path = os.path.join(checkpoint_dir, "umt5-base")

        self.ace_step_transformer = ACEStepTransformer2DModel.from_pretrained(
            ace_step_checkpoint_path, torch_dtype=self.dtype
        )
        
        if self.cpu_offload:
            self.ace_step_transformer = (
                self.ace_step_transformer.to("cpu").eval().to(self.dtype)
            )
        else:
            self.ace_step_transformer = (
                self.ace_step_transformer.to(self.device).eval().to(self.dtype)
            )
        if self.torch_compile:
            self.ace_step_transformer = torch.compile(self.ace_step_transformer)

        self.music_dcae = MusicDCAE(
            dcae_checkpoint_path=dcae_checkpoint_path,
            vocoder_checkpoint_path=vocoder_checkpoint_path,
        )
        
        if self.cpu_offload:
            self.music_dcae = self.music_dcae.to("cpu").eval().to(self.dtype)
        else:
            self.music_dcae = self.music_dcae.to(self.device).eval().to(self.dtype)
        if self.torch_compile:
            self.music_dcae = torch.compile(self.music_dcae)

        lang_segment = LangSegment()
        lang_segment.setfilters(language_filters.default)
        self.lang_segment = lang_segment
        self.lyric_tokenizer = VoiceBpeTokenizer()

        text_encoder_model = UMT5EncoderModel.from_pretrained(
            text_encoder_checkpoint_path, torch_dtype=self.dtype
        ).eval()
        
        if self.cpu_offload:
            text_encoder_model = text_encoder_model.to("cpu").eval().to(self.dtype)
        else:
            text_encoder_model = text_encoder_model.to(self.device).eval().to(self.dtype)
        text_encoder_model.requires_grad_(False)
        self.text_encoder_model = text_encoder_model
        if self.torch_compile:
            self.text_encoder_model = torch.compile(self.text_encoder_model)

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_checkpoint_path
        )
        self.loaded = True

        # Load Tamil tokenizer if not already loaded
        if self.tamil_tokenizer is None:
            self._load_tamil_tokenizer()

    # MODIFIED: Enhanced text embedding method with Tamil support
    @cpu_offload("text_encoder_model")
    def get_text_embeddings(self, texts, text_max_length=256, use_tamil=False):
        """
        Get text embeddings with Tamil support
        """
        if use_tamil and self.tamil_tokenizer is not None:
            # Use Tamil tokenizer
            print("🔤 Using Tamil tokenizer for text encoding")
            encoded_texts = []
            attention_masks = []
            
            for text in texts:
                # Encode with Tamil tokenizer
                encoding = self.tamil_tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=text_max_length,
                    return_tensors="pt"
                )
                encoded_texts.append(encoding['input_ids'].squeeze(0))
                attention_masks.append(encoding['attention_mask'].squeeze(0))
            
            # Pad sequences
            max_len = max(len(seq) for seq in encoded_texts)
            input_ids = torch.zeros(len(texts), max_len, dtype=torch.long)
            attention_mask = torch.zeros(len(texts), max_len, dtype=torch.long)
            
            for i, (seq, mask) in enumerate(zip(encoded_texts, attention_masks)):
                input_ids[i, :len(seq)] = seq
                attention_mask[i, :len(mask)] = mask
            
            inputs = {
                'input_ids': input_ids.to(self.device),
                'attention_mask': attention_mask.to(self.device)
            }
        else:
            # Use original tokenizer
            inputs = self.text_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=text_max_length,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        if self.text_encoder_model.device != self.device:
            self.text_encoder_model.to(self.device)
            
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            
        attention_mask = inputs["attention_mask"]
        return last_hidden_states, attention_mask

    # MODIFIED: Enhanced lyric tokenization with Tamil support
    def tokenize_lyrics(self, lyrics, debug=False, use_tamil=False):
        """
        Tokenize lyrics with Tamil support
        """
        if use_tamil and self.tamil_tokenizer is not None:
            print("🎵 Using Tamil tokenizer for lyrics")
            # Simple Tamil lyric tokenization
            lines = lyrics.split("\n")
            all_tokens = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Tokenize with Tamil tokenizer
                tokens = self.tamil_tokenizer.encode(line, add_special_tokens=False)
                all_tokens.extend(tokens)
                all_tokens.append(self.tamil_tokenizer.sep_token_id)  # Add separator
                
            if all_tokens and all_tokens[-1] == self.tamil_tokenizer.sep_token_id:
                all_tokens = all_tokens[:-1]  # Remove last separator
                
            return all_tokens
        else:
            # Original lyric tokenization
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
                        toks = self.lyric_tokenizer.batch_decode(
                            [[tok_id] for tok_id in token_idx]
                        )
                        logger.info(f"debug {line} --> {lang} --> {toks}")
                    lyric_token_idx = lyric_token_idx + token_idx + [2]
                except Exception as e:
                    print("tokenize error", e, "for line", line, "major_language", lang)
            return lyric_token_idx

    def detect_language(self, text):
        """
        Detect if text contains Tamil characters
        """
        tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
        return bool(tamil_pattern.search(text))

    # MODIFIED: Enhanced main call method with Tamil detection
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
        use_tamil: bool = None,  # New parameter to force Tamil mode
    ):

        start_time = time.time()

        # Auto-detect Tamil if not specified
        if use_tamil is None:
            use_tamil = self.detect_language(prompt) or self.detect_language(lyrics or "")
            
        if use_tamil:
            print("🧮 Tamil mode activated!")

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
        logger.info(f"Model loaded in {load_model_cost:.2f} seconds.")

        start_time = time.time()

        random_generators, actual_seeds = self.set_seeds(batch_size, manual_seeds)
        retake_random_generators, actual_retake_seeds = self.set_seeds(
            batch_size, retake_seeds
        )

        if isinstance(oss_steps, str) and len(oss_steps) > 0:
            oss_steps = list(map(int, oss_steps.split(",")))
        else:
            oss_steps = []

        texts = [prompt]
        
        # MODIFIED: Use Tamil-aware text embedding
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            texts, use_tamil=use_tamil
        )
        encoder_text_hidden_states = encoder_text_hidden_states.repeat(batch_size, 1, 1)
        text_attention_mask = text_attention_mask.repeat(batch_size, 1)

        encoder_text_hidden_states_null = None
        if use_erg_tag:
            encoder_text_hidden_states_null = self.get_text_embeddings_null(texts)
            encoder_text_hidden_states_null = encoder_text_hidden_states_null.repeat(batch_size, 1, 1)

        speaker_embeds = torch.zeros(batch_size, 512).to(self.device).to(self.dtype)

        # MODIFIED: Use Tamil-aware lyric tokenization
        lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        
        if lyrics and len(lyrics) > 0:
            lyric_token_idx_list = self.tokenize_lyrics(lyrics, debug=debug, use_tamil=use_tamil)
            lyric_mask_list = [1] * len(lyric_token_idx_list)
            
            lyric_token_idx = (
                torch.tensor(lyric_token_idx_list)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )
            lyric_mask = (
                torch.tensor(lyric_mask_list)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )

        if audio_duration <= 0:
            audio_duration = random.uniform(30.0, 240.0)
            logger.info(f"random audio duration: {audio_duration}")

        # ... [rest of the method remains the same, using the modified components above]

        # The diffusion process and audio generation remain the same
        # Only the text/lyric processing is modified

        # [Rest of the original __call__ method continues...]
        # This part remains unchanged as it uses the already processed embeddings

        return self._continue_original_call(
            # ... pass all the parameters
        )

    def _continue_original_call(self, *args, **kwargs):
        """
        Continue with the original pipeline logic after Tamil modifications
        """
        # This would contain the rest of the original __call__ method
        # For brevity, I'm showing the key modifications above
        pass

# Keep the original utility methods unchanged
ACEStepTamilPipeline.set_seeds = ACEStepPipeline.set_seeds
ACEStepTamilPipeline.load_lora = ACEStepPipeline.load_lora
ACEStepTamilPipeline.latents2audio = ACEStepPipeline.latents2audio
ACEStepTamilPipeline.save_wav_file = ACEStepPipeline.save_wav_file
ACEStepTamilPipeline.infer_latents = ACEStepPipeline.infer_latents
# ... [other utility methods]