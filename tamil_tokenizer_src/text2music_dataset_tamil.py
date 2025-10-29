# text2music_dataset_tamil.py
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from datasets import load_from_disk
from loguru import logger
import time
import traceback
import torchaudio
from pathlib import Path
import re
from acestep.language_segmentation import LangSegment
from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
import warnings

# Import your Tamil tokenizer
from tokenizer.tamil_tokenizer_ace import AceStyleTamilTokenizer

warnings.simplefilter("ignore", category=FutureWarning)

DEFAULT_TRAIN_PATH = "./data/example_dataset"

def is_silent_audio(audio_tensor, silence_threshold=0.95):
    """
    Determine if an audio is silent and should be discarded
    """
    silent_samples = torch.all(audio_tensor == 0, dim=0)
    silent_ratio = torch.mean(silent_samples.float()).item()
    return silent_ratio > silence_threshold

# Enhanced supported languages with Tamil
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
    "ta": 7000,  # Tamil
}

structure_pattern = re.compile(r"\[.*?\]")
tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')  # Tamil Unicode range

class Text2MusicTamilDataset(Dataset):
    """
    Enhanced Dataset for text-to-music generation with Tamil support
    """

    def __init__(
        self,
        train=True,
        train_dataset_path=DEFAULT_TRAIN_PATH,
        max_duration=240.0,
        sample_size=None,
        shuffle=True,
        minibatch_size=1,
        tamil_tokenizer_path="chkpts/tokenizer_ace",  # Add Tamil tokenizer path
        use_tamil=True,  # Enable Tamil processing
    ):
        """
        Initialize the Text2Music dataset with Tamil support
        """
        self.train_dataset_path = train_dataset_path
        self.max_duration = max_duration
        self.minibatch_size = minibatch_size
        self.train = train
        self.use_tamil = use_tamil

        # Initialize Tamil tokenizer
        self.tamil_tokenizer = None
        if self.use_tamil:
            try:
                self.tamil_tokenizer = AceStyleTamilTokenizer.from_pretrained(tamil_tokenizer_path)
                print(f"✅ Tamil tokenizer loaded with vocab size: {self.tamil_tokenizer.vocab_size}")
            except Exception as e:
                print(f"❌ Failed to load Tamil tokenizer: {e}")
                self.tamil_tokenizer = None

        # Initialize language segmentation
        self.lang_segment = LangSegment()
        self.lang_segment.setfilters([
            "af", "am", "an", "ar", "as", "az", "be", "bg", "bn", "br", "bs", "ca", 
            "cs", "cy", "da", "de", "dz", "el", "en", "eo", "es", "et", "eu", "fa", 
            "fi", "fo", "fr", "ga", "gl", "gu", "he", "hi", "hr", "ht", "hu", "hy", 
            "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky", 
            "la", "lb", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "mt", 
            "nb", "ne", "nl", "nn", "no", "oc", "or", "pa", "pl", "ps", "pt", "qu", 
            "ro", "ru", "rw", "se", "si", "sk", "sl", "sq", "sr", "sv", "sw", "ta",  # Added Tamil
            "te", "th", "tl", "tr", "ug", "uk", "ur", "vi", "vo", "wa", "xh", "zh", "zu",
        ])

        # Initialize original lyric tokenizer (for fallback)
        self.lyric_tokenizer = VoiceBpeTokenizer()

        # Load dataset
        self.setup_full(train, shuffle, sample_size)
        logger.info(f"Dataset size: {len(self)} total {self.total_samples} samples")

    def setup_full(self, train=True, shuffle=True, sample_size=None):
        """
        Load and prepare the dataset
        """
        pretrain_ds = load_from_disk(self.train_dataset_path)

        if sample_size is not None:
            pretrain_ds = pretrain_ds.select(range(sample_size))

        self.pretrain_ds = pretrain_ds
        self.total_samples = len(self.pretrain_ds)

    def __len__(self):
        """Return the number of batches in the dataset"""
        if self.total_samples % self.minibatch_size == 0:
            return self.total_samples // self.minibatch_size
        else:
            return self.total_samples // self.minibatch_size + 1

    def is_tamil_text(self, text):
        """
        Detect if text contains Tamil characters
        """
        return bool(tamil_pattern.search(text))

    def get_lang(self, text):
        """
        Detect the language of a text with Tamil enhancement
        """
        # First check if it's Tamil
        if self.is_tamil_text(text):
            return "ta", [], [["ta", 1.0]]
        
        # Fall back to original detection
        language = "en"
        langs = []
        try:
            langs = self.lang_segment.getTexts(text)
            langCounts = self.lang_segment.getCounts()
            if langCounts:
                language = langCounts[0][0]
                if len(langCounts) > 1 and language == "en":
                    language = langCounts[1][0]
        except Exception:
            language = "en"
        return language, langs, langCounts

    def tokenize_lyrics_tamil(self, lyrics, debug=False):
        """
        Tokenize Tamil lyrics using custom Tamil tokenizer
        """
        if not lyrics or not lyrics.strip():
            return [self.tamil_tokenizer.pad_token_id] if self.tamil_tokenizer else [0]

        try:
            # Tokenize with Tamil tokenizer
            tokens = self.tamil_tokenizer.encode(lyrics, add_special_tokens=False)
            
            if debug:
                decoded = self.tamil_tokenizer.decode(tokens)
                print(f"🔤 Tamil tokenization debug:")
                print(f"   Original: {lyrics}")
                print(f"   Tokens: {tokens}")
                print(f"   Decoded: {decoded}")
            
            return tokens
            
        except Exception as e:
            print(f"❌ Tamil tokenization failed: {e}")
            return [self.tamil_tokenizer.pad_token_id] if self.tamil_tokenizer else [0]

    def tokenize_lyrics_original(self, lyrics, debug=False, key=None):
        """
        Original lyric tokenization (for non-Tamil text)
        """
        lines = lyrics.split("\n")
        lyric_token_idx = [261]  # Start token

        # Detect language
        lang, langs, lang_counter = self.get_lang(lyrics)

        # Determine most common language
        most_common_lang = "en"
        if lang_counter and len(lang_counter) > 0:
            most_common_lang = lang_counter[0][0]
            if most_common_lang == "":
                most_common_lang = "en"

        if most_common_lang not in SUPPORT_LANGUAGES:
            raise ValueError(f"Unsupported language: {most_common_lang}")

        # Process each language segment
        for lang_seg in langs:
            lang = lang_seg["lang"]
            text = lang_seg["text"]

            # Normalize language codes
            if lang not in SUPPORT_LANGUAGES:
                lang = "en"
            if "zh" in lang:
                lang = "zh"
            if "spa" in lang:
                lang = "es"

            # Process each line in the segment
            lines = text.split("\n")
            for line in lines:
                if not line.strip():
                    lyric_token_idx += [2]  # Line break token
                    continue

                try:
                    # Handle structure markers like [Verse], [Chorus]
                    if structure_pattern.match(line):
                        token_idx = self.lyric_tokenizer.encode(line, "en")
                    else:
                        token_idx = self.lyric_tokenizer.encode(line, most_common_lang)

                        # If debug mode, show tokenization results
                        if debug:
                            toks = self.lyric_tokenizer.batch_decode(
                                [[tok_id] for tok_id in token_idx]
                            )
                            logger.info(
                                f"debug using most_common_lang {line} --> {most_common_lang} --> {toks}"
                            )

                        # If tokenization contains unknown token (1), try with segment language
                        if 1 in token_idx:
                            token_idx = self.lyric_tokenizer.encode(line, lang)

                    if debug:
                        toks = self.lyric_tokenizer.batch_decode(
                            [[tok_id] for tok_id in token_idx]
                        )
                        logger.info(f"debug {line} --> {lang} --> {toks}")

                    # Add tokens and line break
                    lyric_token_idx = lyric_token_idx + token_idx + [2]

                except Exception as e:
                    logger.error(
                        f"Tokenize error: {e} for line: {line}, major_language: {lang}"
                    )

        return lyric_token_idx

    def tokenize_lyrics(self, lyrics, debug=False, key=None):
        """
        Unified lyric tokenization with Tamil support
        """
        if not lyrics or not lyrics.strip():
            return [0]

        # Auto-detect Tamil or use Tamil tokenizer if available
        if self.use_tamil and self.tamil_tokenizer and self.is_tamil_text(lyrics):
            print(f"🎵 Using Tamil tokenizer for lyrics: {lyrics[:50]}...")
            return self.tokenize_lyrics_tamil(lyrics, debug)
        else:
            # Use original tokenization for non-Tamil text
            return self.tokenize_lyrics_original(lyrics, debug, key)

    def tokenize_lyrics_map(self, item, debug=False):
        """
        Process and tokenize lyrics in a dataset item with Tamil support
        """
        norm_lyrics = item.get("norm_lyrics", "")

        # Filter out prompts that match pattern "write a .* song that genre is"
        pattern = r"write a .* song that genre is"
        if re.search(pattern, norm_lyrics):
            norm_lyrics = ""
            item["lyric_token_idx"] = [0]
            item["norm_lyrics"] = norm_lyrics
            return item

        key = item.get("keys", "unknown")

        # Handle empty lyrics
        if not norm_lyrics.strip():
            item["lyric_token_idx"] = [0]
            return item

        # Tokenize lyrics with Tamil support
        item["lyric_token_idx"] = self.tokenize_lyrics(norm_lyrics, debug, key)
        return item

    def get_speaker_emb_file(self, speaker_emb_path):
        """
        Load speaker embedding file
        """
        data = None
        try:
            data = torch.load(speaker_emb_path, map_location="cpu")
        except Exception:
            pass
        return data

    def get_audio(self, item):
        """
        Load and preprocess audio file
        """
        filename = item["filename"]
        sr = 48000
        try:
            audio, sr = torchaudio.load(filename)
        except Exception as e:
            logger.error(f"Failed to load audio {item}: {e}")
            return None

        if audio is None:
            logger.error(f"Failed to load audio {item}")
            return None

        # Convert mono to stereo if needed
        if audio.shape[0] == 1:
            audio = torch.cat([audio, audio], dim=0)

        # Take first two channels if more than stereo
        audio = audio[:2]

        # Resample if needed
        if sr != 48000:
            audio = torchaudio.transforms.Resample(sr, 48000)(audio)

        # Clip values to [-1.0, 1.0]
        audio = torch.clamp(audio, -1.0, 1.0)

        # Pad to minimum 3 seconds if needed
        if audio.shape[-1] < 48000 * 3:
            audio = torch.nn.functional.pad(
                audio, (0, 48000 * 3 - audio.shape[-1]), "constant", 0
            )

        # Check if audio is silent
        if is_silent_audio(audio):
            logger.error(f"Silent audio {item}")
            return None

        return audio

    def process(self, item):
        """
        Process a dataset item into model-ready features with Tamil support
        """
        # Get audio
        audio = self.get_audio(item)
        if audio is None:
            return []

        music_wavs = audio

        # Get speaker embedding
        key = item["keys"]
        speaker_emb_path = item.get("speaker_emb_path")
        speaker_emb = None
        
        if speaker_emb_path:
            speaker_emb = self.get_speaker_emb_file(speaker_emb_path)

        if speaker_emb is None:
            speaker_emb = torch.zeros(512)

        # Process prompt/tags
        prompt = item.get("tags", [])
        if len(prompt) == 0:
            prompt = ["music"]

        # Shuffle tags and join with commas
        random.shuffle(prompt)
        prompt = ", ".join(prompt)

        # Handle recaption data if available
        recaption = item.get("recaption", {})
        valid_recaption = []
        for k, v in recaption.items():
            if isinstance(v, str) and len(v) > 0:
                valid_recaption.append(v)

        # Add original prompt to recaption options and randomly select one
        valid_recaption.append(prompt)
        prompt = random.choice(valid_recaption)
        prompt = prompt[:256]  # Limit prompt length

        # Process lyrics with Tamil support
        lyric_token_idx = item["lyric_token_idx"]
        lyric_token_idx = torch.tensor(lyric_token_idx).long()
        lyric_token_idx = lyric_token_idx[:4096]  # Limit lyric context length
        lyric_mask = torch.ones(len(lyric_token_idx))

        # Create lyric chunks for display
        candidate_lyric_chunk = []
        lyrics = item.get("norm_lyrics", "")
        lyrics_lines = lyrics.split("\n")
        for lyric_line in lyrics_lines:
            candidate_lyric_chunk.append(
                {
                    "lyric": lyric_line,
                }
            )

        # Limit audio length
        longest_length = 24 * 10 * 48000  # 240 seconds
        music_wavs = music_wavs[:, :longest_length]
        vocal_wavs = torch.zeros_like(music_wavs)
        wav_len = music_wavs.shape[-1]

        # Create example dictionary
        example = {
            "key": key,
            "vocal_wav": vocal_wavs,
            "target_wav": music_wavs,
            "wav_length": wav_len,
            "prompt": prompt,
            "speaker_emb": speaker_emb,
            "lyric_token_id": lyric_token_idx,
            "lyric_mask": lyric_mask,
            "structured_tag": {"recaption": recaption},
            "candidate_lyric_chunk": candidate_lyric_chunk,
        }
        return [example]

    def get_full_features(self, idx):
        """
        Get full features for a dataset index
        """
        examples = {
            "keys": [],
            "target_wavs": [],
            "vocal_wavs": [],
            "wav_lengths": [],
            "structured_tags": [],
            "prompts": [],
            "speaker_embs": [],
            "lyric_token_ids": [],
            "lyric_masks": [],
            "candidate_lyric_chunks": [],
        }

        item = self.pretrain_ds[idx]
        item["idx"] = idx
        item = self.tokenize_lyrics_map(item)
        features = self.process(item)

        if features:
            for feature in features:
                for k, v in feature.items():
                    # Handle key mapping more explicitly
                    target_key = k + "s"  # Default plural form

                    # Special case handling for keys that don't follow simple plural pattern
                    if k == "key":
                        target_key = "keys"
                    elif k == "wav_length":
                        target_key = "wav_lengths"
                    elif k == "candidate_lyric_chunk":
                        target_key = "candidate_lyric_chunks"

                    if v is not None and target_key in examples:
                        examples[target_key].append(v)

        return examples

    def pack_batch(self, batch):
        """
        Pack a batch of examples
        """
        packed_batch = {}
        for item in batch:
            for k, v in item.items():
                if k not in packed_batch:
                    packed_batch[k] = v
                    continue
                packed_batch[k] += v
        return packed_batch

    def collate_fn(self, batch):
        """
        Collate function for DataLoader
        """
        batch = self.pack_batch(batch)
        output = {}

        for k, v in batch.items():
            if k in ["keys", "structured_tags", "prompts", "candidate_lyric_chunks"]:
                # Pass through lists without modification
                padded_input_list = v
            elif k in ["wav_lengths"]:
                # Convert to LongTensor
                padded_input_list = torch.LongTensor(v)
            elif k in ["src_wavs", "target_wavs", "vocal_wavs"]:
                # Pad audio to max length
                max_length = max(seq.shape[1] for seq in v)
                padded_input_list = torch.stack(
                    [
                        torch.nn.functional.pad(
                            seq, (0, max_length - seq.shape[1]), "constant", 0
                        )
                        for seq in v
                    ]
                )
            elif k in ["clap_conditions"]:
                # Pad time dimension of embeddings
                max_length = max(seq.shape[0] for seq in v)
                v = [
                    torch.nn.functional.pad(
                        seq, (0, 0, 0, max_length - seq.shape[0]), "constant", 0
                    )
                    for seq in v
                ]
                padded_input_list = torch.stack(v)
            elif k == "speaker_embs":
                # Stack speaker embeddings
                padded_input_list = torch.stack(v)
            elif k in [
                "chunk_masks",
                "clap_attention_masks",
                "lyric_token_ids",
                "lyric_masks",
            ]:
                # Pad sequence tensors
                max_length = max(len(seq) for seq in v)
                padded_input_list = torch.stack(
                    [
                        torch.nn.functional.pad(
                            seq, (0, max_length - len(seq)), "constant", 0
                        )
                        for seq in v
                    ]
                )

            output[k] = padded_input_list

        return output

    def __getitem__(self, idx):
        """
        Get item at index with error handling
        """
        try:
            example = self.get_full_features(idx)
            if len(example["keys"]) == 0:
                raise Exception(f"Empty example {idx=}")
            return example
        except Exception as e:
            # Log error and try a different random index
            logger.error(f"Error in getting item {idx}: {e}")
            traceback.print_exc()
            new_idx = random.choice(range(len(self)))
            return self.__getitem__(new_idx)


# Backward compatibility alias
Text2MusicDataset = Text2MusicTamilDataset

if __name__ == "__main__":
    # Example usage with Tamil
    dataset = Text2MusicTamilDataset(
        train_dataset_path="./tamil_dataset",  # Your Tamil dataset path
        use_tamil=True,
        tamil_tokenizer_path="chkpts/tokenizer_ace"
    )
    print(f"Dataset size: {len(dataset)}")
    item = dataset[0]
    print(item)

    for k, v in item.items():
        if len(v) > 0 and isinstance(v[0], torch.Tensor):
            print(k, [v[i].shape for i in range(len(v))])
        else:
            print(k, v)