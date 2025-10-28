import torch
import sentencepiece as spm
import os
import json
from typing import List ,Dict,Union,Optional
import unicodedata

class AceStyleTamilTokenizer:

    SPECIAL_TOKENS = {
        'pad_token': '[PAD]',
        'unk_token': '[UNK]',
        'bos_token': '[BOS]',
        'eos_token':'[EOS]' ,
        'sep_token':'[SEP]',
        'cls_token':'[CLS]',
        'mask_token':'[MASK]',

         #Music-specific tokens (ACE-Step Style)
         'music_token': '[MUSIC]',
         'lyric_token': '[LYRIC]',
         'chorus_token': '[CHORUS]',
        'verse_token': '[VERSE]',
        'bridge_token': '[BRIDGE]',
        'intro_token': '[INTRO]',
        'outro_token': '[OUTRO]',
        'instrumental_token': '[INSTRUMENTAL]',
    }



    def __init__(self, model_path: str =None , device : str ='cpu'):
        self.device = torch.device(device)
        self.sp_model =None
        self.vocab =None 
        self.inverse_vocab =None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        # Initialize special token IDs
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.sep_token_id = 4
        self.cls_token_id = 5
        self.mask_token_id = 6
        self.music_token_id = 7
        self.lyric_token_id = 8
        self.chorus_token_id = 9
        self.verse_token_id = 10
        self.bridge_token_id = 11
        self.intro_token_id = 12
        self.outro_token_id = 13
        self.instrumental_token_id = 14

    def load_model(self, model_path: str):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_path)

        
        # Build vocabulary mapping
        self.vocab_size = self.sp_model.get_piece_size()
        self.vocab = {i: self.sp_model.id_to_piece(i) for i in range(self.vocab_size)}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"✅ Loaded tokenizer with vocab size: {self.vocab_size}")
    
    def train_tokenizer(self, corpus_file: str, model_prefix: str, vocab_size: int = 8000, model_type: str = 'bpe'):
        model_dir =os.path.dirname(model_prefix)
        os.makedirs(model_dir, exist_ok=True)

        train_args={
            'input': corpus_file,
            'model_prefix': model_prefix,
            'vocab_size': vocab_size,
            'model_type': model_type,
            'character_coverage': 1.0,
            'pad_id':self.pad_token_id,
            'pad_piece':self.SPECIAL_TOKENS['pad_token'],
            'unk_id':self.unk_token_id,
            'unk_piece':self.SPECIAL_TOKENS['unk_token'],
            'bos_id':self.bos_token_id,
            'bos_piece':self.SPECIAL_TOKENS['bos_token'],
            'eos_id':self.eos_token_id,
            'eos_piece':self.SPECIAL_TOKENS['eos_token'],

            'user_defined_symbols':[
                self.SPECIAL_TOKENS['sep_token'],   
                self.SPECIAL_TOKENS['cls_token'],
                self.SPECIAL_TOKENS['mask_token'],
                self.SPECIAL_TOKENS['music_token'],
                self.SPECIAL_TOKENS['lyric_token'],
                self.SPECIAL_TOKENS['chorus_token'],
                self.SPECIAL_TOKENS['verse_token'],
                self.SPECIAL_TOKENS['bridge_token'],
                self.SPECIAL_TOKENS['intro_token'],
                self.SPECIAL_TOKENS['outro_token'],
                self.SPECIAL_TOKENS['instrumental_token'],
            ],
            'split_by_unicode_script': True,
            'split_by_whitespace': True,
            'split_by_number': True,
            'treat_whitespace_as_suffix': False,
            'byte_fallback': True,
            'remove_extra_whitespaces': True,
            'add_dummy_prefix': True,
            # Training parameters
            'num_threads': os.cpu_count(),
            'max_sentence_length': 16384,
            'shuffle_input_sentence': True,
            'seed_sentencepiece_size': 1000000,
            'training_size': 10000000

        }
        print("Training tokenizer with the following args:")
        for k, v in train_args.items():
            print(f"  {k}: {v}")
        spm.SentencePieceTrainer.Train(
            **train_args
        )

        #load the trained model
        model_path = f"{model_prefix}.model"
        self.load_model(model_path)

        print(f"✅ Tokenizer trained and saved at: {model_path}")
        return model_path
    
        
    def encode(self, text: str,add_special_tokens: bool =True , max_length:Optional[int] =None ,truncation: bool =True)-> List[int]:
        if not self.sp_model:
            raise ValueError("Tokenizer model not loaded. Call load_model() first.")
               
        text = unicodedata.normalize('NFC', text)

        #Encode with sentencepiece
        tokens = self.sp_model.encode(text, out_type=int)

        #Add special tokens if required
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

        #Truncate if required

        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
            if add_special_tokens and tokens[-1] != self.eos_token_id:
                tokens[-1] = self.eos_token_id  # Ensure EOS token at the end after truncation
        
        return tokens
    
    def decode(self, 
               token_ids: List[int], 
               skip_special_tokens: bool =True) -> str:
        
        if not self.sp_model:
            raise ValueError("Tokenizer model not loaded. Call load_model() first.")
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        if skip_special_tokens:
            token_ids = [tid for tid in token_ids if tid not in {
                self.pad_token_id,
                self.unk_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.sep_token_id,
                self.cls_token_id,
                self.mask_token_id,
                self.music_token_id,
                self.lyric_token_id,
                self.chorus_token_id,
                self.verse_token_id,
                self.bridge_token_id,
                self.intro_token_id,
                self.outro_token_id,
                self.instrumental_token_id,
            }]
        
        return self.sp_model.decode(token_ids)
    
    def __call__(self, 
                  texts: Union[str, List[str]],
                  return_tensors: Optional[str] ="pt",
                  padding: bool =True,
                  truncation: bool =True,
                  max_length: Optional[int] =256,
                  **kwargs) -> Dict[str, torch.Tensor]:
         
        if isinstance(texts, str):
            texts = [texts]

        #Encode all texts
        encoded_texts=[]
        for text in texts:
            token_ids = self.encode(
                text,
                add_special_tokens=True,
                max_length=max_length,
                truncation=truncation
            )
            encoded_texts.append(token_ids)

        #pad sequences
        if padding:
            max_len = max(len(tokens) for tokens in encoded_texts)
            if max_length:
                max_len = min(max_len, max_length)
            
            padded_texts = []
            for tokens in encoded_texts:
                if len(tokens) < max_len:
                    padded_tokens = tokens + [self.pad_token_id] * (max_len - len(tokens))
                else:
                    padded_tokens = tokens[:max_len]
                padded_texts.append(padded_tokens)
            encoded_texts = padded_texts
        
        # Convert to tensors
        input_ids = torch.tensor(encoded_texts, dtype=torch.long, device=self.device)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        return result
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        
        # Save SentencePiece model
        model_path = os.path.join(save_directory, "tamil_tokenizer.model")
        with open(model_path, 'wb') as f:
            f.write(self.sp_model.serialized_model_proto())
        
        # Save tokenizer config
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.SPECIAL_TOKENS,
            "model_type": "sentencepiece"
        }
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Tokenizer saved to: {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str, device: str = 'cpu'):
        """Load pretrained tokenizer"""
        model_path = os.path.join(pretrained_path, "tamil_tokenizer.model")
        tokenizer = cls(device=device)
        tokenizer.load_model(model_path)
        return tokenizer

# ACE-Step compatible wrapper
class AceTamilTokenizer:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.tokenizer = AceStyleTamilTokenizer(model_path, device)
        
        # ACE-Step expected attributes
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id  
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
    
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
    
    def encode(self, text: str, **kwargs):
        return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size