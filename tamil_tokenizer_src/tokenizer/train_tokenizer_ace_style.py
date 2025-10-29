# train_tokenizer_ace_style.py
import os
import pandas as pd
import unicodedata
from tamil_tokenizer_ace import AceStyleTamilTokenizer

class AceTokenizerTrainer:
    """
    ACE-Step inspired tokenizer training pipeline
    """
    
    def __init__(self, 
                 metadata_path: str,
                 audio_dir: str,
                 output_dir: str = "chkpts/tokenizer"):
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.corpus_file = os.path.join(output_dir, "tamil_corpus_ace.txt")
        self.model_name = "tamil_tokenizer_ace"
    
    def load_metadata(self):
        """Load and validate dataset metadata"""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        
        df = pd.read_csv(self.metadata_path)
        print(f"📊 Loaded dataset with {len(df)} samples")
        
        # Validate required columns
        required_cols = ['prompt', 'lyrics', 'audio_path']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            print(f"📋 Available columns: {df.columns.tolist()}")
        
        return df
    
    def load_lyrics(self, lyrics_path: str) -> str:
        """Load lyrics from file with error handling"""
        try:
            if os.path.exists(lyrics_path):
                with open(lyrics_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
        except Exception as e:
            print(f"❌ Error loading lyrics {lyrics_path}: {e}")
        return ""
    
    # IMPROVED VERSION - Key fixes:
  
    def build_corpus(self, df: pd.DataFrame) -> str:
        """Build training corpus from dataset - IMPROVED VERSION"""
        corpus_lines = []
        
        print("📁 Building ACE-Step style corpus...")
        
        # Count statistics
        total_texts = 0
        skipped_samples = 0
        
        for idx, row in df.iterrows():
            texts = []
            
            # 1. Add prompts with music structure tokens
            prompt_text = ""
            for col in ['prompt', 'description', 'text', 'caption']:
                if col in row and pd.notna(row[col]):
                    prompt_text = str(row[col])
                    break
            
            if prompt_text:
                # Enhance prompt with music tokens (ACE-Step style)
                enhanced_prompt = f"[LYRIC] {prompt_text} [MUSIC]"
                texts.append(enhanced_prompt)
            
            # 2. Add lyrics with proper structure
            lyric_text = ""
            for col in ['lyrics', 'lyric', 'transcript']:
                if col in row and pd.notna(row[col]):
                    lyric_content = str(row[col])
                    # Check if it's a file path
                
                    if os.path.exists(os.path.join(self.audio_dir, lyric_content)):
                        lyric_text = self.load_lyrics(os.path.join(self.audio_dir, lyric_content))
                    else:
                        lyric_text = lyric_content
                    break
            
            if lyric_text:
                # Add music structure markers to lyrics
                structured_lyrics = self._add_music_structure(lyric_text)
                texts.append(structured_lyrics)
            
            # 3. Combine all texts for this sample
            if texts:
                combined_text = " ".join(texts)
                # Normalize Tamil text and clean
                combined_text = self._clean_tamil_text(combined_text)
                corpus_lines.append(combined_text)
                total_texts += 1
            else:
                skipped_samples += 1
            
            if (idx + 1) % 50 == 0:  # More frequent updates for small dataset
                print(f"   Processed {idx + 1}/{len(df)} samples")
        
        print(f"   Used {total_texts} samples, skipped {skipped_samples}")
        
        # Save corpus with proper formatting
        corpus_text = "\n\n".join(corpus_lines)
        
        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            f.write(corpus_text)
        
        # Print corpus statistics
        self._print_corpus_stats(corpus_text, corpus_lines)
        
        return corpus_text

    def _add_music_structure(self, lyrics: str) -> str:
        """Add music structure tokens to lyrics (ACE-Step style)"""
        # Simple heuristic-based structure detection
        lines = lyrics.split('\n')
        structured_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Detect common Tamil song structures
            line_lower = line.lower()
            if any(word in line_lower for word in ['பல்லவி', 'pallavi', 'chorus']):
                structured_lines.append(f"[CHORUS] {line}")
            elif any(word in line_lower for word in ['சரணம்', 'charanam', 'verse']):
                structured_lines.append(f"[VERSE] {line}")
            elif any(word in line_lower for word in ['முதல்', 'first', 'intro']):
                structured_lines.append(f"[INTRO] {line}")
            elif any(word in line_lower for word in ['இறுதி', 'end', 'outro']):
                structured_lines.append(f"[OUTRO] {line}")
            else:
                # Default to lyric token for regular lines
                structured_lines.append(f"[LYRIC] {line}")
        
        return "\n".join(structured_lines)

    def _clean_tamil_text(self, text: str) -> str:
        """Clean and normalize Tamil text"""
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove excessive whitespace but preserve structure
        text = ' '.join(text.split())
        
        # Ensure proper spacing around special tokens
        import re
        special_tokens = ['\[MUSIC\]', '\[LYRIC\]', '\[CHORUS\]', '\[VERSE\]', 
                         '\[BRIDGE\]', '\[INTRO\]', '\[OUTRO\]', '\[INSTRUMENTAL\]']
        
        for token in special_tokens:
            text = re.sub(f'\\s*{token}\\s*', f' {token} ', text)
        
        return text.strip()

    def _print_corpus_stats(self, corpus_text: str, corpus_lines: list):
        """Print detailed corpus statistics"""
        print(f"✅ Corpus built: {self.corpus_file}")
        print(f"   Total samples: {len(corpus_lines)}")
        print(f"   Corpus length: {len(corpus_text):,} characters")
        print(f"   Unique chars: {len(set(corpus_text))}")
        print(f"   Avg chars per sample: {len(corpus_text) // len(corpus_lines) if corpus_lines else 0}")
        
        # Count special tokens usage
        special_tokens = ['[MUSIC]', '[LYRIC]', '[CHORUS]', '[VERSE]', '[BRIDGE]', '[INTRO]', '[OUTRO]', '[INSTRUMENTAL]']
        print("   Special token usage:")
        for token in special_tokens:
            count = corpus_text.count(token)
            if count > 0:
                print(f"     {token}: {count} times")

    def calculate_vocab_size(self, corpus_text: str) -> int:
        """Calculate optimal vocabulary size for small dataset"""
        unique_chars = len(set(corpus_text))
        total_chars = len(corpus_text)
        
        # For 390 songs, we need much smaller vocab
        if total_chars < 50000:  # Very small dataset
            vocab_size = min(2000, max(500, unique_chars * 4))
        elif total_chars < 200000:  # Small dataset
            vocab_size = min(4000, max(1000, unique_chars * 3))
        else:  # Medium dataset
            vocab_size = min(8000, max(2000, unique_chars * 2))
        
        print(f"📊 Corpus stats: {total_chars:,} chars, {unique_chars} unique chars")
        print(f"🎯 Calculated vocab size: {vocab_size}")
        
        return vocab_size

    def train_tokenizer(self, vocab_size: int = None):
        """Train the tokenizer with improved parameters"""
        if vocab_size is None:
            # Read corpus to calculate vocab size
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                corpus_text = f.read()
            vocab_size = self.calculate_vocab_size(corpus_text)
        
        print("🔄 Training ACE-Step inspired Tamil tokenizer...")
        
        tokenizer = AceStyleTamilTokenizer()
        
        try:
            model_path = tokenizer.train_tokenizer(
                corpus_file=self.corpus_file,
                model_prefix=os.path.join(self.output_dir, self.model_name),
                vocab_size=vocab_size,
                model_type='bpe'
            )
            
            return tokenizer, model_path
            
        except Exception as e:
            print(f"❌ Tokenizer training failed: {e}")
            # Fallback to smaller vocab size
            print("🔄 Trying with smaller vocabulary size...")
            model_path = tokenizer.train_tokenizer(
                corpus_file=self.corpus_file,
                model_prefix=os.path.join(self.output_dir, self.model_name),
                vocab_size=min(2000, vocab_size // 2),
                model_type='bpe'
            )
            return tokenizer, model_path

    def verify_tokenizer(self, tokenizer: AceStyleTamilTokenizer):
        """Enhanced tokenizer verification"""
        print("\n🔍 Comprehensive Tokenizer Verification")
        print("=" * 60)
        
        # Test samples covering Tamil + music tokens
        test_samples = [
            # Basic Tamil
            "வணக்கம்",  
            "பாடல் பாடுவேன்",
            # With music tokens
            "[LYRIC] என் இனிய பாடல் [MUSIC]",
            "[VERSE] முதல் பகுதி [CHORUS] இரண்டாம் பகுதி",
            "[INTRO] தொடக்கம் [OUTRO] முடிவு",
            # Complex cases
            "[MUSIC] [CHORUS] பல்லவி பகுதி [VERSE] சரணம் பகுதி",
            "இசை [LYRIC] பாடல் வரிகள் [MUSIC] இசை அமைப்பு",
        ]
        
        print("🧪 Testing Tamil + Music Token Integration:")
        print("-" * 60)
        
        all_passed = True
        
        for i, sample in enumerate(test_samples, 1):
            try:
                print(f"\nTest {i}: '{sample}'")
                
                # Encode
                tokens = tokenizer.encode(sample, add_special_tokens=False)
                print(f"   Tokens: {tokens}")
                print(f"   Token count: {len(tokens)}")
                
                # Show token pieces
                token_pieces = [tokenizer.sp_model.id_to_piece(tid) for tid in tokens]
                print(f"   Pieces: {token_pieces}")
                
                # Decode  
                decoded = tokenizer.decode(tokens, skip_special_tokens=False)
                
                # Check round-trip
                passed = sample == decoded
                status = "✅" if passed else "❌"
                print(f"   Round-trip: {status}")
                
                if not passed:
                    print(f"   Original: '{sample}'")
                    print(f"   Decoded:  '{decoded}'")
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                all_passed = False
        
        # Test special token mapping
        print(f"\n🔍 Special Token Mapping:")
        special_tokens = tokenizer.get_special_tokens_dict()
        for name, token_id in special_tokens.items():
            if token_id is not None:
                token_piece = tokenizer.sp_model.id_to_piece(token_id)
                print(f"   {name}: {token_id} -> '{token_piece}'")
            else:
                print(f"   {name}: NOT FOUND")
        
        return all_passed

    def run_training(self):
            """Complete training pipeline"""
            print("🚀 Starting ACE-Step Tamil Tokenizer Training")
            print("=" * 60)
            
            # Step 1: Load data
            df = self.load_metadata()
            
            # Step 2: Build corpus
            corpus_text = self.build_corpus(df)
            
            # Step 3: Calculate vocab size
            vocab_size = self.calculate_vocab_size(corpus_text)
            
            # Step 4: Train tokenizer
            tokenizer, model_path = self.train_tokenizer(vocab_size)
            
            # Step 5: Verify
            success = self.verify_tokenizer(tokenizer)
            
            # Step 6: Save
            tokenizer.save_pretrained(self.output_dir)
            
            if success:
                print("\n🎉 ACE-Step Tamil Tokenizer Training Completed!")
                print(f"📁 Model saved: {model_path}")
                return tokenizer
            else:
                print("\n❌ Tokenizer training completed with errors")
                return None

# Enhanced main function
def main():
    """Main training function with better error handling"""
    
    # Configuration
    config = {
        'metadata_path': "tamil_tokenizer_src/data/metadata.csv",
        'audio_dir': "data/tamil_data/songs/", 
        'output_dir': "chkpts/tokenizer_ace"
    }
    
    # Check if data exists
    if not os.path.exists(config['metadata_path']):
        print(f"❌ Metadata file not found: {config['metadata_path']}")
        print("💡 Please check the path and try again")
        return
    
    try:
        trainer = AceTokenizerTrainer(**config)
        tokenizer = trainer.run_training()
        
        if tokenizer:
            print("\n🎉 ACE-Step Tamil Tokenizer Training Successful!")
            print("🎯 Next steps for LoRA training:")
            print("   1. Use the special token dict for model configuration")
            print("   2. Integrate with your base TTS model")
            print("   3. Start LoRA training with your 390 songs")
            
            # Print special tokens for easy copying
            special_tokens = tokenizer.get_special_tokens_dict()
            print(f"\n📋 Special tokens for model config:")
            for k, v in special_tokens.items():
                print(f'   "{k}": {v}')
                
        else:
            print("\n❌ Training failed - check the errors above")
            
    except Exception as e:
        print(f"💥 Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()