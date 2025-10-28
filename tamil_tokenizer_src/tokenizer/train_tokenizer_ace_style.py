# train_tokenizer_ace_style.py
import os
import pandas as pd
import unicodedata
from tokenizer_ace_style import AceStyleTamilTokenizer

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
    
    def build_corpus(self, df: pd.DataFrame) -> str:
        """Build training corpus from dataset"""
        corpus_lines = []
        
        print("📁 Building ACE-Step style corpus...")
        
        for idx, row in df.iterrows():
            # Get text data from various possible columns
            texts = []
            
            # Try different column names for prompts
            for col in ['prompt', 'description', 'text', 'caption']:
                if col in row and pd.notna(row[col]):
                    texts.append(str(row[col]))
            
            # Try different column names for lyrics
            for col in ['lyrics', 'lyric', 'transcript']:
                if col in row and pd.notna(row[col]):
                    lyric_text = str(row[col])
                    if os.path.exists(os.path.join(self.audio_dir, lyric_text)):
                        lyric_text = self.load_lyrics(os.path.join(self.audio_dir, lyric_text))
                    texts.append(lyric_text)
            
            # Add music structure tokens (ACE-Step style)
            if texts:
                combined_text = " ".join(texts)
                # Normalize Tamil text
                combined_text = unicodedata.normalize('NFC', combined_text)
                corpus_lines.append(combined_text)
            
            if (idx + 1) % 100 == 0:
                print(f" Processed {idx + 1}/{len(df)} samples")
        
        # Join with double newlines (ACE-Step style document separation)
        corpus_text = "\n\n".join(corpus_lines)
        
        # Save corpus
        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            f.write(corpus_text)
        
        print(f"✅ Corpus built: {self.corpus_file}")
        print(f"   Total samples: {len(corpus_lines)}")
        print(f"   Corpus length: {len(corpus_text)} characters")
        print(f"   Unique chars: {len(set(corpus_text))}")
        
        return corpus_text
    
    def calculate_vocab_size(self, corpus_text: str) -> int:
        """Calculate optimal vocabulary size (ACE-Step inspired)"""
        unique_chars = len(set(corpus_text))
        
        # ACE-Step style vocabulary sizing
        if len(corpus_text) < 100000:  # Small dataset
            vocab_size = min(4000, max(1000, unique_chars * 3))
        elif len(corpus_text) < 1000000:  # Medium dataset
            vocab_size = min(8000, max(2000, unique_chars * 2))
        else:  # Large dataset
            vocab_size = min(16000, max(4000, unique_chars * 1.5))
        
        print(f"🎯 Calculated vocab size: {vocab_size}")
        return vocab_size
    
    def train_tokenizer(self, vocab_size: int = 8000):
        """Train the tokenizer"""
        print("🔄 Training ACE-Step inspired Tamil tokenizer...")
        
        tokenizer = AceStyleTamilTokenizer()
        
        model_path = tokenizer.train_tokenizer(
            corpus_path=self.corpus_file,
            model_prefix=os.path.join(self.output_dir, self.model_name),
            vocab_size=vocab_size,
            model_type='bpe'
        )
        
        return tokenizer, model_path
    
    def verify_tokenizer(self, tokenizer: AceStyleTamilTokenizer):
        """Comprehensive tokenizer verification"""
        print("\n🔍 Verifying tokenizer...")
        
        # Test samples covering various Tamil linguistic patterns
        test_samples = [
            "வணக்கம்",  # Basic greeting
            "பாடல் பாடுவேன் இசையுடன்",  # Simple sentence
            "கலை எங்கள் வாழ்க்கை",  # Cultural expression
            "இசை உலகம் அழகானது",  # Descriptive
            "பாடலில் இனிமை நிறைந்தது",  # Emotional
            "மழையின் தூறல் போல் இனிமையான பாடல்",  # Poetic
            "காற்றில் வந்த இசை மனதை தொடும் பாடல்",  # Complex sentence
            "[VERSE] முதல் பாடல் பகுதி [CHORUS] இரண்டாம் பகுதி",  # Music structure
        ]
        
        print("🧪 Tokenizer Tests:")
        print("-" * 60)
        
        all_passed = True
        for sample in test_samples:
            try:
                # Encode
                tokens = tokenizer.encode(sample)
                # Decode  
                decoded = tokenizer.decode(tokens)
                
                # Check round-trip
                passed = sample == decoded
                status = "✅" if passed else "❌"
                
                print(f"{status} '{sample}'")
                print(f"   Tokens: {len(tokens)} | Round-trip: {passed}")
                
                if not passed:
                    print(f"   Original: '{sample}'")
                    print(f"   Decoded:  '{decoded}'")
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ Error with: '{sample}' - {e}")
                all_passed = False
        
        # Test batch encoding
        try:
            batch_result = tokenizer(test_samples[:3], padding=True)
            print(f"✅ Batch encoding: {batch_result['input_ids'].shape}")
        except Exception as e:
            print(f"❌ Batch encoding failed: {e}")
            all_passed = False
        
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

def main():
    """Main training function"""
    trainer = AceTokenizerTrainer(
        metadata_path="data/tamil_data/metadata.csv",
        audio_dir="data/tamil_data/songs/",
        output_dir="chkpts/tokenizer_ace"
    )
    
    tokenizer = trainer.run_training()
    
    if tokenizer:
        print("\n🎯 Next steps:")
        print("   1. Use tokenizer in your LoRA training")
        print("   2. Test with ACE-Step pipeline")
        print("   3. Generate Tamil audio!")

if __name__ == "__main__":
    main()