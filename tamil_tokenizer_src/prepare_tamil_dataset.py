# prepare_tamil_dataset_fixed.py
import os
import json
import pandas as pd
import torch
import torchaudio
from pathlib import Path
import numpy as np
from tokenizer.tamil_tokenizer_ace import AceStyleTamilTokenizer
import shutil

class TamilDatasetPreparer:
    def __init__(self, 
                 metadata_path: str,
                 output_dir: str = "./tamil_dataset",
                 tamil_tokenizer_path: str = "chkpts/tokenizer_ace"):
        
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.tamil_tokenizer_path = tamil_tokenizer_path
        
        # Create directories
        self.dataset_dir = Path(output_dir)
        self.audio_output_dir = self.dataset_dir / "audio"
        self.embeddings_dir = self.dataset_dir / "speaker_embeddings"
        
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.audio_output_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Load Tamil tokenizer
        self.tokenizer = AceStyleTamilTokenizer.from_pretrained(tamil_tokenizer_path)
        
    def load_metadata(self):
        """Load and validate your Tamil metadata"""
        df = pd.read_csv(self.metadata_path)
        print(f"📊 Loaded {len(df)} Tamil samples")
        
        # Show sample of data
        print("\n📋 Sample data structure:")
        for i in range(min(3, len(df))):
            print(f"  Sample {i}:")
            print(f"    Prompt: {df.iloc[i]['prompt'][:50]}...")
            print(f"    Audio: {os.path.basename(df.iloc[i]['audio_path'])}")
            print(f"    Lyrics: {df.iloc[i]['lyrics'][:50]}...")
            print()
            
        return df
    
    def load_file_content(self, file_path):
        """Load content from text files referenced in metadata"""
        try:
            if pd.isna(file_path) or not file_path.strip():
                return ""
            
            full_path = str(file_path).strip().replace('\\', '/')
            
            if os.path.exists(full_path) and os.path.isfile(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                return content
            else:
                # If it's not a file path, treat as direct text
                return str(file_path)
                
        except Exception as e:
            print(f"❌ Error loading file {file_path}: {e}")
            return str(file_path) if not pd.isna(file_path) else ""
    
    def copy_audio_files(self, df):
        """Copy audio files from their original locations to dataset directory"""
        print("🔊 Copying audio files...")
        
        audio_files_copied = 0
        audio_files_skipped = 0
        
        for idx, row in df.iterrows():
            audio_path = row['audio_path']
            
            try:
                if pd.isna(audio_path) or not audio_path.strip():
                    audio_files_skipped += 1
                    continue
                
                full_audio_path = str(audio_path).strip().replace('\\', '/')
                
                if not os.path.exists(full_audio_path):
                    print(f"❌ Audio file not found: {full_audio_path}")
                    audio_files_skipped += 1
                    continue
                
                # Get filename and copy to dataset audio directory
                audio_filename = os.path.basename(full_audio_path)
                dest_path = self.audio_output_dir / audio_filename
                
                # Copy audio file
                shutil.copy2(full_audio_path, dest_path)
                audio_files_copied += 1
                
                if (idx + 1) % 50 == 0:
                    print(f"  Copied {idx + 1}/{len(df)} audio files")
                    
            except Exception as e:
                print(f"❌ Error copying audio file {audio_path}: {e}")
                audio_files_skipped += 1
                continue
        
        print(f"✅ Copied {audio_files_copied} audio files")
        print(f"⚠️  Skipped {audio_files_skipped} audio files")
        return audio_files_copied
    
    def create_speaker_embeddings(self, num_speakers=1):
        """Create dummy speaker embeddings"""
        print("👤 Creating speaker embeddings...")
        
        # Create a dummy speaker embedding (512 dimensions as expected by ACE-Step)
        speaker_embedding = torch.randn(512)
        
        # Save the speaker embedding
        speaker_path = self.embeddings_dir / "speaker_1.pt"
        torch.save(speaker_embedding, speaker_path)
        
        print(f"✅ Created speaker embedding: {speaker_path}")
        return "speaker_1.pt"
    
    def process_prompt(self, prompt_content):
        """Process and enhance Tamil prompts with music tokens"""
        if not prompt_content:
            return "[LYRIC] Tamil music [MUSIC]"
        
        # Ensure prompt has proper music structure tokens
        if not any(token in prompt_content for token in ['[MUSIC]', '[LYRIC]', '[CHORUS]']):
            prompt_content = f"[LYRIC] {prompt_content} [MUSIC]"
        
        # Limit length and clean up
        words = prompt_content.split()
        if len(words) > 20:
            prompt_content = ' '.join(words[:20]) + "..."
            
        return prompt_content
    
    def process_lyrics(self, lyrics_content):
        """Process Tamil lyrics and add music structure"""
        if not lyrics_content:
            return ""
        
        # Clean and structure lyrics
        lines = lyrics_content.split('\n')
        structured_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Detect Tamil song sections
            line_lower = line.lower()
            if any(word in line_lower for word in ['பல்லவி', 'pallavi', 'chorus']):
                structured_lines.append(f"[CHORUS] {line}")
            elif any(word in line_lower for word in ['சரணம்', 'charanam', 'verse']):
                structured_lines.append(f"[VERSE] {line}")
            elif i == 0:
                structured_lines.append(f"[INTRO] {line}")
            elif i == len(lines) - 1:
                structured_lines.append(f"[OUTRO] {line}")
            else:
                structured_lines.append(f"[LYRIC] {line}")
        
        return "\n".join(structured_lines)
    
    def extract_tags_from_content(self, prompt, lyrics):
        """Extract relevant tags from content for ACE-Step"""
        # Base Tamil music tags
        base_tags = ["tamil", "music", "song", "melody", "traditional"]
        
        # Combine prompt and lyrics for tag extraction
        combined_text = (prompt + " " + lyrics).lower()
        
        # Genre detection
        genre_tags = []
        if any(genre in combined_text for genre in ['classical', 'carnatic']):
            genre_tags.extend(["classical", "carnatic"])
        elif any(genre in combined_text for genre in ['film', 'movie', 'kollywood']):
            genre_tags.extend(["film", "movie"])
        elif any(genre in combined_text for genre in ['folk', 'traditional']):
            genre_tags.extend(["folk", "traditional"])
        elif any(genre in combined_text for genre in ['pop', 'modern']):
            genre_tags.extend(["pop", "modern"])
        
        # Mood detection
        mood_tags = []
        if any(mood in combined_text for mood in ['happy', 'joyful', 'celebratory']):
            mood_tags.append("happy")
        elif any(mood in combined_text for mood in ['sad', 'emotional', 'melancholic']):
            mood_tags.append("sad")
        elif any(mood in combined_text for mood in ['romantic', 'love']):
            mood_tags.append("romantic")
        elif any(mood in combined_text for mood in ['energetic', 'fast']):
            mood_tags.append("energetic")
        
        # Instrument detection
        instrument_tags = []
        if any(instr in combined_text for instr in ['violin', 'veena']):
            instrument_tags.append("strings")
        if any(instr in combined_text for instr in ['mridangam', 'tabla']):
            instrument_tags.append("percussion")
        if any(instr in combined_text for instr in ['flute']):
            instrument_tags.append("flute")
        
        # Combine all tags
        all_tags = list(set(base_tags + genre_tags + mood_tags + instrument_tags))
        return all_tags[:10]  # Limit to 10 tags
    
    def create_dataset_jsonl(self, df, speaker_embedding_file):
        """Create the dataset in ACE-Step JSONL format"""
        print("📝 Creating dataset JSONL file...")
        
        dataset_records = []
        skipped_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Get audio filename and path
                audio_path = row['audio_path']
                audio_filename = os.path.basename(str(audio_path))
                
                # Check if audio file exists in our organized directory
                audio_file_path = self.audio_output_dir / audio_filename
                if not audio_file_path.exists():
                    print(f"⚠️ Audio file not found in dataset: {audio_filename}")
                    skipped_count += 1
                    continue
                
                # Load prompt content (could be file path or direct text)
                prompt_content = self.load_file_content(row['prompt'])
                lyrics_content = self.load_file_content(row['lyrics'])
                
                # Process prompt and lyrics
                prompt = self.process_prompt(prompt_content)
                lyrics = self.process_lyrics(lyrics_content)
                
                # Extract tags
                tags = self.extract_tags_from_content(prompt, lyrics)
                
                # Create dataset record
                record = {
                    "keys": f"tamil_song_{idx:04d}",
                    "filename": str(audio_file_path),
                    "speaker_emb_path": str(self.embeddings_dir / speaker_embedding_file),
                    "tags": tags,
                    "norm_lyrics": lyrics,
                    "recaption": {
                        "original": prompt,
                        "enhanced": f"{prompt} - Tamil music with traditional instruments and melodic vocals"
                    }
                }
                
                dataset_records.append(record)
                
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} samples")
                    
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
                skipped_count += 1
                continue
        
        # Save as JSONL file
        jsonl_path = self.dataset_dir / "dataset.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for record in dataset_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"🎉 Dataset created: {len(dataset_records)} samples")
        print(f"⚠️  Skipped: {skipped_count} samples")
        return dataset_records
    
    def create_dataset_info(self, num_samples):
        """Create dataset_info.json with Tamil-specific information"""
        dataset_info = {
            "description": "Tamil Music Dataset for ACE-Step TTS Training",
            "language": "ta",
            "num_samples": num_samples,
            "audio_format": "wav",
            "sample_rate": 48000,
            "tokenizer": "tamil_custom",
            "vocab_size": self.tokenizer.vocab_size,
            "special_tokens": self.tokenizer.get_special_tokens_dict(),
            "created_with": "TamilDatasetPreparer",
            "features": ["prompt", "lyrics", "audio", "speaker_embedding"]
        }
        
        info_path = self.dataset_dir / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        return dataset_info
    
    def convert_to_hf_dataset(self):
        """Convert JSONL to Hugging Face dataset format"""
        print("🔄 Converting to Hugging Face dataset format...")
        
        try:
            from datasets import Dataset
            
            # Read JSONL file
            jsonl_path = self.dataset_dir / "dataset.jsonl"
            records = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    records.append(json.loads(line.strip()))
            
            # Create Hugging Face dataset
            dataset = Dataset.from_list(records)
            
            # Save in Hugging Face format
            hf_dataset_path = self.dataset_dir / "hf_dataset"
            dataset.save_to_disk(str(hf_dataset_path))
            
            print(f"✅ Hugging Face dataset saved: {hf_dataset_path}")
            return hf_dataset_path
            
        except Exception as e:
            print(f"❌ Error converting to HF format: {e}")
            print("💡 Make sure you have 'datasets' library installed: pip install datasets")
            return None
    
    def verify_audio_files(self, df):
        """Verify that all audio files exist and are accessible"""
        print("🔍 Verifying audio files...")
        
        valid_files = 0
        invalid_files = 0
        
        for idx, row in df.iterrows():
            audio_path = row['audio_path']
            
            try:
                if pd.isna(audio_path) or not audio_path.strip():
                    invalid_files += 1
                    continue
                
                full_path = str(audio_path).strip().replace('\\', '/')
                
                if os.path.exists(full_path):
                    # Try to load the audio file to verify it's valid
                    try:
                        audio, sr = torchaudio.load(full_path)
                        if audio.numel() > 0:  # Check if audio has data
                            valid_files += 1
                        else:
                            print(f"⚠️ Empty audio file: {os.path.basename(full_path)}")
                            invalid_files += 1
                    except Exception as e:
                        print(f"❌ Corrupted audio file: {os.path.basename(full_path)} - {e}")
                        invalid_files += 1
                else:
                    print(f"❌ Missing audio file: {full_path}")
                    invalid_files += 1
                    
            except Exception as e:
                print(f"❌ Error verifying {audio_path}: {e}")
                invalid_files += 1
        
        print(f"✅ Valid audio files: {valid_files}")
        print(f"❌ Invalid audio files: {invalid_files}")
        return valid_files, invalid_files
    
    def prepare_dataset(self):
        """Main method to prepare the complete Tamil dataset"""
        print("🚀 Starting Tamil Dataset Preparation")
        print("=" * 50)
        
        # Step 1: Load metadata
        df = self.load_metadata()
        
        # Step 2: Verify audio files
        valid_files, invalid_files = self.verify_audio_files(df)
        
        if valid_files == 0:
            print("❌ No valid audio files found! Cannot create dataset.")
            return None
        
        # Step 3: Copy audio files
        audio_files_copied = self.copy_audio_files(df)
        
        # Step 4: Create speaker embeddings
        speaker_file = self.create_speaker_embeddings()
        
        # Step 5: Create dataset JSONL
        dataset_records = self.create_dataset_jsonl(df, speaker_file)
        
        if len(dataset_records) == 0:
            print("❌ No valid dataset records created!")
            return None
        
        # Step 6: Create dataset info
        dataset_info = self.create_dataset_info(len(dataset_records))
        
        # Step 7: Convert to Hugging Face dataset format
        hf_dataset_path = self.convert_to_hf_dataset()
        
        print(f"\n🎉 Tamil Dataset Preparation Complete!")
        print(f"📁 Dataset location: {self.dataset_dir}")
        print(f"🎵 Total samples: {len(dataset_records)}")
        print(f"🔊 Audio files copied: {audio_files_copied}")
        
        return self.dataset_dir

def main():
    """Main function to prepare your Tamil dataset"""
    
    # Configuration - UPDATE THIS PATH
    config = {
        'metadata_path': "tamil_tokenizer_src/data/metadata.csv",  # Your metadata file
        'output_dir': "./tamil_dataset_ace_step",  # Output directory
        'tamil_tokenizer_path': "chkpts/tokenizer_ace"  # Your Tamil tokenizer
    }
    
    # Verify input path
    if not os.path.exists(config['metadata_path']):
        print(f"❌ Metadata file not found: {config['metadata_path']}")
        return
    
    # Prepare dataset
    preparer = TamilDatasetPreparer(**config)
    dataset_dir = preparer.prepare_dataset()
    
    if dataset_dir:
        print(f"\n🎯 Next Steps:")
        print(f"   1. Your dataset is ready at: {dataset_dir}")
        print(f"   2. Use this path in your training script:")
        print(f"      --dataset_path {dataset_dir}/hf_dataset")
        print(f"   3. Start training with the Tamil-modified trainer")
    else:
        print("\n❌ Dataset preparation failed!")

if __name__ == "__main__":
    main()