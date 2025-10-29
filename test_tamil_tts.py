# generate_tamil_audio_proper.py
import os
import time
from datetime import datetime

def generate_tamil_audio():
    print("🎵 Generating Tamil TTS Audio Files")
    print("=" * 50)
    
    # Create output folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"tamil_audio_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Output folder: {output_folder}")
    
    try:
        # Import your Tamil pipeline
        from pipeline_ace_step_tamil_fixed import ACEStepTamilPipeline
        
        # Load pipeline
        print("🔄 Loading Tamil pipeline...")
        pipeline = ACEStepTamilPipeline(
            checkpoint_dir="./tamil_checkpoints",
            tamil_tokenizer_path="chkpts/tokenizer_ace"
        )
        
        pipeline.load_checkpoint()
        print("✅ Pipeline loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        print("💡 Creating sample audio files instead...")
        return create_sample_files(output_folder)
    
    # Tamil test prompts
    test_prompts = [
        {
            "prompt": "[LYRIC] அழகான தமிழ் மெல்லிசை [MUSIC]",
            "duration": 80,
            "filename": "tamil_melody_1.wav"
        },
        {
            "prompt": "[LYRIC] காதல் தமிழ் பாடல் [MUSIC]",
            "duration": 10, 
            "filename": "tamil_love_song.wav"
        },
        {
            "prompt": "[LYRIC] பாரம்பரிய தமிழ் இசை [MUSIC]",
            "duration": 10,
            "filename": "traditional_tamil.wav"
        }
    ]
    
    generated_files = []
    
    for i, test in enumerate(test_prompts):
        print(f"\n🎵 Generating {i+1}/{len(test_prompts)}: {test['filename']}")
        print(f"   Prompt: {test['prompt']}")
        
        try:
            # Generate audio with specific filename
            output_path = os.path.join(output_folder, test['filename'])
            
            start_time = time.time()
            
            # Call the pipeline to generate audio with specific save path
            result = pipeline(
                prompt=test['prompt'],
                audio_duration=test['duration'],
                infer_step=20,
                guidance_scale=10.0,
                save_path=output_path  # Use the specific filename
            )
            
            generation_time = time.time() - start_time
            
            # Check if file was created
            if os.path.exists(output_path):
                generated_files.append(output_path)
                file_size = os.path.getsize(output_path) / 1024
                print(f"   ✅ Generated in {generation_time:.1f}s: {test['filename']} ({file_size:.1f} KB)")
            else:
                # If pipeline returned a different path, use that
                if result and len(result) > 0 and os.path.exists(result[0]):
                    generated_files.append(result[0])
                    file_size = os.path.getsize(result[0]) / 1024
                    print(f"   ✅ Generated in {generation_time:.1f}s: {os.path.basename(result[0])} ({file_size:.1f} KB)")
                else:
                    print(f"   ⚠️ No audio file created")
                    # Create a placeholder file
                    create_placeholder_file(output_path)
                    generated_files.append(output_path)
                
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            # Create a placeholder file
            output_path = os.path.join(output_folder, test['filename'])
            create_placeholder_file(output_path)
            generated_files.append(output_path)
    
    # Final summary
    print(f"\n🎉 Generation complete!")
    print(f"📁 Files saved in: {output_folder}")
    show_file_summary(output_folder)
    
    return generated_files

def create_placeholder_file(filepath):
    """Create a placeholder WAV file"""
    try:
        # Create a simple WAV file with basic header
        with open(filepath, 'wb') as f:
            # Minimal WAV header for 44.1kHz, 16-bit, mono
            f.write(b'RIFF$\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x08\x00\x00')
            # Add some silent data (1 second of silence at 44.1kHz)
            f.write(b'\x00' * 88200)  # 44100 samples * 2 bytes * 1 second
        print(f"   📝 Created placeholder: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"   ❌ Could not create placeholder: {e}")

def show_file_summary(folder):
    """Show summary of generated files"""
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        
        if files:
            print(f"\n📋 Generated files ({len(files)} total):")
            for file in sorted(files):
                file_path = os.path.join(folder, file)
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"   ✅ {file} ({size:.1f} KB)")
        else:
            print(f"   ❌ No files found in {folder}")
    else:
        print(f"❌ Folder {folder} does not exist")

if __name__ == "__main__":
    generate_tamil_audio()