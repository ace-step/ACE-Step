import os 
import pandas as pd

def convert_txt_to_csv(txt_path: str, csv_path: str ="metadata.csv"):
    try:
        if os.path.exists(txt_path):
            files_list = os.listdir(txt_path)

            prompt_files = [f for f in files_list if f.endswith('_prompt.txt')]
            audio_files = [f for f in files_list if f.endswith((".MP3", ".WAV", ".FLAC"))]
            lyric_files = [f for f in files_list if f.endswith('_lyric.txt')]

            datasets={
                "prompts": [prompt_files],
                "audio_files": [audio_files],
                "lyrics": [lyric_files]
            }

            df = pd.DataFrame(datasets)
            df.to_csv(csv_path, index=False)
            print(f"✅ Converted TXT to CSV at: {csv_path}")

        else:
            print(f"❌ Error: The directory {txt_path} does not exist.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    csv_path = os.getcwd() + "/metadata.csv"
    convert_txt_to_csv(txt_path="data/tamil_dataset/songs" , csv_path=csv_path)