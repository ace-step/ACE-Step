import os 
import pandas as pd

def convert_txt_to_csv(txt_path: str, csv_path: str ="metadata.csv"):
    try:
        if os.path.exists(txt_path):
            files_list = os.listdir(txt_path)

            prompt_files = [f for f in files_list if f.endswith('_prompt.txt')]
            audio_files = [f for f in files_list if f.endswith((".MP3", ".WAV", ".FLAC" , ".mp3", ".wav", ".flac"))]
            lyric_files = [f for f in files_list if f.endswith('_lyric.txt')]

            print(f"Found {len(prompt_files)} prompt files.")
            print(f"Found {len(audio_files)} audio files.")
            print(f"Found {len(lyric_files)} lyric files.")

            datasets={
                "prompt": [],
                "audio_path":[],
                "lyrics": []
            }
            #adding path in datasets
            for audio_files , prompt_files , lyric_files in zip(audio_files , prompt_files , lyric_files):
                datasets["audio_path"].append(os.path.join(txt_path , audio_files))
                datasets["prompt"].append(os.path.join(txt_path , prompt_files))
                datasets["lyrics"].append(os.path.join(txt_path , lyric_files))

            df = pd.DataFrame(datasets)
            df.to_csv(csv_path, index=False)
            print(f"✅ Converted TXT to CSV at: {csv_path}")

        else:
            print(f"❌ Error: The directory {txt_path} does not exist.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    csv_path = os.getcwd() + "\\tamil_tokenizer_src\\data\\metadata.csv"
    txt_path = os.getcwd() + "\\tamil_tokenizer_src\\data\\tamil_dataset\\"
    print(f"CSV will be saved at: {csv_path}")

    convert_txt_to_csv(txt_path=txt_path , csv_path=csv_path)