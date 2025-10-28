from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch
class DataLoader(Dataset):
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path

    def load_metadata(self):
        if self.metadata_path.endswith('.txt'):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            metadata = [line.strip().split('|') for line in lines]

        elif self.metadata_path.endswith('.csv'):
            df = pd.read_csv(self.metadata_path)
            metadata = df.values.tolist()
        else:
            raise ValueError("Unsupported metadata format (must be .txt or .csv)")

        return metadata

    def load_audio_data(self,path):
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate
    
    def load_lyrics_data(self,path):
        lyrics = None
        if path.endswith('.txt'):
            with open(path, 'r', encoding='utf-8') as f:
                lyrics = f.read()

        elif path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(path)
            lyrics = df.to_dict(orient='records')

        return lyrics

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        audio_path, lyrics_path = self.metadata[idx][:2]
        waveform, sr = self.load_audio_data(audio_path)
        lyrics = self.load_lyrics_data(lyrics_path)
        return {
            "audio": waveform,
            "sr": sr,
            "lyrics": lyrics
        }