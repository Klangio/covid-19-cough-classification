import numpy as np
import librosa
import torch

# helper functions
def preproces(fn_wav):
    y, sr = librosa.load(fn_wav, mono=True, duration=5)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    feature_row = {        
        "chroma_stft": np.mean(chroma_stft),
        "rmse": np.mean(rmse),
        "spectral_centroid": np.mean(spectral_centroid),
        "spectral_bandwidth": np.mean(spectral_bandwidth),
        "rolloff": np.mean(rolloff),
        "zero_crossing_rate": np.mean(zcr),        
    }
    for i, c in enumerate(mfcc):
        feature_row[f'mfcc{i+1}'] = np.mean(c)

    return feature_row


class CoughNet(torch.nn.Module):
    def __init__(self, input_size):
        super(CoughNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
        self.l6 = torch.nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))
        x = torch.relu(self.l5(x))
        x = self.l6(x)
        return x