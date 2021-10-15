import torch
import torchaudio
from src.dataset import LibriSpeechDataset
from src.model import ASRModel

df = LibriSpeechDataset('data/', 'train-clean-360')
model = ASRModel()
criterion = torch.nn.CTCLoss()

batch = torch.unsqueeze(df[1][0], dim=0)

tr = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=64,
    )

batch = tr(batch)
out = model(batch)

out = torch.transpose(out, 0, 1)
print(out.shape)
