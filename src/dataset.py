import torch
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
from torchtyping import TensorType
from typing import Tuple
from .text import encode


class LibriSpeechDataset(Dataset):

    def __init__(self, root: str, url: str):
        self.dataset = LIBRISPEECH(root=root, url=url, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[TensorType['wave'], TensorType['text']]:
        wave, _, text, _, _, _ = self.dataset[idx]
        text = encode(text)
        return torch.mean(wave, dim=0), torch.LongTensor(text)
