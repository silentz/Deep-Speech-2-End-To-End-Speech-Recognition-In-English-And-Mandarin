import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchaudio.datasets import LIBRISPEECH, LJSPEECH
from typing import Any, Callable, Dict
from .text import encode


class LibrispeechDataset(Dataset):

    def __init__(self, root: str,
                       url: str,
                       transforms: Callable,
                       spect_transform: Callable):
        self.dataset = LIBRISPEECH(root=root, url=url, download=True)
        self.transforms = transforms
        self.spect_transform = spect_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wave, sample_rate, text, _, _, _ = self.dataset[idx]
        wave = torch.mean(wave, dim=0)
        wave = self.transforms(wave.numpy(), sample_rate=sample_rate)
        wave = torch.from_numpy(wave)
        spectrogram = self.spect_transform(wave)
        text = torch.LongTensor(encode(text))
        return {
                'wave': wave,
                'wave_len': len(wave),
                'text': text,
                'text_len': len(text),
            }


class LJSpeechDataset(Dataset):

    def __init__(self, root: str):
        self.dataset = LJSPEECH(root=root, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wave, _, text, _ = self.dataset[idx]
        wave = torch.mean(wave, dim=0)
        text = torch.LongTensor(encode(text))
        return {
                'wave': wave,
                'wave_len': len(wave),
                'text': text,
                'text_len': len(text),
            }


def librispeech_collate_fn(values: list) -> Dict[str, Any]:
    waves = [x['wave'] for x in values]
    texts = [x['text'] for x in values]

    waves_len = [x['wave_len'] for x in values]
    texts_len = [x['text_len'] for x in values]

    waves = pad_sequence(waves, batch_first=True, padding_value=0)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)

    return {
            'waves': waves,
            'waves_len': torch.LongTensor(waves_len),
            'texts': texts,
            'texts_len': torch.LongTensor(texts_len),
        }

