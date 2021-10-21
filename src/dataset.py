import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchaudio.datasets import LIBRISPEECH, LJSPEECH
from typing import Any, Callable, Dict
from .text import encode


class LibrispeechDataset(Dataset):

    def __init__(self, root: str,
                       url: str,
                       transforms: Callable):
        self.dataset = LIBRISPEECH(root=root, url=url, download=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wave, sample_rate, text, _, _, _ = self.dataset[idx]
        wave = torch.mean(wave, dim=0)
        wave = self.transforms(wave.numpy(), sample_rate=sample_rate)
        wave = torch.from_numpy(wave)
        text = torch.LongTensor(encode(text))
        return {
                'wave': wave,
                'wave_len': len(wave),
                'text': text,
                'text_len': len(text),
            }


class LJSpeechDataset(Dataset):

    def __init__(self, root: str, transforms: Callable):
        self.dataset = LJSPEECH(root=root, download=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wave, sample_rate, _, text = self.dataset[idx]
        wave = torch.mean(wave, dim=0)
        wave = self.transforms(wave.numpy(), sample_rate=sample_rate)
        wave = torch.from_numpy(wave)
        text = torch.LongTensor(encode(text))
        return {
                'wave': wave,
                'wave_len': len(wave),
                'text': text,
                'text_len': len(text),
            }


class PartialDataset(Dataset):

    def __init__(self, dataset: Dataset,
                       start_idx: int,
                       finish_idx: int):
        # interval has form: [start_idx; finish_idx)
        self.dataset = dataset
        self.start_idx = start_idx
        self.finish_idx = finish_idx

    def __getitem__(self, idx: int):
        return self.dataset[idx + self.start_idx]

    def __len__(self):
        return self.finish_idx - self.start_idx


def collate_fn(values: list) -> Dict[str, Any]:
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

