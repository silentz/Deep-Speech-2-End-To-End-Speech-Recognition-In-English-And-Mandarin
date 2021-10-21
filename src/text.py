import torch
import numpy as np

from typing import List
from torchtyping import TensorType
from fast_ctc_decode import beam_search


_idx2char = [
    '@', ' ', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
]

_char2idx = {char: idx for idx, char in enumerate(_idx2char)}

empty_token_str = '@'
empty_token_int = _char2idx[empty_token_str]


def encode(line: str) -> List[int]:
    line = line.lower().replace(empty_token_str, '')
    ids = [_char2idx[x] for x in line if x in _char2idx]
    return ids


def decode(vector: List[int]) -> str:
    line = ''.join(_idx2char[x] for x in vector)
    return line


def ctc_decode(probs: TensorType['time', 'n_classes'], size: int = 1):
    alphabet = ''.join(_idx2char)
    if isinstance(probs, list):
        probs = np.array(probs)
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    line, _ = beam_search(probs, alphabet, beam_size=size)
    return line
