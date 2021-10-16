import torch
import numpy as np

from typing import List
from torchtyping import TensorType
from fast_ctc_decode import beam_search


_idx2char = [
    '@', ' ', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
]

_char2idx = {char: idx for idx, char in enumerate(_idx2char)}

empty_token_str = '@'
empty_token_int = _char2idx[empty_token_str]


def encode(line: str) -> List[int]:
    ids = [_char2idx[x] for x in line]
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
