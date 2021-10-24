import torch
import numpy as np

from typing import List
from torchtyping import TensorType
from fast_ctc_decode import beam_search as lib_beam_search
from .beam import my_beam_search, CustomDecoder


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

_decoder_alphabet = ''.join(_idx2char).replace(empty_token_str, '_')
_decoder = CustomDecoder(_decoder_alphabet)

def ctc_decode(probs: TensorType['batch', 'time', 'n_classes'], size: int = 1):
    #  alphabet = ''.join(_idx2char)
    if isinstance(probs, list):
        probs = torch.tensor(probs)
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)
    #  line = lib_beam_search(probs, alphabet, beam_size=size)
    #  line = my_beam_search(probs, alphabet, empty_token=empty_token_str, beam_size=size)
    lines = _decoder.decode(probs)
    lines = [decode(x) for x in lines]
    return lines
