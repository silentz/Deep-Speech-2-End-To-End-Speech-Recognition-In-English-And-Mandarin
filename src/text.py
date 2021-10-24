from typing import List

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
