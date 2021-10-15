from typing import List


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
