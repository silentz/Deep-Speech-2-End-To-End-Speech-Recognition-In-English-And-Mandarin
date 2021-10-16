import fastwer
from typing import List


def wer(pred: List[str], target: List[str]) -> float:
    return fastwer.score(pred, target) / 100


def cer(pred: List[str], target: List[str]) -> float:
    return fastwer.score(pred, target, char_level=True) / 100
