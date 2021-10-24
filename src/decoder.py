import torch
import kenlm
import numpy as np
from torchtyping import TensorType

from . import text
from .beam import ExternalBeamSearch


_decoder_alphabet = ''.join(text._idx2char).replace(text.empty_token_str, '_')
_decoder = ExternalBeamSearch(_decoder_alphabet)
_lm_model = kenlm.Model('data/4gram.bin')


def ctc_decode(probs: TensorType['batch', 'time', 'n_classes']):
    if isinstance(probs, list):
        probs = torch.tensor(probs)
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)

    lines = _decoder.decode(probs)
    chosen = []

    for batch in lines:
        batch = [text.decode(x) for x in batch]
        scores = [_lm_model.score(x) for x in batch]
        max_idx = np.argmax(scores)
        chosen.append(batch[max_idx])

    return chosen
