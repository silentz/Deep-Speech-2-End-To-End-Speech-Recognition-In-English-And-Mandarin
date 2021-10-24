import numpy as np
from typing import (
    Callable,
    List,
)


def beam_search(probs: np.ndarray,
                alphabet: str,
                empty_token: str,
                beam_size: int,
                language_model: Callable = None) -> str:

    def clean_text(text: str) -> str:
        text = empty_token + text
        unique_ids = [x for x in range(1, len(text)) if text[x - 1] != text[x]]
        text = [text[x] for x in unique_ids]
        text = ''.join(text).replace(empty_token, '')
        return text

    candidates = [('', 1.)]
    length = probs.shape[0]
    letters = probs.shape[1]

    for idx in range(length):
        new_candidates = []

        for cand, cand_prob in candidates:
            for letter_idx in range(letters):
                new_cand = cand + alphabet[letter_idx]
                score = cand_prob + np.log(probs[idx, letter_idx])
                #  if language_model is not None and len(new_cand) > 10:
                #      tmp_text = clean_text(new_cand)
                #      lm_score = language_model.score(tmp_text, bos=True, eos=False)
                #      score += lm_score
                new_candidates.append((new_cand, score))

        new_candidates = sorted(new_candidates, key=lambda x: x[1])
        new_candidates = new_candidates[::-1][:beam_size]
        candidates = new_candidates

    result = candidates[0][0]
    result = clean_text(result)
    return result

