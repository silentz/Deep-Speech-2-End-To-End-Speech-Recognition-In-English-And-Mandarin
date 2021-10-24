import numpy as np
from ctcdecode import CTCBeamDecoder


def my_beam_search(probs: np.ndarray,
                   alphabet: str,
                   empty_token: str,
                   beam_size: int) -> str:

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
                new_candidates.append((new_cand, score))

        new_candidates = sorted(new_candidates, key=lambda x: x[1])
        new_candidates = new_candidates[::-1][:beam_size]
        candidates = new_candidates

    result = candidates[0][0]
    result = clean_text(result)
    return result


class ExternalBeamSearch:

    def __init__(self, labels: str,
                       model_path: str = None,
                       alpha: float = 0.,
                       beta: float = 0.,
                       cutoff_top_n: int = 40,
                       cutoff_prob: float = 1.,
                       beam_width: int = 200,
                       num_processes: int = 32,
                       blank_id: int = 0,
                       log_probs_input: bool = True):
        self.decoder = CTCBeamDecoder(
            labels=labels,
            model_path=model_path,
            alpha=alpha,
            beta=beta,
            cutoff_top_n=cutoff_top_n,
            cutoff_prob=cutoff_prob,
            beam_width=beam_width,
            num_processes=num_processes,
            blank_id=blank_id,
            log_probs_input=log_probs_input,
        )

    def decode(self, probs):
        results, _, _, lens = self.decoder.decode(probs)
        batch_size, n_beams, _ = results.shape
        result = []

        for idx in range(batch_size):
            item = []
            for beam_idx in range(n_beams):
                beam = results[idx][beam_idx]
                beam = beam[:lens[idx][beam_idx]]
                item.append(beam)
            result.append(item)

        return result
