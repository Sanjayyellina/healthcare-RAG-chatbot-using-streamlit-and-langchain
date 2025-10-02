from __future__ import annotations
from typing import List
import numpy as np
from config import CFG

# Trust score = weighted blend of retrieval signal + self-check score
# retrieval signal: mean cosine similarity of top-k hits (requires passing similarities via metadata)
# self-check: LLM-produced 1-5 confidence mapped to 0-1


def score_from_similarities(sims: List[float]) -> float:
    if not sims:
        return 0.0
    sims_arr = np.clip(np.array(sims, dtype=float), -1.0, 1.0)
    sims01 = (sims_arr + 1.0) / 2.0
    return float(np.mean(sims01))


def blend_trust(sim_score: float, self_check_score_1to5: float) -> float:
    s_self = max(0.0, min(1.0, (self_check_score_1to5 - 1) / 4.0))
    w_sim, w_self = CFG["W_SIM"], CFG["W_SELF"]
    return float(np.clip(w_sim * sim_score + w_self * s_self, 0.0, 1.0))


SELF_CHECK_PROMPT = (
    "On a scale of 1 (low) to 5 (high), how confident are you that your answer is strictly "
    "supported by the provided context? Respond with only a single integer from 1 to 5. "
    "Context:\n\n{context}\n\nAnswer:\n{answer}"
)
