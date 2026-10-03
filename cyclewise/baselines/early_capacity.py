"""Naive baseline: keep the cells with the highest discharge capacity near cycle 50
(mean of cycles 46..50). Label-free, so it applies identically to every batch."""

from cyclewise.tools.train_eval import score, select_top_k

RULE = {"kind": "zscore_sum", "weights": {"q50": 1.0}, "fit_on": "none"}


def select(batch: str, k: int, eligible: list[str]) -> dict:
    return select_top_k(score(RULE, batch, eligible), k, eligible)
