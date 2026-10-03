"""Bootstrap CIs over cells (fixed seed). Selections are held fixed; cells are
resampled with replacement within a batch, and recall is recomputed on the
resample. Paired differences use the same resamples for both methods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cyclewise.config import load_config


def _recall_vec(selected: set[str], ids: np.ndarray, pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
    sel = np.array([c in selected for c in ids])
    p = pos[idx]                       # (B, n) positives in each resample
    hit = (p & sel[idx]).sum(axis=1)
    tot = p.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, hit / tot, np.nan)


def recall_ci(selections: dict[str, list[str]], lab: pd.DataFrame, reference: str | None = None) -> dict:
    cfg = load_config()["preregistered"]
    rng = np.random.default_rng(cfg["seed"])
    ids = lab.index.to_numpy()
    pos = lab["long_lived"].astype(bool).to_numpy()
    n, B = len(ids), cfg["bootstrap_resamples"]
    idx = rng.integers(0, n, size=(B, n))
    a = (1 - cfg["ci_level"]) / 2
    vecs = {m: _recall_vec(set(s), ids, pos, idx) for m, s in selections.items()}
    out = {}
    for m, v in vecs.items():
        out[m] = {"ci_low": float(np.nanquantile(v, a)), "ci_high": float(np.nanquantile(v, 1 - a))}
        if reference and m != reference:
            d = v - vecs[reference]
            out[m][f"diff_vs_{reference}"] = {
                "mean": float(np.nanmean(d)), "ci_low": float(np.nanquantile(d, a)),
                "ci_high": float(np.nanquantile(d, 1 - a)), "p_le_0": float(np.nanmean(d <= 0))}
    return out
