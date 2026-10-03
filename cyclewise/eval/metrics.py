"""Scoring harness (NOT an agent). Reads labels from the hidden DB after selections
are committed. Primary metric: recall of long-lived cells at the fixed cell budget.
Secondary: cells (and test cycles) needed to match the best baseline's recall."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cyclewise.config import load_config, path
from cyclewise.record.dblock import connect_ro


def load_labels() -> pd.DataFrame:
    with connect_ro(path(load_config()["storage"]["hidden_db"])) as con:
        df = con.execute("SELECT * FROM labels_hidden").df()
    df["long_lived"] = df["long_lived"].map(lambda v: None if pd.isna(v) else bool(v))
    return df


def scored(labels: pd.DataFrame, batch: str) -> pd.DataFrame:
    """Cells with a known label in this batch (censored-unknown cells excluded)."""
    return labels[(labels["batch"] == batch) & labels["label_known"]].set_index("cell_id")


def recall_at_k(selected: list[str], lab: pd.DataFrame) -> float:
    pos = lab.index[lab["long_lived"].astype(bool)]
    return float(np.isin(pos, selected).mean()) if len(pos) else float("nan")


def precision_at_k(selected: list[str], lab: pd.DataFrame) -> float:
    s = [c for c in selected if c in lab.index]
    return float(lab.loc[s, "long_lived"].astype(bool).mean()) if s else float("nan")


def cells_to_reach(ranking: list[str], lab: pd.DataFrame, target_recall: float) -> int | None:
    """Smallest k such that the top-k of this ranking reaches target_recall."""
    pos = set(lab.index[lab["long_lived"].astype(bool)])
    if not pos:
        return None
    found = 0
    r = [c for c in ranking if c in lab.index]
    for i, c in enumerate(r, start=1):
        found += c in pos
        if found / len(pos) >= target_recall - 1e-12:
            return i
    return None


def cycles_for(ranking: list[str], k: int | None, lab: pd.DataFrame) -> int | None:
    if k is None:
        return None
    r = [c for c in ranking if c in lab.index][:k]
    return int(lab.loc[r, "cycle_life"].sum())


def within_batch_labels(labels: pd.DataFrame, batch: str) -> pd.DataFrame:
    """Pre-registered secondary rule: >= the batch's own 75th percentile of cycle life.
    Censored cells give lower bounds; a censored cell below the threshold is unknown."""
    q = load_config()["preregistered"]["long_lived_quantile"]
    b = labels[labels["batch"] == batch].copy()
    thr = float(np.quantile(b["cycle_life"].astype(float), q))
    known = ~b["censored"] | (b["cycle_life"] >= thr)
    b = b[known].copy()
    b["long_lived"] = b["cycle_life"] >= thr
    return b.set_index("cell_id")
