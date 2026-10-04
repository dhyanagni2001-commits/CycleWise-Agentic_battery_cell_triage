"""Severson-style ΔQ baseline.

Severson's "variance model" is log10(cycle life) = a + b * log10 var(ΔQ(V)) with b < 0.
For SELECTION only the ranking matters, and a one-feature linear model with a
negative slope ranks cells exactly by ascending log var(ΔQ). So the baseline is
label-free and identical for both batches: keep the K cells with the lowest
var(ΔQ_{50-10}). (Our window is cycles 50-10, not Severson's 100-10.)

`fit_supervised` is a label-privileged REFERENCE, not a baseline: the same model
fit on ALL batch-1 labels (information First Fifty never gets), used to show what
full training labels would buy on batch 2.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from cyclewise.tools.cutoff_view import get_features
from cyclewise.tools.train_eval import score, select_top_k

RULE = {"kind": "zscore_sum", "weights": {"dq_logvar": -1.0}, "fit_on": "none"}


def select(batch: str, k: int, eligible: list[str]) -> dict:
    return select_top_k(score(RULE, batch, eligible), k, eligible)


def fit_supervised(train_batch: str, labels) -> dict:
    """Fit on every known-label cell of the train batch (reference only)."""
    lab = labels[(labels["batch"] == train_batch) & ~labels["censored"]]
    X = get_features(train_batch, ["dq_logvar"], lab["cell_id"].tolist()).set_index("cell_id")
    y = np.log10(lab.set_index("cell_id").loc[X.index, "cycle_life"].astype(float))
    m = LinearRegression().fit(X[["dq_logvar"]].values, y.values)
    return {"kind": "ridge", "features": ["dq_logvar"], "mean": {"dq_logvar": 0.0},
            "std": {"dq_logvar": 1.0}, "coef": {"dq_logvar": float(m.coef_[0])},
            "intercept": float(m.intercept_), "alpha": 0.0, "fit_on": f"{train_batch}_all_labels",
            "n_train": int(len(X)), "target": "log10_cycle_life"}
