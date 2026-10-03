"""Executable selection rules: score cells, pick top-K, fit a revised rule.

A rule is a JSON artifact, never prose:

  {"kind": "zscore_sum", "weights": {"dq_logvar": -1.0, ...}, "fit_on": "none"}
      score = sum_f w_f * (x_f - mean_f) / std_f, with mean/std from the batch being
      scored (label-free; a within-batch rank normalisation).

  {"kind": "ridge", "features": [...], "mean": {...}, "std": {...},
   "coef": {...}, "intercept": float, "alpha": float, "target": "log10_cycle_life",
   "fit_on": "b1_revealed", "n_train": int}
      score = predicted log10 cycle life. mean/std are frozen from the training data;
      a new batch is transformed, never re-fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

from cyclewise.data.featurize import FEATURE_CATALOG
from cyclewise.tools.cutoff_view import get_features

ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


class RuleError(ValueError):
    pass


def validate_rule(rule: dict) -> dict:
    kind = rule.get("kind")
    if kind == "zscore_sum":
        w = rule.get("weights") or {}
        if not w:
            raise RuleError("zscore_sum rule needs non-empty weights")
        for f, v in w.items():
            if f not in FEATURE_CATALOG:
                raise RuleError(f"unknown feature {f!r}")
            if not np.isfinite(float(v)):
                raise RuleError(f"non-finite weight for {f}")
    elif kind == "ridge":
        feats = rule.get("features") or []
        if not feats:
            raise RuleError("ridge rule needs features")
        for key in ("mean", "std", "coef"):
            if set(rule.get(key, {})) != set(feats):
                raise RuleError(f"ridge rule {key} keys must match features")
        if any(float(rule["std"][f]) <= 0 for f in feats):
            raise RuleError("ridge rule std must be > 0")
        float(rule["intercept"])
        for f in feats:
            if f not in FEATURE_CATALOG:
                raise RuleError(f"unknown feature {f!r}")
    else:
        raise RuleError(f"unknown rule kind {kind!r}")
    return rule


def score(rule: dict, batch: str, cell_ids: list[str] | None = None) -> pd.Series:
    validate_rule(rule)
    feats = list(rule["weights"]) if rule["kind"] == "zscore_sum" else rule["features"]
    X = get_features(batch, feats, cell_ids).set_index("cell_id")[feats].astype(float)
    # Impute any missing feature with the batch median (label-free).
    X = X.fillna(X.median())
    if rule["kind"] == "zscore_sum":
        Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1.0)
        s = sum(float(w) * Z[f] for f, w in rule["weights"].items())
    else:
        mean = pd.Series(rule["mean"]); std = pd.Series(rule["std"]); coef = pd.Series(rule["coef"])
        Z = (X[feats] - mean[feats]) / std[feats]
        s = Z.dot(coef[feats]) + float(rule["intercept"])
    return s.rename("score")


def select_top_k(scores: pd.Series, k: int, eligible: list[str] | None = None) -> dict:
    """Highest score first; ties broken by cell id (deterministic) and reported."""
    s = scores if eligible is None else scores[scores.index.isin(eligible)]
    df = pd.DataFrame({"cell_id": s.index, "score": s.values}).sort_values(
        ["score", "cell_id"], ascending=[False, True]).reset_index(drop=True)
    k = min(k, len(df))
    chosen = df.iloc[:k]["cell_id"].tolist()
    tie = False
    if 0 < k < len(df):
        tie = bool(np.isclose(df.loc[k - 1, "score"], df.loc[k, "score"]))
    return {"cell_ids": chosen, "tie_at_boundary": tie, "ranking": df["cell_id"].tolist()}


def stratified_explore(scores: pd.Series, k: int, n_explore: int, eligible: list[str] | None = None) -> dict:
    """Top (k - n_explore) by score, plus n_explore cells spread evenly over the rest of
    the score range, so the revealed outcomes span the score axis (better for re-fitting)."""
    top = select_top_k(scores, k - n_explore, eligible)
    rest = [c for c in top["ranking"] if c not in top["cell_ids"]]
    if n_explore > 0 and rest:
        idx = np.unique(np.linspace(0, len(rest) - 1, n_explore).round().astype(int))
        extra = [rest[i] for i in idx]
    else:
        extra = []
    return {"cell_ids": top["cell_ids"] + extra, "exploit_ids": top["cell_ids"],
            "explore_ids": extra, "tie_at_boundary": top["tie_at_boundary"]}


def fit_ridge(batch: str, revealed: pd.DataFrame, features: list[str], fit_on: str) -> dict:
    """Fit log10(cycle life) ~ standardized features on REVEALED cells only.
    Alpha chosen by leave-one-out on those same revealed cells (no other batch touched).
    Censored cells contribute their lower bound (documented limitation)."""
    for f in features:
        if f not in FEATURE_CATALOG:
            raise RuleError(f"unknown feature {f!r}")
    X = get_features(batch, features, revealed["cell_id"].tolist()).set_index("cell_id")[features].astype(float)
    y = np.log10(revealed.set_index("cell_id").loc[X.index, "cycle_life"].astype(float))
    if len(X) < len(features) + 2:
        raise RuleError(f"too few revealed cells ({len(X)}) for {len(features)} features")
    mean, std = X.mean(), X.std(ddof=0).replace(0, 1.0)
    Z = ((X - mean) / std).values
    best = None
    for a in ALPHAS:
        err = []
        for tr, te in LeaveOneOut().split(Z):
            m = Ridge(alpha=a).fit(Z[tr], y.values[tr])
            err.append((m.predict(Z[te])[0] - y.values[te][0]) ** 2)
        mse = float(np.mean(err))
        if best is None or mse < best[1]:
            best = (a, mse)
    m = Ridge(alpha=best[0]).fit(Z, y.values)
    rule = {
        "kind": "ridge", "features": features, "target": "log10_cycle_life",
        "mean": {f: float(mean[f]) for f in features}, "std": {f: float(std[f]) for f in features},
        "coef": {f: float(c) for f, c in zip(features, m.coef_)}, "intercept": float(m.intercept_),
        "alpha": best[0], "loo_rmse_log10": round(best[1] ** 0.5, 4),
        "fit_on": fit_on, "n_train": int(len(X)),
    }
    return validate_rule(rule)
