"""Checkpoint features, computed ONLY from the run's visible DB (paid-for data).

At checkpoint c (50, 100 or 150), for each cell paid to at least c:
  dq_logvar      log10 var(Q_c(V) - Q_10(V))       base signal (Severson 2019), lower = longer life
  dq_logmin      log10 |min(Q_c - Q_10)|
  q_now          mean discharge capacity over cycles c-4..c (Ah)
  fade_slope     slope of Q vs cycle over cycles max(2, c-40)..c (Ah/cycle)
  ir_delta       IR(c) - IR(2) (ohm)
  tmax_mean      mean max temperature over cycles 2..c (C)
  c1, soc1, c2   charging protocol (known before testing)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cyclewise.data.featurize import parse_protocol
from cyclewise.record.dblock import locked
from cyclewise.v2 import visible

FEATURES = {
    "dq_logvar": {"desc": "log10 var(Q_c - Q_10); Severson 2019 base signal", "prior_sign": -1},
    "dq_logmin": {"desc": "log10 |min(Q_c - Q_10)|", "prior_sign": -1},
    "q_now": {"desc": "mean discharge capacity, cycles c-4..c (Ah)", "prior_sign": 1},
    "fade_slope": {"desc": "slope of capacity vs cycle over the last 40 cycles", "prior_sign": 1},
    "ir_delta": {"desc": "IR(c) - IR(2) (ohm)", "prior_sign": -1},
    "tmax_mean": {"desc": "mean max temperature, cycles 2..c (C)", "prior_sign": -1},
    "c1": {"desc": "first-step charge C-rate (protocol)", "prior_sign": -1},
    "soc1": {"desc": "SOC % where the charge rate switches (protocol)", "prior_sign": 0},
    "c2": {"desc": "second-step charge C-rate (protocol)", "prior_sign": -1},
}


def _log_abs(v: float) -> float:
    v = abs(float(v))
    return float(np.log10(v)) if v > 0 and np.isfinite(v) else np.nan


def at_checkpoint(run_id: str, batch: str, c: int, cell_ids: list[str] | None = None) -> pd.DataFrame:
    """Feature table at checkpoint c for alive cells paid to >= c (or the given subset)."""
    p = visible.paid(run_id, batch)
    alive = p[(p["paid_to"] >= c) & (p["eol"].isna() | (p["eol"] > c))]
    ids = sorted(set(alive["cell_id"]) & set(cell_ids)) if cell_ids is not None else sorted(alive["cell_id"])
    if not ids:
        return pd.DataFrame(columns=["cell_id", *FEATURES])
    q10 = visible.qdlin_at(run_id, ids, 10)
    qc = visible.qdlin_at(run_id, ids, c)
    cyc = visible.query_cycles(run_id, ids, ["qd", "ir", "tmax", "glitch"], c)
    with locked(visible.db_path(run_id)) as con:
        proto = dict(con.execute(f"SELECT cell_id, protocol FROM cells WHERE cell_id IN ({', '.join('?' * len(ids))})",
                                 ids).fetchall())
    rows = []
    for cid in ids:
        g = cyc[(cyc["cell_id"] == cid) & ~cyc["glitch"].astype(bool) & (cyc["cycle"] >= 2)]
        r = {"cell_id": cid}
        if cid in q10 and cid in qc:
            d = qc[cid] - q10[cid]
            d = d[np.isfinite(d)]
            r["dq_logvar"], r["dq_logmin"] = _log_abs(np.var(d)), _log_abs(np.min(d))
        r["q_now"] = float(g[g["cycle"] >= c - 4]["qd"].mean())
        w = g[g["cycle"] >= max(2, c - 40)]
        r["fade_slope"] = float(np.polyfit(w["cycle"], w["qd"], 1)[0]) if len(w) >= 3 else np.nan
        ir = g.set_index("cycle")["ir"].replace(0, np.nan).dropna()
        r["ir_delta"] = float(ir.iloc[-1] - ir.iloc[0]) if len(ir) > 1 else np.nan
        r["tmax_mean"] = float(g["tmax"].replace(0, np.nan).mean())
        r.update(parse_protocol(proto.get(cid, "")))
        rows.append(r)
    return pd.DataFrame(rows)


def score(rule: dict, feats: pd.DataFrame) -> pd.Series:
    """zscore_sum: z-scores within the scored cells (label-free). ridge: frozen mean/std/coef."""
    df = feats.set_index("cell_id")
    if rule["kind"] == "zscore_sum":
        cols = list(rule["weights"])
        X = df[cols].astype(float)
        X = X.fillna(X.median()).fillna(0.0)
        Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1.0).fillna(1.0)
        s = sum(float(w) * Z[f] for f, w in rule["weights"].items())
    elif rule["kind"] == "ridge":
        cols = rule["features"]
        X = df[cols].astype(float)
        X = X.fillna(pd.Series(rule["mean"]))
        Z = (X - pd.Series(rule["mean"])[cols]) / pd.Series(rule["std"])[cols]
        s = Z.dot(pd.Series(rule["coef"])[cols]) + float(rule["intercept"])
    else:
        raise ValueError(f"unknown rule kind {rule['kind']!r}")
    return pd.Series(s, index=df.index, name="score").astype(float)


def rank(scores: pd.Series) -> list[str]:
    """Highest score first; ties broken by cell id (deterministic)."""
    return pd.DataFrame({"cell_id": scores.index, "s": scores.values}).sort_values(
        ["s", "cell_id"], ascending=[False, True])["cell_id"].tolist()
