"""v2 scoring harness (not an agent): recall at k_final and channel-cycles, for
First Fifty and every pre-registered comparator, with bootstrap CIs and the
pre-registered claim rules. Headline batch: b3 (when present).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import yaml

from cyclewise import prereg
from cyclewise.config import ROOT, path
from cyclewise.eval import bootstrap, tracking
from cyclewise.record import research_log
from cyclewise.record.dblock import connect_ro
from cyclewise.v2.allocation import Budget
from cyclewise.v2.data import HIDDEN, SCREEN

REPORTS = ROOT / "reports"
COMPARATORS = ("test_all_to_eol", "one_shot_dq50", "one_shot_dq100", "early_capacity50")


def _logvar(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    v = float(np.var(d))
    return float(np.log10(v)) if v > 0 else np.nan


def full_features(batch: str) -> pd.DataFrame:
    """Scorer-side features for the comparators (uses all data; agents never see this)."""
    with connect_ro(path(SCREEN)) as s, connect_ro(path(HIDDEN)) as h:
        q = s.execute("SELECT cell_id, cycle, qdlin FROM qdlin WHERE batch = ? AND cycle IN (10, 50)", [batch]).df()
        qk = h.execute("SELECT cell_id, cycle, qdlin FROM qdlin_ckpt WHERE batch = ? AND cycle = 100", [batch]).df()
        cap = s.execute("SELECT cell_id, avg(qd) q50 FROM cycles WHERE batch = ? AND cycle BETWEEN 46 AND 50 "
                        "AND NOT glitch GROUP BY 1", [batch]).df()
    curves = {(r.cell_id, r.cycle): np.asarray(r.qdlin, float) for r in pd.concat([q, qk]).itertuples()}
    rows = []
    for cid in sorted({c for c, _ in curves}):
        r = {"cell_id": cid}
        if (cid, 10) in curves:
            for c in (50, 100):
                if (cid, c) in curves:
                    r[f"dq_logvar_{c}"] = _logvar(curves[(cid, c)] - curves[(cid, 10)])
        rows.append(r)
    return pd.DataFrame(rows).merge(cap, on="cell_id", how="left").set_index("cell_id")


def labels(batch: str) -> pd.DataFrame:
    with connect_ro(path(HIDDEN)) as h:
        lab = h.execute("SELECT * FROM labels WHERE batch = ?", [batch]).df().set_index("cell_id")
    lab["long_lived"] = lab["long_lived"].map(lambda v: None if pd.isna(v) else bool(v))
    return lab


def secondary(lab: pd.DataFrame) -> pd.DataFrame:
    thr = yaml.safe_load((ROOT / "config" / "frozen.yaml").read_text())["long_lived_threshold_cycles"]
    out = lab.copy()
    out["label_known"] = ~out["censored"] | (out["cycle_life"] >= thr)
    out["long_lived"] = np.where(out["label_known"], out["cycle_life"] >= thr, None)
    return out


def top_k(series: pd.Series, k: int, ascending: bool) -> list[str]:
    df = pd.DataFrame({"c": series.index, "v": series.values}).dropna()
    return df.sort_values(["v", "c"], ascending=[ascending, True])["c"].head(k).tolist()


def evaluate(run_id: str) -> dict:
    pre = prereg.load()
    log = research_log.rows(run_id=run_id)
    start = next(r["output"] for r in log if r["event"] == "run_start")
    out = {"run_id": run_id, "version": 2, "prereg_commit": start.get("prereg_commit"),
           "prereg_sha256": start.get("prereg_sha256"), "llm_backend": start.get("llm_backend"),
           "demo_auto_approve": start.get("DEMO_AUTO_APPROVE"),
           "fallback_steps": sum(bool(r["fallback_used"]) for r in log),
           "revisions": [r["batch"] for r in log if r["event"] == "rule_revised"], "batches": {}}
    for batch, role in start["batches"].items():
        kept = [r["output"] for r in log if r["event"] == "kept_to_eol" and r["batch"] == batch]
        if not kept:
            continue
        lab = labels(batch)
        n = len(lab)
        b = Budget.for_batch(n)
        eol = lab["cycle_life"].astype(float)
        ext = [r["output"] for r in log if r["event"] == "extension_done" and r["batch"] == batch]
        cw_keep = kept[-1]["cells"]
        costs = {"cyclewise": int(eol.clip(upper=50).sum() + sum(e["cost"] for e in ext)
                                  + sum(eol[c] - 150 for c in cw_keep))}
        ff = full_features(batch)
        sels = {"cyclewise": cw_keep, "test_all_to_eol": lab.index.tolist()}
        sels["one_shot_dq50"] = top_k(ff["dq_logvar_50"], b.k_final, ascending=True)
        alive100 = ff.index[eol.reindex(ff.index) > 100]
        sels["one_shot_dq100"] = top_k(ff.loc[alive100, "dq_logvar_100"], b.k_final, ascending=True)
        sels["early_capacity50"] = top_k(ff["q50"], b.k_final, ascending=False)
        costs["test_all_to_eol"] = int(eol.sum())
        costs["one_shot_dq50"] = int(eol.clip(upper=50).sum() + sum(eol[c] - 50 for c in sels["one_shot_dq50"]))
        costs["one_shot_dq100"] = int(eol.clip(upper=100).sum() + sum(eol[c] - 100 for c in sels["one_shot_dq100"]))
        costs["early_capacity50"] = int(eol.clip(upper=50).sum() + sum(eol[c] - 50 for c in sels["early_capacity50"]))

        res = {"role": role, "n": n, "k_final": b.k_final, "extension_budget": b.extension,
               "extension_spent": sum(e["cost"] for e in ext),
               "continued_to_100": len(ext[0]["cells"]) if ext else 0,
               "continued_to_150": len(ext[1]["cells"]) if len(ext) > 1 else 0,
               "n_censored": int(lab["censored"].sum()), "costs": costs, "label_rules": {}}
        for rule_name, rl in (("primary_within_batch_q75", lab), ("secondary_absolute_v1", secondary(lab))):
            known = rl[rl["label_known"]]
            pos = int(known["long_lived"].astype(bool).sum())
            rr = {"n_scored": len(known), "n_long_lived": pos, "n_unknown": int((~rl["label_known"]).sum()),
                  "random_expected": b.k_final / max(1, len(known)), "oracle": min(1.0, b.k_final / pos) if pos else None,
                  "methods": {}}
            if pos == 0:
                rr["undefined"] = "no long-lived cells under this rule"
            else:
                cis = bootstrap.recall_ci(sels, known, reference="cyclewise")
                pos_ids = set(known.index[known["long_lived"].astype(bool)])
                for m, s in sels.items():
                    rec = len(pos_ids & set(s)) / pos
                    d = {"recall": rec, "ci_low": cis[m]["ci_low"], "ci_high": cis[m]["ci_high"],
                         "cost": costs[m], "selected": sorted(s)}
                    if m != "cyclewise":
                        x = cis[m]["diff_vs_cyclewise"]
                        d["cyclewise_minus_this"] = {"mean": -x["mean"], "ci_low": -x["ci_high"], "ci_high": -x["ci_low"]}
                        d["claim"] = claim(d["cyclewise_minus_this"], costs["cyclewise"], costs[m], pre)
                    rr["methods"][m] = d
            res["label_rules"][rule_name] = rr
        prim = res["label_rules"]["primary_within_batch_q75"]["methods"]
        for m, s in sels.items():  # one MLflow run per strategy per batch (selections already committed)
            d = prim.get(m, {})
            tracking.log_selection(f"v2_{m}_{batch}", batch, f"v2_{m}",
                                   {"run_id": run_id, "role": role, "k_final": b.k_final},
                                   {"recall_at_k": d.get("recall", float("nan")), "channel_cycles": costs[m]}, s)
        out["batches"][batch] = res

    REPORTS.mkdir(exist_ok=True)
    for name in (f"v2_{run_id}.json", "v2_latest.json"):
        (REPORTS / name).write_text(json.dumps(out, indent=2, default=str))
    (REPORTS / f"v2_{run_id}.md").write_text(to_markdown(out))
    research_log.append(run_id, "evaluator", "evaluation_v2", {
        b: {"costs": r["costs"], **{rn: {m: {k: v for k, v in d.items() if k != "selected"}
                                         for m, d in rr["methods"].items()} for rn, rr in r["label_rules"].items()}}
        for b, r in out["batches"].items()})
    return out


def claim(diff: dict, cost_cw: int, cost_other: int, pre: dict) -> str:
    """Pre-registered claim rules (prereg_v2.yaml analysis.claim_rules)."""
    if diff["ci_low"] > 0:
        return "better"
    if diff["ci_low"] > -0.10 and cost_cw < cost_other:
        return "non_inferior_and_cheaper"
    return "not_better"


def _f(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def to_markdown(out: dict) -> str:
    L = [f"# First Fifty results: {out['run_id']}", "",
         f"Pre-registration commit `{(out['prereg_commit'] or '')[:10]}` (sha256 `{(out['prereg_sha256'] or '')[:12]}`). "
         f"LLM backend: {out['llm_backend']}; fallback steps: {out['fallback_steps']}; "
         f"DEMO_AUTO_APPROVE={out['demo_auto_approve']}; rule revised after: {out['revisions'] or 'none'}.", ""]
    for batch, r in out["batches"].items():
        L += [f"## Batch {batch} ({r['role']})", "",
              f"N={r['n']}, k_final={r['k_final']}, extension budget {r['extension_budget']} channel-cycles "
              f"(spent {r['extension_spent']}); continued {r['continued_to_100']} to 100 and {r['continued_to_150']} to 150; "
              f"{r['n_censored']} censored.", ""]
        for rn, rr in r["label_rules"].items():
            L += [f"### {rn}", "", f"{rr['n_long_lived']} long-lived of {rr['n_scored']} scored "
                                   f"({rr['n_unknown']} unknown). Random {_f(rr['random_expected'])}, oracle {_f(rr['oracle'])}.", ""]
            if rr.get("undefined"):
                L += [f"**{rr['undefined']}.**", ""]
                continue
            L += ["| strategy | recall@k | 95% CI | channel-cycles | First Fifty − this (95% CI) | claim |",
                  "|---|---|---|---|---|---|"]
            for m, d in rr["methods"].items():
                diff = d.get("cyclewise_minus_this")
                L.append(f"| {m} | {_f(d['recall'])} | [{_f(d['ci_low'])}, {_f(d['ci_high'])}] | {d['cost']:,} | "
                         + (f"{diff['mean']:+.2f} [{diff['ci_low']:+.2f}, {diff['ci_high']:+.2f}]" if diff else "") + " | "
                         + d.get("claim", "") + " |")
            L.append("")
    return "\n".join(L)


def print_summary(out: dict) -> None:
    for batch, r in out["batches"].items():
        rr = r["label_rules"]["primary_within_batch_q75"]
        print(f"\nBatch {batch} ({r['role']}) [primary within-batch Q75]: {rr['n_long_lived']} long-lived")
        for m, d in rr["methods"].items():
            print(f"  {m:18s} recall {_f(d['recall'])} CI [{_f(d['ci_low'])}, {_f(d['ci_high'])}]  "
                  f"cycles {d['cost']:>7,}  {d.get('claim', '')}")
