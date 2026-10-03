"""Critic/Safety agent.

Safety (before planning): anomaly scan -> flags routed to human approval.
Critic (after reveal): decides whether to revise the rule. Revision is allowed
only when the pre-registered trigger fires, at most once per batch, and only
using revealed cells of the train batch. The new rule is an executable artifact.
"""

from __future__ import annotations

import json

import pandas as pd

from cyclewise.agents import llm
from cyclewise.agents.schemas import Critique, CritiqueDraft, Result
from cyclewise.config import load_config
from cyclewise.record import research_log
from cyclewise.tools import anomaly
from cyclewise.tools.train_eval import RuleError, fit_ridge

SYSTEM = """You are the Critic/Safety agent in CycleWise. You review the outcome of a batch of
kept cells. A pre-registered trigger decides whether a revision is permitted; you cannot revise
when it has not fired. When revising, name the early features (from the catalog) the new rule
should be fit on; the fit itself is done by a tool on the revealed cells only. Keep the feature
count small relative to the number of revealed cells. Also propose the next experiment."""


def safety_scan(run_id: str, batch: str) -> list[dict]:
    flags = anomaly.scan(batch)
    research_log.append(run_id, "critic_safety", "safety_flags", flags, batch=batch, inputs=batch)
    return flags


def trigger(metrics: dict) -> dict:
    t = load_config()["preregistered"]["revision_trigger"]
    low_p = metrics["precision_at_k"] < t["min_precision_at_k"]
    low_r = not (metrics["spearman_revealed"] >= t["min_spearman_revealed"])  # NaN counts as low
    return {"fired": bool(low_p or low_r), "precision_at_k": metrics["precision_at_k"],
            "spearman_revealed": metrics["spearman_revealed"], "thresholds": t,
            "low_precision": bool(low_p), "low_rank_corr": bool(low_r)}


def run(run_id: str, result: Result, rule: dict, role: str, revisions_used: int) -> Critique:
    cfg = load_config()
    trig = trigger(result.metrics)
    can_revise = role == "train" and trig["fired"] and revisions_used < cfg["preregistered"]["max_revisions_per_batch"]
    base_feats = list(rule["weights"]) if rule["kind"] == "zscore_sum" else rule["features"]
    max_feats = max(1, len([o for o in result.revealed_outcomes]) // 4)

    def fb() -> CritiqueDraft:
        return CritiqueDraft(
            revise=can_revise,
            reason=("trigger fired on revealed cells; re-fit weights on batch-1 revealed outcomes"
                    if can_revise else
                    "no revision: trigger did not fire" if not trig["fired"] else
                    f"trigger fired but revision is not permitted (role={role}; rules are only "
                    f"revised on the train batch, max {cfg['preregistered']['max_revisions_per_batch']})"),
            new_features=base_feats[:max_feats],
            next_experiment=("Prospective run on new cells with sequential stop/continue checkpoints "
                             "at cycles 50/100/150, then repeat on a second chemistry."),
        )

    def check(d: CritiqueDraft) -> None:
        if d.revise and not can_revise:
            raise ValueError("revise=true is not permitted: the pre-registered trigger did not fire, "
                             "or this is not the train batch, or the revision limit is used")
        if d.revise and not (1 <= len(d.new_features) <= max_feats):
            raise ValueError(f"choose 1..{max_feats} features for {len(result.revealed_outcomes)} revealed cells")

    prompt = (f"Batch {result.batch} (role={role}). Rule used: {json.dumps(rule)}\n"
              f"Revealed outcomes of kept cells: {json.dumps(result.revealed_outcomes, default=str)}\n"
              f"Metrics on revealed cells: {json.dumps(result.metrics)}\n"
              f"Pre-registered trigger: {json.dumps(trig)}\nRevision permitted: {can_revise}\n")
    out = llm.complete(CritiqueDraft, SYSTEM, prompt, fallback=fb, check=check)
    return finalize(run_id, result, rule, out.value, trig, model=out.model, fallback_used=out.fallback_used)


def permitted(result: Result, role: str, revisions_used: int) -> tuple[dict, bool, int]:
    cfg = load_config()
    trig = trigger(result.metrics)
    can = role == "train" and trig["fired"] and revisions_used < cfg["preregistered"]["max_revisions_per_batch"]
    return trig, can, max(1, len(result.revealed_outcomes) // 4)


def finalize(run_id: str, result: Result, rule: dict, d: CritiqueDraft, trig: dict,
             model: str = "", fallback_used: bool = False) -> Critique:
    """Fit the revised rule (if any) on revealed cells and log the critique."""
    cfg = load_config()
    base_feats = list(rule["weights"]) if rule["kind"] == "zscore_sum" else rule["features"]
    max_feats = max(1, len(result.revealed_outcomes) // 4)
    new_rule, reason = None, d.reason
    if d.revise:
        revealed = pd.DataFrame(result.revealed_outcomes)
        try:
            new_rule = fit_ridge(result.batch, revealed, d.new_features or base_feats[:max_feats],
                                 fit_on=f"{result.batch}_revealed")
        except RuleError as e:
            reason += f" | revision abandoned: {e}"
    crit = Critique(revise=new_rule is not None, reason=reason, trigger=trig, new_rule=new_rule,
                    flagged_cells=d.flagged_cells, next_experiment=d.next_experiment)
    research_log.append(run_id, "critic_safety", "critique", crit.model_dump(), batch=result.batch,
                        inputs={"plan": result.plan_id, "metrics": result.metrics},
                        model=model, prompt_version=cfg["agents"]["prompt_version"],
                        fallback_used=fallback_used)
    return crit
