"""Evidence agent: decides which early signals to trust. Sees early data only."""

from __future__ import annotations

import json

from cyclewise.agents import citations, llm
from cyclewise.agents.schemas import FeatureClaim, Hypothesis
from cyclewise.config import load_config
from cyclewise.data.featurize import FEATURE_CATALOG
from cyclewise.record import research_log
from cyclewise.tools.cutoff_view import feature_summary, list_cells

SYSTEM = """You are the Evidence agent in CycleWise, an agentic battery-testing lab.
You see ONLY the first 50 cycles of each LFP/graphite cell (A123 APR18650M1A, fast-charge
protocols, 30 C). No outcome labels are available to you. Choose 1-6 early-cycle features
whose values you expect to separate long-lived cells from short-lived ones, with a sign
(+1 = higher value means longer life) and a weight in (0, 3]. Base this on published evidence
and on the feature distributions you are shown. Cite only ids from the allowed citation list.
Prior knowledge of published findings is allowed and must be cited; you do not know
per-cell outcomes."""


def fallback_hypothesis(previous_rule: dict | None = None) -> Hypothesis:
    if previous_rule and previous_rule.get("kind") == "ridge":
        feats = [FeatureClaim(name=f, direction=1 if c >= 0 else -1, weight=min(3.0, max(0.1, abs(c) * 10)))
                 for f, c in previous_rule["coef"].items()][:6]
        return Hypothesis(features=feats, citations=["severson2019"],
                          rationale="Carry forward the revised rule fit on batch 1 revealed cells; "
                                    "signs and relative weights come from its coefficients.")
    return Hypothesis(
        features=[FeatureClaim(name="dq_logvar", direction=-1, weight=1.0),
                  FeatureClaim(name="dq_logmin", direction=-1, weight=0.5),
                  FeatureClaim(name="qmax_minus_q2", direction=1, weight=0.5)],
        citations=["severson2019"],
        rationale="Severson et al. 2019 report that the variance of ΔQ(V) between an early and a "
                  "later cycle is strongly anti-correlated with log cycle life, and that min ΔQ and "
                  "early capacity rise add information. Weights are priors, not fit to labels.",
    )


def run(run_id: str, batch: str, previous_rule: dict | None = None, critique: dict | None = None) -> Hypothesis:
    n = len(list_cells(batch))
    summary = feature_summary(batch)
    prompt = (
        f"Batch {batch}: {n} eligible cells.\n"
        f"Feature catalog (name: description, literature prior sign):\n"
        + "\n".join(f"- {k}: {v['desc']} (prior {v['prior_sign']:+d})" for k, v in FEATURE_CATALOG.items())
        + f"\n\nLabel-free feature distribution in this batch:\n{json.dumps(summary, indent=1)}\n"
        f"\nAllowed citation ids: {sorted(citations.allowed())}\n"
    )
    if previous_rule:
        prompt += f"\nRule in force from the previous batch (fit on revealed batch-1 cells):\n{json.dumps(previous_rule)}\n"
    if critique:
        prompt += f"\nCritic's last critique:\n{json.dumps(critique)}\n"

    out = llm.complete(Hypothesis, SYSTEM, prompt,
                       fallback=lambda: fallback_hypothesis(previous_rule),
                       check=lambda h: citations.validate(h.citations))
    return record(run_id, batch, out.value, inputs={"batch": batch, "prev": previous_rule, "critique": critique},
                  model=out.model, fallback_used=out.fallback_used)


def record(run_id: str, batch: str, hyp: Hypothesis, inputs=None, model: str = "",
           fallback_used: bool = False) -> Hypothesis:
    citations.validate(hyp.citations)
    research_log.append(run_id, "evidence", "hypothesis", hyp.model_dump(), batch=batch, inputs=inputs,
                        model=model, prompt_version=load_config()["agents"]["prompt_version"],
                        fallback_used=fallback_used)
    return hyp
