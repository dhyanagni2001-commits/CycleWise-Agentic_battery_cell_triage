"""Pydantic schemas for every agent handoff. Every agent output is validated here."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from cyclewise.data.featurize import FEATURE_CATALOG


class FeatureClaim(BaseModel):
    name: str
    direction: Literal[-1, 1]  # sign of association with long life
    weight: float = Field(1.0, gt=0, le=3)

    @field_validator("name")
    @classmethod
    def known_feature(cls, v: str) -> str:
        if v not in FEATURE_CATALOG:
            raise ValueError(f"unknown feature {v!r}; choose from {sorted(FEATURE_CATALOG)}")
        return v


class Hypothesis(BaseModel):
    features: list[FeatureClaim] = Field(min_length=1, max_length=6)
    rationale: str = Field(min_length=20)
    citations: list[str] = Field(min_length=1)
    label: Literal["agent-generated"] = "agent-generated"

    def as_rule(self) -> dict:
        return {"kind": "zscore_sum", "fit_on": "none",
                "weights": {f.name: f.direction * f.weight for f in self.features}}


class TestOption(BaseModel):
    __test__ = False  # not a pytest class
    option_id: str
    strategy: Literal["exploit_top_k", "explore_stratified", "hedge_with_dq"]
    description: str
    n_cells: int = Field(gt=0)
    n_explore: int = Field(0, ge=0)
    expected_learning: str
    est_cost_cycles: Optional[int] = None


class TestPlanDraft(BaseModel):
    """What the Planner LLM returns. Cell IDs are filled in by tools, not the LLM."""
    __test__ = False  # not a pytest class
    options: list[TestOption] = Field(min_length=2)
    chosen: str
    reason: str = Field(min_length=10)

    @model_validator(mode="after")
    def chosen_is_an_option(self):
        ids = [o.option_id for o in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError("option_id values must be unique")
        if self.chosen not in ids:
            raise ValueError(f"chosen {self.chosen!r} is not one of {ids}")
        return self


class TestPlan(BaseModel):
    __test__ = False  # not a pytest class
    plan_id: str
    batch: str
    options: list[TestOption] = Field(min_length=2)
    chosen: str
    reason: str
    cost_cells: int
    est_cost_cycles: Optional[int] = None
    expected_learning: str
    cell_ids: list[str]
    rule: dict
    flagged_cells: list[dict] = []
    tie_at_boundary: bool = False


class Approval(BaseModel):
    plan_id: str
    decision: Literal["approved", "rejected"]
    approver: str
    note: str = ""
    demo_auto_approve: bool = False
    excluded_flagged: list[str] = []


class Result(BaseModel):
    run_id: str
    plan_id: str
    batch: str
    mlflow_run_id: Optional[str]
    selected_ids: list[str]
    revealed_outcomes: list[dict]
    metrics: dict


class CritiqueDraft(BaseModel):
    """What the Critic LLM returns."""
    revise: bool
    reason: str = Field(min_length=10)
    new_features: list[str] = []
    flagged_cells: list[str] = []
    next_experiment: str = ""

    @field_validator("new_features")
    @classmethod
    def known(cls, v: list[str]) -> list[str]:
        bad = [f for f in v if f not in FEATURE_CATALOG]
        if bad:
            raise ValueError(f"unknown features {bad}")
        return v


class Critique(BaseModel):
    revise: bool
    reason: str
    trigger: dict
    new_rule: Optional[dict] = None
    flagged_cells: list[str] = []
    next_experiment: str = ""
