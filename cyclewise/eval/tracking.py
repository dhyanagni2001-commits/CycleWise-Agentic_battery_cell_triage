"""MLflow logging. Every baseline and every agent-chosen test is a run.

Tracking URI: MLFLOW_TRACKING_URI if set (e.g. "databricks"), else the local
SQLite store from config. Set CYCLEWISE_MLFLOW=off to disable (tests).
"""

from __future__ import annotations

import json
import os

from cyclewise.config import ROOT, load_config


def _enabled() -> bool:
    return os.environ.get("CYCLEWISE_MLFLOW", "on") != "off"


def _setup():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    cfg = load_config()["storage"]
    uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow_tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{ROOT / uri.removeprefix('sqlite:///')}"
    mlflow.set_tracking_uri(uri)
    name = os.environ.get("CYCLEWISE_MLFLOW_EXPERIMENT", cfg["mlflow_experiment"])
    if uri.startswith("databricks") and not name.startswith("/"):
        name = f"/Shared/{name}"  # Databricks experiments need an absolute workspace path
    mlflow.set_experiment(name)
    return mlflow


def log_selection(name: str, batch: str, method: str, params: dict, metrics: dict,
                  selected: list[str], rule: dict | None = None) -> str | None:
    if not _enabled():
        return None
    mlflow = _setup()
    with mlflow.start_run(run_name=name) as r:
        mlflow.set_tags({"batch": batch, "method": method})
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})  # drop NaN
        mlflow.log_text(json.dumps(sorted(selected)), "selected_cell_ids.json")
        if rule is not None:
            mlflow.log_text(json.dumps(rule, indent=2), "rule.json")
        return r.info.run_id
