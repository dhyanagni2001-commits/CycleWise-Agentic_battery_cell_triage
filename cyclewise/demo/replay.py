"""Replay a stored run step by step from the research log (no LLM calls, no labels
beyond what the run itself revealed).

  python -m cyclewise.demo.replay <run_id> [--pause]
  python -m cyclewise.demo.replay --list
"""

from __future__ import annotations

import argparse
import json
import time

from cyclewise.record import research_log

SHOW = {
    "run_start": lambda o: f"config {o.get('config_hash')} | LLM {o.get('llm_backend')} ({o.get('model')}) | "
                           f"budget {o.get('budget_cells')} cells | DEMO_AUTO_APPROVE={o.get('DEMO_AUTO_APPROVE')}",
    "hypothesis": lambda o: "features " + ", ".join(f"{f['name']}({f['direction']:+d}x{f['weight']})" for f in o["features"])
                            + f" | cites {o['citations']}\n      {o['rationale']}",
    "safety_flags": lambda o: f"{len(o)} flag(s) " + "; ".join(f"{f['cell_id']}:{f['flag']}" for f in o),
    "test_plan": lambda o: f"plan {o['plan_id']} options " + ", ".join(f"[{x['option_id']}] {x['strategy']}" for x in o["options"])
                           + f" -> chose {o['chosen']}: {o['reason']}\n      cells {o['cell_ids']}",
    "approval": lambda o: f"{o['decision'].upper()} by {o['approver']} {('(DEMO_AUTO_APPROVE)' if o.get('demo_auto_approve') else '')} {o.get('note', '')}",
    "selection_committed": lambda o: f"committed {len(o['cell_ids'])} cells, hash {o['selection_hash']} (before reveal)",
    "reveal": lambda o: f"revealed {len(o['outcomes'])} kept cells",
    "result": lambda o: "metrics " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in o["metrics"].items()})
                        + f" | mlflow {o.get('mlflow_run_id')}",
    "critique": lambda o: f"trigger fired={o['trigger']['fired']} revise={o['revise']}: {o['reason']}"
                          + (f"\n      new rule: {o['new_rule']['features']}" if o.get("new_rule") else "")
                          + (f"\n      next experiment: {o['next_experiment']}" if o.get("next_experiment") else ""),
    "rule_revised": lambda o: f"rule now ridge on {o['features']} (fit_on={o['fit_on']}, n={o['n_train']})",
}


def replay(run_id: str, pause: bool = False) -> None:
    rows = research_log.rows(run_id=run_id)
    if not rows:
        raise SystemExit(f"no rows for run {run_id}")
    for r in rows:
        fmt = SHOW.get(r["event"], lambda o: json.dumps(o, default=str)[:300])
        fb = " [fallback]" if r["fallback_used"] else ""
        print(f"#{r['seq']:<4} {r['batch'] or '--':3} {r['agent']:>13} | {r['event']}{fb}\n      {fmt(r['output'])}")
        if pause:
            time.sleep(1.2)
    print(f"\nlog hash chain intact: {research_log.verify()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--pause", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.run_id:
        for r in research_log.rows(event="run_start"):
            print(r["run_id"], r["ts"], r["output"].get("llm_backend"))
        return
    replay(a.run_id, a.pause)


if __name__ == "__main__":
    main()
