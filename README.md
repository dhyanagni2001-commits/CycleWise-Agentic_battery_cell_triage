# CycleWise

**An agentic battery lab that decides which cells deserve a full life test, after seeing only their first 50 cycles.**

Four specialist agents, orchestrated by Omnigent, propose and run selection rules under a fixed test budget. A human approves every plan before any outcome is revealed. Built for Hack-Nation x Databricks, Challenge 03 "Agentic Scientific Discovery", on the Severson et al. 2019 battery dataset.

The project has two rounds, and the second exists because the first failed:
- **v1** made one decision at cycle 50 and lost to the published ΔQ rule. That result is kept below unchanged.
- **v2** changes the experiment design. It adopts ΔQ as the base signal and makes allocation decisions at checkpoints 50, 100 and 150 under a channel-cycle budget. It was pre-registered in git before its test batch (batch 3) was opened.

## Status at a glance

| | |
|---|---|
| Pipeline (data → features → agents → evaluation) | ✅ Runs end to end from a clean state; reproducible run to run |
| Tests | ✅ 65 passing (leakage, gates, LLM path, Omnigent bundle, concurrency, v2 paid-data control) |
| v1 result (one decision at cycle 50) | ❌ CycleWise does **not** beat the ΔQ baseline (see below) |
| v2 pre-registration | ✅ Committed (`e5725a5`), hash-locked. ⏳ Not yet pushed to GitHub. |
| v2 sequential loop | ✅ Runs end to end on b1/b2 (development only); ⏳ final test on b3 not run yet, b3 not downloaded |
| Live Claude agents via Omnigent | ⚠️ Run as far as the human approval step; full live results not measured yet |
| Databricks (Delta, Unity Catalog, dashboard) | ⚠️ Scripts written, untested (no workspace available) |

## Question

A battery lab can keep only some cells on test. If an agentic lab sees only the first 50 cycles of each cell, can it choose which cells to keep testing and retain more of the truly long-lived cells than (a) a naive early-capacity rule and (b) a Severson-style ΔQ-variance rule?

**Bottleneck.** Cycling a cell to end of life takes hundreds to thousands of cycles, which means weeks of channel time per cell. Choosing which cells earn that time is a decision made with almost no data.

## v1 result: one decision at cycle 50 (measured, honest)

From run `final-check` (`reports/final-check.md`). Budget: 12 cells per batch. Metric: recall of long-lived cells, with 95% bootstrap CIs (2,000 resamples, fixed seed).

| Batch / label rule | CycleWise | Early capacity | ΔQ variance | Random |
|---|---|---|---|---|
| b1, primary (≥ batch-1 Q75) | 0.42 [0.12, 0.71] | 0.17 [0.00, 0.42] | **0.75** [0.50, 1.00] | 0.29 |
| b2, primary | undefined: 0 long-lived cells | | | |
| b2, secondary (≥ within-batch Q75) | 0.42 [0.12, 0.71] | 0.42 [0.12, 0.70] | **0.50** [0.20, 0.80] | 0.28 |

**CycleWise did not beat the ΔQ baseline.**
- On batch 1 it was worse: −0.34 recall, CI [−0.62, −0.08]. The Planner chose an exploration strategy (8 top cells plus 4 spread across the score range) to learn for a revision. The pre-registered revision trigger then did not fire (precision 0.50 was not below 0.50), so that exploration bought nothing.
- On batch 2 it tied early capacity and trailed ΔQ by −0.08 (CI [−0.27, −0.00]).
- Measured acceleration: matching ΔQ's recall needed 12 cells for CycleWise vs 11 for ΔQ on b1, and 13 vs 12 on b2. That is **0.92x, i.e. slower, not faster.**

**What this run is.** Every agent decision came from the deterministic offline policy; no LLM was called (6 steps logged `fallback_used=true`). Approvals were `DEMO_AUTO_APPROVE=true`, logged as such. Two runs from a clean state give identical selections and numbers.

**Primary metric undefined on batch 2.** The long-lived threshold was pre-registered as the 75th percentile of batch-1 cycle life and frozen before any agent ran. Every batch-2 cell falls below it. I did not move the threshold after seeing this. The pre-registered within-batch rule is reported as the sensitivity check.

### Live Omnigent runs

`omni run omnigent/cyclewise_lab` runs real Claude agents through Omnigent's claude-sdk harness, which uses the local Claude login (no API key needed). Three live runs (`omni-63e3cc7c`, `omni-f6098d37`, `omni-424f2025`) got this far:
- the supervisor called `start_run`;
- Safety scanned batch 1 (no flags);
- Evidence chose its own features (ΔQ variance, ΔQ min, early charge time, peak temperature, initial IR), with citations;
- Planner proposed options A/B/C and chose B, with a written rationale and within budget.

None reached approval:
- **First run:** exposed two bugs, now fixed. The approval ASK was raised inside the headless Planner sub-agent, where no human sees it; approval now sits on the supervisor. The strict tool schema made the optional `note` argument required; tools are now non-strict.
- **Later runs:** the one-shot `-p` CLI mode exited while sub-agents were still working. That is a limitation of the non-interactive harness used for testing, not of CycleWise.

**So the live LLM results are not measured yet.** Finish them by running Omnigent interactively ([below](#live-loop-in-omnigent)) and approving each plan yourself.

## v2: sequential checkpoints (pre-registered)

**Why.** v1 asked the agents to out-predict ΔQ from one 50-cycle snapshot, with about 40 cells per batch. Tweaking the rule cannot fix that. v2 changes the experiment design instead:
- ΔQ (Severson 2019, cited) is the base signal.
- The lab's job is **allocation**: stop clear losers at cycle 50, spend extra cycles only where the decision is uncertain, and keep a fixed number of cells to end of life.

**Design** (all fixed in [`config/prereg_v2.yaml`](config/prereg_v2.yaml)):
- **Checkpoints at 50, 100 and 150.** Every cell is screened to 50. The Planner decides how many to continue to 100, then to 150. At 150, k_final = round(0.25·N) cells are kept to end of life.
- **Budget:** extensions share 50·⌈0.75·N⌉ channel-cycles. Options are generated inside the budget, and at least two are offered at every checkpoint. A human approves every checkpoint.
- **Agents see only paid-for data.** Each run has its own database holding cycles ≤ 50. Cycles up to 100 or 150 are copied in only for cells whose extension was approved and committed. Labels appear only for cells that died inside the paid window or were kept to end of life.
- **Roles:** b1 train, b2 validation (the Critic may revise the 150-cycle rule after either, if the pre-registered trigger fires), b3 test (rule frozen; all headline claims).
- **Label:** within-batch top quartile, because v1's absolute threshold broke on b2 (batch shift). v1's threshold is still reported as the secondary label.
- **Comparators,** all with the same cost accounting: test everything to end of life; one-shot ΔQ at 50; one-shot ΔQ at 100 (Severson's stronger setting); early capacity.
- **Claim rules** were fixed in advance: "better" only if the paired recall-difference CI is above 0; "non-inferior and cheaper" only if its lower bound is above −0.10 and it uses fewer channel-cycles.

**Proof of order.** Commit `e5725a5` adds the pre-registration with its SHA-256 (`ac30f55b…`) before any batch-3 code or data. Every v2 run checks that the file is committed and unchanged, and records the commit hash. Batch 3 can only be loaded through that check.

**Development results on b1/b2** (run `v2-dev-001`, `reports/v2_v2-dev-001.md`). **These are not headline results:** b1 and b2 labels were seen during v1. Deterministic agent policy, `DEMO_AUTO_APPROVE=true`. Primary label (within-batch Q75), recall at k_final with 95% CI, and total channel-cycles:

| Batch | CycleWise v2 | One-shot ΔQ@50 | One-shot ΔQ@100 | Test all to EOL | Early capacity |
|---|---|---|---|---|---|
| b1 (k=12) | 0.75 [0.50, 1.00] · 18,338 | 0.75 · 17,755 | 0.75 · 19,488 | 1.00 · 42,245 | 0.17 · 11,461 |
| b2 (k=11) | 0.33 [0.08, 0.62] · 7,783 | 0.42 · 7,143 | 0.42 · 8,928 | 1.00 · 20,348 | 0.42 · 7,017 |

- **b1:** CycleWise matches the recall of both ΔQ rules. It uses 6% fewer channel-cycles than ΔQ@100 (pre-registered claim: *non-inferior and cheaper*) but 3% more than ΔQ@50, and 57% fewer than testing everything.
- **b2:** CycleWise is worse than both ΔQ rules (−0.08 recall). The Critic's trigger just missed (Spearman 0.51 vs the 0.5 threshold), so no revision happened. Nothing was retuned in response.
- **b3 (the real test):** not run yet. It needs the pre-registration pushed, then `python -m cyclewise.v2.data --with-b3`.

## How it works

```
                         Omnigent supervisor (cyclewise_lab)
                                       │
batch 1 (train):  Evidence ─► Safety scan ─► Planner (≥2 options) ─► HUMAN APPROVAL ─► Runner ─► Critic
                                                                    (supervisor ASK)    │ commit ids+hash,
                                                                                        │ then reveal
batch 2 (eval):   same loop, using the revised rule if the Critic revised (fit on batch-1 revealed cells only)
then:             scoring harness vs both baselines, bootstrap CIs, MLflow, research log
```

| Agent | Owns | Tools | Output (Pydantic, `cyclewise/agents/schemas.py`) |
|---|---|---|---|
| Supervisor | sequencing; asking the human | start run, run state, approval request, evaluation | passes each agent's JSON to the next |
| Evidence | which early signals to trust | early features/cycles (≤ 50), citations | `Hypothesis{features[], rationale, citations[], label}` |
| Planner | which test, within budget | budget check, ranking preview, plan submission | `TestPlan{options[≥2], chosen, cost_cells, expected_learning, cell_ids[]}` |
| Runner | none (executes) | commit, gated reveal, MLflow | `Result{run_id, metrics, selected_ids, revealed_outcomes}` |
| Critic/Safety | revise or not; cells needing review | anomaly scan, rule fitting | `Critique{revise, reason, new_rule?, flagged_cells[]}` |

Full agent specs and policies: [AGENTS.md](AGENTS.md).

### Leakage controls (defense in depth)

1. **Physical split.** `warehouse/early.duckdb` holds only cycles ≤ 50 and no labels. Labels and full traces live in `warehouse/hidden.duckdb`, which only `reveal()` and the scorer open. On Databricks this is the `cycles_early` UC view plus grants (`databricks/01_unity_catalog.sql`).
2. **Tool code.** Every cycle argument is checked; anything above 50 raises `LeakageError`.
3. **Omnigent policies.** `leakage_guard` denies any tool call naming cycle > 50 or a hidden table. `approval_gate` ASKs a human before any plan is approved. `reveal_gate` denies `execute_plan` without a recorded approval. `budget_guard` denies over-budget plans.
4. **Reveal protocol.** The selection (cell IDs + hash) is committed to the log before reveal. Reveal checks the commit hash and an approval for that exact plan, and returns only the kept cells.
5. **Train/eval separation.** Batch 2 gets `transform`, never `fit`. The threshold was frozen once (`config/frozen.yaml`, with a hash of the pre-registered settings).

`tests/test_leakage.py` tries cycle 51 through every path (tool, Omnigent tool wrapper, policy) and asserts it fails. It also checks that labels are unreachable before commit + approval.

Inputs agents may use: cell IDs and charging protocol (known before testing). LLMs may know Severson's published findings, which must be cited from `CITATIONS.md`. They do not know per-cell outcomes.

## Data

Severson et al. 2019 (Nature Energy), batches 2017-05-12 (b1, train) and 2017-06-30 (b2, eval) from data.matr.io. Quirks handled in `cyclewise/data/load_raw.py`:

- **HDF5 references** are dereferenced, and uint16 strings are decoded.
- **Continued cells:** b1c0–4 continue as b2c7/8/9/15/16. Their cycles are appended to the batch-1 cell and the batch-2 copies are dropped. An assert checks that no ID appears twice.
- **EOL crossing not stored:** batch-1 files stop on the cycle *before* capacity crosses 0.88 Ah. Cycle life = last + 1 when the last capacity is within 0.005 Ah of EOL. This matches the file's own cycle life for 100% of batch-1 cells, and our b2 labels match the file exactly.
- **Censoring:** 5 cells (b1c8/10/12/13/22) are right-censored. Their lower bound is below the threshold, so their label is unknown and they are excluded from scoring (counted in the report).
- **Glitches and units:** a rolling-median glitch filter is applied, forward-fill is limited to 2 cycles, and value ranges are asserted (Ah, °C, Ω).
- **Window:** features use cycles 2..50, and ΔQ = Q₅₀(V) − Q₁₀(V) on the files' 1000-point voltage grid.
- **Protocol parsing:** one malformed protocol string (`4C(31%)-5`) is handled.

## Run it

Verified from a clean state (warehouse, logs, reports and cache deleted) on 2026-10-03, macOS, Python 3.12, Omnigent 0.16.0.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -e ".[dev]" omnigent
.venv/bin/python -m cyclewise.data.download     # ~5 GB, resumable, size-checked
.venv/bin/python -m cyclewise.data.load_raw     # load + clean + label (prints counts only)
.venv/bin/python -m cyclewise.data.splits       # features, two-DB split, freeze threshold
.venv/bin/python -m cyclewise.v2.data          # v2 warehouse for b1/b2 (Qdlin at 100/150, labels hidden)
.venv/bin/python -m pytest                      # 65 tests
```

Without the data (e.g. a fresh clone), `pytest` gives 53 passed and 12 skipped. The 12 tests marked `requires_data` need the built warehouses, and their skip message lists the data commands.

**v2 loop:**

```bash
DEMO_AUTO_APPROVE=true CYCLEWISE_LLM=offline .venv/bin/python -m cyclewise.v2.loop   # b1, b2 (and b3 if built)
.venv/bin/python -m cyclewise.v2.data --with-b3     # ONLY after the pre-registration is pushed: downloads + loads b3
.venv/bin/python -m cyclewise.v2.loop               # full v2 run incl. the b3 test; approvals at every checkpoint
```

Reports are written to `reports/v2_<run_id>.md` and `.json`.

Expected output of the data steps:

```
$ python -m cyclewise.data.load_raw
b1     eligible    46
b2     eligible    43
cycles_raw rows: 63731, qdlin_early rows: 4361
       size  censored
b1       46         5
b2       43         0

$ python -m cyclewise.data.splits
frozen long-lived threshold written to config/frozen.yaml (value not printed)
{"b1": {"cells": 46, "eligible": 46, "censored": 5, "label_unknown": 5, "features_nan_cells": 0},
 "b2": {"cells": 43, "eligible": 43, "censored": 0, "label_unknown": 0, "features_nan_cells": 0}}
```

Expected output of the deterministic demo run (`DEMO_AUTO_APPROVE=true CYCLEWISE_LLM=offline python -m cyclewise.run_loop`), abridged:

```
>>> DEMO_AUTO_APPROVE=true — approvals in this run are automatic and logged as such <<<
======== BATCH b1 (2017-05-12, role=train) ========
[evidence] features: [('dq_logvar', -1), ('dq_logmin', -1), ('qmax_minus_q2', 1)]
[safety] 0 flag(s)
=== APPROVAL REQUEST  plan b1-…  batch b1 ===
Chosen option B: train batch: a revision can still be applied downstream, so buy learning
 * [B] explore_stratified: top 8 by score + 4 spread over the score range
[runner] revealed 12 cells: 5 long-lived (precision 0.50, spearman 0.83)
[critic] trigger fired=False  revise=False: no revision: trigger did not fire
======== BATCH b2 (2017-06-30, role=eval) ========
[runner] revealed 12 cells: 0 long-lived (precision 0.00, spearman 0.54)
[critic] trigger fired=True  revise=False: trigger fired but revision is not permitted (role=eval; …)
======== EVALUATION (scoring harness, full labels) ========
Batch b1 [primary_train_q75]: 12 long-lived of 41 scored
  cyclewise       recall 0.42  CI [0.12, 0.71]
  early_capacity  recall 0.17  CI [0.00, 0.42]
  delta_q         recall 0.75  CI [0.50, 1.00]
Batch b2 [primary_train_q75]: 0 long-lived of 43 scored
  no long-lived cells under this rule: recall is undefined
Batch b2 [secondary_within_batch_q75]: 12 long-lived of 43 scored
  cyclewise       recall 0.42  CI [0.12, 0.71]
  early_capacity  recall 0.42  CI [0.12, 0.70]
  delta_q         recall 0.50  CI [0.20, 0.80]
```

The report is written to `reports/<run_id>.md` and `.json` (plus `reports/latest.json`).

### Live loop in Omnigent

This is the orchestrator for the submission. Run it in a real terminal and keep the session open:

```bash
omni setup                                       # only if no Claude provider is configured yet
.venv/bin/omni run omnigent/cyclewise_lab        # interactive REPL; also prints a web UI URL
> Run the CycleWise study                        # type this at the prompt
# shared server: omnigent server -c omnigent/server.yaml   (registers the custom policies)
```

When each plan is ready, the supervisor calls `request_approval`. The `approval_gate` policy turns that into an Omnigent ASK in your session (terminal or web UI), showing the chosen option, the alternatives, cost and safety flags. Approving is the human gate; rejecting sends your note back to the Planner once. An unanswered ASK waits (Omnigent's default limit is one day) and is never auto-approved. Avoid `-p "..."` for full runs: in one-shot mode the CLI can exit while sub-agents are still working.

### Headless loop

Same agents, tools, gates and log, driven by plain Python; for CI and replay:

```bash
ANTHROPIC_API_KEY=... .venv/bin/python -m cyclewise.run_loop                  # Claude, interactive approval
DEMO_AUTO_APPROVE=true CYCLEWISE_LLM=offline .venv/bin/python -m cyclewise.run_loop   # deterministic, labeled demo
.venv/bin/python -m cyclewise.demo.replay <run_id> --pause                     # replay a run from the log
.venv/bin/python -m cyclewise.demo.replay --list                               # list runs
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db                    # MLflow runs
```

Without a TTY, approvals wait indefinitely for `python -m cyclewise.agents.approval <plan_id> approve|reject "note"`. Nothing is ever auto-approved unless `DEMO_AUTO_APPROVE=true`.

### Databricks

Scripts written, **untested** (no workspace was available):
- `databricks/upload_tables.py` loads the Delta tables;
- `databricks/01_unity_catalog.sql` creates the `cycles_early` view and the grants;
- `databricks/02_dashboard_queries.sql` feeds the AI/BI dashboard.

Set `MLFLOW_TRACKING_URI=databricks` to log runs to the workspace.

## Project layout

```
config/            cyclewise.yaml (v1 pre-registered settings), frozen.yaml (v1 threshold, written once),
                   prereg_v2.yaml + prereg_v2.lock (v2 pre-registration, hash-locked), exclusions.yaml
cyclewise/
  data/            download, load_raw (HDF5 → tables), featurize (cycles ≤ 50), splits (two-DB split + freeze)
  tools/           cutoff_view (cycle ≤ 50 guard), train_eval (rules), reveal (gated), anomaly, budget,
                   omni_tools (Omnigent tool layer)
  agents/          evidence, planner, runner, critic_safety, approval, schemas, llm (retries/cache/fallback)
  policies/        omni_policies (leakage, approval, reveal, budget)
  baselines/       early_capacity, delta_q_model
  eval/            metrics, bootstrap, report, tracking (MLflow)
  record/          research_log (append-only, hash-chained), dblock (cross-process lock)
  demo/            replay
  run_loop.py      v1 headless orchestrator
  prereg.py        v2 pre-registration guard (committed + hash-locked, else refuse)
  v2/              data (b1-b3, checkpoint curves), visible (paid-for data per run), features,
                   allocation (budgets, options), agents, loop (checkpoints), scoring (comparators, claims)
omnigent/
  cyclewise_lab/   supervisor + agents/{evidence,planner,runner,critic_safety}; tools/python/<tool>.py
  gen_tools.py     regenerates the Omnigent tool files
  server.yaml      registers the custom policies on a shared server
databricks/        Unity Catalog SQL, Delta upload, dashboard queries (untested)
tests/             65 tests
reports/           generated results per run
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `prereg_v2.yaml differs from its frozen hash` / `is not committed` | The v2 pre-registration was edited or never committed. Revert it; it must not change after commit. |
| `raw data missing or incomplete` | Run `python -m cyclewise.data.download` (resumes partial files). |
| `missing [...frozen.yaml, ...early.duckdb]` | Run `load_raw` then `splits` before `run_loop`. Nothing is logged until this passes. |
| `pre-registered part of config ... changed after the threshold was frozen` | You edited `preregistered`, batch roles or continuations. Revert. Paths, URLs and timeouts can change freely. |
| Omnigent: run ends before approval, plan in log but no `approval` row | You used `-p` one-shot mode. Run the REPL interactively and type the request. |
| Omnigent: `tool '…' is not enabled for this agent` | Tool files must be one tool per file, named after the tool. Regenerate with `python omnigent/gen_tools.py`. |
| Omnigent: policy handler not registered (shared server) | Start the server with `omnigent server -c omnigent/server.yaml`. |
| Run waits at `Waiting for a human decision` | Expected without a TTY. Approve or reject with `python -m cyclewise.agents.approval <plan_id> approve "note"`. |
| Ctrl-D / Ctrl-C at the approval prompt | Nothing is approved or revealed. The run stops and logs `approval_aborted`. |
| `fallback_used=true` in the log | The model reply failed validation 3 times, or no LLM backend was configured. The deterministic policy was used and labeled. |

## Robustness

- **Omnigent tool boundary:** tools never raise into the agent loop. Bad input (unknown batch or plan, invalid JSON, hallucinated feature or citation, over budget) comes back as `{"ok": false, "error": …}` so the agent can correct itself.
- **Rules come from the log, not the agent:**
  - one executed plan per batch;
  - batch 2 must use the revised rule unchanged;
  - role and revision count are read from config and the log, never from tool arguments.
- **Approval where the human is:** the approval ASK is raised by the supervisor, whose session the human watches, never inside a headless sub-agent.
- **Concurrent access:** the research log is guarded by a cross-process file lock, because Omnigent runs tools in subprocesses and policies in the server. Four parallel writers × 40 rows give 160 rows with unique sequence numbers and an intact hash chain.
- **LLM retries:** each retry reaches the model again (the attempt number is part of the cache key), so a repeated bad answer is not replayed from cache.
- **Tests (65):** leakage through every path; approval and reveal gates; budget, schema and citation guards; the log tamper check; the LLM path with a scripted fake model, including a full two-batch loop that takes the revision branch; the Omnigent bundle against Omnigent's own loader and dispatch check; the tool-boundary rules; and v2's pre-registration guard, budgets, and paid-for-data leakage control.

## Reproducibility

- Every step is a row in an append-only, hash-chained research log (`warehouse/research_log.duckdb`; `research_log.verify()`).
- Each row records the timestamp, agent, input hash, output JSON, model, prompt version, and `fallback_used`.
- LLM responses are cached by input hash (`cache/llm/`), so replays make no calls. Current Claude models do not accept `temperature`, so determinism comes from the cache rather than temperature 0.
- Seeds are fixed for the bootstrap, and ties at the budget boundary are broken by cell ID and logged.

## Limitations

- Retrospective evaluation only. No physical test time was saved.
- Small samples: 41 and 43 scored cells. CIs are wide, and most differences are not distinguishable from zero.
- One chemistry (LFP/graphite), one temperature (30 °C), fast-charge protocols only.
- v2's design was written after seeing b1/b2 labels (stated in the pre-registration). Only b3 results are claims; b1/b2 v2 numbers are development results.
- Batch 2 is shorter-lived than batch 1, so the pre-registered absolute threshold yields no positives there.
- The revised-rule path (Critic re-fits on revealed cells) did not trigger in the measured run, so it is tested only by unit tests.
- Censored cells contribute lower bounds when fitting revised rules.
- The live LLM agents run through Omnigent up to the approval step, but no live run has been completed, so their results are not measured.
- The Databricks scripts are untested.

## Next experiment

1. Push the v2 pre-registration, then run v2 on batch 3 (the untouched test set) and report the table above for b3.
2. Complete a live Omnigent run with human approvals at each checkpoint, and record it.
3. Run v2 prospectively on new cells, then on a second chemistry.

## Citations

See [CITATIONS.md](CITATIONS.md) (DOIs verified). Primary: Severson et al., *Nature Energy* 4, 383–391 (2019); Attia et al., *Nature* 578, 397–402 (2020).
