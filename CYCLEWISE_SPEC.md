# CycleWise: Build Spec and Edge Cases

Hand this whole file to the coding agent as project context. Read it fully before writing code.

---

## 1. What we are building

A 24h hackathon project (Hack-Nation x Databricks, Challenge 03 "Agentic Scientific Discovery"). Build time budget: ~12 hours.

**Research question:** Under a fixed total cycle budget, can an agentic lab that sees only the first 50 cycles of each battery cell choose which cells to keep testing, and retain more truly long-lived cells than (a) a naive early-capacity rule and (b) a Severson-style ΔQ-variance model?

**Required platform:** Omnigent (Databricks managed or open source). Omnigent must orchestrate the live loop: multiple specialist agents exchanging structured outputs, using tools, and changing the plan after an experimental result.

**Judging weights:** Omnigent orchestration 30%, breakthrough potential 25%, discovery acceleration 20%, scientific rigor 15%, creativity and responsibility 10%.

**Submission must include:** repo, agent specs and policies, 2-minute demo video, cited evidence, experiment code and results, measured improvement, proposed next experiment.

---

## 2. Data

**Primary:** Severson et al. 2019 (Nature Energy) battery cycle-life dataset, data.matr.io. A123 APR18650M1A LFP/graphite cells, nominal capacity 1.1 Ah, fast-charge protocols, chamber at 30 °C.
- Use batches 2017-05-12 (batch 1) and 2017-06-30 (batch 2). Batch 3 (2018-04-12) only if time allows.
- Files are MATLAB v7.3 (HDF5). Load with `h5py`, not `scipy.io`.
- Cycle life = first cycle where discharge capacity < 0.88 Ah (80% of nominal).

**Fallback (switch if data isn't loaded and featurized by hour 2):** ESOL aqueous solubility dataset (~1,100 molecules, single CSV), RDKit descriptors. Same loop: "which molecules to send for an expensive assay under a budget." Keep the code modular so only the data layer changes.

### Known dataset quirks (verify against the authors' reference loader before trusting)
- Some batch 1 cells were continued in batch 2. Their cycle data must be concatenated, and the batch 2 copies removed as separate cells.
- Some batch 1 cells never reach 80% capacity within the test. Treat as right-censored (see 4.2).
- A few cells have noisy channels or broken temperature/IR traces and were excluded in the original paper.
- Cycle 1 (and sometimes cycle 0) has atypical values. Start feature windows at cycle 2 or later.
- Severson's ΔQ feature uses discharge capacity interpolated on a fixed voltage grid (Qdlin). Our window is ΔQ_{50-10}(V) instead of ΔQ_{100-10}(V).

---

## 3. Architecture

```
cyclewise/
  data/        load_raw.py, featurize.py, splits.py
  tools/       cutoff_view.py, train_eval.py, reveal.py, anomaly.py, budget.py
  agents/      evidence.py, planner.py, runner.py, critic_safety.py, schemas.py
  policies/    omnigent policy files (tool permissions, approval gates)
  record/      research_log.py  (append-only Delta table, local DuckDB fallback)
  baselines/   early_capacity.py, delta_q_model.py
  eval/        metrics.py, bootstrap.py
  demo/        replay.py, dashboard queries
  CITATIONS.md
  README.md
```

### Databricks pieces (these are part of the pitch, not optional decoration)
| Piece | Use |
|---|---|
| Delta tables | `cells`, `cycles_raw`, `features_early`, `labels_hidden`, `research_log` |
| Unity Catalog view | `cycles_early` exposes only cycle_index ≤ 50. Agents' tools query only this view. |
| UC permissions | Agent principal has no SELECT on `labels_hidden` or `cycles_raw`. Only `reveal()` (after approval) can read labels. |
| MLflow | Every baseline and every agent-chosen test is a run, with params, metrics, and the cell IDs selected. |
| AI/BI dashboard | CycleWise vs. baselines, per batch, with CIs. |

If running open-source Omnigent locally without Databricks, mirror this with DuckDB tables and a tool-layer check. Document the difference.

### Agents
Every agent output is JSON validated against a Pydantic schema in `agents/schemas.py`.

| Agent | Owns the decision | Tools | Input | Output |
|---|---|---|---|---|
| Evidence | Which early signals to trust | `cycles_early` query, featurize, citation lookup | cell IDs | `Hypothesis{features[], rationale, citations[], label:"agent-generated"}` |
| Planner | Which test to run, within budget | `budget.check`, `request_approval` | Hypothesis, remaining budget | `TestPlan{options[>=2], chosen, cost_cycles, expected_learning, cell_ids[]}` |
| Runner | None (executes) | `train_eval`, MLflow log, `reveal` (gated) | approved TestPlan | `Result{run_id, metrics, selected_ids, revealed_outcomes}` |
| Critic/Safety | Whether to revise the rule; which cells need human review | `anomaly.scan`, metrics | Result, early data | `Critique{revise:bool, reason, new_rule?, flagged_cells[]}` |

### Loop
1. Batch 1: Evidence → Planner (2+ options, picks one) → **human approves** → Runner selects cells under budget → reveal → Critic.
2. If Critic says revise: new rule is fit on **batch 1 revealed data only**, then applied to batch 2 without any tuning on batch 2.
3. Batch 2: same loop with the revised rule. Max 1 revision per batch.
4. Every step appends a row to `research_log` (timestamp, agent, input hash, output JSON, model, prompt version).

### Optional extension (only if core is done by hour 9)
Sequential checkpoints at cycles 50, 100, 150 with continue/stop decisions per cell under a total cycle budget.

---

## 4. Edge cases to handle

### 4.1 Data loading and cleaning
- `.mat` v7.3 references: HDF5 object references must be dereferenced; strings are stored as uint16 arrays.
- Continued cells: merge, renumber cycles contiguously, drop the duplicate entries. Assert no cell ID appears twice.
- NaN / inf in capacity, IR, temperature: drop the cycle if the core field is missing; never forward-fill across more than 2 cycles; record count of repaired values per cell.
- Capacity spikes (single cycles far above neighbors, or > nominal by a wide margin, or negative): mark as glitches with a rolling-median filter and exclude from features.
- Missing cycles inside 2..50 (gaps in cycle_index): if more than ~10% of the window is missing, mark the cell `insufficient_early_data` and exclude it from selection, logged, not silently dropped.
- Cells with fewer than 50 cycles recorded: exclude and log. (Should not occur in this dataset; assert.)
- Voltage grid for ΔQ: interpolation needs monotonic voltage; sort and dedupe before interpolating. Clip to the common voltage range across cycles.
- Units: confirm capacity in Ah, temperature in °C, IR in ohms. Assert value ranges on load.
- Excluded noisy cells: keep the exclusion list in a config file with the reason, and cite the source.

### 4.2 Labels
- Censored cells (never cross 0.88 Ah): cycle life is "≥ last observed cycle." For the binary long-lived label this is fine if last cycle ≥ threshold; if a censored cell's last cycle is below the threshold, its label is unknown. Exclude from scoring and report the count.
- Long-lived threshold: fixed **before** any reveal (e.g., top quartile of cycle life computed on the training batch, or an absolute cycle count). Store it in config and in the research log. Never recompute after seeing eval labels.
- Ties at the threshold: deterministic rule (≥ counts as long-lived).

### 4.3 Leakage (highest priority, judges will check)
- Agents can only reach data through tools that query `cycles_early`. No tool takes a cycle index argument above 50. Enforce in UC permissions and again in the tool code (defense in depth).
- `labels_hidden` readable only by `reveal()`, which checks for an approval record for that exact TestPlan ID.
- Feature scalers, model weights, thresholds: fit on the training batch only. Batch 2 gets `transform`, never `fit`.
- Research log after batch 1 reveal contains batch 1 labels. That is allowed. It must never contain batch 2 labels before batch 2 selection is committed.
- Commit the batch 2 selection (write cell IDs + hash to the log) **before** calling reveal for batch 2.
- Cell IDs and protocol names are allowed inputs (charging protocol is known before testing). Note this in the README.
- LLM prior knowledge: the model may know Severson's general findings. That is fine and should be cited. It does not know per-cell outcomes.
- Add a test that tries to query cycle 51 through every tool and asserts failure. Show it in the demo.

### 4.4 Budget and selection
- Budget expressed in cells or in cycles; pick one and be consistent. Store as an integer.
- Budget ≥ number of eligible cells: select all, log a warning, metric is trivially 100%.
- Budget = 0 or negative: reject at config load.
- Ties in score at the budget boundary: break by cell ID for determinism and log it.
- Planner asks for more than remaining budget: `budget.check` rejects, planner must re-propose (max 2 retries, then fallback to the highest-scoring cells within budget).
- Flagged cells from Safety: do not silently drop or silently keep. They go to the human approval step with the flag reason.

### 4.5 Agent and LLM behavior
- Invalid JSON or schema mismatch: retry with the validation error in the prompt, max 2 retries, then deterministic fallback, logged as `fallback_used=true`.
- Hallucinated cell IDs: validate every ID against the eligible set; reject unknown IDs.
- Hallucinated citations: citations must come from `CITATIONS.md` (a fixed list with DOIs). Reject any citation not in that list.
- Planner proposes fewer than 2 test options: reject (the brief requires at least two).
- Critic revises every time or never: revision triggers only on a defined condition (e.g., recall of long-lived below the ΔQ baseline, or a CI-based drop). Max 1 revision per batch to prevent loops.
- Revised rule must be a concrete artifact (feature list + model params), not prose. Validate it is executable before applying.
- Determinism: temperature 0, fixed seeds for numpy / sklearn / bootstrap, prompt versions stored in the log.
- Timeouts and rate limits: per-call timeout, exponential backoff, max total LLM calls per run.
- Cost/latency: cache LLM responses keyed by input hash so the demo can replay.

### 4.6 Human approval
- Approval request shows: chosen test, alternative tests, cost, flagged cells.
- Rejection path: planner re-proposes once with the human's note; on second rejection, stop the run cleanly.
- No response: run waits, does not auto-approve.
- Demo mode auto-approve is allowed only if clearly labeled `DEMO_AUTO_APPROVE=true` in the log and on screen.

### 4.7 Safety agent
- Flag cells with early temperature well above their batch's early-cycle distribution (z-score based, threshold in config) or sharp IR jumps.
- Flags route to human approval. They are not used as features unless the Evidence agent proposes that and it is logged.
- If anomaly detection itself fails (missing temperature), flag the cell as `sensor_missing`, do not crash.

### 4.8 Statistics and reporting
- Small n (~40 cells per batch): report bootstrap 95% CIs (≥1,000 resamples, fixed seed) for every metric.
- Primary metric: recall of long-lived cells at fixed budget. Secondary: cycles spent to reach the same recall as the best baseline (this is the "Nx faster" number).
- Report against both baselines. If CycleWise does not beat the ΔQ baseline, say so in the README.
- Report the actual measured ratio, even if it is 1.0x or worse.
- Report excluded / censored / flagged cell counts per batch.

### 4.9 Platform and infra
- Check Omnigent docs first: how agents are defined, how tools are registered, how policies and approvals work. Do not guess the API.
- Databricks free tier limits (cluster size, Unity Catalog availability, serverless): verify in hour 0. If UC is unavailable, implement the cutoff in the tool layer and say so.
- Large raw files: featurize once, write compact Delta/Parquet tables, never reload raw `.mat` in the agent loop.
- Secrets (API keys, tokens): environment variables or Databricks secrets, never committed.

### 4.10 Demo
- Every run reproducible from `research_log` + MLflow run IDs. `demo/replay.py` replays a stored run step by step.
- Record a full successful run early (by hour 9) as a backup video in case live calls fail.
- Show on screen: agent handoffs, the two test options and the choice, the approval click, the blocked cycle-51 query, batch 1 result, the critique and revised rule, batch 2 result, dashboard with CIs.

---

## 5. Acceptance criteria
- [ ] Data loaded, cleaned, featurized; quirks handled and logged.
- [ ] Both baselines run and logged to MLflow.
- [ ] Four agents run through Omnigent with schema-validated handoffs.
- [ ] Planner always compares ≥ 2 tests.
- [ ] Human approval gate blocks reveal.
- [ ] Leakage test passes (cycle 51 unreachable, labels unreachable pre-approval).
- [ ] Full loop on batch 1 → revision → batch 2, no tuning on batch 2.
- [ ] Metrics with bootstrap CIs vs both baselines.
- [ ] Research log reconstructs every decision.
- [ ] README states: question, bottleneck, measured improvement, limitations, next experiment.

## 6. Limitations to state in README
- Retrospective evaluation only. No physical test time was saved.
- Small sample, one cell chemistry, one temperature.
- Next experiment: prospective run on new cells with sequential stop/continue decisions, then a different chemistry.

## 7. Citations (seed list for CITATIONS.md; verify DOIs before submission)
- Severson et al., "Data-driven prediction of battery cycle life before capacity degradation," Nature Energy, 2019.
- Attia et al., "Closed-loop optimization of fast-charging protocols for batteries with machine learning," Nature, 2020.
- Hack-Nation x Databricks Challenge 03 brief.
