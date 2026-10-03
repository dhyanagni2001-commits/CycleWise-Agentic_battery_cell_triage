# CycleWise results: fresh-001

LLM backend: offline (claude-opus-5); deterministic-fallback steps: 6; DEMO_AUTO_APPROVE=True; rule revised: False.
Budget: 12 cells per batch. Primary metric: recall of long-lived cells at that budget, 95% bootstrap CIs (cells resampled with replacement, fixed seed).

## Batch b1

46 eligible cells; 5 censored (5 with unknown label under the primary rule, excluded from scoring); 0 safety flags. CycleWise chose option B with rule `zscore_sum` (fit_on=none).

### primary_train_q75

_long-lived = cycle life >= 75th percentile of batch-1 (train) cycle life, frozen before any run._ 12 long-lived of 41 scored.

| method | recall@K | 95% CI | precision@K |
|---|---|---|---|
| cyclewise | 0.42 | [0.12, 0.71] | 0.50 |
| early_capacity | 0.17 | [0.00, 0.42] | 0.18 |
| delta_q | 0.75 | [0.50, 1.00] | 0.82 |
| random (expected) | 0.29 | | |
| oracle (max possible) | 1.00 | | |

- CycleWise minus early_capacity: +0.25 recall (95% CI [-0.20, +0.67])
- CycleWise minus delta_q: -0.34 recall (95% CI [-0.62, -0.08])
- Cells tested in rank order to match delta_q (recall 0.75): cyclewise 12, early_capacity 23, delta_q 11. Ratio = 0.92x

### secondary_within_batch_q75

_long-lived = cycle life >= 75th percentile within the same batch (pre-registered sensitivity check)._ 12 long-lived of 41 scored.

| method | recall@K | 95% CI | precision@K |
|---|---|---|---|
| cyclewise | 0.42 | [0.12, 0.71] | 0.50 |
| early_capacity | 0.17 | [0.00, 0.42] | 0.18 |
| delta_q | 0.75 | [0.50, 1.00] | 0.82 |
| random (expected) | 0.29 | | |
| oracle (max possible) | 1.00 | | |

- CycleWise minus early_capacity: +0.25 recall (95% CI [-0.20, +0.67])
- CycleWise minus delta_q: -0.34 recall (95% CI [-0.62, -0.08])
- Cells tested in rank order to match delta_q (recall 0.75): cyclewise 12, early_capacity 23, delta_q 11. Ratio = 0.92x

## Batch b2

43 eligible cells; 0 censored (0 with unknown label under the primary rule, excluded from scoring); 0 safety flags. CycleWise chose option A with rule `zscore_sum` (fit_on=none).

### primary_train_q75

_long-lived = cycle life >= 75th percentile of batch-1 (train) cycle life, frozen before any run._ 0 long-lived of 43 scored.

**no long-lived cells under this rule: recall is undefined.** Precision of each selection: cyclewise 0.00, early_capacity 0.00, delta_q 0.00, ref_delta_q_all_train_labels 0.00

### secondary_within_batch_q75

_long-lived = cycle life >= 75th percentile within the same batch (pre-registered sensitivity check)._ 12 long-lived of 43 scored.

| method | recall@K | 95% CI | precision@K |
|---|---|---|---|
| cyclewise | 0.42 | [0.12, 0.71] | 0.42 |
| early_capacity | 0.42 | [0.12, 0.70] | 0.42 |
| delta_q | 0.50 | [0.20, 0.80] | 0.50 |
| ref_delta_q_all_train_labels | 0.50 | [0.20, 0.80] | 0.50 |
| random (expected) | 0.28 | | |
| oracle (max possible) | 1.00 | | |

- CycleWise minus early_capacity: +0.00 recall (95% CI [-0.50, +0.50])
- CycleWise minus delta_q: -0.08 recall (95% CI [-0.27, -0.00])
- Cells tested in rank order to match delta_q (recall 0.50): cyclewise 13, early_capacity 13, delta_q 12. Ratio = 0.92x
