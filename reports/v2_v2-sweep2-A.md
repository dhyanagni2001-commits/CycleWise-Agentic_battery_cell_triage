# CycleWise v2 results: v2-sweep2-A

Pre-registration commit `e5725a5e7f` (sha256 `ac30f55bb995`). LLM backend: offline; fallback steps: 8; DEMO_AUTO_APPROVE=True; rule revised after: none.

## Batch b1 (train)

N=46, k_final=12, extension budget 1750 channel-cycles (spent 1750); continued 23 to 100 and 12 to 150; 5 censored.

### primary_within_batch_q75

12 long-lived of 41 scored (5 unknown). Random 0.29, oracle 1.00.

| strategy | recall@k | 95% CI | channel-cycles | CycleWise − this (95% CI) | claim |
|---|---|---|---|---|---|
| cyclewise | 0.75 | [0.50, 1.00] | 18,338 |  |  |
| test_all_to_eol | 1.00 | [1.00, 1.00] | 42,245 | -0.25 [-0.50, -0.00] | not_better |
| one_shot_dq50 | 0.75 | [0.50, 1.00] | 17,755 | -0.00 [-0.23, +0.24] | not_better |
| one_shot_dq100 | 0.75 | [0.50, 1.00] | 19,488 | -0.00 [-0.00, -0.00] | non_inferior_and_cheaper |
| early_capacity50 | 0.17 | [0.00, 0.42] | 11,461 | +0.59 [+0.29, +0.89] | better |

### secondary_absolute_v1

12 long-lived of 41 scored (5 unknown). Random 0.29, oracle 1.00.

| strategy | recall@k | 95% CI | channel-cycles | CycleWise − this (95% CI) | claim |
|---|---|---|---|---|---|
| cyclewise | 0.75 | [0.50, 1.00] | 18,338 |  |  |
| test_all_to_eol | 1.00 | [1.00, 1.00] | 42,245 | -0.25 [-0.50, -0.00] | not_better |
| one_shot_dq50 | 0.75 | [0.50, 1.00] | 17,755 | -0.00 [-0.23, +0.24] | not_better |
| one_shot_dq100 | 0.75 | [0.50, 1.00] | 19,488 | -0.00 [-0.00, -0.00] | non_inferior_and_cheaper |
| early_capacity50 | 0.17 | [0.00, 0.42] | 11,461 | +0.59 [+0.29, +0.89] | better |

## Batch b2 (validation)

N=43, k_final=11, extension budget 1650 channel-cycles (spent 1650); continued 22 to 100 and 11 to 150; 0 censored.

### primary_within_batch_q75

12 long-lived of 43 scored (0 unknown). Random 0.26, oracle 0.92.

| strategy | recall@k | 95% CI | channel-cycles | CycleWise − this (95% CI) | claim |
|---|---|---|---|---|---|
| cyclewise | 0.33 | [0.08, 0.62] | 7,783 |  |  |
| test_all_to_eol | 1.00 | [1.00, 1.00] | 20,348 | -0.67 [-0.92, -0.38] | not_better |
| one_shot_dq50 | 0.42 | [0.12, 0.71] | 7,143 | -0.08 [-0.38, +0.20] | not_better |
| one_shot_dq100 | 0.42 | [0.12, 0.70] | 8,928 | -0.08 [-0.27, -0.00] | not_better |
| early_capacity50 | 0.42 | [0.12, 0.70] | 7,017 | -0.08 [-0.45, +0.29] | not_better |

### secondary_absolute_v1

0 long-lived of 43 scored (0 unknown). Random 0.26, oracle n/a.

**no long-lived cells under this rule.**
