# CycleWise v2 results: v2-sweep2-C

Pre-registration commit `e5725a5e7f` (sha256 `ac30f55bb995`). LLM backend: offline; fallback steps: 8; DEMO_AUTO_APPROVE=True; rule revised after: ['b2'].

## Batch b1 (train)

N=46, k_final=12, extension budget 1750 channel-cycles (spent 1750); continued 18 to 100 and 17 to 150; 5 censored.

### primary_within_batch_q75

12 long-lived of 41 scored (5 unknown). Random 0.29, oracle 1.00.

| strategy | recall@k | 95% CI | channel-cycles | CycleWise − this (95% CI) | claim |
|---|---|---|---|---|---|
| cyclewise | 0.75 | [0.50, 1.00] | 18,277 |  |  |
| test_all_to_eol | 1.00 | [1.00, 1.00] | 42,245 | -0.25 [-0.50, -0.00] | not_better |
| one_shot_dq50 | 0.75 | [0.50, 1.00] | 17,755 | -0.00 [-0.00, -0.00] | not_better |
| one_shot_dq100 | 0.75 | [0.50, 1.00] | 19,488 | +0.00 [-0.24, +0.23] | not_better |
| early_capacity50 | 0.17 | [0.00, 0.42] | 11,461 | +0.59 [+0.17, +0.92] | better |

### secondary_absolute_v1

12 long-lived of 41 scored (5 unknown). Random 0.29, oracle 1.00.

| strategy | recall@k | 95% CI | channel-cycles | CycleWise − this (95% CI) | claim |
|---|---|---|---|---|---|
| cyclewise | 0.75 | [0.50, 1.00] | 18,277 |  |  |
| test_all_to_eol | 1.00 | [1.00, 1.00] | 42,245 | -0.25 [-0.50, -0.00] | not_better |
| one_shot_dq50 | 0.75 | [0.50, 1.00] | 17,755 | -0.00 [-0.00, -0.00] | not_better |
| one_shot_dq100 | 0.75 | [0.50, 1.00] | 19,488 | +0.00 [-0.24, +0.23] | not_better |
| early_capacity50 | 0.17 | [0.00, 0.42] | 11,461 | +0.59 [+0.17, +0.92] | better |

## Batch b2 (validation)

N=43, k_final=11, extension budget 1650 channel-cycles (spent 1650); continued 17 to 100 and 16 to 150; 0 censored.

### primary_within_batch_q75

12 long-lived of 43 scored (0 unknown). Random 0.26, oracle 0.92.

| strategy | recall@k | 95% CI | channel-cycles | CycleWise − this (95% CI) | claim |
|---|---|---|---|---|---|
| cyclewise | 0.58 | [0.27, 0.88] | 7,905 |  |  |
| test_all_to_eol | 1.00 | [1.00, 1.00] | 20,348 | -0.42 [-0.73, -0.12] | not_better |
| one_shot_dq50 | 0.42 | [0.12, 0.71] | 7,143 | +0.17 [-0.00, +0.40] | not_better |
| one_shot_dq100 | 0.42 | [0.12, 0.70] | 8,928 | +0.17 [-0.17, +0.50] | not_better |
| early_capacity50 | 0.42 | [0.12, 0.70] | 7,017 | +0.17 [-0.23, +0.57] | not_better |

### secondary_absolute_v1

0 long-lived of 43 scored (0 unknown). Random 0.26, oracle n/a.

**no long-lived cells under this rule.**
