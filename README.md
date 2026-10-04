# CycleWise

**An AI lab that decides which battery cells are worth testing to the end, after seeing only their first cycles.**

Testing one battery cell until it wears out takes weeks. A lab can't test every cell, so it has to guess early which cells will last longest. CycleWise uses a team of AI agents, run by [Omnigent](https://omnigent.ai), to make those calls under a fixed budget. A human approves every decision before any results are revealed.

Built for Hack-Nation x Databricks, Challenge 03 "Agentic Scientific Discovery", using the public battery dataset from Severson et al., *Nature Energy* 2019.

## Result

On a batch of 40 cells that was kept sealed until the plan was locked in, CycleWise found **8 of the 10 longest-lived cells**:

| Approach | Long-lived cells found | Test cycles used |
|---|---|---|
| **CycleWise** | **8 of 10** | **15,475** |
| Best published method (ΔQ at cycle 100) | 8 of 10 | 16,997 |
| Same method, earlier (ΔQ at cycle 50) | 6 of 10 | 13,697 |
| Test every cell to the end | 10 of 10 | 41,280 |

- **Same accuracy as the best method, with 9% fewer test cycles.**
- **62% fewer test cycles than testing everything.**
- **Honest caveat:** with only 40 cells these differences are not statistically confirmed. Full numbers with confidence intervals are in [`reports/v2_v2-final-001.md`](reports/v2_v2-final-001.md).

## How it works

1. **Every cell runs 50 cycles.**
2. **At cycles 50 and 100,** the agents stop clearly weak cells and pay for more cycles only on the uncertain ones.
3. **At cycle 150,** they pick the cells to test to the end.
4. **Before any new data is revealed, a human approves the decision.**

Four agents share the work:

| Agent | Job |
|---|---|
| Evidence | Chooses which early signals to trust (cites published research) |
| Planner | Proposes at least two options at each checkpoint, within budget |
| Runner | Carries out the approved decision |
| Critic / Safety | Flags unusual cells and decides whether the rule needs fixing |

## Why you can trust the result

- **No peeking.** The agents can only see data the lab has paid for. A request for cycle 51 before it's paid for is blocked, and tests prove it.
- **Plan locked first.** The test plan was committed to GitHub before the test batch was even downloaded, and the timestamps prove the order.
- **Nothing hidden.** Every decision is saved in a tamper-evident log. Our first attempt (one decision at cycle 50) lost to the published method, finding 5 of 12 long-lived cells against its 9, and it is still reported in [`reports/final-check.md`](reports/final-check.md).

## Run it

Needs Python 3.12 and about 8 GB of disk space for the data.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -e ".[dev]" omnigent
.venv/bin/python -m cyclewise.data.download     # get batches 1-2 (~5 GB)
.venv/bin/python -m cyclewise.data.load_raw     # load and clean it
.venv/bin/python -m cyclewise.data.splits
.venv/bin/python -m cyclewise.v2.data --with-b3 # get batch 3 (~3 GB), build checkpoint data
.venv/bin/python -m pytest                      # 65 tests
```

Run the study with automatic approvals (marked as such in the log):

```bash
DEMO_AUTO_APPROVE=true CYCLEWISE_LLM=offline .venv/bin/python -m cyclewise.v2.loop
```

Or run it live with Claude agents in Omnigent, approving each step yourself:

```bash
.venv/bin/omni run omnigent/cyclewise_lab       # then type: Run the CycleWise study
```

## Status

| | |
|---|---|
| Data, agents, checkpoints, evaluation | ✅ Working, 65 tests passing |
| Result on the sealed test batch | ✅ Matches the best method at lower cost (not statistically confirmed) |
| Live run with Claude agents in Omnigent | ⚠️ Works up to the approval step; the Omnigent setup still runs the older one-decision version |
| Databricks (Unity Catalog, dashboard) | ⚠️ Scripts written, not yet tested on a workspace |

## More

- [AGENTS.md](AGENTS.md): agent and policy specs
- [CITATIONS.md](CITATIONS.md): sources
- [config/prereg_v2.yaml](config/prereg_v2.yaml): the locked test plan
