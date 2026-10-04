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
.venv/bin/python -m pytest                      # 70 tests
```

Run the study with automatic approvals (marked as such in the log):

```bash
DEMO_AUTO_APPROVE=true CYCLEWISE_LLM=offline .venv/bin/python -m cyclewise.v2.loop
```

Or run it live with Claude agents in Omnigent, approving each step yourself. Use a normal terminal and keep it open:

```bash
.venv/bin/omni run omnigent/cyclewise_v2        # then type: Run the CycleWise v2 study
```

At each of the 9 checkpoints (3 per batch), Omnigent asks you to approve the plan. `omnigent/cyclewise_lab` is the older one-decision version.

## Live run with Claude agents

A full live run of the one-decision version completed in Omnigent (run `omni-69fa0ff2`):
- four Claude agents handed off to each other on two batches;
- a human approved each plan before any result was revealed.

The log shows every approval recorded before the matching selection was committed. The agents also acted sensibly on their own: the Planner flagged a near-tie at the cutoff, and the Critic refused a revision the rules did not allow. Replay it with `.venv/bin/python -m cyclewise.demo.replay omni-69fa0ff2 --pause`.

## Databricks

The scripts load the data into Unity Catalog and lock the agents out of the hidden labels. They have been checked offline but not yet run on a workspace.

1. Get a workspace (the free [Databricks Free Edition](https://www.databricks.com/learn/free-edition) works) and sign in from your terminal:
   ```bash
   brew tap databricks/tap && brew install databricks
   databricks auth login --host https://<your-workspace>.cloud.databricks.com
   uv pip install --python .venv -e ".[databricks]"
   ```
2. Preview every step without sending anything:
   ```bash
   .venv/bin/python databricks/upload_tables.py --dry-run
   ```
3. Upload the tables, create the `cycles_early` view (cycles ≤ 50 only), and grant the agent access to `lab` only:
   ```bash
   .venv/bin/python databricks/upload_tables.py --catalog workspace --agent-principal <agent user or group>
   ```
   At the end it checks the view in your workspace and should print `cycles_early max cycle = 50`.
4. For the dashboard, open `databricks/02_dashboard_queries.sql` in the SQL editor, replace `${catalog}` with `workspace`, and add the queries to an AI/BI dashboard.
5. To log MLflow runs to the workspace, set this before running the study:
   ```bash
   export MLFLOW_TRACKING_URI=databricks
   ```
   Runs appear under `/Shared/cyclewise`.

## Status

| | |
|---|---|
| Data, agents, checkpoints, evaluation | ✅ Working, 70 tests passing |
| Result on the sealed test batch | ✅ Matches the best method at lower cost (not statistically confirmed) |
| Live run with Claude agents in Omnigent | ✅ Completed for the one-decision version (`omni-69fa0ff2`). ⚠️ The v2 checkpoint version is built and tested but has not been run live yet |
| Databricks (Unity Catalog, dashboard) | ⚠️ Scripts checked offline (dry run + tests); not yet run on a workspace |

## More

- [AGENTS.md](AGENTS.md): agent and policy specs
- [CITATIONS.md](CITATIONS.md): sources
- [config/prereg_v2.yaml](config/prereg_v2.yaml): the locked test plan
