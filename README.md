# First Fifty

**[Omnigent](https://omnigent.ai) agents that decide which battery cells deserve a full life test, after only 50 cycles.**

Testing one battery cell until it wears out takes weeks. A lab can't test every cell, so it has to guess early which cells will last longest. In First Fifty, **Omnigent** runs a team of five Claude agents that make those calls under a fixed budget. Omnigent's policies keep the agents away from data they haven't paid for, and Omnigent asks a person to approve every decision before any results are revealed.

The code package is still named `cyclewise` (the project's working name), so commands and folders use that name.

**Demo:** [interactive replay of the logged runs](https://claude.ai/artifact/CAC2xACXT5Tq8iVduzMkaJ) (also in [`demo/first-fifty-replay.html`](demo/first-fifty-replay.html); open it in a browser).

Built for Hack-Nation x Databricks, Challenge 03 "Agentic Scientific Discovery", using the public battery dataset from Severson et al., *Nature Energy* 2019.

## Result

On a batch of 40 cells that was kept sealed until the plan was locked in, First Fifty was run two ways:

| Approach | Long-lived cells found | Test cycles used |
|---|---|---|
| **First Fifty, live Claude agents in Omnigent, every step approved by a person** | **7 of 10** | **15,338** |
| **First Fifty, frozen policy, automatic approvals** | **8 of 10** | **15,475** |
| Best published method (ΔQ at cycle 100) | 8 of 10 | 16,997 |
| Same method, earlier (ΔQ at cycle 50) | 6 of 10 | 13,697 |
| Test every cell to the end | 10 of 10 | 41,280 |

- **Frozen policy:** this is the pre-registered test. It matched the best method's accuracy with 9% fewer test cycles.
- **Live Claude agents:** they found one long-lived cell fewer than the best method, with 10% fewer test cycles. They made their own choices: a more cautious budget split at cycle 50 than the pre-registered default, a five-feature rule, and a different final ranking on each batch.
- **Both runs** used about 63% fewer test cycles than testing everything, and both beat the same method used at cycle 50.
- **Not statistically confirmed:** with only 40 cells, none of these differences passes the claim rules fixed in advance.

Full numbers with confidence intervals: [`reports/v2_omni2-73292d79.md`](reports/v2_omni2-73292d79.md) (live run) and [`reports/v2_v2-final-001.md`](reports/v2_v2-final-001.md) (frozen policy). The sealed batch was evaluated once per decision-maker; nothing was changed between or after the runs.

## What Omnigent does here

Omnigent is the layer that turns five separate Claude agents into one controlled lab. In this project it provides:

| Omnigent feature | How First Fifty uses it | Where to see it |
|---|---|---|
| **Agents defined in short YAML files** | A supervisor and four specialist agents, each with its own instructions, tools and model, in about 180 lines of YAML in total | [`omnigent/cyclewise_v2/`](omnigent/cyclewise_v2) |
| **Multi-agent orchestration** | The supervisor hands work to the Evidence, Safety/Critic, Planner and Runner agents, and passes each one's structured output to the next | Live run `omni2-73292d79` |
| **Policies on every tool call** | `leakage_guard` blocks requests for cycles past 50 and for the hidden label tables. `reveal_gate` blocks a plan from running until it is approved. `rate_limit` caps tool calls per session. Omnigent enforces these on every tool call, whatever the agent intends. | [`cyclewise/policies/omni_policies.py`](cyclewise/policies/omni_policies.py) |
| **Human approval built in** | The `approval_gate` policy turns every checkpoint into an Omnigent approval request in the terminal or web UI. In the live run a person approved all 9. | `approval` rows in the research log |
| **Python functions as tools** | Every agent tool is a plain Python function that Omnigent exposes with a typed schema | [`cyclewise/tools/omni_tools_v2.py`](cyclewise/tools/omni_tools_v2.py) |
| **Swap the runtime in one line** | The agents run on Claude through Omnigent's claude-sdk harness. Changing the `harness:` line in the YAML swaps the runtime without touching the tools or policies. | `executor:` block in each `config.yaml` |
| **Web UI and saved sessions** | The full live conversation, every agent handoff and every approval can be reopened in Omnigent's web UI | `omni run omnigent/cyclewise_v2` |
| **Runs on Databricks** | Omnigent has a Databricks-managed version, and the data side is set up for Unity Catalog | [Databricks](#databricks) section below |

## How it works

1. **Every cell runs 50 cycles.**
2. **At cycles 50 and 100,** the agents stop clearly weak cells and pay for more cycles only on the uncertain ones.
3. **At cycle 150,** they pick the cells to test to the end.
4. **Before any new data is revealed, a human approves the decision through Omnigent.**

Omnigent runs a supervisor agent and four specialist agents, all Claude:

| Agent | Job |
|---|---|
| Evidence | Chooses which early signals to trust (cites published research) |
| Planner | Proposes at least two options at each checkpoint, within budget |
| Runner | Carries out the approved decision |
| Critic / Safety | Flags unusual cells and decides whether the rule needs fixing |

## Why you can trust the result

- **No peeking.** The agents can only see data the lab has paid for. A request for cycle 51 before it's paid for is blocked by an Omnigent policy and again by the tool code, and tests prove both.
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

Or run it the way it's meant to run: live, with Claude agents in Omnigent, approving each step yourself. Use a normal terminal and keep it open:

```bash
.venv/bin/omni run omnigent/cyclewise_v2        # then type: Run the First Fifty study
```

At each of the 9 checkpoints (3 per batch), Omnigent asks you to approve the plan. `omnigent/cyclewise_lab` is the older one-decision version.

## Live runs in Omnigent

**Checkpoint version** (run `omni2-73292d79`): Omnigent ran five Claude agents through all three batches. A person approved each of the 9 checkpoint plans before any data was paid for or revealed, and all 9 approvals are in the log. The result is in the table above. The supervisor itself reported three process issues, recorded here as it gave them:
- The Planner ranked the final pick by ΔQ alone on batches 1 and 3, but by the full rule on batch 2.
- On batch 2 the revision trigger fired and a revision was allowed, but the Critic skipped it. It was reading an outdated threshold in `AGENTS.md`, now corrected.
- One batch-2 rule was submitted by mistake and then replaced before any checkpoint ran.

Replay with `.venv/bin/python -m cyclewise.demo.replay omni2-73292d79 --pause`.

**One-decision version** (run `omni-69fa0ff2`), the earlier design:
- four Claude agents handed off to each other on two batches;
- a human approved each plan before any result was revealed.

The log shows every approval recorded before the matching selection was committed. The agents also acted sensibly on their own: the Planner flagged a near-tie at the cutoff, and the Critic refused a revision the rules did not allow.

Results (share of long-lived cells found, 12 cells kept per batch; full report in [`reports/omni-69fa0ff2.md`](reports/omni-69fa0ff2.md)):

| Batch | First Fifty (live Claude agents) | ΔQ method | Early capacity |
|---|---|---|---|
| Batch 1 | 0.50 | **0.75** | 0.17 |
| Batch 2 | 0.50 | 0.50 | 0.42 |

This one-decision version trailed the published ΔQ method on batch 1 and tied it on batch 2. That is why the project moved to the checkpoint design shown in the [Result](#result) section. Batch 2 uses the within-batch definition of long-lived; under the stricter batch-1 definition it has no long-lived cells, so that score can't be computed.

Replay the run step by step with `.venv/bin/python -m cyclewise.demo.replay omni-69fa0ff2 --pause`.

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
| Result on the sealed test batch | Frozen policy matches the best method at 9% lower cost; live agents found one cell fewer at 10% lower cost (neither statistically confirmed) |
| Live run with Claude agents in Omnigent | ✅ Completed for both versions: checkpoint version `omni2-73292d79` (9 checkpoints, all approved by a person) and one-decision version `omni-69fa0ff2` |
| Databricks (Unity Catalog, dashboard) | ⚠️ Scripts checked offline (dry run + tests); not yet run on a workspace |

## More

- [AGENTS.md](AGENTS.md): agent and policy specs
- [CITATIONS.md](CITATIONS.md): sources
- [config/prereg_v2.yaml](config/prereg_v2.yaml): the locked test plan
