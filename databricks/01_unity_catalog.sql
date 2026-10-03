-- CycleWise on Databricks: Unity Catalog layout and permissions.
-- UNTESTED in this repo (no workspace was available during the build); mirrors the
-- local two-database split in cyclewise/data/splits.py. Replace the placeholders:
--   ${catalog}         e.g. cyclewise
--   ${agent_principal} the service principal / group the agents run as
--   ${scorer_principal} the principal that runs reveal() and the evaluator

CREATE CATALOG IF NOT EXISTS ${catalog};
CREATE SCHEMA IF NOT EXISTS ${catalog}.lab;      -- agent-visible
CREATE SCHEMA IF NOT EXISTS ${catalog}.hidden;   -- labels and full traces

-- Tables are created by databricks/upload_tables.py (Delta).
--   ${catalog}.lab.cells, ${catalog}.lab.features_early, ${catalog}.lab.research_log
--   ${catalog}.hidden.cycles_raw, ${catalog}.hidden.labels_hidden

-- The only cycle-level data agents can read: cycles <= 50.
CREATE OR REPLACE VIEW ${catalog}.lab.cycles_early AS
SELECT * FROM ${catalog}.hidden.cycles_raw WHERE cycle <= 50;

-- Agents: read the lab schema only. No access to hidden tables.
GRANT USE CATALOG ON CATALOG ${catalog} TO `${agent_principal}`;
GRANT USE SCHEMA ON SCHEMA ${catalog}.lab TO `${agent_principal}`;
GRANT SELECT ON TABLE ${catalog}.lab.cells TO `${agent_principal}`;
GRANT SELECT ON TABLE ${catalog}.lab.features_early TO `${agent_principal}`;
GRANT SELECT ON VIEW  ${catalog}.lab.cycles_early TO `${agent_principal}`;
GRANT MODIFY ON TABLE ${catalog}.lab.research_log TO `${agent_principal}`;  -- append via tools
-- Explicitly no grant on ${catalog}.hidden.* for the agent principal.
-- A view owned by a principal that can read hidden.cycles_raw lets the agent read
-- the view's rows (cycles <= 50) without SELECT on the underlying table.

-- Scorer / reveal(): reads labels, only after the approval check in code.
GRANT USE SCHEMA ON SCHEMA ${catalog}.hidden TO `${scorer_principal}`;
GRANT SELECT ON TABLE ${catalog}.hidden.labels_hidden TO `${scorer_principal}`;
GRANT SELECT ON TABLE ${catalog}.hidden.cycles_raw TO `${scorer_principal}`;

-- Check (run as the agent principal; both must fail):
--   SELECT * FROM ${catalog}.hidden.labels_hidden LIMIT 1;
--   SELECT max(cycle) FROM ${catalog}.lab.cycles_early;   -- returns 50
