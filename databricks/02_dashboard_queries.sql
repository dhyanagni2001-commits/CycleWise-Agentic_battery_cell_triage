-- AI/BI dashboard datasets for CycleWise. UNTESTED (no workspace available).
-- Source: the `evaluation` rows the evaluator appends to research_log.

-- 1. Recall@K with 95% CI per method, batch, and label rule.
SELECT
  run_id,
  b.key                                   AS batch,
  r.key                                   AS label_rule,
  m.key                                   AS method,
  CAST(get_json_object(m.value, '$.recall')  AS DOUBLE) AS recall,
  CAST(get_json_object(m.value, '$.ci_low')  AS DOUBLE) AS ci_low,
  CAST(get_json_object(m.value, '$.ci_high') AS DOUBLE) AS ci_high,
  CAST(get_json_object(m.value, '$.precision') AS DOUBLE) AS precision
FROM ${catalog}.lab.research_log
LATERAL VIEW explode(from_json(output_json, 'map<string,string>')) b AS key, value
LATERAL VIEW explode(from_json(b.value, 'map<string,string>')) r AS key, value
LATERAL VIEW explode(from_json(r.value, 'map<string,string>')) m AS key, value
WHERE event = 'evaluation';

-- 2. Agent handoffs timeline (for the demo).
SELECT seq, ts, run_id, batch, agent, event, fallback_used, model
FROM ${catalog}.lab.research_log
ORDER BY run_id, seq;

-- 3. Approvals, including any DEMO_AUTO_APPROVE.
SELECT run_id, batch, get_json_object(output_json, '$.plan_id') AS plan_id,
       get_json_object(output_json, '$.decision') AS decision,
       get_json_object(output_json, '$.approver') AS approver,
       get_json_object(output_json, '$.demo_auto_approve') AS demo_auto_approve
FROM ${catalog}.lab.research_log WHERE event = 'approval';

-- 4. v2 (sequential checkpoints): recall and channel-cycles per strategy, batch and label rule.
SELECT
  run_id,
  b.key AS batch,
  r.key AS label_rule,
  m.key AS strategy,
  CAST(get_json_object(m.value, '$.recall')  AS DOUBLE) AS recall,
  CAST(get_json_object(m.value, '$.ci_low')  AS DOUBLE) AS ci_low,
  CAST(get_json_object(m.value, '$.ci_high') AS DOUBLE) AS ci_high,
  CAST(get_json_object(m.value, '$.cost')    AS BIGINT) AS channel_cycles,
  get_json_object(m.value, '$.claim') AS claim
FROM ${catalog}.lab.research_log
LATERAL VIEW explode(from_json(output_json, 'map<string,string>')) b AS key, value
LATERAL VIEW explode(from_json(b.value, 'map<string,string>')) r AS key, value
LATERAL VIEW explode(from_json(r.value, 'map<string,string>')) m AS key, value
WHERE event = 'evaluation_v2' AND r.key <> 'costs';
