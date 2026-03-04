# End-to-end Trace (Worked Example)

This walkthrough shows a single OpenClawBrain query, feedback capture, learning event, and a follow-up query that changes retrieval. The example outputs are representative and safe.

## 1) Query the brain (hot path)

```bash
python3 -m openclawbrain.openclaw_adapter.query_brain \
  --state ~/.openclawbrain/main/state.json \
  --chat-id demo-123 \
  --query "deploy the hotfix"
```

Representative output:

```text
[BRAIN_CONTEXT]
Runbook: hotfix deploy checklist
- confirm CI green
- tag release
- run smoke tests

[END_BRAIN_CONTEXT]
```

A fired-nodes entry is appended to `fired_log.jsonl` under the same state directory.

## 2) Capture feedback (cold path)

Assume the assistant skipped tests and the operator marks it as a negative outcome.

```bash
python3 -m openclawbrain.openclaw_adapter.capture_feedback \
  --state ~/.openclawbrain/main/state.json \
  --chat-id demo-123 \
  --outcome -1 \
  --note "skipped tests"
```

Representative output:

```text
captured feedback for demo-123 (outcome=-1)
```

A learning event is appended to `learning_events.jsonl`, for example:

```json
{"ts": 1719965452.44, "chat_id": "demo-123", "outcome": -1, "fired": ["deploy_query", "skip_tests_for_hotfix"]}
```

## 3) Apply learning (cold path)

```bash
openclawbrain learn-by-chat-id \
  --state ~/.openclawbrain/main/state.json \
  --chat-id demo-123
```

Representative output:

```text
applied outcome=-1 to 2 fired nodes
```

## 4) Next query changes (hot path)

```bash
python3 -m openclawbrain.openclaw_adapter.query_brain \
  --state ~/.openclawbrain/main/state.json \
  --chat-id demo-124 \
  --query "deploy the hotfix"
```

Representative output (note the shift away from the skipped-tests path):

```text
[BRAIN_CONTEXT]
Runbook: hotfix deploy checklist
- confirm CI green
- run smoke tests
- verify rollback plan

[END_BRAIN_CONTEXT]
```

## Hot path vs cold path

Hot path: `query_brain` traversal, context assembly, daemon socket reads.

Cold path: feedback capture, learning updates, replay, maintenance, and pruning.
