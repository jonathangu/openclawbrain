# OpenClaw Workflow Proof

Date: 2026-03-06

This proof package tests one narrow claim:

> Given a useful cold-start graph, can async teacher supervision train a runtime `route_fn` that pulls the right history/runbook node immediately on realistic OpenClaw-style workflows?

It is designed to stay aligned with the actual product story:

1. OpenClawBrain is meant to sit inside ongoing OpenClaw use.
2. It should be useful immediately from existing graph priors, then improve in the background.
3. The core loop is async supervision over routing decisions that produces a better learned runtime route policy.

## Production alignment vs simulation constraints

- Production-default operator stack: local BGE-large embeddings plus a local async teacher such as Ollama `qwen3.5:35b`.
- This harness is deterministic on purpose, so it uses `HashEmbedder(dim=64)` and synthetic teacher labels instead of live model calls.
- That makes it reproducible in CI while still exercising the same runtime routing and route-model training surfaces used by the product.

## Reproduce

```bash
python3 -m examples.eval.simulate_openclaw_workflows \
  --output-dir scratch/workflow-proof/latest
```

Key generated files:

- `scratch/workflow-proof/latest/workflow_state.json`
- `scratch/workflow-proof/latest/train_traces.jsonl`
- `scratch/workflow-proof/latest/train_labels.jsonl`
- `scratch/workflow-proof/latest/eval_queries.jsonl`
- `scratch/workflow-proof/latest/learning_curve.csv`
- `scratch/workflow-proof/latest/baseline_eval/summary.json`
- `scratch/workflow-proof/latest/report.md`
- `scratch/workflow-proof/latest/worked_example.md`

## Deterministic result on this repo state

| mode | exact target success | required-node coverage | interpretation |
| --- | --- | --- | --- |
| `vector_topk` | `0.00` | `0.00` | one seed hub is found, but no route learning happens |
| `pointer_chase` | `0.25` | `0.38` | sometimes reaches the right chain, but needs extra turns and still misses often |
| `graph_prior_only` | `0.50` | `0.50` | cold-start graph priors are already useful on half the held-out workflow queries |
| `learned` | `1.00` | `1.00` | learned routing recovers the exact required nodes on all held-out queries |

Learning-curve detail:

- graph-prior target success stays flat at `0.50`
- learned routing reaches `1.00` target success by epoch `14`
- the final learned model at epoch `16` keeps `1.00` exact target success

## Worked example

Held-out query:

```text
the payments canary failed again. which exact flag action and rollback gate did we use last time
```

Required prompt-context nodes:

- `doc::payments_incident_2026_02_14`
- `doc::rollback_gate`

What each mode actually returns:

- `vector_topk` returns only `hub::incident`
- `pointer_chase` follows the wrong chain into dashboard material
- `graph_prior_only` prefers `oncall_schedule` and `monitoring_dashboards`
- `learned` returns `payments_incident_2026_02_14 -> rollback_gate`

The candidate table in `scratch/workflow-proof/latest/worked_example.md` shows why: graph priors still favor the globally common incident tools, while the learned router assigns the highest score to the incident-history note for this query.

## What is actually proven now

- OpenClawBrain can start with partially useful graph priors instead of requiring a fully trained router.
- Async route supervision can improve runtime retrieval on workflow-shaped held-out examples, not just abstract synthetic clusters.
- The learned runtime policy can outperform both vector-only retrieval and a pointer-chasing baseline on exact-node retrieval.
- The evidence is reproducible from committed code and written artifacts, not a one-off screenshot or manual story.

## What is not proven yet

- This is not a live production OpenClaw evaluation.
- It does not yet prove scanner/harvester label quality on real histories; it isolates the teacher-to-runtime route learning path.
- It does not measure downstream answer quality or operator success directly; it measures whether the right memory nodes reach prompt context.
