# Evaluation Plan

## Goal and key claim

Key claim for paper validation:

OpenClawBrain reduces multi-hop pointer chasing, which should reduce turns, tokens, and latency while preserving or improving retrieval quality on recurring workflows.

## Baselines and ablations

Evaluate the following retrieval modes with the same state and query set:

- `vector_only`: top-k vector seeds only; no traversal.
- `graph_prior_only`: learned router path with `router_conf=0.0` (forces graph-prior behavior from edge weight/relevance confidence mix).
- `qtsim_only`: learned router path with `router_conf=1.0` and `relevance_conf=1.0` (forces QTsim score dominance).
- `learned`: default confidence-mixed learned routing.
- `edge_sim_legacy`: deterministic `route_mode=edge+sim` baseline.

`graph_prior_only` and `qtsim_only` use debug-only confidence overrides (`debug_allow_confidence_override=true`) so production defaults are unchanged.

## Metrics

Primary metrics reported per mode:

- Latency: p50 and p95 end-to-end query time.
- Context efficiency: mean/median `prompt_context_len`, mean/median `fired_count`.
- Routing diagnostics: mean `route_router_conf_mean`, `route_relevance_conf_mean`, `route_policy_disagreement_mean`.
- QTsim usage proxies:
  - distribution of `route_router_conf_mean`
  - fraction of queries with `route_router_conf_mean > 0.7`

Secondary optional metrics:

- keyword hit ratio from `expected_keywords` in eval query file.
- downstream user outcomes (task success, correction rate, retry rate).

## Ablation matrix

For each dataset split or benchmark shard, run all five modes:

1. `vector_only`
2. `edge_sim_legacy`
3. `graph_prior_only`
4. `qtsim_only`
5. `learned`

Interpretation:

- `learned` should outperform `vector_only` on multi-hop and history queries.
- `learned` vs `edge_sim_legacy` isolates value of the route model and confidence mixing.
- `graph_prior_only` vs `qtsim_only` bounds the behavior between priors and QTsim router.
- `learned` should land between these bounds, shifting by query difficulty and confidence.

## How to interpret QTsim usage

Use the confidence proxy outputs to characterize routing behavior:

- High router-confidence mass (`>0.7`) indicates QTsim-driven selection dominates.
- Low router-confidence mass indicates fallback to graph-prior behavior.
- Rising policy disagreement with good outcomes can indicate productive re-ranking by QTsim.
- Rising disagreement with poor outcomes can indicate router overreach or calibration drift.

## Ground-truth dataset construction

Recommended process for paper-grade ground truth:

1. Collect real queries from replay logs, route traces, and operator workflows.
2. Stratify into categories (`decision-history`, `project-boundary`, `pointer`, `ops`).
3. For each query, annotate one or more acceptable target nodes/chunks and required facts.
4. Record ambiguity flags (single-answer vs multi-answer) and confidence in labels.
5. Keep train/dev/test splits by time or project slice to avoid leakage.
6. Add adversarial near-miss queries to test pointer-chasing reduction.
7. Track inter-annotator agreement and resolve disagreements with adjudication notes.

## Running the suite

- Eval and ablations:
  - `python examples/eval/run_eval.py --state /path/to/state.json --output /tmp/ocb_eval.json`
- Synthetic simulation:
  - `python examples/eval/simulate_two_cluster_routing.py --output-dir /tmp/ocb_two_cluster`

## Expected evidence package for paper

- per-mode JSON summaries from `run_eval.py`
- simulation CSV + report from `simulate_two_cluster_routing.py`
- qualitative failure analysis for low-confidence and high-disagreement queries
- appendix with dataset construction protocol and annotation rubric
