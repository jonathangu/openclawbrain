# Guardrails and Rollback

## Failure modes

**Drift / pollution**: Non-representative sessions or noisy tool output push retrieval toward irrelevant nodes.

**Bad learning**: Incorrect feedback or automated scoring applies strong negative/positive updates to the wrong path.

**Teacher noise**: Async teacher traces or labels contain low-quality guidance that overfits the graph.

## Guardrails

**Reward-source hierarchy**: Prefer explicit human feedback over auto-scored outcomes, and auto-scored outcomes over heuristic inference. Avoid mixing sources at equal weight.

**Fail-open retrieval**: If the brain query or daemon fails, OpenClaw proceeds with the base prompt unchanged.

**Redaction and exclude paths**: Redact common secrets and exclude sensitive directories from retrieval to avoid contaminated context.

## Rollback playbook

1. Stop the daemon: `openclawbrain serve stop --state ~/.openclawbrain/<agent>/state.json`
2. Restore state: Fast path is copy `state.bak` over `state.json`. Safer path is restore from your latest full backup.
3. Disable the hook if needed: `openclaw hooks disable openclawbrain-context-injector`; `openclaw gateway restart`.
4. Validate health: `openclawbrain info --state ~/.openclawbrain/<agent>/state.json`; run a known-good query and verify `[BRAIN_CONTEXT]` matches expected nodes.

If the issue is tied to noisy learning events, archive or delete the current `learning_events.jsonl` before re-enabling learning.
