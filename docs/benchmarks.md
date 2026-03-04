# Benchmarks

OpenClawBrain ships lightweight benchmarking tools in `benchmarks/` to help you compare:
- retrieval quality before/after learning loops
- replay and maintenance throughput
- cost/latency tradeoffs for different embedder + LLM settings

## Quickstart

```bash
python3 benchmarks/run_benchmark.py --help
```

Common inputs:
- `benchmarks/queries.json` for canned query prompts
- `benchmarks/results.json` for baseline output comparisons

## What to measure

- **Cold start**: fresh `state.json` vs. warmed (after replay/maintain).
- **Loop impact**: run `openclawbrain loop run --mode full` and compare before/after retrieval.
- **Latency**: query time with/without `openclawbrain serve`.

## Notes

- For apples-to-apples results, keep embedder + LLM settings fixed.
- LLM-backed fast-learning can be cost-heavy; edges-only replay is LLM-free.
- Store benchmark artifacts under `~/.openclawbrain/<agent>/scratch` for easier diffing.

## Gold-standard eval harness

See `benchmarks/gold_standard_eval` for the dataset-based + simulation harness:
- LoCoMo memory retrieval eval (dataset-based)
- Minimal agent-loop simulator for tool-use efficiency and call counting
- Toy call-counting tasks to validate LLM/tool step metrics

Planned extensions include GAIA/WebArena task loaders and richer tool-use metrics under the same fairness caps.
