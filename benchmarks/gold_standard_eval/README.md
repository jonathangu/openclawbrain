# Gold-Standard Eval Harness

This suite provides reproducible, dataset-based and simulation-based benchmarks for OpenClawBrain without requiring the OpenClaw runtime. It includes:
- LoCoMo memory retrieval eval (dataset-based)
- A minimal agent-loop simulator for tool-use efficiency and call counting
- A toy call-counting eval (simulation-based)

## A/B Design

All benchmarks are designed for A/B comparisons:
- **Baseline**: No brain context (plain LLM/tool loop)
- **Brain-first**: Inject `BRAIN_CONTEXT` before the user request

A **turn** is defined as one LLM API call. Tool calls are counted separately.

## Fairness Caps

We enforce caps to keep comparisons fair and reproducible:
- **Brain context cap**: `--max-prompt-context-chars` (default 20k; use 80k for extended runs)
- **Tool result cap**: `--tool-result-max-chars` in the agent loop

## Metrics

LoCoMo (dataset-based):
- `recall_at_k`: retrieval hit rate when gold spans are present
- `avg_retrieved_chars`: average raw retrieved context size
- `avg_prompt_context_chars`: average size of final `BRAIN_CONTEXT`
- `proxy_recall`: heuristic overlap recall when gold spans are not provided

Toy call-counting (simulation-based):
- `baseline_avg_llm_calls`, `baseline_avg_tool_calls`
- `brain_avg_llm_calls`, `brain_avg_tool_calls`
- Token usage is counted when the OpenAI API returns usage metadata

## Running

Install dependencies:
```bash
pip install -e .[eval]
```

LoCoMo:
```bash
python3 benchmarks/gold_standard_eval/run_locomo.py --split test --max-examples 200 --max-prompt-context-chars 20000
```

Toy call-counting:
```bash
python3 benchmarks/gold_standard_eval/run_toy_calls.py --max-prompt-context-chars 20000
```

Agent-loop simulator:
```bash
python3 -c "from benchmarks.gold_standard_eval.agent_loop import run_agent_loop; print('ok')"
```

## Extending to GAIA/WebArena

The agent-loop simulator is intentionally minimal. We will extend it to:
- Load GAIA/WebArena task sets as tool-use benchmarks
- Measure tool-call efficiency and total tokens per user-visible exchange
- Compare baseline vs brain-first with identical fairness caps
