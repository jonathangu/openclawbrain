from __future__ import annotations

from argparse import Namespace

from benchmarks.gold_standard_eval import run_toy_calls


def test_toy_eval_runs_in_dry_mode():
    args = Namespace(
        max_tasks=3,
        max_prompt_context_chars=2000,
        tool_result_max_chars=200,
        output=None,
    )
    summary = run_toy_calls.run(args)
    assert summary["tasks"] == 3
    assert summary["baseline_avg_calls"] >= 1
    assert summary["brain_avg_calls"] >= 1
    assert summary["baseline_avg_llm_calls"] >= 1
    assert summary["brain_avg_llm_calls"] >= 1
