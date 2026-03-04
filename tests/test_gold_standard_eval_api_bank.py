from __future__ import annotations

from argparse import Namespace

from benchmarks.gold_standard_eval import run_api_bank


def test_api_bank_runs_in_dry_mode():
    args = Namespace(
        dataset="liminghao1630/API-Bank",
        split="test",
        train_split="train",
        max_examples=2,
        seed=7,
        max_steps=2,
        max_prompt_context_chars=2000,
        brain_examples=2,
        model="stub",
        temperature=0.0,
        allow_openai=False,
        stub_model=True,
        dry_run=True,
        output=None,
    )
    summary = run_api_bank.run(args)
    assert summary["examples_evaluated"] == 2
    assert "baseline_success_rate" in summary
    assert "brain_success_rate" in summary
