from __future__ import annotations

from pathlib import Path

from benchmarks.gold_standard_eval import run_locomo


def test_locomo_loader_smoke():
    fixture = Path("tests/fixtures/locomo10_tiny.json")
    examples = run_locomo._load_locomo10_json(str(fixture))
    assert len(examples) == 2

    first = examples[0]
    messages = first["messages"]
    assert messages[0]["text"] == "Hi Bob."
    assert messages[2]["text"] == "I love pizza."
    assert first["question"] == "What food does Alice love?"
    assert first["answers"] == ["pizza"]
