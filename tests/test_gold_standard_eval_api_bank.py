from __future__ import annotations

from argparse import Namespace
import json
import types
import sys

from benchmarks.gold_standard_eval import run_api_bank


def test_api_bank_runs_in_dry_mode():
    args = Namespace(
        dataset="liminghao1630/API-Bank",
        level=1,
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


def test_api_bank_uses_hf_hub_download(monkeypatch, tmp_path):
    train_payload = [
        {
            "id": "train-1",
            "instruction": "Schedule a yoga session.",
            "input": "Create a Yin yoga session on 2023-06-03 at 19:30 for 90 minutes.",
            "output": (
                "API-Request: [Create_New_Session(session_name='Yin yoga', "
                "session_date='2023-06-03', session_time='19:30', duration_in_minutes=90)]"
            ),
        }
    ]
    test_payload = [
        {
            "id": "test-1",
            "instruction": "Add a medication.",
            "input": "Log aspirin 75mg for every morning.",
            "expected_output": (
                "API-Request: [Add_Medication(medication_name='aspirin 75mg', "
                "dosage='take one tablet every morning')]"
            ),
            "file": "ignored.json",
            "extra": "ignored",
        }
    ]
    train_path = tmp_path / "lv1-api-train.json"
    test_path = tmp_path / "level-1-api.json"
    train_path.write_text(json.dumps(train_payload), encoding="utf-8")
    test_path.write_text(json.dumps(test_payload), encoding="utf-8")

    download_calls: list[tuple[str, str]] = []

    def fake_hf_hub_download(*, repo_id: str, repo_type: str, filename: str) -> str:
        download_calls.append((repo_id, filename))
        if "lv1-api-train" in filename:
            return str(train_path)
        return str(test_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_hf_hub_download))

    args = Namespace(
        dataset="liminghao1630/API-Bank",
        level=1,
        split="test",
        train_split="train",
        max_examples=1,
        seed=7,
        max_steps=1,
        max_prompt_context_chars=2000,
        brain_examples=0,
        model="stub",
        temperature=0.0,
        allow_openai=False,
        stub_model=True,
        dry_run=False,
        output=None,
    )
    summary = run_api_bank.run(args)
    assert summary["examples_evaluated"] == 1
    assert download_calls == [
        ("liminghao1630/API-Bank", "training-data/lv1-api-train.json"),
        ("liminghao1630/API-Bank", "test-data/level-1-api.json"),
    ]
