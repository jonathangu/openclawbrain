from __future__ import annotations

from pathlib import Path
import csv

from examples.eval.simulate_expert_regions import run_expert_regions_simulation


def _run_small(out_dir: Path, *, seed: int) -> dict[str, object]:
    return run_expert_regions_simulation(
        output_dir=out_dir,
        k_experts=8,
        dim=16,
        num_components=20,
        train_queries=900,
        test_queries=320,
        epochs=6,
        rank=6,
        lr=0.08,
        label_temp=0.6,
        seed=seed,
    )


def test_oracle_reward_has_clear_gap_over_random(tmp_path: Path) -> None:
    summary = _run_small(tmp_path / "gap_check", seed=21)
    oracle_reward = float(summary["oracle_reward"])
    random_reward = float(summary["random_reward"])
    assert oracle_reward - random_reward > 0.45


def test_learned_closes_at_least_40_percent_of_oracle_gap(tmp_path: Path) -> None:
    summary = _run_small(tmp_path / "learned_check", seed=22)
    gap_closed = float(summary["final_gap_closed"])
    assert gap_closed >= 0.40


def test_learned_reward_beats_random_by_margin(tmp_path: Path) -> None:
    summary = _run_small(tmp_path / "reward_gap", seed=24)
    learned_reward = float(summary["final_learned_reward"])
    random_reward = float(summary["random_reward"])
    assert learned_reward - random_reward > 0.10


def test_expert_regions_simulation_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    summary_a = _run_small(out_a, seed=23)
    summary_b = _run_small(out_b, seed=23)

    keys_to_compare = [
        "oracle_reward",
        "random_reward",
        "graph_prior_reward",
        "final_learned_reward",
        "final_learned_accuracy",
        "final_gap_closed",
        "train_reward",
        "test_reward",
        "train_gap_closed",
        "test_gap_closed",
        "epochs",
    ]
    for key in keys_to_compare:
        assert summary_a[key] == summary_b[key]

    curve_a = (out_a / "simulation_curve.csv").read_text(encoding="utf-8")
    curve_b = (out_b / "simulation_curve.csv").read_text(encoding="utf-8")
    assert curve_a == curve_b

    report_a = (out_a / "report.md").read_text(encoding="utf-8")
    report_b = (out_b / "report.md").read_text(encoding="utf-8")
    assert report_a == report_b


def test_curve_includes_industry_baselines(tmp_path: Path) -> None:
    summary = _run_small(tmp_path / "curve_check", seed=31)
    curve_path = Path(summary["curve_path"])
    with curve_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        policies = {row["policy"] for row in reader}
    assert {"vector_topk", "vector_topk_rerank", "pointer_chase"}.issubset(policies)


def test_report_includes_industry_baselines_section(tmp_path: Path) -> None:
    summary = _run_small(tmp_path / "report_check", seed=32)
    report_path = Path(summary["report_path"])
    report = report_path.read_text(encoding="utf-8")
    assert "## Industry baselines" in report
