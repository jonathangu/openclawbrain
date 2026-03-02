from __future__ import annotations

from pathlib import Path

import pytest

from examples.eval.run_eval import _load_queries, _percentile
from examples.eval.simulate_two_cluster_routing import run_two_cluster_simulation


def test_percentile_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(values, 50) == 2.5
    assert _percentile(values, 95) == pytest.approx(3.85)


def test_load_queries_jsonl_parses_expected_fields(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        "\n".join(
            [
                '{"id":"q1","query":"alpha","category":"ops"}',
                '{"id":"q2","query":"beta","category":"pointer","expected_keywords":["x","y"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = _load_queries(queries_path)
    assert [item.id for item in parsed] == ["q1", "q2"]
    assert parsed[0].expected_keywords == ()
    assert parsed[1].expected_keywords == ("x", "y")


def test_two_cluster_simulation_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    summary_a = run_two_cluster_simulation(
        output_dir=out_a,
        dim=6,
        samples_per_cluster=12,
        epochs=4,
        lr=0.1,
        rank=3,
        label_temp=0.5,
        seed=11,
    )
    summary_b = run_two_cluster_simulation(
        output_dir=out_b,
        dim=6,
        samples_per_cluster=12,
        epochs=4,
        lr=0.1,
        rank=3,
        label_temp=0.5,
        seed=11,
    )

    assert summary_a["initial_cluster_accuracy"] == summary_b["initial_cluster_accuracy"]
    assert summary_a["final_cluster_accuracy"] == summary_b["final_cluster_accuracy"]
    assert summary_a["initial_top1_accuracy"] == summary_b["initial_top1_accuracy"]
    assert summary_a["final_top1_accuracy"] == summary_b["final_top1_accuracy"]
    assert summary_a["initial_ce_loss"] == summary_b["initial_ce_loss"]
    assert summary_a["final_ce_loss"] == summary_b["final_ce_loss"]
    assert summary_a["ce_loss_overall_decrease"] == summary_b["ce_loss_overall_decrease"]
    assert summary_a["ce_loss_monotonic_nonincreasing"] == summary_b["ce_loss_monotonic_nonincreasing"]

    curve_a = (out_a / "simulation_curve.csv").read_text(encoding="utf-8")
    curve_b = (out_b / "simulation_curve.csv").read_text(encoding="utf-8")
    assert curve_a == curve_b
    assert curve_a.splitlines()[0] == "epoch,ce_loss,cluster_accuracy,top1_accuracy"
