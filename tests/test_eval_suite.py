from __future__ import annotations

import csv
from pathlib import Path

import pytest

from examples.eval.run_eval import _load_queries, _percentile, _resolve_embed as _resolve_eval_embed
from examples.eval.simulate_openclaw_workflows import run_openclaw_workflow_simulation
from examples.eval.simulate_two_cluster_routing import run_two_cluster_simulation
from openclawbrain.eval.runner import _resolve_embed as _resolve_baseline_embed


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
                '{"id":"q2","query":"beta","category":"pointer","expected_keywords":["x","y"],"acceptable_node_ids":["n1"],"required_node_ids":["n2","n3"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = _load_queries(queries_path)
    assert [item.id for item in parsed] == ["q1", "q2"]
    assert parsed[0].expected_keywords == ()
    assert parsed[1].expected_keywords == ("x", "y")
    assert parsed[1].acceptable_node_ids == ("n1",)
    assert parsed[1].required_node_ids == ("n2", "n3")


def test_hash_embedder_resolution_uses_state_dimension() -> None:
    meta = {"embedder_name": "hash-v1", "embedder_dim": 64}
    eval_embed = _resolve_eval_embed(meta, "auto")
    baseline_embed = _resolve_baseline_embed(meta, "auto")

    assert len(eval_embed("alpha")) == 64
    assert len(baseline_embed("alpha")) == 64


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


def test_openclaw_workflow_simulation_improves_over_graph_prior(tmp_path: Path) -> None:
    summary = run_openclaw_workflow_simulation(
        output_dir=tmp_path / "workflow",
        embed_dim=64,
        epochs=16,
        rank=16,
        lr=0.25,
        label_temp=0.3,
    )

    assert float(summary["vector_topk_target_success_rate"]) == pytest.approx(0.0)
    assert float(summary["graph_prior_target_success_rate"]) == pytest.approx(0.5)
    assert float(summary["learned_target_success_rate"]) == pytest.approx(1.0)
    assert float(summary["learned_minus_graph_prior_target_success"]) >= 0.49
    assert int(summary["first_full_success_epoch"]) <= 16
    assert Path(summary["report_path"]).exists()
    assert Path(summary["per_query_matrix_csv_path"]).exists()
    assert Path(summary["per_query_matrix_md_path"]).exists()
    assert Path(summary["worked_example_path"]).exists()

    with Path(summary["per_query_matrix_csv_path"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 16
    payments_learned = next(
        row
        for row in rows
        if row["query_id"] == "payments_recovery_eval" and row["mode"] == "learned"
    )
    assert payments_learned["scenario"] == "payments_recovery"
    assert payments_learned["category"] == "decision-history"
    assert (
        payments_learned["prompt_context_included_node_ids"]
        == "doc::payments_incident_2026_02_14|doc::rollback_gate|hub::incident"
    )
    assert payments_learned["target_success"] == "1.0"
    assert payments_learned["required_node_coverage"] == "1.0"
    assert payments_learned["pointer_turns"] == ""

    report_text = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "| vector_topk | 0/4 (0.00) | 0.00 | - |" in report_text
    assert "| pointer_chase | 1/4 (0.25) | 0.38 | 1.25 |" in report_text
    assert "| graph_prior_only | 2/4 (0.50) | 0.50 | - |" in report_text
    assert "| learned | 4/4 (1.00) | 1.00 | - |" in report_text
    assert "## Scenario by mode" in report_text

    matrix_md = Path(summary["per_query_matrix_md_path"]).read_text(encoding="utf-8")
    assert "| payments_recovery_eval | payments_recovery | decision-history | learned |" in matrix_md
    assert (
        "| oncall_dashboard_recall_eval | oncall_dashboard_recall | ops | pointer_chase | "
        "doc::oncall_schedule|doc::monitoring_dashboards | doc::monitoring_dashboards|hub::incident | 0.00 | 0.50 | 1 |"
        in matrix_md
    )
