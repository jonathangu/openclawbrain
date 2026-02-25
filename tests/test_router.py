from __future__ import annotations

import pytest

from crabpath.router import Router, RouterConfig, RouterError


def test_fallback_picks_highest_weight():
    router = Router()
    decision = router.fallback(
        [
            ("low", 0.15),
            ("high", 0.94),
            ("mid", 0.70),
        ],
        "reflex",
    )

    assert decision.chosen_target == "high"
    assert decision.tier == "reflex"
    assert decision.confidence > 0.9
    assert decision.alternatives[0][0] == "mid"


def test_parse_json_valid():
    router = Router()
    raw = '{"target":"node-1","confidence":0.81,"rationale":"best match","alternatives":[["node-2",0.2]]}'
    payload = router.parse_json(raw)

    assert payload["target"] == "node-1"
    assert payload["confidence"] == 0.81
    assert payload["rationale"] == "best match"


def test_parse_json_invalid_raises():
    router = Router()

    with pytest.raises(RouterError):
        router.parse_json("not json")

    with pytest.raises(RouterError):
        router.parse_json('{"target":123,"confidence":2.0,"rationale":7}')


def test_build_prompt_under_budget():
    router = Router()
    query = "How do I deploy a hotfix for memory pressure in production under high latency conditions?"
    candidates = [(f"node-{i}", float(i) / 10.0) for i in range(30)]
    context = {
        "node_summary": "Production incident runbook notes with multiple rollout steps."
    }

    prompt = router.build_prompt(query, candidates, context, budget=140)

    assert len(prompt.split()) < 200
    assert len(prompt.split()) <= 140
    assert "System:" in prompt
    assert "Query:" in prompt


def test_router_config_defaults():
    cfg = RouterConfig()

    assert cfg.model == "gpt-5-mini"
    assert cfg.temperature == 0.2
    assert cfg.timeout_s == 8.0
    assert cfg.max_retries == 2
    assert cfg.fallback_behavior == "heuristic"
