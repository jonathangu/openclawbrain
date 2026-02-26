from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from crabpath import Graph, Node
from crabpath import embeddings as embeddings_lib
from crabpath.providers.heuristic import HeuristicRouter
from crabpath.providers.registry import auto_detect_providers


def test_heuristic_router_picks_realistic_candidate() -> None:
    graph = Graph()
    graph.add_node(Node(id="deploy", content="Deploy release steps and smoke tests."))
    graph.add_node(Node(id="rollback", content="Rollback plan and incident recovery."))
    graph.add_node(Node(id="infra", content="Infrastructure maintenance"))

    candidates = [
        {
            "node_id": node.id,
            "summary": node.content,
            "weight": 0.2 + idx * 0.1,
            "current_node_id": "start",
        }
        for idx, node in enumerate(graph.nodes())
    ]
    # Keep the rollback node intentionally strongest for this rollout-focused query.
    for candidate in candidates:
        if candidate["node_id"] == "rollback":
            candidate["summary"] = (
                "Rollback plan and incident recovery after a failed deployment."
            )
            candidate["weight"] = 1.2
    candidates[0]["summary"] = (
        "Deploy release steps and smoke tests for safe rollout and validation."
    )
    router = HeuristicRouter(top_k=3)
    decision = router.route(
        "How do we rollback after a bad deploy?",
        candidates,
        schema={"tier": "habitual"},
    )

    assert decision["target"] == "rollback"
    assert decision["provider"] == "heuristic"
    assert decision["confidence"] > 0.0
    assert "alternatives" in decision


def test_auto_detect_respects_empty_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(
        "crabpath.providers.ollama_provider.ollama_embed",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no local service")),
    )

    embedding_provider, router_provider = auto_detect_providers()
    assert embedding_provider is None
    assert router_provider.name == "heuristic"


def test_openai_embed_chunks_requests(monkeypatch) -> None:
    create_calls: list[int] = []

    class FakeOpenAIEmbeddingService:
        def create(self, model: str, input: list[str]):
            create_calls.append(len(input))
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in input]
            )

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda: SimpleNamespace(
        embeddings=FakeOpenAIEmbeddingService()
    )

    original = sys.modules.get("openai")
    sys.modules["openai"] = fake_openai
    try:
        # Keep the chunk size deterministic even if a constant changes in future.
        old_limit = embeddings_lib.OPENAI_EMBEDDING_MAX_BATCH_SIZE
        embeddings_lib.OPENAI_EMBEDDING_MAX_BATCH_SIZE = 4
        try:
            embed_fn = embeddings_lib.openai_embed("text-embedding-3-small")
            vectors = embed_fn(["q"] * 5)
            assert len(vectors) == 5
            assert create_calls == [4, 1]
        finally:
            embeddings_lib.OPENAI_EMBEDDING_MAX_BATCH_SIZE = old_limit
    finally:
        if original is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = original


def test_gemini_embed_raises_readable_error() -> None:
    fake_genai = types.ModuleType("google.generativeai")
    fake_google = types.ModuleType("google")

    def configure(**_kwargs: object) -> None:
        return None

    def embed_content(**_kwargs: object) -> list[float]:
        error = RuntimeError("backend down")
        setattr(error, "status", 500)
        raise error

    fake_genai.configure = configure
    fake_genai.embed_content = embed_content
    fake_google.generativeai = fake_genai
    fake_google.__path__ = []  # type: ignore[attr-defined]

    original_google = sys.modules.get("google")
    original_genai = sys.modules.get("google.generativeai")
    sys.modules["google"] = fake_google
    sys.modules["google.generativeai"] = fake_genai
    try:
        with pytest.raises(ValueError, match="Gemini embeddings service returned HTTP 500"):
            _ = embeddings_lib.gemini_embed()(["q"])
    finally:
        if original_genai is None:
            sys.modules.pop("google.generativeai", None)
        else:
            sys.modules["google.generativeai"] = original_genai
        if original_google is None:
            sys.modules.pop("google", None)
        else:
            sys.modules["google"] = original_google


def test_ollama_embed_reports_disconnect(monkeypatch) -> None:
    import requests

    def fail(*_args: object, **_kwargs: object) -> None:
        raise requests.RequestException("service disconnected")

    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(ValueError, match="Ollama unavailable"):
        _ = embeddings_lib.ollama_embed()(["x"])
