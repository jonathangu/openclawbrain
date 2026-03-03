"""Tests for Ollama LLM integration."""

from __future__ import annotations

import importlib


def test_ollama_llm_fn_honors_env_model(monkeypatch) -> None:
    """ollama_llm_fn uses OPENCLAWBRAIN_OLLAMA_MODEL when model is None."""
    monkeypatch.setenv("OPENCLAWBRAIN_OLLAMA_MODEL", "sentinel-model")

    module = importlib.import_module("openclawbrain.ollama_llm")
    calls: list[str] = []

    def _fake_chat(system: str, user: str, *, model: str) -> str:
        calls.append(model)
        return "ok"

    monkeypatch.setattr(module, "_ollama_chat", _fake_chat)

    assert module.ollama_llm_fn("system", "user") == "ok"
    assert calls == ["sentinel-model"]
