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


def test_resolve_ollama_timeout_seconds(monkeypatch) -> None:
    """Timeout resolver prefers OPENCLAWBRAIN_OLLAMA_TIMEOUT_SECONDS with validation."""
    module = importlib.import_module("openclawbrain.ollama_llm")

    monkeypatch.delenv("OPENCLAWBRAIN_OLLAMA_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("OLLAMA_TIMEOUT_SECONDS", raising=False)
    assert module._resolve_ollama_timeout_seconds() == 600

    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "120")
    assert module._resolve_ollama_timeout_seconds() == 120

    monkeypatch.setenv("OPENCLAWBRAIN_OLLAMA_TIMEOUT_SECONDS", "42")
    assert module._resolve_ollama_timeout_seconds() == 42

    monkeypatch.setenv("OPENCLAWBRAIN_OLLAMA_TIMEOUT_SECONDS", "0")
    assert module._resolve_ollama_timeout_seconds() == 600

    monkeypatch.setenv("OPENCLAWBRAIN_OLLAMA_TIMEOUT_SECONDS", "-5")
    assert module._resolve_ollama_timeout_seconds() == 600

    monkeypatch.setenv("OPENCLAWBRAIN_OLLAMA_TIMEOUT_SECONDS", "not-a-number")
    assert module._resolve_ollama_timeout_seconds() == 600
