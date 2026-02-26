"""Provider registry and auto-detection helpers."""

from __future__ import annotations

import os
from typing import Any

from .base import EmbeddingProvider, RouterProvider


def _instantiate_router_provider(factory: Any) -> RouterProvider | None:
    try:
        return factory()
    except Exception:
        return None


def _env_has_value(name: str) -> bool:
    value = os.getenv(name)
    return bool((value or "").strip())


def auto_detect_providers() -> tuple[EmbeddingProvider | None, RouterProvider]:
    """Auto-detect best available providers."""
    embedding_provider: EmbeddingProvider | None = None
    router_provider: RouterProvider

    # Custom endpoint providers:
    llm_url = (os.getenv("CRABPATH_LLM_URL") or "").strip()
    llm_token = (os.getenv("CRABPATH_LLM_TOKEN") or "").strip() or None
    llm_model = (os.getenv("CRABPATH_LLM_MODEL") or "").strip() or "gpt-5-mini"

    embed_url = (os.getenv("CRABPATH_EMBEDDINGS_URL") or "").strip()
    embed_token = (os.getenv("CRABPATH_EMBEDDINGS_TOKEN") or "").strip() or None
    embed_model = (
        (os.getenv("CRABPATH_EMBEDDINGS_MODEL") or "").strip() or "text-embedding-3-small"
    )

    if embed_url:
        from .endpoint_provider import EndpointEmbeddingProvider

        try:
            embedding_provider = EndpointEmbeddingProvider(
                embed_url,
                embed_token,
                embed_model,
            )
        except Exception:
            embedding_provider = None

    if embedding_provider is None:
        # Embeddings priority:
        # 1) OpenAI
        # 2) Gemini
        # 3) Ollama
        # 4) Local TF-IDF fallback
        if _env_has_value("OPENAI_API_KEY"):
            from .openai_provider import OpenAIEmbeddingProvider

            try:
                embedding_provider = OpenAIEmbeddingProvider()
            except Exception:
                embedding_provider = None

        if embedding_provider is None and (
            _env_has_value("GEMINI_API_KEY") or _env_has_value("GOOGLE_API_KEY")
        ):
            from .gemini_provider import GeminiEmbeddingProvider

            try:
                embedding_provider = GeminiEmbeddingProvider()
            except Exception:
                embedding_provider = None

        if embedding_provider is None:
            from .ollama_provider import OllamaEmbeddingProvider

            candidate = _instantiate_router_provider(lambda: OllamaEmbeddingProvider())
            if isinstance(candidate, EmbeddingProvider):
                embedding_provider = candidate

        if embedding_provider is None:
            from .tfidf_provider import TfidfEmbeddingProvider

            embedding_provider = TfidfEmbeddingProvider()

    # Routing provider
    if llm_url:
        from .endpoint_provider import EndpointRouterProvider

        router_provider = _instantiate_router_provider(
            lambda: EndpointRouterProvider(llm_url, llm_token, llm_model)
        )
    else:
        router_provider = None

    if router_provider is None:
        # Routing priority:
        # 1) OpenAI
        # 2) Gemini
        # 3) Heuristic (always available)
        from .openai_provider import OpenAIRouterProvider
        from .gemini_provider import GeminiRouterProvider
        from .heuristic import HeuristicRouter

        router_provider = _instantiate_router_provider(
            lambda: OpenAIRouterProvider()
        )  # type: ignore[assignment]
        if router_provider is None:
            router_provider = _instantiate_router_provider(
                lambda: GeminiRouterProvider()
            )  # type: ignore[assignment]
        if router_provider is None:
            router_provider = HeuristicRouter()

    return embedding_provider, router_provider


def get_embedding_provider(name: str | None = None) -> EmbeddingProvider | None:
    """Get a specific embedding provider by name, or auto-detect."""
    if name is None or str(name).lower() == "auto":
        embedding, _ = auto_detect_providers()
        return embedding

    provider_name = str(name).lower().strip()
    if provider_name == "heuristic":
        return None

    if provider_name == "openai":
        from .openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider()

    if provider_name == "gemini":
        from .gemini_provider import GeminiEmbeddingProvider

        return GeminiEmbeddingProvider()

    if provider_name == "ollama":
        from .ollama_provider import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider()

    raise ValueError(f"Unknown embedding provider: {name}")


def get_router_provider(name: str | None = None) -> RouterProvider:
    """Get a specific router provider by name, or auto-detect."""
    if name is None or str(name).lower() == "auto":
        _, router = auto_detect_providers()
        return router

    provider_name = str(name).lower().strip()
    if provider_name == "heuristic":
        from .heuristic import HeuristicRouter

        return HeuristicRouter()

    if provider_name == "openai":
        from .openai_provider import OpenAIRouterProvider

        return OpenAIRouterProvider()

    if provider_name == "gemini":
        from .gemini_provider import GeminiRouterProvider

        return GeminiRouterProvider()

    if provider_name == "ollama":
        from .ollama_provider import OllamaRouterProvider

        return OllamaRouterProvider()

    raise ValueError(f"Unknown router provider: {name}")
