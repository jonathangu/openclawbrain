"""Provider registry and abstractions for embeddings and routing."""

from .base import EmbeddingProvider, RouterProvider
from .registry import auto_detect_providers, get_embedding_provider, get_router_provider

__all__ = [
    "EmbeddingProvider",
    "RouterProvider",
    "auto_detect_providers",
    "get_embedding_provider",
    "get_router_provider",
]
