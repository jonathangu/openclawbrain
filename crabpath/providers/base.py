"""Base provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class EmbeddingProvider(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns list of vectors."""
        ...

    @abstractmethod
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        ...


class RouterProvider(ABC):
    name: str

    @abstractmethod
    def route(self, query: str, candidates: list[dict], schema: dict | None = None) -> dict[str, Any]:
        """Select which candidates to follow. Returns JSON decision."""
        ...
