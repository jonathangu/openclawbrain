"""
🦀 CrabPath — Neuron-inspired memory graphs for AI agents.
Everything evolves into this.
"""

__version__ = "0.4.0"

from .graph import Graph, Node, Edge
from .activation import activate, learn, Firing
from .embeddings import EmbeddingIndex, openai_embed

__all__ = [
    "Graph", "Node", "Edge",
    "activate", "learn", "Firing",
    "EmbeddingIndex", "openai_embed",
]
