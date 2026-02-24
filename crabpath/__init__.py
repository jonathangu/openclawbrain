"""
🦀 CrabPath — Activation-driven memory graphs for AI agents.
Everything evolves into this.
"""

__version__ = "0.0.1"

from .graph import Graph, Node, Edge
from .activation import activate, learn, Result

__all__ = ["Graph", "Node", "Edge", "activate", "learn", "Result"]
