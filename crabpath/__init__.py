"""CrabPath public API."""

from .graph import Edge, Graph, Node
from .index import VectorIndex
from ._batch import batch_or_single, batch_or_single_embed
from .learn import apply_outcome
from .split import split_workspace
from .traverse import traverse

__all__ = [
    "Node",
    "Edge",
    "Graph",
    "VectorIndex",
    "traverse",
    "split_workspace",
    "apply_outcome",
    "batch_or_single",
    "batch_or_single_embed",
]

__version__ = "3.0.0"
