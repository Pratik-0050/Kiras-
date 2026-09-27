# contextflow/vectorstores/__init__.py
"""
Vector-store subsystem for ContextFlow.

Provides persistent nearest-neighbour search over message embeddings so
retrieval scales without scanning every message in Python.
"""

from .base import (
    VectorStore,
    BaseVectorStore,
    VectorHit,
    VectorStoreError,
    similarity_from_distance,
)
from .chroma_store import ChromaVectorStore

__all__ = [
    "VectorStore",
    "BaseVectorStore",
    "VectorHit",
    "VectorStoreError",
    "similarity_from_distance",
    "ChromaVectorStore",
]
