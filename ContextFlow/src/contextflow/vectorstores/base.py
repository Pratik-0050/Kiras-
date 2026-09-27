# contextflow/vectorstores/base.py
"""
Step 15: VectorStore interface for persistent semantic search.

A VectorStore persists message IDs, content, metadata, and embeddings, and
answers nearest-neighbour queries without scanning every message in Python.
Backends (ChromaDB today, others later) implement this interface; retrievers
program against the interface so backends stay interchangeable.

Distances use cosine space. Scores map cosine distance d in [0.0, 2.0] to a
0.0-1.0 similarity matching SemanticRetriever's scale::

    score = round(max(0.0, 1.0 - d / 2.0), 3)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


class VectorStoreError(Exception):
    """Raised when vector-store operations fail.

    Covers missing backends, connection problems, invalid arguments that
    reach the backend, and malformed backend responses. A query with no
    matches is NOT an error -- it returns an empty list.
    """


@dataclass
class VectorHit:
    """One nearest-neighbour hit from a vector-store query.

    Attributes:
        id:       Stable record ID (ContextFlow uses the message content hash).
        content:  Stored document text (the message content).
        metadata: Stored metadata (role, importance, protected, ...).
        score:    Similarity in 0.0-1.0 (higher is more relevant).
        distance: Raw cosine distance from the backend.
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    distance: float = 0.0


def similarity_from_distance(distance: float) -> float:
    """Map a cosine distance in [0.0, 2.0] to a 0.0-1.0 similarity score."""
    return round(max(0.0, 1.0 - float(distance) / 2.0), 3)


class VectorStore(ABC):
    """Abstract interface that every vector-store backend must implement."""

    # ── validation helpers (shared by backends and test fakes) ──────

    @staticmethod
    def _validate_ids(ids: Sequence[str], label: str = "ids") -> List[str]:
        items = list(ids)
        if not items:
            raise VectorStoreError(f"{label} must not be empty.")
        for value in items:
            if not isinstance(value, str) or not value.strip():
                raise VectorStoreError(
                    f"{label} must be non-empty strings, got {value!r}."
                )
        if len(set(items)) != len(items):
            raise VectorStoreError(f"{label} must be unique (duplicates found).")
        return items

    @staticmethod
    def _validate_write(
        ids: Sequence[str],
        contents: Sequence[str],
        metadatas: Optional[Sequence[Optional[Mapping[str, Any]]]],
        embeddings: Optional[Sequence[Sequence[float]]],
    ) -> tuple[List[str], List[str], List[Dict[str, Any]], Optional[List[List[float]]]]:
        clean_ids = VectorStore._validate_ids(ids)
        items = list(contents)
        if len(items) != len(clean_ids):
            raise VectorStoreError(
                f"contents ({len(items)}) and ids ({len(clean_ids)}) must match in length."
            )
        for content in items:
            if not isinstance(content, str) or not content.strip():
                raise VectorStoreError("contents must be non-empty strings.")
        clean_metas: List[Dict[str, Any]] = (
            [dict(m) if m is not None else {} for m in metadatas]
            if metadatas is not None
            else [{} for _ in clean_ids]
        )
        if len(clean_metas) != len(clean_ids):
            raise VectorStoreError("metadatas length must match ids length.")
        clean_vecs: Optional[List[List[float]]] = None
        if embeddings is not None:
            clean_vecs = []
            raw_vecs = list(embeddings)
            if len(raw_vecs) != len(clean_ids):
                raise VectorStoreError("embeddings length must match ids length.")
            for vector in raw_vecs:
                if isinstance(vector, tuple):
                    vector = list(vector)
                if (
                    not isinstance(vector, list)
                    or not vector
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vector)
                ):
                    raise VectorStoreError(
                        "embeddings must be non-empty lists of numbers."
                    )
                clean_vecs.append([float(v) for v in vector])
        return clean_ids, items, clean_metas, clean_vecs

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise VectorStoreError(
                f"top_k must be a positive integer, got {type(top_k).__name__!r}."
            )
        if top_k <= 0:
            raise VectorStoreError(f"top_k must be a positive integer, got {top_k}.")
        return top_k

    # ── abstract operations ─────────────────────────────────────────

    @abstractmethod
    def add(
        self,
        ids: Sequence[str],
        contents: Sequence[str],
        metadatas: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> int:
        """Add new records (raises VectorStoreError on duplicate IDs)."""
        pass

    @abstractmethod
    def upsert(
        self,
        ids: Sequence[str],
        contents: Sequence[str],
        metadatas: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> int:
        """Insert records or replace the ones sharing their IDs."""
        pass

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> int:
        """Delete records by ID. Returns the number of IDs requested."""
        pass

    @abstractmethod
    def exists(self, ids: Sequence[str]) -> List[bool]:
        """Return one flag per ID (True when the record is stored)."""
        pass

    @abstractmethod
    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        top_k: int = 5,
    ) -> List[VectorHit]:
        """Return the top_k nearest hits for the first query vector."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored records."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete every record in the store."""
        pass


# BaseVectorStore is an alias kept for naming consistency with the other
# subsystems (BaseSummarizer, BaseValidator, BaseRetriever, ...).
BaseVectorStore = VectorStore
