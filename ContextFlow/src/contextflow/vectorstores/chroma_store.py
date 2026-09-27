# contextflow/vectorstores/chroma_store.py
"""
Step 15: ChromaDB backend for the VectorStore interface.

ChromaVectorStore persists message IDs, content, metadata, and embeddings in
a local ChromaDB collection (in-memory by default, on-disk when
CHROMA_PERSIST_DIR is set) and answers nearest-neighbour queries with cosine
distance -- no full scan in Python.

Configuration (via environment variables or constructor args):
    CHROMA_COLLECTION  -- Optional. Collection name (default "contextflow").
    CHROMA_PERSIST_DIR -- Optional. Directory for on-disk persistence.
                          Unset (default) keeps everything in memory.

A ready-made client can be injected (useful for tests); otherwise one is
built from persist_dir. All backend failures surface as VectorStoreError.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .base import (
    VectorHit,
    VectorStore,
    VectorStoreError,
    similarity_from_distance,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_DEFAULT_COLLECTION = "contextflow"

# Metadata keys ContextFlow reserves when round-tripping Messages.
_RESERVED_META = ("role", "importance", "protected", "token_count", "extra")


def _sanitize_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce a metadata mapping into Chroma-safe primitives.

    Chroma accepts str / int / float / bool values only. None values are
    dropped; anything else is JSON-encoded (with str() as a last resort).
    """
    clean: Dict[str, Any] = {}
    for key, value in dict(metadata).items():
        if value is None:
            continue
        if isinstance(value, bool) or isinstance(value, (str, int, float)):
            clean[str(key)] = value
        else:
            try:
                clean[str(key)] = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                clean[str(key)] = str(value)
    return clean


class ChromaVectorStore(VectorStore):
    """Local ChromaDB backend (ephemeral or persistent).

    Args:
        collection_name: Collection to use. Defaults to CHROMA_COLLECTION
                         env var or "contextflow".
        persist_dir:     Directory for on-disk storage. Defaults to
                         CHROMA_PERSIST_DIR env var or None (in-memory).
        client:          Optional pre-built chromadb client (for tests or
                         custom configuration); skips client construction.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        env_collection = os.getenv("CHROMA_COLLECTION", "").strip()
        env_persist = os.getenv("CHROMA_PERSIST_DIR", "").strip()

        name = (collection_name if collection_name is not None else env_collection)
        name = name.strip() if isinstance(name, str) else name
        if not isinstance(name, str) or not name:
            if collection_name is not None or not env_collection:
                raise ValueError(
                    f"collection_name must be a non-empty string, got {collection_name!r}."
                    if collection_name is not None else
                    "collection_name must be a non-empty string."
                )
            name = _DEFAULT_COLLECTION
        if not name.strip():
            raise ValueError("collection_name must be a non-empty string.")
        self.collection_name: str = name.strip()

        if persist_dir is not None and not isinstance(persist_dir, str):
            raise TypeError(
                "persist_dir must be a directory path string or None, "
                f"got {type(persist_dir).__name__!r}."
            )
        resolved = persist_dir if persist_dir is not None else (env_persist or None)
        self.persist_dir: Optional[str] = resolved

        if client is not None:
            self._client = client
        else:
            try:
                import chromadb as _chromadb
            except ImportError as exc:
                raise VectorStoreError(
                    "The 'chromadb' package is not installed. "
                    "Run: pip install chromadb"
                ) from exc
            try:
                if self.persist_dir:
                    self._client = _chromadb.PersistentClient(path=self.persist_dir)
                else:
                    self._client = _chromadb.Client()
            except Exception as exc:
                raise VectorStoreError(
                    f"Cannot create ChromaDB client ({exc})."
                ) from exc

        try:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Cannot open ChromaDB collection {self.collection_name!r} ({exc})."
            ) from exc

    # ── writes ─────────────────────────────────────────────────────

    def add(
        self,
        ids: Sequence[str],
        contents: Sequence[str],
        metadatas: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> int:
        """Add new records (backend rejects duplicate IDs)."""
        clean_ids, items, metas, vecs = self._validate_write(ids, contents, metadatas, embeddings)
        kwargs: Dict[str, Any] = {
            "ids": clean_ids,
            "documents": items,
            "metadatas": [_sanitize_metadata(m) for m in metas],
        }
        if vecs is not None:
            kwargs["embeddings"] = vecs
        try:
            self._collection.add(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB add failed ({exc}).") from exc
        return len(clean_ids)

    def upsert(
        self,
        ids: Sequence[str],
        contents: Sequence[str],
        metadatas: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> int:
        """Insert records or replace the ones sharing their IDs."""
        clean_ids, items, metas, vecs = self._validate_write(ids, contents, metadatas, embeddings)
        kwargs: Dict[str, Any] = {
            "ids": clean_ids,
            "documents": items,
            "metadatas": [_sanitize_metadata(m) for m in metas],
        }
        if vecs is not None:
            kwargs["embeddings"] = vecs
        try:
            self._collection.upsert(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB upsert failed ({exc}).") from exc
        return len(clean_ids)

    def delete(self, ids: Sequence[str]) -> int:
        """Delete records by ID; returns the number of IDs requested."""
        clean_ids = self._validate_ids(ids)
        try:
            self._collection.delete(ids=clean_ids)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB delete failed ({exc}).") from exc
        return len(clean_ids)

    # ── reads ──────────────────────────────────────────────────────

    def exists(self, ids: Sequence[str]) -> List[bool]:
        """Return one stored-flag per ID, in input order."""
        clean_ids = self._validate_ids(ids)
        try:
            found = self._collection.get(ids=clean_ids, include=[])
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB exists check failed ({exc}).") from exc
        present = set(found.get("ids", []) or [])
        return [i in present for i in clean_ids]

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        top_k: int = 5,
    ) -> List[VectorHit]:
        """Return the top_k nearest hits for the first query vector."""
        limit = self._validate_top_k(top_k)
        vecs = list(query_embeddings)
        if not vecs:
            return []
        for vector in vecs:
            if isinstance(vector, tuple):
                vector = list(vector)
            if (
                not isinstance(vector, list)
                or not vector
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vector)
            ):
                raise VectorStoreError("query_embeddings must be non-empty lists of numbers.")
        try:
            total = self._collection.count()
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB count failed ({exc}).") from exc
        if total == 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[[float(v) for v in vecs[0]]],
                n_results=min(limit, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed ({exc}).") from exc
        try:
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            dists = (result.get("distances") or [[]])[0]
        except (AttributeError, IndexError, TypeError) as exc:
            raise VectorStoreError(
                f"ChromaDB returned a malformed query response ({exc})."
            ) from exc
        hits: List[VectorHit] = []
        for rid, doc, meta, dist in zip(ids, docs, metas, dists):
            distance = float(dist)
            hits.append(VectorHit(
                id=str(rid),
                content=str(doc),
                metadata=dict(meta) if isinstance(meta, dict) else {},
                score=similarity_from_distance(distance),
                distance=distance,
            ))
        hits.sort(key=lambda h: h.distance)
        return hits[:limit]

    def count(self) -> int:
        """Return the number of stored records."""
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB count failed ({exc}).") from exc

    def clear(self) -> None:
        """Delete every record in the collection."""
        try:
            found = self._collection.get(include=[])
            stored = found.get("ids", []) or []
            if stored:
                self._collection.delete(ids=list(stored))
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB clear failed ({exc}).") from exc
