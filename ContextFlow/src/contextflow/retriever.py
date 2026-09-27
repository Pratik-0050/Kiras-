# contextflow/retriever.py
"""
Step 13: Relevant context retrieval (deterministic keyword matching).

KeywordRetriever searches persisted conversation messages -- including
compaction summaries (they are plain Messages with
``metadata["type"] == "compaction_summary"``) -- and returns the most
relevant ones ranked by a 0.0-1.0 relevance score.

The implementation is deliberately dependency-free and deterministic:
no embeddings, no vector databases, no randomness. Scoring uses
case-insensitive keyword overlap between the query and each message:

    score = min(1.0, 0.7 * coverage + 0.3 * frequency + phrase_bonus)

where ``coverage`` is the fraction of distinct query terms found in the
message, ``frequency`` rewards repeated mentions (capped), and
``phrase_bonus`` (+0.15) rewards an exact-phrase hit. Ties break by
original message order, so repeated calls return identical rankings.

Duplicate content (identical text after lowercasing and whitespace
collapsing) is returned only once -- the highest-scoring copy wins.

Step 15 adds VectorStoreRetriever: messages are indexed once (ID = content
hash, plus role / importance / protected metadata and embeddings) and each
query embeds only the query text, letting the vector store -- not a Python
scan -- find the neighbours. Keyword and in-memory semantic retrieval stay
available as fallbacks.

Step 16 adds HybridRetriever and ContextAssembler (see hybrid.py), which fuse
keyword + semantic rankings and pack the final budget-aware context.
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union

from .conversation import Conversation
from .embeddings import (
    _content_key,
    EmbeddingError,
    EmbeddingProvider,
    validate_vector,
)
from .message import Message
from .vectorstores.base import VectorStore, VectorStoreError

if TYPE_CHECKING:
    from .store import ContextStore

# Sources accepted by Retriever.retrieve().
RetrievalSource = Union["Conversation", "ContextStore", Sequence["Message"]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")

# Common English stopwords -- dropped from queries and message matching so
# that "what is the ..." style queries score on content words, not grammar.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "have", "has", "had", "having",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "and", "or", "but", "if", "then", "else", "when", "at", "by", "for",
    "with", "about", "into", "through", "during", "of", "on", "in", "to",
    "from", "up", "down", "out", "over", "under", "as", "so", "than",
    "too", "very", "can", "will", "just", "should", "now", "how", "what",
    "which", "who", "whom", "this", "that", "these", "those", "there",
    "here", "not", "no", "yes",
})

_PHRASE_BONUS = 0.15
_FREQ_CAP = 8.0


class RetrieverError(Exception):
    """Raised when retrieval cannot run (bad source type, bad parameters)."""


@dataclass
class RetrievalResult:
    """One ranked retrieval hit.

    Attributes:
        score:         Relevance in 0.0-1.0 (higher is more relevant).
        message:       The matching Message object (from the conversation).
        matched_terms: Sorted list of distinct query terms found in the message.
        rank:          1-based position in the ranked result list.
    """
    score: float
    message: Message
    matched_terms: List[str] = field(default_factory=list)
    rank: int = 0

    def __str__(self) -> str:
        preview = self.message.content[:60] + ("..." if len(self.message.content) > 60 else "")
        terms = ", ".join(self.matched_terms) if self.matched_terms else "-"
        return (
            f"#{self.rank} score={self.score:.3f} "
            f"[{self.message.role}] {preview} (terms: {terms})"
        )


class Retriever(ABC):
    """Abstract interface that every retriever must implement.

    Subclasses must implement retrieve(): rank messages from a Conversation,
    ContextStore, or plain message list against a query string.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        source: RetrievalSource,
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """Return the most relevant messages for *query*, best first."""
        pass


# BaseRetriever is an alias kept for naming consistency with the other
# subsystems (BaseSummarizer, BaseValidator, BasePriorityScorer).
BaseRetriever = Retriever


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, minus stopwords and single chars."""
    return [
        tok for tok in _TOKEN_RE.findall(text.lower())
        if tok not in STOPWORDS and len(tok) > 1
    ]


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase (for dedup and phrase matching)."""
    return _WS_RE.sub(" ", text.strip().lower())


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1.0, 1.0] (0.0 when either vector is zero)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def _coerce_messages(source: RetrievalSource) -> List[Message]:
    """Extract a message list from a Conversation, ContextStore, or list."""
    if isinstance(source, Conversation):
        return source.get_messages()
    # ContextStore duck-typed (avoids a hard import cycle): any object
    # exposing a Conversation under `.conversation` qualifies.
    conv = getattr(source, "conversation", None)
    if isinstance(conv, Conversation):
        return conv.get_messages()
    if isinstance(source, (list, tuple)):
        items = list(source)
        for item in items:
            if not isinstance(item, Message):
                raise RetrieverError(
                    "message list must contain only Message instances, "
                    f"got {type(item).__name__!r}."
                )
        return items
    raise TypeError(
        "source must be a Conversation, ContextStore, or list of Messages, "
        f"got {type(source).__name__!r}."
    )


def _finalize(
    scored: List[tuple[float, int, Message, List[str]]],
    limit: int,
) -> List[RetrievalResult]:
    """Sort by score (ties keep conversation order), dedupe, rank, slice."""
    scored.sort(key=lambda item: (-item[0], item[1]))
    seen: set[str] = set()
    results: List[RetrievalResult] = []
    for score, _, msg, matched in scored:
        key = _normalize(msg.content)
        if key in seen:
            continue
        seen.add(key)
        results.append(RetrievalResult(
            score=score,
            message=msg,
            matched_terms=matched,
            rank=len(results) + 1,
        ))
        if len(results) >= limit:
            break
    return results


class KeywordRetriever(Retriever):
    """Deterministic keyword-based retriever (no embeddings, stdlib only).

    Args:
        top_k:     Default maximum results per call (positive int).
        min_score: Minimum relevance score for a hit to be included
                   (0.0-1.0; messages scoring at or below it are dropped).

    Usage::

        retriever = KeywordRetriever(top_k=3)
        hits = retriever.retrieve("postgresql storage decision", conv)
        hits = conv.search("postgresql storage decision", top_k=3)  # same
    """

    def __init__(self, top_k: int = 5, min_score: float = 0.0) -> None:
        self.top_k: int = self._validate_top_k(top_k, "top_k")
        self.min_score: float = self._validate_min_score(min_score)

    # ── validation helpers ───────────────────────────────────────

    @staticmethod
    def _validate_top_k(value: int, label: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"{label} must be a positive integer, got {type(value).__name__!r}."
            )
        if value <= 0:
            raise ValueError(f"{label} must be a positive integer, got {value}.")
        return value

    @staticmethod
    def _validate_min_score(value: float) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"min_score must be a number between 0 and 1, got {type(value).__name__!r}."
            )
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"min_score must be between 0 and 1, got {value}.")
        return float(value)

    # ── public API ───────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        source: RetrievalSource,
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """Rank messages in *source* against *query* (best first).

        Args:
            query:  User query string (case-insensitive; empty or
                    stopword-only queries return []).
            source: A Conversation, ContextStore, or list of Messages.
                    Compaction summaries are included automatically --
                    they are ordinary messages in the conversation.
            top_k:  Override for this call (positive int). Defaults to
                    the constructor value.

        Returns:
            Ranked RetrievalResult list (deduplicated by content, at most
            *top_k* entries). Empty list when nothing matches.

        Raises:
            TypeError:      If *query* is not a string or *source* is not a
                            Conversation / ContextStore / message list.
            RetrieverError: If a message list holds non-Message items.
        """
        if not isinstance(query, str):
            raise TypeError(
                f"query must be a string, got {type(query).__name__!r}."
            )
        limit = self.top_k if top_k is None else self._validate_top_k(top_k, "top_k")
        messages = self._coerce_messages(source)

        query_terms = _tokenize(query)
        if not query_terms or not messages:
            return []

        distinct_query = set(query_terms)
        norm_query = _normalize(query)

        scored: List[tuple[float, int, Message, List[str]]] = []
        for index, msg in enumerate(messages):
            msg_terms = _tokenize(msg.content)
            if not msg_terms:
                continue
            counts = Counter(msg_terms)
            matched = sorted(t for t in distinct_query if t in counts)
            if not matched:
                continue
            coverage = len(matched) / len(distinct_query)
            hits = sum(counts[t] for t in matched)
            frequency = min(1.0, hits / _FREQ_CAP)
            bonus = _PHRASE_BONUS if len(norm_query) >= 8 and norm_query in _normalize(msg.content) else 0.0
            score = round(min(1.0, 0.7 * coverage + 0.3 * frequency + bonus), 3)
            if score > self.min_score:
                scored.append((score, index, msg, matched))

        # Best score first; ties keep original conversation order (stable,
        # deterministic -- no randomness anywhere). Duplicates collapsed.
        return _finalize(scored, limit)

    # ── source handling ──────────────────────────────────────────

    @staticmethod
    def _coerce_messages(source: RetrievalSource) -> List[Message]:
        return _coerce_messages(source)


class SemanticRetriever(Retriever):
    """Embedding-based retriever: ranks by semantic (cosine) similarity.

    Message and query texts are embedded via an EmbeddingProvider and scored
    by cosine similarity mapped to 0.0-1.0::

        score = round((cosine + 1.0) / 2.0, 3)

    Unchanged messages are embedded only once: vectors are cached in memory
    keyed by content hash, and a persisted ContextStore's embeddings can be
    loaded into the cache via prime_cache(). On provider failure the
    retriever gracefully falls back to keyword retrieval (configurable).

    Args:
        provider:  EmbeddingProvider used for queries and messages.
        top_k:     Default maximum results per call (positive int).
        min_score: Minimum score for a hit (0.0-1.0; at-or-below dropped).
        fallback:  Retriever used when the provider raises EmbeddingError.
                   Defaults to a KeywordRetriever (graceful degradation).
                   Pass False to disable the fallback and propagate
                   failures as RetrieverError instead.

    Usage::

        retriever = SemanticRetriever(OpenAIEmbeddingProvider())
        hits = retriever.retrieve("how do we store data?", conv)
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        top_k: int = 5,
        min_score: float = 0.0,
        fallback: Union[Retriever, bool, None] = None,
    ) -> None:
        if not isinstance(provider, EmbeddingProvider):
            raise TypeError(
                "provider must be an EmbeddingProvider instance, "
                f"got {type(provider).__name__!r}."
            )
        self.provider: EmbeddingProvider = provider
        self.top_k: int = KeywordRetriever._validate_top_k(top_k, "top_k")
        self.min_score: float = KeywordRetriever._validate_min_score(min_score)
        if fallback is None:
            # Default: degrade gracefully to keyword retrieval on API failure.
            # Pass fallback=False to propagate failures as RetrieverError.
            fallback = KeywordRetriever(top_k=top_k, min_score=min_score)
        elif fallback is False:
            fallback = None
        if fallback is not None and not isinstance(fallback, Retriever):
            raise TypeError(
                "fallback must be a Retriever instance, False, or None, "
                f"got {type(fallback).__name__!r}."
            )
        self.fallback: Optional[Retriever] = fallback
        self._cache: Dict[str, List[float]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    # ── cache ────────────────────────────────────────────────────

    def cache_info(self) -> Dict[str, int]:
        """Return cache statistics (hits, misses, size)."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Empty the embedding cache and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def prime_cache(self, source: Union["ContextStore", Dict[str, List[float]]]) -> int:
        """Load persisted embeddings into the cache.

        Accepts a ContextStore (reads its ``embeddings`` map) or a plain
        ``{content_hash: vector}`` dict. Returns the number of vectors added.
        Entries that are missing or malformed are skipped (they will simply
        be embedded on demand).
        """
        if hasattr(source, "embeddings"):
            mapping = getattr(source, "embeddings")
        else:
            mapping = source
        if not isinstance(mapping, dict):
            raise TypeError(
                "prime_cache expects a ContextStore or {hash: vector} dict, "
                f"got {type(source).__name__!r}."
            )
        added = 0
        for key, vector in mapping.items():
            if not isinstance(key, str) or key in self._cache:
                continue
            try:
                clean = validate_vector(vector)
            except EmbeddingError:
                continue
            self._cache[key] = clean
            added += 1
        return added

    def _vectors_for(self, texts: List[str]) -> List[List[float]]:
        """Return vectors for *texts*, embedding only cache misses in one call."""
        vectors: List[Optional[List[float]]] = []
        missing: List[str] = []
        missing_keys: List[str] = []
        for text in texts:
            key = _content_key(text)
            hit = self._cache.get(key)
            if hit is not None:
                self._cache_hits += 1
                vectors.append(hit)
            else:
                self._cache_misses += 1
                vectors.append(None)
                missing.append(text)
                missing_keys.append(key)
        if missing:
            fresh = self.provider.embed(missing)
            if len(fresh) != len(missing):
                raise EmbeddingError(
                    f"Provider returned {len(fresh)} vector(s) "
                    f"for {len(missing)} text(s)."
                )
            try:
                clean_fresh = [validate_vector(v) for v in fresh]
            except EmbeddingError as exc:
                raise EmbeddingError(f"Provider returned a malformed vector ({exc}).") from exc
            for key, vector in zip(missing_keys, clean_fresh):
                self._cache[key] = vector
            cursor = iter(clean_fresh)
            vectors = [v if v is not None else next(cursor) for v in vectors]
        return [v for v in vectors if v is not None]

    # ── public API ───────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        source: RetrievalSource,
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """Rank messages in *source* by semantic similarity (best first).

        On EmbeddingError the configured fallback retriever runs instead
        (default: keyword retrieval); with ``fallback=False`` the failure is
        re-raised as RetrieverError.
        """
        if not isinstance(query, str):
            raise TypeError(
                f"query must be a string, got {type(query).__name__!r}."
            )
        limit = self.top_k if top_k is None else KeywordRetriever._validate_top_k(top_k, "top_k")
        messages = _coerce_messages(source)
        if not query.strip() or not messages:
            return []

        try:
            query_vector = self._vectors_for([query])[0]
            msg_vectors = self._vectors_for([m.content for m in messages])
            if any(len(v) != len(query_vector) for v in msg_vectors):
                raise EmbeddingError("Inconsistent embedding dimensions.")
        except EmbeddingError:
            if self.fallback is not None:
                return self.fallback.retrieve(query, messages, top_k=limit)
            raise RetrieverError(
                "Semantic retrieval failed and no fallback is configured."
            )

        scored: List[tuple[float, int, Message, List[str]]] = []
        for index, (msg, vector) in enumerate(zip(messages, msg_vectors)):
            score = round((_cosine_similarity(query_vector, vector) + 1.0) / 2.0, 3)
            if score > self.min_score:
                scored.append((score, index, msg, []))
        return _finalize(scored, limit)


# ── vector-store message record helpers ──────────────────────────


def _message_id(message: Message) -> str:
    """Stable record ID for a message (content hash)."""
    return _content_key(message.content)


def _message_to_record_meta(message: Message) -> Dict[str, object]:
    """Flatten a Message into vector-store-safe metadata."""
    return {
        "role": message.role,
        "importance": message.importance.value,
        "protected": message.protected,
        "token_count": message.token_count,
        "extra": json.dumps(message.metadata, ensure_ascii=False),
    }


def _message_from_record(content: str, metadata: Dict[str, object]) -> Optional[Message]:
    """Rebuild a Message from stored content + metadata (None if corrupt)."""
    try:
        role = str(metadata.get("role", "user"))
        importance = str(metadata.get("importance", "normal"))
        protected = bool(metadata.get("protected", False))
        raw_extra = metadata.get("extra", "{}")
        extra = json.loads(raw_extra) if isinstance(raw_extra, str) else {}
        return Message(
            role=role,
            content=content,
            metadata=extra if isinstance(extra, dict) else {},
            importance=importance,
            protected=protected,
        )
    except (ValueError, TypeError):
        return None


class VectorStoreRetriever(SemanticRetriever):
    """Vector-store-backed retriever: queries an index, not a scan.

    Messages are embedded once via index() (or lazily on the first
    retrieve()) and stored with their IDs, content, metadata, and vectors.
    Each query then embeds only the query text and asks the store for its
    nearest neighbours -- Python never scores every message.

    Fallback chain (graceful degradation):
        vector store  -->  in-memory semantic  -->  keyword  -->  [].

    A store failure falls back to in-memory cosine scoring; a provider
    failure falls back further to the configured keyword retriever
    (pass ``fallback=False`` to raise RetrieverError instead).

    Args:
        provider:     EmbeddingProvider used for queries and indexing.
        vector_store: VectorStore backend holding the index.
        top_k:        Default maximum results per call (positive int).
        min_score:    Minimum score for a hit (0.0-1.0; at-or-below dropped).
        fallback:     Retriever used when embeddings fail (default:
                      KeywordRetriever). Pass False to disable.

    Usage::

        store = ChromaVectorStore()
        retriever = VectorStoreRetriever(provider, store)
        retriever.index(conv)
        hits = retriever.retrieve("how do we store data?", conv)
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int = 5,
        min_score: float = 0.0,
        fallback: Union[Retriever, bool, None] = None,
    ) -> None:
        super().__init__(provider, top_k=top_k, min_score=min_score, fallback=fallback)
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                "vector_store must be a VectorStore instance, "
                f"got {type(vector_store).__name__!r}."
            )
        self.vector_store: VectorStore = vector_store
        self._indexed_ids: set[str] = set()

    # ── indexing ─────────────────────────────────────────────────

    def index(self, source: RetrievalSource) -> int:
        """Embed and upsert every message from *source* (idempotent).

        Returns the number of messages passed (all are guaranteed indexed
        afterwards). Already-indexed messages are skipped without
        re-embedding; vectors come from the cache or the provider.
        """
        messages = _coerce_messages(source)
        self._ensure_indexed(messages)
        return len(messages)

    @property
    def indexed_count(self) -> int:
        """Number of message IDs this retriever has indexed."""
        return len(self._indexed_ids)

    def remove(self, ids: Sequence[str]) -> int:
        """Delete records from the vector store by ID.

        IDs are content hashes (see index()); unknown IDs are ignored by
        the backend. Forgets the IDs locally so re-added content is
        embedded fresh. Raises VectorStoreError when deletion fails.
        """
        clean = VectorStore._validate_ids(ids)
        removed = self.vector_store.delete(clean)
        self._indexed_ids.difference_update(clean)
        return removed

    def _ensure_indexed(self, messages: List[Message]) -> None:
        """Embed + upsert any message IDs missing from the index."""
        fresh = [(m, _message_id(m)) for m in messages]
        unknown = [(m, i) for m, i in fresh if i not in self._indexed_ids]
        if not unknown:
            return
        # Ask the store first so a fresh retriever over a persistent,
        # pre-indexed store embeds nothing (VectorStoreError propagates
        # to the fallback chain in retrieve()).
        present = self.vector_store.exists([i for _, i in unknown])
        missing = [(m, i) for (m, i), found in zip(unknown, present) if not found]
        if missing:
            vectors = self._vectors_for([m.content for m, _ in missing])
            self.vector_store.upsert(
                ids=[i for _, i in missing],
                contents=[m.content for m, _ in missing],
                metadatas=[_message_to_record_meta(m) for m, _ in missing],
                embeddings=vectors,
            )
        self._indexed_ids.update(i for _, i in unknown)

    # ── public API ───────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        source: Optional[RetrievalSource] = None,
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """Query the vector store for *query* (best first).

        Args:
            query:  User query string (empty queries return []).
            source: Optional Conversation / ContextStore / message list.
                    New messages are indexed on the fly; pass None to
                    query a pre-indexed store directly.
            top_k:  Override for this call (positive int).

        Falls back to in-memory semantic scoring when the store fails,
        and to keyword retrieval when embeddings fail (unless disabled).
        """
        if not isinstance(query, str):
            raise TypeError(
                f"query must be a string, got {type(query).__name__!r}."
            )
        limit = self.top_k if top_k is None else KeywordRetriever._validate_top_k(top_k, "top_k")
        messages = _coerce_messages(source) if source is not None else []
        if not query.strip():
            return []
        if source is not None and not messages:
            return []

        try:
            self._ensure_indexed(messages)
            query_vector = self._vectors_for([query])[0]
            hits = self.vector_store.query([query_vector], top_k=limit)
        except (VectorStoreError, EmbeddingError):
            if source is None:
                raise RetrieverError(
                    "Vector-store query failed and no source is available "
                    "for fallback retrieval."
                )
            return super().retrieve(query, messages, top_k=limit)

        scored: List[tuple[float, int, Message, List[str]]] = []
        for position, hit in enumerate(hits):
            message = _message_from_record(hit.content, hit.metadata)
            if message is None:
                continue
            if hit.score > self.min_score:
                scored.append((hit.score, position, message, []))
        return _finalize(scored, limit)
