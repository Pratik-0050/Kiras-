# contextflow/conversation.py
"""
Defines the Conversation class.

A Conversation is an ordered list of Message objects with a token budget
and context-pressure detection. It reports usage and status but never
automatically removes or compresses messages -- that belongs to later steps.
"""

from typing import List, Optional, TYPE_CHECKING
from .message import Message
from .status import ContextStatus

if TYPE_CHECKING:
    from .embeddings import EmbeddingProvider
    from .retriever import RetrievalResult
    from .vectorstores.base import VectorStore


# Default thresholds (as percentages of max_tokens).
DEFAULT_WARN_AT    = 70.0   # above this -> WARNING
DEFAULT_COMPACT_AT = 90.0   # above this -> COMPACTION_NEEDED


class Conversation:
    """An ordered collection of Message objects with token-budget and
    context-pressure tracking.

    Args:
        name:        Human-readable label for this conversation.
        max_tokens:  Token budget (positive int). Pass None for no limit.
        warn_at:     Percentage threshold for WARNING status   (default 70.0).
        compact_at:  Percentage threshold for COMPACTION_NEEDED (default 90.0).

    Pressure status:
        OK                -- usage < warn_at %
        WARNING           -- warn_at % <= usage < compact_at %
        COMPACTION_NEEDED -- usage >= compact_at %
        (Status is always OK when no max_tokens is set.)

    Methods (budget):
        total_tokens()      -- Sum of token_count across all messages.
        remaining_tokens()  -- Tokens left before the limit (None if no limit).
        usage_percentage()  -- Percentage of budget used (None if no limit).
        is_over_limit()     -- True if total_tokens() > max_tokens.

    Methods (pressure):
        get_status()         -- Return a ContextStatus enum value.
        get_status_message() -- Return a human-readable status string.

    Methods (management):
        add(message)       -- Append a Message; raises TypeError for non-Message.
        remove(index)      -- Remove and return the Message at *index* (0-based).
        clear()            -- Remove all messages.
        get_messages()     -- Return a safe copy of the message list.
        message_count()    -- Number of messages currently stored.

    Methods (retrieval, Step 13 -- additive only, no behavior change):
        search(query)      -- Rank messages by keyword relevance (best first).
    """

    def __init__(
        self,
        name: str = "Conversation",
        max_tokens: Optional[int] = None,
        warn_at: float = DEFAULT_WARN_AT,
        compact_at: float = DEFAULT_COMPACT_AT,
    ) -> None:
        self.name: str = name

        # --- validate max_tokens ---
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
                raise TypeError(
                    f"max_tokens must be a positive integer, "
                    f"got {type(max_tokens).__name__!r}."
                )
            if max_tokens <= 0:
                raise ValueError(
                    f"max_tokens must be a positive integer, got {max_tokens}."
                )

        # --- validate thresholds ---
        for label, value in (("warn_at", warn_at), ("compact_at", compact_at)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"{label} must be a number between 0 and 100, "
                    f"got {type(value).__name__!r}."
                )
            if not (0.0 < value <= 100.0):
                raise ValueError(
                    f"{label} must be between 0 and 100, got {value}."
                )
        if warn_at >= compact_at:
            raise ValueError(
                f"warn_at ({warn_at}) must be less than compact_at ({compact_at})."
            )

        self.max_tokens: Optional[int]  = max_tokens
        self.warn_at: float             = float(warn_at)
        self.compact_at: float          = float(compact_at)
        self._messages: List[Message]   = []

    # ------------------------------------------------------------------
    # Core mutation methods
    # ------------------------------------------------------------------

    def add(self, message: Message) -> None:
        """Append *message* to the conversation.

        Raises:
            TypeError: If *message* is not a Message instance.
        """
        if not isinstance(message, Message):
            raise TypeError(
                f"Expected a Message instance, got {type(message).__name__!r}. "
                "Create your turns with Message(role=..., content=...)."
            )
        self._messages.append(message)

    def remove(self, index: int) -> Message:
        """Remove and return the Message at *index* (0-based).

        Raises:
            IndexError: If *index* is out of range.
        """
        if index < 0 or index >= len(self._messages):
            raise IndexError(
                f"Index {index} is out of range. "
                f"Conversation has {len(self._messages)} message(s)."
            )
        return self._messages.pop(index)

    def clear(self) -> None:
        """Remove all messages from the conversation."""
        self._messages.clear()

    # ------------------------------------------------------------------
    # Read-only query methods
    # ------------------------------------------------------------------

    def get_messages(self) -> List[Message]:
        """Return a shallow copy of the message list (safe to iterate)."""
        return list(self._messages)

    def message_count(self) -> int:
        """Return the number of messages currently in the conversation."""
        return len(self._messages)

    # ------------------------------------------------------------------
    # Retrieval (Step 13 -- convenience wrapper, read-only)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        provider: Optional["EmbeddingProvider"] = None,
        vector_store: Optional["VectorStore"] = None,
    ) -> List["RetrievalResult"]:
        """Return the most relevant messages for *query* (best first).

        Keyword-based and deterministic by default; pass an
        EmbeddingProvider for semantic (cosine-similarity) ranking with
        graceful keyword fallback on API failure; pass both a provider
        and a VectorStore to query a persistent index instead of
        scanning (with the same fallbacks). The conversation is
        never modified.

        Args:
            query:        User query string (case-insensitive for keywords).
            top_k:        Max results (positive int). Defaults to the
                          retriever default (5) when None.
            provider:     Optional EmbeddingProvider enabling semantic search.
            vector_store: Optional VectorStore enabling indexed search
                          (requires *provider*).

        Returns:
            Ranked RetrievalResult list (possibly empty).
        """
        if vector_store is not None and provider is None:
            raise TypeError(
                "vector_store search requires an EmbeddingProvider; "
                "pass provider=... alongside vector_store=..."
            )
        if vector_store is not None:
            from .retriever import VectorStoreRetriever  # lazy: avoids import cycle
            return VectorStoreRetriever(provider, vector_store).retrieve(
                query, self, top_k=top_k
            )
        if provider is None:
            from .retriever import KeywordRetriever  # lazy: avoids import cycle
            return KeywordRetriever().retrieve(query, self, top_k=top_k)
        from .retriever import SemanticRetriever  # lazy: avoids import cycle
        return SemanticRetriever(provider).retrieve(query, self, top_k=top_k)

    # ------------------------------------------------------------------
    # Budget methods
    # ------------------------------------------------------------------

    def total_tokens(self) -> int:
        """Return the sum of token_count for every message."""
        return sum(m.token_count for m in self._messages)

    def remaining_tokens(self) -> Optional[int]:
        """Return tokens left before the budget is exhausted.

        Returns:
            Tokens remaining (can be negative if over limit).
            None if no max_tokens was set.
        """
        if self.max_tokens is None:
            return None
        return self.max_tokens - self.total_tokens()

    def usage_percentage(self) -> Optional[float]:
        """Return token usage as a percentage of the budget (0.0 - 100.0+).

        Returns:
            Float percentage, e.g. 65.4 means 65.4% used.
            None if no max_tokens was set.
        """
        if self.max_tokens is None:
            return None
        return round((self.total_tokens() / self.max_tokens) * 100, 1)

    def is_over_limit(self) -> bool:
        """Return True if total_tokens() exceeds max_tokens."""
        if self.max_tokens is None:
            return False
        return self.total_tokens() > self.max_tokens

    # ------------------------------------------------------------------
    # Pressure / status methods
    # ------------------------------------------------------------------

    def get_status(self) -> ContextStatus:
        """Return the current context-pressure status.

        Returns:
            ContextStatus.OK                if no limit or usage < warn_at
            ContextStatus.WARNING           if warn_at  <= usage% < compact_at
            ContextStatus.COMPACTION_NEEDED if usage% >= compact_at
        """
        pct = self.usage_percentage()
        if pct is None:
            return ContextStatus.OK
        if pct >= self.compact_at:
            return ContextStatus.COMPACTION_NEEDED
        if pct >= self.warn_at:
            return ContextStatus.WARNING
        return ContextStatus.OK

    def get_status_message(self) -> str:
        """Return a human-readable explanation of the current status.

        Examples:
            "OK -- 1,200 / 8,000 tokens used (15.0%). Plenty of space remaining."
            "WARNING -- 6,100 / 8,000 tokens used (76.3%). Consider compaction soon."
            "COMPACTION_NEEDED -- 7,500 / 8,000 tokens used (93.8%). Act now."
        """
        status = self.get_status()
        total  = self.total_tokens()

        if self.max_tokens is None:
            return f"OK -- {total:,} tokens used (no limit set)."

        pct  = self.usage_percentage()
        limit = self.max_tokens

        base = f"{status} -- {total:,} / {limit:,} tokens used ({pct}%)."

        if status == ContextStatus.OK:
            return base + " Plenty of space remaining."
        if status == ContextStatus.WARNING:
            return base + " Approaching the limit -- consider compaction soon."
        # COMPACTION_NEEDED
        if self.is_over_limit():
            over = abs(self.remaining_tokens())
            return base + f" OVER LIMIT by {over:,} tokens -- compaction required."
        return base + " Very close to the limit -- compaction required."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _budget_line(self) -> str:
        """Short formatted budget string for __str__."""
        total = self.total_tokens()
        if self.max_tokens is None:
            return f"{total:,} tokens  (no limit set)"
        pct   = self.usage_percentage()
        limit = self.max_tokens
        over  = "  [OVER LIMIT]" if self.is_over_limit() else ""
        return f"{total:,} / {limit:,} tokens ({pct}%){over}"

    def _status_icon(self) -> str:
        icons = {
            ContextStatus.OK:                "[OK]",
            ContextStatus.WARNING:           "[WARNING]",
            ContextStatus.COMPACTION_NEEDED: "[COMPACTION NEEDED]",
        }
        return icons[self.get_status()]

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        limit_part = (
            f"max_tokens={self.max_tokens}"
            if self.max_tokens is not None
            else "no_limit"
        )
        return (
            f"Conversation(name={self.name!r}, "
            f"messages={self.message_count()}, "
            f"total_tokens={self.total_tokens()}, "
            f"{limit_part}, "
            f"status={self.get_status()})"
        )

    def __str__(self) -> str:
        sep = "=" * 58
        lines = [sep, f"  {self.name}", sep]
        for msg in self._messages:
            lines.append(str(msg))
        lines.append("")
        lines.append(f"  Messages : {self.message_count()}")
        lines.append(f"  Usage    : {self._budget_line()}")
        if self.max_tokens is not None:
            lines.append(f"  Status   : {self._status_icon()}")
            lines.append(f"  Detail   : {self.get_status_message()}")
            if self.remaining_tokens() is not None and not self.is_over_limit():
                lines.append(f"  Remaining: {self.remaining_tokens():,} tokens")
        lines.append(sep)
        return "\n".join(lines)
