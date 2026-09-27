# contextflow/manager.py
"""Simple facade over ContextFlow's engine (Step: professional library).

ContextManager wraps a Conversation with retrieval, budget-aware assembly,
and threshold-gated compaction behind five methods: add_message(),
get_context(), compact(), search(), and save()/load(). Everything else in
the package remains available for advanced use; nothing here duplicates
engine logic -- it only composes Conversation, Retriever,
ContextAssembler, and ContextPipeline.
"""

from __future__ import annotations

from typing import List, Optional, Union

from .conversation import Conversation
from .compactor import Compactor
from .hybrid import AssembledContext, ContextAssembler
from .message import ImportanceLevel, Message
from .pipeline import ContextPipeline, PipelineResult
from .retriever import KeywordRetriever, RetrievalResult, Retriever
from .store import ContextStore
from .summarizers.placeholder import PlaceholderSummarizer
from .validators.heuristic_validator import HeuristicValidator


class ContextManager:
    """One-object entry point for everyday ContextFlow use.

    Args:
        max_tokens: Assembly budget and conversation limit
                    (positive int, default 8000).
        name:       Conversation label (default "Context").
        system:     Optional system instructions, stored as the leading
                    system message.
        retriever:  Retriever for get_context() (defaults to KeywordRetriever()).
        assembler:  ContextAssembler override (built per call from
                     max_tokens when not given).
        pipeline:   ContextPipeline override for compact() (built with an
                     offline-safe validator when not given).
        keep_recent: Recent messages kept by the per-call assembler
                     (non-negative int, default 5).
        retrieval_top_k: Hits fetched per get_context() call (default 5).

    Usage::

        cf = ContextManager()
        cf.add_message("user", "Hello")
        cf.add_message("assistant", "Hi! How can I help?")
        print(cf.get_context())
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        name: str = "Context",
        system: Optional[str] = None,
        retriever: Optional[Retriever] = None,
        assembler: Optional[ContextAssembler] = None,
        pipeline: Optional[ContextPipeline] = None,
        keep_recent: int = 5,
        retrieval_top_k: int = 5,
    ) -> None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise TypeError(
                f"max_tokens must be a positive integer, got {type(max_tokens).__name__!r}."
            )
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}.")
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__!r}.")
        if system is not None and not isinstance(system, str):
            raise TypeError(
                f"system must be a string or None, got {type(system).__name__!r}."
            )
        if retriever is not None and not isinstance(retriever, Retriever):
            raise TypeError(
                "retriever must be a Retriever instance or None, "
                f"got {type(retriever).__name__!r}."
            )
        if assembler is not None and not isinstance(assembler, ContextAssembler):
            raise TypeError(
                "assembler must be a ContextAssembler instance or None, "
                f"got {type(assembler).__name__!r}."
            )
        if pipeline is not None and not isinstance(pipeline, ContextPipeline):
            raise TypeError(
                "pipeline must be a ContextPipeline instance or None, "
                f"got {type(pipeline).__name__!r}."
            )
        if not isinstance(keep_recent, int) or isinstance(keep_recent, bool):
            raise TypeError(
                "keep_recent must be a non-negative integer, "
                f"got {type(keep_recent).__name__!r}."
            )
        if keep_recent < 0:
            raise ValueError(
                f"keep_recent must be a non-negative integer, got {keep_recent}."
            )
        if not isinstance(retrieval_top_k, int) or isinstance(retrieval_top_k, bool):
            raise TypeError(
                "retrieval_top_k must be a positive integer, "
                f"got {type(retrieval_top_k).__name__!r}."
            )
        if retrieval_top_k <= 0:
            raise ValueError(
                f"retrieval_top_k must be a positive integer, got {retrieval_top_k}."
            )

        self.max_tokens: int = max_tokens
        self.conversation: Conversation = Conversation(name=name, max_tokens=max_tokens)
        if system is not None:
            self.conversation.add(Message(role="system", content=system))
        self.retriever: Retriever = (
            retriever if retriever is not None else KeywordRetriever()
        )
        self.assembler: Optional[ContextAssembler] = assembler
        self.pipeline: ContextPipeline = pipeline if pipeline is not None else ContextPipeline(
            compactor=Compactor(
                summarizer=PlaceholderSummarizer(),
                validator=HeuristicValidator(),
            )
        )
        self.keep_recent: int = keep_recent
        self.retrieval_top_k: int = retrieval_top_k

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        importance: Union[ImportanceLevel, str] = ImportanceLevel.NORMAL,
        protected: bool = False,
        metadata: Optional[dict] = None,
    ) -> Message:
        """Append a turn and return the stored Message (validated)."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata,
            importance=importance,
            protected=protected,
        )
        self.conversation.add(message)
        return message

    @property
    def history(self) -> List[Message]:
        """Conversation history (safe copy)."""
        return self.conversation.get_messages()

    @property
    def message_count(self) -> int:
        """Number of stored messages."""
        return self.conversation.message_count()

    @property
    def tokens(self) -> int:
        """Total tokens currently stored."""
        return self.conversation.total_tokens()

    def clear(self) -> None:
        """Remove all messages from history."""
        self.conversation.clear()

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def _assembler(self) -> ContextAssembler:
        return self.assembler or ContextAssembler(
            max_tokens=self.max_tokens, keep_recent=self.keep_recent
        )

    def _focal_request(self, query: Optional[str]) -> Union[str, Message]:
        """Resolve what the assembled context is built around.

        An explicit query becomes a new (unrecorded) request; otherwise the
        latest user message object is reused so it appears exactly once;
        with no history at all, an empty request yields system-only output.
        """
        if query is not None:
            if not isinstance(query, str):
                raise TypeError(
                    f"query must be a string or None, got {type(query).__name__!r}."
                )
            return query
        for message in reversed(self.conversation.get_messages()):
            if message.role == "user":
                return message
        return ""

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Return ranked retrieval hits for *query* (best first)."""
        return self.retriever.retrieve(
            query, self.conversation,
            top_k=self.retrieval_top_k if top_k is None else top_k,
        )

    def get_context(
        self,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> AssembledContext:
        """Assemble prompt-ready context within the token budget.

        Retrieves relevant history for *query* (or reuses the latest user
        message when omitted), then packs system, protected, retrieved,
        recent, and request messages. Never mutates history.
        """
        focal = self._focal_request(query)
        text = focal if isinstance(focal, str) else focal.content
        hits = self.retriever.retrieve(
            text, self.conversation,
            top_k=self.retrieval_top_k if top_k is None else top_k,
        ) if text.strip() else []
        return self._assembler().assemble(focal, self.conversation, hits)

    def compact(self) -> PipelineResult:
        """Compact when pressure requires it (skips healthy chats)."""
        return self.pipeline.run(self.conversation)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> str:
        """Persist history to a JSON file; returns the path."""
        ContextStore(self.conversation).save(path)
        return path

    @classmethod
    def load(cls, path: str, **kwargs) -> "ContextManager":
        """Load history from a JSON file into a new manager.

        Extra keyword arguments (max_tokens, system, retriever, ...) are
        forwarded to the constructor; the stored conversation replaces the
        fresh one (stored max_tokens win when max_tokens is not given).
        """
        store = ContextStore.load_file(path)
        stored_max = store.conversation.max_tokens
        if "max_tokens" not in kwargs and stored_max is not None:
            kwargs["max_tokens"] = stored_max
        manager = cls(**kwargs)
        manager.conversation = store.conversation
        return manager

    def __repr__(self) -> str:
        return (
            f"ContextManager(name={self.conversation.name!r}, "
            f"messages={self.message_count}, "
            f"tokens={self.tokens}/{self.max_tokens}, "
            f"status={self.conversation.get_status()})"
        )
