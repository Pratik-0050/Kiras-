# contextflow/hybrid.py
"""
Step 16: Hybrid retrieval and context assembly.

HybridRetriever fuses keyword retrieval (exact terms, deterministic) with
semantic / vector retrieval (meaning, embeddings) into one ranked list.
Each side's 0.0-1.0 scores are min-max normalized across its own hits so
that different scoring scales stay comparable, then combined with
configurable weights::

    fused = keyword_weight * norm_keyword + semantic_weight * norm_semantic

A message seen by only one side scores 0.0 on the other side. Duplicate
content is returned once (best fused score wins).

ContextAssembler packs the final prompt-ready context: leading system
prompt, protected messages, retrieved hits, recent messages, and the
current user request -- in that order -- while respecting a token budget.
Retrieved candidates fill the budget best-first; recent messages fill
newest-first; system, protected, and request messages are mandatory and
are never excluded (over-budget is reported honestly instead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .conversation import Conversation
from .message import Message
from .retriever import (
    _finalize,
    _normalize,
    KeywordRetriever,
    RetrievalResult,
    Retriever,
    RetrievalSource,
)


# ── hybrid retrieval ─────────────────────────────────────────────


def _min_max_normalize(scores: List[float]) -> List[float]:
    """Scale *scores* to [0.0, 1.0] (uniform 1.0 when all equal)."""
    if not scores:
        return []
    top, bottom = max(scores), min(scores)
    if top == bottom:
        return [1.0 for _ in scores]
    return [(s - bottom) / (top - bottom) for s in scores]


class HybridRetriever(Retriever):
    """Fuses keyword and semantic/vector retrieval with weighted ranking.

    Args:
        keyword:  Keyword-side retriever (defaults to KeywordRetriever()).
        semantic: Semantic-side retriever (SemanticRetriever,
                  VectorStoreRetriever, or any Retriever). None disables
                  the semantic side (keyword-only hybrid). For
                  semantic-only ranking, keep both sides and pass
                  keyword_weight=0.0.
        keyword_weight:  Weight for normalized keyword scores (0.0-1.0).
        semantic_weight: Weight for normalized semantic scores (0.0-1.0).
                         At least one weight must be positive.
        top_k:     Default maximum results per call (positive int).
        min_score: Minimum fused score for a hit (0.0-1.0; at-or-below
                   dropped).
        per_source_k: Hits fetched per side before fusion (positive int).
                      Defaults to top_k. Raise for broader recall.

    Usage::

        hybrid = HybridRetriever(
            semantic=SemanticRetriever(provider),
            keyword_weight=0.4,
            semantic_weight=0.6,
        )
        hits = hybrid.retrieve("where do we persist records?", conv)
    """

    def __init__(
        self,
        keyword: Optional[Retriever] = None,
        semantic: Optional[Retriever] = None,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
        top_k: int = 5,
        min_score: float = 0.0,
        per_source_k: Optional[int] = None,
    ) -> None:
        if keyword is None:
            keyword = KeywordRetriever()
        for name, retriever in (("keyword", keyword), ("semantic", semantic)):
            if retriever is not None and not isinstance(retriever, Retriever):
                raise TypeError(
                    f"{name} must be a Retriever instance or None, "
                    f"got {type(retriever).__name__!r}."
                )
        self.keyword: Optional[Retriever] = keyword
        self.semantic: Optional[Retriever] = semantic
        self.keyword_weight: float = self._validate_weight(keyword_weight, "keyword_weight")
        self.semantic_weight: float = self._validate_weight(semantic_weight, "semantic_weight")
        if self.keyword_weight + self.semantic_weight <= 0.0:
            raise ValueError("At least one of keyword_weight / semantic_weight must be positive.")
        self.top_k: int = KeywordRetriever._validate_top_k(top_k, "top_k")
        self.min_score: float = KeywordRetriever._validate_min_score(min_score)
        self.per_source_k: int = (
            self.top_k if per_source_k is None
            else KeywordRetriever._validate_top_k(per_source_k, "per_source_k")
        )

    @staticmethod
    def _validate_weight(value: float, label: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"{label} must be a number between 0 and 1, "
                f"got {type(value).__name__!r}."
            )
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{label} must be between 0 and 1, got {value}.")
        return float(value)

    def retrieve(
        self,
        query: str,
        source: RetrievalSource,
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """Fuse both sides' rankings for *query* (best fused score first).

        Raises:
            TypeError: If *query* is not a string.
        """
        if not isinstance(query, str):
            raise TypeError(
                f"query must be a string, got {type(query).__name__!r}."
            )
        limit = self.top_k if top_k is None else KeywordRetriever._validate_top_k(top_k, "top_k")

        keyword_hits = (
            self.keyword.retrieve(query, source, top_k=self.per_source_k)
            if self.keyword is not None else []
        )
        semantic_hits = (
            self.semantic.retrieve(query, source, top_k=self.per_source_k)
            if self.semantic is not None else []
        )
        if not keyword_hits and not semantic_hits:
            return []

        keyword_norm = _min_max_normalize([h.score for h in keyword_hits])
        semantic_norm = _min_max_normalize([h.score for h in semantic_hits])

        fused_scores: Dict[str, float] = {}
        fused_messages: Dict[str, Message] = {}
        fused_terms: Dict[str, set] = {}
        order: List[str] = []
        for hit, norm in zip(keyword_hits, keyword_norm):
            key = _normalize(hit.message.content)
            fused_scores[key] = self.keyword_weight * norm
            fused_messages[key] = hit.message
            fused_terms[key] = set(hit.matched_terms)
            order.append(key)
        for hit, norm in zip(semantic_hits, semantic_norm):
            key = _normalize(hit.message.content)
            contribution = self.semantic_weight * norm
            if key in fused_scores:
                fused_scores[key] = round(fused_scores[key] + contribution, 3)
                fused_terms[key].update(hit.matched_terms)
            else:
                fused_scores[key] = contribution
                fused_messages[key] = hit.message
                fused_terms[key] = set(hit.matched_terms)
                order.append(key)

        scored: List[Tuple[float, int, Message, List[str]]] = []
        for position, key in enumerate(order):
            score = round(fused_scores[key], 3)
            if score > self.min_score:
                scored.append((score, position, fused_messages[key], sorted(fused_terms[key])))
        return _finalize(scored, limit)


# ── context assembly ─────────────────────────────────────────────


@dataclass
class ExcludedItem:
    """A message left out of the assembled context, with the reason why."""
    message: Message
    reason: str

    def __str__(self) -> str:
        preview = self.message.content[:60] + ("..." if len(self.message.content) > 60 else "")
        return f"[{self.message.role}] {preview} -- {self.reason}"


@dataclass
class AssembledContext:
    """Prompt-ready context built by ContextAssembler.

    Attributes:
        messages:     Final ordered messages (system, protected, retrieved,
                      recent, request).
        total_tokens: Sum of token_count across messages.
        max_tokens:   Budget the assembly respected (mandatory messages may
                      still push usage over budget -- see within_budget).
        excluded:     Messages considered but left out, with reasons.
    """
    messages: List[Message] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 0
    excluded: List[ExcludedItem] = field(default_factory=list)

    @property
    def within_budget(self) -> bool:
        """True when total_tokens fits max_tokens."""
        return self.total_tokens <= self.max_tokens

    @property
    def usage_percentage(self) -> float:
        """Token usage as a percentage of the budget."""
        if not self.max_tokens:
            return 0.0
        return round((self.total_tokens / self.max_tokens) * 100, 1)

    def __str__(self) -> str:
        sep = "-" * 52
        state = "within budget" if self.within_budget else "OVER BUDGET"
        lines = [
            sep,
            "Assembled Context",
            sep,
            f"  Messages : {len(self.messages)} "
            f"({self.total_tokens:,} / {self.max_tokens:,} tokens, "
            f"{self.usage_percentage}% -- {state})",
            f"  Excluded : {len(self.excluded)}",
        ]
        for i, msg in enumerate(self.messages, 1):
            preview = msg.content[:44] + ("..." if len(msg.content) > 44 else "")
            lines.append(f"    {i}. [{msg.role}] {preview}")
        for item in self.excluded:
            lines.append(f"    x {item}")
        lines.append(sep)
        return "\n".join(lines)


class ContextAssembler:
    """Packs retrieved + protected + recent + request context into a budget.

    Final order is always: leading system prompt, protected messages
    (conversation order), retrieved hits (rank order), recent messages
    (conversation order), current user request (last).

    Budget discipline: system, protected, and request messages are
    mandatory and never excluded. Retrieved candidates fill remaining
    budget best-first; recent messages fill newest-first. Anything that
    does not fit, or duplicates an included message, lands in
    ``AssembledContext.excluded`` with a reason.

    Args:
        max_tokens:  Token budget (positive int, required).
        keep_recent: How many trailing conversation messages count as
                     "recent" (non-negative int, default 5).

    Usage::

        assembler = ContextAssembler(max_tokens=2000, keep_recent=5)
        ctx = assembler.assemble("where do we persist records?", conv, hits)
    """

    def __init__(self, max_tokens: int, keep_recent: int = 5) -> None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise TypeError(
                f"max_tokens must be a positive integer, "
                f"got {type(max_tokens).__name__!r}."
            )
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}.")
        if not isinstance(keep_recent, int) or isinstance(keep_recent, bool):
            raise TypeError(
                f"keep_recent must be a non-negative integer, "
                f"got {type(keep_recent).__name__!r}."
            )
        if keep_recent < 0:
            raise ValueError(f"keep_recent must be a non-negative integer, got {keep_recent}.")
        self.max_tokens: int = max_tokens
        self.keep_recent: int = keep_recent

    def assemble(
        self,
        request: Union[str, Message],
        conversation: Conversation,
        retrieved: Sequence[RetrievalResult],
    ) -> AssembledContext:
        """Build the final ordered context (never mutates the conversation).

        Args:
            request:      Current user request (string becomes a user Message).
            conversation: Source of system prompt, protected, and recent
                          messages.
            retrieved:    Ranked retrieval hits (any Retriever output).

        Returns:
            AssembledContext with ordered messages, token usage, and
            excluded items. System / protected / request messages are
            always included, even over budget.
        """
        if isinstance(request, str):
            request_msg = Message(role="user", content=request)
        elif isinstance(request, Message):
            request_msg = request
        else:
            raise TypeError(
                f"request must be a string or Message, got {type(request).__name__!r}."
            )
        if not isinstance(conversation, Conversation):
            raise TypeError(
                f"conversation must be a Conversation instance, "
                f"got {type(conversation).__name__!r}."
            )
        hits = list(retrieved)
        for hit in hits:
            if not isinstance(hit, RetrievalResult):
                raise TypeError(
                    "retrieved must contain only RetrievalResult instances, "
                    f"got {type(hit).__name__!r}."
                )

        messages = conversation.get_messages()
        system_prompt = messages[0] if messages and messages[0].role == "system" else None
        protected = [
            m for m in messages
            if m.protected and (system_prompt is None or m is not system_prompt)
        ]
        recent_pool = messages[-self.keep_recent:] if self.keep_recent else []

        included: List[Message] = []
        excluded: List[ExcludedItem] = []
        used_tokens = 0
        seen_content: set[str] = set()
        seen_ids: set[int] = set()

        def note_duplicate(message: Message, label: str) -> None:
            """Record *message* as excluded in favor of an included duplicate."""
            excluded.append(ExcludedItem(
                message=message,
                reason=f"duplicate of {label}",
            ))

        def fits(message: Message) -> bool:
            """True when *message* fits the remaining token budget."""
            return used_tokens + message.token_count <= self.max_tokens

        def take_mandatory(message: Message) -> None:
            """Include *message* unconditionally (system/protected/request)."""
            nonlocal used_tokens
            included.append(message)
            used_tokens += message.token_count
            seen_content.add(_normalize(message.content))
            seen_ids.add(id(message))

        def offer(message: Message, kind: str, dest: List[Message]) -> None:
            """Include *message* in *dest* if new and within budget, else exclude."""
            nonlocal used_tokens
            if id(message) in seen_ids or _normalize(message.content) in seen_content:
                note_duplicate(message, f"included {kind} message")
                return
            if fits(message):
                dest.append(message)
                used_tokens += message.token_count
                seen_content.add(_normalize(message.content))
                seen_ids.add(id(message))
            else:
                excluded.append(ExcludedItem(
                    message=message,
                    reason=f"over token budget ({kind} slot kept higher-priority content)",
                ))

        # 1-2. Mandatory: system prompt + protected (never excluded).
        if system_prompt is not None:
            take_mandatory(system_prompt)
        for message in protected:
            if id(message) in seen_ids:
                continue
            take_mandatory(message)

        # 3. Retrieved candidates, best rank first.
        for hit in sorted(hits, key=lambda h: h.rank if h.rank > 0 else 10**9):
            offer(hit.message, "retrieved", included)

        # 4. Recent messages fill newest-first (oldest excluded first) but
        #    join the final order in conversation order.
        recent_taken: List[Message] = []
        for message in reversed(recent_pool):
            if id(message) in seen_ids or _normalize(message.content) in seen_content:
                continue  # already present via system/protected/retrieved: not "excluded"
            offer(message, "recent", recent_taken)
        included.extend(reversed(recent_taken))

        # 5. Current request: always last, always included.
        if id(request_msg) not in seen_ids:
            included.append(request_msg)
            used_tokens += request_msg.token_count

        return AssembledContext(
            messages=included,
            total_tokens=used_tokens,
            max_tokens=self.max_tokens,
            excluded=excluded,
        )
