# contextflow/pipeline.py
"""
Step 17: Automatic end-to-end compaction pipeline.

ContextPipeline wires the engine's stages into one deterministic call:
monitor token usage, detect compaction pressure, retrieve relevant older
context, compact eligible messages (protected / critical content is always
preserved verbatim), validate before committing, and persist the updated
context. Compaction runs only when the configured threshold is exceeded;
when validation fails the original conversation is preserved untouched.

This is orchestration, not agency: a single pass with no loops, no goals,
no tool calling. Infrastructure failures (summarizer / validator /
persistence errors) propagate to the caller instead of being hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .compactor import CompactionResult, Compactor
from .conversation import Conversation
from .message import ImportanceLevel, Message
from .retriever import RetrievalResult, Retriever
from .status import ContextStatus
from .store import ContextStore
from .validators.base import ValidationResult


@dataclass
class PipelineResult:
    """Detailed outcome of one ContextPipeline.run() call.

    Attributes:
        triggered:           True when usage exceeded the trigger threshold
                             and compaction was attempted.
        reason:              Human-readable explanation of the decision
                             (e.g. threshold comparison, skip cause).
        tokens_before:       Total tokens before the run.
        tokens_after:        Total tokens after the run (equal when skipped
                             or rolled back).
        compression_ratio:   tokens_after / tokens_before (1.0 when nothing
                             changed or the conversation was empty; lower
                             means more compressed).
        messages_before:     Message count before the run.
        messages_after:      Message count after the run.
        status_before:       ContextStatus before the run.
        status_after:        ContextStatus after the run.
        committed:           True when the conversation was left in its final
                             state with nothing outstanding (including clean
                             skips). False only when validation failed and
                             the original context was preserved by rollback.
        preserved_messages:  Messages kept verbatim (system, protected,
                             critical, recent).
        summarized_messages: Older messages condensed into the summary.
        validation_passed:   True/False from the validator, or None when no
                             validation ran (skipped or no validator).
        validation_result:   Full ValidationResult, if any.
        warnings:            Non-fatal notes (validator warnings, retrieval
                             skips, persistence notes, rollback explanation).
        protected_count:     Protected messages identified in the run.
        critical_count:      Critical (explicit or auto-scored) messages found.
        important_count:     Important messages found.
        retrieved:           Relevant older context retrieved during the run.
        retrieval_query:     Query used for retrieval, if any.
        persisted_path:      File the updated context was saved to, if any.
        compaction:          Underlying CompactionResult, if compaction ran.
    """
    triggered: bool
    reason: str
    tokens_before: int = 0
    tokens_after: int = 0
    compression_ratio: float = 1.0
    messages_before: int = 0
    messages_after: int = 0
    status_before: ContextStatus = ContextStatus.OK
    status_after: ContextStatus = ContextStatus.OK
    committed: bool = True
    preserved_messages: List[Message] = field(default_factory=list)
    summarized_messages: List[Message] = field(default_factory=list)
    validation_passed: Optional[bool] = None
    validation_result: Optional[ValidationResult] = None
    warnings: List[str] = field(default_factory=list)
    protected_count: int = 0
    critical_count: int = 0
    important_count: int = 0
    retrieved: List[RetrievalResult] = field(default_factory=list)
    retrieval_query: Optional[str] = None
    persisted_path: Optional[str] = None
    compaction: Optional[CompactionResult] = None

    @property
    def tokens_saved(self) -> int:
        """Token reduction achieved (negative when the summary is longer)."""
        return self.tokens_before - self.tokens_after

    def __str__(self) -> str:
        sep = "-" * 52
        state = "TRIGGERED" if self.triggered else "SKIPPED"
        commit = "committed" if self.committed else "ROLLED BACK"
        validation = (
            "n/a" if self.validation_passed is None
            else ("PASSED" if self.validation_passed else "FAILED")
        )
        lines = [
            sep,
            f"Pipeline Result [{state}, {commit}]",
            sep,
            f"  Reason     : {self.reason}",
            f"  Tokens     : {self.tokens_before:,} -> {self.tokens_after:,} "
            f"(ratio {self.compression_ratio:.3f})",
            f"  Messages   : {self.messages_before} -> {self.messages_after}",
            f"  Status     : {self.status_before} -> {self.status_after}",
            f"  Validation : {validation}",
            f"  Preserved  : {len(self.preserved_messages)} "
            f"({self.protected_count} protected, {self.critical_count} critical, "
            f"{self.important_count} important)",
            f"  Summarized : {len(self.summarized_messages)}",
            f"  Retrieved  : {len(self.retrieved)}",
        ]
        if self.persisted_path is not None:
            lines.append(f"  Persisted  : {self.persisted_path}")
        for warning in self.warnings:
            lines.append(f"    ! {warning}")
        lines.append(sep)
        return "\n".join(lines)


class ContextPipeline:
    """Automatic end-to-end compaction pipeline (single deterministic pass).

    Stage order per run():
      1. Monitor  -- snapshot tokens, usage %, and ContextStatus.
      2. Pressure -- compact only when usage meets the trigger threshold.
      3. Retrieve -- surface relevant older context (optional retriever).
      4. Compact  -- Compactor preserves protected/critical content,
                     summarizes eligible messages, validates before commit.
      5. Identify -- count protected / critical / important messages from
                     the compactor's own priority scores (no extra pass).
      6. Persist  -- record + save the updated context (committed runs only).

    Args:
        compactor:        Compactor performing steps 4-5. Defaults to Compactor().
        trigger_pct:      Usage % at/above which compaction runs. Defaults to
                          None, meaning the conversation's own compact_at
                          threshold at run time.
        retriever:        Optional Retriever for stage 3 (keyword, semantic,
                          hybrid, or vector-backed).
        query:            Explicit retrieval query. Defaults to None, meaning
                          the latest user message content at run time.
        retrieval_top_k:  Hits fetched in stage 3 (positive int, default 5).
        store:            Optional ContextStore used for stage 6.
        persist_path:     Optional file path for stage 6. Persistence runs
                          only when set (a store is created when none given).

    Usage::

        pipeline = ContextPipeline(
            retriever=KeywordRetriever(),
            store=ContextStore(conv),
            persist_path="session.json",
        )
        result = pipeline.run(conv)
        print(result)
    """

    def __init__(
        self,
        compactor: Optional[Compactor] = None,
        trigger_pct: Optional[float] = None,
        retriever: Optional[Retriever] = None,
        query: Optional[str] = None,
        retrieval_top_k: int = 5,
        store: Optional[ContextStore] = None,
        persist_path: Optional[str] = None,
    ) -> None:
        if compactor is not None and not isinstance(compactor, Compactor):
            raise TypeError(
                "compactor must be a Compactor instance, "
                f"got {type(compactor).__name__!r}."
            )
        if trigger_pct is not None:
            if not isinstance(trigger_pct, (int, float)) or isinstance(trigger_pct, bool):
                raise TypeError(
                    "trigger_pct must be a number between 0 and 100 or None, "
                    f"got {type(trigger_pct).__name__!r}."
                )
            if not (0.0 < trigger_pct <= 100.0):
                raise ValueError(
                    f"trigger_pct must be between 0 and 100, got {trigger_pct}."
                )
        if retriever is not None and not isinstance(retriever, Retriever):
            raise TypeError(
                "retriever must be a Retriever instance or None, "
                f"got {type(retriever).__name__!r}."
            )
        if query is not None and not isinstance(query, str):
            raise TypeError(
                f"query must be a string or None, got {type(query).__name__!r}."
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
        if store is not None and not isinstance(store, ContextStore):
            raise TypeError(
                "store must be a ContextStore instance or None, "
                f"got {type(store).__name__!r}."
            )
        if persist_path is not None:
            if not isinstance(persist_path, str):
                raise TypeError(
                    "persist_path must be a file-path string or None, "
                    f"got {type(persist_path).__name__!r}."
                )
            if not persist_path.strip():
                raise ValueError("persist_path must not be empty.")

        self.compactor: Compactor = compactor or Compactor()
        self.trigger_pct: Optional[float] = (
            float(trigger_pct) if trigger_pct is not None else None
        )
        self.retriever: Optional[Retriever] = retriever
        self.query: Optional[str] = query
        self.retrieval_top_k: int = retrieval_top_k
        self.store: Optional[ContextStore] = store
        self.persist_path: Optional[str] = persist_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        conversation: Conversation,
        query: Optional[str] = None,
    ) -> PipelineResult:
        """Run one automatic compaction pass over *conversation*.

        Args:
            conversation: The Conversation to monitor and possibly compact
                          (modified in place on commit only).
            query:        Per-run retrieval query, overriding the
                          constructor query for this call only.

        Returns:
            PipelineResult with the trigger decision, token before/after,
            compression ratio, preserved/summarized messages, validation
            status, warnings, and persistence info.

        Raises:
            TypeError:       If *conversation* is not a Conversation.
            SummarizerError: If summarization fails (conversation untouched).
            ValidatorError:  If validation errors (conversation untouched).
            StoreError:      If persistence fails (compaction already
                             committed in memory; disk state unchanged).
        """
        if not isinstance(conversation, Conversation):
            raise TypeError(
                "conversation must be a Conversation instance, "
                f"got {type(conversation).__name__!r}."
            )
        if query is not None and not isinstance(query, str):
            raise TypeError(
                f"query must be a string or None, got {type(query).__name__!r}."
            )

        # ── 1. Monitor ───────────────────────────────────────────────
        tokens_before = conversation.total_tokens()
        messages_before = conversation.message_count()
        status_before = conversation.get_status()
        usage = conversation.usage_percentage()

        if usage is None:
            return PipelineResult(
                triggered=False,
                reason="skipped: conversation has no token limit set",
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                compression_ratio=1.0,
                messages_before=messages_before,
                messages_after=messages_before,
                status_before=status_before,
                status_after=status_before,
                committed=True,
                warnings=["monitor: no max_tokens set, pressure is undefined"],
            )

        trigger = self.trigger_pct if self.trigger_pct is not None else conversation.compact_at

        # ── 2. Pressure gate ─────────────────────────────────────────
        if usage < trigger:
            return PipelineResult(
                triggered=False,
                reason=(
                    f"skipped: usage {usage}% below trigger {trigger}% "
                    f"({tokens_before:,} tokens)"
                ),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                compression_ratio=1.0,
                messages_before=messages_before,
                messages_after=messages_before,
                status_before=status_before,
                status_after=status_before,
                committed=True,
                warnings=[f"monitor: {conversation.get_status_message()}"],
            )

        # ── 3. Retrieve relevant older context (observability stage) ─
        effective_query = query if query is not None else self.query
        if effective_query is None:
            effective_query = _latest_user_content(conversation)
        retrieved: List[RetrievalResult] = []
        warnings: List[str] = []
        if self.retriever is None:
            warnings.append("retrieval skipped: no retriever configured")
            effective_query = None
        elif not effective_query or not effective_query.strip():
            warnings.append("retrieval skipped: no query available")
            effective_query = None
        else:
            retrieved = self.retriever.retrieve(
                effective_query, conversation, top_k=self.retrieval_top_k
            )

        # ── 4-5. Compact (validates before commit) + identify ────────
        result = self.compactor.compact(conversation)

        protected = [m for m in result.preserved_messages if m.protected]
        critical = [
            m for m in result.preserved_messages
            if m.importance == ImportanceLevel.CRITICAL
        ]
        auto_critical = [
            ps for ps in result.priority_scores
            if ps.classification == ImportanceLevel.CRITICAL
        ]
        important = [
            m for m in result.preserved_messages + result.summarized_messages
            if m.importance == ImportanceLevel.IMPORTANT
        ]
        # Union explicit-critical with auto-scored-critical message identities.
        critical_ids = {id(m) for m in critical} | {id(ps.message) for ps in auto_critical}

        tokens_after = conversation.total_tokens()
        if result.validation_result is not None:
            warnings.extend(result.validation_result.warnings)
            validation_passed: Optional[bool] = result.validation_result.passed
        else:
            validation_passed = None

        persisted_path: Optional[str] = None
        if result.committed and self.persist_path is not None:
            target_store = self.store or ContextStore(conversation)
            if target_store.conversation is not conversation:
                # Provided store tracks a different conversation: point it at
                # this one so the persisted state matches what was compacted.
                target_store.conversation = conversation
            target_store.record(result)
            target_store.save(self.persist_path)
            persisted_path = self.persist_path
            warnings.append(f"persisted updated context to {self.persist_path!r}")
        elif not result.committed:
            warnings.append(
                "validation failed: original context preserved, nothing persisted"
            )

        if tokens_before > 0:
            ratio = round(tokens_after / tokens_before, 3)
        else:
            ratio = 1.0

        return PipelineResult(
            triggered=True,
            reason=(
                f"triggered: usage {usage}% >= trigger {trigger}% "
                f"({tokens_before:,} tokens)"
            ),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compression_ratio=ratio,
            messages_before=messages_before,
            messages_after=conversation.message_count(),
            status_before=status_before,
            status_after=conversation.get_status(),
            committed=result.committed,
            preserved_messages=list(result.preserved_messages),
            summarized_messages=list(result.summarized_messages),
            validation_passed=validation_passed,
            validation_result=result.validation_result,
            warnings=warnings,
            protected_count=len(protected),
            critical_count=len(critical_ids),
            important_count=len(important),
            retrieved=retrieved,
            retrieval_query=effective_query,
            persisted_path=persisted_path,
            compaction=result,
        )


def _latest_user_content(conversation: Conversation) -> Optional[str]:
    """Return the latest user message content, if any."""
    for message in reversed(conversation.get_messages()):
        if message.role == "user" and message.content.strip():
            return message.content
    return None
