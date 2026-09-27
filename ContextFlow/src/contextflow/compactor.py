# contextflow/compactor.py
"""
Defines Compactor and CompactionResult.

The Compactor identifies which messages to compress, delegates the
actual text generation to a modular Summarizer, optionally validates the
generated summary, then replaces the older messages and recalculates token usage.

Step 7: Modular summarizer injected via the Summarizer interface.
Step 8: Modular validator injected via the Validator interface.
        Validation runs BEFORE committing changes. If it fails, the Conversation
        is rolled back and left completely unchanged.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .message import Message, ImportanceLevel
from .conversation import Conversation
from .summarizers.base import Summarizer, SummarizerError
from .summarizers.placeholder import PlaceholderSummarizer
from .validators.base import Validator, ValidationResult
from .scorers.base import PriorityScorer, PriorityScore
from .scorers.heuristic_scorer import HeuristicPriorityScorer


@dataclass
class CompactionResult:
    """Before/after statistics from a single compaction run.

    Attributes:
        original_message_count   -- Messages before compaction.
        original_token_count     -- Total tokens before compaction.
        compacted_message_count  -- Messages after compaction.
        compacted_token_count    -- Total tokens after compaction.
        messages_removed         -- How many messages were removed/replaced.
        tokens_saved             -- Reduction in token count (negative if
                                    the summary is longer than the originals).
        summary_message          -- The Message object inserted as summary.
        was_needed               -- False if nothing was compacted.
        summarizer_used          -- Name of the summarizer class that ran.
        validation_result        -- ValidationResult from the validator, or None
                                    if no validator was configured.
        committed                -- True if the compacted conversation was saved.
                                    False if validation failed and it was rolled back.
        preserved_messages       -- Messages preserved verbatim in the conversation.
        summarized_messages      -- Messages selected and passed to the summarizer.
        removed_messages         -- Messages removed from the conversation (discarded + replaced).
        priority_scores          -- PriorityScore objects for each message evaluated.
    """
    original_message_count:  int
    original_token_count:    int
    compacted_message_count: int
    compacted_token_count:   int
    messages_removed:        int
    tokens_saved:            int
    summary_message:         Optional[Message]
    was_needed:              bool
    summarizer_used:         str                      = "none"
    validation_result:       Optional[ValidationResult] = field(default=None)
    committed:               bool                     = True
    preserved_messages:      List[Message]            = field(default_factory=list)
    summarized_messages:     List[Message]            = field(default_factory=list)
    removed_messages:        List[Message]            = field(default_factory=list)
    priority_scores:         List[PriorityScore]      = field(default_factory=list)

    @property
    def discarded_messages(self) -> List[Message]:
        """Messages that were discarded directly without being summarized."""
        summarized_ids = {id(m) for m in self.summarized_messages}
        return [m for m in self.removed_messages if id(m) not in summarized_ids]

    def __str__(self) -> str:
        if not self.was_needed:
            return "CompactionResult: no compaction was needed."

        sep = "-" * 52
        saved_label = (
            f"saved {self.tokens_saved:,}"
            if self.tokens_saved >= 0
            else f"increased by {abs(self.tokens_saved):,} (summary longer than originals)"
        )
        token_line = (
            f"  Tokens     : {self.original_token_count:,} -> "
            f"{self.compacted_token_count:,} ({saved_label})"
            if self.committed
            else f"  Tokens     : {self.original_token_count:,} (unchanged -- rolled back)"
        )
        lines = [
            sep,
            "Compaction Result",
            sep,
        ]
        if not self.committed:
            lines.append("  ** ROLLED BACK -- validation failed **")
        lines.extend([
            f"  Summarizer : {self.summarizer_used}",
            f"  Messages   : {self.original_message_count} -> "
            f"{self.compacted_message_count} "
            f"(-{self.messages_removed} removed)"
            if self.committed
            else f"  Messages   : {self.original_message_count} (unchanged -- rolled back)",
            token_line,
        ])
        if self.validation_result is not None:
            vr = self.validation_result
            vstatus = "PASSED" if vr.passed else "FAILED"
            lines.append(f"  Validation : {vstatus} via {vr.validator_used}")
            for item in vr.missing_items:
                lines.append(f"    missing: {item}")
            for w in vr.warnings:
                lines.append(f"    warning: {w}")

        if self.priority_scores:
            lines.append(f"  Priority Scores ({len(self.priority_scores)}) :")
            for ps in self.priority_scores:
                prot_str = " [protected]" if ps.message.protected else ""
                preview = ps.message.content[:36] + ("..." if len(ps.message.content) > 36 else "")
                lines.append(
                    f"    - [{ps.message.role}] {ps.score:4.1f}/100 ({ps.classification.value}){prot_str}: {preview}"
                )

        if self.committed:
            lines.append(f"  Preserved ({len(self.preserved_messages)}) :")
            if self.preserved_messages:
                for m in self.preserved_messages:
                    reasons = []
                    if m.protected:
                        reasons.append("protected")
                    if m.importance == ImportanceLevel.CRITICAL:
                        reasons.append("critical")
                    tag = f" [{', '.join(reasons)}]" if reasons else ""
                    content_preview = m.content[:38] + ("..." if len(m.content) > 38 else "")
                    lines.append(f"    - [{m.role}]{tag} {content_preview}")
            else:
                lines.append("    (none)")

            lines.append(f"  Summarized ({len(self.summarized_messages)}) :")
            if self.summarized_messages:
                for m in self.summarized_messages:
                    content_preview = m.content[:38] + ("..." if len(m.content) > 38 else "")
                    lines.append(f"    - [{m.role}] ({m.importance.value}) {content_preview}")
            else:
                lines.append("    (none)")

            lines.append(f"  Removed ({len(self.removed_messages)}) :")
            if self.removed_messages:
                discarded_ids = {id(m) for m in self.discarded_messages}
                for m in self.removed_messages:
                    how = "discarded" if id(m) in discarded_ids else "replaced by summary"
                    content_preview = m.content[:38] + ("..." if len(m.content) > 38 else "")
                    lines.append(f"    - [{m.role}] ({how}) {content_preview}")
            else:
                lines.append("    (none)")

        lines.append(sep)
        return "\n".join(line for line in lines if line is not None)


class Compactor:
    """Compacts a Conversation by summarising older messages.

    Strategy:
      1. Always preserve the first system message verbatim if present.
      2. Preserve the most recent keep_recent non-system messages verbatim.
      3. Automatically score all messages via PriorityScorer (evaluating role,
         recency, explicit importance, protected status, and content).
      4. Preserve all older messages marked as protected=True, importance=critical,
         or scored as critical.
      5. Discard older messages classified as discardable without summarising.
      6. Generate a summary of remaining older messages (normal/important) via Summarizer.
      7. If a Validator is configured, run it BEFORE modifying the Conversation.
         If validation fails, discard the summary and leave the Conversation
         completely unchanged (the result has committed=False).
      8. Replace the summarized/discarded messages with the summary message and
         recalculate token usage.

    Args:
        keep_recent:         How many recent messages to leave untouched.
                             Must be a positive integer. Defaults to 5.
        summarizer:          A Summarizer instance. Defaults to PlaceholderSummarizer.
        fallback_summarizer: Optional secondary Summarizer used if the primary
                             raises SummarizerError.
        validator:           Optional Validator instance. When set, validation runs
                             before committing; failures roll back the compaction.
        on_validation_fail:  What to do when validation fails.
                             "rollback" (default): keep original messages unchanged.
                             "warn":               apply compaction anyway but record failure.
        scorer:              Optional PriorityScorer instance. Defaults to HeuristicPriorityScorer.

    Usage::

        from contextflow import Compactor, Conversation, Message, ImportanceLevel
        from contextflow.summarizers import OpenAISummarizer
        from contextflow.validators import HeuristicValidator
        from contextflow.scorers import HeuristicPriorityScorer

        compactor = Compactor(
            keep_recent=5,
            summarizer=OpenAISummarizer(),
            validator=HeuristicValidator(),
            scorer=HeuristicPriorityScorer(),
        )
        result = compactor.compact(conversation)
        print(result)
    """

    _VALID_FAIL_MODES = frozenset({"rollback", "warn"})

    def __init__(
        self,
        keep_recent:         int                  = 5,
        summarizer:          Optional[Summarizer] = None,
        fallback_summarizer: Optional[Summarizer] = None,
        validator:           Optional[Validator]  = None,
        on_validation_fail:  str                  = "rollback",
        scorer:              Optional[PriorityScorer] = None,
    ) -> None:
        if not isinstance(keep_recent, int) or isinstance(keep_recent, bool):
            raise TypeError(
                f"keep_recent must be a positive integer, "
                f"got {type(keep_recent).__name__!r}."
            )
        if keep_recent <= 0:
            raise ValueError(
                f"keep_recent must be a positive integer, got {keep_recent}."
            )
        if summarizer is not None and not isinstance(summarizer, Summarizer):
            raise TypeError(
                f"summarizer must implement the Summarizer interface, "
                f"got {type(summarizer).__name__!r}."
            )
        if fallback_summarizer is not None and not isinstance(fallback_summarizer, Summarizer):
            raise TypeError(
                f"fallback_summarizer must implement the Summarizer interface, "
                f"got {type(fallback_summarizer).__name__!r}."
            )
        if validator is not None and not isinstance(validator, Validator):
            raise TypeError(
                f"validator must implement the Validator interface, "
                f"got {type(validator).__name__!r}."
            )
        if scorer is not None and not isinstance(scorer, PriorityScorer):
            raise TypeError(
                f"scorer must implement the PriorityScorer interface, "
                f"got {type(scorer).__name__!r}."
            )
        if on_validation_fail not in self._VALID_FAIL_MODES:
            raise ValueError(
                f"on_validation_fail must be one of {sorted(self._VALID_FAIL_MODES)!r}, "
                f"got {on_validation_fail!r}."
            )

        self.keep_recent:         int                  = keep_recent
        self.summarizer:          Summarizer           = summarizer or PlaceholderSummarizer()
        self.fallback_summarizer: Optional[Summarizer] = fallback_summarizer
        self.validator:           Optional[Validator]  = validator
        self.on_validation_fail:  str                  = on_validation_fail
        self.scorer:              PriorityScorer       = scorer or HeuristicPriorityScorer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compact(self, conversation: Conversation) -> CompactionResult:
        """Compact conversation (optionally validating first) and return a CompactionResult.

        Workflow:
          1. Separate leading system prompt (always preserved).
          2. Keep the most recent keep_recent messages untouched.
          3. Classify older messages:
             - protected=True or importance=CRITICAL: preserved verbatim.
             - importance=DISCARDABLE: discarded directly.
             - importance=NORMAL or IMPORTANT: selected for summarization.
          4. If nothing to summarize and nothing to discard, compaction is not needed.
          5. Generate summary text via the summarizer for summarizable messages.
          6. If a validator is configured, run it against the summary BEFORE
             committing. If validation fails and on_validation_fail="rollback",
             return a result with committed=False and the conversation unchanged.
          7. Rebuild the conversation in-place with summary and preserved messages.
          8. Recalculate token usage and status.

        Args:
            conversation: The Conversation to compact (modified in-place, unless
                          validation fails with on_validation_fail="rollback").

        Returns:
            A CompactionResult describing what happened, including breakdown of
            preserved, summarized, and removed messages.

        Raises:
            SummarizerError: If the summarizer (and fallback) both fail.
            ValidatorError:  If the validator encounters an infrastructure failure.
        """
        messages      = conversation.get_messages()
        before_count  = conversation.message_count()
        before_tokens = conversation.total_tokens()

        # ── 1. Automatically score all messages in conversation ──────────────
        priority_scores: List[PriorityScore] = self.scorer.score_messages(messages)
        score_map = {id(ps.message): ps for ps in priority_scores}

        # ── 2. Separate leading system prompt ────────────────────────────────
        system_prompt: Optional[Message] = None
        rest: List[Message]              = []

        if messages and messages[0].role == "system":
            system_prompt = messages[0]
            rest          = messages[1:]
        else:
            rest = messages

        # ── 3. Early-exit if not enough messages to compact ──────────────────
        if len(rest) <= self.keep_recent:
            return CompactionResult(
                original_message_count  = before_count,
                original_token_count    = before_tokens,
                compacted_message_count = before_count,
                compacted_token_count   = before_tokens,
                messages_removed        = 0,
                tokens_saved            = 0,
                summary_message         = None,
                was_needed              = False,
                committed               = True,
                summarizer_used         = type(self.summarizer).__name__,
                preserved_messages      = list(messages),
                summarized_messages     = [],
                removed_messages        = [],
                priority_scores         = priority_scores,
            )

        older:   List[Message] = rest[: len(rest) - self.keep_recent]
        to_keep: List[Message] = rest[len(rest) - self.keep_recent :]

        # ── 4. Classify older messages using priority scores & protection ─────
        preserved_older: List[Message] = []
        to_summarize:    List[Message] = []
        to_discard:      List[Message] = []

        for msg in older:
            ps = score_map.get(id(msg))
            cls = ps.classification if ps is not None else msg.importance

            # Always protect protected and critical messages
            if msg.protected or msg.importance == ImportanceLevel.CRITICAL or cls == ImportanceLevel.CRITICAL:
                preserved_older.append(msg)
            elif cls == ImportanceLevel.DISCARDABLE:
                to_discard.append(msg)
            else:
                to_summarize.append(msg)

        # If nothing to summarize and nothing to discard, no compaction needed
        if not to_summarize and not to_discard:
            return CompactionResult(
                original_message_count  = before_count,
                original_token_count    = before_tokens,
                compacted_message_count = before_count,
                compacted_token_count   = before_tokens,
                messages_removed        = 0,
                tokens_saved            = 0,
                summary_message         = None,
                was_needed              = False,
                committed               = True,
                summarizer_used         = type(self.summarizer).__name__,
                preserved_messages      = list(messages),
                summarized_messages     = [],
                removed_messages        = [],
                priority_scores         = priority_scores,
            )

        # ── 5. Generate summary (if there are messages to summarize) ─────────
        summary_msg:       Optional[Message]          = None
        validation_result: Optional[ValidationResult] = None
        used_name:         str                        = "none"

        if to_summarize:
            used_name = type(self.summarizer).__name__
            try:
                summary_text = self.summarizer.summarize(to_summarize)
            except SummarizerError as exc:
                if self.fallback_summarizer is not None:
                    summary_text = self.fallback_summarizer.summarize(to_summarize)
                    used_name    = f"{type(self.fallback_summarizer).__name__} (fallback)"
                else:
                    raise exc

            # ── 6. Validate BEFORE committing ────────────────────────────────
            if self.validator is not None:
                validation_result = self.validator.validate(to_summarize, summary_text)

                if not validation_result.passed and self.on_validation_fail == "rollback":
                    # Conversation is unchanged -- return a rolled-back result.
                    candidate_removed = [m for m in older if m in to_discard or m in to_summarize]
                    return CompactionResult(
                        original_message_count  = before_count,
                        original_token_count    = before_tokens,
                        compacted_message_count = before_count,
                        compacted_token_count   = before_tokens,
                        messages_removed        = len(candidate_removed),
                        tokens_saved            = 0,
                        summary_message         = None,
                        was_needed              = True,
                        committed               = False,
                        summarizer_used         = used_name,
                        validation_result       = validation_result,
                        preserved_messages      = list(messages),
                        summarized_messages     = list(to_summarize),
                        removed_messages        = candidate_removed,
                        priority_scores         = priority_scores,
                    )

            # Build summary message
            summary_msg = Message(
                role     = "system",
                content  = summary_text,
                metadata = {
                    "type":                "compaction_summary",
                    "messages_summarised": len(to_summarize),
                    "messages_discarded":  len(to_discard),
                    "summarizer":          used_name,
                    "validation_passed":   (
                        validation_result.passed if validation_result else None
                    ),
                },
            )

        # ── 7. Rebuild conversation in-place ──────────────────────────────────
        conversation.clear()
        if system_prompt:
            conversation.add(system_prompt)
        if summary_msg:
            conversation.add(summary_msg)
        for msg in preserved_older:
            conversation.add(msg)
        for msg in to_keep:
            conversation.add(msg)

        after_count  = conversation.message_count()
        after_tokens = conversation.total_tokens()

        preserved_all = ([system_prompt] if system_prompt else []) + preserved_older + to_keep
        removed_all   = [m for m in older if m in to_discard or m in to_summarize]

        return CompactionResult(
            original_message_count  = before_count,
            original_token_count    = before_tokens,
            compacted_message_count = after_count,
            compacted_token_count   = after_tokens,
            messages_removed        = len(removed_all),
            tokens_saved            = before_tokens - after_tokens,
            summary_message         = summary_msg,
            was_needed              = True,
            committed               = True,
            summarizer_used         = used_name,
            validation_result       = validation_result,
            preserved_messages      = preserved_all,
            summarized_messages     = list(to_summarize),
            removed_messages        = removed_all,
            priority_scores         = priority_scores,
        )
