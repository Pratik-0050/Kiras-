# contextflow/validators/heuristic_validator.py
"""
HeuristicValidator: offline, zero-dependency validation using keyword and
pattern extraction from original messages and the generated summary.

Strategy:
    1. Extract candidate "important phrases" from each original message using
       simple heuristics:
           - Quoted strings  (e.g. "gpt-4o", "Python")
           - Capitalised multi-word phrases (e.g. GitHub App, Azure OpenAI)
           - Numeric values and version strings  (e.g. 8000, v3.2)
           - Action-bearing sentences containing indicator words:
             must, should, require, decide, agree, use, implement, never, always
       These are treated as candidates for "important information".

    2. Check which candidates are NOT present in the summary text (case-insensitive).

    3. A fixed threshold controls how many missing candidates constitutes a failure:
           missing_ratio = missing / total_candidates
           if missing_ratio > fail_threshold  ->  passed=False

    4. Return a ValidationResult with `passed`, `warnings`, and `missing_items`.

This validator is deterministic, instant, and requires no external services.
It is intentionally conservative: it produces warnings but only fails when a
significant fraction of key terms cannot be found.
"""

from __future__ import annotations

import re
from typing import List, TYPE_CHECKING

from .base import Validator, ValidationResult

if TYPE_CHECKING:
    from ..message import Message


# Indicator words whose sentences are considered important.
_ACTION_WORDS = frozenset({
    "must", "should", "shall", "require", "required",
    "decide", "decided", "decision", "agree", "agreed",
    "use", "using", "implement", "implementing",
    "never", "always", "critical", "important", "ensure",
    "choose", "chosen", "select", "selected",
    "goal", "objective", "constraint", "requirement",
    "todo", "to-do", "pending", "unresolved", "open",
    "action", "next", "follow-up",
})


def _extract_candidates(messages: "List[Message]") -> List[str]:
    """Extract a deduplicated list of candidate phrases from messages."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(phrase: str) -> None:
        """Append *phrase* unless blank, tiny, or already collected."""
        key = phrase.lower().strip()
        if key and key not in seen and len(key) > 2:
            seen.add(key)
            candidates.append(phrase.strip())

    for msg in messages:
        text = msg.content

        # 1. Quoted strings  "like this" or 'like this'
        for m in re.finditer(r'["\']([^"\']{2,60})["\']', text):
            add(m.group(1))

        # 2. Capitalised multi-word phrases (Title Case, at least 2 words)
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            add(m.group(1))

        # 3. Numeric / version values (stand-alone numbers, e.g. 8000, v3.2, 2.0)
        for m in re.finditer(r'\bv?\d+(?:\.\d+)+\b|\b\d{3,}\b', text):
            add(m.group(0))

        # 4. Sentences containing action/indicator words
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            words_lower = sentence.lower().split()
            if any(w in _ACTION_WORDS for w in words_lower):
                # Use the whole sentence but trim to ≤ 80 chars for readability
                trimmed = sentence.strip()
                if len(trimmed) > 80:
                    trimmed = trimmed[:77] + "..."
                add(trimmed)

    return candidates


class HeuristicValidator(Validator):
    """Validates summaries using keyword and pattern extraction (no API calls).

    Args:
        fail_threshold:  Fraction of missing candidates that triggers a failure.
                         0.5 means "fail if more than half of key terms are absent".
                         Defaults to 0.5.
        warn_threshold:  Fraction of missing candidates that triggers a warning.
                         Defaults to 0.25.
        min_candidates:  Minimum number of candidates needed to run validation.
                         If fewer are found, validation is skipped (passed=True).
                         Defaults to 3.
    """

    def __init__(
        self,
        fail_threshold: float = 0.50,
        warn_threshold: float = 0.25,
        min_candidates: int   = 3,
    ) -> None:
        if not (0.0 < fail_threshold <= 1.0):
            raise ValueError("fail_threshold must be between 0 (exclusive) and 1 (inclusive).")
        if not (0.0 < warn_threshold <= 1.0):
            raise ValueError("warn_threshold must be between 0 (exclusive) and 1 (inclusive).")
        if warn_threshold > fail_threshold:
            raise ValueError("warn_threshold must be <= fail_threshold.")
        if min_candidates < 1:
            raise ValueError("min_candidates must be at least 1.")

        self.fail_threshold = fail_threshold
        self.warn_threshold = warn_threshold
        self.min_candidates = min_candidates

    def validate(
        self,
        original_messages: "List[Message]",
        summary_text: str,
    ) -> ValidationResult:
        """Run heuristic validation and return a ValidationResult.

        Args:
            original_messages: The Message objects that were compressed.
            summary_text:      The plain-text summary to validate.

        Returns:
            ValidationResult with passed, warnings, and missing_items populated.
        """
        name = type(self).__name__

        if not original_messages:
            return ValidationResult(
                passed=True,
                details="No messages to validate.",
                validator_used=name,
            )

        if not summary_text or not summary_text.strip():
            return ValidationResult(
                passed=False,
                missing_items=["<entire summary is empty>"],
                details="The summary is blank.",
                validator_used=name,
            )

        candidates = _extract_candidates(original_messages)

        if len(candidates) < self.min_candidates:
            return ValidationResult(
                passed=True,
                warnings=[
                    f"Only {len(candidates)} candidate phrase(s) found "
                    f"(min_candidates={self.min_candidates}); skipping coverage check."
                ],
                details="Insufficient candidates to validate.",
                validator_used=name,
            )

        summary_lower = summary_text.lower()
        missing: list[str] = []

        for phrase in candidates:
            if phrase.lower() not in summary_lower:
                missing.append(phrase)

        missing_ratio = len(missing) / len(candidates)
        passed   = missing_ratio <= self.fail_threshold
        warnings: list[str] = []

        if missing_ratio > self.warn_threshold and passed:
            warnings.append(
                f"{len(missing)}/{len(candidates)} candidate phrase(s) not found "
                f"in summary ({missing_ratio:.0%} missing -- approaching threshold)."
            )

        details = (
            f"Checked {len(candidates)} candidate phrase(s); "
            f"{len(missing)} missing ({missing_ratio:.0%}). "
            f"Threshold: fail>{self.fail_threshold:.0%}, warn>{self.warn_threshold:.0%}."
        )

        return ValidationResult(
            passed=passed,
            warnings=warnings,
            missing_items=missing,
            details=details,
            validator_used=name,
        )
