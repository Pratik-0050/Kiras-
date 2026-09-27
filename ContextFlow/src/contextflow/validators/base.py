# contextflow/validators/base.py
"""
Abstract base class, ValidationResult, and error type for all validators.

A Validator has one responsibility: compare the original messages (the ones
about to be replaced) with the generated summary text, and decide whether the
summary adequately preserves important information.

The Compactor calls validate() BEFORE committing any changes to the Conversation.
If validation fails (passed=False), the Compactor rolls back and leaves the
Conversation completely unchanged.

Design:
    - ValidationResult is a simple, inspectable data object.
    - Validator is an abstract interface: to add a new validation strategy,
      subclass Validator and implement validate(). Nothing else changes.
    - ValidatorError is raised on infrastructure failures (API errors, etc.),
      not on validation failures. A failed validation is expressed as
      ValidationResult(passed=False, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..message import Message


@dataclass
class ValidationResult:
    """Result from a single validation run.

    Attributes:
        passed:        True if the summary passes validation (safe to commit).
        warnings:      Non-fatal issues found -- informational, won't block compaction.
        missing_items: Specific pieces of information believed to be missing
                       from the summary (requirements, decisions, tasks, facts).
        validator_used: Name of the validator class that produced this result.
        details:       Free-form diagnostic text for debugging or display.
    """
    passed:         bool
    warnings:       List[str] = field(default_factory=list)
    missing_items:  List[str] = field(default_factory=list)
    validator_used: str       = "none"
    details:        str       = ""

    @property
    def has_warnings(self) -> bool:
        """True if there are any non-fatal warnings."""
        return len(self.warnings) > 0

    @property
    def has_missing_items(self) -> bool:
        """True if specific missing items were detected."""
        return len(self.missing_items) > 0

    def __str__(self) -> str:
        sep = "-" * 52
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            sep,
            f"Validation Result  [{status}]",
            sep,
            f"  Validator : {self.validator_used}",
        ]

        if self.missing_items:
            lines.append(f"  Missing ({len(self.missing_items)}):")
            for item in self.missing_items:
                lines.append(f"    - {item}")

        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    ! {w}")

        if self.details:
            lines.append(f"  Details : {self.details}")

        lines.append(sep)
        return "\n".join(lines)


class Validator(ABC):
    """Abstract interface that every validator must implement.

    Subclasses must implement the validate() method to compare original
    messages against the generated summary and return a ValidationResult.
    """

    @abstractmethod
    def validate(
        self,
        original_messages: "List[Message]",
        summary_text: str,
    ) -> ValidationResult:
        """Compare *original_messages* with *summary_text* and return a result.

        Args:
            original_messages: The Message objects that were compressed.
            summary_text:      The plain-text summary produced by the Summarizer.

        Returns:
            A ValidationResult indicating whether important information
            was preserved, along with any warnings or identified missing items.

        Raises:
            ValidatorError: On infrastructure failures (API errors, network issues).
                            A simple validation failure is NOT an exception --
                            it is expressed via ValidationResult(passed=False, ...).
        """
        pass


# BaseValidator is an alias kept for naming consistency with BaseSummarizer.
BaseValidator = Validator


class ValidatorError(Exception):
    """Raised when a validator encounters an infrastructure failure.

    This is distinct from a validation failure (ValidationResult.passed=False).
    Use this only for problems like API timeouts, missing credentials, etc.
    """
    pass
