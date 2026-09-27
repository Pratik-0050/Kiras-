# contextflow/scorers/base.py
"""
Defines the PriorityScorer abstract base class and PriorityScore data structure.

Every priority scorer evaluates messages based on factors such as role,
recency, explicit importance, protected status, and content characteristics,
producing a 0-100 score and an ImportanceLevel classification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from ..message import Message, ImportanceLevel


class ScorerError(Exception):
    """Raised when priority scoring fails."""
    pass


@dataclass
class PriorityScore:
    """The result of scoring a single message for context priority.

    Attributes:
        score:          Numeric priority score from 0.0 to 100.0.
        classification: ImportanceLevel (CRITICAL, IMPORTANT, NORMAL, DISCARDABLE).
        message:        The Message object that was evaluated.
        factors:        Dictionary of individual factor contributions to the score.
        reason:         Human-readable explanation of why this score was assigned.
    """
    score:          float
    classification: ImportanceLevel
    message:        Message
    factors:        Dict[str, float] = field(default_factory=dict)
    reason:         str              = ""

    def __post_init__(self) -> None:
        # Clamp score to [0.0, 100.0]
        self.score = max(0.0, min(100.0, round(float(self.score), 1)))

        # Ensure classification is an ImportanceLevel
        if isinstance(self.classification, str):
            self.classification = ImportanceLevel(self.classification.lower())

    def __str__(self) -> str:
        prot_str = " [protected]" if self.message.protected else ""
        preview = self.message.content[:45] + ("..." if len(self.message.content) > 45 else "")
        return (
            f"Score: {self.score:5.1f} ({self.classification.value}){prot_str} "
            f"[{self.message.role}] {preview}"
        )


class PriorityScorer(ABC):
    """Abstract base class for all context priority scorers.

    Subclasses must implement `score()`, which evaluates a single message
    within the context of a conversation. `score_messages()` is provided with
    a default implementation that scores a sequence of messages.
    """

    @abstractmethod
    def score(
        self,
        message: Message,
        index: int = 0,
        total_messages: int = 1,
    ) -> PriorityScore:
        """Assign a priority score (0-100) and classification to *message*.

        Args:
            message:        The Message to score.
            index:          0-based position of the message in the conversation.
            total_messages: Total number of messages in the conversation.

        Returns:
            A PriorityScore dataclass instance.
        """
        pass

    def score_messages(self, messages: List[Message]) -> List[PriorityScore]:
        """Score an ordered sequence of messages in a conversation.

        Args:
            messages: List of Message objects to evaluate.

        Returns:
            List of PriorityScore objects corresponding to each input message.
        """
        total = len(messages)
        return [self.score(msg, index=i, total_messages=total) for i, msg in enumerate(messages)]


# Alias for consistency with BaseSummarizer and BaseValidator
BasePriorityScorer = PriorityScorer
