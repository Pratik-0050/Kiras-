# contextflow/summarizers/base.py
"""
Abstract base class and error definitions for all summarizers.

A Summarizer has one responsibility: take a list of Message objects and
produce a concise, high-density summary string. The Compactor wraps this
summary string into a new system Message inserted into the Conversation.

To add support for any LLM provider (OpenAI, Anthropic, local models, etc.),
subclass Summarizer (or BaseSummarizer) and implement the summarize() method.
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..message import Message


class Summarizer(ABC):
    """Abstract interface that every summarizer must implement.

    Subclasses must implement the summarize() method to convert a list of
    Message objects into a concise summary string.
    """

    @abstractmethod
    def summarize(self, messages: "List[Message]") -> str:
        """Convert *messages* into a concise summary string.

        Args:
            messages: The older Message objects to compress.

        Returns:
            A plain-text summary. This will become the content of a
            new system Message inserted into the Conversation.

        Raises:
            SummarizerError: If the summary could not be produced due to
                             API failure, network issues, or invalid response.
        """
        pass


# BaseSummarizer alias for backwards-compatibility and convention preference.
BaseSummarizer = Summarizer


class SummarizerError(Exception):
    """Raised when a summarizer fails to produce a summary."""
    pass
