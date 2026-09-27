# contextflow/summarizers/__init__.py
"""
Summarizers module for ContextFlow.
Provides modular summarizer implementations and the abstract Summarizer interface.
"""

from .base import Summarizer, BaseSummarizer, SummarizerError
from .placeholder import PlaceholderSummarizer
from .openai_summarizer import OpenAISummarizer

__all__ = [
    "Summarizer",
    "BaseSummarizer",
    "SummarizerError",
    "PlaceholderSummarizer",
    "OpenAISummarizer",
]
