# contextflow/summarizers/placeholder.py
"""
PlaceholderSummarizer: local fallback summarizer.

Produces a structured listing of older messages without calling
any external API. Useful for testing the compaction pipeline
offline or when no API key is configured.
"""

from typing import List, TYPE_CHECKING
from .base import Summarizer

if TYPE_CHECKING:
    from ..message import Message


class PlaceholderSummarizer(Summarizer):
    """Summarizes messages by listing their roles and content verbatim.

    No API call is made. The output is a structured text block, not
    a real prose summary. Use OpenAISummarizer for actual LLM compression.
    """

    def summarize(self, messages: "List[Message]") -> str:
        """Produce a formatted listing of messages without external API calls."""
        if not messages:
            return ""

        lines = [
            f"[CONTEXT SUMMARY -- {len(messages)} older message(s) compacted]",
            "(Placeholder: no LLM was used. Install and configure an API key for",
            " real summarization.)",
            "",
        ]
        for i, msg in enumerate(messages, start=1):
            excerpt = (
                msg.content if len(msg.content) <= 120
                else msg.content[:117] + "..."
            )
            lines.append(f"  {i}. [{msg.role.upper()}] {excerpt}")
        lines.append("")
        lines.append("[End of summary. Conversation continues below.]")
        return "\n".join(lines)
