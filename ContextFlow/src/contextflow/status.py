# contextflow/status.py
"""
Defines ContextStatus: the three pressure levels a Conversation can be in.

OK               -- Usage is below the warning threshold. No action needed.
WARNING          -- Usage is approaching the limit. Consider planning compaction.
COMPACTION_NEEDED -- Usage is at or above the compaction threshold. Act now.
"""

from enum import Enum


class ContextStatus(Enum):
    """Pressure level of a Conversation relative to its token budget."""
    OK                = "OK"
    WARNING           = "WARNING"
    COMPACTION_NEEDED = "COMPACTION_NEEDED"

    def __str__(self) -> str:
        return self.value
