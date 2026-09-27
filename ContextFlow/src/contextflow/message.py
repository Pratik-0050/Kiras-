# contextflow/message.py
"""
Defines the Message class used throughout ContextFlow.
Each message represents a single turn in a conversation.
Token counts are computed automatically via tiktoken.
"""

import tiktoken
from enum import Enum
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Shared encoder -- created once and reused for every Message instance.
# cl100k_base is the encoding used by GPT-3.5-turbo and GPT-4.
# ---------------------------------------------------------------------------
_ENCODER = tiktoken.get_encoding("cl100k_base")

# The set of roles that are valid in a chat-style conversation.
VALID_ROLES = {"system", "user", "assistant", "tool"}


class ImportanceLevel(str, Enum):
    """Importance level of a message for context management and compaction.

    Levels:
        CRITICAL    -- Never removed or summarized; always preserved verbatim.
        IMPORTANT   -- High priority; preserved or summarized with high fidelity.
        NORMAL      -- Default; subject to summarization when in older context.
        DISCARDABLE -- Ephemeral / low value; dropped first during compaction.
    """
    CRITICAL    = "critical"
    IMPORTANT   = "important"
    NORMAL      = "normal"
    DISCARDABLE = "discardable"


VALID_IMPORTANCE = {level.value for level in ImportanceLevel}


class Message:
    """A single message in a conversation.

    Attributes:
        role        -- Who sent the message (system / user / assistant / tool).
        content     -- The text of the message.
        token_count -- Number of tokens in content, computed automatically.
        metadata    -- Optional dict for extra info (e.g. timestamps, IDs).
        importance  -- ImportanceLevel (critical, important, normal, discardable).
        protected   -- If True, compaction will never remove this message.
    """

    def __init__(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
        importance: Union[ImportanceLevel, str] = ImportanceLevel.NORMAL,
        protected: bool = False,
    ) -> None:
        # Validate role before storing anything.
        if role not in VALID_ROLES:
            raise ValueError(
                f"Invalid role '{role}'. "
                f"Allowed roles: {sorted(VALID_ROLES)}"
            )

        # Validate importance
        if isinstance(importance, str):
            val = importance.lower()
            if val not in VALID_IMPORTANCE:
                raise ValueError(
                    f"Invalid importance '{importance}'. "
                    f"Allowed importance levels: {sorted(VALID_IMPORTANCE)}"
                )
            self.importance: ImportanceLevel = ImportanceLevel(val)
        elif isinstance(importance, ImportanceLevel):
            self.importance = importance
        else:
            raise TypeError(
                f"importance must be an ImportanceLevel or str, "
                f"got {type(importance).__name__!r}."
            )

        # Validate protected
        if not isinstance(protected, bool):
            raise TypeError(
                f"protected must be a bool, got {type(protected).__name__!r}."
            )

        # Validate content and metadata before counting tokens.
        if not isinstance(content, str):
            raise TypeError(
                f"content must be a string, got {type(content).__name__!r}."
            )
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be a dict or None, got {type(metadata).__name__!r}."
            )

        self.role: str = role
        self.content: str = content
        self.token_count: int = self._count_tokens(content)
        self.metadata: dict = metadata or {}
        self.protected: bool = protected

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Encode *text* and return the token count."""
        return len(_ENCODER.encode(text))

    # ------------------------------------------------------------------
    # Nice string representations
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        flags = []
        if self.importance != ImportanceLevel.NORMAL:
            flags.append(f"importance={self.importance.value!r}")
        if self.protected:
            flags.append("protected=True")
        flag_str = f", {', '.join(flags)}" if flags else ""
        return (
            f"Message(role={self.role!r}, "
            f"tokens={self.token_count}{flag_str}, "
            f"content={self.content[:40]!r}{'...' if len(self.content) > 40 else ''})"
        )

    def __str__(self) -> str:
        sep = "-" * 50
        lines = [
            sep,
            f"Role       : {self.role}",
            f"Tokens     : {self.token_count}",
        ]
        if self.importance != ImportanceLevel.NORMAL:
            lines.append(f"Importance : {self.importance.value}")
        if self.protected:
            lines.append(f"Protected  : {self.protected}")
        lines.append(f"Content    : {self.content}")
        lines.append(sep)
        return "\n".join(lines)
