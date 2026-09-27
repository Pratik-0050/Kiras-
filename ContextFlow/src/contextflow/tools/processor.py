# contextflow/tools/processor.py
"""
Step 21: Intelligent tool-result compaction.

ToolResultProcessor detects large tool outputs and reduces them before
they enter active context: errors always pass through untouched, small
results pass through untouched, and large outputs are reduced by
strategy (plain text, file listings, structured JSON). Reduction keeps
head/tail context plus important lines (errors, file paths, commands)
while collapsing redundant repeats and omitting the rest with markers.

The full original output is never lost: it stays in the processor's
in-memory archive (lookup by call ID) and travels in the tool message's
``full_output`` metadata, which persists to JSON via ContextStore for
later retrieval. Token counts always reflect the reduced text actually
sent to the model.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..conversation import Conversation
from ..message import Message
from .base import ToolResult

_VALID_STRATEGIES = frozenset({"text", "listing", "json"})

_DEFAULT_TOOL_STRATEGIES = {
    "list_directory": "listing",
    "read_file": "text",
}

_ERROR_RE = re.compile(
    r"(?i)\b(error|errors|failed|failure|exception|traceback|warning|"
    r"critical|fatal|denied|refused|invalid|missing|not found|timeout|"
    r"cannot|unable)\b"
)
_PATH_RE = re.compile(r"(^|[ \t\"'])(/[^\s:]*|\.{0,2}/[\w\-.]+|[A-Za-z]:\\[^\s]*)")
_COMMAND_RE = re.compile(r"^\s*[$>#%]\s*\S")


def _count_tokens(text: str) -> int:
    """Exact ContextFlow token count (same accounting as Message)."""
    return Message(role="tool", content=text).token_count


def _call_id_for(tool_name: str, output: str) -> str:
    digest = hashlib.sha256(f"{tool_name}\0{output}".encode("utf-8")).hexdigest()
    return digest[:16]


def _line_score(line: str) -> int:
    """Importance score: errors/paths weigh most, commands weigh some."""
    score = 0
    if _ERROR_RE.search(line):
        score += 2
    if _PATH_RE.search(line):
        score += 2
    if _COMMAND_RE.match(line):
        score += 1
    return score


def _collapse_repeats(lines: List[str]) -> Tuple[List[str], int]:
    """Collapse runs of 3+ identical lines into one + a marker.

    Returns (collapsed lines, number of lines removed).
    """
    collapsed: List[str] = []
    removed = 0
    i = 0
    while i < len(lines):
        run = 1
        while i + run < len(lines) and lines[i + run] == lines[i]:
            run += 1
        if run >= 3:
            collapsed.append(lines[i])
            collapsed.append(f"[... identical line repeated {run}x ...]")
            removed += run - 2
            i += run
        else:
            collapsed.append(lines[i])
            i += 1
    return collapsed, removed


@dataclass
class ProcessedToolResult:
    """A tool result paired with its reduced, context-ready form.

    Attributes:
        call_id:         Stable ID (content hash) for archive lookup.
        tool_name:       Tool that produced the result.
        strategy:        Strategy used ("text", "listing", "json",
                         "none" for untouched-small, "passthrough" for
                         errors passed through verbatim).
        was_reduced:     True when reduced text differs from the original.
        original:        Full original output text.
        reduced:         Context-ready text (== original when untouched).
        original_tokens: Token count of the original output.
        reduced_tokens:  Token count of the reduced text.
        original_lines:  Line count of the original output.
        reduced_lines:   Line count of the reduced text.
    """
    call_id: str
    tool_name: str
    strategy: str = "none"
    was_reduced: bool = False
    original: str = ""
    reduced: str = ""
    error: str = ""
    original_tokens: int = 0
    reduced_tokens: int = 0
    original_lines: int = 0
    reduced_lines: int = 0

    @property
    def saved_tokens(self) -> int:
        """Tokens kept out of active context."""
        return self.original_tokens - self.reduced_tokens

    def __str__(self) -> str:
        state = "reduced" if self.was_reduced else "untouched"
        return (
            f"[{self.tool_name}:{self.strategy} {state}] "
            f"{self.original_tokens}->{self.reduced_tokens} tokens "
            f"({self.original_lines}->{self.reduced_lines} lines)"
        )

    def to_message_content(self) -> str:
        """Full observation text stored as the tool message."""
        if self.error:
            name = self.tool_name or "?"
            return f"[Tool '{name}' error]\n{self.error}"
        return f"[Tool '{self.tool_name}' result]\n{self.reduced}"


class ToolResultProcessor:
    """Reduces large tool outputs for active context (deterministic, offline).

    Processing triggers when an output exceeds ANY threshold: max_chars,
    max_lines, or max_tokens. Failed results (ok=False) and small results
    always pass through untouched.

    Args:
        max_chars:        Character threshold triggering processing (default 2000).
        max_lines:        Line threshold triggering processing (default 50).
        max_tokens:       Token threshold triggering processing (default 500).
        keep_head_lines:  Leading lines always kept by text/listing
                          strategies (default 10).
        keep_tail_lines:  Trailing lines always kept (default 10).
        max_json_items:   List items kept per JSON array (default 20).
        max_json_depth:   Structure depth kept for JSON (default 3).
        max_string_chars: Longest single JSON string kept (default 300).
        tool_strategies:  Extra {tool_name: strategy} mappings merged over
                          the defaults ({"list_directory": "listing",
                          "read_file": "text"}). Values must be "text",
                          "listing", or "json".

    Usage::

        processor = ToolResultProcessor()
        processed = processor.process(result)   # ToolResult -> ProcessedToolResult
        processor.lookup(processed.call_id)     # full original, any time later
    """

    def __init__(
        self,
        max_chars: int = 2000,
        max_lines: int = 50,
        max_tokens: int = 500,
        keep_head_lines: int = 10,
        keep_tail_lines: int = 10,
        max_json_items: int = 20,
        max_json_depth: int = 3,
        max_string_chars: int = 300,
        tool_strategies: Optional[Dict[str, str]] = None,
    ) -> None:
        for label, value in (
            ("max_chars", max_chars),
            ("max_lines", max_lines),
            ("max_tokens", max_tokens),
            ("keep_head_lines", keep_head_lines),
            ("keep_tail_lines", keep_tail_lines),
            ("max_json_items", max_json_items),
            ("max_json_depth", max_json_depth),
            ("max_string_chars", max_string_chars),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{label} must be a positive integer, "
                    f"got {type(value).__name__!r}."
                )
            if value <= 0:
                raise ValueError(f"{label} must be a positive integer, got {value}.")
        strategies = dict(_DEFAULT_TOOL_STRATEGIES)
        for key, value in dict(tool_strategies or {}).items():
            if not isinstance(key, str):
                raise TypeError(
                    "tool_strategies keys must be tool-name strings, "
                    f"got {type(key).__name__!r}."
                )
            if value not in _VALID_STRATEGIES:
                raise ValueError(
                    f"tool_strategies[{key!r}] must be one of "
                    f"{sorted(_VALID_STRATEGIES)}, got {value!r}."
                )
            strategies[key] = value

        self.max_chars: int = max_chars
        self.max_lines: int = max_lines
        self.max_tokens: int = max_tokens
        self.keep_head_lines: int = keep_head_lines
        self.keep_tail_lines: int = keep_tail_lines
        self.max_json_items: int = max_json_items
        self.max_json_depth: int = max_json_depth
        self.max_string_chars: int = max_string_chars
        self.tool_strategies: Dict[str, str] = strategies
        self.archive: Dict[str, ProcessedToolResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        result: ToolResult,
        strategy: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> ProcessedToolResult:
        """Reduce *result* for active context and archive the original.

        Args:
            result:   ToolResult to process.
            strategy: "text", "listing", or "json" to force a strategy;
                      None resolves via tool_strategies, else JSON sniffing,
                      else text.
            call_id:  Archive key override (defaults to a content hash).

        Returns:
            ProcessedToolResult (also stored in self.archive).
        """
        if not isinstance(result, ToolResult):
            raise TypeError(
                f"result must be a ToolResult, got {type(result).__name__!r}."
            )
        if strategy is not None and strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_VALID_STRATEGIES)} or None, "
                f"got {strategy!r}."
            )
        output = result.output or ""
        original_tokens = _count_tokens(output)
        original_lines = len(output.splitlines())
        key = call_id or _call_id_for(result.tool_name, output)

        if not result.ok:
            processed = ProcessedToolResult(
                call_id=key, tool_name=result.tool_name, strategy="passthrough",
                was_reduced=False, original=output, reduced=output,
                error=result.error,
                original_tokens=original_tokens, reduced_tokens=original_tokens,
                original_lines=original_lines, reduced_lines=original_lines,
            )
            self.archive[key] = processed
            return processed

        if (len(output) <= self.max_chars
                and original_lines <= self.max_lines
                and original_tokens <= self.max_tokens):
            processed = ProcessedToolResult(
                call_id=key, tool_name=result.tool_name, strategy="none",
                was_reduced=False, original=output, reduced=output,
                error="",
                original_tokens=original_tokens, reduced_tokens=original_tokens,
                original_lines=original_lines, reduced_lines=original_lines,
            )
            self.archive[key] = processed
            return processed

        resolved = strategy or self._resolve_strategy(result.tool_name, output)
        if resolved == "listing":
            reduced = self._reduce_listing(output)
        elif resolved == "json":
            reduced, resolved = self._reduce_json(output)
        else:
            resolved = "text"
            reduced = self._reduce_text(output)

        reduced = self._enforce_char_cap(reduced)
        processed = ProcessedToolResult(
            call_id=key, tool_name=result.tool_name, strategy=resolved,
            was_reduced=reduced != output, original=output, reduced=reduced,
            error=result.error,
            original_tokens=original_tokens, reduced_tokens=_count_tokens(reduced),
            original_lines=original_lines, reduced_lines=len(reduced.splitlines()),
        )
        self.archive[key] = processed
        return processed

    def lookup(self, call_id: str) -> Optional[ProcessedToolResult]:
        """Return the archived processed result for *call_id*, if any."""
        return self.archive.get(call_id)

    def restore_from_conversation(self, conversation: Conversation) -> int:
        """Rebuild archive entries from persisted tool-message metadata.

        Tool messages carrying ``full_output`` metadata (written by the
        agent when a result was reduced) are re-indexed by call ID so the
        originals stay retrievable after a JSON store round-trip. Returns
        the number of entries restored.
        """
        if not isinstance(conversation, Conversation):
            raise TypeError(
                "conversation must be a Conversation instance, "
                f"got {type(conversation).__name__!r}."
            )
        restored = 0
        for message in conversation.get_messages():
            meta = message.metadata or {}
            if meta.get("type") != "tool_result" or "full_output" not in meta:
                continue
            original = meta["full_output"]
            if not isinstance(original, str):
                continue
            key = meta.get("call_id") or _call_id_for(str(meta.get("tool", "")), original)
            if key in self.archive:
                continue
            self.archive[key] = ProcessedToolResult(
                call_id=key,
                tool_name=str(meta.get("tool", "")),
                strategy="restored",
                was_reduced=True,
                original=original,
                reduced=message.content,
                error="",
                original_tokens=_count_tokens(original),
                reduced_tokens=message.token_count,
                original_lines=len(original.splitlines()),
                reduced_lines=len(message.content.splitlines()),
            )
            restored += 1
        return restored

    # ------------------------------------------------------------------
    # Strategy dispatch + implementations
    # ------------------------------------------------------------------

    def _resolve_strategy(self, tool_name: str, output: str) -> str:
        if tool_name in self.tool_strategies:
            return self.tool_strategies[tool_name]
        stripped = output.strip()
        if stripped[:1] in ("{", "["):
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                pass
            else:
                if isinstance(parsed, (dict, list)):
                    return "json"
        return "text"

    def _reduce_text(self, output: str) -> str:
        lines, _ = _collapse_repeats(output.splitlines())
        head, tail = self.keep_head_lines, self.keep_tail_lines
        if len(lines) <= head + tail:
            return "\n".join(lines)
        middle = lines[head: len(lines) - tail if tail else len(lines)]
        important = [line for line in middle if _line_score(line) >= 2]
        omitted = len(middle) - len(important)
        kept = lines[:head] + important + (lines[len(lines) - tail:] if tail else [])
        if omitted > 0:
            noun = "line" if omitted == 1 else "lines"
            kept = (
                lines[:head]
                + [f"[... {omitted} {noun} omitted ...]"]
                + important
                + (lines[len(lines) - tail:] if tail else [])
            )
        return "\n".join(kept)

    def _reduce_listing(self, output: str) -> str:
        entries = [line for line in output.splitlines() if line.strip()]
        if len(entries) <= self.max_lines:
            return "\n".join(entries)
        dirs = [e for e in entries if e.rstrip().endswith("/")]
        files = [e for e in entries if not e.rstrip().endswith("/")]
        budget = self.max_lines
        if len(dirs) >= budget:
            kept = entries[:self.keep_head_lines] + [f"[... {len(entries) - self.keep_head_lines} of {len(entries)} entries omitted ...]"] if self.keep_head_lines < len(entries) else entries
            return "\n".join(kept)
        room = budget - len(dirs)
        # Head gets the extra slot on odd budgets.
        head_count = min(len(files), (room + 1) // 2)
        tail_count = min(len(files) - head_count, room - head_count)
        omitted = len(files) - head_count - tail_count
        kept = list(dirs) + files[:head_count]
        if omitted > 0:
            kept.append(f"[... {omitted} of {len(entries)} entries omitted ...]")
        kept.extend(files[len(files) - tail_count:] if tail_count else [])
        return "\n".join(kept)

    def _reduce_json(self, output: str) -> Tuple[str, str]:
        """Prune JSON; returns (text, strategy_used). Falls back to text."""
        try:
            parsed = json.loads(output.strip())
        except (json.JSONDecodeError, ValueError):
            return self._reduce_text(output), "text"
        if not isinstance(parsed, (dict, list)):
            return self._reduce_text(output), "text"
        pruned = self._prune(parsed, depth=0)
        return json.dumps(pruned, indent=2, ensure_ascii=False), "json"

    def _prune(self, node: Any, depth: int) -> Any:
        if depth > self.max_json_depth:
            return "<... truncated>"
        if isinstance(node, dict):
            return {str(k): self._prune(v, depth + 1) for k, v in node.items()}
        if isinstance(node, list):
            kept = [self._prune(v, depth + 1) for v in node[: self.max_json_items]]
            if len(node) > self.max_json_items:
                kept.append(f"... ({len(node) - self.max_json_items} more items)")
            return kept
        if isinstance(node, str) and len(node) > self.max_string_chars:
            return node[: self.max_string_chars] + "...[truncated]"
        return node

    def _enforce_char_cap(self, text: str) -> str:
        """Last-resort truncation for pathological (e.g. single-line) outputs."""
        if len(text) <= self.max_chars:
            return text
        return (
            text[: self.max_chars]
            + f"\n[... output truncated to {self.max_chars} chars ...]"
        )
