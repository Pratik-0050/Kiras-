# contextflow/tools/filesystem.py
"""
Step 20: Safe example filesystem tools (no shell execution).

ReadFileTool and ListDirectoryTool operate inside a fixed root directory:
every requested path is resolved and must stay within the root (symlink
and `..` escapes rejected), missing paths and wrong types raise clear
ToolErrors, file reads are size-capped UTF-8, and listings are capped and
sorted. There is deliberately no shell tool in this step.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

from .base import Tool, ToolError, ToolResult

PathLike = Union[str, os.PathLike]

_DEFAULT_MAX_BYTES = 100_000
_DEFAULT_MAX_ENTRIES = 500


def _resolve_root(root: PathLike) -> Path:
    if not isinstance(root, (str, os.PathLike)):
        raise TypeError(
            f"root must be a directory path, got {type(root).__name__!r}."
        )
    resolved = Path(os.fspath(root)).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"root must be an existing directory, got {str(root)!r}.")
    return resolved


def _resolve_within_root(root: Path, rel: PathLike, *, what: str = "path") -> Path:
    """Resolve *rel* against *root*, rejecting escapes outside the root."""
    if not isinstance(rel, (str, os.PathLike)):
        raise ToolError(f"{what} must be a path string, got {type(rel).__name__!r}.")
    text = os.fspath(rel).strip()
    if not text:
        raise ToolError(f"{what} must not be empty.")
    candidate = (root / text).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ToolError(
            f"{what} {text!r} escapes the allowed root directory."
        ) from None
    return candidate


class ReadFileTool(Tool):
    """Read a UTF-8 text file inside the root directory.

    Arguments: path (relative path inside root, required).
    """

    name = "read_file"
    description = "Read a UTF-8 text file inside the allowed directory."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def __init__(self, root: PathLike, max_bytes: int = _DEFAULT_MAX_BYTES) -> None:
        self.root: Path = _resolve_root(root)
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
            raise TypeError(
                f"max_bytes must be a positive integer, got {type(max_bytes).__name__!r}."
            )
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be a positive integer, got {max_bytes}.")
        self.max_bytes: int = max_bytes

    def run(self, **kwargs: Any) -> ToolResult:
        """Read and return the file at kwargs['path'] (raises ToolError)."""
        path = kwargs.get("path")
        target = _resolve_within_root(self.root, path) if path is not None else None
        if target is None:
            raise ToolError("read_file requires a 'path' argument.")
        if not target.exists():
            raise ToolError(f"file not found: {path!r}.")
        if not target.is_file():
            raise ToolError(f"not a file: {path!r}.")
        size = target.stat().st_size
        if size > self.max_bytes:
            raise ToolError(
                f"file {path!r} is {size:,} bytes, over the "
                f"{self.max_bytes:,} byte limit."
            )
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"file {path!r} is not decodable as UTF-8 ({exc}).") from exc
        except OSError as exc:
            raise ToolError(f"cannot read file {path!r} ({exc}).") from exc
        return ToolResult(tool_name=self.name, ok=True, output=text)


class ListDirectoryTool(Tool):
    """List entries of a directory inside the root directory.

    Arguments: path (relative directory inside root, default ".").
    Directories are suffixed with "/"; entries are sorted by name.
    """

    name = "list_directory"
    description = "List files and subdirectories inside the allowed directory."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }

    def __init__(self, root: PathLike, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self.root: Path = _resolve_root(root)
        if not isinstance(max_entries, int) or isinstance(max_entries, bool):
            raise TypeError(
                "max_entries must be a positive integer, "
                f"got {type(max_entries).__name__!r}."
            )
        if max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, got {max_entries}."
            )
        self.max_entries: int = max_entries

    def run(self, **kwargs: Any) -> ToolResult:
        """List and return the directory at kwargs['path'] (raises ToolError)."""
        target = _resolve_within_root(
            self.root, kwargs.get("path", "."), what="directory")
        if not target.exists():
            raise ToolError(f"directory not found: {kwargs.get('path', '.')!r}.")
        if not target.is_dir():
            raise ToolError(f"not a directory: {kwargs.get('path', '.')!r}.")
        try:
            entries = sorted(target.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            raise ToolError(f"cannot list directory ({exc}).") from exc
        if len(entries) > self.max_entries:
            raise ToolError(
                f"directory holds {len(entries)} entries, over the "
                f"{self.max_entries} entry limit."
            )
        lines = [
            entry.name + ("/" if entry.is_dir() else "")
            for entry in entries
        ]
        return ToolResult(tool_name=self.name, ok=True, output="\n".join(lines))


def default_filesystem_tools(
    root: PathLike,
    extra: Optional[list] = None,
) -> list:
    """Build [ReadFileTool(root), ListDirectoryTool(root)] plus *extra* tools."""
    tools = [ReadFileTool(root), ListDirectoryTool(root)]
    tools.extend(extra or [])
    return tools
