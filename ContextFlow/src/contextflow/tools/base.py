# contextflow/tools/base.py
"""
Step 20: Tool interface, results, registry, and the tool-call wire format.

A Tool is a named, described callable the agent can invoke mid-turn. Tools
run locally with validated arguments and return a ToolResult; failures
(including unknown tool names) become failed results, never exceptions, so
the agent loop can feed the outcome back to the model deterministically.

The model requests a tool with a fenced JSON block (provider-independent --
no function-calling API required)::

    ```tool
    {"name": "read_file", "arguments": {"path": "notes.txt"}}
    ```

No shell execution exists in this step: only the safe, path-validated
tools in filesystem.py are provided.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# A model requests tools with ```tool fenced JSON blocks (DOTALL so the
# JSON may span lines). Anything outside a block is plain reply text.
_TOOL_BLOCK_RE = re.compile(r"```tool\s*\n?(.*?)\n?```", re.DOTALL)


class ToolError(Exception):
    """Raised for tool problems: bad registration, bad arguments, bad paths.

    Execution-time failures are converted to failed ToolResults by
    ToolRegistry.execute(); only programming errors (wrong types,
    duplicates) propagate as-is from the registry methods.
    """


@dataclass
class ToolResult:
    """Outcome of one tool execution.

    Attributes:
        tool_name: Name of the tool that ran ("" when unparsable).
        ok:        True when the tool succeeded.
        output:    Tool output text (meaningful when ok is True).
        error:     Error description (meaningful when ok is False).
    """
    tool_name: str
    ok: bool
    output: str = ""
    error: str = ""

    def __str__(self) -> str:
        if self.ok:
            preview = self.output[:100] + ("..." if len(self.output) > 100 else "")
            return f"[Tool '{self.tool_name}' result]\n{preview}"
        return f"[Tool '{self.tool_name or '?'}' error]\n{self.error}"

    def to_message_content(self) -> str:
        """Full text stored as the tool observation message."""
        if self.ok:
            return f"[Tool '{self.tool_name}' result]\n{self.output}"
        name = self.tool_name or "?"
        return f"[Tool '{name}' error]\n{self.error}"


@dataclass
class ToolCall:
    """One parsed tool request plus its execution result."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[ToolResult] = None

    def __str__(self) -> str:
        status = "ok" if self.result is None or self.result.ok else "FAILED"
        return f"{self.name}({self.arguments}) -> {status}"


@dataclass
class ToolCallRequest:
    """A raw parsed ```tool block (before execution)."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    error: Optional[str] = None


def parse_tool_calls(text: str) -> List[ToolCallRequest]:
    """Extract ```tool JSON blocks from model reply *text*, in order.

    Malformed blocks (bad JSON, non-object, missing name, non-object
    arguments) become requests carrying an error instead of raising, so
    the agent can report them back deterministically.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__!r}.")
    requests: List[ToolCallRequest] = []
    for match in _TOOL_BLOCK_RE.finditer(text):
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            requests.append(ToolCallRequest(
                name="", arguments={}, raw=raw,
                error=f"invalid tool block JSON: {exc}"))
            continue
        if not isinstance(payload, dict):
            requests.append(ToolCallRequest(
                name="", arguments={}, raw=raw,
                error="tool block must be a JSON object with 'name' and 'arguments'"))
            continue
        name = payload.get("name", "")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            requests.append(ToolCallRequest(
                name="", arguments={}, raw=raw,
                error="tool block is missing a non-empty string 'name'"))
            continue
        if not isinstance(arguments, dict):
            requests.append(ToolCallRequest(
                name=name.strip(), arguments={}, raw=raw,
                error="tool block 'arguments' must be a JSON object"))
            continue
        requests.append(ToolCallRequest(
            name=name.strip(), arguments=dict(arguments), raw=raw))
    return requests


class Tool(ABC):
    """Abstract interface that every tool must implement.

    Attributes (class-level recommended):
        name:        Unique registry key, e.g. "read_file".
        description: One-line summary shown in tool catalogs.
        parameters: JSON-schema-style argument docs (informational only;
                    validation lives in run()).
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool; return a ToolResult (raise ToolError on failure)."""
        pass


# BaseTool is an alias kept for naming consistency with the other
# subsystems (BaseSummarizer, BaseValidator, BaseRetriever, ...).
BaseTool = Tool


class ToolRegistry:
    """Named collection of tools with execution and catalog support.

    Usage::

        registry = ToolRegistry([ReadFileTool(root), ListDirectoryTool(root)])
        result = registry.execute("read_file", {"path": "notes.txt"})
        print(registry.describe())  # prompt-ready catalog for the model
    """

    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add *tool* (raises TypeError for non-tools, ValueError on dupes)."""
        if not isinstance(tool, Tool):
            raise TypeError(
                f"tool must be a Tool instance, got {type(tool).__name__!r}."
            )
        if not tool.name or not tool.name.strip():
            raise ValueError("tool must define a non-empty name.")
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove *name* (raises KeyError when unknown)."""
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}.")
        del self._tools[name]

    def get(self, name: str) -> Optional[Tool]:
        """Return the tool registered as *name*, or None."""
        return self._tools.get(name)

    @property
    def names(self) -> List[str]:
        """Registered tool names in registration order."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def describe(self) -> str:
        """Prompt-ready catalog listing each tool and its parameters."""
        if not self._tools:
            return "No tools available."
        lines = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            if tool.parameters:
                lines.append(f"  arguments: {json.dumps(tool.parameters)}")
        return "\n".join(lines)

    def execute(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Run tool *name* with *arguments*, never raising for tool problems.

        Unknown tool names and tool-raised errors become failed ToolResults
        (ok=False) so the agent loop can report them back to the model.
        Non-dict *arguments* is a programming error and raises TypeError.
        """
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise TypeError(
                f"arguments must be a dict, got {type(arguments).__name__!r}."
            )
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(self._tools) or "(none registered)"
            return ToolResult(
                tool_name=name, ok=False,
                error=f"unknown tool {name!r}. Known tools: {known}.")
        try:
            result = tool.run(**arguments)
        except ToolError as exc:
            return ToolResult(tool_name=name, ok=False, error=str(exc))
        except Exception as exc:  # defensive: tools must not crash the loop
            return ToolResult(
                tool_name=name, ok=False,
                error=f"tool {name!r} raised {type(exc).__name__}: {exc}")
        if not isinstance(result, ToolResult):
            return ToolResult(
                tool_name=name, ok=False,
                error=f"tool {name!r} returned {type(result).__name__!r}, "
                      "expected ToolResult.")
        if not result.tool_name:
            result.tool_name = name
        return result
