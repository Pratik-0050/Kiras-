# contextflow/tools/__init__.py
"""
Tool subsystem for ContextFlow.

Provides the Tool interface, ToolRegistry, safe filesystem tools, and the
```tool fenced-JSON wire format the agent loop recognizes in model replies.
"""

from .base import (
    Tool,
    BaseTool,
    ToolCall,
    ToolCallRequest,
    ToolError,
    ToolRegistry,
    ToolResult,
    parse_tool_calls,
)
from .filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    default_filesystem_tools,
)
from .processor import (
    ProcessedToolResult,
    ToolResultProcessor,
)

__all__ = [
    "Tool",
    "BaseTool",
    "ToolCall",
    "ToolCallRequest",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "parse_tool_calls",
    "ListDirectoryTool",
    "ReadFileTool",
    "default_filesystem_tools",
    "ProcessedToolResult",
    "ToolResultProcessor",
]
