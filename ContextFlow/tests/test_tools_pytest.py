# tests/test_tools_pytest.py
"""Step 20 (pytest): basic tool calling.

Covers Tool/ToolRegistry registration and execution, the safe filesystem
tools (real files under tmp_path), the agent tool loop with a scripted
model (no network), invalid tools, errors, and context/token updates.
"""

import os

import pytest

from contextflow import (
    Agent,
    BaseTool,
    Conversation,
    ListDirectoryTool,
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    ReadFileTool,
    Tool,
    ToolCall,
    ToolError,
    ToolRegistry,
    ToolResult,
    default_filesystem_tools,
    parse_tool_calls,
)


class ScriptedClient(LLMClient):
    """Replies from a script; records every complete() call."""

    def __init__(self, replies, model="fake"):
        self.replies = list(replies)
        self.model = model
        self.calls = 0

    def complete(self, context=None, system=None, request=None,
                 temperature=None, max_tokens=None):
        self.calls += 1
        return LLMResponse(content=self.replies.pop(0), model=self.model,
                           input_tokens=30, output_tokens=10,
                           latency_seconds=0.01)


def _tool_block(name, arguments):
    import json
    return "Checking.\n```tool\n%s\n```" % json.dumps(
        {"name": name, "arguments": arguments})


def _write(root, name, content="content"):
    path = os.path.join(str(root), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


# ── wire format ──────────────────────────────────────────────────────


def test_parse_single_block():
    requests = parse_tool_calls(_tool_block("read_file", {"path": "a.txt"}))
    assert len(requests) == 1
    assert requests[0].name == "read_file"
    assert requests[0].arguments == {"path": "a.txt"}
    assert requests[0].error is None


def test_parse_multiple_blocks_in_order():
    text = (_tool_block("a", {}) + "\n" + _tool_block("b", {"x": 1}))
    assert [r.name for r in parse_tool_calls(text)] == ["a", "b"]


def test_parse_ignores_plain_text():
    assert parse_tool_calls("Just a normal reply.") == []


def test_parse_malformed_blocks_become_errors():
    requests = parse_tool_calls("```tool\n{not json\n```")
    assert len(requests) == 1
    assert requests[0].error is not None
    requests = parse_tool_calls('```tool\n{"arguments": {}}\n```')
    assert requests[0].error is not None
    requests = parse_tool_calls('```tool\n{"name": "x", "arguments": [1]}\n```')
    assert requests[0].error is not None
    with pytest.raises(TypeError):
        parse_tool_calls(123)  # type: ignore


# ── Tool interface + registry ────────────────────────────────────────


def test_tool_is_abstract():
    class Incomplete(Tool):
        pass
    with pytest.raises(TypeError):
        Incomplete()  # type: ignore
    assert BaseTool is Tool


def test_registry_register_get_list_describe():
    registry = ToolRegistry()

    class Echo(Tool):
        name = "echo"
        description = "Echoes."
        parameters = {"type": "object"}

        def run(self, **kwargs):
            return ToolResult(tool_name=self.name, ok=True, output="hi")

    registry.register(Echo())
    assert registry.names == ["echo"]
    assert len(registry) == 1
    assert isinstance(registry.get("echo"), Echo)
    assert registry.get("missing") is None
    assert "echo" in registry.describe()


def test_registry_rejects_bad_registration():
    registry = ToolRegistry()
    with pytest.raises(TypeError, match="Tool instance"):
        registry.register("nope")  # type: ignore

    class Nameless(Tool):
        def run(self, **kwargs):
            return ToolResult(tool_name="", ok=True)

    with pytest.raises(ValueError, match="non-empty name"):
        registry.register(Nameless())

    class Dup(Tool):
        name = "dup"
        description = "x"

        def run(self, **kwargs):
            return ToolResult(tool_name=self.name, ok=True)

    registry.register(Dup())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Dup())


def test_registry_unregister():
    registry = ToolRegistry()

    class Temp(Tool):
        name = "temp"
        description = "x"

        def run(self, **kwargs):
            return ToolResult(tool_name=self.name, ok=True)

    registry.register(Temp())
    registry.unregister("temp")
    assert registry.names == []
    with pytest.raises(KeyError, match="unknown tool"):
        registry.unregister("temp")


def test_execute_unknown_tool_returns_failed_result():
    registry = ToolRegistry()
    result = registry.execute("ghost", {})
    assert result.ok is False
    assert "unknown tool" in result.error


def test_execute_converts_tool_errors_to_results():
    class Boom(Tool):
        name = "boom"
        description = "x"

        def run(self, **kwargs):
            raise ToolError("kaput")

    class Wild(Tool):
        name = "wild"
        description = "x"

        def run(self, **kwargs):
            raise RuntimeError("unexpected")

    class BadReturn(Tool):
        name = "bad"
        description = "x"

        def run(self, **kwargs):
            return "not a result"  # type: ignore

    registry = ToolRegistry([Boom(), Wild(), BadReturn()])
    assert registry.execute("boom", {}).error == "kaput"
    assert "RuntimeError" in registry.execute("wild", {}).error
    assert "ToolResult" in registry.execute("bad", {}).error
    with pytest.raises(TypeError, match="must be a dict"):
        registry.execute("boom", ["nope"])  # type: ignore


def test_tool_result_message_content():
    ok = ToolResult(tool_name="t", ok=True, output="out")
    assert "result" in ok.to_message_content()
    failed = ToolResult(tool_name="t", ok=False, error="bad")
    assert "bad" in failed.to_message_content()


# ── filesystem tools (real files under tmp_path) ─────────────────────


def test_read_file_round_trip(tmp_path):
    _write(tmp_path, "notes.txt", "Hello file.")
    tool = ReadFileTool(tmp_path)
    result = tool.run(path="notes.txt")
    assert result.ok is True
    assert result.output == "Hello file."
    assert result.tool_name == "read_file"


def test_read_file_rejects_escape_and_missing(tmp_path):
    _write(tmp_path, "ok.txt", "x")
    tool = ReadFileTool(tmp_path)
    with pytest.raises(ToolError, match="requires a 'path'"):
        tool.run()
    with pytest.raises(ToolError, match="escapes"):
        tool.run(path="../outside.txt")
    with pytest.raises(ToolError, match="escapes"):
        tool.run(path="/etc/passwd")
    with pytest.raises(ToolError, match="not found"):
        tool.run(path="missing.txt")
    sub = os.path.join(str(tmp_path), "sub")
    os.mkdir(sub)
    with pytest.raises(ToolError, match="not a file"):
        tool.run(path="sub")


def test_read_file_rejects_bad_root_and_limits(tmp_path):
    with pytest.raises(ValueError, match="existing directory"):
        ReadFileTool(str(tmp_path / "nope"))
    with pytest.raises(TypeError, match="directory path"):
        ReadFileTool(123)  # type: ignore
    _write(tmp_path, "big.txt", "x" * 100)
    with pytest.raises(ToolError, match="over the"):
        ReadFileTool(tmp_path, max_bytes=10).run(path="big.txt")


def test_list_directory_marks_dirs_sorted(tmp_path):
    _write(tmp_path, "b.txt", "x")
    _write(tmp_path, "a.txt", "x")
    os.mkdir(os.path.join(str(tmp_path), "sub"))
    result = ListDirectoryTool(tmp_path).run(path=".")
    assert result.ok is True
    assert result.output.split("\n") == ["a.txt", "b.txt", "sub/"]


def test_list_directory_rejects_bad_paths(tmp_path):
    tool = ListDirectoryTool(tmp_path)
    with pytest.raises(ToolError, match="not found"):
        tool.run(path="nope")
    with pytest.raises(ToolError, match="escapes"):
        tool.run(path="..")
    _write(tmp_path, "f.txt", "x")
    with pytest.raises(ToolError, match="not a directory"):
        tool.run(path="f.txt")


def test_default_filesystem_tools(tmp_path):
    tools = default_filesystem_tools(tmp_path)
    assert [t.name for t in tools] == ["read_file", "list_directory"]
    registry = ToolRegistry(tools)
    assert len(registry) == 2


# ── agent tool loop ──────────────────────────────────────────────────


def test_agent_executes_tool_and_continues(tmp_path):
    _write(tmp_path, "notes.txt", "Deploy on Fridays is forbidden.")
    replies = [
        _tool_block("read_file", {"path": "notes.txt"}),
        "Per notes: never deploy on Fridays.",
    ]
    conv = Conversation(name="T", max_tokens=8000)
    agent = Agent(llm=ScriptedClient(replies), conversation=conv,
                  tools=[ReadFileTool(tmp_path)])
    result = agent.run("What do the notes say?")

    assert result.response == "Per notes: never deploy on Fridays."
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.name == "read_file"
    assert call.arguments == {"path": "notes.txt"}
    assert call.result.ok is True
    assert "forbidden" in call.result.output
    assert result.tool_rounds == 1
    assert result.tools_truncated is False
    roles = [m.role for m in agent.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_msg = agent.history[2]
    assert "Deploy on Fridays is forbidden." in tool_msg.content
    assert tool_msg.metadata["tool"] == "read_file"
    # Two model calls: initial + follow-up after the observation.
    assert agent.llm.calls == 2


def test_unknown_tool_error_fed_back_to_model(tmp_path):
    replies = [
        _tool_block("ghost", {}),
        "Understood, no such tool.",
    ]
    agent = Agent(llm=ScriptedClient(replies),
                  conversation=Conversation(name="T", max_tokens=8000),
                  tools=[ReadFileTool(tmp_path)])
    result = agent.run("Do the ghost thing.")
    assert result.tool_calls[0].result.ok is False
    assert "unknown tool" in result.tool_calls[0].result.error
    assert result.response == "Understood, no such tool."
    assert agent.history[2].role == "tool"


def test_tool_error_observation_recorded(tmp_path):
    replies = [_tool_block("read_file", {"path": "missing.txt"}), "Noted."]
    agent = Agent(llm=ScriptedClient(replies),
                  conversation=Conversation(name="T", max_tokens=8000),
                  tools=[ReadFileTool(tmp_path)])
    result = agent.run("Read it.")
    assert result.tool_calls[0].result.ok is False
    assert "not found" in agent.history[2].content


def test_max_tool_rounds_truncates_loop():
    block = _tool_block("read_file", {"path": "x"})
    # Bypass constructor validation with a stub tool instead:
    from contextflow import ToolRegistry as Registry

    class Stub(Tool):
        name = "read_file"
        description = "stub"

        def run(self, **kwargs):
            return ToolResult(tool_name=self.name, ok=True, output="data")

    agent = Agent(llm=ScriptedClient([block, block, "Final answer."]),
                  conversation=Conversation(name="T", max_tokens=8000),
                  tools=Registry([Stub()]),
                  max_tool_rounds=1)
    result = agent.run("Go.")
    assert result.tools_truncated is True
    assert result.tool_rounds == 1
    assert len(result.tool_calls) == 1
    # Truncation stops the loop: the unfulfilled tool request is the reply,
    # and the third scripted answer is never consumed.
    assert result.response == block
    assert agent.llm.calls == 2


def test_no_tools_registered_treats_blocks_as_text():
    block = _tool_block("read_file", {"path": "x"})
    agent = Agent(llm=ScriptedClient([block]),
                  conversation=Conversation(name="T", max_tokens=8000))
    result = agent.run("Go.")
    assert result.tool_calls == []
    assert result.response == block
    assert [m.role for m in agent.history] == ["user", "assistant"]


def test_zero_max_tool_rounds_disables_execution():
    from contextflow import ToolRegistry as Registry

    class Stub(Tool):
        name = "read_file"
        description = "stub"

        def run(self, **kwargs):
            return ToolResult(tool_name=self.name, ok=True, output="data")

    block = _tool_block("read_file", {"path": "x"})
    agent = Agent(llm=ScriptedClient([block]),
                  conversation=Conversation(name="T", max_tokens=8000),
                  tools=Registry([Stub()]),
                  max_tool_rounds=0)
    result = agent.run("Go.")
    assert result.tool_calls == []
    assert result.tools_truncated is False
    assert result.response == block
    assert [m.role for m in agent.history] == ["user", "assistant"]


def test_malformed_tool_block_reported():
    bad = "Hmm.\n```tool\n{not json\n```\nDone."
    agent = Agent(llm=ScriptedClient([bad, "OK then."]),
                  conversation=Conversation(name="T", max_tokens=8000),
                  tools=[])
    # Empty registry: blocks are plain text, no execution attempted.
    result = agent.run("Go.")
    assert result.tool_calls == []
    assert result.response == bad


def test_tool_results_count_toward_tokens_and_compaction(tmp_path):
    _write(tmp_path, "notes.txt", "Word " * 200)
    replies = [_tool_block("read_file", {"path": "notes.txt"}), "Summarized."]
    conv = Conversation(name="Tight", max_tokens=60)
    conv.add(Message(role="system", content="You are helpful."))
    for i in range(6):
        conv.add(Message(role="user", content="Filler turn %d here." % i))
    agent = Agent(llm=ScriptedClient(replies), conversation=conv,
                  tools=[ReadFileTool(tmp_path)])
    before = conv.total_tokens()
    result = agent.run("Read the notes.")

    tool_msg = next(m for m in agent.history if m.role == "tool")
    assert tool_msg.token_count > 0
    assert result.prompt_tokens > before  # second assembly includes the observation
    assert result.compaction_occurred is True  # large observation trips the trigger
    assert result.pipeline_result.triggered is True


def test_model_failure_mid_loop_leaves_history_untouched(tmp_path):
    class Flaky(ScriptedClient):
        def complete(self, context=None, system=None, request=None,
                     temperature=None, max_tokens=None):
            self.calls += 1
            if self.calls == 1:
                return super().complete(context, system, request)
            raise LLMError("Died on follow-up.")

    _write(tmp_path, "a.txt", "data")
    agent = Agent(llm=Flaky([_tool_block("read_file", {"path": "a.txt"})]),
                  conversation=Conversation(name="T", max_tokens=8000),
                  tools=[ReadFileTool(tmp_path)])
    before = [m.content for m in agent.history]
    with pytest.raises(LLMError, match="Died on follow-up"):
        agent.run("Go.")
    assert [m.content for m in agent.history] == before


def test_agent_tool_registration_and_validation(tmp_path):
    agent = Agent(llm=ScriptedClient(["hi"]),
                  conversation=Conversation(name="T", max_tokens=8000))
    assert agent.tools.names == []
    agent.register_tool(ReadFileTool(tmp_path))
    assert agent.tools.names == ["read_file"]
    with pytest.raises(ValueError, match="already registered"):
        agent.register_tool(ReadFileTool(tmp_path))
    with pytest.raises(TypeError, match="ToolRegistry"):
        Agent(llm=ScriptedClient(["hi"]), tools="nope")  # type: ignore
    with pytest.raises(ValueError, match="non-negative integer"):
        Agent(llm=ScriptedClient(["hi"]), max_tool_rounds=-1)
    with pytest.raises(TypeError, match="non-negative integer"):
        Agent(llm=ScriptedClient(["hi"]), max_tool_rounds=True)  # type: ignore


def test_shared_registry_used_directly(tmp_path):
    from contextflow import ToolRegistry as Registry
    registry = Registry([ListDirectoryTool(tmp_path)])
    agent = Agent(llm=ScriptedClient(["done"]),
                  conversation=Conversation(name="T", max_tokens=8000),
                  tools=registry)
    assert agent.tools is registry
    assert agent.run("Hi?").response == "done"
