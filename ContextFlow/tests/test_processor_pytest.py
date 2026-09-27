# tests/test_processor_pytest.py
"""Step 21 (pytest): intelligent tool-result compaction.

Covers threshold triggering (small untouched, large reduced), error
passthrough, text strategy (important lines kept, repeats collapsed),
file-listing strategy, JSON strategy, archive lookup + restore, agent
integration with token accounting, and constructor validation. No
network; filesystem tests use tmp_path.
"""

import json
import os

import pytest

from contextflow import (
    Agent,
    ContextStore,
    Conversation,
    ListDirectoryTool,
    LLMClient,
    LLMResponse,
    Message,
    ReadFileTool,
    ToolResult,
    ToolResultProcessor,
)


class ScriptedClient(LLMClient):
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, context=None, system=None, request=None,
                 temperature=None, max_tokens=None):
        return LLMResponse(content=self.replies.pop(0), model="fake",
                           input_tokens=30, output_tokens=10,
                           latency_seconds=0.01)


def _ok_result(output, tool_name="read_file"):
    return ToolResult(tool_name=tool_name, ok=True, output=output)


def _big_log():
    lines = ["Starting batch job."]
    lines += ["heartbeat ok"] * 8  # redundant repeats
    lines += ["Processing /var/data/input_014.csv ..."]
    lines += ["filler detail line %d" % i for i in range(40)]
    lines += ["ERROR: connection refused on db-primary:5432"]
    lines += ["more trailing filler %d" % i for i in range(10)]
    lines += ["Run finished."]
    return "\n".join(lines)


# ── triggering ───────────────────────────────────────────────────────


def test_small_result_passes_through_untouched():
    processed = ToolResultProcessor().process(_ok_result("Hello file."))
    assert processed.was_reduced is False
    assert processed.strategy == "none"
    assert processed.reduced == "Hello file."
    assert processed.saved_tokens == 0


def test_thresholds_trigger_processing():
    text = "Word " * 600  # over default char/token thresholds
    assert ToolResultProcessor().process(_ok_result(text)).was_reduced is True
    assert ToolResultProcessor(max_chars=10**9, max_tokens=10**9,
                               max_lines=10**9).process(
        _ok_result(text)).was_reduced is False
    many_lines = "\n".join("line %d" % i for i in range(60))
    assert ToolResultProcessor().process(_ok_result(many_lines)).was_reduced is True


def test_failed_results_are_never_reduced():
    big_error = ToolResult(tool_name="read_file", ok=False,
                           error="E: " + "boom " * 500)
    processed = ToolResultProcessor().process(big_error)
    assert processed.was_reduced is False
    assert processed.strategy == "passthrough"
    assert "boom" in processed.reduced or processed.reduced == ""


def test_failed_result_message_keeps_error_text():
    failed = ToolResult(tool_name="read_file", ok=False, error="file not found: x")
    processed = ToolResultProcessor().process(failed)
    assert "file not found" in processed.to_message_content()


def test_invalid_process_inputs_rejected():
    processor = ToolResultProcessor()
    with pytest.raises(TypeError, match="ToolResult"):
        processor.process("nope")  # type: ignore
    with pytest.raises(ValueError, match="strategy must be"):
        processor.process(_ok_result("x" * 5000), strategy="weird")


def test_invalid_constructor_arguments_rejected():
    with pytest.raises(TypeError, match="max_chars"):
        ToolResultProcessor(max_chars="lots")  # type: ignore
    with pytest.raises(ValueError, match="max_lines"):
        ToolResultProcessor(max_lines=0)
    with pytest.raises(ValueError, match="tool_strategies"):
        ToolResultProcessor(tool_strategies={"read_file": "magic"})
    with pytest.raises(TypeError, match="tool-name strings"):
        ToolResultProcessor(tool_strategies={123: "text"})  # type: ignore


# ── text strategy ────────────────────────────────────────────────────


def test_text_keeps_head_tail_and_important_lines():
    processed = ToolResultProcessor().process(_ok_result(_big_log()))
    assert processed.was_reduced is True
    assert processed.strategy == "text"
    assert "Starting batch job." in processed.reduced  # head kept
    assert "Run finished." in processed.reduced  # tail kept
    assert "ERROR: connection refused on db-primary:5432" in processed.reduced
    assert "/var/data/input_014.csv" in processed.reduced  # path kept
    assert "omitted" in processed.reduced  # marker present
    assert "filler detail line 20" not in processed.reduced  # bulk dropped
    assert processed.reduced_tokens < processed.original_tokens
    assert "filler detail" in processed.original  # original intact


def test_repeated_lines_collapsed():
    text = "\n".join(["ping"] * 6 + ["unique line %d" % i for i in range(60)])
    processed = ToolResultProcessor().process(_ok_result(text))
    assert processed.was_reduced is True
    assert "repeated 6x" in processed.reduced
    assert processed.reduced.count("ping") == 1  # run collapsed to one line


def test_explicit_strategy_override():
    text = '{"a": 1, "b": 2}' + "x" * 3000
    assert ToolResultProcessor().process(
        _ok_result(text), strategy="text").strategy == "text"


# ── listing strategy ─────────────────────────────────────────────────


def test_listing_keeps_dirs_and_counts_omitted(tmp_path):
    for i in range(80):
        with open(os.path.join(str(tmp_path), "file_%02d.txt" % i), "w") as fh:
            fh.write("x")
    os.mkdir(os.path.join(str(tmp_path), "src"))
    tool = ListDirectoryTool(tmp_path)
    output = tool.run(path=".").output
    processed = ToolResultProcessor().process(
        ToolResult(tool_name="list_directory", ok=True, output=output))
    assert processed.strategy == "listing"
    assert processed.was_reduced is True
    assert "src/" in processed.reduced  # directories always kept
    assert "entries omitted" in processed.reduced
    assert "file_00.txt" in processed.reduced  # head kept
    assert "file_79.txt" in processed.reduced  # tail kept
    assert "file_40.txt" not in processed.reduced  # middle dropped


def test_small_listing_untouched(tmp_path):
    with open(os.path.join(str(tmp_path), "a.txt"), "w") as fh:
        fh.write("x")
    output = ListDirectoryTool(tmp_path).run(path=".").output
    processed = ToolResultProcessor().process(
        ToolResult(tool_name="list_directory", ok=True, output=output))
    assert processed.was_reduced is False
    assert processed.reduced == output


# ── JSON strategy ────────────────────────────────────────────────────


def test_json_arrays_capped_with_marker():
    payload = json.dumps({"items": [{"id": i} for i in range(50)]})
    # Trigger via tokens only, so the pruned JSON (with its marker) fits.
    processor = ToolResultProcessor(max_chars=10**9, max_lines=10**9, max_tokens=50)
    processed = processor.process(
        ToolResult(tool_name="api", ok=True, output=payload))
    assert processed.strategy == "json"
    assert processed.was_reduced is True
    assert "more items" in processed.reduced
    assert processed.reduced_tokens < processed.original_tokens
    # Original parses back exactly (nothing lost).
    assert json.loads(processed.original)["items"][49] == {"id": 49}


def test_json_depth_and_long_strings_truncated():
    payload = json.dumps({
        "nested": {"deeper": {"deepest": {"too": "deep"}}},
        "blob": "z" * 1000,
    })
    processor = ToolResultProcessor(max_chars=200, max_lines=5, max_tokens=50)
    processed = processor.process(
        ToolResult(tool_name="api", ok=True, output=payload))
    assert processed.strategy == "json"
    assert "truncated" in processed.reduced


def test_json_like_output_auto_detected():
    payload = json.dumps({"k": list(range(100))})
    processor = ToolResultProcessor(max_chars=200, max_lines=5, max_tokens=50)
    assert processor.process(
        _ok_result(payload, tool_name="custom")).strategy == "json"


def test_invalid_json_falls_back_to_text():
    text = "{not valid json" + "y" * 3000
    processed = ToolResultProcessor().process(
        _ok_result(text, tool_name="custom"), strategy="json")
    assert processed.strategy == "text"
    assert processed.was_reduced is True


# ── archive + persistence ────────────────────────────────────────────


def test_archive_lookup_returns_original():
    processor = ToolResultProcessor()
    processed = processor.process(_ok_result(_big_log()))
    found = processor.lookup(processed.call_id)
    assert found is processed
    assert "filler detail line 20" in found.original
    assert processor.lookup("missing") is None
    assert "reduced" in str(processed)


def test_call_ids_stable_for_same_content():
    processor = ToolResultProcessor()
    first = processor.process(_ok_result("same text here " * 200))
    second = processor.process(_ok_result("same text here " * 200))
    assert first.call_id == second.call_id


def test_restore_from_persisted_conversation(tmp_path):
    conv = Conversation(name="T", max_tokens=8000)
    conv.add(Message(role="system", content="sys"))
    processor = ToolResultProcessor()
    processed = processor.process(_ok_result(_big_log()))
    conv.add(Message(
        role="tool",
        content=processed.to_message_content(),
        metadata={"type": "tool_result", "tool": "read_file", "ok": True,
                  "call_id": processed.call_id, "reduced": True,
                  "full_output": processed.original},
    ))
    path = str(tmp_path / "tools.json")
    ContextStore(conv).save(path)

    fresh = ToolResultProcessor()
    restored = fresh.restore_from_conversation(ContextStore.load_file(path).conversation)
    assert restored == 1
    found = fresh.lookup(processed.call_id)
    assert found is not None
    assert "filler detail line 20" in found.original
    with pytest.raises(TypeError, match="Conversation"):
        fresh.restore_from_conversation("nope")  # type: ignore


# ── agent integration ────────────────────────────────────────────────


def _block(name, arguments):
    return "Checking.\n```tool\n%s\n```" % json.dumps(
        {"name": name, "arguments": arguments})


def test_agent_reduces_large_tool_output(tmp_path):
    with open(os.path.join(str(tmp_path), "big.log"), "w") as fh:
        fh.write(_big_log())
    replies = [_block("read_file", {"path": "big.log"}), "Summarized."]
    conv = Conversation(name="T", max_tokens=8000)
    agent = Agent(llm=ScriptedClient(replies), conversation=conv,
                  tools=[ReadFileTool(tmp_path)])
    result = agent.run("Read the big log.")

    tool_msg = next(m for m in agent.history if m.role == "tool")
    assert tool_msg.metadata["reduced"] is True
    assert "omitted" in tool_msg.content
    assert "ERROR: connection refused" in tool_msg.content
    assert tool_msg.metadata["full_output"] == _big_log()
    assert tool_msg.token_count < len(_big_log().split())  # charged reduced only
    # Archive serves the original later in the session.
    archived = agent.result_processor.lookup(tool_msg.metadata["call_id"])
    assert archived is not None
    assert "filler detail line 20" in archived.original
    assert result.response == "Summarized."


def test_agent_small_outputs_untouched(tmp_path):
    with open(os.path.join(str(tmp_path), "tiny.txt"), "w") as fh:
        fh.write("Tiny.")
    replies = [_block("read_file", {"path": "tiny.txt"}), "Done."]
    agent = Agent(llm=ScriptedClient(replies),
                  conversation=Conversation(name="T", max_tokens=8000),
                  tools=[ReadFileTool(tmp_path)])
    agent.run("Read it.")
    tool_msg = next(m for m in agent.history if m.role == "tool")
    assert tool_msg.metadata["reduced"] is False
    assert "Tiny." in tool_msg.content
    assert "full_output" not in tool_msg.metadata


def test_agent_processor_disabled_records_raw(tmp_path):
    with open(os.path.join(str(tmp_path), "big.log"), "w") as fh:
        fh.write(_big_log())
    replies = [_block("read_file", {"path": "big.log"}), "Done."]
    agent = Agent(llm=ScriptedClient(replies),
                  conversation=Conversation(name="T", max_tokens=8000),
                  tools=[ReadFileTool(tmp_path)],
                  result_processor=False)
    agent.run("Read it.")
    tool_msg = next(m for m in agent.history if m.role == "tool")
    assert "filler detail line 20" in tool_msg.content  # raw, unreduced
    assert agent.result_processor is None


def test_agent_rejects_bad_processor():
    with pytest.raises(TypeError, match="result_processor"):
        Agent(llm=ScriptedClient(["hi"]), result_processor="fancy")  # type: ignore


def test_reduced_observation_keeps_compaction_honest(tmp_path):
    with open(os.path.join(str(tmp_path), "big.log"), "w") as fh:
        fh.write(_big_log())
    replies = [_block("read_file", {"path": "big.log"}), "Summarized."]
    conv = Conversation(name="Tight", max_tokens=60)
    conv.add(Message(role="system", content="You are helpful."))
    for i in range(6):
        conv.add(Message(role="user", content="Filler turn %d here." % i))
    agent = Agent(llm=ScriptedClient(replies), conversation=conv,
                  tools=[ReadFileTool(tmp_path)])
    result = agent.run("Read the big log.")
    # prompt_tokens reflects reduced (not raw) tool text end to end.
    tool_msg = next(m for m in agent.history if m.role == "tool")
    archived = agent.result_processor.lookup(tool_msg.metadata["call_id"])
    assert tool_msg.content.endswith(archived.reduced)
    assert tool_msg.token_count < archived.original_tokens  # header + reduced << raw
    assert result.prompt_tokens < len(_big_log().split()) * 2
    assert result.pipeline_result is not None
