# tests/test_pipeline_pytest.py
"""Step 17 (pytest): automatic end-to-end compaction pipeline.

Covers the full run (trigger -> retrieve -> compact -> validate -> persist),
threshold gating, validation-failure rollback, protected content, and error
handling. Summarizers and validators are deterministic fakes (no network).
"""

from typing import List

import pytest

from contextflow import (
    Compactor,
    ContextPipeline,
    ContextStatus,
    ContextStore,
    Conversation,
    ImportanceLevel,
    KeywordRetriever,
    Message,
    PipelineResult,
    StoreError,
    ValidationResult,
)
from contextflow.summarizers.base import Summarizer, SummarizerError
from contextflow.validators.base import Validator, ValidatorError


class EchoSummarizer(Summarizer):
    def summarize(self, messages: List[Message]) -> str:
        return "Summary of %d messages." % len(messages)


class FailingSummarizer(Summarizer):
    def summarize(self, messages: List[Message]) -> str:
        raise SummarizerError("Provider unavailable.")


class PassValidator(Validator):
    def validate(self, original_messages, summary_text) -> ValidationResult:
        return ValidationResult(passed=True, validator_used="PassValidator")


class FailValidator(Validator):
    def validate(self, original_messages, summary_text) -> ValidationResult:
        return ValidationResult(
            passed=False,
            missing_items=["Requirement: must use Python"],
            warnings=["summary is thin"],
            validator_used="FailValidator",
        )


class ErrorValidator(Validator):
    def validate(self, original_messages, summary_text) -> ValidationResult:
        raise ValidatorError("Validator API unavailable.")


def _pressured_conversation(max_tokens=60, assert_pressured=True):
    conv = Conversation(name="Pressured", max_tokens=max_tokens)
    conv.add(Message(role="system", content="You are helpful."))
    conv.add(Message(
        role="user",
        content="Never store secrets in plain text.",
        protected=True,
    ))
    conv.add(Message(
        role="assistant",
        content="Core decision recorded here.",
        importance=ImportanceLevel.CRITICAL,
    ))
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        conv.add(Message(role=role, content="Discussion turn number %d about architecture." % i))
    if assert_pressured:
        assert conv.get_status() == ContextStatus.COMPACTION_NEEDED
    return conv


def _pipeline(**kwargs):
    kwargs.setdefault(
        "compactor",
        Compactor(keep_recent=2, summarizer=EchoSummarizer(), validator=PassValidator()),
    )
    return ContextPipeline(**kwargs)


# ── successful run ─────────────────────────────────────────────────


def test_successful_run_compacts_and_reports():
    conv = _pressured_conversation()
    before_tokens = conv.total_tokens()
    before_count = conv.message_count()
    result = _pipeline().run(conv)

    assert isinstance(result, PipelineResult)
    assert result.triggered is True
    assert result.committed is True
    assert result.tokens_before == before_tokens
    assert result.tokens_after == conv.total_tokens()
    assert result.messages_before == before_count
    assert result.messages_after == conv.message_count()
    assert result.compaction is not None
    assert result.compaction.committed is True
    assert result.compression_ratio == round(result.tokens_after / before_tokens, 3)
    assert result.compression_ratio < 1.0
    assert result.tokens_saved == before_tokens - result.tokens_after
    assert result.status_before == ContextStatus.COMPACTION_NEEDED
    assert "triggered" in result.reason
    # Preserved / summarized mirror the compactor breakdown.
    assert [m.content for m in result.preserved_messages] == [
        m.content for m in result.compaction.preserved_messages]
    assert [m.content for m in result.summarized_messages] == [
        m.content for m in result.compaction.summarized_messages]
    assert result.summarized_messages
    # Validation status surfaces the validator verdict.
    assert result.validation_passed is True
    assert result.validation_result.passed is True


def test_protected_and_critical_content_identified_and_kept():
    conv = _pressured_conversation()
    result = _pipeline().run(conv)
    assert result.protected_count == 1
    assert result.critical_count >= 1  # explicit critical tag always counts
    contents = [m.content for m in conv.get_messages()]
    assert "Never store secrets in plain text." in contents
    assert "Core decision recorded here." in contents


def test_retrieval_stage_records_relevant_context():
    conv = _pressured_conversation()
    pipeline = _pipeline(retriever=KeywordRetriever())
    result = pipeline.run(conv)
    assert result.retrieved  # latest user turn matches architecture chatter
    assert result.retrieval_query is not None
    assert "Discussion turn" in result.retrieval_query


def test_explicit_query_overrides_default():
    conv = _pressured_conversation()
    pipeline = _pipeline(retriever=KeywordRetriever())
    by_default = pipeline.run(conv).retrieval_query
    conv2 = _pressured_conversation()
    by_explicit = pipeline.run(conv2, query="Core decision").retrieval_query
    assert by_explicit == "Core decision"
    assert by_default != "Core decision"


def test_no_retriever_warns_and_skips():
    conv = _pressured_conversation()
    result = _pipeline().run(conv)  # no retriever configured
    assert result.retrieved == []
    assert result.retrieval_query is None
    assert any("retrieval skipped" in w for w in result.warnings)


def test_result_str_shows_workflow():
    conv = _pressured_conversation()
    text = str(_pipeline().run(conv))
    assert "TRIGGERED" in text
    assert "ratio" in text
    assert "Preserved" in text
    assert "Retrieved" in text


# ── threshold gating ───────────────────────────────────────────────


def test_below_trigger_skips_without_touching_conversation():
    conv = Conversation(name="Calm", max_tokens=8000)
    conv.add(Message(role="user", content="Hello there."))
    before = [m.content for m in conv.get_messages()]
    result = _pipeline().run(conv)

    assert result.triggered is False
    assert result.committed is True
    assert result.compaction is None
    assert result.tokens_after == result.tokens_before
    assert result.compression_ratio == 1.0
    assert result.validation_passed is None
    assert "skipped" in result.reason
    assert [m.content for m in conv.get_messages()] == before
    assert "TRIGGERED" not in str(result)


def test_custom_trigger_pct_overrides_conversation_threshold():
    conv = _pressured_conversation(max_tokens=1000, assert_pressured=False)  # ~6%, status OK
    assert conv.get_status() == ContextStatus.OK
    # Default trigger (compact_at=90): skipped.
    assert _pipeline().run(conv).triggered is False
    # Custom low trigger forces a run.
    assert _pipeline(trigger_pct=5.0).run(conv).triggered is True


def test_high_custom_trigger_skips_pressured_conversation():
    # ~95% usage: above the default 90% trigger but below a custom 99.5%.
    conv = _pressured_conversation(max_tokens=67, assert_pressured=False)
    assert 90.0 <= conv.usage_percentage() < 99.5
    assert _pipeline().run(conv).triggered is True
    fresh = _pressured_conversation(max_tokens=67, assert_pressured=False)
    assert _pipeline(trigger_pct=99.5).run(fresh).triggered is False
    assert fresh.message_count() == 9  # skip leaves everything untouched


def test_conversation_without_limit_never_triggers():
    conv = Conversation(name="NoLimit")
    for i in range(10):
        conv.add(Message(role="user", content="Filler turn number %d here." % i))
    result = _pipeline().run(conv)
    assert result.triggered is False
    assert "no token limit" in result.reason


def test_empty_conversation_skips():
    conv = Conversation(name="Empty", max_tokens=100)
    result = _pipeline().run(conv)
    assert result.triggered is False
    assert result.tokens_before == 0
    assert result.compression_ratio == 1.0


def test_triggered_but_nothing_eligible_stays_unchanged():
    conv = Conversation(name="AllProtected", max_tokens=10)
    conv.add(Message(role="system", content="System prompt here."))
    conv.add(Message(role="user", content="Keep this rule.", protected=True))
    conv.add(Message(role="assistant", content="Keep that rule.", importance="critical"))
    conv.add(Message(role="user", content="Recent one here."))
    conv.add(Message(role="assistant", content="Recent two here."))
    assert conv.is_over_limit()
    result = _pipeline().run(conv)
    assert result.triggered is True
    assert result.committed is True
    assert result.compaction is not None
    assert result.compaction.was_needed is False
    assert conv.message_count() == 5


# ── validation failure ─────────────────────────────────────────────


def test_failed_validation_preserves_original():
    conv = _pressured_conversation()
    before_contents = [m.content for m in conv.get_messages()]
    before_tokens = conv.total_tokens()
    pipeline = ContextPipeline(compactor=Compactor(
        keep_recent=2, summarizer=EchoSummarizer(), validator=FailValidator()))
    result = pipeline.run(conv)

    assert result.triggered is True
    assert result.committed is False
    assert result.validation_passed is False
    assert result.validation_result.missing_items == ["Requirement: must use Python"]
    # Validator warnings propagate into the pipeline warnings.
    assert "summary is thin" in result.warnings
    assert any("preserved" in w for w in result.warnings)
    assert [m.content for m in conv.get_messages()] == before_contents
    assert conv.total_tokens() == before_tokens
    assert result.tokens_after == before_tokens
    assert result.compression_ratio == 1.0
    assert result.persisted_path is None
    assert "ROLLED BACK" in str(result)


def test_failed_validation_skips_persistence(tmp_path):
    conv = _pressured_conversation()
    path = str(tmp_path / "should-not-exist.json")
    pipeline = ContextPipeline(
        compactor=Compactor(
            keep_recent=2, summarizer=EchoSummarizer(), validator=FailValidator()),
        persist_path=path,
    )
    result = pipeline.run(conv)
    assert result.committed is False
    assert result.persisted_path is None
    import os
    assert not os.path.exists(path)


def test_validator_error_propagates_untouched():
    conv = _pressured_conversation()
    before = [m.content for m in conv.get_messages()]
    pipeline = ContextPipeline(compactor=Compactor(
        keep_recent=2, summarizer=EchoSummarizer(), validator=ErrorValidator()))
    with pytest.raises(ValidatorError):
        pipeline.run(conv)
    assert [m.content for m in conv.get_messages()] == before


def test_summarizer_error_propagates_untouched():
    conv = _pressured_conversation()
    before = [m.content for m in conv.get_messages()]
    pipeline = ContextPipeline(compactor=Compactor(
        keep_recent=2, summarizer=FailingSummarizer()))
    with pytest.raises(Exception):
        pipeline.run(conv)
    assert [m.content for m in conv.get_messages()] == before


# ── persistence ────────────────────────────────────────────────────


def test_persist_path_saves_loadable_context(tmp_path):
    conv = _pressured_conversation()
    path = str(tmp_path / "pipeline.json")
    result = _pipeline(persist_path=path).run(conv)

    assert result.persisted_path == path
    restored = ContextStore.load_file(path)
    assert restored.conversation.message_count() == conv.message_count()
    assert restored.conversation.total_tokens() == conv.total_tokens()
    assert restored.last_compaction is not None
    assert any("persisted" in w for w in result.warnings)


def test_provided_store_is_recorded(tmp_path):
    conv = _pressured_conversation()
    store = ContextStore(Conversation(name="Other"))
    path = str(tmp_path / "pipeline.json")
    result = _pipeline(store=store, persist_path=path).run(conv)
    assert result.persisted_path == path
    assert store.conversation is conv  # repointed so disk matches memory
    assert store.last_compaction is result.compaction


def test_no_persist_path_saves_nothing(tmp_path):
    conv = _pressured_conversation()
    result = _pipeline().run(conv)  # no persist_path configured
    assert result.persisted_path is None


def test_persistence_failure_raises_store_error(tmp_path):
    conv = _pressured_conversation()
    missing = str(tmp_path / "no-such-dir" / "pipeline.json")
    with pytest.raises(StoreError):
        _pipeline(persist_path=missing).run(conv)


# ── constructor / call validation ──────────────────────────────────


def test_invalid_pipeline_arguments_rejected():
    with pytest.raises(TypeError, match="Compactor"):
        ContextPipeline(compactor="nope")  # type: ignore
    with pytest.raises(TypeError, match="trigger_pct"):
        ContextPipeline(trigger_pct="high")  # type: ignore
    with pytest.raises(ValueError, match="trigger_pct"):
        ContextPipeline(trigger_pct=0.0)
    with pytest.raises(ValueError, match="trigger_pct"):
        ContextPipeline(trigger_pct=101.0)
    with pytest.raises(TypeError, match="Retriever"):
        ContextPipeline(retriever="nope")  # type: ignore
    with pytest.raises(TypeError, match="query must be"):
        ContextPipeline(query=123)  # type: ignore
    with pytest.raises(ValueError, match="retrieval_top_k"):
        ContextPipeline(retrieval_top_k=0)
    with pytest.raises(TypeError, match="ContextStore"):
        ContextPipeline(store="nope")  # type: ignore
    with pytest.raises(TypeError, match="persist_path"):
        ContextPipeline(persist_path=123)  # type: ignore
    with pytest.raises(ValueError, match="must not be empty"):
        ContextPipeline(persist_path="   ")


def test_run_rejects_non_conversation():
    with pytest.raises(TypeError, match="Conversation"):
        _pipeline().run("nope")  # type: ignore
    with pytest.raises(TypeError, match="query must be"):
        _pipeline().run(_pressured_conversation(), query=123)  # type: ignore
