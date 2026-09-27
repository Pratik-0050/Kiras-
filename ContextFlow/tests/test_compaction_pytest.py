# tests/test_compaction_pytest.py
"""Step 11 (pytest): compaction, priority scoring, protected messages, validation.

Covers context limits in action, pressure-driven compaction, auto priority
scoring, protected/critical preservation, discardable drops, and both
validation outcomes (rollback keeps data, warn commits anyway).

All tests are independent: helpers build fresh Conversations per test and
fakes avoid any network access.
"""

from typing import List

import pytest

from contextflow import (
    Compactor,
    Conversation,
    HeuristicPriorityScorer,
    ImportanceLevel,
    Message,
    PriorityScore,
    ValidationResult,
)
from contextflow.summarizers.base import Summarizer, SummarizerError
from contextflow.validators.base import Validator, ValidatorError


# ── fakes (no network, fully deterministic) ──────────────────────────


class EchoSummarizer(Summarizer):
    def summarize(self, messages: List[Message]) -> str:
        return f"Summary of {len(messages)} messages."


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
            validator_used="FailValidator",
        )


class ErrorValidator(Validator):
    def validate(self, original_messages, summary_text) -> ValidationResult:
        raise ValidatorError("Validator API unavailable.")


def _conv(*contents, max_tokens=1000, **kw):
    """Build a fresh Conversation from content strings (user turns by default)."""
    conv = Conversation(name=kw.pop("name", "Test"), max_tokens=max_tokens, **kw)
    for c in contents:
        if isinstance(c, Message):
            conv.add(c)
        else:
            conv.add(Message(role="user", content=c))
    return conv


def _long_conv(n=7, max_tokens=500):
    conv = Conversation(name="Long", max_tokens=max_tokens)
    conv.add(Message(role="system", content="System prompt: You are helpful."))
    for i in range(n - 1):
        role = "user" if i % 2 == 0 else "assistant"
        conv.add(Message(role=role, content=f"Turn {i}: substantive discussion point {i}."))
    return conv


# ── compaction basics ────────────────────────────────────────────────


def test_no_compaction_when_few_messages():
    conv = _conv("a", "b", "c")
    result = Compactor(keep_recent=3, summarizer=EchoSummarizer()).compact(conv)
    assert result.was_needed is False
    assert result.committed is True
    assert result.summary_message is None
    assert conv.message_count() == 3


def test_empty_conversation_needs_no_compaction():
    conv = Conversation(name="Empty", max_tokens=500)
    result = Compactor(keep_recent=2, summarizer=EchoSummarizer()).compact(conv)
    assert result.was_needed is False
    assert conv.message_count() == 0


def test_compaction_replaces_older_messages_and_updates_budget():
    conv = _long_conv(n=7)
    before_tokens = conv.total_tokens()
    result = Compactor(keep_recent=2, summarizer=EchoSummarizer()).compact(conv)

    assert result.was_needed is True
    assert result.committed is True
    assert result.original_message_count == 7
    assert result.summary_message is not None
    assert result.summary_message.metadata["type"] == "compaction_summary"
    # system prompt + summary + 2 recent = 4
    assert conv.message_count() == 4
    assert conv.total_tokens() == result.compacted_token_count
    assert result.tokens_saved == before_tokens - result.compacted_token_count
    assert conv.get_messages()[0].content.startswith("System prompt")


def test_compaction_without_leading_system_prompt():
    conv = _conv("t1", "t2", "t3", "t4")
    result = Compactor(keep_recent=2, summarizer=EchoSummarizer()).compact(conv)
    assert result.was_needed is True
    msgs = conv.get_messages()
    assert len(msgs) == 3
    assert msgs[0].role == "system"  # inserted summary
    assert msgs[1].content == "t3"
    assert msgs[2].content == "t4"


def test_summarizer_error_leaves_conversation_unchanged():
    conv = _conv("m1", "m2", "m3", "m4")
    before = [m.content for m in conv.get_messages()]
    with pytest.raises(SummarizerError):
        Compactor(keep_recent=2, summarizer=FailingSummarizer()).compact(conv)
    assert [m.content for m in conv.get_messages()] == before


def test_fallback_summarizer_used_on_primary_failure():
    conv = _conv("m1", "m2", "m3", "m4")
    result = Compactor(
        keep_recent=2,
        summarizer=FailingSummarizer(),
        fallback_summarizer=EchoSummarizer(),
    ).compact(conv)
    assert result.committed is True
    assert "fallback" in result.summarizer_used


def test_invalid_compactor_args_rejected():
    with pytest.raises((TypeError, ValueError)):
        Compactor(keep_recent=0)
    with pytest.raises(TypeError):
        Compactor(summarizer="nope")
    with pytest.raises(TypeError):
        Compactor(validator="nope")
    with pytest.raises(TypeError):
        Compactor(scorer="nope")
    with pytest.raises(ValueError):
        Compactor(on_validation_fail="explode")


# ── protected messages & importance ──────────────────────────────────


def test_protected_older_message_preserved_verbatim():
    conv = Conversation(name="P", max_tokens=1000)
    conv.add(Message(role="system", content="System prompt"))
    conv.add(Message(role="user", content="Never forget this rule", protected=True))
    conv.add(Message(role="assistant", content="filler one"))
    conv.add(Message(role="user", content="filler two"))
    conv.add(Message(role="assistant", content="Recent one"))
    conv.add(Message(role="user", content="Recent two"))

    result = Compactor(keep_recent=2, summarizer=EchoSummarizer()).compact(conv)
    contents = [m.content for m in conv.get_messages()]
    assert "Never forget this rule" in contents
    assert "Never forget this rule" not in result.summary_message.content
    assert any(m.content == "Never forget this rule" for m in result.preserved_messages)


def test_critical_message_preserved_and_discardable_dropped():
    conv = Conversation(name="Mix", max_tokens=1000)
    conv.add(Message(role="system", content="System prompt"))
    conv.add(Message(role="user", content="Critical DB setting", importance="critical"))
    conv.add(Message(role="user", content="Ping healthcheck", importance="discardable"))
    conv.add(Message(role="assistant", content="Normal discussion point"))
    conv.add(Message(role="assistant", content="Recent one"))
    conv.add(Message(role="user", content="Recent two"))

    result = Compactor(keep_recent=2, summarizer=EchoSummarizer()).compact(conv)
    contents = [m.content for m in conv.get_messages()]
    assert "Critical DB setting" in contents
    assert "Ping healthcheck" not in contents
    assert "Ping healthcheck" not in result.summary_message.content
    assert any(m.content == "Ping healthcheck" for m in result.discarded_messages)


def test_all_older_protected_means_no_compaction():
    conv = Conversation(name="AllP", max_tokens=1000)
    conv.add(Message(role="system", content="System prompt"))
    conv.add(Message(role="user", content="Constraint 1", protected=True))
    conv.add(Message(role="assistant", content="Constraint 2", importance="critical"))
    conv.add(Message(role="user", content="Recent 1"))
    conv.add(Message(role="assistant", content="Recent 2"))

    result = Compactor(keep_recent=2, summarizer=EchoSummarizer()).compact(conv)
    assert result.was_needed is False
    assert conv.message_count() == 5


def test_all_discardable_older_drops_without_summarizer_call():
    class ExplodingSummarizer(Summarizer):
        def summarize(self, messages):
            raise AssertionError("summarizer must not be called")

    conv = Conversation(name="AllD", max_tokens=1000)
    conv.add(Message(role="system", content="System prompt"))
    conv.add(Message(role="user", content="Log 1", importance="discardable"))
    conv.add(Message(role="assistant", content="Log 2", importance="discardable"))
    conv.add(Message(role="user", content="Recent 1"))
    conv.add(Message(role="assistant", content="Recent 2"))

    result = Compactor(keep_recent=2, summarizer=ExplodingSummarizer()).compact(conv)
    assert result.was_needed is True
    assert result.summary_message is None
    assert conv.message_count() == 3  # system + 2 recent


def test_compaction_result_labels_discarded_messages():
    conv = Conversation(name="Labels", max_tokens=1000)
    conv.add(Message(role="system", content="System instruction"))
    conv.add(Message(role="assistant", content="Trash greeting", importance="discardable"))
    conv.add(Message(role="user", content="Normal point to summarize"))
    conv.add(Message(role="assistant", content="Recent 1"))
    conv.add(Message(role="user", content="Recent 2"))

    result = Compactor(keep_recent=2, summarizer=EchoSummarizer()).compact(conv)
    text = str(result)
    assert "Preserved" in text
    assert "Summarized" in text
    assert "Removed" in text
    assert "discarded" in text
    assert "replaced by summary" in text


def test_auto_discarded_message_labeled_discarded_not_replaced():
    """Regression: scorer-auto-discarded chatter must be labeled 'discarded'."""
    conv = Conversation(name="AutoLabels", max_tokens=1000)
    conv.add(Message(role="system", content="sys"))
    conv.add(Message(role="user", content="ok thanks"))  # auto-discardable, importance normal
    conv.add(Message(role="assistant", content="Normal discussion point here for summary"))
    conv.add(Message(role="assistant", content="Recent 1"))
    conv.add(Message(role="user", content="Recent 2"))

    result = Compactor(keep_recent=2, summarizer=EchoSummarizer()).compact(conv)
    assert any(m.content == "ok thanks" for m in result.discarded_messages)
    assert "(discarded) ok thanks" in str(result)


# ── priority scoring ─────────────────────────────────────────────────


def test_priority_scores_cover_every_message_and_stay_in_bounds():
    conv = _long_conv(n=6)
    result = Compactor(keep_recent=2, summarizer=EchoSummarizer()).compact(conv)
    assert len(result.priority_scores) == 6
    for ps in result.priority_scores:
        assert isinstance(ps, PriorityScore)
        assert 0.0 <= ps.score <= 100.0
        assert ps.classification in set(ImportanceLevel)


def test_protected_scores_100_critical():
    scorer = HeuristicPriorityScorer()
    ps = scorer.score(Message(role="user", content="trivial", protected=True))
    assert ps.score == 100.0
    assert ps.classification == ImportanceLevel.CRITICAL


def test_explicit_critical_and_discardable_overrides():
    scorer = HeuristicPriorityScorer()
    critical = scorer.score(Message(role="assistant", content="plain", importance="critical"))
    assert critical.score >= 90.0
    assert critical.classification == ImportanceLevel.CRITICAL

    disc = scorer.score(Message(role="user", content="plain note", importance="discardable"))
    assert disc.score <= 25.0
    assert disc.classification == ImportanceLevel.DISCARDABLE


def test_auto_classification_preserves_vital_and_drops_chatter():
    conv = Conversation(name="Auto", max_tokens=1000)
    conv.add(Message(role="system", content="System instruction"))
    conv.add(Message(
        role="user",
        content="CRITICAL SECURITY REQUIREMENT: Never store unencrypted passwords or secrets in database.",
    ))
    conv.add(Message(role="assistant", content="Understood, all secrets are hashed."))
    conv.add(Message(role="user", content="ok thanks"))
    conv.add(Message(role="assistant", content="Recent turn 1"))
    conv.add(Message(role="user", content="Recent turn 2"))

    result = Compactor(keep_recent=2, summarizer=EchoSummarizer()).compact(conv)
    preserved = [m.content for m in result.preserved_messages]
    discarded = [m.content for m in result.discarded_messages]
    assert any("CRITICAL SECURITY REQUIREMENT" in c for c in preserved)
    assert any("ok thanks" in c for c in discarded)
    assert "Priority Scores" in str(result)


def test_custom_scorer_can_be_injected():
    from contextflow.scorers import PriorityScorer

    class FlatScorer(PriorityScorer):
        def score(self, message, index=0, total_messages=1):
            return PriorityScore(score=50.0, classification=ImportanceLevel.NORMAL, message=message)

    compactor = Compactor(keep_recent=2, summarizer=EchoSummarizer(), scorer=FlatScorer())
    assert isinstance(compactor.scorer, FlatScorer)
    conv = _conv("a", "b", "c", "d")
    result = compactor.compact(conv)
    assert result.was_needed is True


# ── validation behaviour ─────────────────────────────────────────────


def test_validation_pass_commits_and_records_metadata():
    conv = _conv("m1", "m2", "m3", "m4")
    result = Compactor(
        keep_recent=2, summarizer=EchoSummarizer(), validator=PassValidator()
    ).compact(conv)
    assert result.committed is True
    assert result.validation_result is not None
    assert result.validation_result.passed is True
    assert result.summary_message.metadata["validation_passed"] is True
    assert conv.message_count() == 3


def test_failed_validation_rollback_leaves_conversation_unchanged():
    conv = Conversation(name="Rollback", max_tokens=500)
    conv.add(Message(role="system", content="System prompt."))
    conv.add(Message(role="user", content="Message 1"))
    conv.add(Message(role="assistant", content="Message 2"))
    conv.add(Message(role="user", content="Message 3"))
    conv.add(Message(role="assistant", content="Message 4"))
    before_contents = [m.content for m in conv.get_messages()]
    before_tokens = conv.total_tokens()

    result = Compactor(
        keep_recent=2,
        summarizer=EchoSummarizer(),
        validator=FailValidator(),
        on_validation_fail="rollback",
    ).compact(conv)

    assert result.was_needed is True
    assert result.committed is False
    assert result.validation_result.passed is False
    assert "Requirement: must use Python" in result.validation_result.missing_items
    assert [m.content for m in conv.get_messages()] == before_contents
    assert conv.total_tokens() == before_tokens
    assert "ROLLED BACK" in str(result)


def test_failed_validation_warn_still_commits():
    conv = _conv("m1", "m2", "m3", "m4")
    result = Compactor(
        keep_recent=2,
        summarizer=EchoSummarizer(),
        validator=FailValidator(),
        on_validation_fail="warn",
    ).compact(conv)
    assert result.committed is True
    assert result.validation_result.passed is False
    assert conv.message_count() == 3  # compaction applied despite failure


def test_validator_error_propagates_and_leaves_state_for_caller():
    conv = _conv("m1", "m2", "m3", "m4")
    with pytest.raises(ValidatorError):
        Compactor(
            keep_recent=2, summarizer=EchoSummarizer(), validator=ErrorValidator()
        ).compact(conv)
