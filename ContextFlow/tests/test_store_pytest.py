# tests/test_store_pytest.py
"""Step 12 (pytest): persistent context storage via ContextStore (JSON).

Covers saving, loading, corrupted files, and empty conversations.
Every test is independent (tmp_path fixture, fresh Conversations).
"""

import json

import pytest

from contextflow import (
    Compactor,
    ContextStore,
    Conversation,
    ImportanceLevel,
    Message,
    StoreError,
    ValidationResult,
)
from contextflow.summarizers.base import Summarizer


class EchoSummarizer(Summarizer):
    def summarize(self, messages):
        return f"Summary of {len(messages)} messages."


class PassValidator:
    """Minimal stub with the Validator interface (no import cycle in tests)."""

    def validate(self, original_messages, summary_text):
        return ValidationResult(passed=True, validator_used="PassValidator")


def _rich_conversation():
    conv = Conversation(name="Session", max_tokens=8000, warn_at=70.0, compact_at=90.0)
    conv.add(Message(role="system", content="You are helpful."))
    conv.add(Message(
        role="user",
        content="Security Rule: AES-256 required.",
        protected=True,
    ))
    conv.add(Message(
        role="assistant",
        content="Decision: use Postgres.",
        importance=ImportanceLevel.CRITICAL,
    ))
    conv.add(Message(
        role="user",
        content="Debug ping 200 ok.",
        importance=ImportanceLevel.DISCARDABLE,
    ))
    conv.add(Message(
        role="user",
        content="What library for tokens?",
        importance=ImportanceLevel.IMPORTANT,
        metadata={"topic": "tokens"},
    ))
    return conv


def _assert_conversations_equal(original: Conversation, loaded: Conversation):
    assert loaded.name == original.name
    assert loaded.max_tokens == original.max_tokens
    assert loaded.warn_at == original.warn_at
    assert loaded.compact_at == original.compact_at
    assert loaded.message_count() == original.message_count()
    for a, b in zip(original.get_messages(), loaded.get_messages()):
        assert b.role == a.role
        assert b.content == a.content
        assert b.importance == a.importance
        assert b.protected == a.protected
        assert b.metadata == a.metadata
        assert b.token_count == a.token_count  # recomputed identically
    # Behavioural parity: budgets and pressure match exactly.
    assert loaded.total_tokens() == original.total_tokens()
    assert loaded.remaining_tokens() == original.remaining_tokens()
    assert loaded.usage_percentage() == original.usage_percentage()
    assert loaded.is_over_limit() == original.is_over_limit()
    assert loaded.get_status() == original.get_status()


# ── saving & loading ─────────────────────────────────────────────


def test_save_creates_valid_json_file(tmp_path):
    conv = _rich_conversation()
    path = tmp_path / "session.json"
    ContextStore(conv).save(path)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "contextflow/context-store"
    assert payload["version"] == 1
    assert payload["conversation"]["name"] == "Session"
    assert len(payload["conversation"]["messages"]) == 5


def test_round_trip_preserves_everything(tmp_path):
    conv = _rich_conversation()
    path = tmp_path / "session.json"
    ContextStore(conv).save(path)

    restored = ContextStore.load_file(path)
    _assert_conversations_equal(conv, restored.conversation)


def test_round_trip_preserves_token_counts_and_pressure(tmp_path):
    conv = Conversation(name="Budget", max_tokens=40)
    conv.add(Message(role="system", content="System prompt"))
    conv.add(Message(role="user", content="A fairly long user turn about architecture."))
    conv.add(Message(role="assistant", content="A fairly long assistant reply about layering."))
    path = tmp_path / "b.json"
    ContextStore(conv).save(str(path))  # str paths work too

    loaded = ContextStore.load_file(path).conversation
    assert loaded.total_tokens() == conv.total_tokens()
    assert loaded.get_status() == conv.get_status()
    assert loaded.get_status_message() == conv.get_status_message()


def test_loaded_conversation_behaves_like_new(tmp_path):
    conv = _rich_conversation()
    path = tmp_path / "s.json"
    ContextStore(conv).save(path)
    loaded = ContextStore.load_file(path).conversation

    # Can keep chatting, querying, and compacting after load.
    loaded.add(Message(role="user", content="Follow-up question?"))
    assert loaded.message_count() == conv.message_count() + 1
    result = Compactor(keep_recent=2, summarizer=EchoSummarizer()).compact(loaded)
    assert result.was_needed is True
    assert result.committed is True
    assert loaded.total_tokens() == result.compacted_token_count


def test_compaction_summary_and_validation_survive_round_trip(tmp_path):
    conv = Conversation(name="Compacted", max_tokens=1000)
    conv.add(Message(role="system", content="System prompt"))
    for i in range(5):
        conv.add(Message(role="user", content=f"Discussion point {i} with substance."))
    conv.add(Message(role="assistant", content="Recent one"))
    conv.add(Message(role="user", content="Recent two"))

    from contextflow.validators.heuristic_validator import HeuristicValidator

    compactor = Compactor(
        keep_recent=2,
        summarizer=EchoSummarizer(),
        validator=HeuristicValidator(fail_threshold=0.99, min_candidates=10),
    )
    result = compactor.compact(conv)
    assert result.committed is True

    store = ContextStore(conv)
    store.record(result)
    path = tmp_path / "compacted.json"
    store.save(path)

    restored = ContextStore.load_file(path)
    _assert_conversations_equal(conv, restored.conversation)

    # Summary message (compaction summary) is still in the conversation.
    summaries = [
        m for m in restored.conversation.get_messages()
        if m.metadata.get("type") == "compaction_summary"
    ]
    assert len(summaries) == 1
    assert summaries[0].content == result.summary_message.content

    # Validation result round-tripped.
    assert restored.last_validation is not None
    assert restored.last_validation.passed == result.validation_result.passed
    assert restored.last_validation.validator_used == result.validation_result.validator_used

    # Compaction metrics round-tripped.
    assert restored.last_compaction is not None
    assert restored.last_compaction.messages_removed == result.messages_removed
    assert restored.last_compaction.tokens_saved == result.tokens_saved
    assert restored.last_compaction.summarizer_used == result.summarizer_used
    assert restored.last_compaction.summary_message.content == result.summary_message.content


def test_instance_load_replaces_state_and_returns_self(tmp_path):
    conv = _rich_conversation()
    path = tmp_path / "s.json"
    ContextStore(conv).save(path)

    other = ContextStore(Conversation(name="Blank"))
    returned = other.load(path)
    assert returned is other
    _assert_conversations_equal(conv, other.conversation)


def test_save_overwrites_existing_file(tmp_path):
    path = tmp_path / "s.json"
    ContextStore(Conversation(name="First")).save(path)
    ContextStore(Conversation(name="Second")).save(path)
    assert ContextStore.load_file(path).conversation.name == "Second"


# ── empty conversations ────────────────────────────────────────────


def test_empty_conversation_round_trip(tmp_path):
    conv = Conversation(name="Empty", max_tokens=500)
    path = tmp_path / "empty.json"
    ContextStore(conv).save(path)
    loaded = ContextStore.load_file(path).conversation
    assert loaded.message_count() == 0
    assert loaded.total_tokens() == 0
    assert loaded.get_status() == conv.get_status()
    _assert_conversations_equal(conv, loaded)


def test_no_limit_conversation_round_trip(tmp_path):
    conv = Conversation(name="NoLimit")
    conv.add(Message(role="user", content="hello"))
    path = tmp_path / "n.json"
    ContextStore(conv).save(path)
    loaded = ContextStore.load_file(path).conversation
    assert loaded.max_tokens is None
    assert loaded.remaining_tokens() is None
    _assert_conversations_equal(conv, loaded)


# ── missing & corrupted files ──────────────────────────────────────


def test_missing_file_raises_store_error(tmp_path):
    with pytest.raises(StoreError, match="not found"):
        ContextStore.load_file(tmp_path / "does-not-exist.json")


def test_invalid_json_raises_store_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ this is not valid JSON", encoding="utf-8")
    with pytest.raises(StoreError, match="invalid JSON"):
        ContextStore.load_file(path)


def test_empty_file_raises_store_error(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")
    with pytest.raises(StoreError, match="invalid JSON"):
        ContextStore.load_file(path)


def test_wrong_kind_rejected(tmp_path):
    path = tmp_path / "k.json"
    path.write_text(json.dumps({"kind": "other", "version": 1}), encoding="utf-8")
    with pytest.raises(StoreError, match="unexpected 'kind'"):
        ContextStore.load_file(path)


def test_unsupported_version_rejected(tmp_path):
    conv = _rich_conversation()
    store = ContextStore(conv)
    path = tmp_path / "v.json"
    store.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StoreError, match="Unsupported store version"):
        ContextStore.load_file(path)


def test_missing_conversation_key_rejected(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps({"kind": "contextflow/context-store", "version": 1}),
        encoding="utf-8",
    )
    with pytest.raises(StoreError, match="missing required key 'conversation'"):
        ContextStore.load_file(path)


def test_bad_message_role_rejected(tmp_path):
    conv = _rich_conversation()
    path = tmp_path / "r.json"
    ContextStore(conv).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["conversation"]["messages"][0]["role"] = "admin"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StoreError, match="[Cc]orrupt"):
        ContextStore.load_file(path)


def test_bad_importance_rejected(tmp_path):
    conv = _rich_conversation()
    path = tmp_path / "i.json"
    ContextStore(conv).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["conversation"]["messages"][1]["importance"] = "mega"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StoreError, match="[Cc]orrupt"):
        ContextStore.load_file(path)


def test_bad_limits_rejected(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(
        json.dumps({
            "kind": "contextflow/context-store",
            "version": 1,
            "conversation": {
                "name": "Bad",
                "max_tokens": -5,
                "warn_at": 70.0,
                "compact_at": 90.0,
                "messages": [],
            },
            "last_compaction": None,
            "last_validation": None,
        }),
        encoding="utf-8",
    )
    with pytest.raises(StoreError, match="[Cc]orrupt"):
        ContextStore.load_file(path)


def test_top_level_array_rejected(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(StoreError, match="[Cc]orrupt"):
        ContextStore.load_file(path)


# ── save-side validation ───────────────────────────────────────────


def test_save_rejects_bad_path_type(tmp_path):
    with pytest.raises(TypeError, match="path must be"):
        ContextStore(Conversation(name="T")).save(123)


def test_save_rejects_empty_path():
    with pytest.raises(ValueError, match="must not be empty"):
        ContextStore(Conversation(name="T")).save("   ")


def test_save_to_missing_directory_raises_store_error(tmp_path):
    missing = tmp_path / "no-such-dir" / "s.json"
    with pytest.raises(StoreError, match="Cannot save"):
        ContextStore(Conversation(name="T")).save(missing)


def test_load_rejects_bad_path_type():
    with pytest.raises(TypeError, match="path must be"):
        ContextStore.load_file(None)


def test_store_rejects_wrong_types():
    with pytest.raises(TypeError):
        ContextStore(conversation="nope")
    with pytest.raises(TypeError):
        ContextStore().record("nope")
