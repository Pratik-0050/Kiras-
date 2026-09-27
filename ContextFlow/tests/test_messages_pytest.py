# tests/test_messages_pytest.py
"""Step 11 (pytest): Message creation, token counting, and field validation.

Each test is independent: it builds its own Message objects and asserts
one focused behaviour. Covers invalid roles, importance, and protected flags.
"""

import pytest
import tiktoken

from contextflow import ImportanceLevel, Message

_ENCODER = tiktoken.get_encoding("cl100k_base")


# ── creation & defaults ──────────────────────────────────────────────


def test_message_defaults_to_normal_unprotected():
    msg = Message(role="user", content="Hello world")
    assert msg.role == "user"
    assert msg.content == "Hello world"
    assert msg.importance == ImportanceLevel.NORMAL
    assert msg.importance == "normal"  # ImportanceLevel is a str Enum
    assert msg.protected is False
    assert msg.metadata == {}


def test_message_metadata_stored():
    msg = Message(role="tool", content="result", metadata={"id": "1"})
    assert msg.metadata == {"id": "1"}


@pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
def test_all_valid_roles_accepted(role):
    msg = Message(role=role, content="hi")
    assert msg.role == role


@pytest.mark.parametrize(
    "bad_role", ["admin", "SYSTEM", "User", "", "human", "ai", None, 42]
)
def test_invalid_roles_rejected(bad_role):
    with pytest.raises(ValueError, match="Invalid role"):
        Message(role=bad_role, content="hi")


# ── token counting ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello",
        "Hello world",
        "We want to build an AI coding agent.",
        "Use pyproject.toml with flit or setuptools. " * 10,
    ],
)
def test_token_count_matches_tiktoken(text):
    msg = Message(role="user", content=text)
    assert msg.token_count == len(_ENCODER.encode(text))


def test_empty_content_counts_zero_tokens():
    assert Message(role="user", content="").token_count == 0


def test_longer_text_uses_more_tokens():
    short = Message(role="user", content="Hi")
    long = Message(role="user", content="Hi " * 200)
    assert long.token_count > short.token_count


# ── importance & protected ───────────────────────────────────────────


def test_string_importance_is_coerced_case_insensitively():
    assert Message(role="user", content="x", importance="critical").importance == ImportanceLevel.CRITICAL
    assert Message(role="user", content="x", importance="IMPORTANT").importance == ImportanceLevel.IMPORTANT
    assert Message(role="user", content="x", importance="Discardable").importance == ImportanceLevel.DISCARDABLE


def test_enum_importance_accepted_directly():
    msg = Message(role="user", content="x", importance=ImportanceLevel.IMPORTANT)
    assert msg.importance is ImportanceLevel.IMPORTANT


def test_invalid_importance_string_rejected():
    with pytest.raises(ValueError, match="Invalid importance"):
        Message(role="user", content="x", importance="super_high")


def test_non_string_non_enum_importance_rejected():
    with pytest.raises(TypeError, match="must be an ImportanceLevel or str"):
        Message(role="user", content="x", importance=42)


def test_protected_flag_defaults_false_and_accepts_true():
    assert Message(role="user", content="x").protected is False
    assert Message(role="user", content="x", protected=True).protected is True


@pytest.mark.parametrize("bad", ["true", "yes", 1, 0, None])
def test_non_bool_protected_rejected(bad):
    with pytest.raises(TypeError, match="protected must be a bool"):
        Message(role="user", content="x", protected=bad)


@pytest.mark.parametrize("bad", [123, None, ["x"], {"text": "x"}])
def test_non_string_content_rejected(bad):
    with pytest.raises(TypeError, match="content must be a string"):
        Message(role="user", content=bad)


@pytest.mark.parametrize("bad", ["meta", 123, [("a", 1)]])
def test_non_dict_metadata_rejected(bad):
    with pytest.raises(TypeError, match="metadata must be a dict or None"):
        Message(role="user", content="x", metadata=bad)


def test_repr_and_str_surface_importance_and_protected():
    msg = Message(
        role="user",
        content="Important constraint",
        importance=ImportanceLevel.CRITICAL,
        protected=True,
    )
    assert "importance='critical'" in repr(msg)
    assert "protected=True" in repr(msg)
    assert "Importance : critical" in str(msg)
    assert "Protected  : True" in str(msg)


def test_plain_message_repr_has_no_extra_flags():
    msg = Message(role="user", content="plain")
    assert "importance" not in repr(msg)
    assert "protected" not in repr(msg)
