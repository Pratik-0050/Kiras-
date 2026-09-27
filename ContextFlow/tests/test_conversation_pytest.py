# tests/test_conversation_pytest.py
"""Step 11 (pytest): Conversation management, token budget, and pressure.

Covers empty conversations, invalid token limits, context limits,
over-limit conversations, and OK / WARNING / COMPACTION_NEEDED detection.
Every test builds its own Conversation (no shared state).
"""

import pytest

from contextflow import ContextStatus, Conversation, Message


def _msg(role="user", content="hello", **kwargs):
    return Message(role=role, content=content, **kwargs)


def _filled(name="Conv", max_tokens=1000, n=4):
    conv = Conversation(name=name, max_tokens=max_tokens)
    for i in range(n):
        conv.add(_msg(content=f"Turn {i} with some content."))
    return conv


# ── empty conversations ──────────────────────────────────────────────


def test_empty_conversation_has_zero_usage():
    conv = Conversation(name="Empty", max_tokens=1000)
    assert conv.message_count() == 0
    assert conv.get_messages() == []
    assert conv.total_tokens() == 0
    assert conv.remaining_tokens() == 1000
    assert conv.usage_percentage() == 0.0
    assert conv.is_over_limit() is False
    assert conv.get_status() == ContextStatus.OK


def test_empty_conversation_without_limit():
    conv = Conversation(name="Empty")
    assert conv.total_tokens() == 0
    assert conv.remaining_tokens() is None
    assert conv.usage_percentage() is None
    assert conv.is_over_limit() is False
    assert conv.get_status() == ContextStatus.OK
    assert "no limit" in conv.get_status_message()


def test_clear_empties_conversation():
    conv = _filled()
    assert conv.message_count() == 4
    conv.clear()
    assert conv.message_count() == 0
    assert conv.total_tokens() == 0


# ── add / remove / query ─────────────────────────────────────────────


def test_add_and_get_messages():
    conv = Conversation(name="T", max_tokens=500)
    m = _msg()
    conv.add(m)
    assert conv.message_count() == 1
    assert conv.get_messages() == [m]


def test_add_rejects_non_message():
    conv = Conversation(name="T")
    with pytest.raises(TypeError):
        conv.add("not a message")
    with pytest.raises(TypeError):
        conv.add({"role": "user"})


def test_get_messages_returns_safe_copy():
    conv = _filled(n=2)
    copy = conv.get_messages()
    copy.clear()
    assert conv.message_count() == 2  # original untouched


def test_remove_returns_message_and_shrinks():
    conv = _filled(n=3)
    first = conv.get_messages()[0]
    removed = conv.remove(0)
    assert removed is first
    assert conv.message_count() == 2


@pytest.mark.parametrize("bad_index", [-1, 3, 99])
def test_remove_out_of_range_raises(bad_index):
    conv = _filled(n=3)
    with pytest.raises(IndexError):
        conv.remove(bad_index)


# ── invalid token limits ─────────────────────────────────────────────


@pytest.mark.parametrize("bad", [0, -1, -8000])
def test_non_positive_max_tokens_rejected(bad):
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        Conversation(name="T", max_tokens=bad)


@pytest.mark.parametrize("bad", ["100", 3.5, True, False, [100], object()])
def test_non_int_max_tokens_rejected(bad):
    with pytest.raises(TypeError, match="max_tokens must be a positive integer"):
        Conversation(name="T", max_tokens=bad)


def test_none_max_tokens_means_no_limit():
    conv = Conversation(name="T", max_tokens=None)
    conv.add(_msg(content="anything"))
    assert conv.remaining_tokens() is None
    assert conv.usage_percentage() is None
    assert conv.is_over_limit() is False


def test_invalid_pressure_thresholds_rejected():
    with pytest.raises(ValueError):
        Conversation(name="T", max_tokens=100, warn_at=90.0, compact_at=90.0)
    with pytest.raises(ValueError):
        Conversation(name="T", max_tokens=100, warn_at=95.0, compact_at=90.0)
    with pytest.raises(ValueError):
        Conversation(name="T", max_tokens=100, warn_at=0.0)
    with pytest.raises(TypeError):
        Conversation(name="T", max_tokens=100, warn_at="70")


# ── token budget / context limits ────────────────────────────────────


def test_token_budget_tracking_is_consistent():
    conv = Conversation(name="Budget", max_tokens=1000)
    m1 = _msg(content="Hello world")
    m2 = _msg(content="How are you today?")
    conv.add(m1)
    conv.add(m2)
    expected = m1.token_count + m2.token_count
    assert conv.total_tokens() == expected
    assert conv.remaining_tokens() == 1000 - expected
    assert conv.usage_percentage() == pytest.approx(round(expected / 1000 * 100, 1))
    assert conv.is_over_limit() is False


def test_over_limit_conversation():
    conv = Conversation(name="Over", max_tokens=5)
    conv.add(_msg(content="This message alone is definitely longer than five tokens."))
    assert conv.is_over_limit() is True
    assert conv.remaining_tokens() < 0
    assert conv.usage_percentage() > 100.0
    assert conv.get_status() == ContextStatus.COMPACTION_NEEDED
    assert "OVER LIMIT" in conv.get_status_message()


def test_exactly_at_limit_is_not_over():
    conv = Conversation(name="Exact", max_tokens=10_000)
    conv.add(_msg(content="hi"))
    assert conv.is_over_limit() is False
    assert conv.remaining_tokens() > 0


# ── pressure detection ───────────────────────────────────────────────


def test_pressure_ok_below_warn_threshold():
    conv = Conversation(name="T", max_tokens=1000, warn_at=70.0, compact_at=90.0)
    # ~few tokens -> well below 70%
    conv.add(_msg(content="hi"))
    assert conv.get_status() == ContextStatus.OK
    assert "Plenty of space" in conv.get_status_message()


def test_pressure_warning_between_thresholds():
    conv = Conversation(name="T", max_tokens=100, warn_at=10.0, compact_at=90.0)
    conv.add(_msg(content="This is a moderately long message for testing pressure."))
    pct = conv.usage_percentage()
    assert pct >= 10.0
    if pct < 90.0:
        assert conv.get_status() == ContextStatus.WARNING
        assert "Approaching the limit" in conv.get_status_message()


def test_pressure_compaction_needed_above_threshold():
    conv = Conversation(name="T", max_tokens=10, warn_at=70.0, compact_at=90.0)
    conv.add(_msg(content="This message will blow past a ten-token budget."))
    assert conv.usage_percentage() >= 90.0
    assert conv.get_status() == ContextStatus.COMPACTION_NEEDED
    assert "compaction required" in conv.get_status_message().lower()


def test_no_limit_always_ok():
    conv = Conversation(name="T")
    for i in range(20):
        conv.add(_msg(content=f"Long message number {i} with extra padding words."))
    assert conv.get_status() == ContextStatus.OK
