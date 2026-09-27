# tests/test_importance.py
"""
Unit tests for Step 9: Message importance levels and protected context.
"""

import unittest
from typing import List

from contextflow import (
    Message,
    Conversation,
    ContextStatus,
    Compactor,
    ImportanceLevel,
    VALID_IMPORTANCE,
    ValidationResult,
)
from contextflow.summarizers.base import Summarizer
from contextflow.validators.base import Validator


class MockEchoSummarizer(Summarizer):
    """Concatenates all input messages into a simple summary."""
    def summarize(self, messages: List[Message]) -> str:
        return "Summary of: " + " | ".join(m.content for m in messages)


class MockFailValidator(Validator):
    """Always fails validation."""
    def validate(self, original_messages: List[Message], summary_text: str) -> ValidationResult:
        return ValidationResult(
            passed=False,
            missing_items=["Critical fact missing"],
            validator_used="MockFailValidator",
        )


class TestMessageImportance(unittest.TestCase):
    """Test Message class extension for importance and protected flag."""

    def test_default_importance_and_protected(self):
        msg = Message(role="user", content="Hello world")
        self.assertEqual(msg.importance, ImportanceLevel.NORMAL)
        self.assertEqual(msg.importance, "normal")
        self.assertFalse(msg.protected)

    def test_importance_enum_values(self):
        self.assertEqual(ImportanceLevel.CRITICAL.value, "critical")
        self.assertEqual(ImportanceLevel.IMPORTANT.value, "important")
        self.assertEqual(ImportanceLevel.NORMAL.value, "normal")
        self.assertEqual(ImportanceLevel.DISCARDABLE.value, "discardable")
        self.assertEqual(
            VALID_IMPORTANCE,
            {"critical", "important", "normal", "discardable"},
        )

    def test_string_importance_coercion(self):
        msg1 = Message(role="user", content="Test", importance="critical")
        self.assertEqual(msg1.importance, ImportanceLevel.CRITICAL)

        msg2 = Message(role="user", content="Test", importance="IMPORTANT")
        self.assertEqual(msg2.importance, ImportanceLevel.IMPORTANT)

        msg3 = Message(role="user", content="Test", importance="Discardable")
        self.assertEqual(msg3.importance, ImportanceLevel.DISCARDABLE)

    def test_invalid_importance_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            Message(role="user", content="Test", importance="super_high")
        self.assertIn("Invalid importance", str(ctx.exception))

    def test_non_string_non_enum_importance_raises_type_error(self):
        with self.assertRaises(TypeError) as ctx:
            Message(role="user", content="Test", importance=42)  # type: ignore
        self.assertIn("must be an ImportanceLevel or str", str(ctx.exception))

    def test_protected_flag_boolean(self):
        msg = Message(role="user", content="Keep me forever", protected=True)
        self.assertTrue(msg.protected)

    def test_non_bool_protected_raises_type_error(self):
        with self.assertRaises(TypeError) as ctx:
            Message(role="user", content="Test", protected="true")  # type: ignore
        self.assertIn("protected must be a bool", str(ctx.exception))

        with self.assertRaises(TypeError):
            Message(role="user", content="Test", protected=1)  # type: ignore

    def test_repr_and_str(self):
        msg = Message(
            role="user",
            content="Important constraint",
            importance=ImportanceLevel.CRITICAL,
            protected=True,
        )
        r = repr(msg)
        self.assertIn("importance='critical'", r)
        self.assertIn("protected=True", r)

        s = str(msg)
        self.assertIn("Importance : critical", s)
        self.assertIn("Protected  : True", s)


class TestCompactorWithImportance(unittest.TestCase):
    """Test Compactor handling of protected and importance-tagged messages."""

    def test_protected_older_message_preserved_verbatim(self):
        """Older message with protected=True must not be removed or summarized."""
        conv = Conversation(name="Protected Test", max_tokens=1000)
        conv.add(Message(role="system", content="System prompt"))
        conv.add(Message(role="user", content="Never forget this rule", protected=True))
        conv.add(Message(role="assistant", content="Got it, rule noted."))
        conv.add(Message(role="user", content="What about feature X?"))
        conv.add(Message(role="assistant", content="Feature X is planned."))
        conv.add(Message(role="user", content="Recent question"))
        conv.add(Message(role="assistant", content="Recent answer"))

        # keep_recent=2 -> older messages are [rule, got it, feature X, feature X planned]
        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        self.assertTrue(result.committed)

        # The protected message must still be in the conversation
        messages = conv.get_messages()
        contents = [m.content for m in messages]
        self.assertIn("System prompt", contents)
        self.assertIn("Never forget this rule", contents)
        self.assertIn("Recent question", contents)
        self.assertIn("Recent answer", contents)

        # It must NOT be in the summary
        self.assertNotIn("Never forget this rule", result.summary_message.content)

        # Breakdown checks
        self.assertTrue(any(m.content == "Never forget this rule" for m in result.preserved_messages))
        self.assertFalse(any(m.content == "Never forget this rule" for m in result.summarized_messages))

    def test_critical_older_message_preserved_verbatim(self):
        """Older message with importance='critical' must be preserved."""
        conv = Conversation(name="Critical Test", max_tokens=1000)
        conv.add(Message(role="system", content="System prompt"))
        conv.add(Message(role="user", content="Critical DB setting", importance="critical"))
        conv.add(Message(role="assistant", content="Normal chatter 1"))
        conv.add(Message(role="user", content="Normal chatter 2"))
        conv.add(Message(role="assistant", content="Recent turn 1"))
        conv.add(Message(role="user", content="Recent turn 2"))

        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        contents = [m.content for m in conv.get_messages()]
        self.assertIn("Critical DB setting", contents)
        self.assertNotIn("Critical DB setting", result.summary_message.content)
        self.assertTrue(any(m.content == "Critical DB setting" for m in result.preserved_messages))

    def test_discardable_message_dropped_without_summarization(self):
        """Older message with importance='discardable' is removed without LLM summary."""
        conv = Conversation(name="Discardable Test", max_tokens=1000)
        conv.add(Message(role="system", content="System prompt"))
        conv.add(Message(role="user", content="Ping health check", importance="discardable"))
        conv.add(Message(role="assistant", content="Pong ok", importance="discardable"))
        conv.add(Message(role="user", content="Discuss architecture"))
        conv.add(Message(role="assistant", content="Architecture is microservices."))
        conv.add(Message(role="user", content="Recent question"))
        conv.add(Message(role="assistant", content="Recent answer"))

        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)

        # Discarded messages are NOT in conversation
        contents = [m.content for m in conv.get_messages()]
        self.assertNotIn("Ping health check", contents)
        self.assertNotIn("Pong ok", contents)

        # Discarded messages are NOT in summary
        self.assertNotIn("Ping health check", result.summary_message.content)
        self.assertNotIn("Pong ok", result.summary_message.content)

        # But normal older messages ARE in summary
        self.assertIn("Discuss architecture", result.summary_message.content)

        # Check result lists
        self.assertEqual(len(result.discarded_messages), 2)
        discarded_contents = [m.content for m in result.discarded_messages]
        self.assertIn("Ping health check", discarded_contents)
        self.assertIn("Pong ok", discarded_contents)

        # Total messages removed = 2 discarded + 2 summarized = 4
        self.assertEqual(result.messages_removed, 4)

    def test_all_older_messages_protected_means_no_compaction(self):
        """If all older messages are protected/critical, compaction is not needed."""
        conv = Conversation(name="All Protected", max_tokens=1000)
        conv.add(Message(role="system", content="System prompt"))
        conv.add(Message(role="user", content="Constraint 1", protected=True))
        conv.add(Message(role="assistant", content="Constraint 2", importance="critical"))
        conv.add(Message(role="user", content="Recent 1"))
        conv.add(Message(role="assistant", content="Recent 2"))

        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        self.assertFalse(result.was_needed)
        self.assertEqual(result.messages_removed, 0)
        self.assertEqual(result.tokens_saved, 0)
        self.assertIsNone(result.summary_message)
        self.assertEqual(conv.message_count(), 5)

    def test_all_older_messages_discardable_frees_tokens_without_summarizer(self):
        """If all older messages are discardable, they are dropped without invoking summarizer."""
        class FailIfCalledSummarizer(Summarizer):
            def summarize(self, messages):
                raise RuntimeError("Summarizer should not be called!")

        conv = Conversation(name="All Discardable", max_tokens=1000)
        conv.add(Message(role="system", content="System prompt"))
        conv.add(Message(role="user", content="Log 1", importance="discardable"))
        conv.add(Message(role="assistant", content="Log 2", importance="discardable"))
        conv.add(Message(role="user", content="Recent 1"))
        conv.add(Message(role="assistant", content="Recent 2"))

        compactor = Compactor(keep_recent=2, summarizer=FailIfCalledSummarizer())
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        self.assertEqual(result.messages_removed, 2)
        self.assertIsNone(result.summary_message)
        self.assertEqual(conv.message_count(), 3)  # system + recent 1 + recent 2
        self.assertEqual(len(result.summarized_messages), 0)
        self.assertEqual(len(result.discarded_messages), 2)

    def test_token_budget_recalculation_after_compaction(self):
        """Token budget, remaining tokens, and pressure status update correctly after compaction."""
        conv = Conversation(name="Budget Test", max_tokens=40)
        # Add messages until status is COMPACTION_NEEDED
        conv.add(Message(role="system", content="System prompt"))
        conv.add(Message(role="user", content="Protected message", protected=True))
        conv.add(Message(role="assistant", content="Very long normal message with lots of text to use up token budget"))
        conv.add(Message(role="user", content="Another long normal message consuming more tokens in the conversation"))
        conv.add(Message(role="assistant", content="Recent response turn"))
        conv.add(Message(role="user", content="Recent user turn"))

        self.assertIn(conv.get_status(), [ContextStatus.WARNING, ContextStatus.COMPACTION_NEEDED])

        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        self.assertTrue(result.committed)

        # Total tokens in conversation matches result.compacted_token_count
        self.assertEqual(conv.total_tokens(), result.compacted_token_count)
        self.assertEqual(conv.remaining_tokens(), conv.max_tokens - result.compacted_token_count)
        self.assertEqual(
            conv.usage_percentage(),
            round((result.compacted_token_count / conv.max_tokens) * 100, 1),
        )
        self.assertFalse(conv.is_over_limit())

    def test_validator_receives_only_summarizable_messages(self):
        """Validator should only receive messages that were actually summarized."""
        recorded_messages = []

        class RecordingValidator(Validator):
            def validate(self, original_messages, summary_text):
                recorded_messages.extend(original_messages)
                return ValidationResult(passed=True, validator_used="RecordingValidator")

        conv = Conversation(name="Validator Test", max_tokens=1000)
        conv.add(Message(role="system", content="System prompt"))
        conv.add(Message(role="user", content="Protected instruction", protected=True))
        conv.add(Message(role="assistant", content="Discardable chatter", importance="discardable"))
        conv.add(Message(role="user", content="Normal discussion"))
        conv.add(Message(role="assistant", content="Recent 1"))
        conv.add(Message(role="user", content="Recent 2"))

        compactor = Compactor(
            keep_recent=2,
            summarizer=MockEchoSummarizer(),
            validator=RecordingValidator(),
        )
        result = compactor.compact(conv)

        self.assertTrue(result.committed)
        # Validator should have received ONLY "Normal discussion"
        self.assertEqual(len(recorded_messages), 1)
        self.assertEqual(recorded_messages[0].content, "Normal discussion")

    def test_validation_rollback_with_importance(self):
        """If validation fails and rollback is active, conversation remains untouched."""
        conv = Conversation(name="Rollback Test", max_tokens=1000)
        conv.add(Message(role="system", content="System prompt"))
        conv.add(Message(role="user", content="Keep me", protected=True))
        conv.add(Message(role="assistant", content="Summarize me"))
        conv.add(Message(role="user", content="Recent 1"))
        conv.add(Message(role="assistant", content="Recent 2"))

        initial_count = conv.message_count()
        initial_tokens = conv.total_tokens()

        compactor = Compactor(
            keep_recent=2,
            summarizer=MockEchoSummarizer(),
            validator=MockFailValidator(),
            on_validation_fail="rollback",
        )
        result = compactor.compact(conv)

        self.assertFalse(result.committed)
        self.assertEqual(conv.message_count(), initial_count)
        self.assertEqual(conv.total_tokens(), initial_tokens)
        self.assertEqual(len(result.summarized_messages), 1)
        self.assertEqual(result.summarized_messages[0].content, "Summarize me")

    def test_compaction_result_display_shows_breakdown(self):
        """Verify CompactionResult string output shows Preserved, Summarized, and Removed."""
        conv = Conversation(name="Display Test", max_tokens=1000)
        conv.add(Message(role="system", content="System instruction"))
        conv.add(Message(role="user", content="Protected rule", protected=True))
        conv.add(Message(role="assistant", content="Trash greeting", importance="discardable"))
        conv.add(Message(role="user", content="Important requirement", importance="important"))
        conv.add(Message(role="assistant", content="Recent 1"))
        conv.add(Message(role="user", content="Recent 2"))

        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        s = str(result)
        self.assertIn("Preserved", s)
        self.assertIn("Summarized", s)
        self.assertIn("Removed", s)
        self.assertIn("protected", s)
        self.assertIn("discarded", s)
        self.assertIn("replaced by summary", s)

    def test_mixed_importance_full_breakdown(self):
        """Test full mix: system prompt, protected, critical, discardable, normal, and recent."""
        conv = Conversation(name="Mixed Test", max_tokens=1000)
        m_sys = Message(role="system", content="System Prompt")
        m_prot = Message(role="user", content="Protected Goal", protected=True)
        m_crit = Message(role="assistant", content="Critical Architecture", importance=ImportanceLevel.CRITICAL)
        m_disc = Message(role="user", content="Discardable debug log", importance=ImportanceLevel.DISCARDABLE)
        m_norm1 = Message(role="assistant", content="Normal turn 1", importance=ImportanceLevel.NORMAL)
        m_norm2 = Message(role="user", content="Normal turn 2", importance="important")
        m_rec1 = Message(role="assistant", content="Recent turn 1")
        m_rec2 = Message(role="user", content="Recent turn 2")

        for m in [m_sys, m_prot, m_crit, m_disc, m_norm1, m_norm2, m_rec1, m_rec2]:
            conv.add(m)

        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        self.assertTrue(result.committed)

        # Preserved messages: system prompt, protected, critical, and 2 recent turns = 5
        self.assertEqual(len(result.preserved_messages), 5)
        self.assertIn(m_sys, result.preserved_messages)
        self.assertIn(m_prot, result.preserved_messages)
        self.assertIn(m_crit, result.preserved_messages)
        self.assertIn(m_rec1, result.preserved_messages)
        self.assertIn(m_rec2, result.preserved_messages)

        # Summarized: normal 1 + normal 2 (important) = 2
        self.assertEqual(len(result.summarized_messages), 2)
        self.assertIn(m_norm1, result.summarized_messages)
        self.assertIn(m_norm2, result.summarized_messages)

        # Removed: discardable + normal 1 + normal 2 = 3
        self.assertEqual(len(result.removed_messages), 3)
        self.assertEqual(result.messages_removed, 3)
        self.assertIn(m_disc, result.removed_messages)
        self.assertIn(m_norm1, result.removed_messages)
        self.assertIn(m_norm2, result.removed_messages)

        # Discarded only:
        self.assertEqual(len(result.discarded_messages), 1)
        self.assertIn(m_disc, result.discarded_messages)

        # Total compacted message count: 5 preserved + 1 summary = 6
        self.assertEqual(conv.message_count(), 6)
        self.assertEqual(result.compacted_message_count, 6)

        # Order of messages in conversation:
        # [system_prompt, summary_msg, protected, critical, recent 1, recent 2]
        compacted = conv.get_messages()
        self.assertEqual(compacted[0], m_sys)
        self.assertEqual(compacted[1], result.summary_message)
        self.assertEqual(compacted[2], m_prot)
        self.assertEqual(compacted[3], m_crit)
        self.assertEqual(compacted[4], m_rec1)
        self.assertEqual(compacted[5], m_rec2)

    def test_no_system_prompt_conversation(self):
        """Compactor works when conversation does not begin with a system message."""
        conv = Conversation(name="No System", max_tokens=1000)
        m_prot = Message(role="user", content="Protected user request", protected=True)
        m_norm = Message(role="assistant", content="Normal assistant response")
        m_rec1 = Message(role="user", content="Recent turn 1")
        m_rec2 = Message(role="assistant", content="Recent turn 2")

        for m in [m_prot, m_norm, m_rec1, m_rec2]:
            conv.add(m)

        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        # Compacted: [summary_msg, protected, rec1, rec2]
        self.assertEqual(conv.message_count(), 4)
        messages = conv.get_messages()
        self.assertEqual(messages[0], result.summary_message)
        self.assertEqual(messages[1], m_prot)
        self.assertEqual(messages[2], m_rec1)
        self.assertEqual(messages[3], m_rec2)


if __name__ == "__main__":
    unittest.main()
