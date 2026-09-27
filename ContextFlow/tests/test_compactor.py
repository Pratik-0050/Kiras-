# tests/test_compactor.py
import unittest
from typing import List

from contextflow.message import Message
from contextflow.conversation import Conversation
from contextflow.compactor import Compactor, CompactionResult
from contextflow.summarizers.base import Summarizer, SummarizerError
from contextflow.summarizers.placeholder import PlaceholderSummarizer
from contextflow.validators.base import Validator, ValidationResult, ValidatorError


class MockEchoSummarizer(Summarizer):
    """Simple test summarizer that returns a fixed concise summary."""
    def summarize(self, messages: List[Message]) -> str:
        return f"Summary of {len(messages)} messages."


class MockFailingSummarizer(Summarizer):
    """Test summarizer that always raises SummarizerError."""
    def summarize(self, messages: List[Message]) -> str:
        raise SummarizerError("Provider unavailable.")


class MockPassValidator(Validator):
    """Test validator that always passes."""
    def validate(self, original_messages, summary_text) -> ValidationResult:
        return ValidationResult(passed=True, validator_used="MockPassValidator")


class MockFailValidator(Validator):
    """Test validator that always fails with missing items."""
    def validate(self, original_messages, summary_text) -> ValidationResult:
        return ValidationResult(
            passed=False,
            missing_items=["Requirement: must use Python", "Decision: microservices"],
            validator_used="MockFailValidator",
        )


class MockErrorValidator(Validator):
    """Test validator that raises ValidatorError (infrastructure failure)."""
    def validate(self, original_messages, summary_text) -> ValidationResult:
        raise ValidatorError("Validator API unavailable.")


class TestCompactor(unittest.TestCase):

    # ── initialization ───────────────────────────────────────────────────────

    def test_initialization_validation_keep_recent(self):
        """Verify Compactor validates keep_recent type and value."""
        with self.assertRaises(TypeError):
            Compactor(keep_recent="5")
        with self.assertRaises(ValueError):
            Compactor(keep_recent=0)
        with self.assertRaises(ValueError):
            Compactor(keep_recent=-2)

    def test_initialization_validation_summarizer_type(self):
        with self.assertRaises(TypeError):
            Compactor(summarizer="not a summarizer")

    def test_initialization_validation_validator_type(self):
        with self.assertRaises(TypeError):
            Compactor(validator="not a validator")

    def test_initialization_invalid_on_validation_fail(self):
        with self.assertRaises(ValueError):
            Compactor(on_validation_fail="unknown_mode")

    def test_default_summarizer(self):
        """Verify Compactor defaults to PlaceholderSummarizer."""
        compactor = Compactor(keep_recent=2)
        self.assertIsInstance(compactor.summarizer, PlaceholderSummarizer)

    def test_default_validator_is_none(self):
        compactor = Compactor(keep_recent=2)
        self.assertIsNone(compactor.validator)

    # ── no compaction needed ─────────────────────────────────────────────────

    def test_not_needed_when_messages_few(self):
        """Verify compaction is skipped when messages <= keep_recent."""
        conv = Conversation(name="Short Conv", max_tokens=1000)
        conv.add(Message(role="system", content="System instructions."))
        conv.add(Message(role="user", content="Hello."))
        conv.add(Message(role="assistant", content="Hi."))

        compactor = Compactor(keep_recent=3, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        self.assertFalse(result.was_needed)
        self.assertEqual(conv.message_count(), 3)
        self.assertEqual(result.messages_removed, 0)
        self.assertEqual(result.tokens_saved, 0)
        self.assertIsNone(result.summary_message)
        self.assertTrue(result.committed)

    # ── successful compaction ─────────────────────────────────────────────────

    def test_replaces_older_messages_and_recalculates_tokens(self):
        """Verify compactor replaces older messages with summary and recalculates token usage."""
        conv = Conversation(name="Test Conversation", max_tokens=500)
        conv.add(Message(role="system", content="System prompt: You are a helpful assistant."))
        conv.add(Message(role="user", content="Turn 1: We need to design a feature."))
        conv.add(Message(role="assistant", content="Turn 2: Let's consider architecture options."))
        conv.add(Message(role="user", content="Turn 3: We prefer option B."))
        conv.add(Message(role="assistant", content="Turn 4: Option B is selected and documented."))
        conv.add(Message(role="user", content="Turn 5: What is next?"))
        conv.add(Message(role="assistant", content="Turn 6: Now we implement the models."))

        initial_count  = conv.message_count()
        initial_tokens = conv.total_tokens()

        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result    = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        self.assertTrue(result.committed)
        self.assertEqual(result.original_message_count, initial_count)
        self.assertEqual(result.original_token_count, initial_tokens)
        self.assertEqual(result.messages_removed, 4)
        self.assertIsNone(result.validation_result)  # no validator configured

        messages = conv.get_messages()
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[0].content, "System prompt: You are a helpful assistant.")
        self.assertEqual(messages[1].role, "system")
        self.assertEqual(messages[1].content, "Summary of 4 messages.")
        self.assertEqual(messages[1].metadata.get("type"), "compaction_summary")
        self.assertEqual(messages[1].metadata.get("messages_summarised"), 4)
        self.assertEqual(messages[2].content, "Turn 5: What is next?")
        self.assertEqual(messages[3].content, "Turn 6: Now we implement the models.")

        expected_tokens = sum(m.token_count for m in messages)
        self.assertEqual(conv.total_tokens(), expected_tokens)
        self.assertEqual(result.compacted_token_count, expected_tokens)
        self.assertEqual(result.tokens_saved, initial_tokens - expected_tokens)

    def test_without_leading_system_prompt(self):
        """Verify compactor works when conversation has no leading system prompt."""
        conv = Conversation(name="No System Prompt", max_tokens=500)
        conv.add(Message(role="user", content="Turn 1"))
        conv.add(Message(role="assistant", content="Turn 2"))
        conv.add(Message(role="user", content="Turn 3"))
        conv.add(Message(role="assistant", content="Turn 4"))

        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        self.assertTrue(result.committed)
        messages = conv.get_messages()
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].content, "Turn 3")
        self.assertEqual(messages[2].content, "Turn 4")

    # ── summarizer errors ─────────────────────────────────────────────────────

    def test_summarizer_error_leaves_conversation_unchanged(self):
        """Verify that if the summarizer raises, the conversation is untouched."""
        conv = Conversation(name="Protected Conv", max_tokens=500)
        conv.add(Message(role="user", content="Message 1"))
        conv.add(Message(role="assistant", content="Message 2"))
        conv.add(Message(role="user", content="Message 3"))
        conv.add(Message(role="assistant", content="Message 4"))

        before_messages = [m.content for m in conv.get_messages()]
        compactor = Compactor(keep_recent=2, summarizer=MockFailingSummarizer())

        with self.assertRaises(SummarizerError):
            compactor.compact(conv)

        after_messages = [m.content for m in conv.get_messages()]
        self.assertEqual(before_messages, after_messages)

    def test_fallback_summarizer_used_on_failure(self):
        """Verify that fallback_summarizer is used if primary fails."""
        conv = Conversation(name="Fallback Conv", max_tokens=500)
        conv.add(Message(role="user", content="Message 1"))
        conv.add(Message(role="assistant", content="Message 2"))
        conv.add(Message(role="user", content="Message 3"))
        conv.add(Message(role="assistant", content="Message 4"))

        compactor = Compactor(
            keep_recent=2,
            summarizer=MockFailingSummarizer(),
            fallback_summarizer=MockEchoSummarizer(),
        )
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        self.assertTrue(result.committed)
        self.assertIn("fallback", result.summarizer_used)
        self.assertEqual(conv.get_messages()[0].content, "Summary of 2 messages.")

    # ── validation: passing ───────────────────────────────────────────────────

    def test_validation_pass_allows_compaction(self):
        """Verify validation pass lets compaction proceed normally."""
        conv = Conversation(name="Validated Conv", max_tokens=500)
        conv.add(Message(role="user", content="Message 1"))
        conv.add(Message(role="assistant", content="Message 2"))
        conv.add(Message(role="user", content="Message 3"))
        conv.add(Message(role="assistant", content="Message 4"))

        compactor = Compactor(
            keep_recent=2,
            summarizer=MockEchoSummarizer(),
            validator=MockPassValidator(),
        )
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        self.assertTrue(result.committed)
        self.assertIsNotNone(result.validation_result)
        self.assertTrue(result.validation_result.passed)
        self.assertEqual(conv.message_count(), 3)

    # ── validation: rollback ──────────────────────────────────────────────────

    def test_validation_fail_rollback_leaves_conversation_unchanged(self):
        """Verify that validation failure with rollback leaves conversation intact."""
        conv = Conversation(name="Rollback Conv", max_tokens=500)
        conv.add(Message(role="system", content="System prompt."))
        conv.add(Message(role="user", content="Message 1"))
        conv.add(Message(role="assistant", content="Message 2"))
        conv.add(Message(role="user", content="Message 3"))
        conv.add(Message(role="assistant", content="Message 4"))

        before_contents = [m.content for m in conv.get_messages()]
        before_tokens   = conv.total_tokens()

        compactor = Compactor(
            keep_recent=2,
            summarizer=MockEchoSummarizer(),
            validator=MockFailValidator(),
            on_validation_fail="rollback",
        )
        result = compactor.compact(conv)

        # Result should indicate what happened without committing
        self.assertTrue(result.was_needed)
        self.assertFalse(result.committed)
        self.assertIsNotNone(result.validation_result)
        self.assertFalse(result.validation_result.passed)
        self.assertIn("Requirement: must use Python", result.validation_result.missing_items)

        # Conversation must be completely unchanged
        after_contents = [m.content for m in conv.get_messages()]
        self.assertEqual(before_contents, after_contents)
        self.assertEqual(conv.total_tokens(), before_tokens)
        self.assertEqual(conv.message_count(), 5)

    def test_validation_fail_warn_applies_compaction_anyway(self):
        """Verify on_validation_fail='warn' commits even on validation failure."""
        conv = Conversation(name="Warn Conv", max_tokens=500)
        conv.add(Message(role="user", content="Message 1"))
        conv.add(Message(role="assistant", content="Message 2"))
        conv.add(Message(role="user", content="Message 3"))
        conv.add(Message(role="assistant", content="Message 4"))

        compactor = Compactor(
            keep_recent=2,
            summarizer=MockEchoSummarizer(),
            validator=MockFailValidator(),
            on_validation_fail="warn",
        )
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        self.assertTrue(result.committed)   # committed despite failure
        self.assertFalse(result.validation_result.passed)
        # Conversation WAS modified
        self.assertEqual(conv.message_count(), 3)

    def test_validation_result_in_summary_message_metadata(self):
        """Verify that validation_passed is recorded in summary message metadata."""
        conv = Conversation(name="Meta Conv", max_tokens=500)
        conv.add(Message(role="user", content="Message 1"))
        conv.add(Message(role="assistant", content="Message 2"))
        conv.add(Message(role="user", content="Message 3"))
        conv.add(Message(role="assistant", content="Message 4"))

        compactor = Compactor(
            keep_recent=2,
            summarizer=MockEchoSummarizer(),
            validator=MockPassValidator(),
        )
        result = compactor.compact(conv)

        self.assertIsNotNone(result.summary_message)
        self.assertTrue(result.summary_message.metadata.get("validation_passed"))

    def test_validator_error_propagates(self):
        """Verify ValidatorError from the validator propagates to the caller."""
        conv = Conversation(name="Error Conv", max_tokens=500)
        conv.add(Message(role="user", content="Message 1"))
        conv.add(Message(role="assistant", content="Message 2"))
        conv.add(Message(role="user", content="Message 3"))
        conv.add(Message(role="assistant", content="Message 4"))

        compactor = Compactor(
            keep_recent=2,
            summarizer=MockEchoSummarizer(),
            validator=MockErrorValidator(),
        )
        with self.assertRaises(ValidatorError):
            compactor.compact(conv)

    def test_compaction_result_str_shows_validation(self):
        """Verify CompactionResult.__str__ includes validation info."""
        vr = ValidationResult(
            passed=False,
            missing_items=["Decision X"],
            validator_used="MockFailValidator",
        )
        result = CompactionResult(
            original_message_count=5,
            original_token_count=200,
            compacted_message_count=5,
            compacted_token_count=200,
            messages_removed=3,
            tokens_saved=0,
            summary_message=None,
            was_needed=True,
            committed=False,
            summarizer_used="MockEchoSummarizer",
            validation_result=vr,
        )
        s = str(result)
        self.assertIn("FAILED", s)
        self.assertIn("Decision X", s)
        self.assertIn("ROLLED BACK", s)


if __name__ == "__main__":
    unittest.main()
