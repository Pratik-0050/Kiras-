# tests/test_scorers.py
"""
Unit tests for Step 10: Automatic Context Priority Scoring.
"""

import unittest
from typing import List

from contextflow import (
    Message,
    Conversation,
    Compactor,
    ImportanceLevel,
    PriorityScorer,
    BasePriorityScorer,
    PriorityScore,
    ScorerError,
    HeuristicPriorityScorer,
)
from contextflow.summarizers.base import Summarizer


class MockEchoSummarizer(Summarizer):
    def summarize(self, messages: List[Message]) -> str:
        return "Summary of " + str(len(messages)) + " messages."


class TestPriorityScoreDataclass(unittest.TestCase):
    """Test PriorityScore dataclass and clamping behavior."""

    def test_score_clamping_high_and_low(self):
        msg = Message(role="user", content="Hello")
        ps_high = PriorityScore(score=150.0, classification=ImportanceLevel.CRITICAL, message=msg)
        self.assertEqual(ps_high.score, 100.0)

        ps_low = PriorityScore(score=-25.0, classification=ImportanceLevel.DISCARDABLE, message=msg)
        self.assertEqual(ps_low.score, 0.0)

    def test_string_classification_coercion(self):
        msg = Message(role="user", content="Hello")
        ps = PriorityScore(score=75.0, classification="important", message=msg)  # type: ignore
        self.assertEqual(ps.classification, ImportanceLevel.IMPORTANT)

    def test_priority_score_str(self):
        msg = Message(role="user", content="Requirement: Must support Python 3.12", protected=True)
        ps = PriorityScore(score=100.0, classification=ImportanceLevel.CRITICAL, message=msg)
        s = str(ps)
        self.assertIn("100.0", s)
        self.assertIn("critical", s)
        self.assertIn("[protected]", s)
        self.assertIn("Requirement: Must support", s)


class TestHeuristicPriorityScorer(unittest.TestCase):
    """Test HeuristicPriorityScorer factors, rules, and classifications."""

    def setUp(self):
        self.scorer = HeuristicPriorityScorer()

    def test_protected_message_always_scores_100_critical(self):
        msg = Message(role="user", content="Trivial greeting", protected=True)
        ps = self.scorer.score(msg, index=0, total_messages=10)
        self.assertEqual(ps.score, 100.0)
        self.assertEqual(ps.classification, ImportanceLevel.CRITICAL)
        self.assertIn("protected", ps.reason.lower())

    def test_critical_importance_scores_high(self):
        msg = Message(role="assistant", content="Normal architecture response", importance="critical")
        ps = self.scorer.score(msg, index=0, total_messages=10)
        self.assertGreaterEqual(ps.score, 90.0)
        self.assertEqual(ps.classification, ImportanceLevel.CRITICAL)

    def test_discardable_importance_scores_low(self):
        msg = Message(role="user", content="Ping healthcheck 200 ok", importance="discardable")
        ps = self.scorer.score(msg, index=0, total_messages=10)
        self.assertLessEqual(ps.score, 25.0)
        self.assertEqual(ps.classification, ImportanceLevel.DISCARDABLE)

    def test_role_weighting(self):
        """System prompt should score higher than user, which scores higher than assistant, which scores higher than tool."""
        text = "Database configuration parameters"
        ps_sys = self.scorer.score(Message(role="system", content=text), index=5, total_messages=10)
        ps_usr = self.scorer.score(Message(role="user", content=text), index=5, total_messages=10)
        ps_ast = self.scorer.score(Message(role="assistant", content=text), index=5, total_messages=10)
        ps_tol = self.scorer.score(Message(role="tool", content=text), index=5, total_messages=10)

        self.assertGreater(ps_sys.score, ps_usr.score)
        self.assertGreater(ps_usr.score, ps_ast.score)
        self.assertGreater(ps_ast.score, ps_tol.score)

    def test_recency_weighting(self):
        """Later message in conversation trajectory gets higher recency points."""
        msg = Message(role="user", content="Discuss feature design")
        ps_early = self.scorer.score(msg, index=0, total_messages=10)
        ps_late  = self.scorer.score(msg, index=9, total_messages=10)

        self.assertGreater(ps_late.score, ps_early.score)
        self.assertGreater(ps_late.factors["recency"], ps_early.factors["recency"])

    def test_high_importance_keywords_boost_score(self):
        """Keywords like 'requirement', 'must', 'security' boost score."""
        msg_plain = Message(role="user", content="We should discuss the application layout.")
        msg_vital = Message(role="user", content="CRITICAL REQUIREMENT: The system must enforce security compliance.")

        ps_plain = self.scorer.score(msg_plain, index=2, total_messages=10)
        ps_vital = self.scorer.score(msg_vital, index=2, total_messages=10)

        self.assertGreater(ps_vital.score, ps_plain.score)
        self.assertGreater(ps_vital.factors["content"], ps_plain.factors["content"])

    def test_code_blocks_boost_score(self):
        """Code blocks in content award additional content points."""
        msg_code = Message(role="assistant", content="Here is the implementation:\n```python\ndef run():\n    pass\n```")
        msg_text = Message(role="assistant", content="Here is the implementation in Python.")

        ps_code = self.scorer.score(msg_code, index=3, total_messages=10)
        ps_text = self.scorer.score(msg_text, index=3, total_messages=10)

        self.assertGreater(ps_code.factors["content"], ps_text.factors["content"])

    def test_ephemeral_and_pleasantry_terms_penalize_score(self):
        """Trivial greetings, healthchecks, and debug logs get penalized."""
        msg_ack = Message(role="user", content="ok thanks")
        msg_dbg = Message(role="assistant", content="Debug trace: healthcheck ping 200 ok")

        ps_ack = self.scorer.score(msg_ack, index=2, total_messages=10)
        ps_dbg = self.scorer.score(msg_dbg, index=2, total_messages=10)

        self.assertEqual(ps_ack.classification, ImportanceLevel.DISCARDABLE)
        self.assertEqual(ps_dbg.classification, ImportanceLevel.DISCARDABLE)

    def test_score_messages_returns_full_list(self):
        messages = [
            Message(role="system", content="System Prompt"),
            Message(role="user", content="Question 1"),
            Message(role="assistant", content="Answer 1"),
        ]
        scores = self.scorer.score_messages(messages)
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores[0].message, messages[0])
        self.assertEqual(scores[1].message, messages[1])
        self.assertEqual(scores[2].message, messages[2])

    def test_invalid_thresholds_raise_value_error(self):
        with self.assertRaises(ValueError):
            HeuristicPriorityScorer(critical_threshold=50.0, important_threshold=60.0)

        with self.assertRaises(ValueError):
            HeuristicPriorityScorer(important_threshold=30.0, normal_threshold=40.0)


class TestCompactorWithScorer(unittest.TestCase):
    """Test Compactor integration with PriorityScorer."""

    def test_compactor_uses_heuristic_scorer_by_default(self):
        compactor = Compactor()
        self.assertIsInstance(compactor.scorer, HeuristicPriorityScorer)

    def test_compactor_rejects_invalid_scorer_type(self):
        with self.assertRaises(TypeError):
            Compactor(scorer="not_a_scorer")  # type: ignore

    def test_custom_scorer_injection(self):
        class InvertScorer(PriorityScorer):
            def score(self, message, index=0, total_messages=1):
                return PriorityScore(score=50.0, classification=ImportanceLevel.NORMAL, message=message)

        compactor = Compactor(scorer=InvertScorer())
        self.assertIsInstance(compactor.scorer, InvertScorer)

    def test_auto_classification_protects_critical_scored_message(self):
        """A message scoring >= 85 (e.g. system prompt or vital requirement) is preserved."""
        conv = Conversation(name="Auto Critical", max_tokens=1000)
        conv.add(Message(role="system", content="System instruction"))
        # Unprotected user message with strong critical keywords that score >= 85
        conv.add(Message(
            role="user",
            content="CRITICAL SECURITY REQUIREMENT: Never store unencrypted passwords or secrets in database.",
        ))
        conv.add(Message(role="assistant", content="Understood, all secrets are hashed with argon2."))
        conv.add(Message(role="user", content="ok thanks"))  # Trivial pleasantry -> discardable
        conv.add(Message(role="assistant", content="Recent turn 1"))
        conv.add(Message(role="user", content="Recent turn 2"))

        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        self.assertTrue(result.was_needed)
        # Verify priority scores populated
        self.assertEqual(len(result.priority_scores), 6)

        # The vital requirement should have been scored as CRITICAL and preserved
        preserved_contents = [m.content for m in result.preserved_messages]
        self.assertTrue(any("CRITICAL SECURITY REQUIREMENT" in c for c in preserved_contents))

        # "ok thanks" should have been classified as DISCARDABLE and removed/discarded
        discarded_contents = [m.content for m in result.discarded_messages]
        self.assertTrue(any("ok thanks" in c for c in discarded_contents))

    def test_priority_scores_displayed_in_compaction_result_str(self):
        """Verify str(result) contains priority scores and classifications."""
        conv = Conversation(name="Score Display", max_tokens=1000)
        conv.add(Message(role="system", content="System prompt"))
        conv.add(Message(role="user", content="CRITICAL REQUIREMENT: Data must be encrypted.", protected=True))
        conv.add(Message(role="assistant", content="Debug trace: ping 200 ok"))
        conv.add(Message(role="user", content="Normal turn to summarize"))
        conv.add(Message(role="assistant", content="Recent 1"))
        conv.add(Message(role="user", content="Recent 2"))

        compactor = Compactor(keep_recent=2, summarizer=MockEchoSummarizer())
        result = compactor.compact(conv)

        s = str(result)
        self.assertIn("Priority Scores", s)
        self.assertIn("100.0/100 (critical)", s)
        self.assertIn("(discardable)", s)
        self.assertIn("Preserved", s)
        self.assertIn("Summarized", s)
        self.assertIn("Removed", s)

    def test_base_priority_scorer_alias(self):
        self.assertIs(BasePriorityScorer, PriorityScorer)

    def test_custom_role_weights(self):
        custom_weights = {"user": 40.0, "system": 10.0, "assistant": 5.0, "tool": 0.0}
        scorer = HeuristicPriorityScorer(role_weights=custom_weights)
        msg_user = Message(role="user", content="Test message")
        msg_sys = Message(role="system", content="Test message")

        ps_user = scorer.score(msg_user)
        ps_sys = scorer.score(msg_sys)
        self.assertGreater(ps_user.score, ps_sys.score)
        self.assertEqual(ps_user.factors["role"], 40.0)

    def test_scorer_error_exception(self):
        err = ScorerError("Scoring failure")
        self.assertIsInstance(err, Exception)
        self.assertEqual(str(err), "Scoring failure")

    def test_factors_dictionary_populated(self):
        scorer = HeuristicPriorityScorer()
        msg = Message(role="user", content="Discuss deployment strategy")
        ps = scorer.score(msg, index=1, total_messages=5)
        self.assertIn("base", ps.factors)
        self.assertIn("role", ps.factors)
        self.assertIn("recency", ps.factors)
        self.assertIn("importance", ps.factors)
        self.assertIn("content", ps.factors)


if __name__ == "__main__":
    unittest.main()
