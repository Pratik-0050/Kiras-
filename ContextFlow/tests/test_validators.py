# tests/test_validators.py
"""
Unit tests for the validators subsystem.
Covers:
  - ValidationResult data model
  - Validator ABC (cannot be instantiated directly)
  - HeuristicValidator: threshold logic, candidate extraction, edge cases
  - OpenAIValidator: mock API success, failure, malformed JSON, empty summary
"""

import os
import json
import unittest
from unittest.mock import MagicMock

from contextflow.message import Message
from contextflow.validators.base import (
    Validator, BaseValidator, ValidationResult, ValidatorError,
)
from contextflow.validators.heuristic_validator import HeuristicValidator
from contextflow.validators.openai_validator import OpenAIValidator


# ── ValidationResult ────────────────────────────────────────────────────────

class TestValidationResult(unittest.TestCase):
    def test_defaults(self):
        vr = ValidationResult(passed=True)
        self.assertTrue(vr.passed)
        self.assertEqual(vr.warnings, [])
        self.assertEqual(vr.missing_items, [])
        self.assertEqual(vr.validator_used, "none")
        self.assertEqual(vr.details, "")

    def test_has_warnings(self):
        vr = ValidationResult(passed=True, warnings=["some warning"])
        self.assertTrue(vr.has_warnings)

    def test_has_missing_items(self):
        vr = ValidationResult(passed=False, missing_items=["Decision X"])
        self.assertTrue(vr.has_missing_items)

    def test_str_passed(self):
        vr = ValidationResult(passed=True, validator_used="TestValidator")
        s = str(vr)
        self.assertIn("PASSED", s)
        self.assertIn("TestValidator", s)

    def test_str_failed_with_items(self):
        vr = ValidationResult(
            passed=False,
            missing_items=["Requirement A", "Decision B"],
            validator_used="TestValidator",
        )
        s = str(vr)
        self.assertIn("FAILED", s)
        self.assertIn("Requirement A", s)
        self.assertIn("Decision B", s)


# ── Validator ABC ────────────────────────────────────────────────────────────

class TestValidatorABC(unittest.TestCase):
    def test_cannot_instantiate_directly(self):
        with self.assertRaises(TypeError):
            Validator()

    def test_base_validator_alias(self):
        self.assertIs(Validator, BaseValidator)

    def test_custom_validator_works(self):

        class AlwaysPass(Validator):
            def validate(self, original_messages, summary_text):
                return ValidationResult(passed=True, validator_used="AlwaysPass")

        v = AlwaysPass()
        result = v.validate([], "summary text")
        self.assertTrue(result.passed)


# ── HeuristicValidator ───────────────────────────────────────────────────────

class TestHeuristicValidator(unittest.TestCase):
    def _msgs(self, *contents):
        """Helper: build a list of user Messages from strings."""
        return [Message(role="user", content=c) for c in contents]

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            HeuristicValidator(fail_threshold=0.0)
        with self.assertRaises(ValueError):
            HeuristicValidator(warn_threshold=0.0)
        with self.assertRaises(ValueError):
            HeuristicValidator(warn_threshold=0.8, fail_threshold=0.5)
        with self.assertRaises(ValueError):
            HeuristicValidator(min_candidates=0)

    def test_empty_messages_passes(self):
        v = HeuristicValidator()
        result = v.validate([], "Some summary text.")
        self.assertTrue(result.passed)
        self.assertIn("No messages", result.details)

    def test_empty_summary_fails(self):
        msgs = self._msgs("The system must use Python.", "We decided on microservices.")
        v = HeuristicValidator(min_candidates=1)
        result = v.validate(msgs, "")
        self.assertFalse(result.passed)
        self.assertTrue(result.has_missing_items)

    def test_whitespace_summary_fails(self):
        msgs = self._msgs("Requirement: must use Python.")
        v = HeuristicValidator(min_candidates=1)
        result = v.validate(msgs, "   ")
        self.assertFalse(result.passed)

    def test_passes_when_all_keywords_present(self):
        # Use a high fail_threshold so that missing sentences (which summaries
        # paraphrase rather than quote verbatim) don't cause a failure.
        # The key quoted term "Python" appears in the summary, so at most
        # the full-sentence candidates are missing (which is expected).
        msgs = self._msgs(
            'We must use "Python" for the backend.',
            'The decision was to adopt "PostgreSQL" for persistence.',
        )
        # Summary reproduces the quoted terms exactly.
        summary = 'Python is used for the backend. PostgreSQL handles data persistence.'
        # Only quoted strings are "must-find" candidates here.
        # Even if action sentences are missing, fail_threshold=0.8 means we
        # tolerate up to 80% missing (sentences rarely appear verbatim).
        v = HeuristicValidator(fail_threshold=0.8, min_candidates=1)
        result = v.validate(msgs, summary)
        self.assertTrue(result.passed)

    def test_fails_when_too_many_keywords_missing(self):
        msgs = self._msgs(
            "We must use Redis for caching.",
            "The system must support PostgreSQL version 14.",
            "We decided to use Kubernetes for deployment.",
            "Critical: authentication must use OAuth 2.0.",
        )
        # Summary that mentions nothing relevant
        summary = "The conversation discussed various technical aspects."
        v = HeuristicValidator(fail_threshold=0.4, min_candidates=1)
        result = v.validate(msgs, summary)
        self.assertFalse(result.passed)
        self.assertTrue(result.has_missing_items)

    def test_warning_when_approaching_threshold(self):
        # Construct a scenario where missing_ratio > warn_threshold but <= fail_threshold.
        # We use quoted strings as candidates since they are exact-match friendly.
        msgs = [
            Message(role="user",      content='Use "Redis" for caching.'),
            Message(role="assistant", content='Use "PostgreSQL" for storage.'),
            Message(role="user",      content='We also need "Nginx" as a proxy.'),
            Message(role="assistant", content='And "Docker" for containerization.'),
        ]
        # Summary only covers Redis -- PostgreSQL, Nginx, Docker are missing.
        # Quoted candidates: Redis, PostgreSQL, Nginx, Docker  -> 4 total
        # Missing: PostgreSQL, Nginx, Docker  -> 3/4 = 75%
        # fail_threshold=0.9 (won't fail), warn_threshold=0.4 (will warn)
        summary = "Redis is used for caching."
        v = HeuristicValidator(fail_threshold=0.9, warn_threshold=0.4, min_candidates=1)
        result = v.validate(msgs, summary)
        self.assertTrue(result.passed)
        self.assertTrue(result.has_warnings)

    def test_insufficient_candidates_skips_check(self):
        # Very short messages produce few candidates
        msgs = self._msgs("hi", "ok")
        v = HeuristicValidator(min_candidates=5)
        result = v.validate(msgs, "A brief summary.")
        self.assertTrue(result.passed)
        self.assertTrue(result.has_warnings)
        self.assertIn("Insufficient", result.details)

    def test_validator_used_name(self):
        v = HeuristicValidator()
        result = v.validate([], "")
        # empty messages → early pass
        self.assertEqual(result.validator_used, "HeuristicValidator")


# ── OpenAIValidator ──────────────────────────────────────────────────────────

class TestOpenAIValidator(unittest.TestCase):
    def _msgs(self, *contents):
        return [Message(role="user", content=c) for c in contents]

    def _make_mock_response(self, content: str) -> MagicMock:
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    def test_missing_api_key_raises(self):
        import unittest.mock as mock
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidatorError) as ctx:
                OpenAIValidator(api_key="")
            self.assertIn("No API key found", str(ctx.exception))

    def test_empty_messages_passes_without_api_call(self):
        mock_client = MagicMock()
        v = OpenAIValidator(api_key="key", client=mock_client)
        result = v.validate([], "summary")
        self.assertTrue(result.passed)
        mock_client.chat.completions.create.assert_not_called()

    def test_empty_summary_fails_without_api_call(self):
        mock_client = MagicMock()
        v = OpenAIValidator(api_key="key", client=mock_client)
        msgs = self._msgs("Important requirement here.")
        result = v.validate(msgs, "")
        self.assertFalse(result.passed)
        mock_client.chat.completions.create.assert_not_called()

    def test_passed_verdict(self):
        mock_client = MagicMock()
        verdict_json = json.dumps({"passed": True, "missing": []})
        mock_client.chat.completions.create.return_value = (
            self._make_mock_response(verdict_json)
        )
        v = OpenAIValidator(api_key="key", client=mock_client)
        msgs = self._msgs("We must use Python.")
        result = v.validate(msgs, "Python is the language of choice.")
        self.assertTrue(result.passed)
        self.assertEqual(result.missing_items, [])

    def test_failed_verdict_with_missing_items(self):
        mock_client = MagicMock()
        verdict = {
            "passed": False,
            "missing": [
                "Requirement: system must use Python",
                "Decision: GitHub App for authentication",
            ]
        }
        mock_client.chat.completions.create.return_value = (
            self._make_mock_response(json.dumps(verdict))
        )
        v = OpenAIValidator(api_key="key", client=mock_client)
        msgs = self._msgs("We must use Python.", "GitHub App for auth.")
        result = v.validate(msgs, "Generic summary.")
        self.assertFalse(result.passed)
        self.assertEqual(len(result.missing_items), 2)
        self.assertIn("Requirement: system must use Python", result.missing_items)

    def test_strips_markdown_fences(self):
        mock_client = MagicMock()
        fenced = "```json\n" + json.dumps({"passed": True, "missing": []}) + "\n```"
        mock_client.chat.completions.create.return_value = (
            self._make_mock_response(fenced)
        )
        v = OpenAIValidator(api_key="key", client=mock_client)
        result = v.validate(self._msgs("Hello."), "Summary text.")
        self.assertTrue(result.passed)

    def test_malformed_json_raises_validator_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            self._make_mock_response("NOT JSON AT ALL")
        )
        v = OpenAIValidator(api_key="key", client=mock_client)
        with self.assertRaises(ValidatorError) as ctx:
            v.validate(self._msgs("Some message."), "Some summary.")
        self.assertIn("non-JSON", str(ctx.exception))

    def test_api_exception_raises_validator_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Timeout")
        v = OpenAIValidator(api_key="key", client=mock_client)
        with self.assertRaises(ValidatorError) as ctx:
            v.validate(self._msgs("Some message."), "Some summary.")
        self.assertIn("API call failed", str(ctx.exception))

    def test_env_var_config(self):
        import unittest.mock as mock
        env = {
            "OPENAI_API_KEY":  "env-key",
            "OPENAI_MODEL":    "gpt-custom",
            "OPENAI_BASE_URL": "https://custom.host/v1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            mock_client = MagicMock()
            v = OpenAIValidator(client=mock_client)
            self.assertEqual(v.api_key, "env-key")
            self.assertEqual(v.model, "gpt-custom")
            self.assertEqual(v.base_url, "https://custom.host/v1")

    def test_prompt_contains_original_messages_and_summary(self):
        """Verify both the transcript and summary are included in the user prompt."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            self._make_mock_response(json.dumps({"passed": True, "missing": []}))
        )
        v = OpenAIValidator(api_key="key", client=mock_client)
        msgs = self._msgs("We need feature X.", "Use Redis for caching.")
        v.validate(msgs, "Feature X needed. Redis for caching.")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        user_msg_content = call_kwargs["messages"][1]["content"]
        self.assertIn("[USER]: We need feature X.", user_msg_content)
        self.assertIn("[USER]: Use Redis for caching.", user_msg_content)
        self.assertIn("Feature X needed. Redis for caching.", user_msg_content)


if __name__ == "__main__":
    unittest.main()
