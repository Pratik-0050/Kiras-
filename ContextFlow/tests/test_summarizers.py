# tests/test_summarizers.py
import os
import unittest
from unittest.mock import MagicMock, patch

from contextflow.message import Message
from contextflow.summarizers.base import Summarizer, BaseSummarizer, SummarizerError
from contextflow.summarizers.placeholder import PlaceholderSummarizer
from contextflow.summarizers.openai_summarizer import OpenAISummarizer


class TestSummarizers(unittest.TestCase):
    def test_summarizer_abc(self):
        """Verify Summarizer is an abstract class and BaseSummarizer is an alias."""
        self.assertIs(Summarizer, BaseSummarizer)
        with self.assertRaises(TypeError):
            Summarizer()  # Cannot instantiate abstract class

    def test_placeholder_summarizer(self):
        """Verify PlaceholderSummarizer produces structured text without external calls."""
        summarizer = PlaceholderSummarizer()
        self.assertIsInstance(summarizer, Summarizer)

        # Empty messages
        self.assertEqual(summarizer.summarize([]), "")

        # Non-empty messages
        msgs = [
            Message(role="user", content="Hello world"),
            Message(role="assistant", content="Hi there! How can I help?"),
        ]
        result = summarizer.summarize(msgs)
        self.assertIn("[CONTEXT SUMMARY -- 2 older message(s) compacted]", result)
        self.assertIn("[USER] Hello world", result)
        self.assertIn("[ASSISTANT] Hi there! How can I help?", result)
        self.assertIn("[End of summary. Conversation continues below.]", result)

    def test_openai_summarizer_missing_api_key(self):
        """Verify SummarizerError is raised when no API key is provided or found in env."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SummarizerError) as ctx:
                OpenAISummarizer(api_key="")
            self.assertIn("No API key found", str(ctx.exception))

    def test_openai_summarizer_env_var_config(self):
        """Verify OpenAISummarizer reads OPENAI_API_KEY, OPENAI_MODEL, and OPENAI_BASE_URL."""
        env_vars = {
            "OPENAI_API_KEY": "env-test-key",
            "OPENAI_MODEL": "gpt-custom-model",
            "OPENAI_BASE_URL": "https://custom.endpoint.com/v1",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            # Pass a mock client so network initialization isn't attempted
            mock_client = MagicMock()
            summarizer = OpenAISummarizer(client=mock_client)
            self.assertEqual(summarizer.api_key, "env-test-key")
            self.assertEqual(summarizer.model, "gpt-custom-model")
            self.assertEqual(summarizer.base_url, "https://custom.endpoint.com/v1")

    def test_openai_summarizer_explicit_args_override_env(self):
        """Verify constructor arguments override environment variables."""
        env_vars = {
            "OPENAI_API_KEY": "env-key",
            "OPENAI_MODEL": "env-model",
            "OPENAI_BASE_URL": "https://env.endpoint/v1",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            mock_client = MagicMock()
            summarizer = OpenAISummarizer(
                api_key="arg-key",
                model="arg-model",
                base_url="https://arg.endpoint/v1",
                client=mock_client,
            )
            self.assertEqual(summarizer.api_key, "arg-key")
            self.assertEqual(summarizer.model, "arg-model")
            self.assertEqual(summarizer.base_url, "https://arg.endpoint/v1")

    def test_openai_summarizer_mock_success(self):
        """Verify summarize() formats messages into prompt and returns formatted summary."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "User requested a Python agent. Architecture agreed."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        summarizer = OpenAISummarizer(
            api_key="dummy-key",
            model="gpt-4o-mini",
            client=mock_client,
        )

        msgs = [
            Message(role="user", content="We want an AI agent."),
            Message(role="assistant", content="Let's build it with a 3-layer architecture."),
        ]
        result = summarizer.summarize(msgs)

        # Ensure API was called with system and user prompts
        self.assertTrue(mock_client.chat.completions.create.called)
        kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        messages_sent = kwargs["messages"]
        self.assertEqual(messages_sent[0]["role"], "system")
        self.assertIn("Requirements", messages_sent[0]["content"])
        self.assertIn("Decisions", messages_sent[0]["content"])
        self.assertIn("Important facts", messages_sent[0]["content"])
        self.assertIn("Unresolved tasks", messages_sent[0]["content"])
        self.assertIn("Relevant context", messages_sent[0]["content"])

        self.assertEqual(messages_sent[1]["role"], "user")
        self.assertIn("[USER]: We want an AI agent.", messages_sent[1]["content"])
        self.assertIn("[ASSISTANT]: Let's build it with a 3-layer architecture.", messages_sent[1]["content"])

        # Result has header and footer
        self.assertIn("[CONTEXT SUMMARY -- 2 older message(s) compacted by LLM]", result)
        self.assertIn("User requested a Python agent. Architecture agreed.", result)
        self.assertIn("[End of summary. Conversation continues below.]", result)

    def test_openai_summarizer_empty_messages(self):
        """Verify summarize() returns empty string immediately for empty message list."""
        mock_client = MagicMock()
        summarizer = OpenAISummarizer(api_key="dummy-key", client=mock_client)
        result = summarizer.summarize([])
        self.assertEqual(result, "")
        mock_client.chat.completions.create.assert_not_called()

    def test_openai_summarizer_api_exception(self):
        """Verify SummarizerError is raised on API network/server errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Connection timeout")
        summarizer = OpenAISummarizer(api_key="dummy-key", client=mock_client)

        msgs = [Message(role="user", content="Test message")]
        with self.assertRaises(SummarizerError) as ctx:
            summarizer.summarize(msgs)
        self.assertIn("OpenAI API call failed: Connection timeout", str(ctx.exception))

    def test_openai_summarizer_empty_response(self):
        """Verify SummarizerError is raised if the API returns empty text or null content."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "   "  # whitespace
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        summarizer = OpenAISummarizer(api_key="dummy-key", client=mock_client)
        msgs = [Message(role="user", content="Test message")]
        with self.assertRaises(SummarizerError) as ctx:
            summarizer.summarize(msgs)
        self.assertIn("The API returned an empty summary", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
