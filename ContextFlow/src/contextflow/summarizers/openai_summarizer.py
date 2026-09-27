# contextflow/summarizers/openai_summarizer.py
"""
OpenAISummarizer: calls any OpenAI-compatible chat API to summarize
a list of messages into a compact, information-dense summary.

Configuration (via environment variables or constructor args):
    OPENAI_API_KEY  -- Required. Your API key.
    OPENAI_MODEL    -- Optional. Defaults to "gpt-4o-mini".
    OPENAI_BASE_URL -- Optional. Defaults to "https://api.openai.com/v1".
                      Override to use any OpenAI-compatible endpoint
                      (e.g. Azure OpenAI, Groq, local Ollama, OpenRouter, etc.)

The summarizer formats older messages into a prompt and asks the LLM
to produce a concise summary preserving:
    - User requirements, goals, and constraints
    - Decisions made or agreed upon
    - Key facts, concepts, definitions, and state
    - Unresolved tasks, open questions, and next steps
    - Relevant context needed to continue the conversation seamlessly
"""

import os
from typing import List, Optional, Any, TYPE_CHECKING

from .base import Summarizer, SummarizerError

if TYPE_CHECKING:
    from ..message import Message

# Automatically load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default values used when env vars are not set.
_DEFAULT_MODEL    = "gpt-4o-mini"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# System prompt sent to the LLM to guide its summary.
_SYSTEM_PROMPT = """\
You are a context compaction assistant for an LLM conversation manager.

You will receive a sequence of older conversation messages that need to be
compressed to save token space. Your job is to write a single, concise
summary of those messages.

The summary MUST preserve:
- Requirements: all user requirements, goals, constraints, and preferences
- Decisions: all architectural, technical, or design decisions made or agreed upon
- Important facts: key facts, concepts, definitions, configurations, and state
- Unresolved tasks: any open tasks, pending questions, or next steps
- Relevant context: any technical background needed to continue the conversation seamlessly

Rules:
- Write in clear, dense prose (avoid unnecessary bullet points unless summarizing a list)
- Be as concise as possible while retaining all essential facts and decisions
- Do NOT include conversational filler, pleasantries, or meta-commentary
- Do NOT describe the conversation mechanics (e.g., avoid "the user asked" or "the assistant replied")
- Output ONLY the summary text, nothing else
"""


class OpenAISummarizer(Summarizer):
    """Summarizes messages using an OpenAI-compatible chat completions API.

    Args:
        api_key:     API key. Defaults to OPENAI_API_KEY environment variable.
        model:       Model name. Defaults to OPENAI_MODEL env var or "gpt-4o-mini".
        base_url:    API base URL. Defaults to OPENAI_BASE_URL env var or "https://api.openai.com/v1".
        temperature: Sampling temperature for generation (default: 0.2).
        client:      Optional pre-configured OpenAI client (useful for testing or custom proxies).
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       Optional[str] = None,
        base_url:    Optional[str] = None,
        temperature: float = 0.2,
        client:      Optional[Any] = None,
    ) -> None:
        env_key = os.getenv("OPENAI_API_KEY", "").strip()
        env_model = os.getenv("OPENAI_MODEL", "").strip()
        env_base_url = os.getenv("OPENAI_BASE_URL", "").strip()

        self.api_key: str = (api_key if api_key is not None else env_key).strip()
        self.model: str = (model if model is not None else (env_model or _DEFAULT_MODEL)).strip()
        self.base_url: str = (base_url if base_url is not None else (env_base_url or _DEFAULT_BASE_URL)).strip()
        self.temperature: float = temperature

        if client is not None:
            self._client = client
        else:
            if not self.api_key:
                raise SummarizerError(
                    "No API key found. "
                    "Set the OPENAI_API_KEY environment variable or pass api_key=... "
                    "to OpenAISummarizer()."
                )

            # Import openai lazily so the rest of ContextFlow works even if
            # the package is not installed.
            try:
                import openai as _openai
                self._client = _openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise SummarizerError(
                    "The 'openai' package is not installed. "
                    "Run: pip install openai"
                )

    def summarize(self, messages: "List[Message]") -> str:
        """Call the LLM and return a concise summary of *messages*.

        Args:
            messages: Older Message objects to compress.

        Returns:
            A prose summary string.

        Raises:
            SummarizerError: On API errors, network problems, invalid configuration,
                             or empty responses.
        """
        if not messages:
            return ""

        # Format older messages into a readable transcript for the LLM.
        transcript_lines = []
        for msg in messages:
            transcript_lines.append(f"[{msg.role.upper()}]: {msg.content}")
        transcript = "\n\n".join(transcript_lines)

        user_prompt = (
            f"Please summarize the following {len(messages)} conversation "
            f"message(s) into a compact context summary preserving all requirements, "
            f"decisions, important facts, unresolved tasks, and relevant context:\n\n{transcript}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT.strip()},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=self.temperature,
            )
        except Exception as exc:
            raise SummarizerError(
                f"OpenAI API call failed: {exc}"
            ) from exc

        # Extract and validate the summary text from the response.
        if not response or not getattr(response, "choices", None):
            raise SummarizerError(
                "OpenAI API returned an invalid response structure (no choices found)."
            )

        try:
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message else None
        except (IndexError, AttributeError) as exc:
            raise SummarizerError(
                f"Unexpected API response format: {exc}"
            ) from exc

        if content is None:
            raise SummarizerError(
                "The model returned null content (content may have been filtered or refused)."
            )

        summary = content.strip()
        if not summary:
            raise SummarizerError(
                "The API returned an empty summary. "
                "Try a different model or check your prompt."
            )

        # Wrap with a header and footer so it is clear this is a compacted block.
        header = f"[CONTEXT SUMMARY -- {len(messages)} older message(s) compacted by LLM]"
        footer = "[End of summary. Conversation continues below.]"
        return f"{header}\n\n{summary}\n\n{footer}"
