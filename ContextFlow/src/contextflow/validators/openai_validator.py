# contextflow/validators/openai_validator.py
"""
OpenAIValidator: calls any OpenAI-compatible chat API to judge whether a
summary adequately preserves the important information from the original
conversation messages.

The validator sends the original messages and the generated summary to the LLM
and asks it to:
    - Identify requirements, decisions, facts, and tasks from the originals
    - Check which (if any) are absent from the summary
    - Return a structured JSON verdict: { "passed": bool, "missing": [...] }

Configuration (via environment variables or constructor args) -- same as OpenAISummarizer:
    OPENAI_API_KEY  -- Required. Your API key.
    OPENAI_MODEL    -- Optional. Defaults to "gpt-4o-mini".
    OPENAI_BASE_URL -- Optional. Defaults to "https://api.openai.com/v1".
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, TYPE_CHECKING

from .base import Validator, ValidationResult, ValidatorError

if TYPE_CHECKING:
    from ..message import Message

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_DEFAULT_MODEL    = "gpt-4o-mini"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"

_SYSTEM_PROMPT = """\
You are a validation assistant for an LLM context compaction system.

You will receive:
  1. ORIGINAL MESSAGES: a sequence of conversation messages that were compressed.
  2. SUMMARY: the compact summary that was generated to replace those messages.

Your task is to identify whether any of the following categories of important
information from the original messages are missing or inadequately represented
in the summary:

Categories to check:
  - REQUIREMENTS: explicit user requirements, goals, constraints, or preferences
  - DECISIONS: architectural, technical, or design decisions that were made
  - IMPORTANT FACTS: key facts, definitions, configurations, domain knowledge, or state
  - UNRESOLVED TASKS: open action items, pending questions, or next steps

Rules:
  - Be precise: only report items that are genuinely missing or significantly distorted
  - Do NOT report minor paraphrasing differences or stylistic changes
  - Do NOT penalise the summary for omitting pleasantries or meta-conversation
  - If all important information is preserved, "missing" must be an empty list

You MUST respond with ONLY a valid JSON object in exactly this format, no other text:
{
  "passed": true,
  "missing": []
}

or if items are missing:
{
  "passed": false,
  "missing": [
    "Requirement: the system must support Python 3.11+",
    "Decision: use GitHub App instead of personal access token"
  ]
}
"""


class OpenAIValidator(Validator):
    """Validates summaries using an OpenAI-compatible LLM as the judge.

    The LLM receives the original messages and summary, then returns a structured
    JSON verdict identifying whether important information was preserved and what
    (if anything) is missing.

    Uses the same environment variables as OpenAISummarizer for zero-config
    integration when the API is already configured.

    Args:
        api_key:     API key. Defaults to OPENAI_API_KEY environment variable.
        model:       Model name. Defaults to OPENAI_MODEL env var or "gpt-4o-mini".
        base_url:    API base URL. Defaults to OPENAI_BASE_URL env var.
        temperature: Sampling temperature (default: 0.1 for deterministic verdicts).
        client:      Optional pre-configured OpenAI client (for testing/proxies).
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       Optional[str] = None,
        base_url:    Optional[str] = None,
        temperature: float = 0.1,
        client:      Optional[Any] = None,
    ) -> None:
        env_key      = os.getenv("OPENAI_API_KEY",  "").strip()
        env_model    = os.getenv("OPENAI_MODEL",    "").strip()
        env_base_url = os.getenv("OPENAI_BASE_URL", "").strip()

        self.api_key:     str   = (api_key  if api_key  is not None else env_key).strip()
        self.model:       str   = (model    if model    is not None else (env_model    or _DEFAULT_MODEL   )).strip()
        self.base_url:    str   = (base_url if base_url is not None else (env_base_url or _DEFAULT_BASE_URL)).strip()
        self.temperature: float = temperature

        if client is not None:
            self._client = client
        else:
            if not self.api_key:
                raise ValidatorError(
                    "No API key found. "
                    "Set the OPENAI_API_KEY environment variable or pass api_key=... "
                    "to OpenAIValidator()."
                )
            try:
                import openai as _openai
                self._client = _openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ValidatorError(
                    "The 'openai' package is not installed. Run: pip install openai"
                )

    def validate(
        self,
        original_messages: "List[Message]",
        summary_text: str,
    ) -> ValidationResult:
        """Use the LLM to judge whether the summary preserves important content.

        Args:
            original_messages: The Message objects that were compressed.
            summary_text:      The plain-text summary to evaluate.

        Returns:
            ValidationResult with passed, missing_items, and details.

        Raises:
            ValidatorError: On API / network failures or malformed responses.
        """
        name = type(self).__name__

        if not original_messages:
            return ValidationResult(
                passed=True,
                details="No messages to validate.",
                validator_used=name,
            )

        if not summary_text or not summary_text.strip():
            return ValidationResult(
                passed=False,
                missing_items=["<entire summary is empty>"],
                details="The summary is blank.",
                validator_used=name,
            )

        # Format original messages as a readable transcript.
        transcript_lines = [
            f"[{msg.role.upper()}]: {msg.content}"
            for msg in original_messages
        ]
        transcript = "\n\n".join(transcript_lines)

        user_prompt = (
            f"ORIGINAL MESSAGES ({len(original_messages)} total):\n\n"
            f"{transcript}\n\n"
            f"---\n\n"
            f"SUMMARY:\n\n{summary_text}"
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
            raise ValidatorError(
                f"OpenAI validation API call failed: {exc}"
            ) from exc

        # Extract raw text response.
        if not response or not getattr(response, "choices", None):
            raise ValidatorError(
                "OpenAI validator received an invalid response (no choices)."
            )

        try:
            raw_text = response.choices[0].message.content
        except (IndexError, AttributeError) as exc:
            raise ValidatorError(
                f"Unexpected validator API response format: {exc}"
            ) from exc

        if raw_text is None:
            raise ValidatorError(
                "The validator model returned null content."
            )

        raw_text = raw_text.strip()

        # Parse and validate the JSON verdict.
        verdict = self._parse_verdict(raw_text, name)
        return verdict

    def _parse_verdict(self, raw_text: str, validator_name: str) -> ValidationResult:
        """Parse the LLM JSON verdict into a ValidationResult.

        Raises ValidatorError on malformed JSON or unexpected structure.
        """
        # Strip optional markdown fences the model may include.
        text = raw_text
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first line (``` or ```json) and trailing ```
            text = "\n".join(
                line for line in lines[1:]
                if line.strip() != "```"
            ).strip()

        try:
            verdict = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidatorError(
                f"Validator model returned non-JSON response: {exc!r}\n"
                f"Raw response was: {raw_text[:300]!r}"
            ) from exc

        if not isinstance(verdict, dict):
            raise ValidatorError(
                f"Validator response is not a JSON object. Got: {type(verdict).__name__!r}."
            )

        passed_raw = verdict.get("passed")
        if not isinstance(passed_raw, bool):
            raise ValidatorError(
                f"'passed' field must be a boolean; got {type(passed_raw).__name__!r}."
            )

        missing_raw = verdict.get("missing", [])
        if not isinstance(missing_raw, list):
            raise ValidatorError(
                f"'missing' field must be a list; got {type(missing_raw).__name__!r}."
            )

        missing_items = [str(item) for item in missing_raw]

        warnings: list[str] = []
        if passed_raw and missing_items:
            warnings.append(
                "Validator returned passed=true but listed missing items. "
                "Treating as passed with warnings."
            )

        details = (
            f"LLM judge ({self.model}) checked {len(missing_items)} potential "
            f"missing item(s) and returned passed={passed_raw}."
        )

        return ValidationResult(
            passed=passed_raw,
            warnings=warnings,
            missing_items=missing_items,
            details=details,
            validator_used=validator_name,
        )
