# contextflow/llm.py
"""
Step 18: LLM context adapter (provider-independent).

LLMClient is the seam between ContextFlow's assembled context and any chat
model: it accepts an AssembledContext (or Conversation / message list),
optional system instructions, and the current user request, then returns an
LLMResponse with the model text plus input tokens, output tokens, latency,
and model name.

ContextFlow itself never imports a provider SDK -- only the OpenAILLMClient
implementation talks to an OpenAI-compatible chat completions API. Custom
providers subclass LLMClient and implement complete().

Configuration (via environment variables or constructor args):
    OPENAI_API_KEY  -- Required. Your API key (shared with the other
                       OpenAI-backed components).
    OPENAI_MODEL    -- Optional. Defaults to "gpt-4o-mini".
    OPENAI_BASE_URL -- Optional. Defaults to "https://api.openai.com/v1".
    OPENAI_TIMEOUT  -- Optional. Request timeout in seconds (default 60.0).

API errors and timeouts surface as LLMError with the underlying cause
chained -- never as raw SDK exceptions.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from .conversation import Conversation
from .hybrid import AssembledContext
from .message import Message

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_TIMEOUT = 60.0

# Sources accepted as the "assembled context" argument.
LLMContext = Union["AssembledContext", "Conversation", Sequence["Message"]]


class LLMError(Exception):
    """Raised when a model call cannot be completed.

    Covers missing credentials, API / network failures, timeouts, and
    malformed responses. An empty shortlist of context is NOT an error --
    sending zero messages is (nothing to complete).
    """


@dataclass
class LLMResponse:
    """Result of one model call.

    Attributes:
        content:         Model reply text (may be "").
        model:           Model name that produced the reply.
        input_tokens:    Prompt tokens reported by the API (0 if unreported).
        output_tokens:   Completion tokens reported by the API (0 if unreported).
        latency_seconds: Wall-clock time of the API call, rounded to 3 dp.
        finish_reason:   Backend stop reason (default "stop").
    """
    content: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        """Input + output tokens."""
        return self.input_tokens + self.output_tokens

    def __str__(self) -> str:
        sep = "-" * 52
        preview = self.content[:120] + ("..." if len(self.content) > 120 else "")
        return "\n".join([
            sep,
            "LLM Response",
            sep,
            f"  Model    : {self.model}",
            f"  Tokens   : {self.input_tokens} in + {self.output_tokens} out "
            f"(= {self.total_tokens})",
            f"  Latency  : {self.latency_seconds:.3f}s ({self.finish_reason})",
            f"  Content  : {preview}",
            sep,
        ])


class LLMClient(ABC):
    """Abstract interface that every LLM adapter must implement.

    The interface speaks ContextFlow types (AssembledContext, Message) so
    callers stay provider-independent; only implementations know about SDKs.
    """

    @abstractmethod
    def complete(
        self,
        context: Optional[LLMContext] = None,
        system: Optional[str] = None,
        request: Optional[Union[str, Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send assembled context + instructions + request to the model.

        Args:
            context:     AssembledContext, Conversation, or message list
                         providing the background context.
            system:      System instructions, sent first as a system message.
            request:     Current user request (string or Message), sent last.
            temperature: Per-call sampling temperature override.
            max_tokens:  Per-call cap on completion tokens.

        Returns:
            LLMResponse with content, token usage, latency, and model name.

        Raises:
            LLMError: On empty payloads, bad arguments reaching the backend,
                      API failures, timeouts, or malformed responses.
        """
        pass


# BaseLLMClient is an alias kept for naming consistency with the other
# subsystems (BaseSummarizer, BaseValidator, BaseRetriever, ...).
BaseLLMClient = LLMClient


def _coerce_context_messages(context: Optional[LLMContext]) -> List[Message]:
    """Extract Message objects from any accepted context source."""
    if context is None:
        return []
    if isinstance(context, AssembledContext):
        return list(context.messages)
    if isinstance(context, Conversation):
        return context.get_messages()
    if isinstance(context, (list, tuple)):
        items = list(context)
        for item in items:
            if not isinstance(item, Message):
                raise LLMError(
                    "context lists must contain only Message instances, "
                    f"got {type(item).__name__!r}."
                )
        return items
    raise TypeError(
        "context must be an AssembledContext, Conversation, or list of "
        f"Messages, got {type(context).__name__!r}."
    )


def _coerce_request(request: Optional[Union[str, Message]]) -> Optional[Message]:
    """Normalize the user request to a Message (None stays None)."""
    if request is None:
        return None
    if isinstance(request, Message):
        return request
    if isinstance(request, str):
        return Message(role="user", content=request)
    raise TypeError(
        f"request must be a string or Message, got {type(request).__name__!r}."
    )


class OpenAILLMClient(LLMClient):
    """Sends ContextFlow context to any OpenAI-compatible chat API.

    Args:
        api_key:     API key. Defaults to OPENAI_API_KEY environment variable.
        model:       Model name. Defaults to OPENAI_MODEL env var or "gpt-4o-mini".
        base_url:    API base URL. Defaults to OPENAI_BASE_URL env var or
                     "https://api.openai.com/v1".
        timeout:     Request timeout in seconds. Defaults to OPENAI_TIMEOUT
                     env var or 60.0.
        temperature: Default sampling temperature (default 0.7). Overridable
                     per complete() call.
        client:      Optional pre-configured OpenAI client (for testing/proxies).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: float = 0.7,
        client: Optional[Any] = None,
    ) -> None:
        env_key = os.getenv("OPENAI_API_KEY", "").strip()
        env_model = os.getenv("OPENAI_MODEL", "").strip()
        env_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        env_timeout = os.getenv("OPENAI_TIMEOUT", "").strip()

        self.api_key: str = (api_key if api_key is not None else env_key).strip()
        self.model: str = (model if model is not None else (env_model or _DEFAULT_MODEL)).strip()
        self.base_url: str = (base_url if base_url is not None else (env_base_url or _DEFAULT_BASE_URL)).strip()
        self.timeout: float = self._resolve_timeout(
            timeout if timeout is not None else (env_timeout or None),
            "timeout" if timeout is not None else "OPENAI_TIMEOUT",
        )
        self.temperature: float = self._validate_temperature(temperature, "temperature")

        if client is not None:
            self._client = client
        else:
            if not self.api_key:
                raise LLMError(
                    "No API key found. "
                    "Set the OPENAI_API_KEY environment variable or pass api_key=... "
                    "to OpenAILLMClient()."
                )
            try:
                import openai as _openai
                self._client = _openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except ImportError:
                raise LLMError(
                    "The 'openai' package is not installed. "
                    "Run: pip install openai"
                )

    # ── validation helpers ───────────────────────────────────────

    @staticmethod
    def _resolve_timeout(value: Any, label: str) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{label} must be a positive number of seconds.")
        try:
            seconds = float(value) if value is not None else _DEFAULT_TIMEOUT
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} must be a positive number of seconds, got {value!r}."
            ) from exc
        if seconds <= 0.0:
            raise ValueError(
                f"{label} must be a positive number of seconds, got {value!r}."
            )
        return seconds

    @staticmethod
    def _validate_temperature(value: Any, label: str = "temperature") -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be a number between 0 and 2.")
        if not (0.0 <= float(value) <= 2.0):
            raise ValueError(f"{label} must be between 0 and 2, got {value!r}.")
        return float(value)

    @staticmethod
    def _validate_max_tokens(value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"max_tokens must be a positive integer, got {type(value).__name__!r}."
            )
        if value <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {value}.")
        return value

    # ── public API ───────────────────────────────────────────────

    @property
    def config(self) -> Dict[str, Any]:
        """Adapter configuration (safe to log: no API key included)."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "temperature": self.temperature,
        }

    def complete(
        self,
        context: Optional[LLMContext] = None,
        system: Optional[str] = None,
        request: Optional[Union[str, Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Assemble the chat payload and call the model.

        Payload order: system instructions first, then context messages in
        order, then the user request last.
        """
        if system is not None and not isinstance(system, str):
            raise TypeError(
                f"system must be a string or None, got {type(system).__name__!r}."
            )
        messages = _coerce_context_messages(context)
        request_msg = _coerce_request(request)

        payload: List[Dict[str, str]] = []
        if system is not None and system.strip():
            payload.append({"role": "system", "content": system})
        payload.extend({"role": m.role, "content": m.content} for m in messages)
        if request_msg is not None:
            payload.append({"role": request_msg.role, "content": request_msg.content})
        if not payload:
            raise LLMError(
                "Nothing to send: provide context, system instructions, "
                "or a user request."
            )

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": (
                self.temperature if temperature is None
                else self._validate_temperature(temperature)
            ),
        }
        if max_tokens is not None:
            call_kwargs["max_tokens"] = self._validate_max_tokens(max_tokens)

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            raise LLMError(f"OpenAI chat call failed: {exc}") from exc
        latency = round(time.perf_counter() - started, 3)

        if not response or not getattr(response, "choices", None):
            raise LLMError("OpenAI chat call returned no choices.")
        try:
            choice = response.choices[0]
        except (IndexError, TypeError) as exc:
            raise LLMError(f"OpenAI chat call returned a malformed response ({exc}).") from exc

        reply = getattr(getattr(choice, "message", None), "content", None)
        if reply is None:
            raise LLMError("The model returned null content.")
        if not isinstance(reply, str):
            raise LLMError(
                f"The model returned non-text content ({type(reply).__name__!r})."
            )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
        try:
            input_tokens = int(input_tokens or 0)
            output_tokens = int(output_tokens or 0)
        except (TypeError, ValueError):
            input_tokens, output_tokens = 0, 0

        model_name = getattr(response, "model", None) or self.model
        finish = getattr(choice, "finish_reason", None) or "stop"

        return LLMResponse(
            content=reply,
            model=str(model_name),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            finish_reason=str(finish),
        )
