# tests/test_llm_pytest.py
"""Step 18 (pytest): provider-independent LLM context adapter.

All API calls are mocked -- no network, no keys, no real models. Covers the
LLMClient interface, OpenAI configuration, payload assembly from ContextFlow
context, response parsing (tokens, latency, model), and clean API/timeout
error handling.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from contextflow import (
    AssembledContext,
    BaseLLMClient,
    Conversation,
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    OpenAILLMClient,
)


def _mock_client(text="Hello!", prompt=10, completion=4, model="gpt-4o-mini"):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    client.chat.completions.create.return_value = response
    return client


def _conversation():
    conv = Conversation(name="Adapter", max_tokens=8000)
    conv.add(Message(role="system", content="You are helpful."))
    conv.add(Message(role="user", content="Hi there."))
    return conv


# ── interface ────────────────────────────────────────────────────────


def test_client_is_abstract():
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore
    assert BaseLLMClient is LLMClient


def test_custom_provider_keeps_core_independent():
    class EchoClient(LLMClient):
        def complete(self, context=None, system=None, request=None,
                     temperature=None, max_tokens=None):
            return LLMResponse(content="echo", model="custom")

    response = EchoClient().complete(request="hi")
    assert response.content == "echo"
    assert response.model == "custom"
    assert response.total_tokens == 0


def test_response_helpers_and_str():
    response = LLMResponse(content="Hi!", model="m", input_tokens=10,
                           output_tokens=4, latency_seconds=0.123)
    assert response.total_tokens == 14
    text = str(response)
    assert "m" in text and "10 in + 4 out" in text and "0.123s" in text


# ── configuration ────────────────────────────────────────────────────


def test_openai_reads_env():
    env = {
        "OPENAI_API_KEY": "env-key",
        "OPENAI_MODEL": "env-model",
        "OPENAI_BASE_URL": "https://env.host/v1",
        "OPENAI_TIMEOUT": "12.5",
    }
    with patch.dict(os.environ, env, clear=True):
        client = OpenAILLMClient(client=MagicMock())
        assert client.api_key == "env-key"
        assert client.model == "env-model"
        assert client.base_url == "https://env.host/v1"
        assert client.timeout == 12.5


def test_openai_defaults():
    with patch.dict(os.environ, {}, clear=True):
        client = OpenAILLMClient(api_key="k", client=MagicMock())
        assert client.model == "gpt-4o-mini"
        assert client.base_url == "https://api.openai.com/v1"
        assert client.timeout == 60.0
        assert client.temperature == 0.7


def test_explicit_args_override_env():
    env = {"OPENAI_API_KEY": "env", "OPENAI_MODEL": "env-m",
           "OPENAI_BASE_URL": "https://env/v1", "OPENAI_TIMEOUT": "5"}
    with patch.dict(os.environ, env, clear=True):
        client = OpenAILLMClient(api_key="arg", model="arg-m",
                                 base_url="https://arg/v1", timeout=9,
                                 temperature=0.1, client=MagicMock())
        assert (client.api_key, client.model, client.base_url,
                client.timeout, client.temperature) == (
            "arg", "arg-m", "https://arg/v1", 9.0, 0.1)


def test_missing_api_key_rejected():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(LLMError, match="No API key"):
            OpenAILLMClient(api_key="")


def test_invalid_timeout_rejected():
    with pytest.raises(ValueError, match="positive number"):
        OpenAILLMClient(api_key="k", timeout=0, client=MagicMock())
    with pytest.raises(ValueError, match="positive number"):
        OpenAILLMClient(api_key="k", timeout=-3, client=MagicMock())
    with pytest.raises(TypeError, match="positive number"):
        OpenAILLMClient(api_key="k", timeout=True, client=MagicMock())  # type: ignore
    with patch.dict(os.environ, {"OPENAI_TIMEOUT": "soon"}, clear=True):
        with pytest.raises(ValueError, match="positive number"):
            OpenAILLMClient(api_key="k", client=MagicMock())


def test_invalid_temperature_rejected():
    with pytest.raises(ValueError, match="between 0 and 2"):
        OpenAILLMClient(api_key="k", temperature=2.5, client=MagicMock())
    with pytest.raises(TypeError, match="between 0 and 2"):
        OpenAILLMClient(api_key="k", temperature="hot", client=MagicMock())  # type: ignore


def test_config_exposes_settings_without_key():
    client = OpenAILLMClient(api_key="secret", client=MagicMock())
    assert client.config == {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "timeout": 60.0,
        "temperature": 0.7,
    }


# ── payload assembly ─────────────────────────────────────────────────


def test_payload_order_system_context_request():
    client = OpenAILLMClient(api_key="k", client=_mock_client())
    conv = _conversation()
    client.complete(context=conv, system="Be helpful.", request="What now?")
    sent = client._client.chat.completions.create.call_args[1]["messages"]
    assert sent[0] == {"role": "system", "content": "Be helpful."}
    assert sent[1] == {"role": "system", "content": "You are helpful."}
    assert sent[2] == {"role": "user", "content": "Hi there."}
    assert sent[3] == {"role": "user", "content": "What now?"}


def test_assembled_context_source():
    ctx = AssembledContext(
        messages=[Message(role="assistant", content="Fact.")],
        total_tokens=3, max_tokens=100)
    client = OpenAILLMClient(api_key="k", client=_mock_client())
    response = client.complete(context=ctx, request="Q?")
    assert response.content == "Hello!"
    sent = client._client.chat.completions.create.call_args[1]["messages"]
    assert sent == [
        {"role": "assistant", "content": "Fact."},
        {"role": "user", "content": "Q?"},
    ]


def test_message_list_and_message_request():
    client = OpenAILLMClient(api_key="k", client=_mock_client())
    response = client.complete(
        context=[Message(role="user", content="Ping.")],
        request=Message(role="user", content="Pong?"))
    assert response.content == "Hello!"


def test_temperature_and_max_tokens_passthrough():
    client = OpenAILLMClient(api_key="k", client=_mock_client())
    client.complete(request="Prompt?", temperature=0.1, max_tokens=50)
    kwargs = client._client.chat.completions.create.call_args[1]
    assert kwargs["temperature"] == 0.1
    assert kwargs["max_tokens"] == 50
    assert kwargs["model"] == "gpt-4o-mini"


def test_invalid_per_call_options_rejected():
    client = OpenAILLMClient(api_key="k", client=_mock_client())
    with pytest.raises(ValueError, match="between 0 and 2"):
        client.complete(request="hi", temperature=5.0)
    with pytest.raises(ValueError, match="positive integer"):
        client.complete(request="hi", max_tokens=0)


def test_empty_payload_rejected_without_api_call():
    client = OpenAILLMClient(api_key="k", client=_mock_client())
    with pytest.raises(LLMError, match="Nothing to send"):
        client.complete()
    with pytest.raises(LLMError, match="Nothing to send"):
        client.complete(context=[], system="   ")
    client._client.chat.completions.create.assert_not_called()


def test_invalid_payload_types_rejected():
    client = OpenAILLMClient(api_key="k", client=_mock_client())
    with pytest.raises(TypeError, match="AssembledContext"):
        client.complete(context="nope")  # type: ignore
    with pytest.raises(TypeError, match="string or Message"):
        client.complete(request=123)  # type: ignore
    with pytest.raises(TypeError, match="string or None"):
        client.complete(system=123)  # type: ignore
    with pytest.raises(LLMError, match="only Message"):
        client.complete(context=["nope"])  # type: ignore


# ── response parsing ─────────────────────────────────────────────────


def test_response_fields_and_latency():
    client = OpenAILLMClient(api_key="k", model="chat-model",
                             client=_mock_client("Done.", 7, 3, "served-model"))
    response = client.complete(request="Go.")
    assert response.content == "Done."
    assert response.input_tokens == 7
    assert response.output_tokens == 3
    assert response.total_tokens == 10
    assert response.model == "served-model"  # server-reported name wins
    assert response.latency_seconds >= 0.0
    assert response.finish_reason == "stop"


def test_missing_usage_defaults_to_zero():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "Hi."
    choice.finish_reason = None
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    response.model = None
    client.chat.completions.create.return_value = response
    result = OpenAILLMClient(api_key="k", client=client).complete(request="Go.")
    assert (result.input_tokens, result.output_tokens) == (0, 0)
    assert result.model == "gpt-4o-mini"  # falls back to configured model
    assert result.finish_reason == "stop"


# ── error handling ───────────────────────────────────────────────────


def test_api_exception_wrapped():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("timeout after 60s")
    with pytest.raises(LLMError, match="chat call failed.*timeout after 60s"):
        OpenAILLMClient(api_key="k", client=client).complete(request="Go.")


def test_empty_choices_rejected():
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(choices=[])
    with pytest.raises(LLMError, match="no choices"):
        OpenAILLMClient(api_key="k", client=client).complete(request="Go.")


def test_null_content_rejected():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = None
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    with pytest.raises(LLMError, match="null content"):
        OpenAILLMClient(api_key="k", client=client).complete(request="Go.")


def test_original_error_chained():
    client = MagicMock()
    cause = TimeoutError("timed out")
    client.chat.completions.create.side_effect = cause
    with pytest.raises(LLMError) as exc_info:
        OpenAILLMClient(api_key="k", client=client).complete(request="Go.")
    assert exc_info.value.__cause__ is cause
