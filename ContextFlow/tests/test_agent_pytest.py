# tests/test_agent_pytest.py
"""Step 19 (pytest): simple ContextFlow agent workflow.

The model is a deterministic fake -- no network, no keys, no real calls.
Covers the full turn (retrieve -> assemble -> complete -> record -> maybe
compact), history handling, token statistics, compaction reporting, and
constructor/request validation.
"""

import pytest

from contextflow import (
    Agent,
    AgentResult,
    Compactor,
    ContextAssembler,
    ContextPipeline,
    Conversation,
    KeywordRetriever,
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
)


class FakeClient(LLMClient):
    """Canned replies; records every complete() call for assertions."""

    def __init__(self, reply="Noted.", model="fake"):
        self.reply = reply
        self.model = model
        self.calls = []

    def complete(self, context=None, system=None, request=None,
                 temperature=None, max_tokens=None):
        from contextflow import AssembledContext
        if isinstance(context, AssembledContext):
            sent = list(context.messages)
        elif context is None:
            sent = []
        else:
            sent = list(context)
        if request is not None:
            sent = sent + [request if isinstance(request, Message)
                           else Message(role="user", content=request)]
        self.calls.append({"messages": sent, "system": system})
        return LLMResponse(content=self.reply, model=self.model,
                           input_tokens=20, output_tokens=3,
                           latency_seconds=0.01)


class FailingClient(LLMClient):
    def complete(self, context=None, system=None, request=None,
                 temperature=None, max_tokens=None):
        raise LLMError("Simulated outage.")


def _agent(**kwargs):
    kwargs.setdefault("llm", FakeClient())
    return Agent(**kwargs)


def _seeded_conversation(max_tokens=8000):
    conv = Conversation(name="Agent", max_tokens=max_tokens)
    conv.add(Message(role="system", content="You are helpful."))
    conv.add(Message(
        role="user",
        content="Never store secrets in plain text.",
        protected=True,
    ))
    conv.add(Message(role="assistant", content="PostgreSQL storage decision."))
    return conv


# ── complete workflow ────────────────────────────────────────────────


def test_run_returns_reply_and_statistics():
    agent = _agent(conversation=_seeded_conversation(), system="Be concise.")
    result = agent.run("where do we persist records?")

    assert isinstance(result, AgentResult)
    assert result.response == "Noted."
    assert result.llm_response.model == "fake"
    assert result.assistant_message.role == "assistant"
    assert result.assistant_message.content == "Noted."
    assert result.prompt_tokens == result.assembled.total_tokens
    assert result.completion_tokens == 3
    assert result.total_tokens == result.prompt_tokens + 3
    assert result.compaction_occurred is False
    assert result.pipeline_result is not None
    assert result.pipeline_result.triggered is False
    assert result.turns == 5
    assert "Agent Result" in str(result)


def test_history_accumulates_across_runs():
    agent = _agent(conversation=_seeded_conversation())
    agent.run("First question?")
    agent.run("Second question?")
    history = agent.history
    assert [m.content for m in history[-4:]] == [
        "First question?", "Noted.", "Second question?", "Noted."]
    assert all(m.role in ("user", "assistant", "system") for m in history)
    # Returned history is a safe copy.
    history.clear()
    assert agent.conversation.message_count() == 7


def test_request_message_object_accepted():
    agent = _agent(conversation=_seeded_conversation())
    result = agent.run(Message(role="user", content="Object request?"))
    assert result.response == "Noted."
    assert agent.history[-2].content == "Object request?"


def test_system_instructions_reach_the_model():
    fake = FakeClient()
    agent = _agent(llm=fake, conversation=_seeded_conversation(), system="Be concise.")
    agent.run("Hi?")
    assert fake.calls[0]["system"] == "Be concise."
    # Assembled prompt ends with the request (no duplication by the client).
    sent = fake.calls[0]["messages"]
    assert sent[-1].content == "Hi?"


def test_retrieved_context_flows_into_assembly():
    agent = _agent(conversation=_seeded_conversation())
    result = agent.run("PostgreSQL storage")
    assert result.retrieved  # keyword side matches the decision message
    assert result.assembled is not None


def test_custom_components_are_used():
    fake = FakeClient(reply="Custom.")
    assembler = ContextAssembler(max_tokens=500, keep_recent=1)
    pipeline = ContextPipeline()
    agent = Agent(llm=fake, conversation=_seeded_conversation(),
                  retriever=KeywordRetriever(top_k=1),
                  assembler=assembler, pipeline=pipeline,
                  max_tokens=500)
    result = agent.run("Hi?")
    assert result.response == "Custom."
    assert result.assembled.max_tokens == 500


def test_explicit_max_tokens_overrides_conversation():
    conv = _seeded_conversation(max_tokens=8000)
    agent = _agent(conversation=conv, max_tokens=100)
    result = agent.run("Hi?")
    assert result.assembled.max_tokens == 100


# ── compaction triggering ────────────────────────────────────────────


def test_compaction_triggered_under_pressure():
    conv = Conversation(name="Tight", max_tokens=60)
    conv.add(Message(role="system", content="You are helpful."))
    conv.add(Message(role="user", content="Never store secrets.", protected=True))
    for i in range(6):
        conv.add(Message(role="user", content="Filler discussion turn %d here." % i))
    agent = _agent(conversation=conv)
    result = agent.run("One more turn please?")

    assert result.compaction_occurred is True
    assert result.pipeline_result.triggered is True
    assert result.pipeline_result.committed is True
    assert "Never store secrets." in [m.content for m in agent.history]


def test_no_compaction_when_healthy():
    agent = _agent(conversation=_seeded_conversation())
    result = agent.run("Hello?")
    assert result.compaction_occurred is False
    assert result.turns == 5  # 3 seeded + user + assistant


# ── failure handling ─────────────────────────────────────────────────


def test_model_failure_leaves_history_untouched():
    agent = Agent(llm=FailingClient(), conversation=_seeded_conversation())
    before = [m.content for m in agent.history]
    with pytest.raises(LLMError, match="Simulated outage"):
        agent.run("Hello?")
    assert [m.content for m in agent.history] == before


# ── validation ───────────────────────────────────────────────────────


def test_invalid_requests_rejected():
    agent = _agent(conversation=_seeded_conversation())
    with pytest.raises(TypeError, match="string or Message"):
        agent.run(123)  # type: ignore
    with pytest.raises(ValueError, match="must not be blank"):
        agent.run("   ")
    with pytest.raises(ValueError, match="role 'user'"):
        agent.run(Message(role="assistant", content="Sneaky."))


def test_missing_budget_rejected_at_run_time():
    conv = Conversation(name="NoBudget")  # no max_tokens
    agent = _agent(conversation=conv)
    with pytest.raises(ValueError, match="No token budget"):
        agent.run("Hello?")


def test_invalid_agent_arguments_rejected():
    conv = _seeded_conversation()
    with pytest.raises(TypeError, match="LLMClient"):
        Agent(llm="nope", conversation=conv)  # type: ignore
    with pytest.raises(TypeError, match="Conversation"):
        Agent(llm=FakeClient(), conversation="nope")  # type: ignore
    with pytest.raises(TypeError, match="Retriever"):
        Agent(llm=FakeClient(), retriever="nope")  # type: ignore
    with pytest.raises(TypeError, match="ContextAssembler"):
        Agent(llm=FakeClient(), assembler="nope")  # type: ignore
    with pytest.raises(TypeError, match="ContextPipeline"):
        Agent(llm=FakeClient(), pipeline=Compactor())  # type: ignore
    with pytest.raises(TypeError, match="string or None"):
        Agent(llm=FakeClient(), system=123)  # type: ignore
    with pytest.raises(ValueError, match="max_tokens"):
        Agent(llm=FakeClient(), max_tokens=0)
    with pytest.raises(ValueError, match="keep_recent"):
        Agent(llm=FakeClient(), keep_recent=-1)
    with pytest.raises(ValueError, match="retrieval_top_k"):
        Agent(llm=FakeClient(), retrieval_top_k=0)


def test_default_conversation_created():
    agent = _agent(max_tokens=500)
    agent.run("Hello?")
    assert agent.conversation.name == "Agent Session"
    assert agent.conversation.message_count() == 2
