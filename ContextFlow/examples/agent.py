# examples/agent.py
"""Agent: retrieve -> assemble -> answer -> record -> compact on pressure.

Run:  python examples/agent.py
Offline: uses a scripted stand-in model (no API key). Swap DemoModel for
OpenAILLMClient() with OPENAI_API_KEY set to call a real model. No tools
or planning -- one deterministic turn per run() call.
"""

from contextflow import Agent, Conversation, LLMClient, LLMResponse, Message


class DemoModel(LLMClient):
    """Deterministic stand-in: acknowledges the latest user message."""

    def complete(self, context=None, system=None, request=None,
                 temperature=None, max_tokens=None):
        from contextflow import AssembledContext
        turns = (list(context.messages) if isinstance(context, AssembledContext)
                 else list(context or []))
        text = next((m.content for m in reversed(turns) if m.role == "user"), "...")
        return LLMResponse(content="Acknowledged: %s" % text[:60],
                           model="demo-model", input_tokens=20, output_tokens=8,
                           latency_seconds=0.01)


conv = Conversation(name="Agent Demo", max_tokens=200)
conv.add(Message(role="system", content="You are a concise assistant."))
conv.add(Message(role="user", content="Never store secrets in plain text.",
                 protected=True))

agent = Agent(llm=DemoModel(), conversation=conv, system="Be concise.")

for question in ("What should we build first?", "And after that?"):
    result = agent.run(question)
    print(result)
    print("compacted:", result.compaction_occurred,
          "| history:", len(agent.history), "messages")
    print()
