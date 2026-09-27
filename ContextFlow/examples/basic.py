# examples/basic.py
"""Basic usage: track a chat, check pressure, print assembled context.

Run from the repository root (after `pip install -e .`) or anywhere once
`contextflow` is installed:  python examples/basic.py
"""

from contextflow import ContextManager

cf = ContextManager(max_tokens=8000, system="You are a concise assistant.")

cf.add_message("user", "What is blockchain?")
cf.add_message("assistant", "Blockchain is a distributed ledger shared across nodes.")
cf.add_message("user", "And what is a smart contract?", protected=False)

print("history:", cf.message_count, "messages,", cf.tokens, "tokens")
print("status:", cf.conversation.get_status())
print()
print(cf.get_context("smart contract"))
