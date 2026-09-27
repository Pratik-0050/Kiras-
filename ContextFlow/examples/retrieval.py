# examples/retrieval.py
"""Retrieval: keyword search over persisted-style history.

Run:  python examples/retrieval.py
Offline: KeywordRetriever needs no API key. Semantic/vector retrieval
needs an embedding provider (OpenAIEmbeddingProvider + OPENAI_API_KEY),
e.g.:  SemanticRetriever(OpenAIEmbeddingProvider()).retrieve(query, conv)
"""

from contextflow import ContextManager, KeywordRetriever

cf = ContextManager(max_tokens=8000)
cf.add_message("system", "You are an expert software architect.")
cf.add_message("assistant", "Decision: use PostgreSQL 16 with pgvector for storage.")
cf.add_message("user", "How should we handle GitHub authentication?")
cf.add_message("assistant", "Use pyproject.toml with flit for packaging.")

retriever = KeywordRetriever(top_k=2)
for query in ("PostgreSQL storage decision", "packaging", "quantum banana"):
    hits = retriever.retrieve(query, cf.conversation)
    print("query %r -> %d hit(s)" % (query, len(hits)))
    for hit in hits:
        print("   ", hit)

print()
print("same ranking via the Conversation helper:")
for hit in cf.conversation.search("GitHub authentication", top_k=1):
    print("   ", hit)
