# tests/test_semantic_pytest.py
"""Step 14 (pytest): semantic retrieval with mocked embeddings (no network).

Uses deterministic fake providers -- never real API calls. Covers the
EmbeddingProvider interface, OpenAI provider configuration and failure
handling, semantic ranking, top_k, caching, persisted embeddings, graceful
keyword fallback, and Conversation.search with a provider.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from contextflow import (
    BaseEmbeddingProvider,
    ContextStore,
    Conversation,
    EmbeddingError,
    EmbeddingProvider,
    KeywordRetriever,
    Message,
    OpenAIEmbeddingProvider,
    RetrievalResult,
    RetrieverError,
    SemanticRetriever,
    StoreError,
)


# ── fakes (no network, fully deterministic) ──────────────────────────


class FakeProvider(EmbeddingProvider):
    """Maps exact texts to vectors; records every embed() batch."""

    def __init__(self, mapping=None, default=(0.0, 1.0)):
        self.mapping = dict(mapping or {})
        self.default = list(default)
        self.batches = []

    def embed(self, texts):
        items = list(texts)
        self.batches.append(items)
        return [list(self.mapping.get(t, self.default)) for t in items]

    @property
    def call_count(self):
        return len(self.batches)

    @property
    def embedded_texts(self):
        return [t for batch in self.batches for t in batch]


class FailingProvider(EmbeddingProvider):
    """Always raises EmbeddingError (simulates an API outage)."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        raise EmbeddingError("Simulated API outage.")


def _semantic_conversation():
    conv = Conversation(name="Semantic", max_tokens=8000)
    conv.add(Message(
        role="assistant",
        content="PostgreSQL storage decision for persistence.",
    ))
    conv.add(Message(
        role="assistant",
        content="Packaging release with flit and setuptools.",
    ))
    conv.add(Message(
        role="user",
        content="Unrelated note about lunch plans.",
    ))
    return conv


def _paraphrase_setup():
    """Query shares NO keywords with the target message (proves semantics)."""
    query = "where do we persist records?"
    mapping = {
        query: [1.0, 0.0],
        "PostgreSQL storage decision for persistence.": [1.0, 0.0],
        "Packaging release with flit and setuptools.": [0.0, 1.0],
        "Unrelated note about lunch plans.": [-1.0, 0.0],
    }
    return query, mapping


# ── provider interface ───────────────────────────────────────────────


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        EmbeddingProvider()  # type: ignore
    assert BaseEmbeddingProvider is EmbeddingProvider


def test_embed_one_wraps_embed():
    provider = FakeProvider({"hi": [0.5, 0.5]})
    assert provider.embed_one("hi") == [0.5, 0.5]


def test_invalid_provider_rejected():
    with pytest.raises(TypeError, match="EmbeddingProvider"):
        SemanticRetriever(provider="nope")  # type: ignore


def test_invalid_fallback_rejected():
    with pytest.raises(TypeError, match="fallback must be"):
        SemanticRetriever(FakeProvider(), fallback="nope")


# ── semantic ranking (mocked vectors) ───────────────────────────────


def test_paraphrase_query_finds_semantic_match():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    hits = SemanticRetriever(FakeProvider(mapping)).retrieve(query, conv)
    assert len(hits) >= 1
    assert hits[0].message.content.startswith("PostgreSQL")
    assert hits[0].score == 1.0
    # Keyword retrieval finds nothing for the same paraphrase query.
    assert KeywordRetriever().retrieve(query, conv) == []


def test_scores_descend_and_ranks_assigned():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    hits = SemanticRetriever(FakeProvider(mapping), top_k=10).retrieve(query, conv)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    for hit in hits:
        assert isinstance(hit, RetrievalResult)
        assert 0.0 <= hit.score <= 1.0


def test_opposite_vector_excluded_by_default_min_score():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    # Lunch note is opposite ([-1,0] vs [1,0]) -> score 0.0 -> filtered.
    hits = SemanticRetriever(FakeProvider(mapping), top_k=10).retrieve(query, conv)
    assert all("lunch" not in h.message.content for h in hits)


def test_top_k_limits_semantic_results():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    hits = SemanticRetriever(FakeProvider(mapping), top_k=1).retrieve(query, conv)
    assert len(hits) == 1
    assert hits[0].message.content.startswith("PostgreSQL")


def test_per_call_top_k_override():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    retriever = SemanticRetriever(FakeProvider(mapping), top_k=5)
    assert len(retriever.retrieve(query, conv, top_k=1)) == 1


def test_empty_query_and_empty_conversation():
    conv = _semantic_conversation()
    provider = FakeProvider()
    assert SemanticRetriever(provider).retrieve("", conv) == []
    assert SemanticRetriever(provider).retrieve("   ", conv) == []
    assert SemanticRetriever(provider).retrieve("query", Conversation(name="E")) == []
    assert provider.call_count == 0  # no embedding wasted on empties


def test_min_score_filters_weak_hits():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    hits = SemanticRetriever(FakeProvider(mapping), min_score=0.99).retrieve(query, conv)
    assert len(hits) == 1  # only the identical-vector match survives
    assert hits[0].score == 1.0


def test_duplicate_content_returned_once():
    provider = FakeProvider({
        "Repeated summary text.": [1.0, 0.0],
        "query text": [1.0, 0.0],
    })
    conv = Conversation(name="Dupes")
    conv.add(Message(role="system", content="Repeated summary text."))
    conv.add(Message(role="assistant", content="Repeated summary text."))
    hits = SemanticRetriever(provider, top_k=10).retrieve("query text", conv)
    assert len(hits) == 1


# ── caching ──────────────────────────────────────────────────────────


def test_unchanged_messages_embedded_once():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    provider = FakeProvider(mapping)
    retriever = SemanticRetriever(provider)
    retriever.retrieve(query, conv)
    first_calls = provider.call_count
    assert first_calls > 0
    retriever.retrieve(query, conv)
    assert provider.call_count == first_calls  # everything cached
    info = retriever.cache_info()
    assert info["hits"] > 0
    assert info["size"] == 4  # query + 3 messages


def test_new_message_triggers_single_targeted_embed():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    provider = FakeProvider(mapping)
    retriever = SemanticRetriever(provider)
    retriever.retrieve(query, conv)
    before = provider.call_count
    conv.add(Message(role="user", content="Brand new follow-up turn."))
    retriever.retrieve(query, conv)
    assert provider.call_count == before + 1
    assert provider.batches[-1] == ["Brand new follow-up turn."]


def test_clear_cache_forces_reembed():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    provider = FakeProvider(mapping)
    retriever = SemanticRetriever(provider)
    retriever.retrieve(query, conv)
    retriever.clear_cache()
    assert retriever.cache_info() == {"hits": 0, "misses": 0, "size": 0}
    retriever.retrieve(query, conv)
    assert provider.call_count == 4  # query batch + messages batch, twice


# ── graceful fallback ────────────────────────────────────────────────


def test_api_failure_falls_back_to_keyword():
    conv = _semantic_conversation()
    retriever = SemanticRetriever(FailingProvider())
    hits = retriever.retrieve("PostgreSQL storage", conv)
    expected = KeywordRetriever().retrieve("PostgreSQL storage", conv)
    assert [h.message.content for h in hits] == [h.message.content for h in expected]
    assert len(hits) >= 1


def test_api_failure_without_fallback_raises():
    conv = _semantic_conversation()
    retriever = SemanticRetriever(FailingProvider(), fallback=False)
    with pytest.raises(RetrieverError, match="no fallback"):
        retriever.retrieve("PostgreSQL storage", conv)


def test_malformed_provider_vectors_fall_back():
    class RaggedProvider(EmbeddingProvider):
        def embed(self, texts):
            return [[1.0, 0.0]] * len(list(texts)) + [[1.0]]  # count mismatch

    conv = _semantic_conversation()
    hits = SemanticRetriever(RaggedProvider()).retrieve("PostgreSQL storage", conv)
    assert len(hits) >= 1  # keyword fallback saved the call


def test_keyword_retrieval_still_default_without_provider():
    conv = _semantic_conversation()
    hits = conv.search("PostgreSQL storage")
    assert len(hits) >= 1
    assert "PostgreSQL" in hits[0].message.content


def test_conversation_search_with_provider_is_semantic():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    hits = conv.search(query, provider=FakeProvider(mapping))
    assert len(hits) >= 1
    assert hits[0].message.content.startswith("PostgreSQL")
    # ...while the keyword path finds nothing for this paraphrase.
    assert conv.search(query) == []


# ── persisted embeddings ─────────────────────────────────────────────


def test_embed_missing_and_round_trip(tmp_path):
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    store = ContextStore(conv)
    assert store.embed_missing(FakeProvider(mapping)) == 3
    assert store.embed_missing(FakeProvider(mapping)) == 0  # cached now
    assert store.get_embedding("PostgreSQL storage decision for persistence.") == [1.0, 0.0]

    path = tmp_path / "emb.json"
    store.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert all("embedding" in m for m in payload["conversation"]["messages"])

    restored = ContextStore.load_file(path)
    assert restored.embeddings == store.embeddings
    assert restored.conversation.message_count() == 3


def test_primed_retriever_skips_message_embeds():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    store = ContextStore(conv)
    store.embed_missing(FakeProvider(mapping))

    provider = FakeProvider(mapping)
    retriever = SemanticRetriever(provider)
    assert retriever.prime_cache(store) == 3
    retriever.retrieve(query, conv)
    # Only the uncached query was embedded -- messages came from the store.
    assert provider.embedded_texts == [query]


def test_old_file_without_embeddings_loads_cleanly(tmp_path):
    conv = _semantic_conversation()
    path = tmp_path / "old.json"
    ContextStore(conv).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for message in payload["conversation"]["messages"]:
        message.pop("embedding", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = ContextStore.load_file(path)
    assert restored.embeddings == {}
    assert restored.conversation.message_count() == 3


def test_corrupted_embedding_rejected(tmp_path):
    conv = _semantic_conversation()
    path = tmp_path / "bad.json"
    ContextStore(conv).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["conversation"]["messages"][0]["embedding"] = "not-a-vector"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StoreError, match="[Cc]orrupt"):
        ContextStore.load_file(path)


def test_set_embedding_validates_vectors():
    store = ContextStore(Conversation(name="T"))
    with pytest.raises(ValueError, match="invalid embedding"):
        store.set_embedding("text", [])
    with pytest.raises(TypeError):
        store.set_embedding(123, [1.0])  # type: ignore


def test_embed_missing_rejects_bad_provider():
    store = ContextStore(_semantic_conversation())
    with pytest.raises(TypeError, match="implement embed"):
        store.embed_missing(object())


# ── OpenAI provider (mocked client -- no network) ────────────────────


def _mock_client(vectors):
    client = MagicMock()
    entries = []
    for vector in vectors:
        entry = MagicMock()
        entry.embedding = vector
        entries.append(entry)
    response = MagicMock()
    response.data = entries
    client.embeddings.create.return_value = response
    return client


def test_openai_provider_reads_env():
    env = {
        "OPENAI_API_KEY": "env-key",
        "OPENAI_EMBEDDING_MODEL": "env-embed-model",
        "OPENAI_BASE_URL": "https://custom.host/v1",
    }
    with patch.dict(os.environ, env, clear=True):
        provider = OpenAIEmbeddingProvider(client=MagicMock())
        assert provider.api_key == "env-key"
        assert provider.model == "env-embed-model"
        assert provider.base_url == "https://custom.host/v1"


def test_openai_provider_defaults():
    with patch.dict(os.environ, {}, clear=True):
        provider = OpenAIEmbeddingProvider(api_key="k", client=MagicMock())
        assert provider.model == "text-embedding-3-small"
        assert provider.base_url == "https://api.openai.com/v1"


def test_openai_provider_explicit_args_win():
    env = {"OPENAI_API_KEY": "env", "OPENAI_EMBEDDING_MODEL": "env-model"}
    with patch.dict(os.environ, env, clear=True):
        provider = OpenAIEmbeddingProvider(
            api_key="arg", model="arg-model", base_url="https://arg.host/v1",
            client=MagicMock(),
        )
        assert (provider.api_key, provider.model, provider.base_url) == (
            "arg", "arg-model", "https://arg.host/v1")


def test_openai_provider_missing_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EmbeddingError, match="No API key"):
            OpenAIEmbeddingProvider(api_key="")


def test_openai_provider_embed_success():
    client = _mock_client([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    provider = OpenAIEmbeddingProvider(api_key="k", client=client)
    vectors = provider.embed(["hello", "world"])
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    kwargs = client.embeddings.create.call_args[1]
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["input"] == ["hello", "world"]


def test_openai_provider_empty_input_skips_call():
    client = MagicMock()
    provider = OpenAIEmbeddingProvider(api_key="k", client=client)
    assert provider.embed([]) == []
    client.embeddings.create.assert_not_called()


def test_openai_provider_api_error():
    client = MagicMock()
    client.embeddings.create.side_effect = RuntimeError("timeout")
    provider = OpenAIEmbeddingProvider(api_key="k", client=client)
    with pytest.raises(EmbeddingError, match="embeddings call failed"):
        provider.embed(["hello"])


def test_openai_provider_count_mismatch():
    client = _mock_client([[0.1, 0.2]])  # 1 vector for 2 texts
    provider = OpenAIEmbeddingProvider(api_key="k", client=client)
    with pytest.raises(EmbeddingError, match="for 2 input"):
        provider.embed(["a", "b"])


def test_openai_provider_malformed_vector():
    client = _mock_client(["oops"])
    provider = OpenAIEmbeddingProvider(api_key="k", client=client)
    with pytest.raises(EmbeddingError, match="malformed vector"):
        provider.embed(["a"])


def test_openai_provider_rejects_empty_text():
    provider = OpenAIEmbeddingProvider(api_key="k", client=MagicMock())
    with pytest.raises(EmbeddingError, match="must not be empty"):
        provider.embed(["   "])


def test_openai_provider_powers_semantic_retrieval():
    query, mapping = _paraphrase_setup()
    conv = _semantic_conversation()
    client = MagicMock()

    def fake_create(model, input):
        return MagicMock(data=[
            MagicMock(embedding=mapping.get(t, [0.0, 1.0])) for t in input
        ])

    client.embeddings.create.side_effect = fake_create
    provider = OpenAIEmbeddingProvider(api_key="k", client=client)
    hits = SemanticRetriever(provider).retrieve(query, conv)
    assert hits[0].message.content.startswith("PostgreSQL")
