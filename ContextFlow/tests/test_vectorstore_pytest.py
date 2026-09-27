# tests/test_vectorstore_pytest.py
"""Step 15 (pytest): vector-store operations and vector-backed retrieval.

All backends are mocked or in-memory fakes -- no ChromaDB server, no network.
Covers the VectorStore interface, the ChromaDB backend mapping, and
VectorStoreRetriever indexing/querying with keyword + in-memory fallbacks.
"""

import math
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from contextflow import (
    BaseVectorStore,
    ChromaVectorStore,
    ContextStore,
    Conversation,
    EmbeddingError,
    EmbeddingProvider,
    ImportanceLevel,
    KeywordRetriever,
    Message,
    RetrievalResult,
    RetrieverError,
    VectorHit,
    VectorStore,
    VectorStoreError,
    VectorStoreRetriever,
)
from contextflow.vectorstores.base import similarity_from_distance


# ── fakes (no backend, no network) ──────────────────────────────────


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


class FailingProvider(EmbeddingProvider):
    def embed(self, texts):
        raise EmbeddingError("Simulated outage.")


class FakeVectorStore(VectorStore):
    """In-memory dict backend with cosine scoring (mirrors Chroma semantics)."""

    def __init__(self):
        self.records = {}
        self.calls = {"query": 0}

    def add(self, ids, contents, metadatas=None, embeddings=None):
        clean_ids, items, metas, vecs = self._validate_write(ids, contents, metadatas, embeddings)
        for i in clean_ids:
            if i in self.records:
                raise VectorStoreError(f"Duplicate id {i!r}.")
        for i, c, m, v in zip(clean_ids, items, metas,
                              vecs if vecs is not None else [None] * len(clean_ids)):
            self.records[i] = (c, m, v or [1.0])
        return len(clean_ids)

    def upsert(self, ids, contents, metadatas=None, embeddings=None):
        clean_ids, items, metas, vecs = self._validate_write(ids, contents, metadatas, embeddings)
        for i, c, m, v in zip(clean_ids, items, metas,
                              vecs if vecs is not None else [None] * len(clean_ids)):
            self.records[i] = (c, m, v or [1.0])
        return len(clean_ids)

    def delete(self, ids):
        clean_ids = self._validate_ids(ids)
        for i in clean_ids:
            self.records.pop(i, None)
        return len(clean_ids)

    def exists(self, ids):
        clean_ids = self._validate_ids(ids)
        return [i in self.records for i in clean_ids]

    def query(self, query_embeddings, top_k=5):
        limit = self._validate_top_k(top_k)
        self.calls["query"] += 1
        vecs = list(query_embeddings)
        if not vecs or not self.records:
            return []
        q = [float(v) for v in vecs[0]]
        ranked = []
        for rid, (content, meta, vector) in self.records.items():
            dot = sum(a * b for a, b in zip(q, vector))
            na = math.sqrt(sum(a * a for a in q)) or 1.0
            nb = math.sqrt(sum(b * b for b in vector)) or 1.0
            dist = 1.0 - max(-1.0, min(1.0, dot / (na * nb)))
            ranked.append(VectorHit(
                id=rid, content=content, metadata=dict(meta),
                score=similarity_from_distance(dist), distance=dist))
        ranked.sort(key=lambda h: h.distance)
        return ranked[:limit]

    def count(self):
        return len(self.records)

    def clear(self):
        self.records.clear()


class BrokenVectorStore(VectorStore):
    """Fails every backend call (tests the fallback chain)."""

    def add(self, *a, **k):
        raise VectorStoreError("Store is down.")

    def upsert(self, *a, **k):
        raise VectorStoreError("Store is down.")

    def delete(self, *a, **k):
        raise VectorStoreError("Store is down.")

    def exists(self, *a, **k):
        raise VectorStoreError("Store is down.")

    def query(self, *a, **k):
        raise VectorStoreError("Store is down.")

    def count(self, *a, **k):
        raise VectorStoreError("Store is down.")

    def clear(self, *a, **k):
        raise VectorStoreError("Store is down.")


def _vector_conversation():
    conv = Conversation(name="Vector", max_tokens=8000)
    conv.add(Message(
        role="assistant",
        content="PostgreSQL storage decision for persistence.",
        importance=ImportanceLevel.CRITICAL,
    ))
    conv.add(Message(
        role="assistant",
        content="Packaging release with flit and setuptools.",
    ))
    conv.add(Message(
        role="user",
        content="Debug healthcheck ping.",
        importance=ImportanceLevel.DISCARDABLE,
    ))
    return conv


def _paraphrase_mapping():
    query = "where do we persist records?"
    return query, {
        query: [1.0, 0.0],
        "PostgreSQL storage decision for persistence.": [1.0, 0.0],
        "Packaging release with flit and setuptools.": [0.0, 1.0],
        "Debug healthcheck ping.": [-1.0, 0.0],
    }


def _mock_chroma_collection():
    return MagicMock()


def _chroma_with_mock():
    client = MagicMock()
    collection = _mock_chroma_collection()
    client.get_or_create_collection.return_value = collection
    with patch.dict(os.environ, {}, clear=True):
        store = ChromaVectorStore(collection_name="test", client=client)
    return store, client, collection


# ── interface ────────────────────────────────────────────────────────


def test_vectorstore_is_abstract():
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore
    assert BaseVectorStore is VectorStore


def test_similarity_from_distance_scale():
    assert similarity_from_distance(0.0) == 1.0
    assert similarity_from_distance(1.0) == 0.5
    assert similarity_from_distance(2.0) == 0.0
    assert similarity_from_distance(5.0) == 0.0  # clamped, never negative


@pytest.mark.parametrize("bad", [[], ["", "b"], ["a", "a"], [123]])
def test_fake_store_rejects_bad_ids(bad):
    store = FakeVectorStore()
    with pytest.raises(VectorStoreError):
        store.add(bad, ["x"] * max(len(bad), 1))


def test_fake_store_rejects_mismatched_lengths():
    store = FakeVectorStore()
    with pytest.raises(VectorStoreError, match="must match"):
        store.add(["a", "b"], ["only one"])
    with pytest.raises(VectorStoreError, match="must match"):
        store.add(["a"], ["x"], embeddings=[[1.0], [2.0]])


def test_fake_store_rejects_bad_top_k():
    store = FakeVectorStore()
    with pytest.raises(VectorStoreError, match="positive integer"):
        store.query([[1.0]], top_k=0)


# ── fake backend CRUD ────────────────────────────────────────────────


def test_add_exists_count_delete_clear():
    store = FakeVectorStore()
    assert store.count() == 0
    assert store.add(["a", "b"], ["hello", "world"],
                     [{"role": "user"}, {"role": "assistant"}],
                     [[1.0, 0.0], [0.0, 1.0]]) == 2
    assert store.count() == 2
    assert store.exists(["a", "missing"]) == [True, False]
    assert store.delete(["a"]) == 1
    assert store.exists(["a"]) == [False]
    store.clear()
    assert store.count() == 0


def test_add_duplicate_ids_rejected():
    store = FakeVectorStore()
    store.add(["a"], ["hello"], embeddings=[[1.0]])
    with pytest.raises(VectorStoreError, match="Duplicate"):
        store.add(["a"], ["other"], embeddings=[[0.0]])


def test_upsert_updates_content_and_vector():
    store = FakeVectorStore()
    store.add(["a"], ["old text"], embeddings=[[1.0, 0.0]])
    store.upsert(["a"], ["new text"], embeddings=[[0.0, 1.0]])
    hits = store.query([[0.0, 1.0]], top_k=5)
    assert hits[0].content == "new text"
    assert hits[0].score == 1.0


def test_query_ranks_by_cosine_and_respects_top_k():
    store = FakeVectorStore()
    store.add(["a", "b", "c"], ["x", "y", "z"],
              embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    hits = store.query([[1.0, 0.0]], top_k=2)
    assert [h.id for h in hits] == ["a", "c"]
    assert hits[0].score > hits[1].score
    assert store.query([[1.0, 0.0]], top_k=10)[-1].id == "b"


def test_query_empty_store_returns_empty():
    assert FakeVectorStore().query([[1.0, 0.0]], top_k=3) == []


# ── Chroma backend mapping (mocked client) ───────────────────────────


def test_chroma_add_maps_arguments():
    store, _client, collection = _chroma_with_mock()
    assert store.add(["a"], ["hello"], [{"role": "user"}], [[1.0, 0.0]]) == 1
    kwargs = collection.add.call_args[1]
    assert kwargs["ids"] == ["a"]
    assert kwargs["documents"] == ["hello"]
    assert kwargs["metadatas"] == [{"role": "user"}]
    assert kwargs["embeddings"] == [[1.0, 0.0]]


def test_chroma_metadata_sanitized():
    store, _client, collection = _chroma_with_mock()
    store.add(["a"], ["hi"],
              [{"role": "user", "protected": True, "gone": None,
                "nested": {"a": 1}, "count": 3}])
    sent = collection.add.call_args[1]["metadatas"][0]
    assert sent["role"] == "user"
    assert sent["protected"] is True
    assert "gone" not in sent
    assert sent["nested"] == '{"a": 1}'
    assert sent["count"] == 3


def test_chroma_query_maps_response_to_hits():
    store, _client, collection = _chroma_with_mock()
    collection.count.return_value = 2
    collection.query.return_value = {
        "ids": [["a", "b"]],
        "documents": [["hello", "world"]],
        "metadatas": [[{"role": "user"}, {"role": "assistant"}]],
        "distances": [[0.0, 1.0]],
    }
    hits = store.query([[1.0, 0.0]], top_k=2)
    assert [h.id for h in hits] == ["a", "b"]
    assert hits[0].score == 1.0
    assert hits[1].score == 0.5
    assert hits[0].metadata == {"role": "user"}
    assert collection.query.call_args[1]["n_results"] == 2


def test_chroma_query_empty_store_skips_backend():
    store, _client, collection = _chroma_with_mock()
    collection.count.return_value = 0
    assert store.query([[1.0, 0.0]], top_k=3) == []
    collection.query.assert_not_called()


def test_chroma_errors_wrapped():
    store, _client, collection = _chroma_with_mock()
    collection.add.side_effect = RuntimeError("boom")
    with pytest.raises(VectorStoreError, match="add failed"):
        store.add(["a"], ["hi"])
    collection.query.side_effect = RuntimeError("boom")
    collection.count.return_value = 1
    with pytest.raises(VectorStoreError, match="query failed"):
        store.query([[1.0]], top_k=1)
    collection.count.side_effect = RuntimeError("boom")
    with pytest.raises(VectorStoreError, match="count failed"):
        store.count()


def test_chroma_missing_package_hint():
    with patch.dict(sys.modules, {"chromadb": None}):
        with pytest.raises(VectorStoreError, match="pip install chromadb"):
            ChromaVectorStore(collection_name="x")


def test_chroma_env_configuration():
    client = MagicMock()
    env = {"CHROMA_COLLECTION": "env-coll", "CHROMA_PERSIST_DIR": "/tmp/chroma-x"}
    with patch.dict(os.environ, env, clear=True):
        store = ChromaVectorStore(client=client)
        assert store.collection_name == "env-coll"
        assert store.persist_dir == "/tmp/chroma-x"
        client.get_or_create_collection.assert_called_once()


def test_chroma_invalid_configuration():
    with pytest.raises(ValueError, match="non-empty string"):
        ChromaVectorStore(collection_name="  ", client=MagicMock())
    with pytest.raises(TypeError, match="persist_dir"):
        ChromaVectorStore(collection_name="c", persist_dir=123, client=MagicMock())  # type: ignore


def test_chroma_rejects_bad_query_vectors():
    store, _client, _collection = _chroma_with_mock()
    with pytest.raises(VectorStoreError, match="non-empty lists"):
        store.query([[]], top_k=1)
    with pytest.raises(VectorStoreError, match="positive integer"):
        store.query([[1.0]], top_k=-2)


def test_chroma_clear_deletes_all():
    store, _client, collection = _chroma_with_mock()
    collection.get.return_value = {"ids": ["a", "b"]}
    store.clear()
    collection.delete.assert_called_once_with(ids=["a", "b"])


# ── VectorStoreRetriever ─────────────────────────────────────────────


def test_index_then_query_without_scanning():
    query, mapping = _paraphrase_mapping()
    conv = _vector_conversation()
    provider = FakeProvider(mapping)
    backend = FakeVectorStore()
    retriever = VectorStoreRetriever(provider, backend)

    assert retriever.index(conv) == 3
    assert backend.count() == 3
    assert retriever.indexed_count == 3

    backend.calls["query"] = 0
    hits = retriever.retrieve(query, conv)
    assert backend.calls["query"] == 1  # one store query, no Python scan
    assert hits[0].message.content.startswith("PostgreSQL")
    assert hits[0].score == 1.0
    assert isinstance(hits[0], RetrievalResult)


def test_second_retrieve_embeds_nothing_new():
    query, mapping = _paraphrase_mapping()
    conv = _vector_conversation()
    provider = FakeProvider(mapping)
    retriever = VectorStoreRetriever(provider, FakeVectorStore())
    retriever.retrieve(query, conv)
    calls = provider.call_count
    retriever.retrieve(query, conv)
    assert provider.call_count == calls


def test_hit_messages_restore_role_and_flags():
    query, mapping = _paraphrase_mapping()
    conv = _vector_conversation()
    retriever = VectorStoreRetriever(FakeProvider(mapping), FakeVectorStore())
    hits = retriever.retrieve(query, conv, top_k=3)
    first = next(h for h in hits if h.message.content.startswith("PostgreSQL"))
    assert first.message.role == "assistant"
    assert first.message.importance == ImportanceLevel.CRITICAL


def test_update_reindex_replaces_vector():
    conv = _vector_conversation()
    provider = FakeProvider({
        "PostgreSQL storage decision for persistence.": [1.0, 0.0],
        "Totally rewritten packaging note.": [1.0, 0.0],
        "q": [1.0, 0.0],
    })
    backend = FakeVectorStore()
    retriever = VectorStoreRetriever(provider, backend)
    retriever.index(conv)
    conv.get_messages()[1].content = "Totally rewritten packaging note."
    # Same ID scheme would collide on content hash; new content = new ID.
    retriever.index(conv)
    hits = retriever.retrieve("q", conv)
    assert any("rewritten" in h.message.content for h in hits)


def test_delete_removes_hits():
    conv = _vector_conversation()
    provider = FakeProvider({
        "PostgreSQL storage decision for persistence.": [1.0, 0.0],
        "q": [1.0, 0.0],
    }, default=[0.0, 1.0])
    backend = FakeVectorStore()
    retriever = VectorStoreRetriever(provider, backend)
    retriever.index(conv)
    from contextflow.embeddings import _content_key
    assert retriever.remove([_content_key("PostgreSQL storage decision for persistence.")]) == 1
    assert retriever.indexed_count == 2
    conv.remove(0)  # drop from the conversation too (else retrieve re-indexes it)
    hits = retriever.retrieve("q", conv, top_k=10)
    assert all("PostgreSQL" not in h.message.content for h in hits)
    calls = provider.call_count
    retriever.retrieve("q", conv, top_k=10)
    assert provider.call_count == calls  # nothing re-embedded
    # Store-only queries exclude it as well.
    assert all("PostgreSQL" not in h.message.content
               for h in retriever.retrieve("q", None, top_k=10))


def test_remove_rejects_bad_ids():
    retriever = VectorStoreRetriever(FakeProvider(), FakeVectorStore())
    with pytest.raises(VectorStoreError, match="must not be empty"):
        retriever.remove([])


def test_source_none_queries_indexed_store():
    query, mapping = _paraphrase_mapping()
    conv = _vector_conversation()
    retriever = VectorStoreRetriever(FakeProvider(mapping), FakeVectorStore())
    retriever.index(conv)
    hits = retriever.retrieve(query)  # no source: store-only query
    assert hits[0].message.content.startswith("PostgreSQL")


def test_store_failure_falls_back_to_in_memory():
    query, mapping = _paraphrase_mapping()
    conv = _vector_conversation()
    retriever = VectorStoreRetriever(FakeProvider(mapping), BrokenVectorStore())
    hits = retriever.retrieve(query, conv)
    assert hits[0].message.content.startswith("PostgreSQL")
    assert hits[0].score == 1.0


def test_store_and_provider_failure_falls_back_to_keyword():
    conv = _vector_conversation()
    retriever = VectorStoreRetriever(FailingProvider(), BrokenVectorStore())
    hits = retriever.retrieve("PostgreSQL storage", conv)
    expected = KeywordRetriever().retrieve("PostgreSQL storage", conv)
    assert [h.message.content for h in hits] == [h.message.content for h in expected]


def test_no_fallback_raises_on_total_failure():
    conv = _vector_conversation()
    retriever = VectorStoreRetriever(FailingProvider(), BrokenVectorStore(), fallback=False)
    with pytest.raises(RetrieverError):
        retriever.retrieve("PostgreSQL", conv)


def test_source_none_with_broken_store_raises():
    retriever = VectorStoreRetriever(FakeProvider(), BrokenVectorStore(), fallback=False)
    with pytest.raises(RetrieverError, match="no source"):
        retriever.retrieve("query")


def test_top_k_min_score_and_empty_query():
    query, mapping = _paraphrase_mapping()
    conv = _vector_conversation()
    retriever = VectorStoreRetriever(FakeProvider(mapping), FakeVectorStore())
    assert len(retriever.retrieve(query, conv, top_k=1)) == 1
    assert retriever.retrieve("", conv) == []
    assert retriever.retrieve("   ", conv) == []
    strict = VectorStoreRetriever(FakeProvider(mapping), FakeVectorStore(), min_score=0.99)
    assert [h.message.content for h in strict.retrieve(query, conv)] == [
        "PostgreSQL storage decision for persistence."]


def test_invalid_inputs_rejected():
    conv = _vector_conversation()
    with pytest.raises(TypeError, match="VectorStore"):
        VectorStoreRetriever(FakeProvider(), vector_store="nope")  # type: ignore
    retriever = VectorStoreRetriever(FakeProvider(), FakeVectorStore())
    with pytest.raises(TypeError, match="query must be a string"):
        retriever.retrieve(123, conv)
    with pytest.raises(TypeError, match="source must be"):
        retriever.retrieve("q", source="nope")  # type: ignore


def test_conversation_search_with_vector_store():
    query, mapping = _paraphrase_mapping()
    conv = _vector_conversation()
    hits = conv.search(query, provider=FakeProvider(mapping),
                       vector_store=FakeVectorStore())
    assert hits[0].message.content.startswith("PostgreSQL")


def test_conversation_search_store_without_provider_rejected():
    conv = _vector_conversation()
    with pytest.raises(TypeError, match="requires an EmbeddingProvider"):
        conv.search("q", vector_store=FakeVectorStore())


def test_search_does_not_mutate_conversation():
    conv = _vector_conversation()
    before = [m.content for m in conv.get_messages()]
    conv.search("PostgreSQL storage", provider=FakeProvider(),
                vector_store=FakeVectorStore())
    assert [m.content for m in conv.get_messages()] == before


def test_index_from_contextstore_source():
    query, mapping = _paraphrase_mapping()
    store = ContextStore(_vector_conversation())
    retriever = VectorStoreRetriever(FakeProvider(mapping), FakeVectorStore())
    assert retriever.index(store) == 3
    assert retriever.retrieve(query, store)[0].message.content.startswith("PostgreSQL")


def test_keyword_and_in_memory_still_available():
    from contextflow import SemanticRetriever
    conv = _vector_conversation()
    assert KeywordRetriever().retrieve("PostgreSQL storage", conv)
    query, mapping = _paraphrase_mapping()
    in_memory = SemanticRetriever(FakeProvider(mapping))
    assert in_memory.retrieve(query, conv)[0].message.content.startswith("PostgreSQL")
