# tests/test_retriever_pytest.py
"""Step 13 (pytest): keyword-based relevant context retrieval.

Covers relevant matches, irrelevant queries, ranking order, empty results,
top_k limits, duplicate-content suppression, compaction-summary search, and
Conversation.search integration. Every test builds fresh objects.
"""

import pytest

from contextflow import (
    ContextStore,
    Conversation,
    ImportanceLevel,
    KeywordRetriever,
    Message,
    RetrievalResult,
    Retriever,
    RetrieverError,
)


def _retrieval_conversation() -> Conversation:
    conv = Conversation(name="Retrieval", max_tokens=8000)
    conv.add(Message(
        role="system",
        content="You are an expert software architect.",
    ))
    conv.add(Message(
        role="assistant",
        content="Core Decision: Architecture uses PostgreSQL 16 with pgvector for storage.",
        importance=ImportanceLevel.CRITICAL,
    ))
    conv.add(Message(
        role="user",
        content="How should we handle GitHub authentication for the agent?",
    ))
    conv.add(Message(
        role="assistant",
        content="Use pyproject.toml with flit or setuptools for packaging.",
    ))
    conv.add(Message(
        role="user",
        content="Debug: Pod healthcheck HTTP 200 latency 5ms.",
        importance=ImportanceLevel.DISCARDABLE,
    ))
    return conv


# ── relevant matches ─────────────────────────────────────────────


def test_relevant_query_returns_matching_message():
    conv = _retrieval_conversation()
    hits = KeywordRetriever().retrieve("PostgreSQL pgvector storage", conv)
    assert len(hits) >= 1
    assert "PostgreSQL" in hits[0].message.content
    assert hits[0].score > 0.0
    assert isinstance(hits[0], RetrievalResult)


def test_match_reports_query_terms():
    conv = _retrieval_conversation()
    hits = KeywordRetriever().retrieve("github authentication", conv)
    assert len(hits) >= 1
    assert "github" in hits[0].matched_terms
    assert "authentication" in hits[0].matched_terms


def test_search_is_case_insensitive():
    conv = _retrieval_conversation()
    lower = KeywordRetriever().retrieve("postgresql storage", conv)
    upper = KeywordRetriever().retrieve("POSTGRESQL STORAGE", conv)
    assert [h.message.content for h in lower] == [h.message.content for h in upper]
    assert [h.score for h in lower] == [h.score for h in upper]


def test_scores_descend_in_rank_order():
    conv = _retrieval_conversation()
    hits = KeywordRetriever(top_k=10).retrieve("PostgreSQL storage packaging authentication", conv)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


# ── ranking ──────────────────────────────────────────────────────


def test_message_matching_more_terms_ranks_first():
    conv = Conversation(name="Rank")
    conv.add(Message(role="user", content="PostgreSQL is great."))
    conv.add(Message(role="user", content="PostgreSQL pgvector storage decision for persistence layer."))
    hits = KeywordRetriever().retrieve("postgresql pgvector storage decision", conv)
    assert len(hits) == 2
    assert "persistence layer" in hits[0].message.content
    assert hits[0].score > hits[1].score


def test_exact_phrase_beats_scattered_terms():
    conv = Conversation(name="Phrase")
    conv.add(Message(role="user", content="Storage matters. PostgreSQL matters. Decisions matter."))
    conv.add(Message(role="user", content="PostgreSQL storage decision"))
    hits = KeywordRetriever().retrieve("PostgreSQL storage decision", conv)
    assert hits[0].message.content == "PostgreSQL storage decision"


def test_repeated_calls_are_deterministic():
    conv = _retrieval_conversation()
    retriever = KeywordRetriever()
    first = retriever.retrieve("storage packaging github", conv)
    second = retriever.retrieve("storage packaging github", conv)
    assert [(h.message.content, h.score) for h in first] == [
        (h.message.content, h.score) for h in second
    ]


# ── irrelevant queries & empty results ───────────────────────────


def test_irrelevant_query_returns_empty():
    conv = _retrieval_conversation()
    assert KeywordRetriever().retrieve("quantum banana trombone", conv) == []


def test_empty_query_returns_empty():
    conv = _retrieval_conversation()
    assert KeywordRetriever().retrieve("", conv) == []
    assert KeywordRetriever().retrieve("   ", conv) == []


def test_stopword_only_query_returns_empty():
    conv = _retrieval_conversation()
    assert KeywordRetriever().retrieve("what is the", conv) == []


def test_empty_conversation_returns_empty():
    conv = Conversation(name="Empty")
    assert KeywordRetriever().retrieve("postgresql", conv) == []


def test_empty_message_list_returns_empty():
    assert KeywordRetriever().retrieve("postgresql", []) == []


# ── top_k ────────────────────────────────────────────────────────


def test_top_k_limits_results():
    conv = _retrieval_conversation()
    hits = KeywordRetriever(top_k=2).retrieve(
        "PostgreSQL storage packaging authentication healthcheck architect", conv
    )
    assert len(hits) == 2


def test_per_call_top_k_overrides_default():
    conv = _retrieval_conversation()
    retriever = KeywordRetriever(top_k=5)
    query = "PostgreSQL storage packaging authentication healthcheck architect"
    assert len(retriever.retrieve(query, conv, top_k=1)) == 1
    assert len(retriever.retrieve(query, conv)) >= 2


def test_top_k_larger_than_matches_returns_all():
    conv = _retrieval_conversation()
    hits = KeywordRetriever(top_k=50).retrieve("PostgreSQL", conv)
    assert 1 <= len(hits) < 50


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_top_k_value_rejected(bad):
    with pytest.raises(ValueError, match="positive integer"):
        KeywordRetriever(top_k=bad)


@pytest.mark.parametrize("bad", ["3", 2.5, True])
def test_invalid_top_k_type_rejected(bad):
    with pytest.raises(TypeError, match="positive integer"):
        KeywordRetriever(top_k=bad)


def test_invalid_per_call_top_k_rejected():
    conv = _retrieval_conversation()
    with pytest.raises(ValueError, match="positive integer"):
        KeywordRetriever().retrieve("postgresql", conv, top_k=0)


# ── duplicates & summaries ───────────────────────────────────────


def test_duplicate_content_returned_once():
    conv = Conversation(name="Dupes")
    conv.add(Message(role="user", content="PostgreSQL storage decision."))
    conv.add(Message(role="assistant", content="PostgreSQL storage decision."))  # exact copy
    conv.add(Message(role="user", content="Something about packaging."))
    hits = KeywordRetriever(top_k=10).retrieve("PostgreSQL storage decision", conv)
    contents = [h.message.content for h in hits]
    assert contents.count("PostgreSQL storage decision.") == 1


def test_whitespace_variant_counts_as_duplicate():
    conv = Conversation(name="Dupes")
    conv.add(Message(role="user", content="PostgreSQL  storage decision."))
    conv.add(Message(role="user", content="postgresql storage decision."))
    hits = KeywordRetriever(top_k=10).retrieve("PostgreSQL storage decision", conv)
    assert len(hits) == 1


def test_compaction_summaries_are_searchable():
    conv = _retrieval_conversation()
    conv.add(Message(
        role="system",
        content="[CONTEXT SUMMARY -- 3 older message(s) compacted] PostgreSQL storage decided.",
        metadata={"type": "compaction_summary"},
    ))
    hits = KeywordRetriever(top_k=10).retrieve("PostgreSQL storage", conv)
    assert any(m.metadata.get("type") == "compaction_summary"
               for m in (h.message for h in hits))


# ── sources & integration ────────────────────────────────────────


def test_message_list_source_works():
    msgs = [
        Message(role="user", content="PostgreSQL storage decision."),
        Message(role="user", content="Unrelated packaging note."),
    ]
    hits = KeywordRetriever().retrieve("PostgreSQL storage", msgs)
    assert len(hits) == 1
    assert hits[0].message.content.startswith("PostgreSQL")


def test_contextstore_source_works():
    conv = _retrieval_conversation()
    store = ContextStore(conv)
    hits = KeywordRetriever().retrieve("PostgreSQL storage", store)
    assert len(hits) >= 1
    assert "PostgreSQL" in hits[0].message.content


def test_conversation_search_matches_retriever():
    conv = _retrieval_conversation()
    via_method = conv.search("PostgreSQL storage")
    via_retriever = KeywordRetriever().retrieve("PostgreSQL storage", conv)
    assert [(h.message.content, h.score) for h in via_method] == [
        (h.message.content, h.score) for h in via_retriever
    ]


def test_conversation_search_honours_top_k():
    conv = _retrieval_conversation()
    hits = conv.search("PostgreSQL storage packaging authentication", top_k=1)
    assert len(hits) == 1


def test_search_does_not_modify_conversation():
    conv = _retrieval_conversation()
    before = [m.content for m in conv.get_messages()]
    tokens_before = conv.total_tokens()
    conv.search("PostgreSQL")
    KeywordRetriever().retrieve("PostgreSQL", conv)
    assert [m.content for m in conv.get_messages()] == before
    assert conv.total_tokens() == tokens_before
    assert conv.message_count() == len(before)


def test_existing_conversation_behavior_unchanged():
    conv = _retrieval_conversation()
    count, tokens, status = conv.message_count(), conv.total_tokens(), conv.get_status()
    conv.search("anything at all")
    assert (conv.message_count(), conv.total_tokens(), conv.get_status()) == (count, tokens, status)
    conv.add(Message(role="user", content="Follow-up."))
    assert conv.message_count() == count + 1


# ── invalid inputs ───────────────────────────────────────────────


def test_non_string_query_rejected():
    conv = _retrieval_conversation()
    with pytest.raises(TypeError, match="query must be a string"):
        KeywordRetriever().retrieve(123, conv)
    with pytest.raises(TypeError, match="query must be a string"):
        conv.search(None)


def test_bad_source_type_rejected():
    with pytest.raises(TypeError, match="source must be"):
        KeywordRetriever().retrieve("postgresql", "not a source")


def test_non_message_list_rejected():
    with pytest.raises(RetrieverError, match="only Message instances"):
        KeywordRetriever().retrieve("postgresql", ["nope"])


def test_retriever_is_abstract():
    with pytest.raises(TypeError):
        Retriever()  # type: ignore


def test_min_score_filters_weak_hits():
    conv = _retrieval_conversation()
    strict = KeywordRetriever(min_score=0.99)
    assert strict.retrieve("PostgreSQL storage packaging", conv) == []


def test_invalid_min_score_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        KeywordRetriever(min_score=1.5)
    with pytest.raises(TypeError, match="min_score must be"):
        KeywordRetriever(min_score="high")  # type: ignore
