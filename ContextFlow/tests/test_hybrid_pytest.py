# tests/test_hybrid_pytest.py
"""Step 16 (pytest): hybrid retrieval fusion and budget-aware context assembly.

Covers weighted ranking (weights flip the order), score normalization,
deduplication, top_k, token-budget exclusion order, protected/request
guarantees, and empty retrieval results. Every test builds fresh objects;
embedding providers are deterministic fakes (no network).
"""

import pytest

from contextflow import (
    AssembledContext,
    ContextAssembler,
    Conversation,
    ExcludedItem,
    HybridRetriever,
    KeywordRetriever,
    Message,
    RetrievalResult,
    Retriever,
    SemanticRetriever,
)
from contextflow.embeddings import EmbeddingProvider


class FakeProvider(EmbeddingProvider):
    """Maps exact texts to vectors (unmapped texts get the default)."""

    def __init__(self, mapping=None, default=(0.0, 1.0)):
        self.mapping = dict(mapping or {})
        self.default = list(default)
        self.batches = []

    def embed(self, texts):
        items = list(texts)
        self.batches.append(items)
        return [list(self.mapping.get(t, self.default)) for t in items]


def _hybrid_conversation():
    """Two topical messages engineered for weight-flip tests.

    Query "postgres storage": message A matches keywords strongly
    ("postgres" twice + "storage") but is semantically weak; message B
    matches nothing by keyword but is semantically identical to the query.
    """
    conv = Conversation(name="Hybrid", max_tokens=8000)
    conv.add(Message(
        role="assistant",
        content="Postgres postgres storage notes and storage details.",
    ))
    conv.add(Message(
        role="assistant",
        content="Completely unrelated wording about nothing relevant here.",
    ))
    conv.add(Message(
        role="user",
        content="Recent turn stays recent.",
    ))
    return conv


def _flip_provider(query):
    return FakeProvider({
        query: [1.0, 0.0],
        "Postgres postgres storage notes and storage details.": [0.0, 1.0],
        "Completely unrelated wording about nothing relevant here.": [1.0, 0.0],
        "Recent turn stays recent.": [0.0, -1.0],
    })


def _assembler_conversation():
    conv = Conversation(name="Assemble", max_tokens=8000)
    conv.add(Message(role="system", content="You are helpful."))
    conv.add(Message(
        role="user",
        content="Never store secrets in plain text.",
        protected=True,
    ))
    conv.add(Message(role="assistant", content="PostgreSQL storage decision for persistence."))
    conv.add(Message(role="assistant", content="Packaging release with flit and setuptools."))
    conv.add(Message(role="user", content="Older filler turn one."))
    conv.add(Message(role="user", content="Recent question about deploy?"))
    conv.add(Message(role="assistant", content="Recent answer about deploy steps."))
    return conv


def _hit(content, score=0.9, rank=1, role="assistant"):
    return RetrievalResult(
        score=score, message=Message(role=role, content=content),
        matched_terms=[], rank=rank)


# ── hybrid ranking ─────────────────────────────────────────────────


def test_keyword_heavy_weights_rank_keyword_match_first():
    conv = _hybrid_conversation()
    query = "postgres storage"
    hybrid = HybridRetriever(
        semantic=SemanticRetriever(_flip_provider(query)),
        keyword_weight=1.0, semantic_weight=0.0,
    )
    hits = hybrid.retrieve(query, conv)
    assert hits[0].message.content.startswith("Postgres postgres")


def test_semantic_heavy_weights_flip_the_order():
    conv = _hybrid_conversation()
    query = "postgres storage"
    hybrid = HybridRetriever(
        semantic=SemanticRetriever(_flip_provider(query)),
        keyword_weight=0.0, semantic_weight=1.0,
    )
    hits = hybrid.retrieve(query, conv)
    assert hits[0].message.content.startswith("Completely unrelated")


def test_fused_scores_stay_in_bounds_and_descend():
    conv = _hybrid_conversation()
    hybrid = HybridRetriever(
        semantic=SemanticRetriever(_flip_provider("postgres storage")),
        keyword_weight=0.5, semantic_weight=0.5,
    )
    hits = hybrid.retrieve("postgres storage", conv, top_k=10)
    scores = [h.score for h in hits]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_default_hybrid_is_keyword_only():
    conv = _hybrid_conversation()
    hybrid = HybridRetriever()
    hits = hybrid.retrieve("postgres storage", conv)
    expected = KeywordRetriever().retrieve("postgres storage", conv)
    assert [h.message.content for h in hits] == [h.message.content for h in expected]


def test_semantic_only_side_needs_no_keyword_weight():
    conv = _hybrid_conversation()
    query = "postgres storage"
    hybrid = HybridRetriever(
        semantic=SemanticRetriever(_flip_provider(query)),
        keyword_weight=0.0, semantic_weight=1.0,
    )
    assert hybrid.retrieve(query, conv)[0].message.content.startswith("Completely unrelated")


def test_top_k_and_per_source_k():
    conv = _hybrid_conversation()
    hybrid = HybridRetriever(
        semantic=SemanticRetriever(_flip_provider("postgres storage")))
    assert len(hybrid.retrieve("postgres storage", conv, top_k=1)) == 1
    wide = HybridRetriever(
        semantic=SemanticRetriever(_flip_provider("postgres storage")),
        top_k=1, per_source_k=10)
    assert len(wide.retrieve("postgres storage", conv)) == 1


def test_min_score_filters_fused_hits():
    conv = _hybrid_conversation()
    hybrid = HybridRetriever(
        semantic=SemanticRetriever(_flip_provider("postgres storage")),
        min_score=0.99)
    for hit in hybrid.retrieve("postgres storage", conv, top_k=10):
        assert hit.score > 0.99


def test_hybrid_deduplicates_shared_content():
    conv = Conversation(name="Dupes")
    conv.add(Message(role="user", content="Postgres storage answer."))
    conv.add(Message(role="assistant", content="Postgres storage answer."))  # same text
    hybrid = HybridRetriever(
        semantic=SemanticRetriever(FakeProvider({
            "postgres storage": [1.0, 0.0],
            "Postgres storage answer.": [1.0, 0.0],
        })))
    hits = hybrid.retrieve("postgres storage", conv, top_k=10)
    assert len(hits) == 1


def test_empty_results_when_nothing_matches():
    conv = _hybrid_conversation()
    keyword_only = HybridRetriever()  # keyword side only: nonsense matches nothing
    assert keyword_only.retrieve("quantum banana trombone xyzzy", conv) == []
    assert keyword_only.retrieve("", conv) == []
    assert keyword_only.retrieve("postgres", Conversation(name="Empty")) == []
    # A strict fused threshold can also empty a non-empty candidate set.
    strict = HybridRetriever(
        semantic=SemanticRetriever(_flip_provider("postgres storage")),
        min_score=1.0)
    assert strict.retrieve("postgres storage", conv) == []


def test_invalid_weights_and_inputs_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        HybridRetriever(keyword_weight=1.5)
    with pytest.raises(TypeError, match="must be a number"):
        HybridRetriever(semantic_weight="high")  # type: ignore
    with pytest.raises(ValueError, match="At least one"):
        HybridRetriever(keyword_weight=0.0, semantic_weight=0.0)
    with pytest.raises(TypeError, match="Retriever instance"):
        HybridRetriever(keyword="nope")  # type: ignore
    hybrid = HybridRetriever()
    with pytest.raises(TypeError, match="query must be a string"):
        hybrid.retrieve(123, _hybrid_conversation())
    with pytest.raises(ValueError, match="positive integer"):
        hybrid.retrieve("postgres", _hybrid_conversation(), top_k=0)


def test_hybrid_is_a_retriever():
    assert isinstance(HybridRetriever(), Retriever)


# ── assembly order ─────────────────────────────────────────────────


def test_assembly_order_system_protected_retrieved_recent_request():
    conv = _assembler_conversation()
    hits = [_hit("PostgreSQL storage decision for persistence.", score=0.9, rank=1)]
    ctx = ContextAssembler(max_tokens=8000, keep_recent=2).assemble(
        "where do we persist records?", conv, hits)
    assert isinstance(ctx, AssembledContext)
    roles_contents = [(m.role, m.content) for m in ctx.messages]
    assert roles_contents[0] == ("system", "You are helpful.")
    assert roles_contents[1] == ("user", "Never store secrets in plain text.")
    assert roles_contents[2][1] == "PostgreSQL storage decision for persistence."
    # Recent pair keeps conversation order, request is always last.
    assert roles_contents[-3][1] == "Recent question about deploy?"
    assert roles_contents[-2][1] == "Recent answer about deploy steps."
    assert roles_contents[-1] == ("user", "where do we persist records?")
    assert ctx.within_budget is True
    assert ctx.total_tokens == sum(m.token_count for m in ctx.messages)


def test_request_accepts_message_object():
    conv = _assembler_conversation()
    req = Message(role="user", content="Custom request object.")
    ctx = ContextAssembler(max_tokens=8000).assemble(req, conv, [])
    assert ctx.messages[-1] is req


def test_invalid_request_and_inputs_rejected():
    conv = _assembler_conversation()
    assembler = ContextAssembler(max_tokens=100)
    with pytest.raises(TypeError, match="string or Message"):
        assembler.assemble(123, conv, [])  # type: ignore
    with pytest.raises(TypeError, match="Conversation"):
        assembler.assemble("hi", "nope", [])  # type: ignore
    with pytest.raises(TypeError, match="RetrievalResult"):
        assembler.assemble("hi", conv, ["nope"])  # type: ignore
    with pytest.raises(ValueError, match="positive integer"):
        ContextAssembler(max_tokens=0)
    with pytest.raises(TypeError, match="positive integer"):
        ContextAssembler(max_tokens="100")  # type: ignore
    with pytest.raises(ValueError, match="non-negative"):
        ContextAssembler(max_tokens=100, keep_recent=-1)


# ── budget discipline ──────────────────────────────────────────────


def test_lowest_scoring_retrieved_excluded_first():
    conv = _assembler_conversation()
    mandatory = (
        conv.get_messages()[0].token_count  # system
        + conv.get_messages()[1].token_count  # protected
    )
    hits = [
        _hit("PostgreSQL storage decision for persistence.", score=0.95, rank=1),
        _hit("Packaging release with flit and setuptools.", score=0.50, rank=2),
        _hit("Older filler turn one.", score=0.10, rank=3),
    ]
    # Room for mandatory + best retrieved hit + the request only (keep_recent=0).
    best = hits[0].message.token_count
    request_tokens = Message(role="user", content="q?").token_count
    budget = mandatory + best + request_tokens
    ctx = ContextAssembler(max_tokens=budget, keep_recent=0).assemble("q?", conv, hits)
    contents = [m.content for m in ctx.messages]
    assert "PostgreSQL storage decision for persistence." in contents
    assert "Packaging release with flit and setuptools." not in contents
    assert "Older filler turn one." not in contents
    assert ctx.within_budget is True
    assert len(ctx.excluded) == 2
    assert all("over token budget" in e.reason for e in ctx.excluded)
    assert all(isinstance(e, ExcludedItem) for e in ctx.excluded)


def test_oldest_recent_excluded_first():
    conv = _assembler_conversation()
    mandatory = (
        conv.get_messages()[0].token_count
        + conv.get_messages()[1].token_count
    )
    newest = conv.get_messages()[-1].token_count
    budget = mandatory + newest  # only the newest recent fits
    ctx = ContextAssembler(max_tokens=budget, keep_recent=5).assemble("q?", conv, [])
    contents = [m.content for m in ctx.messages]
    assert "Recent answer about deploy steps." in contents
    assert "Recent question about deploy?" not in contents
    # Newest-first fill means the excluded recent is reported, oldest excluded first.
    assert any("Recent question about deploy?" in e.message.content for e in ctx.excluded)


def test_protected_and_request_survive_tiny_budget():
    conv = _assembler_conversation()
    ctx = ContextAssembler(max_tokens=5, keep_recent=5).assemble("request?", conv, [
        _hit("PostgreSQL storage decision for persistence.", score=0.9, rank=1),
    ])
    contents = [m.content for m in ctx.messages]
    assert "You are helpful." in contents  # system mandatory
    assert "Never store secrets in plain text." in contents  # protected mandatory
    assert "request?" in contents  # request mandatory
    assert ctx.within_budget is False  # honestly reported
    assert ctx.usage_percentage > 100.0


def test_usage_percentage_and_str():
    conv = _assembler_conversation()
    ctx = ContextAssembler(max_tokens=8000).assemble("q?", conv, [])
    assert ctx.usage_percentage == round(ctx.total_tokens / 8000 * 100, 1)
    text = str(ctx)
    assert "Assembled Context" in text
    assert "within budget" in text
    assert "Excluded" in text


# ── dedupe + empty retrieval ───────────────────────────────────────


def test_retrieved_duplicate_of_protected_is_excluded_with_reason():
    conv = _assembler_conversation()
    dup = RetrievalResult(
        score=0.99,
        message=Message(role="user", content="Never store secrets in plain text."),
        matched_terms=["secrets"], rank=1)
    ctx = ContextAssembler(max_tokens=8000).assemble("q?", conv, [dup])
    assert [m.content for m in ctx.messages].count("Never store secrets in plain text.") == 1
    assert len(ctx.excluded) == 1
    assert "duplicate" in ctx.excluded[0].reason


def test_empty_retrieval_still_assembles_backbone():
    conv = _assembler_conversation()
    ctx = ContextAssembler(max_tokens=8000, keep_recent=2).assemble("fresh question?", conv, [])
    contents = [m.content for m in ctx.messages]
    assert contents[0] == "You are helpful."
    assert "Never store secrets in plain text." in contents
    assert contents[-1] == "fresh question?"
    assert ctx.excluded == []


def test_empty_conversation_assembles_just_request():
    conv = Conversation(name="Empty")
    ctx = ContextAssembler(max_tokens=100).assemble("hello?", conv, [])
    assert [m.content for m in ctx.messages] == ["hello?"]
    assert ctx.within_budget is True


def test_assemble_does_not_mutate_conversation():
    conv = _assembler_conversation()
    before = [m.content for m in conv.get_messages()]
    tokens = conv.total_tokens()
    ContextAssembler(max_tokens=50, keep_recent=2).assemble(
        "q?", conv, [_hit("PostgreSQL storage decision for persistence.")])
    assert [m.content for m in conv.get_messages()] == before
    assert conv.total_tokens() == tokens


def test_end_to_end_hybrid_then_assemble():
    conv = _assembler_conversation()
    query = "where do we persist records?"
    provider = FakeProvider({
        query: [1.0, 0.0],
        "PostgreSQL storage decision for persistence.": [1.0, 0.0],
    })
    hybrid = HybridRetriever(semantic=SemanticRetriever(provider))
    hits = hybrid.retrieve(query, conv)
    assert hits  # semantic side finds the paraphrase
    ctx = ContextAssembler(max_tokens=8000, keep_recent=2).assemble(query, conv, hits)
    assert ctx.messages[-1].content == query
    assert ctx.within_budget is True
