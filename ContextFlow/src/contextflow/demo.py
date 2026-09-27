# main.py
"""
ContextFlow -- Step 21: Intelligent Tool-Result Compaction

Demonstrates sixteen scenarios that show the compaction, validation, importance,
priority-scoring, persistence, retrieval, assembly, pipeline, LLM, agent, tool,
and result-processing workflows:
  1. HeuristicValidator (offline) -- keyword/pattern matching, no API key needed.
  2. OpenAIValidator (real LLM)   -- LLM-as-judge, requires OPENAI_API_KEY.
  3. Validation failure rollback  -- shows how a bad summary is safely discarded.
  4. on_validation_fail='warn'    -- applies compaction even when validation fails.
  5. Message Importance & Protected Context -- preserves protected/critical, summarizes normal, drops discardable.
  6. Automatic Priority Scoring -- HeuristicPriorityScorer scores every message 0-100,
     auto-preserves critical-scored content and auto-discards ephemeral chatter.
  7. Persistent Context Storage -- ContextStore saves a compacted conversation
     (messages, token counts, importance, protected flags, summary, validation)
     to JSON and loads it back with identical behaviour.
  8. Relevant Context Retrieval -- KeywordRetriever ranks persisted messages
     (including compaction summaries) by keyword relevance with top_k limits.
  9. Semantic Context Retrieval -- SemanticRetriever ranks by embedding cosine
     similarity (OpenAI-compatible provider, cached, keyword fallback).
  10. Vector Database Support -- VectorStoreRetriever indexes messages in
      local ChromaDB and queries the index (add/update/delete/search).
  11. Hybrid Retrieval & Assembly -- HybridRetriever fuses keyword + semantic
      scores; ContextAssembler packs system, protected, retrieved, recent,
      and the request into a token budget.
  12. Compaction Pipeline -- ContextPipeline monitors usage, compacts only
      past the trigger threshold, validates, retrieves context, and persists.
  13. LLM Context Adapter -- LLMClient sends assembled context + instructions
      + request to the model; returns reply, tokens, latency, model name.
  14. Simple Agent -- Agent.run() retrieves, assembles, answers, records
      history, and compacts when needed. No tools, no planning.
  15. Basic Tool Calling -- Agent executes read_file / list_directory via
      ```tool blocks, feeds observations back, and records token-counted
      tool messages. No shell execution.
  16. Tool-Result Compaction -- ToolResultProcessor shrinks large outputs
      (text / listings / JSON) for active context while archiving originals.

Set up your .env file before running for live LLM calls:
    copy .env.example .env
    # Add your OPENAI_API_KEY
"""

import os
from contextflow import (
    Message, Conversation, Compactor, ImportanceLevel,
    PlaceholderSummarizer, OpenAISummarizer, SummarizerError,
    HeuristicValidator, OpenAIValidator, ValidatorError,
    ValidationResult, HeuristicPriorityScorer,
    ContextStore, StoreError,
    KeywordRetriever,
    SemanticRetriever, EmbeddingProvider, OpenAIEmbeddingProvider, EmbeddingError,
    VectorStoreRetriever, ChromaVectorStore, VectorStoreError,
    HybridRetriever, ContextAssembler,
    ContextPipeline,
    LLMClient, OpenAILLMClient, LLMError,
    Agent, AssembledContext,
    ReadFileTool, ListDirectoryTool,
    ToolResultProcessor,
)
from contextflow.summarizers.base import Summarizer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── helpers ──────────────────────────────────────────────────────────────────

def build_conversation(name: str, max_tokens: int) -> Conversation:
    """Build a realistic multi-turn conversation for demo purposes."""
    conv = Conversation(name=name, max_tokens=max_tokens)
    conv.add(Message(
        role="system",
        content="You are an expert software architect helping a team design an AI agent system.",
    ))
    conv.add(Message(
        role="user",
        content="We want to build an AI coding agent. It must be able to read files, run tests, and open pull requests.",
    ))
    conv.add(Message(
        role="assistant",
        content='I recommend a three-layer architecture: a planning layer (LLM), a tool layer (file I/O, shell, GitHub API), and a memory layer. We should use "Python" as the primary language.',
    ))
    conv.add(Message(
        role="user",
        content="How should we handle GitHub authentication?",
    ))
    conv.add(Message(
        role="assistant",
        content='Use a "GitHub App" rather than a personal token. Apps have scoped permissions, better rate limits, and can be installed on specific repos. Store the private key as an environment variable.',
    ))
    conv.add(Message(
        role="user",
        content="What about context management for the agent?",
    ))
    conv.add(Message(
        role="assistant",
        content="Context management is the most critical component. The agent must stay within its token limit across long sessions. Implement ContextFlow-style compaction from day one.",
    ))
    conv.add(Message(
        role="user",
        content="What is the first thing we should build?",
    ))
    conv.add(Message(
        role="assistant",
        content="Start with the tool layer. Define a clear interface for each tool, then build a planning loop that calls tools based on LLM output. Keep the initial scope to file reading and test running.",
    ))
    return conv


def print_status(conv: Conversation, label: str) -> None:
    print(f"  [{label}]")
    print(f"    Messages : {conv.message_count()}")
    print(f"    Tokens   : {conv._budget_line()}")
    print(f"    Status   : {conv._status_icon()}")


def print_validation(vr: ValidationResult) -> None:
    print(str(vr))


# ── Scenario 1: HeuristicValidator (offline, no API key) ─────────────────────

def scenario_heuristic_validator() -> None:
    print("=" * 60)
    print("  Scenario 1: HeuristicValidator (offline, no API key)")
    print("=" * 60)
    conv = build_conversation("Heuristic Validation Demo", max_tokens=500)

    print()
    print_status(conv, "BEFORE compaction")

    # PlaceholderSummarizer + HeuristicValidator
    compactor = Compactor(
        keep_recent=3,
        summarizer=PlaceholderSummarizer(),
        validator=HeuristicValidator(fail_threshold=0.8, warn_threshold=0.3),
        on_validation_fail="rollback",
    )
    result = compactor.compact(conv)

    print_status(conv, "AFTER compaction")
    print()
    print(result)

    if result.validation_result:
        print("  Validation Details:")
        print_validation(result.validation_result)
    print()


# ── Scenario 2: OpenAIValidator (real LLM judge) ─────────────────────────────

def scenario_openai_validator() -> None:
    print("=" * 60)
    print("  Scenario 2: OpenAIValidator (LLM-as-judge, requires API key)")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print()
        print("  OPENAI_API_KEY is not set -- skipping this scenario.")
        print("  To run with a live LLM validator:")
        print("    1. Copy .env.example to .env")
        print("    2. Add your OPENAI_API_KEY to .env")
        print("    3. Run python main.py again")
        print()
        return

    conv = build_conversation("OpenAI Validation Demo", max_tokens=500)
    print()
    print_status(conv, "BEFORE compaction")

    try:
        compactor = Compactor(
            keep_recent=3,
            summarizer=OpenAISummarizer(),
            validator=OpenAIValidator(),
            on_validation_fail="rollback",
        )
        result = compactor.compact(conv)

        print_status(conv, "AFTER compaction")
        print()
        print(result)

        if result.validation_result:
            print("  Validation Details:")
            print_validation(result.validation_result)

        if result.committed and result.summary_message:
            print("\n  Summary content (generated by LLM):")
            print("  " + "-" * 48)
            for line in result.summary_message.content.splitlines():
                print(f"    {line}")

    except (SummarizerError, ValidatorError) as exc:
        print(f"  Error: {exc}")
    print()


# ── Scenario 3: Rollback on validation failure ───────────────────────────────

def scenario_rollback_on_bad_summary() -> None:
    print("=" * 60)
    print("  Scenario 3: Validation failure -- rollback keeps original context")
    print("=" * 60)
    print()
    print("  Simulating a deliberately poor summary + strict validator:")

    class EmptySummarizer(Summarizer):
        """Produces a nearly empty summary (simulates a bad/truncated API response)."""
        def summarize(self, messages):
            return "The conversation discussed some technical topics."

    conv = build_conversation("Rollback Demo", max_tokens=500)
    print()
    print_status(conv, "BEFORE compaction")
    before_count  = conv.message_count()
    before_tokens = conv.total_tokens()

    compactor = Compactor(
        keep_recent=3,
        summarizer=EmptySummarizer(),
        validator=HeuristicValidator(
            fail_threshold=0.3,   # very strict: fail if >30% of key terms are missing
            warn_threshold=0.1,
            min_candidates=1,
        ),
        on_validation_fail="rollback",
    )
    result = compactor.compact(conv)

    print_status(conv, "AFTER attempted compaction")
    print()
    print(result)

    if result.validation_result:
        print("  Validation Details:")
        print_validation(result.validation_result)

    # Verify conversation is unchanged
    assert conv.message_count() == before_count, "Bug: conversation was modified despite rollback!"
    assert conv.total_tokens()  == before_tokens, "Bug: token count changed despite rollback!"
    print(f"\n  [OK] Conversation preserved: {conv.message_count()} messages, "
          f"{conv.total_tokens()} tokens -- unchanged.")
    print()


# ── Scenario 4: on_validation_fail='warn' ────────────────────────────────────

def scenario_warn_mode() -> None:
    print("=" * 60)
    print("  Scenario 4: on_validation_fail='warn' (applies despite failure)")
    print("=" * 60)
    print()

    class EmptySummarizer(Summarizer):
        def summarize(self, messages):
            return "Brief summary."

    conv = build_conversation("Warn Mode Demo", max_tokens=500)
    print()
    print_status(conv, "BEFORE compaction")

    compactor = Compactor(
        keep_recent=3,
        summarizer=EmptySummarizer(),
        validator=HeuristicValidator(fail_threshold=0.3, min_candidates=1),
        on_validation_fail="warn",       # apply compaction even if validation fails
    )
    result = compactor.compact(conv)

    print_status(conv, "AFTER compaction")
    print()
    print(result)

    if result.validation_result:
        print("  Validation Details:")
        print_validation(result.validation_result)

    # Conversation WAS modified even though validation failed
    committed_str = "applied (warn mode)" if result.committed else "rolled back"
    print(f"\n  Result: compaction was {committed_str}.")
    print()


# ── Scenario 5: Message Importance & Protected Context (Step 9) ──────────────

def scenario_importance_and_protected_context() -> None:
    print("=" * 60)
    print("  Scenario 5: Message Importance & Protected Context (Step 9)")
    print("=" * 60)
    print()
    print("  Demonstrates how protected & critical messages are kept verbatim,")
    print("  discardable messages are dropped, and normal messages are summarized.")
    print()

    conv = Conversation(name="Protected & Importance Demo", max_tokens=600)
    conv.add(Message(
        role="system",
        content="You are an expert software architect assisting in agent infrastructure.",
    ))
    conv.add(Message(
        role="user",
        content="CRITICAL REQUIREMENT: Data must be encrypted with AES-256 at rest and in transit.",
        protected=True,
    ))
    conv.add(Message(
        role="assistant",
        content="CORE DECISION: We will use PostgreSQL with pgvector for storage.",
        importance=ImportanceLevel.CRITICAL,
    ))
    conv.add(Message(
        role="user",
        content="Debug log: Cluster healthcheck returned HTTP 200 with latency 14ms.",
        importance=ImportanceLevel.DISCARDABLE,
    ))
    conv.add(Message(
        role="assistant",
        content="Debug trace: Background garbage collection completed in 2ms.",
        importance=ImportanceLevel.DISCARDABLE,
    ))
    conv.add(Message(
        role="user",
        content="What library should we use for token counting in Python?",
        importance=ImportanceLevel.NORMAL,
    ))
    conv.add(Message(
        role="assistant",
        content='Use "tiktoken" with the cl100k_base encoding for GPT-4 compatibility.',
        importance=ImportanceLevel.IMPORTANT,
    ))
    conv.add(Message(
        role="user",
        content="How do we package the module for distribution?",
    ))
    conv.add(Message(
        role="assistant",
        content="Use pyproject.toml with flit or setuptools.",
    ))

    print_status(conv, "BEFORE compaction")
    print()

    compactor = Compactor(
        keep_recent=2,
        summarizer=PlaceholderSummarizer(),
        validator=HeuristicValidator(fail_threshold=0.8, warn_threshold=0.3),
        on_validation_fail="rollback",
    )
    result = compactor.compact(conv)

    print_status(conv, "AFTER compaction")
    print()
    print(result)

    print("\n  Active conversation messages after compaction:")
    for i, msg in enumerate(conv.get_messages(), 1):
        prot_str = " [PROTECTED]" if msg.protected else ""
        imp_str = f" [{msg.importance.value.upper()}]" if msg.importance != ImportanceLevel.NORMAL else ""
        print(f"    {i}. [{msg.role}]{prot_str}{imp_str}: {msg.content[:55]}...")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def scenario_priority_scoring() -> None:
    print("=" * 60)
    print("  Scenario 6: Automatic Priority Scoring (Step 10)")
    print("=" * 60)
    print()
    print("  HeuristicPriorityScorer scores every message 0-100 using")
    print("  role + recency + explicit importance + content signals.")
    print("  Protected => 100/critical. High-scoring older messages are")
    print("  auto-preserved; low-scoring ones are auto-discarded.")
    print()

    scorer = HeuristicPriorityScorer()
    messages = [
        Message(
            role="system",
            content="You are an expert software architect assisting in agent infrastructure.",
        ),
        Message(
            role="user",
            content="CRITICAL SECURITY REQUIREMENT: Never store unencrypted passwords or secrets in database.",
        ),
        Message(
            role="assistant",
            content="Understood, all secrets are hashed with argon2.",
        ),
        Message(
            role="user",
            content="ok thanks",
        ),
        Message(
            role="assistant",
            content="Debug trace: healthcheck ping 200 ok latency 5ms.",
        ),
        Message(
            role="user",
            content="Here is the auth module:\n```python\ndef login(u, p):\n    pass\n```",
        ),
        Message(
            role="user",
            content="How do we package the module for distribution?",
        ),
        Message(
            role="assistant",
            content="Use pyproject.toml with flit or setuptools.",
        ),
    ]

    print("  Individual scores (index / total = recency):")
    for i, msg in enumerate(messages):
        ps = scorer.score(msg, index=i, total_messages=len(messages))
        print(f"    {i + 1}. {ps}")
        print(f"        factors={ps.factors} reason={ps.reason}")
    print()

    # Custom weights demo: user outranks system
    custom = HeuristicPriorityScorer(
        role_weights={"user": 40.0, "system": 10.0, "assistant": 5.0, "tool": 0.0}
    )
    ps_user = custom.score(Message(role="user", content="Test message"))
    ps_sys = custom.score(Message(role="system", content="Test message"))
    print("  Custom role_weights demo (user=40 > system=10):")
    print(f"    user={ps_user.score} system={ps_sys.score}")
    print()

    # Compactor integration: vital requirement auto-preserved, pleasantry auto-discarded
    conv = Conversation(name="Priority Scoring Demo", max_tokens=1000)
    for m in messages:
        conv.add(m)

    print_status(conv, "BEFORE compaction")
    print()

    compactor = Compactor(
        keep_recent=2,
        summarizer=PlaceholderSummarizer(),
        validator=HeuristicValidator(fail_threshold=0.8, warn_threshold=0.3),
        on_validation_fail="rollback",
        scorer=scorer,  # default is HeuristicPriorityScorer(); shown explicitly
    )
    result = compactor.compact(conv)

    print_status(conv, "AFTER compaction")
    print()
    print(result)
    print()


def scenario_persistent_storage() -> None:
    print("=" * 60)
    print("  Scenario 7: Persistent Context Storage (Step 12)")
    print("=" * 60)
    print()
    print("  ContextStore saves a compacted conversation to JSON")
    print("  (messages, token counts, importance, protected flags,")
    print("  compaction summary, validation result) and loads it back")
    print("  so the restored conversation behaves exactly like new.")
    print()

    conv = Conversation(name="Persistent Demo", max_tokens=600)
    conv.add(Message(
        role="system",
        content="You are an expert software architect assisting in agent infrastructure.",
    ))
    conv.add(Message(
        role="user",
        content="CRITICAL REQUIREMENT: Data must be encrypted with AES-256 at rest and in transit.",
        protected=True,
    ))
    conv.add(Message(
        role="assistant",
        content="CORE DECISION: We will use PostgreSQL with pgvector for storage.",
        importance=ImportanceLevel.CRITICAL,
    ))
    conv.add(Message(
        role="user",
        content="What library should we use for token counting in Python?",
    ))
    conv.add(Message(
        role="assistant",
        content='Use "tiktoken" with the cl100k_base encoding for GPT-4 compatibility.',
        importance=ImportanceLevel.IMPORTANT,
    ))
    conv.add(Message(
        role="user",
        content="How do we package the module for distribution?",
    ))
    conv.add(Message(
        role="assistant",
        content="Use pyproject.toml with flit or setuptools.",
    ))

    compactor = Compactor(
        keep_recent=2,
        summarizer=PlaceholderSummarizer(),
        validator=HeuristicValidator(fail_threshold=0.8, warn_threshold=0.3),
        on_validation_fail="rollback",
    )
    result = compactor.compact(conv)
    print_status(conv, "AFTER compaction (before save)")
    print()

    store = ContextStore(conv)
    store.record(result)
    path = "demo_context.json"
    try:
        store.save(path)
        print(f"  Saved {conv.message_count()} messages "
              f"({conv.total_tokens()} tokens) to {path!r}.")
    except StoreError as exc:
        print(f"  Save failed: {exc}")
        print()
        return

    try:
        restored = ContextStore.load_file(path)
    except StoreError as exc:
        print(f"  Load failed: {exc}")
        print()
        return

    loaded = restored.conversation
    print(f"  Loaded {loaded.message_count()} messages "
          f"({loaded.total_tokens()} tokens) from {path!r}.")
    assert loaded.message_count() == conv.message_count()
    assert loaded.total_tokens() == conv.total_tokens()
    assert loaded.get_status() == conv.get_status()
    if restored.last_validation is not None:
        print(f"  Restored validation: "
              f"{'PASSED' if restored.last_validation.passed else 'FAILED'} "
              f"via {restored.last_validation.validator_used}.")
    if restored.last_compaction is not None:
        print(f"  Restored compaction: {restored.last_compaction.messages_removed} "
              f"removed, summarizer={restored.last_compaction.summarizer_used}.")

    # The restored conversation behaves like new: keep chatting.
    loaded.add(Message(role="user", content="What should we build next?"))
    print(f"  After one follow-up turn: {loaded.message_count()} messages, "
          f"{loaded.total_tokens()} tokens, status {loaded.get_status()}.")
    print()


def scenario_relevant_retrieval() -> None:
    print("=" * 60)
    print("  Scenario 8: Relevant Context Retrieval (Step 13)")
    print("=" * 60)
    print()
    print("  KeywordRetriever ranks persisted messages (including")
    print("  compaction summaries) by keyword relevance. Deterministic,")
    print("  dependency-free: no embeddings, no vector database.")
    print()

    conv = Conversation(name="Retrieval Demo", max_tokens=1000)
    conv.add(Message(
        role="system",
        content="You are an expert software architect assisting in agent infrastructure.",
    ))
    conv.add(Message(
        role="assistant",
        content="CORE DECISION: Architecture uses PostgreSQL 16 with pgvector for storage.",
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
        role="system",
        content="[CONTEXT SUMMARY -- 2 older message(s) compacted] Earlier: PostgreSQL storage decided.",
        metadata={"type": "compaction_summary"},
    ))

    retriever = KeywordRetriever(top_k=3)

    queries = [
        "PostgreSQL storage decision",
        "GitHub authentication",
        "quantum banana trombone",  # irrelevant: expect no hits
    ]
    for query in queries:
        hits = retriever.retrieve(query, conv)
        print(f'  Query: {query!r} -> {len(hits)} hit(s)')
        for hit in hits:
            print(f"    {hit}")
        if not hits:
            print("    (no relevant context found)")
        print()

    # top_k + Conversation.search integration (same ranking, read-only).
    before = conv.message_count()
    top1 = conv.search("PostgreSQL storage", top_k=1)
    print(f"  conv.search(..., top_k=1) -> {len(top1)} hit(s):")
    for hit in top1:
        print(f"    {hit}")
    assert conv.message_count() == before, "Bug: search modified the conversation!"
    print(f"\n  [OK] Conversation unchanged after search: {conv.message_count()} messages.")
    print()


def scenario_semantic_retrieval() -> None:
    print("=" * 60)
    print("  Scenario 9: Semantic Context Retrieval (Step 14)")
    print("=" * 60)
    print()
    print("  SemanticRetriever ranks by embedding cosine similarity, caches")
    print("  vectors so unchanged messages are embedded once, persists them")
    print("  in the JSON store, and falls back to keyword search on failure.")
    print()

    # Deterministic stand-in provider (offline): paraphrases of the storage
    # message share its direction, so meaning -- not keywords -- wins.
    class DemoEmbeddingProvider(EmbeddingProvider):
        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            vectors = []
            for text in texts:
                lowered = text.lower()
                if any(w in lowered for w in ("persist", "records", "postgresql", "storage")):
                    vectors.append([1.0, 0.0])
                elif "packag" in lowered or "flit" in lowered:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([-1.0, 0.0])
            return vectors

    conv = Conversation(name="Semantic Demo", max_tokens=1000)
    conv.add(Message(
        role="assistant",
        content="CORE DECISION: Architecture uses PostgreSQL 16 with pgvector for storage.",
        importance=ImportanceLevel.CRITICAL,
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

    provider = DemoEmbeddingProvider()
    retriever = SemanticRetriever(provider, top_k=2)

    # A paraphrase query with ZERO keyword overlap -- keyword finds nothing.
    query = "where do we persist records?"
    keyword_hits = KeywordRetriever().retrieve(query, conv)
    semantic_hits = retriever.retrieve(query, conv)
    print(f"  Query: {query!r}")
    print(f"    keyword hits : {len(keyword_hits)}")
    print(f"    semantic hits: {len(semantic_hits)}")
    for hit in semantic_hits:
        print(f"    {hit}")
    print()

    # Caching: the second identical call embeds nothing new.
    calls_after_first = provider.calls
    retriever.retrieve(query, conv)
    print(f"  Provider calls after 1st retrieval: {calls_after_first}, "
          f"after 2nd: {provider.calls} (unchanged messages cached).")
    print(f"  Cache info: {retriever.cache_info()}")
    print()

    # Persisted embeddings: embed once, save, reload, prime a fresh retriever.
    store = ContextStore(conv)
    print(f"  Embedded {store.embed_missing(provider)} message(s) into the store.")
    store.save("demo_semantic.json")
    restored = ContextStore.load_file("demo_semantic.json")
    fresh = SemanticRetriever(DemoEmbeddingProvider())
    print(f"  Primed {fresh.prime_cache(restored)} persisted vector(s) into a fresh retriever.")
    print()

    # Graceful failure: a broken provider falls back to keyword results.
    class BrokenProvider(EmbeddingProvider):
        def embed(self, texts):
            raise EmbeddingError("Simulated API outage.")

    degraded = SemanticRetriever(BrokenProvider()).retrieve("PostgreSQL storage", conv)
    print(f"  Broken provider -> keyword fallback: {len(degraded)} hit(s).")
    print()

    # Live provider configuration (no network call here -- needs a key).
    if not os.getenv("OPENAI_API_KEY", "").strip():
        print("  OPENAI_API_KEY is not set -- skipping the live OpenAI embedding call.")
        print("  Configure OPENAI_API_KEY + OPENAI_EMBEDDING_MODEL, then:")
        print("    provider = OpenAIEmbeddingProvider()  # reads env vars")
        print("    conv.search('...', provider=provider)")
    else:
        live = OpenAIEmbeddingProvider()
        print(f"  Live provider ready: model={live.model} base_url={live.base_url}.")
    print()


def scenario_vector_store() -> None:
    print("=" * 60)
    print("  Scenario 10: Vector Database Support (Step 15)")
    print("=" * 60)
    print()
    print("  VectorStoreRetriever indexes messages in local ChromaDB and")
    print("  queries the index instead of scanning. Add / update / delete")
    print("  stay in sync, and keyword + in-memory retrieval cover failures.")
    print()

    try:
        store = ChromaVectorStore(collection_name="demo")
    except VectorStoreError as exc:
        print(f"  ChromaDB unavailable -- skipping vector-store demo: {exc}")
        print("  Run: pip install chromadb")
        print()
        return

    class DemoEmbeddingProvider(EmbeddingProvider):
        def embed(self, texts):
            vectors = []
            for text in texts:
                lowered = text.lower()
                if any(w in lowered for w in ("persist", "records", "postgresql", "storage")):
                    vectors.append([1.0, 0.0])
                elif "packag" in lowered or "flit" in lowered:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([-1.0, 0.0])
            return vectors

    conv = Conversation(name="Vector Demo", max_tokens=1000)
    conv.add(Message(
        role="assistant",
        content="CORE DECISION: Architecture uses PostgreSQL 16 with pgvector for storage.",
        importance=ImportanceLevel.CRITICAL,
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

    provider = DemoEmbeddingProvider()
    retriever = VectorStoreRetriever(provider, store, top_k=2)
    print(f"  Indexed {retriever.index(conv)} message(s); store holds {store.count()} records.")

    query = "where do we persist records?"
    hits = retriever.retrieve(query, conv)
    print(f"  Query: {query!r} -> {len(hits)} hit(s) from the index:")
    for hit in hits:
        print(f"    {hit}")
    print()

    # Update = upsert the revised content, then drop the stale record.
    import hashlib
    old_text = conv.get_messages()[1].content
    old_id = hashlib.sha256(old_text.encode("utf-8")).hexdigest()
    conv.get_messages()[1].content = "Use Hatch with hatch-vcs for packaging releases."
    retriever.index(conv)          # embeds only the new text (rest cached/known)
    retriever.remove([old_id])     # drop the stale pre-edit record
    print(f"  After update, store holds {store.count()} records.")
    updated = retriever.retrieve("hatch packaging releases", conv)
    print(f"  'hatch packaging releases' -> {len(updated)} hit(s):")
    for hit in updated:
        print(f"    {hit}")
    print()

    # Delete: removing the debug note drops it from the index.
    debug_text = "Debug: Pod healthcheck HTTP 200 latency 5ms."
    retriever.remove([hashlib.sha256(debug_text.encode("utf-8")).hexdigest()])
    print(f"  After delete, store holds {store.count()} records.")
    print()

    # Fallback chain: when the index is down, retrieval degrades to
    # in-memory semantic scoring, then to keyword matching.
    broken_store = ChromaVectorStore(collection_name="demo-broken")

    def _failing_query(*args, **kwargs):
        raise VectorStoreError("Simulated index outage.")

    broken_store.query = _failing_query  # type: ignore[method-assign]
    degraded = VectorStoreRetriever(provider, broken_store).retrieve(query, conv)
    print(f"  Broken index -> fallback chain: {len(degraded)} hit(s).")
    print("\n  [OK] Vector store queried instead of scanning; fallbacks intact.")
    print()


def scenario_hybrid_assembly() -> None:
    print("=" * 60)
    print("  Scenario 11: Hybrid Retrieval & Assembly (Step 16)")
    print("=" * 60)
    print()
    print("  HybridRetriever fuses keyword + semantic scores with weighted")
    print("  ranking; ContextAssembler packs system, protected, retrieved,")
    print("  recent, and the request into a token budget.")
    print()

    class DemoEmbeddingProvider(EmbeddingProvider):
        def embed(self, texts):
            vectors = []
            for text in texts:
                lowered = text.lower()
                if any(w in lowered for w in ("persist", "records", "postgresql", "storage")):
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

    conv = Conversation(name="Hybrid Demo", max_tokens=8000)
    conv.add(Message(
        role="system",
        content="You are an expert software architect.",
    ))
    conv.add(Message(
        role="user",
        content="Never store secrets in plain text.",
        protected=True,
    ))
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
        content="Recent question about deploy?",
    ))
    conv.add(Message(
        role="assistant",
        content="Recent answer about deploy steps.",
    ))

    provider = DemoEmbeddingProvider()
    query = "where do we persist records?"
    hybrid = HybridRetriever(
        semantic=SemanticRetriever(provider),
        keyword_weight=0.4,
        semantic_weight=0.6,
        top_k=3,
    )
    hits = hybrid.retrieve(query, conv)
    print(f"  Hybrid query: {query!r} -> {len(hits)} hit(s):")
    for hit in hits:
        print(f"    {hit}")
    print()

    # Generous budget: everything fits.
    full = ContextAssembler(max_tokens=8000, keep_recent=2).assemble(query, conv, hits)
    print("  Assembled with a generous budget:")
    print(full)
    print()

    # Tight budget: low-priority content is excluded with reasons,
    # while system / protected / request are always kept (even over budget).
    tight = ContextAssembler(max_tokens=27, keep_recent=2).assemble(query, conv, hits)
    print("  Assembled with a tight (27-token) budget:")
    print(tight)
    print(f"\n  [OK] {len(tight.messages)} messages, "
          f"{tight.total_tokens}/{tight.max_tokens} tokens; "
          f"{len(tight.excluded)} excluded with reasons; mandatory context kept.")
    print()


def scenario_compaction_pipeline() -> None:
    print("=" * 60)
    print("  Scenario 12: Compaction Pipeline (Step 17)")
    print("=" * 60)
    print()
    print("  ContextPipeline monitors usage, compacts only past the trigger")
    print("  threshold, validates before commit, retrieves relevant context,")
    print("  and persists the updated conversation.")
    print()

    compactor = Compactor(
        keep_recent=2,
        summarizer=PlaceholderSummarizer(),
        validator=HeuristicValidator(fail_threshold=0.8, warn_threshold=0.3),
        on_validation_fail="rollback",
    )

    # Case 1: healthy conversation -- pipeline observes and stands down.
    calm = Conversation(name="Calm Session", max_tokens=8000)
    calm.add(Message(role="system", content="You are helpful."))
    calm.add(Message(role="user", content="Hello there."))
    skipped = ContextPipeline(compactor=compactor).run(calm)
    print(f"  Healthy conversation: triggered={skipped.triggered} "
          f"({skipped.reason})")
    print()

    # Case 2: pressured conversation -- full run with retrieval + persistence.
    busy = Conversation(name="Busy Session", max_tokens=80)
    busy.add(Message(role="system", content="You are an expert software architect."))
    busy.add(Message(
        role="user",
        content="Never store secrets in plain text.",
        protected=True,
    ))
    for i in range(8):
        role = "user" if i % 2 == 0 else "assistant"
        busy.add(Message(role=role, content="Design discussion turn %d about architecture." % i))

    pipeline = ContextPipeline(
        compactor=compactor,
        retriever=KeywordRetriever(),
        persist_path="demo_pipeline.json",
    )
    result = pipeline.run(busy)
    print(result)
    print(f"  Retrieved {len(result.retrieved)} relevant message(s); "
          f"protected={result.protected_count}, critical={result.critical_count}.")
    if result.persisted_path:
        restored = ContextStore.load_file(result.persisted_path)
        print(f"  Reloaded persisted context: {restored.conversation.message_count()} "
              f"messages, {restored.conversation.total_tokens()} tokens.")
    print(f"\n  [OK] Pipeline complete: {result.tokens_before} -> "
          f"{result.tokens_after} tokens (ratio {result.compression_ratio:.3f}).")
    print()


def scenario_llm_adapter() -> None:
    print("=" * 60)
    print("  Scenario 13: LLM Context Adapter (Step 18)")
    print("=" * 60)
    print()
    print("  LLMClient sends assembled ContextFlow context + system")
    print("  instructions + the user request to the model, and returns")
    print("  the reply with token usage, latency, and model name.")
    print("  This demo uses a mocked client -- no real API calls.")
    print()

    from unittest.mock import MagicMock

    conv = Conversation(name="Adapter Demo", max_tokens=8000)
    conv.add(Message(role="system", content="You are a concise assistant."))
    conv.add(Message(
        role="user",
        content="Never store secrets in plain text.",
        protected=True,
    ))
    conv.add(Message(
        role="assistant",
        content="PostgreSQL storage decision for persistence.",
    ))

    assembler = ContextAssembler(max_tokens=2000, keep_recent=5)
    ctx = assembler.assemble("where do we persist records?", conv, [])

    # Mocked OpenAI-compatible client: fixed reply + usage, zero network.
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Persist records in PostgreSQL 16 with pgvector."
    mock_choice.finish_reason = "stop"
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 64
    mock_usage.completion_tokens = 12
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = "gpt-4o-mini"
    mock_client.chat.completions.create.return_value = mock_response

    client: LLMClient = OpenAILLMClient(api_key="demo-key", client=mock_client)
    response = client.complete(
        context=ctx,
        system="Answer in one sentence.",
        request="where do we persist records?",
    )
    print(response)
    sent = mock_client.chat.completions.create.call_args[1]["messages"]
    print(f"  Sent {len(sent)} chat message(s): "
          f"{[m['role'] for m in sent]} (system instructions first).")
    print()

    # Error handling: a failing backend surfaces as LLMError, cause chained.
    failing = MagicMock()
    failing.chat.completions.create.side_effect = TimeoutError("timed out")
    try:
        OpenAILLMClient(api_key="demo-key", client=failing).complete(request="Hi?")
    except LLMError as exc:
        print(f"  Failing backend -> LLMError: {exc}")
    print()

    # Live configuration (constructed, never called without consent).
    if not os.getenv("OPENAI_API_KEY", "").strip():
        print("  OPENAI_API_KEY is not set -- skipping the live call.")
        print("  Configure OPENAI_API_KEY (+ OPENAI_MODEL / OPENAI_TIMEOUT), then:")
        print("    client = OpenAILLMClient()  # reads env vars")
        print("    client.complete(context=ctx, system='...', request='...')")
    else:
        live = OpenAILLMClient()
        print(f"  Live client ready: {live.config} (call .complete() to use).")
    print("\n  [OK] Adapter complete: reply + usage + latency, provider-independent.")
    print()


def scenario_simple_agent() -> None:
    print("=" * 60)
    print("  Scenario 14: Simple Agent (Step 19)")
    print("=" * 60)
    print()
    print("  Agent.run() retrieves relevant context, assembles the prompt")
    print("  within budget, calls the model, records history, and compacts")
    print("  when token pressure requires it. Mocked model -- no real calls.")
    print()

    from contextflow.llm import LLMResponse

    class DemoModel(LLMClient):
        """Deterministic stand-in model: acknowledges the latest request."""
        def complete(self, context=None, system=None, request=None,
                     temperature=None, max_tokens=None):
            if isinstance(context, AssembledContext):
                turns = list(context.messages)
            elif context is None:
                turns = []
            else:
                turns = list(context)
            if request is not None:
                text = request.content if isinstance(request, Message) else str(request)
            else:
                # The agent sends the request as the last assembled message.
                text = next((m.content for m in reversed(turns)
                             if m.role == "user"), "...")
            reply = "Acknowledged: %s" % text[:60]
            return LLMResponse(content=reply, model="demo-model",
                               input_tokens=40, output_tokens=8,
                               latency_seconds=0.01)

    conv = Conversation(name="Agent Demo", max_tokens=70)
    conv.add(Message(role="system", content="You are a concise assistant."))
    conv.add(Message(
        role="user",
        content="Never store secrets in plain text.",
        protected=True,
    ))
    for i in range(4):
        conv.add(Message(role="user", content="Design discussion turn %d here." % i))

    agent = Agent(llm=DemoModel(), conversation=conv, system="Be concise.")
    first = agent.run("What about the design discussion next?")
    print(first)
    print(f"  History now holds {len(agent.history)} message(s); "
          f"compacted={first.compaction_occurred}.")
    print()

    second = agent.run("And after that?")
    print(second)
    print(f"  History now holds {len(agent.history)} message(s); "
          f"compacted={second.compaction_occurred}.")
    print("  Final history:")
    for i, msg in enumerate(agent.history, 1):
        print(f"    {i}. [{msg.role}]: {msg.content[:55]}...")
    print("\n  [OK] Agent loop complete: retrieve, assemble, answer, record, compact.")
    print()


def scenario_tool_calling() -> None:
    print("=" * 60)
    print("  Scenario 15: Basic Tool Calling (Step 20)")
    print("=" * 60)
    print()
    print("  The agent executes read_file / list_directory from ```tool")
    print("  blocks, feeds observations back to the model, and records")
    print("  token-counted tool messages. Mocked model -- no real calls.")
    print("  No shell execution exists in this step.")
    print()

    import tempfile
    from contextflow.llm import LLMResponse

    workspace = tempfile.mkdtemp(prefix="contextflow-demo-")
    with open(os.path.join(workspace, "notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("Deploy on Fridays is forbidden.")
    with open(os.path.join(workspace, "todo.txt"), "w", encoding="utf-8") as fh:
        fh.write("Buy milk.")

    tool_block = (
        "Let me check the workspace first.\n"
        "```tool\n"
        '{"name": "list_directory", "arguments": {"path": "."}}\n'
        "```\n"
        "```tool\n"
        '{"name": "read_file", "arguments": {"path": "notes.txt"}}\n'
        "```"
    )

    class DemoModel(LLMClient):
        """Scripted stand-in: requests tools once, then answers."""
        def __init__(self):
            self.calls = 0

        def complete(self, context=None, system=None, request=None,
                     temperature=None, max_tokens=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content=tool_block, model="demo-model",
                                   input_tokens=40, output_tokens=20,
                                   latency_seconds=0.01)
            return LLMResponse(
                content="Per notes.txt: never deploy on Fridays.",
                model="demo-model", input_tokens=60, output_tokens=10,
                latency_seconds=0.01)

    conv = Conversation(name="Tool Demo", max_tokens=8000)
    agent = Agent(
        llm=DemoModel(),
        conversation=conv,
        system="Use tools when files are mentioned.",
        tools=[ReadFileTool(workspace), ListDirectoryTool(workspace)],
    )
    result = agent.run("What do the workspace files say?")
    print(result)
    for call in result.tool_calls:
        status = "ok" if call.result.ok else "FAILED"
        print(f"    tool {call.name} {call.arguments} -> {status}")
    print("  History roles:", [m.role for m in agent.history])
    print("  Tool observations are token-counted Messages: "
          f"{sum(m.token_count for m in agent.history if m.role == 'tool')} token(s).")
    print("\n  [OK] Tool loop complete: execute, observe, answer, record.")
    print()


def scenario_result_compaction() -> None:
    print("=" * 60)
    print("  Scenario 16: Tool-Result Compaction (Step 21)")
    print("=" * 60)
    print()
    print("  ToolResultProcessor shrinks large outputs (text / listings /")
    print("  JSON) for active context while archiving originals. Errors")
    print("  always pass through untouched. Mocked model -- no real calls.")
    print()

    import tempfile
    from contextflow.llm import LLMResponse

    workspace = tempfile.mkdtemp(prefix="contextflow-results-")
    log_lines = ["Starting nightly batch."]
    log_lines += ["heartbeat ok"] * 10
    log_lines += ["Processing /var/data/input_014.csv ..."]
    log_lines += ["detail line %d: all nominal" % i for i in range(80)]
    log_lines += ["ERROR: connection refused on db-primary:5432"]
    log_lines += ["trailing filler %d" % i for i in range(15)]
    log_lines += ["Run finished."]
    with open(os.path.join(workspace, "batch.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines))
    for i in range(60):
        with open(os.path.join(workspace, "out_%02d.txt" % i), "w", encoding="utf-8") as fh:
            fh.write("x")

    # Standalone: reduce a large log without any agent involved.
    processor = ToolResultProcessor()
    demo_result = ReadFileTool(workspace).run(path="batch.log")
    demo = processor.process(demo_result)
    print(f"  Standalone: {demo}")
    print(f"    head kept : {'Starting nightly batch.' in demo.reduced}")
    print(f"    error kept: {'ERROR: connection refused' in demo.reduced}")
    print(f"    path kept : {'/var/data/input_014.csv' in demo.reduced}")
    print(f"    archive   : lookup returns {demo.saved_tokens} saved tokens")
    print()

    # Integrated: the agent records reduced observations; originals persist.
    tool_block = (
        "Let me inspect the workspace.\n"
        "```tool\n"
        '{"name": "list_directory", "arguments": {"path": "."}}\n'
        "```\n"
        "```tool\n"
        '{"name": "read_file", "arguments": {"path": "batch.log"}}\n'
        "```"
    )

    class DemoModel(LLMClient):
        def __init__(self):
            self.calls = 0

        def complete(self, context=None, system=None, request=None,
                     temperature=None, max_tokens=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content=tool_block, model="demo-model",
                                   input_tokens=40, output_tokens=20,
                                   latency_seconds=0.01)
            return LLMResponse(content="Batch failed on db-primary; see log.",
                               model="demo-model", input_tokens=60,
                               output_tokens=10, latency_seconds=0.01)

    conv = Conversation(name="Result Demo", max_tokens=8000)
    agent = Agent(
        llm=DemoModel(),
        conversation=conv,
        tools=[ReadFileTool(workspace), ListDirectoryTool(workspace)],
    )
    result = agent.run("Did the batch succeed?")
    print(result)
    for msg in agent.history:
        if msg.role != "tool":
            continue
        meta = msg.metadata
        print(f"    tool '{meta['tool']}': {msg.token_count} tokens in context, "
              f"reduced={meta['reduced']}, "
              f"original={meta.get('original_tokens', msg.token_count)} tokens archived")
    listing_msg = next(m for m in agent.history
                       if m.metadata.get("tool") == "list_directory")
    print(f"  Listing kept directories/files head+tail, "
          f"omitted marker present: {'omitted' in listing_msg.content}")
    print("\n  [OK] Large outputs reduced; errors and originals preserved.")
    print()


def main() -> None:
    scenario_heuristic_validator()
    scenario_openai_validator()
    scenario_rollback_on_bad_summary()
    scenario_warn_mode()
    scenario_importance_and_protected_context()
    scenario_priority_scoring()
    scenario_persistent_storage()
    scenario_relevant_retrieval()
    scenario_semantic_retrieval()
    scenario_vector_store()
    scenario_hybrid_assembly()
    scenario_compaction_pipeline()
    scenario_llm_adapter()
    scenario_simple_agent()
    scenario_tool_calling()
    scenario_result_compaction()


if __name__ == "__main__":
    main()
