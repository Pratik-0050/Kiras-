# ContextFlow

Intelligent context management for LLMs: token budgets, compaction, retrieval, agents.

## 1. What ContextFlow Is

ContextFlow is a Python library that manages the conversation context you send
to a large language model. It tracks token consumption across multi-turn
conversations, detects context pressure, compacts history with validated
summaries, retrieves relevant past context, persists sessions, and runs a
simple agent loop — all offline-capable, with live LLM calls strictly optional.

```python
from contextflow import ContextManager

cf = ContextManager()

cf.add_message("user", "Hello")
cf.add_message("assistant", "Hi! How can I help?")

context = cf.get_context()
print(context)
```

## 2. Why It Exists

LLM context windows are finite and expensive. Long conversations either
overflow the window, silently drop early turns, or burn tokens on irrelevant
history. ContextFlow solves this by:

- **Budgeting** every turn with exact `tiktoken` counts and pressure levels
  (`OK` / `WARNING` / `COMPACTION_NEEDED`)
- **Compacting** older turns into validated summaries — rolling back instead
  of committing anything unfaithful, and never touching protected content
- **Retrieving** only the relevant past (keyword, semantic, hybrid, or vector
  search) instead of replaying everything
- **Assembling** prompt-ready context (system + protected + retrieved +
  recent + request) that fits a declared token budget, with exclusions reported

## 3. Main Features

- Token counting, conversations, and pressure detection (offline)
- Compaction with protected/critical preservation and validation rollback
- Automatic priority scoring of every message (0–100, offline heuristics)
- JSON persistence with round-trip fidelity (`ContextStore`)
- Keyword, semantic (embeddings), hybrid, and ChromaDB vector retrieval
- Budget-aware context assembly with duplicate removal
- One-pass automatic compaction pipeline with detailed reports
- Provider-independent LLM adapter (`LLMClient` + OpenAI-compatible client)
- Simple agent loop with safe tool calling (`read_file`, `list_directory`)
- Intelligent tool-result compaction (large outputs shrunk, originals archived)
- `ContextManager` facade for everyday use; full engine underneath

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Facade: ContextManager  (add_message/get_context/compact/…) │
├─────────────────────────────────────────────────────────────┤
│ Agent  (retrieve → assemble → complete → tools → record → …) │
├──────────────┬──────────────────────┬────────────────────────┤
│ Compaction   │ Retrieval            │ Persistence            │
│ Compactor    │ KeywordRetriever     │ ContextStore (JSON)    │
│ Summarizer   │ SemanticRetriever    │ ChromaVectorStore      │
│ Validator    │ HybridRetriever      │                        │
│ PriorityScorer│ VectorStoreRetriever│                        │
├──────────────┴──────────────────────┴────────────────────────┤
│ Assembly: ContextAssembler → AssembledContext (budgeted)     │
│ Pipeline: ContextPipeline → PipelineResult (gated runs)     │
│ Model I/O: LLMClient → LLMResponse (provider-independent)   │
├─────────────────────────────────────────────────────────────┤
│ Core: Message · Conversation · ContextStatus (tiktoken)     │
└─────────────────────────────────────────────────────────────┘
```

Layering rules: the facade and `Agent` compose engine pieces but the engine
never imports them back (no cycles). Provider SDKs (`openai`, `chromadb`)
are imported lazily — the core installs and runs without them.

### Extending ContextFlow

Every strategy is an interface with an offline default. Subclass and inject:

```python
from contextflow import Message, ImportanceLevel
from contextflow.scorers import PriorityScorer, PriorityScore

class LengthScorer(PriorityScorer):
    def score(self, message, index=0, total_messages=1):
        if len(message.content) > 500:
            return PriorityScore(score=85.0, classification=ImportanceLevel.CRITICAL,
                                 message=message)
        if len(message.content) < 10:
            return PriorityScore(score=10.0, classification=ImportanceLevel.DISCARDABLE,
                                 message=message)
        return PriorityScore(score=50.0, classification=ImportanceLevel.NORMAL,
                             message=message)
```

The same pattern holds for `Summarizer`, `Validator`, `Retriever`,
`EmbeddingProvider`, `VectorStore`, `LLMClient`, and `Tool`.

## 5. Installation

Requires **Python 3.10+**.

```powershell
# From the built wheel (see dist/)
pip install dist/contextflow-1.0.0-py3-none-any.whl

# Or from source (editable development install)
pip install -e ".[dev]"
```

Optional extras (the core works without them; missing backends raise clear
errors naming the extra to install):

| Extra | Installs | Enables |
|---|---|---|
| *(base)* | `tiktoken`, `python-dotenv` | Everything except live models and ChromaDB |
| `contextflow[llm]` | `openai` | `OpenAISummarizer`, `OpenAIValidator`, `OpenAIEmbeddingProvider`, `OpenAILLMClient` |
| `contextflow[vector]` | `chromadb` | `ChromaVectorStore` |
| `contextflow[dev]` | `pytest`, `ruff`, `build` | Test suite, lint, packaging |
| `contextflow[all]` | `openai` + `chromadb` | All backends |

> The package is not on PyPI yet — install from `dist/` or source (see
> “Exact next steps” from the maintainer before publishing).

## 6. Quick Start

```python
from contextflow import ContextManager

cf = ContextManager(max_tokens=8000, system="You are a concise assistant.")

cf.add_message("user", "What is blockchain?")
cf.add_message("assistant", "Blockchain is a distributed ledger shared across nodes.")

print(cf.tokens, cf.conversation.get_status())
print(cf.get_context("smart contracts"))  # retrieval + budget-aware assembly

result = cf.compact()  # no-op unless pressure requires it
print(result.triggered, result.tokens_before, "->", result.tokens_after)

cf.save("session.json")
restored = ContextManager.load("session.json")
```

## 7. Basic Usage

`ContextManager` methods:

| Method | Purpose |
|---|---|
| `add_message(role, content, importance=..., protected=..., metadata=...)` | Append a validated turn; returns the `Message` |
| `get_context(query=None, top_k=None)` | Retrieve + assemble an `AssembledContext` within budget (never mutates history) |
| `compact()` | Threshold-gated compaction via `ContextPipeline`; returns `PipelineResult` |
| `search(query, top_k=None)` | Ranked `RetrievalResult` hits, best first |
| `save(path)` / `ContextManager.load(path, **kwargs)` | JSON persistence round-trip |
| `history`, `message_count`, `tokens`, `clear()` | History inspection and reset |

Need finer control? Use the engine directly — `Conversation`, `Message`,
`Compactor`, `Conversation.search()`, and friends are all public:

```python
from contextflow import Conversation, Message, ImportanceLevel

conv = Conversation(name="Agent Session", max_tokens=8000)
conv.add(Message(role="user", content="Security rule: encrypt everything.",
                 protected=True))  # never summarized, never removed
conv.add(Message(role="assistant", content="Acknowledged.",
                 importance=ImportanceLevel.CRITICAL))
print(conv.total_tokens(), conv.remaining_tokens(), conv.usage_percentage())
```

## 8. Context Compaction

```python
from contextflow import Compactor, PlaceholderSummarizer, HeuristicValidator

compactor = Compactor(
    keep_recent=5,                          # newest turns stay verbatim
    summarizer=PlaceholderSummarizer(),     # offline; swap OpenAISummarizer() live
    validator=HeuristicValidator(),         # safety check before committing
    on_validation_fail="rollback",          # or "warn" to commit anyway
)
result = compactor.compact(conv)
print(result.tokens_saved, result.validation_result.passed)
```

Rules that always hold: the leading system prompt, `protected=True`, and
`critical` messages are preserved verbatim; `discardable` messages are
dropped without spending summarizer calls; failed validation rolls back and
leaves the conversation untouched. For hands-off operation use
`ContextPipeline`, which monitors usage, compacts only past the trigger
threshold, retrieves relevant context, validates, and persists.

Threshold guidance: `HeuristicValidator` matches verbatim, so its strict
default suits extractive output; abstractive LLM summaries need a tolerant
threshold (`fail_threshold=0.8`) or the `OpenAIValidator` LLM judge.

## 9. Retrieval

```python
hits = conv.search("postgresql storage", top_k=3)          # keyword, offline
hits = conv.search("where is data kept?", provider=provider)  # semantic
retriever = HybridRetriever(semantic=SemanticRetriever(provider),
                            keyword_weight=0.4, semantic_weight=0.6)
store = ChromaVectorStore()                                # needs contextflow[vector]
indexed = VectorStoreRetriever(provider, store)
indexed.index(conv)
```

Backends, weakest to strongest: `KeywordRetriever` (deterministic, offline) →
`SemanticRetriever` (embeddings + cache + keyword fallback) →
`HybridRetriever` (min-max normalized weighted fusion) →
`VectorStoreRetriever` (persistent ChromaDB index, same fallbacks).

## 10. Agents and Tools

```python
from contextflow import Agent, ReadFileTool

agent = Agent(llm=client, conversation=conv, system="You are concise.",
              tools=[ReadFileTool("docs")], max_tool_rounds=3)
turn = agent.run("summarize notes.txt")
print(turn.response, turn.total_tokens, turn.compaction_occurred)
print(turn.tool_calls)  # [ToolCall(name, arguments, result), ...]
```

One `run()` call performs exactly one deterministic turn: retrieve →
assemble → complete → execute any ```tool fenced-JSON requests → feed
observations back → record history → compact on pressure. Failed turns leave
history untouched. Only safe, path-validated tools ship (`read_file`,
`list_directory`); large outputs are reduced by `ToolResultProcessor` while
originals stay archived. There is no shell execution and no autonomous
planning.

## 11. Configuration

All settings are constructor arguments first; environment variables are a
convenience layer (see `.env.example`). Copy it to start:

```powershell
copy .env.example .env
```

| Variable | Default | Used by |
|---|---|---|
| `OPENAI_API_KEY` | *(required for live calls)* | All OpenAI-backed components |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat, summarization, validation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Semantic retrieval |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint (Groq, Ollama, OpenRouter, Azure) |
| `OPENAI_TIMEOUT` | `60.0` | Chat request timeout (seconds) |
| `CHROMA_COLLECTION` | `contextflow` | Vector-store collection name |
| `CHROMA_PERSIST_DIR` | *(unset = in-memory)* | Vector-store disk persistence |

## 12. Examples

Runnable offline scripts in `examples/` (run from the repo root):

| Script | Demonstrates |
|---|---|
| `examples/basic.py` | `ContextManager` tracking, pressure, assembled context |
| `examples/compaction.py` | Pressure → compaction → protected content kept |
| `examples/retrieval.py` | Keyword search hits, misses, and empty results |
| `examples/agent.py` | Agent turns with a scripted stand-in model |

```powershell
python examples/basic.py
python examples/compaction.py
python examples/retrieval.py
python examples/agent.py
```

The full 16-scenario tour lives in `main.py` (or `contextflow-demo` once
installed).

## 13. Development Setup

```powershell
git clone <repo-url>
cd ContextFlow
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Layout is `src/`-based (`src/contextflow/…`); `tests/`, `examples/`, and
`benchmarks/` sit at the root. `ruff check` is the lint gate (formatting is
intentionally not enforced — the repo keeps a consistent hand style).

## 14. Running Tests

```powershell
pytest                    # full suite: 464 pytest tests
pytest -v                 # verbose
python -m pytest tests/test_compactor.py -q   # one file
python -m unittest discover tests              # legacy runner: 94 tests
```

`tests/test_package.py` additionally covers the installable surface
(imports, `__version__`, facade flows, persistence, adapter, agent) and
proves the package imports without optional dependencies.

## 15. Benchmarks

```powershell
python benchmarks/benchmark.py
```

Offline and deterministic (no network, no keys). Latest measured results:

| Workload | Ratio | Saved | Validation |
|---|---|---|---|
| Compaction, dense simulated-LLM (30 msgs) | 0.42 | 268 tokens | PASS |
| Compaction, verbatim listing (30 msgs) | 1.37 | −170 (expands — offline stand-in) | PASS |
| Compaction, lossy / empty summaries | 1.00 | 0 (rolled back) | FAIL → rollback |
| Tool log reduction | 0.17 | 699 tokens | n/a |
| Tool listing reduction | 0.34 | 499 tokens | n/a |
| Tool JSON reduction | 0.29 | 1068 tokens | n/a |

Protected/critical content survived every run; no failed validation was ever
committed. The script exits non-zero if any invariant breaks.

## 16. Project Structure

```text
ContextFlow/
├── pyproject.toml            # metadata, deps, extras, pytest/ruff config
├── README.md / LICENSE / CHANGELOG.md / COMMANDS.md
├── main.py                   # demo shim (real demo: contextflow.demo)
├── .env.example
├── src/contextflow/
│   ├── __init__.py           # public API + __version__
│   ├── manager.py            # ContextManager facade
│   ├── message.py / conversation.py / status.py
│   ├── compactor.py / pipeline.py
│   ├── retriever.py / hybrid.py / embeddings.py
│   ├── store.py / llm.py / agent.py / demo.py
│   ├── scorers/ / summarizers/ / validators/
│   ├── vectorstores/ / tools/
├── tests/                    # pytest suite + legacy unittest modules
├── examples/                 # basic / compaction / retrieval / agent
└── benchmarks/               # offline compression + token report
```

Internal helpers (leading `_`) are not part of the public API and may
change; everything in `contextflow.__all__` is stable.

## 17. Contributing

1. Open an issue describing the change first for anything beyond a small fix.
2. Keep offline determinism: new features must work without API keys; live
   calls stay behind injectable clients.
3. Add/extend tests — do not weaken existing ones to make them pass.
4. Run the gates before pushing: `pytest`, `python -m unittest discover tests`,
   `ruff check .`, `python benchmarks/benchmark.py`, `python main.py`.
5. Update `CHANGELOG.md` and the README sections your change affects.

## 18. License

MIT — see [LICENSE](LICENSE).
