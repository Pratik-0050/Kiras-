# Changelog

All notable changes to ContextFlow are documented here. Dates are in UTC.

## [1.0.0] - 2026-09-27

First packaged release. The complete Steps 1–21 engine, unchanged in
behavior, repackaged under `src/` with a clean public API:

- Token counting (`tiktoken`), structured messages, conversations with
  token budgets and pressure detection (`OK` / `WARNING` / `COMPACTION_NEEDED`)
- Context compaction with protected/critical preservation, discardable
  drops, validation with rollback, and automatic priority scoring
- Persistence to JSON (`ContextStore`), keyword/semantic/hybrid retrieval,
  local ChromaDB vector storage, budget-aware context assembly
- Automatic compaction pipeline, provider-independent LLM adapter,
  simple agent loop, safe tool calling (`read_file`, `list_directory`),
  intelligent tool-result compaction
- New facade: `ContextManager` (`add_message` / `get_context` / `compact` /
  `search` / `save` / `load`)
- Packaging: `src` layout, `pyproject.toml` (`contextflow[llm]`,
  `contextflow[vector]`, `contextflow[dev]`, `contextflow[all]`),
  `contextflow-demo` console script, `examples/`, offline `benchmarks/`
- 455 pytest tests + 94 legacy unittest tests, all passing
