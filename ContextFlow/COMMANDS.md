# ContextFlow — Commands to Run This Project

All commands run from the `ContextFlow/` folder in Windows PowerShell.
Requires **Python 3.10+**.

---

## 1. Setup (first time only)

```powershell
# Go to the project folder
cd C:\Users\prati\OneDrive\Desktop\Projects\kivas\ContextFlow

# (Optional) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the package with dev tools (replaces the old requirements.txt)
pip install -e ".[dev]"

# Configure environment (needed only for live OpenAI calls)
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Extras for users (instead of the full dev install):

```powershell
pip install -e .                 # base: offline engine only
pip install -e ".[llm]"          # + OpenAI-backed components
pip install -e ".[vector]"       # + ChromaDB vector store
pip install -e ".[all]"          # everything
```

---

## 2. Run the Demonstration (16 scenarios, offline-safe)

```powershell
python main.py
# or, once installed:
contextflow-demo
```

Runs every scenario end to end. Works without an API key
(Scenario 2 and live calls are skipped automatically).

---

## 3. Run the Tests

```powershell
# Full suite (recommended)
python -m pytest tests -q

# Verbose output
python -m pytest tests -v

# One test file only
python -m pytest tests/test_compactor.py -q

# Legacy unittest runner (94 tests)
python -m unittest discover tests
```

---

## 4. Run the Benchmark (compression quality + token savings)

```powershell
python benchmarks/benchmark.py
```

Offline and deterministic (no network, no API keys).
Exits non-zero if any invariant breaks.

---

## 5. Build and Verify the Package

```powershell
# Build wheel + sdist into dist/
python -m build

# Lint everything (ruff format is intentionally not enforced)
ruff check contextflow tests main.py benchmarks
```

---

## 6. Code Quality Check (optional)

```powershell
# Install the linter (already in .[dev])
pip install ruff

# Lint everything
ruff check src tests main.py benchmarks examples
```

---

## Quick Reference

| Goal | Command |
|---|---|
| Install (dev) | `pip install -e ".[dev]"` |
| Install (wheel) | `pip install dist/contextflow-1.0.0-py3-none-any.whl` |
| Demo | `python main.py` or `contextflow-demo` |
| All tests | `python -m pytest tests -q` |
| One test file | `python -m pytest tests/<file>.py -q` |
| Legacy tests | `python -m unittest discover tests` |
| Benchmark | `python benchmarks/benchmark.py` |
| Lint | `ruff check src tests main.py benchmarks examples` |
| Build | `python -m build` |
