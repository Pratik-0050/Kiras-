# tests/test_package.py
"""Package-level tests: imports, version, public API, and ContextManager.

Covers the installable-library surface: `import contextflow` works,
`__version__` matches pyproject.toml, the facade flows work offline, and
the package imports cleanly WITHOUT optional dependencies (openai,
chromadb) via a subprocess import-blocker.
"""

import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import contextflow
from contextflow import (
    Agent,
    AssembledContext,
    ContextManager,
    Conversation,
    LLMClient,
    LLMResponse,
    Message,
)

ROOT = Path(__file__).resolve().parent.parent


def test_package_imports_and_version():
    assert contextflow.__version__ == "1.0.0"
    assert isinstance(contextflow.__all__, list)
    assert "ContextManager" in contextflow.__all__


def test_version_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == contextflow.__version__


def test_dist_metadata_version_matches():
    # Works once the distribution is installed (editable or wheel).
    try:
        installed = importlib.metadata.version("contextflow")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("contextflow distribution not installed")
    assert installed == contextflow.__version__


def test_public_api_names_importable():
    for name in ("ContextManager", "ContextPipeline", "Compactor", "Retriever",
                 "Agent", "AgentResult", "Conversation", "Message",
                 "KeywordRetriever", "ContextAssembler", "ContextStore",
                 "LLMClient", "VectorStoreRetriever", "HybridRetriever",
                 "ToolRegistry", "ReadFileTool", "ToolResultProcessor"):
        assert getattr(contextflow, name, None) is not None, name


def test_context_manager_basic_flow():
    cf = ContextManager()
    cf.add_message("user", "Hello")
    cf.add_message("assistant", "Hi! How can I help?")
    assert cf.message_count == 2
    assert cf.tokens > 0
    context = cf.get_context()
    assert isinstance(context, AssembledContext)
    assert len(context.messages) == 2
    print(context)


def test_context_manager_search_compact_save_load(tmp_path):
    cf = ContextManager(max_tokens=70, system="You are helpful.")
    cf.add_message("user", "Never store secrets in plain text.", protected=True)
    for i in range(8):
        cf.add_message("user" if i % 2 == 0 else "assistant",
                       "Design discussion turn %d here." % i)
    assert len(cf.search("secrets")) >= 1
    result = cf.compact()
    assert result.triggered is True
    assert result.committed is True

    path = str(tmp_path / "manager.json")
    assert cf.save(path) == path
    loaded = ContextManager.load(path)
    assert loaded.message_count == cf.message_count
    assert loaded.tokens == cf.tokens


def test_context_manager_invalid_args():
    with pytest.raises(TypeError):
        ContextManager(max_tokens="lots")  # type: ignore
    with pytest.raises(ValueError):
        ContextManager(max_tokens=0)
    cf = ContextManager()
    with pytest.raises(TypeError):
        cf.add_message("user", 123)  # type: ignore
    with pytest.raises(ValueError):
        cf.add_message("nobody", "Hi.")


class _FakeModel(LLMClient):
    def complete(self, context=None, system=None, request=None,
                 temperature=None, max_tokens=None):
        return LLMResponse(content="Noted.", model="fake",
                           input_tokens=5, output_tokens=2,
                           latency_seconds=0.01)


def test_llm_adapter_and_agent_flows():
    from contextflow import ContextAssembler, KeywordRetriever
    conv = Conversation(name="Pkg", max_tokens=8000)
    conv.add(Message(role="user", content="Hello there."))
    hits = KeywordRetriever().retrieve("hello", conv)
    ctx = ContextAssembler(max_tokens=8000).assemble("Hello?", conv, hits)
    response = _FakeModel().complete(context=ctx, system="Be concise.",
                                     request="Hello?")
    assert response.content == "Noted."
    assert response.total_tokens == 7

    agent = Agent(llm=_FakeModel(), conversation=conv)
    turn = agent.run("Second question?")
    assert turn.response == "Noted."
    assert turn.compaction_occurred is False
    assert len(agent.history) == 3  # 1 seeded + user + assistant


def test_imports_without_optional_dependencies():
    """Block openai/chromadb imports: the package must still import and run."""
    code = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'openai' or name.startswith('openai.'):\n"
        "            return self\n"
        "        if name == 'chromadb' or name.startswith('chromadb.'):\n"
        "            return self\n"
        "        return None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('blocked: %s' % name)\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import contextflow\n"
        "cf = contextflow.ContextManager()\n"
        "cf.add_message('user', 'Hello')\n"
        "cf.add_message('assistant', 'Hi!')\n"
        "assert cf.message_count == 2\n"
        "assert len(cf.get_context().messages) == 2\n"
        "assert len(cf.search('hello')) >= 1\n"
        "print('optional-deps-free OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(ROOT), timeout=120,
                          env={**os.environ,
                               "PYTHONPATH": str(ROOT / "src")})
    assert proc.returncode == 0, proc.stderr
    assert "optional-deps-free OK" in proc.stdout
