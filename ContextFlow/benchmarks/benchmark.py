# benchmarks/benchmark.py
"""Offline compression-quality + token-savings benchmark (Step: hardening).

Deterministic, no network, no API keys. Exercises Compactor with four
summarizer profiles (verbatim listing, dense simulated-LLM, lossy
truncation, total loss) across three conversation sizes, plus
ToolResultProcessor reduction on log/listing/JSON outputs.

Reports per-run: tokens before/after, compression ratio, tokens saved,
messages removed, validation verdict, missing-item count, protected
preservation, and wall time. Exits non-zero if any invariant breaks
(protected content lost, unexpected exception, empty summary committed).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
import time
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contextflow import (
    Compactor,
    Conversation,
    HeuristicValidator,
    ImportanceLevel,
    Message,
    PlaceholderSummarizer,
)
from contextflow.summarizers.base import Summarizer
from contextflow.tools import ToolResult, ToolResultProcessor


# ── corpus ───────────────────────────────────────────────────────────

TOPICS = [
    ('We must use "Python" for the backend service.', "normal"),
    ('The decision was to adopt "PostgreSQL" version 14 for persistence.', "important"),
    ("How should we handle GitHub authentication?", "normal"),
    ('Use a "GitHub App" rather than a personal token.', "important"),
    ("Debug trace: healthcheck ping 200 ok latency 5ms.", "discardable"),
    ("What about context management for the agent?", "normal"),
    ("Context caching must stay under 8000 tokens per session.", "normal"),
    ("ok thanks", "normal"),
]


def build_conversation(size: int) -> Conversation:
    conv = Conversation(name="Bench-%d" % size, max_tokens=100_000)
    conv.add(Message(role="system", content="You are an expert software architect."))
    conv.add(Message(
        role="user",
        content="CRITICAL REQUIREMENT: Data must be encrypted with AES-256.",
        protected=True,
    ))
    conv.add(Message(
        role="assistant",
        content="CORE DECISION: We will use PostgreSQL with pgvector.",
        importance=ImportanceLevel.CRITICAL,
    ))
    for i in range(size):
        text, importance = TOPICS[i % len(TOPICS)]
        role = "user" if i % 2 == 0 else "assistant"
        conv.add(Message(role=role, content="Turn %d: %s" % (i, text),
                         importance=importance))
    return conv


# ── summarizer profiles ──────────────────────────────────────────────

class DenseSummarizer(Summarizer):
    """Simulated good LLM: keeps every quoted term + number, capped length."""

    def summarize(self, messages: List[Message]) -> str:
        seen: List[str] = []
        for msg in messages:
            for term in re.findall(r'"([^"]+)"', msg.content):
                if term not in seen:
                    seen.append(term)
            for num in re.findall(r"\b\d[\w.]*\b", msg.content):
                if num not in seen:
                    seen.append(num)
        gist = "Dense summary of %d messages covering: %s." % (
            len(messages), "; ".join(seen[:30]))
        return gist


class TruncatingSummarizer(Summarizer):
    """Simulated lossy model: keeps only the first 60 chars overall."""

    def summarize(self, messages: List[Message]) -> str:
        return " ".join(m.content for m in messages)[:60]


class EmptySummarizer(Summarizer):
    """Simulated total failure: content-free summary."""

    def summarize(self, messages: List[Message]) -> str:
        return "The conversation discussed some technical topics."


PROFILES = {
    "placeholder": PlaceholderSummarizer(),
    "dense-llm": DenseSummarizer(),
    "truncating": TruncatingSummarizer(),
    "empty": EmptySummarizer(),
}


# ── benchmark ────────────────────────────────────────────────────────

def bench_compaction() -> List[dict]:
    rows = []
    for size in (10, 30, 100):
        for profile_name, summarizer in PROFILES.items():
            conv = build_conversation(size)
            compactor = Compactor(
                keep_recent=5,
                summarizer=summarizer,
                # 0.8 tolerates abstractive paraphrase (verbatim full
                # sentences are not expected in real summaries); the
                # strict 0.5 default suits extractive/verbatim output.
                validator=HeuristicValidator(fail_threshold=0.8, min_candidates=3),
                on_validation_fail="rollback",
            )
            started = time.perf_counter()
            try:
                result = compactor.compact(conv)
                error = ""
            except Exception as exc:  # noqa: BLE001 -- benchmark must report, not crash
                result, error = None, "%s: %s" % (type(exc).__name__, exc)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            contents = [m.content for m in conv.get_messages()]
            protected_ok = (
                "CRITICAL REQUIREMENT: Data must be encrypted with AES-256." in contents
                and "CORE DECISION: We will use PostgreSQL with pgvector." in contents
            )
            if result is None:
                rows.append({"size": size, "profile": profile_name, "error": error,
                             "protected_ok": protected_ok, "ms": round(elapsed_ms, 1)})
                continue
            vr = result.validation_result
            rows.append({
                "size": size,
                "profile": profile_name,
                "msgs": "%d->%d" % (result.original_message_count,
                                    result.compacted_message_count),
                "tokens": "%d->%d" % (result.original_token_count,
                                       result.compacted_token_count),
                "ratio": round(result.compacted_token_count / result.original_token_count, 3)
                         if result.original_token_count else 1.0,
                "saved": result.tokens_saved,
                "removed": result.messages_removed,
                "committed": result.committed,
                "validation": "n/a" if vr is None else ("PASS" if vr.passed else "FAIL"),
                "missing": len(vr.missing_items) if vr else 0,
                "protected_ok": protected_ok,
                "ms": round(elapsed_ms, 1),
            })
    return rows


def bench_tool_processing() -> List[dict]:
    processor = ToolResultProcessor()
    log = "\n".join(
        ["start"]
        + ["heartbeat ok"] * 20
        + ["Processing /var/log/app_%02d.log ..." % i for i in range(5)]
        + ["worker line %d nominal" % i for i in range(120)]
        + ["ERROR: disk nearly full on /dev/sda1"]
        + ["end"]
    )
    listing = "\n".join(
        ["src/", "docs/"]
        + ["module_%03d.py" % i for i in range(150)]
    )
    payload = json.dumps({"events": [{"id": i, "msg": "event number %d" % i}
                                     for i in range(100)]})
    rows = []
    for label, text, tool in (("log", log, "read_file"),
                              ("listing", listing, "list_directory"),
                              ("json", payload, "api")):
        started = time.perf_counter()
        processed = processor.process(ToolResult(tool_name=tool, ok=True, output=text))
        ms = round((time.perf_counter() - started) * 1000.0, 1)
        rows.append({
            "kind": label,
            "strategy": processed.strategy,
            "tokens": "%d->%d" % (processed.original_tokens, processed.reduced_tokens),
            "ratio": round(processed.reduced_tokens / processed.original_tokens, 3),
            "saved": processed.saved_tokens,
            "ms": ms,
        })
    return rows


def main() -> int:
    failures = 0
    print("=" * 100)
    print("COMPACTION BENCHMARK (offline, deterministic)")
    print("=" * 100)
    header = ("%-6s %-12s %-11s %-15s %-7s %-7s %-7s %-9s %-10s %-6s %-13s %7s"
              % ("size", "profile", "msgs", "tokens", "ratio", "saved",
                 "removed", "committed", "validation", "miss", "protected_ok", "ms"))
    print(header)
    print("-" * 100)
    for row in bench_compaction():
        if "error" in row:
            print("ERROR size=%s profile=%s: %s" % (row["size"], row["profile"], row["error"]))
            failures += 1
            continue
        print("%-6d %-12s %-11s %-15s %-7.3f %-7d %-7d %-9s %-10s %-6d %-13s %7.1f" % (
            row["size"], row["profile"], row["msgs"], row["tokens"], row["ratio"],
            row["saved"], row["removed"], row["committed"], row["validation"],
            row["missing"], row["protected_ok"], row["ms"]))
        if not row["protected_ok"]:
            print("  ^^ BUG: protected/critical content lost!")
            failures += 1
        if row["committed"] and row["validation"] == "FAIL":
            print("  ^^ BUG: failed validation was committed!")
            failures += 1

    print()
    print("=" * 100)
    print("TOOL-RESULT PROCESSING BENCHMARK")
    print("=" * 100)
    print("%-8s %-9s %-15s %-7s %-7s %7s" % ("kind", "strategy", "tokens", "ratio", "saved", "ms"))
    print("-" * 100)
    for row in bench_tool_processing():
        print("%-8s %-9s %-15s %-7.3f %-7d %7.1f" % (
            row["kind"], row["strategy"], row["tokens"], row["ratio"],
            row["saved"], row["ms"]))
        if row["ratio"] >= 1.0:
            print("  ^^ BUG: no reduction achieved!")
            failures += 1

    print()
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
