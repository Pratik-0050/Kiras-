# contextflow/store.py
"""
Step 12: Persistent context storage (JSON implementation).

ContextStore saves and loads everything needed to resume a session exactly
where it left off:

  - Conversation settings: name, max_tokens, warn_at, compact_at
  - Messages: role, content, importance, protected, metadata
  - Token counts: stored for audit, recomputed on load via tiktoken
  - Compaction summaries: summary Message objects (metadata type
    ``compaction_summary``) are stored like any other message, and the most
    recent CompactionResult metrics are stored under ``last_compaction``
  - Validation results: the most recent ValidationResult under
    ``last_validation`` (also nested inside ``last_compaction`` when present)

Only JSON is used in this step -- no database, no embeddings, no RAG, no
agent loops. Files that are missing or corrupted raise StoreError with a
clear message; the raw FileNotFoundError / JSONDecodeError is never leaked
bare to callers (it is chained for debugging).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .conversation import Conversation
from .compactor import CompactionResult
from .embeddings import _content_key, validate_vector, EmbeddingError
from .message import ImportanceLevel, Message
from .scorers.base import PriorityScore
from .validators.base import ValidationResult

PathLike = Union[str, os.PathLike]

FORMAT_VERSION = 1
STORE_KIND = "contextflow/context-store"


class StoreError(Exception):
    """Raised when a context file cannot be saved or loaded.

    Covers: missing files, invalid JSON, schema violations (wrong types,
    missing keys, bad roles / importance levels), and OS write failures.
    A validation *failure* (ValidationResult.passed=False) is NOT a
    StoreError -- it is normal persisted data.
    """


# ── message helpers ──────────────────────────────────────────────────


def _message_to_dict(
    msg: Message,
    embeddings: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    payload = {
        "role": msg.role,
        "content": msg.content,
        "token_count": msg.token_count,
        "importance": msg.importance.value,
        "protected": msg.protected,
        "metadata": msg.metadata,
    }
    if embeddings is not None:
        payload["embedding"] = embeddings.get(_content_key(msg.content))
    return payload


def _parse_embedding(data: Any) -> Optional[List[float]]:
    """Validate an optional persisted "embedding" entry (None when absent)."""
    if data is None:
        return None
    if isinstance(data, tuple):
        data = list(data)
    if (
        not isinstance(data, list)
        or not data
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in data)
    ):
        raise StoreError(
            "Corrupted file: message 'embedding' must be null or a "
            f"non-empty list of numbers, got {str(data)[:80]!r}."
        )
    try:
        values = [float(v) for v in data]
    except (TypeError, ValueError) as exc:
        raise StoreError(
            f"Corrupted file: message 'embedding' values must be numbers ({exc})."
        ) from exc
    if any(v != v or v in (float("inf"), float("-inf")) for v in values):
        raise StoreError("Corrupted file: message 'embedding' values must be finite.")
    return values


def _message_from_dict(data: Any) -> tuple[Message, Optional[List[float]]]:
    """Rebuild a Message plus its optional persisted embedding vector."""
    if not isinstance(data, dict):
        raise StoreError(f"Corrupted file: message entry must be an object, got {type(data).__name__!r}.")
    try:
        role = data["role"]
        content = data["content"]
    except KeyError as exc:
        raise StoreError(f"Corrupted file: message is missing required key {exc}.") from exc
    if not isinstance(role, str):
        raise StoreError(f"Corrupted file: message 'role' must be a string, got {type(role).__name__!r}.")
    if not isinstance(content, str):
        raise StoreError(f"Corrupted file: message 'content' must be a string, got {type(content).__name__!r}.")
    importance = data.get("importance", ImportanceLevel.NORMAL.value)
    protected = data.get("protected", False)
    metadata = data.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise StoreError(
            "Corrupted file: message 'metadata' must be an object, "
            f"got {type(metadata).__name__!r}."
        )
    embedding = _parse_embedding(data.get("embedding"))
    try:
        message = Message(
            role=role,
            content=content,
            metadata=metadata,
            importance=importance,
            protected=protected,
        )
    except (ValueError, TypeError) as exc:
        raise StoreError(f"Corrupted file: invalid message ({exc}).") from exc
    return message, embedding


# ── validation-result helpers ────────────────────────────────────────


def _validation_to_dict(vr: ValidationResult) -> Dict[str, Any]:
    return {
        "passed": vr.passed,
        "warnings": list(vr.warnings),
        "missing_items": list(vr.missing_items),
        "validator_used": vr.validator_used,
        "details": vr.details,
    }


def _validation_from_dict(data: Any) -> ValidationResult:
    if not isinstance(data, dict):
        raise StoreError(
            "Corrupted file: validation result must be an object, "
            f"got {type(data).__name__!r}."
        )
    try:
        passed = data["passed"]
    except KeyError as exc:
        raise StoreError(f"Corrupted file: validation result is missing key {exc}.") from exc
    if not isinstance(passed, bool):
        raise StoreError(
            "Corrupted file: validation 'passed' must be a bool, "
            f"got {type(passed).__name__!r}."
        )
    warnings = data.get("warnings", [])
    missing_items = data.get("missing_items", [])
    validator_used = data.get("validator_used", "none")
    details = data.get("details", "")
    if not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
        raise StoreError("Corrupted file: validation 'warnings' must be a list of strings.")
    if not isinstance(missing_items, list) or not all(isinstance(m, str) for m in missing_items):
        raise StoreError("Corrupted file: validation 'missing_items' must be a list of strings.")
    if not isinstance(validator_used, str):
        raise StoreError("Corrupted file: validation 'validator_used' must be a string.")
    if not isinstance(details, str):
        raise StoreError("Corrupted file: validation 'details' must be a string.")
    return ValidationResult(
        passed=passed,
        warnings=warnings,
        missing_items=missing_items,
        validator_used=validator_used,
        details=details,
    )


# ── priority-score helpers ───────────────────────────────────────────


def _score_to_dict(ps: PriorityScore) -> Dict[str, Any]:
    return {
        "score": ps.score,
        "classification": ps.classification.value,
        "factors": dict(ps.factors),
        "reason": ps.reason,
        "message": _message_to_dict(ps.message),
    }


def _score_from_dict(data: Any) -> PriorityScore:
    if not isinstance(data, dict):
        raise StoreError(
            "Corrupted file: priority score must be an object, "
            f"got {type(data).__name__!r}."
        )
    try:
        message, _embedding = _message_from_dict(data["message"])
    except KeyError as exc:
        raise StoreError(f"Corrupted file: priority score is missing key {exc}.") from exc
    try:
        score = float(data.get("score", 0.0))
    except (TypeError, ValueError) as exc:
        raise StoreError(f"Corrupted file: priority score 'score' is not a number ({exc}).") from exc
    classification = data.get("classification", ImportanceLevel.NORMAL.value)
    factors = data.get("factors", {})
    reason = data.get("reason", "")
    if not isinstance(factors, dict):
        raise StoreError("Corrupted file: priority score 'factors' must be an object.")
    if not isinstance(reason, str):
        raise StoreError("Corrupted file: priority score 'reason' must be a string.")
    try:
        clean_factors = {str(k): float(v) for k, v in factors.items()}
    except (TypeError, ValueError) as exc:
        raise StoreError(f"Corrupted file: priority score 'factors' values must be numbers ({exc}).") from exc
    try:
        return PriorityScore(
            score=score,
            classification=classification,  # coerced by PriorityScore.__post_init__
            message=message,
            factors=clean_factors,
            reason=reason,
        )
    except (ValueError, TypeError) as exc:
        raise StoreError(f"Corrupted file: invalid priority score ({exc}).") from exc


# ── compaction-result helpers ────────────────────────────────────────


def _compaction_to_dict(result: CompactionResult) -> Dict[str, Any]:
    return {
        "original_message_count": result.original_message_count,
        "original_token_count": result.original_token_count,
        "compacted_message_count": result.compacted_message_count,
        "compacted_token_count": result.compacted_token_count,
        "messages_removed": result.messages_removed,
        "tokens_saved": result.tokens_saved,
        "summary_message": _message_to_dict(result.summary_message) if result.summary_message else None,
        "was_needed": result.was_needed,
        "summarizer_used": result.summarizer_used,
        "validation_result": _validation_to_dict(result.validation_result) if result.validation_result else None,
        "committed": result.committed,
        "preserved_messages": [_message_to_dict(m) for m in result.preserved_messages],
        "summarized_messages": [_message_to_dict(m) for m in result.summarized_messages],
        "removed_messages": [_message_to_dict(m) for m in result.removed_messages],
        "priority_scores": [_score_to_dict(ps) for ps in result.priority_scores],
    }


def _compaction_from_dict(data: Any) -> CompactionResult:
    if not isinstance(data, dict):
        raise StoreError(
            "Corrupted file: compaction result must be an object, "
            f"got {type(data).__name__!r}."
        )
    required_ints = (
        "original_message_count",
        "original_token_count",
        "compacted_message_count",
        "compacted_token_count",
        "messages_removed",
        "tokens_saved",
    )
    for key in required_ints:
        if key not in data:
            raise StoreError(f"Corrupted file: compaction result is missing key {key!r}.")
        if not isinstance(data[key], int) or isinstance(data[key], bool):
            raise StoreError(
                f"Corrupted file: compaction {key!r} must be an integer, "
                f"got {type(data[key]).__name__!r}."
            )
    for key in ("was_needed", "committed"):
        if key not in data:
            raise StoreError(f"Corrupted file: compaction result is missing key {key!r}.")
        if not isinstance(data[key], bool):
            raise StoreError(
                f"Corrupted file: compaction {key!r} must be a bool, "
                f"got {type(data[key]).__name__!r}."
            )
    summarizer_used = data.get("summarizer_used", "none")
    if not isinstance(summarizer_used, str):
        raise StoreError("Corrupted file: compaction 'summarizer_used' must be a string.")

    raw_summary = data.get("summary_message")
    if raw_summary is not None:
        summary_message, _ = _message_from_dict(raw_summary)
    else:
        summary_message = None

    raw_validation = data.get("validation_result")
    validation_result = _validation_from_dict(raw_validation) if raw_validation is not None else None

    def _msg_list(key: str) -> List[Message]:
        raw = data.get(key, [])
        if not isinstance(raw, list):
            raise StoreError(f"Corrupted file: compaction {key!r} must be a list.")
        return [_message_from_dict(item)[0] for item in raw]

    raw_scores = data.get("priority_scores", [])
    if not isinstance(raw_scores, list):
        raise StoreError("Corrupted file: compaction 'priority_scores' must be a list.")
    priority_scores = [_score_from_dict(item) for item in raw_scores]

    return CompactionResult(
        original_message_count=data["original_message_count"],
        original_token_count=data["original_token_count"],
        compacted_message_count=data["compacted_message_count"],
        compacted_token_count=data["compacted_token_count"],
        messages_removed=data["messages_removed"],
        tokens_saved=data["tokens_saved"],
        summary_message=summary_message,
        was_needed=data["was_needed"],
        summarizer_used=summarizer_used,
        validation_result=validation_result,
        committed=data["committed"],
        preserved_messages=_msg_list("preserved_messages"),
        summarized_messages=_msg_list("summarized_messages"),
        removed_messages=_msg_list("removed_messages"),
        priority_scores=priority_scores,
    )


# ── conversation helpers ─────────────────────────────────────────────


def _conversation_to_dict(
    conv: Conversation,
    embeddings: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    return {
        "name": conv.name,
        "max_tokens": conv.max_tokens,
        "warn_at": conv.warn_at,
        "compact_at": conv.compact_at,
        "messages": [_message_to_dict(m, embeddings) for m in conv.get_messages()],
    }


def _conversation_from_dict(data: Any) -> tuple[Conversation, Dict[str, List[float]]]:
    """Rebuild a Conversation plus its {content-hash: embedding} map."""
    if not isinstance(data, dict):
        raise StoreError(
            "Corrupted file: 'conversation' must be an object, "
            f"got {type(data).__name__!r}."
        )
    name = data.get("name", "Conversation")
    max_tokens = data.get("max_tokens")
    warn_at = data.get("warn_at", 70.0)
    compact_at = data.get("compact_at", 90.0)
    if not isinstance(name, str):
        raise StoreError("Corrupted file: conversation 'name' must be a string.")
    raw_messages = data.get("messages")
    if raw_messages is None:
        raise StoreError("Corrupted file: conversation is missing key 'messages'.")
    if not isinstance(raw_messages, list):
        raise StoreError("Corrupted file: conversation 'messages' must be a list.")
    try:
        conv = Conversation(
            name=name,
            max_tokens=max_tokens,
            warn_at=warn_at,
            compact_at=compact_at,
        )
    except (TypeError, ValueError) as exc:
        raise StoreError(f"Corrupted file: invalid conversation settings ({exc}).") from exc
    embeddings: Dict[str, List[float]] = {}
    for item in raw_messages:
        message, vector = _message_from_dict(item)
        conv.add(message)
        if vector is not None:
            embeddings[_content_key(message.content)] = vector
    return conv, embeddings


def _coerce_path(path: PathLike) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(
            f"path must be a string or path-like object, got {type(path).__name__!r}."
        )
    text = os.fspath(path)
    if not text.strip():
        raise ValueError("path must not be empty.")
    return Path(text)


# ── ContextStore ─────────────────────────────────────────────────────


class ContextStore:
    """JSON-backed persistent storage for one conversation and its history.

    Attributes:
        conversation:    The live Conversation being tracked.
        last_compaction: Most recent CompactionResult (if any) -- holds the
                         compaction summary message plus before/after metrics.
        last_validation: Most recent ValidationResult (if any).
        embeddings:      ``{content_hash: vector}`` map of persisted message
                         embeddings (Step 14). Vectors are stored alongside
                         their messages in the JSON file.

    Typical usage::

        store = ContextStore(conversation)
        result = compactor.compact(conversation)
        store.record(result)          # remember summary + validation
        store.embed_missing(provider) # embed messages once, cache vectors
        store.save("session.json")    # persist to disk

        restored = ContextStore.load_file("session.json")
        restored.conversation         # behaves like a freshly built Conversation
        retriever.prime_cache(restored)  # reuse vectors without re-embedding

    No database is used in this step -- everything is plain JSON.
    """

    def __init__(
        self,
        conversation: Optional[Conversation] = None,
        last_compaction: Optional[CompactionResult] = None,
        last_validation: Optional[ValidationResult] = None,
        embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        if conversation is not None and not isinstance(conversation, Conversation):
            raise TypeError(
                "conversation must be a Conversation instance, "
                f"got {type(conversation).__name__!r}."
            )
        if last_compaction is not None and not isinstance(last_compaction, CompactionResult):
            raise TypeError(
                "last_compaction must be a CompactionResult instance, "
                f"got {type(last_compaction).__name__!r}."
            )
        if last_validation is not None and not isinstance(last_validation, ValidationResult):
            raise TypeError(
                "last_validation must be a ValidationResult instance, "
                f"got {type(last_validation).__name__!r}."
            )
        self.conversation: Conversation = conversation or Conversation(name="Conversation")
        self.last_compaction: Optional[CompactionResult] = last_compaction
        self.last_validation: Optional[ValidationResult] = last_validation
        self.embeddings: Dict[str, List[float]] = self._validate_embeddings(embeddings)

    @staticmethod
    def _validate_embeddings(value: Optional[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(
                "embeddings must be a {content-hash: vector} dict, "
                f"got {type(value).__name__!r}."
            )
        clean: Dict[str, List[float]] = {}
        for key, vector in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "embeddings keys must be content-hash strings, "
                    f"got {type(key).__name__!r}."
                )
            try:
                clean[key] = validate_vector(vector, label=f"embeddings[{key[:8]}...]")
            except EmbeddingError as exc:
                raise TypeError(f"Invalid embeddings map ({exc}).") from exc
        return clean

    # ── recording ────────────────────────────────────────────────

    def record(self, result: CompactionResult) -> None:
        """Remember a CompactionResult (and its nested ValidationResult)."""
        if not isinstance(result, CompactionResult):
            raise TypeError(
                "result must be a CompactionResult instance, "
                f"got {type(result).__name__!r}."
            )
        self.last_compaction = result
        if result.validation_result is not None:
            self.last_validation = result.validation_result

    # ── embeddings (Step 14) ─────────────────────────────────────

    @staticmethod
    def _key_for(message_or_text: Union[Message, str]) -> str:
        if isinstance(message_or_text, Message):
            return _content_key(message_or_text.content)
        if isinstance(message_or_text, str):
            return _content_key(message_or_text)
        raise TypeError(
            "expected a Message or string, "
            f"got {type(message_or_text).__name__!r}."
        )

    def set_embedding(self, message_or_text: Union[Message, str], vector: List[float]) -> None:
        """Store an embedding vector for a message (keyed by content hash)."""
        try:
            clean = validate_vector(vector)
        except EmbeddingError as exc:
            raise ValueError(f"Cannot store invalid embedding ({exc}).") from exc
        self.embeddings[self._key_for(message_or_text)] = clean

    def get_embedding(self, message_or_text: Union[Message, str]) -> Optional[List[float]]:
        """Return the stored vector for a message, or None if not embedded."""
        return self.embeddings.get(self._key_for(message_or_text))

    def embed_missing(self, provider: Any) -> int:
        """Embed conversation messages lacking vectors via *provider*.

        Returns the number of messages newly embedded (0 when everything
        is already cached). Raises EmbeddingError if the provider fails.
        """
        if not hasattr(provider, "embed"):
            raise TypeError(
                "provider must implement embed(texts), "
                f"got {type(provider).__name__!r}."
            )
        missing = [
            m for m in self.conversation.get_messages()
            if _content_key(m.content) not in self.embeddings
        ]
        if not missing:
            return 0
        vectors = provider.embed([m.content for m in missing])
        if len(vectors) != len(missing):
            raise EmbeddingError(
                f"Provider returned {len(vectors)} vector(s) "
                f"for {len(missing)} message(s)."
            )
        for message, vector in zip(missing, vectors):
            self.set_embedding(message, vector)
        return len(missing)

    # ── serialization ────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return the full store payload as a JSON-serializable dict."""
        return {
            "kind": STORE_KIND,
            "version": FORMAT_VERSION,
            "conversation": _conversation_to_dict(self.conversation, self.embeddings),
            "last_compaction": _compaction_to_dict(self.last_compaction) if self.last_compaction else None,
            "last_validation": _validation_to_dict(self.last_validation) if self.last_validation else None,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ContextStore":
        """Rebuild a ContextStore from a decoded JSON payload."""
        if not isinstance(data, dict):
            raise StoreError(
                "Corrupted file: top-level JSON must be an object, "
                f"got {type(data).__name__!r}."
            )
        if data.get("kind") != STORE_KIND:
            raise StoreError(
                f"Corrupted file: unexpected 'kind' {data.get('kind')!r} "
                f"(expected {STORE_KIND!r})."
            )
        if data.get("version") != FORMAT_VERSION:
            raise StoreError(
                f"Unsupported store version {data.get('version')!r} "
                f"(this build reads version {FORMAT_VERSION})."
            )
        if "conversation" not in data:
            raise StoreError("Corrupted file: missing required key 'conversation'.")
        conversation, embeddings = _conversation_from_dict(data["conversation"])
        raw_compaction = data.get("last_compaction")
        raw_validation = data.get("last_validation")
        last_compaction = _compaction_from_dict(raw_compaction) if raw_compaction is not None else None
        last_validation = _validation_from_dict(raw_validation) if raw_validation is not None else None
        return cls(
            conversation=conversation,
            last_compaction=last_compaction,
            last_validation=last_validation,
            embeddings=embeddings,
        )

    # ── file I/O ─────────────────────────────────────────────────

    def save(self, path: PathLike) -> Path:
        """Save this store to *path* as JSON (overwrites existing files).

        Raises:
            TypeError:  If *path* is not a string / path-like object.
            ValueError: If *path* is empty.
            StoreError: If the file cannot be written (e.g. missing parent
                        directory) or the payload is not JSON-serializable.
        """
        target = _coerce_path(path)
        try:
            payload = self.to_dict()
            text = json.dumps(payload, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise StoreError(f"Cannot serialize context store ({exc}).") from exc
        try:
            target.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise StoreError(f"Cannot save context to {str(target)!r} ({exc}).") from exc
        return target

    def load(self, path: PathLike) -> "ContextStore":
        """Load store state from *path* into this object (replaces state).

        Returns self so calls can be chained. See load_file() for a
        classmethod that creates a new store directly.

        Raises:
            TypeError:  If *path* is not a string / path-like object.
            ValueError: If *path* is empty.
            StoreError: If the file is missing, holds invalid JSON, or fails
                        schema validation.
        """
        other = self.load_file(path)
        self.conversation = other.conversation
        self.last_compaction = other.last_compaction
        self.last_validation = other.last_validation
        self.embeddings = other.embeddings
        return self

    @classmethod
    def load_file(cls, path: PathLike) -> "ContextStore":
        """Load and return a new ContextStore from the JSON file at *path*.

        Raises:
            TypeError:  If *path* is not a string / path-like object.
            ValueError: If *path* is empty.
            StoreError: If the file is missing ("not found"), invalid JSON
                        ("invalid JSON"), or fails schema validation
                        ("corrupted ...").
        """
        target = _coerce_path(path)
        try:
            text = target.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise StoreError(f"Context file not found: {str(target)!r}.") from exc
        except OSError as exc:
            raise StoreError(f"Cannot read context file {str(target)!r} ({exc}).") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StoreError(
                f"Corrupted file {str(target)!r}: invalid JSON ({exc})."
            ) from exc
        try:
            return cls.from_dict(data)
        except StoreError as exc:
            # Add the file location for easier debugging, preserving the cause.
            raise StoreError(f"Corrupted file {str(target)!r}: {exc}") from exc
