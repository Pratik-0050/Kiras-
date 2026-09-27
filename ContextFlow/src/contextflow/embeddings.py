# contextflow/embeddings.py
"""
Step 14: Semantic context retrieval using embeddings.

This module defines the EmbeddingProvider interface and an OpenAI-compatible
implementation. Providers turn message / query text into dense float vectors
so that retrieval can rank by *meaning* (cosine similarity) instead of raw
keyword overlap.

Configuration (via environment variables or constructor args):
    OPENAI_API_KEY        -- Required. Your API key (shared with the
                             summarizer / validator).
    OPENAI_EMBEDDING_MODEL -- Optional. Defaults to "text-embedding-3-small".
    OPENAI_BASE_URL       -- Optional. Defaults to "https://api.openai.com/v1".
                             Override for any OpenAI-compatible endpoint.

No vector database is used in this step -- vectors live in the in-memory
retriever cache and, when persisted, alongside messages in the JSON store.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


class EmbeddingError(Exception):
    """Raised when embeddings cannot be produced.

    Covers missing credentials, API / network failures, and malformed
    responses. A *failed similarity comparison* is never an error -- it is
    simply a low score. Only infrastructure problems raise this.
    """


def _content_key(text: str) -> str:
    """Stable cache key for a message / query string (sha256 hex)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_vector(vector: Any, *, label: str = "embedding") -> List[float]:
    """Validate *vector* and return it as a list of floats.

    Raises:
        EmbeddingError: If *vector* is not a non-empty list of numbers.
    """
    if isinstance(vector, tuple):
        vector = list(vector)
    if (
        not isinstance(vector, list)
        or not vector
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vector)
    ):
        raise EmbeddingError(
            f"Invalid {label}: expected a non-empty list of numbers, "
            f"got {str(vector)[:80]!r}."
        )
    try:
        values = [float(v) for v in vector]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(f"Invalid {label}: values must be numbers ({exc}).") from exc
    if any(v != v or v in (float("inf"), float("-inf")) for v in values):
        raise EmbeddingError(f"Invalid {label}: values must be finite numbers.")
    return values


class EmbeddingProvider(ABC):
    """Abstract interface that every embedding provider must implement.

    Subclasses must implement embed(): turn a batch of texts into dense
    vectors. embed_one() is provided as a convenience wrapper.
    """

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts.

        Args:
            texts: Non-empty strings to embed.

        Returns:
            One float vector per input text, in the same order.

        Raises:
            EmbeddingError: On configuration problems, API failures, or
                            malformed responses.
        """
        pass

    def embed_one(self, text: str) -> List[float]:
        """Embed a single text (convenience wrapper around embed())."""
        return self.embed([text])[0]


# BaseEmbeddingProvider is an alias kept for naming consistency with the
# other subsystems (BaseSummarizer, BaseValidator, BaseRetriever, ...).
BaseEmbeddingProvider = EmbeddingProvider


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings via any OpenAI-compatible embeddings API.

    Args:
        api_key:  API key. Defaults to OPENAI_API_KEY environment variable.
        model:    Embedding model. Defaults to OPENAI_EMBEDDING_MODEL env var
                  or "text-embedding-3-small".
        base_url: API base URL. Defaults to OPENAI_BASE_URL env var or
                  "https://api.openai.com/v1".
        client:   Optional pre-configured OpenAI client (for testing/proxies).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        env_key = os.getenv("OPENAI_API_KEY", "").strip()
        env_model = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()
        env_base_url = os.getenv("OPENAI_BASE_URL", "").strip()

        self.api_key: str = (api_key if api_key is not None else env_key).strip()
        self.model: str = (model if model is not None else (env_model or _DEFAULT_EMBEDDING_MODEL)).strip()
        self.base_url: str = (base_url if base_url is not None else (env_base_url or _DEFAULT_BASE_URL)).strip()

        if client is not None:
            self._client = client
        else:
            if not self.api_key:
                raise EmbeddingError(
                    "No API key found. "
                    "Set the OPENAI_API_KEY environment variable or pass api_key=... "
                    "to OpenAIEmbeddingProvider()."
                )
            try:
                import openai as _openai
                self._client = _openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise EmbeddingError(
                    "The 'openai' package is not installed. "
                    "Run: pip install openai"
                )

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed *texts* in a single API call.

        Raises:
            EmbeddingError: If any text is not a non-empty string, the API
                            call fails, or the response is malformed.
        """
        items = list(texts)
        if not items:
            return []
        for text in items:
            if not isinstance(text, str):
                raise EmbeddingError(
                    f"Texts to embed must be strings, got {type(text).__name__!r}."
                )
            if not text.strip():
                raise EmbeddingError("Texts to embed must not be empty.")

        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=items,
            )
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embeddings call failed: {exc}") from exc

        data = getattr(response, "data", None)
        if not data:
            raise EmbeddingError("OpenAI embeddings call returned no data.")
        if len(data) != len(items):
            raise EmbeddingError(
                f"OpenAI embeddings returned {len(data)} vector(s) "
                f"for {len(items)} input text(s)."
            )

        vectors: List[List[float]] = []
        for entry in data:
            raw = getattr(entry, "embedding", None)
            if raw is None:
                raise EmbeddingError("OpenAI embeddings entry is missing its vector.")
            try:
                vectors.append(validate_vector(raw))
            except EmbeddingError as exc:
                raise EmbeddingError(f"OpenAI returned a malformed vector ({exc}).") from exc

        first_len = len(vectors[0])
        if any(len(v) != first_len for v in vectors):
            raise EmbeddingError("OpenAI returned inconsistent vector dimensions.")
        return vectors

    def cache_key(self) -> Dict[str, str]:
        """Identify this provider configuration (for cache namespacing)."""
        return {"model": self.model, "base_url": self.base_url}
