from __future__ import annotations
"""
CrimeVision-QA — Embedding Service

Provider is resolved ONCE at module import time via config.py.
No per-request fallback loops.

    from llm.get_voyage_embed import embedding_service

    vectors = embedding_service.embed(["text one", "text two"])
    single  = embedding_service.embed_single("text one")
"""

import hashlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import (
    EMBED_PROVIDER,
    FIREWORKS_API_KEY,
    FIREWORKS_API_BASE,
    FIREWORKS_EMBED_MODEL,
    VOYAGE_API_KEY,
    VOYAGE_EMBED_MODEL,
)

_VOYAGE_BATCH = 128   # max texts per Voyage API call
_FIREWORKS_BATCH = 96 # max texts per Fireworks embedding call
_EMBED_DIM = 1024     # expected output dimension


def _chunk(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


class EmbeddingService:
    """Embeds text into 1024-dim vectors using the provider chosen at startup."""

    def __init__(self) -> None:
        # Provider already resolved in config.py; we just use it.
        self.provider: str = EMBED_PROVIDER
        print(f"[Embeddings] Using provider: {self.provider}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of 1024-dim float vectors."""
        if not texts:
            return []
        if self.provider == "voyage":
            return self._voyage_embed(texts)
        return self._fireworks_embed(texts)

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text. Uses LRU cache to avoid re-embedding duplicates."""
        return self._cached_embed_single(text)

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cached_embed_single(self, text: str) -> list[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        result = self.embed([text])[0]
        self._cache[key] = result
        return result

    # Simple dict-based LRU (max 10k entries)
    _cache: dict[str, list[float]] = {}
    _CACHE_MAX = 10_000

    def _put_cache(self, key: str, value: list[float]) -> None:
        if len(self._cache) >= self._CACHE_MAX:
            # evict first key (insertion order in Python 3.7+)
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    # ------------------------------------------------------------------
    # Voyage AI implementation
    # ------------------------------------------------------------------

    def _voyage_embed(self, texts: list[str]) -> list[list[float]]:
        import voyageai  # imported lazily — only installed if Voyage is used
        client = voyageai.Client(api_key=VOYAGE_API_KEY)
        all_embeddings: list[list[float]] = []
        for batch in _chunk(texts, _VOYAGE_BATCH):
            result = client.embed(batch, model=VOYAGE_EMBED_MODEL, input_type="document")
            all_embeddings.extend(result.embeddings)
        return all_embeddings

    # ------------------------------------------------------------------
    # Fireworks GTE-large implementation
    # ------------------------------------------------------------------

    def _fireworks_embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
        }
        for batch in _chunk(texts, _FIREWORKS_BATCH):
            resp = requests.post(
                f"{FIREWORKS_API_BASE}/embeddings",
                headers=headers,
                json={"model": FIREWORKS_EMBED_MODEL, "input": batch},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            # data is sorted by index
            batch_embeddings = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings


# Module-level singleton — imported by all other modules
embedding_service = EmbeddingService()
