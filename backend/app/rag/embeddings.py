import hashlib
import logging
import re
from typing import Protocol

import httpx
import numpy as np

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[0-9a-zA-Z가-힣]+")


class Embeddings(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) L2-normalized matrix."""


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class HashingEmbeddings:
    """Offline embeddings over character n-grams.

    Character n-grams beat word tokens for Korean, where particles and compounds
    make whole-word matching brittle.
    """

    name = "hashing-char-ngram"

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                index = int.from_bytes(digest, "big") % self.dim
                matrix[row, index] += 1.0
        return _normalize(np.log1p(matrix))

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN.findall(text.lower())
        features = list(tokens)
        joined = " ".join(tokens)
        for size in (2, 3):
            features.extend(joined[i : i + size] for i in range(len(joined) - size + 1))
        return features


class OpenAIEmbeddings:
    # Shared across instances/requests within one process: once this provider's
    # endpoint proves unavailable, stop paying the round-trip on every chunk.
    _unavailable_base_urls: set[str] = set()

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._fallback = HashingEmbeddings()
        self.name = f"openai:{settings.openai_embedding_model}"

    async def embed(self, texts: list[str]) -> np.ndarray:
        base_url = self._settings.openai_base_url
        if base_url in self._unavailable_base_urls:
            return await self._fallback.embed(texts)
        try:
            async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
                response = await client.post(
                    f"{base_url.rstrip('/')}/embeddings",
                    json={"model": self._settings.openai_embedding_model, "input": texts},
                    headers={"Authorization": f"Bearer {self._settings.openai_api_key}"},
                )
                response.raise_for_status()
                vectors = [item["embedding"] for item in response.json()["data"]]
            return _normalize(np.array(vectors, dtype=np.float32))
        except httpx.HTTPStatusError as exc:
            logger.exception("OpenAI embedding call failed; using offline embeddings")
            if exc.response.status_code in (404, 501):
                self._unavailable_base_urls.add(base_url)
            return await self._fallback.embed(texts)
        except Exception:
            logger.exception("OpenAI embedding call failed; using offline embeddings")
            return await self._fallback.embed(texts)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    if settings.live_llm:
        return OpenAIEmbeddings(settings)
    return HashingEmbeddings()
