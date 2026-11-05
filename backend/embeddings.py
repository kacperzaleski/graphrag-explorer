"""Text embeddings over the OpenAI-compatible API, batched and cached per text.

Cache key = provider + model + dimensions + text: only new or changed chunks are re-embedded.
Rate limit is counted in *texts*, not HTTP requests: Gemini's free tier counts every text in a
batch as one request (100/minute), so one 100-text batch would use up a whole minute.
"""

from openai import AsyncOpenAI

from backend.cache import DiskCache, cache_key
from backend.config import Endpoint
from backend.llm import RateLimiter

BATCH_SIZE = 20


class Embedder:
    def __init__(self, endpoint: Endpoint, cache: DiskCache | None = None, timeout: float = 60.0,
                 texts_per_minute: int = 80):
        self.endpoint = endpoint
        self.cache = cache
        self.calls = 0
        # one limiter slot per batch: batches/minute = texts/minute / batch size
        self.limiter = RateLimiter(max(1, texts_per_minute // BATCH_SIZE))
        self.client = AsyncOpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key,
                                  timeout=timeout, max_retries=5)

    def _key(self, text: str) -> str:
        ep = self.endpoint
        return cache_key(provider=ep.provider, model=ep.model, dimensions=ep.dimensions, text=text)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: dict[int, list[float]] = {}
        missing: list[int] = []
        for i, text in enumerate(texts):
            hit = self.cache.get(self._key(text)) if self.cache else None
            if hit is None:
                missing.append(i)
            else:
                vectors[i] = hit
        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start:start + BATCH_SIZE]
            result = await self._request([texts[i] for i in batch])
            for i, vector in zip(batch, result, strict=True):
                vectors[i] = vector
                if self.cache:
                    self.cache.put(self._key(texts[i]), vector)
        return [vectors[i] for i in range(len(texts))]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    async def _request(self, texts: list[str]) -> list[list[float]]:
        await self.limiter.wait()
        ep = self.endpoint
        extra = {"dimensions": ep.dimensions} if ep.dimensions else {}
        result = await self.client.embeddings.create(model=ep.model, input=texts, **extra)
        self.calls += 1
        data = result.data
        # OpenAI sets `index` per item; Gemini's compatible endpoint returns null and keeps order
        if all(d.index is not None for d in data):
            data = sorted(data, key=lambda d: d.index)
        return [d.embedding for d in data]
