"""Async chat client for any OpenAI-compatible provider, hardened for free tiers.

- timeout:      free tiers sometimes hang instead of failing (we saw a 45 s hang in Phase 0)
- retries:      the openai SDK retries 429 / 5xx (incl. "503 overloaded") with exponential backoff
- fallback:     if the model is still overloaded / out of quota after the retries, try the next
                model in the chain (free-tier quotas are per model, e.g. 20 requests/day for flash)
- rate limiter: spaces requests to LLM_RPM, so we don't trigger 429s in the first place
- disk cache:   identical request -> stored reply, zero cost (re-running ingestion is free)
- usage:        token counts per call, needed for the cost column in the evaluation
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from openai import APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from pydantic import BaseModel

from backend.cache import DiskCache, cache_key
from backend.config import Endpoint

log = logging.getLogger(__name__)


class RateLimiter:
    """At most `rpm` request starts per minute, evenly spaced (safe across concurrent tasks)."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.next_slot = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            if self.next_slot > now:
                await asyncio.sleep(self.next_slot - now)
            self.next_slot = max(now, self.next_slot) + self.interval


@dataclass
class Usage:
    calls: int = 0  # real API calls (cache hits excluded)
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0  # wall time spent in API calls


@dataclass
class Completion:
    """One reply plus what it cost. Returned per call, so concurrent requests (API server) each
    get their own numbers instead of reading a shared counter."""

    reply: str
    model: str
    cached: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM:
    def __init__(self, endpoint: Endpoint, rpm: int = 10, timeout: float = 60.0,
                 cache: DiskCache | None = None, temperature: float = 0.0, max_retries: int = 3):
        self.endpoint = endpoint
        self.cache = cache
        self.temperature = temperature  # 0 = as repeatable as the API allows
        self.limiter = RateLimiter(rpm)
        self.usage = Usage()
        self.client = AsyncOpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key,
                                  timeout=timeout, max_retries=max_retries)

    async def complete(self, messages: list[dict], schema: type[BaseModel] | None = None) -> str:
        return (await self.generate(messages, schema)).reply

    async def generate(self, messages: list[dict],
                       schema: type[BaseModel] | None = None) -> Completion:
        """One chat completion. With `schema`, the provider is asked to return JSON matching it
        (structured output); the caller still validates, because not every provider enforces it.

        Tries the main model, then each fallback model when one is overloaded or times out."""
        response_format = _response_format(schema) if schema else None
        models = [self.endpoint.model, *self.endpoint.fallback_models]
        keys = {m: cache_key(provider=self.endpoint.provider, model=m, temperature=self.temperature,
                             messages=messages, response_format=response_format) for m in models}
        if self.cache is not None:
            for model in models:  # a reply from a fallback model counts too: re-runs stay free
                if (hit := self.cache.get(keys[model])) is not None:
                    self.usage.cache_hits += 1
                    return Completion(hit["reply"], hit["model"], cached=True)

        for i, model in enumerate(models):
            try:
                reply, usage = await self._request(model, messages, response_format)
            # overloaded (503), hanging, or out of quota (429; quotas are per model) after retries
            except (InternalServerError, APITimeoutError, RateLimitError) as exc:
                if i == len(models) - 1:
                    raise
                log.warning("%s unavailable (%s), falling back to %s", model,
                            type(exc).__name__, models[i + 1])
                continue
            if self.cache is not None:
                self.cache.put(keys[model], {"model": model, "reply": reply, "usage": usage})
            usage = usage or {}
            return Completion(reply, model, cached=False,
                              prompt_tokens=usage.get("prompt_tokens") or 0,
                              completion_tokens=usage.get("completion_tokens") or 0)
        raise AssertionError("unreachable")

    async def _request(self, model: str, messages: list[dict],
                       response_format: dict | None) -> tuple[str, dict | None]:
        await self.limiter.wait()
        start = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            **({"response_format": response_format} if response_format else {}),
        )
        self.usage.calls += 1
        self.usage.seconds += time.perf_counter() - start
        usage = response.usage
        if usage:
            self.usage.prompt_tokens += usage.prompt_tokens
            self.usage.completion_tokens += usage.completion_tokens
        return response.choices[0].message.content or "", usage.model_dump() if usage else None


def _response_format(schema: type[BaseModel]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
    }


def parse_json(reply: str) -> dict:
    """Some models wrap JSON in ```json fences even in JSON mode; strip them before parsing."""
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)
