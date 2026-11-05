"""Smoke test of the environment: `python -m backend.check`.

Checks Neo4j, one chat call and one embedding call, so a broken .env shows up before ingestion.
"""

import asyncio
import time

from openai import AsyncOpenAI

from backend.config import Settings, get_settings
from backend.db import create_driver, server_info


async def check_neo4j(settings: Settings) -> str:
    driver = create_driver(settings)
    try:
        return await server_info(driver)
    finally:
        await driver.close()


async def check_chat(settings: Settings) -> str:
    ep = settings.chat_endpoint()
    client = AsyncOpenAI(base_url=ep.base_url, api_key=ep.api_key, timeout=60)
    reply = await client.chat.completions.create(
        model=ep.model, messages=[{"role": "user", "content": "Reply with the single word: ok"}]
    )
    return f"{ep.provider}/{ep.model} -> {reply.choices[0].message.content.strip()!r}"


async def check_embeddings(settings: Settings) -> str:
    ep = settings.embed_endpoint()
    client = AsyncOpenAI(base_url=ep.base_url, api_key=ep.api_key, timeout=60)
    extra = {"dimensions": ep.dimensions} if ep.dimensions else {}
    result = await client.embeddings.create(model=ep.model, input="BAWAG Group", **extra)
    return f"{ep.provider}/{ep.model} -> {len(result.data[0].embedding)} dims"


async def main() -> None:
    settings = get_settings()
    for name, check in [("neo4j", check_neo4j), ("chat", check_chat), ("embed", check_embeddings)]:
        start = time.perf_counter()
        try:
            result = await check(settings)
            print(f"[ok]   {name:6} {result}  ({time.perf_counter() - start:.1f}s)")
        # a smoke test reports every failure instead of stopping at the first
        except Exception as exc:
            print(f"[FAIL] {name:6} {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
