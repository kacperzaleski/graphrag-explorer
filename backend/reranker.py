"""Second-stage reranking with a local cross-encoder (from my earlier RAG demo).

Embeddings are a *bi-encoder*: question and chunk are embedded separately and compared afterwards.
A *cross-encoder* reads question and chunk together in one pass and outputs a relevance score, so
it sees interactions ("before Cerberus" -> the chunk about the earlier owner). More precise, but
too slow to score every chunk, so it only reorders a candidate pool from the first stage.
Consequence: it can only rescue what is inside the pool (candidate recall matters).

The model runs on the CPU and is CPU-bound: calling it directly inside `async def` would block
the event loop and freeze every concurrent request, so it runs in a worker thread.
"""

import asyncio
from functools import lru_cache

from backend.search import Hit

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~90 MB, ~50 ms for 30 pairs on CPU


@lru_cache(maxsize=2)
def _load(model_name: str):
    # imported lazily: torch takes seconds to import, and `RERANKER=none` should not pay for it
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

    def warm_up(self) -> None:
        """Load the model at startup instead of on the first user request."""
        _load(self.model_name)

    async def rerank(self, question: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        model = _load(self.model_name)
        pairs = [(question, h.text) for h in hits]
        scores = await asyncio.to_thread(model.predict, pairs)
        order = sorted(range(len(hits)), key=lambda i: -scores[i])
        return [Hit(hits[i].id, float(scores[i]), hits[i].text,
                    {**hits[i].details, "rerank": {"rank": rank, "score": float(scores[i])}})
                for rank, i in enumerate(order[:k], 1)]


class NoReranker:
    """Keeps the first-stage order (for the eval: 'with vs without reranker')."""

    def warm_up(self) -> None:
        pass

    async def rerank(self, question: str, hits: list[Hit], k: int) -> list[Hit]:
        return hits[:k]


def make_reranker(name: str):
    if name == "cross-encoder":
        return CrossEncoderReranker()
    if name == "none":
        return NoReranker()
    raise ValueError(f"Unknown reranker '{name}', use cross-encoder or none")
