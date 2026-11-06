"""Ask the demo questions once, so the live demo is served from the cache:
`uv run python -m scripts.warm_demo`

During a talk a free-tier API can be slow, overloaded or out of quota. Answers are cached by the
exact request, so after this script the same questions in the UI return instantly and identically
to the rehearsal. Prints each answer so they can be checked beforehand.
"""

import asyncio

from backend.config import get_settings
from backend.db import create_driver
from backend.pipeline import Pipeline

DEMO = [
    ("Who have been the owners of BAWAG over time?", "graph"),
    ("How is BayernLB connected to Addiko Bank?", "graph"),
    ("How is BayernLB connected to Addiko Bank?", "vector"),
    ("How is the Austrian Trade Union Federation connected to GoldenTree?", "graph"),
    ("Which bank held 46% of BAWAG before the trade union bought it back?", "graph"),
    ("Who owned Hypo Alpe-Adria-Bank before it was nationalised?", "graph"),
    # reserves (verified correct in the evaluation)
    ("Which Munich-based bank bought the shares of the shareholder Konsum after it went "
     "bankrupt in 1995?", "graph"),
    ("How is Česká spořitelna connected to Slovenská Sporiteľňa?", "graph"),
    ("Which bank merged on October 1, 2005, with the postal savings bank that was owned by the "
     "Austrian Post Office?", "graph"),
    ("Which American company did BAWAG buy the piano manufacturer Bösendorfer from in 2001?",
     "graph"),
]


async def main() -> None:
    settings = get_settings()
    driver = create_driver(settings)
    try:
        pipeline = Pipeline.from_settings(settings, driver)
        for question, mode in DEMO:
            result = await pipeline.ask(question, mode, settings.hops)
            a = result.answer
            flags = "  (cached)" if a.cached else ""
            flags += f"  ISSUES {a.issues}" if a.issues else ""
            print(f"\n[{mode}] {question}\n  -> {a.answer}\n  citations {a.cited_chunks}{flags}")
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
