"""Download the corpus: English Wikipedia articles on Austrian banks, owners and regulators.

`python scripts/fetch_data.py` writes data/raw/<slug>.md plus data/raw/sources.json.

Uses the MediaWiki API's plain-text extracts (no HTML parsing). Section headings come back as
"== History ==" and are converted to markdown ("## History"), which the chunker understands.
Each article is cut to MAX_CHARS (lead + first sections) so ingestion stays cheap.
Wikipedia text is CC BY-SA 4.0: sources.json keeps URL and revision id for attribution.
"""

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import httpx

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
API = "https://en.wikipedia.org/w/api.php"
# Wikimedia's robot policy blocks generic clients. Format:
# "<tool>/<version> (<contact>) <library>/<version>".
# The contact (URL or email) is optional here and only read from the environment.
_CONTACT = os.getenv("WIKIPEDIA_CONTACT")
HEADERS = {
    "User-Agent": f"GraphRAGExplorerDemo/0.1{f' ({_CONTACT})' if _CONTACT else ''} "
    f"httpx/{httpx.__version__}",
    # httpx's default "gzip, deflate, br, zstd" gets a 403 from Wikimedia's bot filter;
    # plain gzip (what curl sends) works
    "Accept-Encoding": "gzip",
}
MAX_CHARS = 12_000
SKIP_SECTIONS = {"see also", "references", "external links", "further reading", "notes",
                 "bibliography", "sources", "literature", "citations"}

TITLES = [
    # BAWAG: ownership chain trade union -> Cerberus -> listed company, Refco scandal
    "BAWAG",  # the article title is "BAWAG Group"; this name resolves via redirect
    "Österreichische Postsparkasse",
    "Austrian Trade Union Federation",
    "Cerberus Capital Management",
    "Refco",
    # Erste and its Central European subsidiaries
    "Erste Group",
    "Česká spořitelna",
    "Slovenská sporiteľňa",
    # Raiffeisen
    "Raiffeisen Bank International",
    "Raiffeisen Zentralbank",
    # UniCredit / Bank Austria / Creditanstalt
    "UniCredit",
    "UniCredit Bank Austria",
    "Creditanstalt",
    # Hypo Alpe Adria collapse: BayernLB, nationalisation, Addiko (Heta is in the Hypo article)
    "Hypo Alpe-Adria-Bank International",
    "BayernLB",
    "Addiko Bank",
    # Regional banks
    "Oberbank",
    "Volksbank Wien",
    # Regulators
    "Oesterreichische Nationalbank",
    "Financial Market Authority (Austria)",
    "European Central Bank",
    "Single Supervisory Mechanism",
]


def slugify(title: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")


def to_markdown(title: str, extract: str) -> str:
    """'== History ==' -> '## History'; drop reference-style sections; cut to MAX_CHARS."""
    lines = [f"# {title}", ""]
    skipping = False
    for line in extract.split("\n"):
        heading = re.match(r"^(={2,6})\s*(.+?)\s*\1\s*$", line)
        if heading:
            level, name = len(heading.group(1)), heading.group(2)
            if level == 2:  # a top-level section decides whether its subsections are kept
                skipping = name.lower() in SKIP_SECTIONS
            if not skipping:
                lines.append("#" * level + " " + name)
            continue
        if not skipping:
            lines.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(text) > MAX_CHARS:
        # cut at a paragraph boundary, never mid-sentence
        text = text[: text.rfind("\n\n", 0, MAX_CHARS)]
    return text + "\n"


def fetch(client: httpx.Client, title: str) -> dict | None:
    params = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "extracts|revisions|info", "explaintext": 1, "exsectionformat": "wiki",
        "rvprop": "ids", "inprop": "url", "titles": title,
    }
    for attempt in range(5):
        response = client.get(API, params=params)
        if response.status_code != 429:
            break
        wait = int(response.headers.get("Retry-After", 2 ** (attempt + 2)))
        print(f"  rate limited, waiting {wait}s", file=sys.stderr)
        time.sleep(wait)
    page = response.raise_for_status().json()["query"]["pages"][0]
    if page.get("missing"):
        return None
    return {
        "requested": title,
        "title": page["title"],
        "url": page["fullurl"],
        "revision_id": page["revisions"][0]["revid"],
        "license": "CC BY-SA 4.0",
        "text": page["extract"],
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sources, missing = [], []
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        for title in TITLES:
            page = fetch(client, title)
            if page is None:
                missing.append(title)
                continue
            slug = slugify(page["title"])
            if any(s["doc"] == slug for s in sources):  # two requested titles redirect to one page
                continue
            markdown = to_markdown(page["title"], page.pop("text"))
            (RAW_DIR / f"{slug}.md").write_text(markdown, encoding="utf-8")
            sources.append({"doc": slug, **page, "chars": len(markdown)})
            print(f"{slug:45} {len(markdown):>6} chars")
            time.sleep(1.0)  # be polite to the public API (it answers bursts with 429)
    (RAW_DIR / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False))
    total = sum(s["chars"] for s in sources)
    print(f"\n{len(sources)} documents, {total:,} chars (~{total // 4:,} tokens)")
    if missing:
        print(f"Not found: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
