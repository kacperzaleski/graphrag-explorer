"""Split a markdown document into heading-aware, overlapping chunks.

Adapted from my earlier RAG demo (minus its table handling; Wikipedia extracts have no tables):
- Every chunk remembers its document title and heading path ("BAWAG | History > Sale to Cerberus").
  That prefix is part of the chunk text, so the embedding and the extraction LLM know which bank
  "the bank" refers to.
- Sentences are packed up to a token budget. The next chunk starts by repeating the previous
  chunk's last sentences (up to `overlap_tokens`), so a fact that spans a chunk border is
  complete in at least one chunk.
- Chunks never cross a section boundary: sections are topical units.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# split after . ! ? when the next sentence starts with a capital, digit or quote; imperfect on
# abbreviations ("Dr. Karl Renner"), which only costs a slightly odd split point
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(„])")

TokenCounter = Callable[[str], int]


def approx_tokens(text: str) -> int:
    """~4 characters per token for English. Good enough for a budget: we have no local tokenizer
    for API models, and the embedding model accepts far more than 512 tokens anyway."""
    return len(text) // 4 + 1


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc>#<n>": deterministic, so re-ingesting MERGEs onto the same node
    doc: str
    title: str
    section: str  # heading path below the title, e.g. "History > Sale to Cerberus"
    body: str

    @property
    def text(self) -> str:
        """What is embedded, sent to extraction and shown to the answering LLM."""
        header = f"{self.title} | {self.section}" if self.section else self.title
        return f"{header}\n\n{self.body}"


def chunk_document(doc: str, markdown: str, max_tokens: int = 512, overlap_tokens: int = 64,
                   count: TokenCounter = approx_tokens) -> list[Chunk]:
    title, sections = _sections(markdown)
    chunks: list[Chunk] = []
    for section, paragraphs in sections:
        prefix = count(f"{title} | {section}\n\n")
        for body in _pack(paragraphs, max_tokens - prefix, overlap_tokens, count):
            chunks.append(Chunk(f"{doc}#{len(chunks):03d}", doc, title, section, body))
    return chunks


def _sections(markdown: str) -> tuple[str, list[tuple[str, list[str]]]]:
    """(title, [(heading path, paragraphs)]) using a heading stack; "#" is the document title."""
    title = ""
    sections: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []  # open headings below the title, as (level, name)
    paragraphs: list[str] = []

    def flush() -> None:
        if paragraphs:
            sections.append((" > ".join(name for _, name in stack), paragraphs.copy()))
        paragraphs.clear()

    for line in markdown.split("\n"):
        heading = _HEADING.match(line)
        if heading and len(heading.group(1)) == 1:
            title = heading.group(2)
        elif heading:
            flush()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2)))
        elif line.strip():  # Wikipedia extracts: one paragraph per line
            paragraphs.append(line.strip())
    flush()
    return title, sections


def _pack(paragraphs: list[str], budget: int, overlap: int, count: TokenCounter) -> list[str]:
    """Greedy sentence packing with sentence-level overlap. A single sentence longer than the
    budget becomes its own (oversized) chunk rather than being cut mid-sentence."""
    units = [(i, s) for i, p in enumerate(paragraphs) for s in _SENTENCE_END.split(p)]
    groups: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    used = 0
    for unit in units:
        size = count(unit[1]) + 1
        if current and used + size > budget:
            groups.append(current)
            current, used = _tail(current, overlap, count)
        current.append(unit)
        used += size
    if current:
        groups.append(current)
    return [_render(group) for group in groups]


def _tail(units: list[tuple[int, str]], overlap: int, count: TokenCounter):
    """The last sentences of a chunk that fit in the overlap budget."""
    tail: list[tuple[int, str]] = []
    used = 0
    for unit in reversed(units):
        size = count(unit[1]) + 1
        if used + size > overlap:
            break
        tail.insert(0, unit)
        used += size
    return tail, used


def _render(units: list[tuple[int, str]]) -> str:
    """Sentences of one paragraph joined by spaces, paragraphs by newlines."""
    lines: list[str] = []
    last = None
    for paragraph, sentence in units:
        if paragraph == last:
            lines[-1] += " " + sentence
        else:
            lines.append(sentence)
        last = paragraph
    return "\n".join(lines)
