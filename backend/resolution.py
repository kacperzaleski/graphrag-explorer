"""Entity resolution: decide which extracted names are the same real-world entity.

Strategy (deliberately conservative; in banking a false merge invents ownership that does not
exist, while a missed merge only loses a connection):
1. normalize(): fold accents and case, drop punctuation and legal-form suffixes (AG, GmbH, plc...).
   "Erste Group Bank AG" and "erste group bank" get the same key.
2. A curated alias file (data/aliases.yaml) maps known variants to one canonical name, e.g.
   "BAWAG P.S.K." -> "BAWAG Group". Every merge beyond step 1 is explicit and reviewable.
3. The key ignores the entity type: the LLM may call the same bank BANK in one chunk and COMPANY
   in another. The type is decided by majority vote over all mentions.

Known limits: no fuzzy matching, so spelling variants not in the alias file stay separate; two
different entities with the same normalized name (rare here) would be merged.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from backend.config import REPO_ROOT
from backend.extraction import ChunkExtraction

ALIASES_FILE = REPO_ROOT / "data" / "aliases.yaml"

# legal forms, after punctuation removal and joining of initials ("S.p.A." -> "spa", "a.s." -> "as")
LEGAL_SUFFIXES = {"ag", "gmbh", "mbh", "plc", "spa", "sa", "se", "nv", "bv", "inc", "ltd", "llc",
                  "kg", "as", "dd", "ad", "ab", "oyj", "corp", "corporation", "limited", "lp",
                  "jsc", "aktiengesellschaft", "zrt", "ead", "srl"}


# German umlauts are often written transliterated ("Oesterreich"): map them the same way before
# folding the remaining accents ("Česká" -> "ceska")
_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def normalize_text(text: str) -> str:
    """Lowercase, umlauts transliterated, other accents folded: "Österreich" -> "oesterreich"."""
    text = text.lower().translate(_UMLAUTS)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def normalize(name: str) -> str:
    text = normalize_text(name)
    text = text.replace("&", " and ")
    tokens = re.sub(r"[^a-z0-9]+", " ", text).split()
    tokens = _join_initials(tokens)
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:  # never strip the whole name
        tokens = tokens[:-1]
    return " ".join(tokens)


def _join_initials(tokens: list[str]) -> list[str]:
    """['bawag', 'p', 's', 'k'] -> ['bawag', 'psk'] so that "P.S.K." equals "PSK"."""
    out: list[str] = []
    run = ""
    for token in [*tokens, ""]:
        if len(token) == 1:
            run += token
            continue
        if run:
            out.append(run)
            run = ""
        if token:
            out.append(token)
    return out


class Resolver:
    def __init__(self, aliases: dict[str, list[str]] | None = None):
        # normalized variant -> canonical display name
        self.canonical: dict[str, str] = {}
        for canonical, variants in (aliases or {}).items():
            for variant in [canonical, *(variants or [])]:
                self.canonical[normalize(variant)] = canonical

    @classmethod
    def from_file(cls, path: Path = ALIASES_FILE) -> "Resolver":
        if not path.exists():
            return cls()
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def key(self, name: str) -> str:
        norm = normalize(name)
        return normalize(self.canonical[norm]) if norm in self.canonical else norm

    def alias_name(self, name: str) -> str | None:
        return self.canonical.get(normalize(name))


@dataclass
class EntityRecord:
    key: str
    names: Counter = field(default_factory=Counter)  # surface forms as written in the chunks
    types: Counter = field(default_factory=Counter)
    descriptions: list[str] = field(default_factory=list)
    chunk_ids: set[str] = field(default_factory=set)
    canonical: str | None = None  # from the alias file

    @property
    def name(self) -> str:
        if self.canonical:
            return self.canonical
        # most frequent surface form; ties -> the longer (more complete) one
        return max(self.names, key=lambda n: (self.names[n], len(n)))

    @property
    def type(self) -> str:
        # majority vote; ties -> anything is better than OTHER, then alphabetical (deterministic)
        return max(self.types, key=lambda t: (self.types[t], t != "OTHER", t))

    @property
    def description(self) -> str:
        return max(self.descriptions, key=len, default="")


@dataclass
class RelationRecord:
    source: str
    type: str
    target: str
    evidence: dict[str, str] = field(default_factory=dict)  # chunk id -> quote
    # set by the enrichment pass (OWNS / ACQUIRED / MERGED_WITH only)
    start_year: int | None = None
    end_year: int | None = None
    stake_percent: float | None = None

    @property
    def id(self) -> str:
        return f"{self.source}|{self.type}|{self.target}"


@dataclass
class ResolvedGraph:
    entities: dict[str, EntityRecord]
    relations: dict[str, RelationRecord]
    mentions: set[tuple[str, str]]  # (chunk id, entity key)


def resolve(extractions: list[ChunkExtraction], resolver: Resolver) -> ResolvedGraph:
    """Merge per-chunk extractions into one graph keyed by resolved entity keys."""
    entities: dict[str, EntityRecord] = {}
    relations: dict[str, RelationRecord] = {}
    mentions: set[tuple[str, str]] = set()

    for ex in extractions:
        for e in ex.entities:
            key = resolver.key(e.name)
            record = entities.setdefault(key, EntityRecord(key))
            record.names[e.name] += 1
            record.types[e.type] += 1
            record.canonical = record.canonical or resolver.alias_name(e.name)
            if e.description:
                record.descriptions.append(e.description)
            record.chunk_ids.add(ex.chunk_id)
            mentions.add((ex.chunk_id, key))

        for r in ex.relations:
            source, target = resolver.key(r.source), resolver.key(r.target)
            if source == target:  # e.g. "BAWAG MERGED_WITH BAWAG PSK" once aliases merge them
                continue
            if r.type == "MERGED_WITH" and target < source:  # symmetric: store one direction only
                source, target = target, source
            record = RelationRecord(source, r.type, target)
            record = relations.setdefault(record.id, record)
            record.evidence.setdefault(ex.chunk_id, r.evidence)
    return ResolvedGraph(entities, relations, mentions)


# first words shared by unrelated names ("European Central Bank" / "European Commission")
_GENERIC_FIRST_WORDS = {"austrian", "european", "national", "federal", "international", "first",
                        "bank", "central", "new", "united", "general", "state", "world", "global"}


def alias_candidates(entities: dict[str, EntityRecord]) -> list[list[EntityRecord]]:
    """Groups of entities whose keys share the first word: likely variants to review by hand
    ("bawag", "bawag psk", "bawag group"). Only a suggestion; nothing is merged automatically."""
    groups: dict[str, list[EntityRecord]] = {}
    for record in entities.values():
        first = record.key.split()[0]
        if len(first) >= 3 and first not in _GENERIC_FIRST_WORDS:
            groups.setdefault(first, []).append(record)
    return [sorted(g, key=lambda r: -len(r.chunk_ids)) for g in groups.values() if len(g) > 1]
