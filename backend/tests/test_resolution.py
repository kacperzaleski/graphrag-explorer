import pytest

from backend.extraction import ChunkExtraction, ExtractedEntity, ExtractedRelation
from backend.resolution import Resolver, normalize, resolve


@pytest.mark.parametrize(("name", "key"), [
    ("Erste Group Bank AG", "erste group bank"),
    ("ERSTE GROUP BANK", "erste group bank"),
    ("Česká spořitelna, a.s.", "ceska sporitelna"),
    ("Raiffeisen Zentralbank Österreich AG", "raiffeisen zentralbank oesterreich"),
    ("Raiffeisen Zentralbank Oesterreich Aktiengesellschaft", "raiffeisen zentralbank oesterreich"),
    ("UniCredit S.p.A.", "unicredit"),
    ("BAWAG P.S.K.", "bawag psk"),
    ("BAWAG PSK", "bawag psk"),
    ("The European Central Bank", "european central bank"),
    ("AG", "ag"),  # never strip a name down to nothing
])
def test_normalize(name, key):
    assert normalize(name) == key


def test_alias_file_merges_variants():
    resolver = Resolver({"BAWAG Group": ["BAWAG", "BAWAG P.S.K."]})
    assert resolver.key("BAWAG PSK") == resolver.key("bawag") == "bawag group"
    assert resolver.alias_name("Bawag P.S.K.") == "BAWAG Group"
    assert resolver.key("Raiffeisen Bank International") == "raiffeisen bank international"


def entity(name, type_="BANK"):
    return ExtractedEntity(name=name, type=type_, description=f"about {name}")


def relation(source, type_, target):
    return ExtractedRelation(source=source, type=type_, target=target, evidence="some quote")


def test_resolve_majority_type_and_aliases():
    resolver = Resolver({"BAWAG Group": ["BAWAG PSK"]})
    extractions = [
        ChunkExtraction("a#000", [entity("BAWAG"), entity("ÖGB", "GOVERNMENT")],
                        [relation("ÖGB", "OWNS", "BAWAG")]),
        ChunkExtraction("b#000", [entity("BAWAG PSK", "COMPANY"), entity("BAWAG Group")], []),
        ChunkExtraction("c#000", [entity("Bawag Group")], []),
    ]
    graph = resolve(extractions, Resolver())
    assert {"bawag", "bawag psk", "bawag group"} <= set(graph.entities)  # without aliases: split

    graph = resolve(extractions, resolver)
    bawag = graph.entities["bawag group"]
    assert bawag.name == "BAWAG Group"
    assert bawag.type == "BANK"  # majority vote: 2x BANK vs 1x COMPANY
    assert bawag.chunk_ids == {"b#000", "c#000"}
    assert ("a#000", "oegb") in graph.mentions
    [owns] = graph.relations.values()
    assert (owns.source, owns.type, owns.target) == ("oegb", "OWNS", "bawag")
    assert owns.evidence == {"a#000": "some quote"}


def test_merged_with_is_stored_once_and_self_loops_vanish():
    resolver = Resolver({"BAWAG Group": ["BAWAG PSK"]})
    extractions = [
        ChunkExtraction("a#000", [entity("PSK"), entity("BAWAG")],
                        [relation("PSK", "MERGED_WITH", "BAWAG")]),
        ChunkExtraction("b#000", [entity("PSK"), entity("BAWAG")],
                        [relation("BAWAG", "MERGED_WITH", "PSK")]),
        ChunkExtraction("c#000", [entity("BAWAG PSK"), entity("BAWAG Group")],
                        [relation("BAWAG PSK", "RELATED_TO", "BAWAG Group")]),
    ]
    graph = resolve(extractions, resolver)
    assert list(graph.relations) == ["bawag|MERGED_WITH|psk"]
    assert set(graph.relations["bawag|MERGED_WITH|psk"].evidence) == {"a#000", "b#000"}
