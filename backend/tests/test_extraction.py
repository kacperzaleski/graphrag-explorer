import asyncio
import json

import pytest
from openai import InternalServerError, RateLimitError

from backend.chunking import Chunk
from backend.config import Endpoint
from backend.extraction import extract, validate_items
from backend.llm import LLM

CHUNK = Chunk("bawag#001", "bawag", "BAWAG", "History",
              "The Austrian Trade Union Federation (ÖGB) retained 70% of the shares.")


def ent(name, type_="BANK"):
    return {"name": name, "type": type_, "description": ""}


def rel(source, target, evidence="retained 70% of the shares", type_="OWNS"):
    return {"source": source, "type": type_, "target": target, "evidence": evidence}


def test_valid_items_are_kept():
    raw = {"entities": [ent("ÖGB", "GOVERNMENT"), ent("BAWAG")],
           "relations": [rel("ÖGB", "BAWAG")]}
    result = validate_items(CHUNK, raw)
    assert len(result.entities) == 2 and len(result.relations) == 1 and result.dropped == {}


def test_invalid_items_are_dropped_individually():
    raw = {
        "entities": [ent("ÖGB", "GOVERNMENT"), ent("BAWAG"), ent("XY", "SPACESHIP")],
        "relations": [
            rel("ÖGB", "BAWAG"),                                 # ok
            rel("ÖGB", "BAWAG", type_="LOVES"),                  # not an allowed type
            rel("ÖGB", "Cerberus"),                              # endpoint not extracted
            rel("ÖGB", "BAWAG", evidence="owned 100% forever"),  # quote not in the text
            rel("BAWAG", "BAWAG"),                               # self loop
        ],
    }
    result = validate_items(CHUNK, raw)
    assert [e.name for e in result.entities] == ["ÖGB", "BAWAG"]
    assert len(result.relations) == 1
    assert result.dropped == {"invalid_entity": 1, "invalid_relation": 1, "unknown_endpoint": 1,
                              "ungrounded_evidence": 1, "self_loop": 1}


@pytest.mark.parametrize("evidence", [
    "RETAINED  70%\nof the shares",                  # case and whitespace
    "ÖGB) retained ... of the shares",               # elision
    "BAWAG | History The Austrian Trade Union Federation (ÖGB) retained 70%",  # header + line
])
def test_evidence_grounding_is_word_based(evidence):
    raw = {"entities": [ent("ÖGB"), ent("BAWAG")],
           "relations": [rel("ÖGB", "BAWAG", evidence=evidence)]}
    assert len(validate_items(CHUNK, raw).relations) == 1


def test_garbled_names_are_dropped():
    # the model sometimes drops umlauts: " sterr ischische Gewerkschaftsbund" is not in the text
    raw = {"entities": [ent(" sterr ischische Gewerkschaftsbund"), ent("ÖGB")], "relations": []}
    result = validate_items(CHUNK, raw)
    assert [e.name for e in result.entities] == ["ÖGB"]
    assert result.dropped == {"ungrounded_name": 1}


class FakeLLM(LLM):
    """LLM with a scripted _request: no network."""

    def __init__(self, replies, **kwargs):
        super().__init__(Endpoint("gemini", "http://fake", "main-model", "key",
                                  fallback_models=("backup-model",)), rpm=6000, **kwargs)
        self.replies = list(replies)
        self.models_called = []

    async def _request(self, model, messages, response_format):
        self.models_called.append(model)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, None


def api_error(cls, status):
    import httpx

    request = httpx.Request("POST", "http://fake")
    return cls(str(status), response=httpx.Response(status, request=request), body=None)


def test_non_json_reply_is_skipped_not_fatal():
    result = asyncio.run(extract(FakeLLM(["sorry, I can't"]), CHUNK))
    assert result.entities == [] and result.dropped == {"invalid_json": 1}


@pytest.mark.parametrize("error", [api_error(InternalServerError, 503),  # overloaded
                                   api_error(RateLimitError, 429)])     # daily quota used up
def test_fallback_model_and_cache(tmp_path, error):
    from backend.cache import DiskCache

    reply = json.dumps({"entities": [ent("BAWAG")], "relations": []})
    llm = FakeLLM([error, reply], cache=DiskCache("llm", root=tmp_path))
    result = asyncio.run(extract(llm, CHUNK))
    assert llm.models_called == ["main-model", "backup-model"]
    assert [e.name for e in result.entities] == ["BAWAG"]

    # second run: answered from the cache (stored under the fallback model), no request at all
    again = FakeLLM([], cache=DiskCache("llm", root=tmp_path))
    assert [e.name for e in asyncio.run(extract(again, CHUNK)).entities] == ["BAWAG"]
    assert again.models_called == [] and again.usage.cache_hits == 1
