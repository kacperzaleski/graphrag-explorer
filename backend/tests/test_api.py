"""API tests with fakes injected via dependency_overrides: no Neo4j, no LLM, no model download.
TestClient is used without `with`, so the lifespan (DB connection, reranker warm-up) never runs."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from backend import api
from backend.answer import AnswerResult
from backend.config import Endpoint
from backend.pipeline import AskResult
from backend.search import Hit

SUBGRAPH = {
    "seeds": ["bawag group"], "truncated": False,
    "nodes": [{"key": "bawag group", "name": "BAWAG Group", "type": "BANK", "degree": 35, "hop": 0},
              {"key": "cerberus", "name": "Cerberus", "type": "COMPANY", "degree": 42, "hop": 1}],
    "edges": [{"id": "cerberus|OWNS|bawag group", "source": "cerberus", "target": "bawag group",
               "type": "OWNS", "chunk_ids": ["bawag#005"], "on_path": False, "used": False}],
}
CHUNK = {"id": "bawag#005", "doc": "bawag", "title": "BAWAG", "section": "History",
         "url": "https://en.wikipedia.org/wiki/BAWAG", "text": "Cerberus bought 90% of BAWAG."}


class FakeGraph:
    def __init__(self):
        self.calls = []

    async def neighbors(self, key, hops, limit):
        self.calls.append((key, hops, limit))
        return SUBGRAPH if key == "bawag group" else None

    async def chunks(self, ids):
        return [CHUNK for i in ids if i == CHUNK["id"]]

    async def search_entities(self, q, limit):
        return [{"key": "bawag group", "name": "BAWAG Group", "type": "BANK", "degree": 35,
                 "description": "Austrian bank", "score": 3.2}]


class FakePipeline:
    llm = type("L", (), {"endpoint": Endpoint("gemini", "x", "fake-model", "k")})()

    async def ask(self, question, mode, hops):
        answer = AnswerResult("Cerberus owns BAWAG [bawag#005].", True, ["bawag#005"],
                              ["cerberus|OWNS|bawag group"], [], "fake-model", False, 100, 20)
        return AskResult(question, mode, answer, [Hit("bawag#005", 1.0, CHUNK["text"])],
                         SUBGRAPH, {"total": 12.0})


class DownDriver:
    async def verify_connectivity(self):
        raise ConnectionError("down")


@pytest.fixture
def client():
    graph = FakeGraph()
    api.app.dependency_overrides = {
        api.get_graph: lambda: graph,
        api.get_pipeline: lambda: FakePipeline(),
        api.get_ask_slots: lambda: asyncio.Semaphore(1),
        api.get_driver: lambda: DownDriver(),
    }
    yield TestClient(api.app), graph
    api.app.dependency_overrides = {}


def test_health(client):
    c, _ = client
    response = c.get("/health")
    assert response.status_code == 200 and response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]  # every response carries a request id


def test_request_id_is_propagated(client):
    c, _ = client
    assert c.get("/health", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"


def test_ready_reports_neo4j_down_as_503(client):
    c, _ = client
    response = c.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "degraded"


def test_neighbors(client):
    c, graph = client
    response = c.get("/graph/neighbors", params={"node_id": "bawag group", "hops": 1})
    assert response.status_code == 200
    body = response.json()
    assert {n["name"] for n in body["nodes"]} == {"BAWAG Group", "Cerberus"}
    assert body["edges"][0]["type"] == "OWNS"
    assert graph.calls == [("bawag group", 1, 30)]  # default limit applied


def test_neighbors_unknown_node_is_404(client):
    c, _ = client
    assert c.get("/graph/neighbors", params={"node_id": "nope"}).status_code == 404


@pytest.mark.parametrize("params", [
    {},                                          # node_id missing
    {"node_id": "bawag group", "hops": 0},       # below minimum
    {"node_id": "bawag group", "hops": 9},       # above MAX_HOPS
    {"node_id": "bawag group", "limit": 5000},   # above limit cap
])
def test_neighbors_validation_errors(client, params):
    c, graph = client
    assert c.get("/graph/neighbors", params=params).status_code == 422
    assert graph.calls == []  # invalid input never reaches the database


@pytest.mark.parametrize("body", [
    {"question": "hi"},                                   # too short
    {"question": "x" * 501},                              # too long
    {"question": "Who owns BAWAG?", "mode": "magic"},     # unknown mode
    {"question": "Who owns BAWAG?", "hops": 7},           # too many hops
])
def test_ask_validation_errors(client, body):
    c, _ = client
    response = c.post("/ask", json=body)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][0] == "body"  # precise error location


def test_ask_returns_answer_citations_and_subgraph(client):
    c, _ = client
    response = c.post("/ask", json={"question": "Who owns BAWAG?", "mode": "graph"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == [{"chunk_id": "bawag#005", **{k: v for k, v in CHUNK.items()
                                                              if k != "id"}}]
    assert body["subgraph"]["nodes"][0]["key"] == "bawag group"
    assert body["usage"] == {"model": "fake-model", "cached": False, "prompt_tokens": 100,
                             "completion_tokens": 20}


def test_search(client):
    c, _ = client
    assert c.get("/graph/search", params={"q": "bawa"}).json()[0]["name"] == "BAWAG Group"
    assert c.get("/graph/search", params={"q": ""}).status_code == 422
