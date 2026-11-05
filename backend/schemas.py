"""API request/response models (Pydantic v2).

They are the API's contract: FastAPI validates every request against them (422 with a precise
error on bad input), filters every response through them (no accidental field leaks), and
generates the OpenAPI schema at /docs from them. The React client's TypeScript types mirror them.
"""

from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["BANK", "COMPANY", "PERSON", "REGULATOR", "GOVERNMENT", "LOCATION", "EVENT",
                     "OTHER"]

# --- input limits: bound cost and latency per request, and the blast radius of abuse -----------
MAX_QUESTION_CHARS = 500
MAX_HOPS = 3


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS,
                          examples=["How is the Austrian Trade Union Federation connected to "
                                    "GoldenTree?"])
    mode: Literal["vector", "graph"] = "graph"
    hops: int = Field(default=2, ge=1, le=MAX_HOPS)


class Citation(BaseModel):
    chunk_id: str
    doc: str
    title: str
    section: str
    url: str | None = None
    text: str


class GraphNode(BaseModel):
    key: str
    name: str
    type: str
    degree: int | None = None
    hop: int = 0
    community: int | None = None


class GraphEdge(BaseModel):
    id: str
    source: str  # entity keys
    target: str
    type: str
    chunk_ids: list[str] = []
    on_path: bool = False
    used: bool = False  # the answer relied on this edge
    start_year: int | None = None  # enrichment pass, ownership-like edges only
    end_year: int | None = None
    stake_percent: float | None = None


class Subgraph(BaseModel):
    seeds: list[str] = []
    truncated: bool = False  # node limit reached: the UI shows a warning
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []


class Usage(BaseModel):
    model: str
    cached: bool
    prompt_tokens: int
    completion_tokens: int


class AskResponse(BaseModel):
    question: str
    mode: str
    answer: str
    found_in_context: bool
    citations: list[Citation]  # passages the answer cites, in citation order
    passages: list[str]  # ids of all passages the LLM saw
    issues: list[str]  # verification problems (e.g. removed invented citations)
    subgraph: Subgraph
    timings_ms: dict[str, float]
    usage: Usage


class EntityHit(BaseModel):
    key: str
    name: str
    type: str
    degree: int | None = None
    description: str | None = None
    score: float


class Relation(BaseModel):
    id: str
    type: str
    outgoing: bool  # True: this entity is the source
    other_key: str
    other_name: str
    other_type: str
    chunk_ids: list[str]
    evidence: list[str]
    start_year: int | None = None
    end_year: int | None = None
    stake_percent: float | None = None


class Passage(BaseModel):
    id: str
    doc: str
    title: str
    section: str
    url: str | None = None
    text: str


class NodeDetails(BaseModel):
    key: str
    name: str
    type: str
    description: str | None = None
    aliases: list[str] = []
    degree: int | None = None
    mention_count: int | None = None
    relations: list[Relation]
    chunks: list[Passage]


class RankedEntity(BaseModel):
    key: str
    name: str
    type: str
    degree: int | None = None
    owner_rank: float | None = None
    group_size: int | None = None
    betweenness: float | None = None
    community: int | None = None


class CommunityMember(BaseModel):
    key: str
    name: str
    type: str


class Community(BaseModel):
    id: int
    size: int
    members: list[CommunityMember]  # most connected first; the first one names the community


class Insights(BaseModel):
    top_owners: list[RankedEntity]  # PageRank over reversed ownership edges
    largest_groups: list[RankedEntity]  # entities reachable via OWNS chains
    bridges: list[RankedEntity]  # betweenness centrality
    communities: list[Community]  # Louvain


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    neo4j: str  # server version, or the error
    llm: str  # provider/model used for answers
