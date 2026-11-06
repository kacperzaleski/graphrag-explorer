# Presentation plan (about 20 minutes + Q&A)

**Storyline:** "I built a tool where people can *discover* how Austrian banks, owners and regulators are
connected, with every fact traceable to its source. Then I measured honestly whether the graph also makes
question answering better. Mostly it doesn't, and I can explain why."

Order: **demo first** (2 min of slides for context, then the product), then how it's built, then the evaluation.
Reason: the goal is visual exploration, so show the product before the plumbing. The evaluation is the strongest
data-science part, so it gets its own block at the end, where questions will start anyway.

## 0. Before the presentation (checklist)
- `docker compose up -d neo4j`, `uv run uvicorn backend.api:app --no-access-log`, `cd frontend && npm run dev`
- Run `uv run python -m scripts.warm_demo`: asks all demo questions once, so the answers come **from the cache**
  (instant, identical to the rehearsal, no quota risk during the talk).
- Browser zoom ~110%, graph cleared, Insights tab open.
- Backup: the screenshots in `slides/img/` are in the deck in case the live demo fails.

## 1. Slides 1–3: context (2 min)
Goal, stack (same path as the BAWAG team: documents → knowledge graph → hybrid retrieval → FastAPI → React),
corpus (22 Wikipedia articles on Austrian banks).

## 2. Live demo (6–7 min): "discover, then ask"
| # | Action | What to say |
|---|---|---|
| 1 | **Insights tab**: show "Most influential owners", then *colour by community* (toggle in the graph legend) after loading a few nodes | "Graph analytics computed at ingest: PageRank over ownership, betweenness for bridges, Louvain communities. A starting point when you know nothing." |
| 2 | **Search "BAWAG"**, then the entity card | Scroll the connections: *BayernLB 46% (1995) → ÖGB 100% (2004) → Cerberus 90% (2006) → GoldenTree 21.8%*. "Did you know BayernLB owned almost half of BAWAG in the 90s? Every line has its source quote." |
| 3 | Click **Cerberus** in the graph (expand) | Expand-on-click; node limit and warning; size = connectivity. |
| 4 | **Connect tab**: *Austrian Trade Union Federation* ↔ *GoldenTree* | "Both owned BAWAG, at different times. Shortest path, no LLM, instant." |
| 5 | Connect: *Refco* ↔ *Cerberus Capital Management* | "The Refco scandal hit BAWAG, which was then sold to Cerberus." |
| 6 | Connect: *Česká spořitelna* ↔ *Slovenská sporiteľňa* | "Siblings: both owned by Erste Group." |
| 7 | **Ask (graph)**: "Who have been the owners of BAWAG over time?" | Answer with citations; the edges used are highlighted in red, the rest faded. |
| 8 | Ask (graph): "How is BayernLB connected to Addiko Bank?" | Multi-hop via Hypo Alpe Adria; open a source passage. Note: the ECB SUPERVISES edges are highlighted too (the path finder found them), because the graph lacks "Addiko spun off from Hypo": the answer comes from the passages. Good lead-in to step 10. |
| 9 | *Optional:* the same question in **vector** mode | "Plain RAG often gets it right too: that's the honest result of my evaluation." |
| 10 | *Limitation, on purpose:* Connect *BayernLB* ↔ *Addiko Bank* | The path finder shows "both supervised by the ECB", because the extraction **missed** "Addiko was spun off from Hypo". "The graph is only as good as the extraction, so I report it." |

Questions that are **verified correct** in graph mode (from the eval runs), as reserves:
- "Which Munich-based bank bought the shares of the shareholder Konsum after it went bankrupt in 1995?" → BayernLB
- "Which bank merged on October 1, 2005, with the postal savings bank that was owned by the Austrian Post Office?" → BAWAG
- "How is Česká spořitelna connected to Slovenská Sporiteľňa?" → both Erste Group
- "Which American company did BAWAG buy the piano manufacturer Bösendorfer from in 2001?" → Kimball International

Avoid live: corpus-wide questions ("Which banks does the ECB supervise?") give only partial answers (a known limitation, slides 11 and 13).

## 3. Slides 5–9: how it's built (6 min)
Pipeline, extraction with validation, entity resolution (Cerberus/ÖGB story), retrieval (RRF, paths, reserved
slots), discovery features (time on edges, analytics, SIA story), engineering for unreliable free-tier APIs.
For each slide: **one decision and one thing that went wrong**. Interviewers remember the stories.

## 4. Slides 10–13: evaluation (5 min): the data-science core
1. Design: questions from the **text, not the graph** (no leakage); judge from another vendor; paired statistics.
2. Round 1: no difference, and the **ceiling effect** explains why.
3. Round 2: pre-registered hard test, H1 **not supported**; the reranker hurts connection questions.
4. Takeaway: where the graph helps (exploration, explainability, provenance) and where it doesn't (QA accuracy
   on a small, well-written corpus).

## 5. Slide 14: next steps (1 min), then Q&A
Prepare for: "Why not Microsoft GraphRAG?", "How would this scale?", "Why would a bank trust it?",
"What would you do with 2 more weeks?" (answers in `NOTES.md` and `docs/INTERVIEW_NOTES.md`).
