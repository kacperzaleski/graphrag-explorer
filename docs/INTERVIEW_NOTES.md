# Interview cheat sheet

One page to read before the interview. The details and reasoning are in `NOTES.md`, the talk order is in
`slides/PRESENTATION_PLAN.md`.

## 30-second pitch
"I built a tool that turns 22 Wikipedia articles on Austrian banks into a knowledge graph, so people can
*discover* connections they didn't know about, like BayernLB owning 46% of BAWAG in the 90s, with every fact
linked to its source sentence. Then I measured honestly whether the graph also improves question answering
compared with plain vector RAG. On this corpus it doesn't, measurably, and I can show why: a ceiling effect,
and a reranker that hurts connection questions. The graph's value is exploration, provenance and explainability."

## Numbers to know
| | |
|---|---|
| Corpus | 22 articles, chunks of 512 tokens with 64 overlap, about 160 extraction calls |
| Graph | 726 entities, 585 relations; 8 entity types, 9 relation types |
| Retrieval | RRF k = 60, 30 candidates, top 8 passages, 2 hops, max 40 nodes |
| Main eval (n = 28) | vector 0.91 vs graph 0.93, diff +0.02 [0.00, +0.05] |
| Hard set (n = 8) | vector 0.38 vs graph 0.44, diff +0.06 [−0.12, +0.25], H1 not supported |
| Recall@8, hard set | vector 0.56 → **0.81 without reranker** |
| Judge agreement | κ = 0.62 (main), 0.69 (hard) with a second judge from another model family |
| Tests | 77 (pytest), no network: fake LLM, dependency overrides |

## The 5 stories (one decision + one failure each)
1. **Grounding.** Exact-substring checks rejected about 70 valid quotes, so I switched to word-level support
   (≥ 0.9). The lesson: validators need their own error analysis.
2. **Entity resolution.** "Cerberus" and "ÖGB" appeared under several names, plus a typo ("Ceberus"), which
   broke the ownership chain. Rules handle the easy merges; a reviewed alias file handles the rest, because a
   wrong merge is worse than a missed one.
3. **Hubs.** Paths ran through "New York" and connected everything to everything. A degree threshold blocked
   the main banks instead, because here the high-degree nodes *are* the answer. Fix: the node *type* is the signal
   (LOCATION is never expanded or used as a path step); degree is only a safety net (100).
4. **Analytics bug.** A wrong ACQUIRED edge made SIA a "top owner". PageRank now uses OWNS only, and I spotted
   the bug only because the result looked implausible: sanity-check the output.
5. **Evaluation.** Round 1 showed no difference because of a ceiling effect. I pre-registered a harder test
   (with one disclosed amendment, made before seeing results). The hypothesis was not supported, and I report
   that instead of tuning on the test set.

## Likely questions, short answers
- **Why not Microsoft GraphRAG?** It is built for global "summarize the corpus" questions via community
  summaries, which means many LLM calls and is expensive on a free tier. My use case is entity-centric
  exploration and traceability. Own code is also easier to explain and to control.
- **How would this scale?** Ingestion is per chunk and idempotent (MERGE), so it parallelises and can run
  incrementally per document. Neo4j handles millions of nodes. The bottlenecks are LLM cost and entity resolution:
  blocking plus embedding similarity, and human review only for the uncertain cases.
- **Why would a bank trust it?** Every edge has a source quote and chunk ID. Citations are checked, and answers
  that cite unseen passages are corrected. For confidential data: EU-hosted or local models, an audit log, no
  training on the data.
- **Why LLM-as-a-judge?** It's cheap and consistent, but biased. Mitigations: another vendor, a reference
  answer, reasoning before the label, a second judge (κ), and human calibration (pending).
- **Why RRF?** Vector and BM25 scores live on different scales. RRF uses only ranks, so nothing needs tuning
  or normalising.
- **Why paired statistics?** Both modes answer the same questions, so per-question differences remove the
  question difficulty. Bootstrap CI on the difference; McNemar on the discordant pairs.
- **What would you do with 2 more weeks?** A messier corpus, measured extraction recall, a question router
  (graph mode only where it helps), human-written evaluation questions, a timeline filter, CI/CD.

## What I would say about the limits
The graph is only as good as the extraction ("Addiko spun off from Hypo" is missing). Corpus-wide questions need
aggregation in Cypher, not top-k retrieval. n = 8 can't detect small effects. The judge isn't validated against
humans yet.
