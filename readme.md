# Cited Research Agent

**A conversational research agent whose every answer sentence carries a
machine-verifiable citation to a specific chunk and page — and which refuses,
explicitly, when the corpus does not contain the answer.**

> The agent holds a multi-turn conversation about a corpus of research papers. Every
> answer sentence carries a machine-verifiable citation to a specific chunk and page;
> when the corpus does not support an answer it refuses explicitly; when the question
> is ambiguous it asks rather than guesses. Conversation state, citation records, and
> API quota accounting are persisted in SQLite.

> **Build status: Step 0 of 13 (Phase 0 · Foundation).** Sections below marked `TBD`
> are not yet measured. Per the project's own rules, no number appears in this README
> until it traces to a file in `outputs/` produced by a real run — placeholders stay
> `TBD` rather than being estimated.

---

## 1 · What it does

TBD — a pasted terminal transcript lands here at Step 7: a question in, a cited answer
with page numbers out, followed by a refusal example and a clarification example.

## 2 · Quickstart

TBD — five copy-pasteable commands ending in a working answer. **The Ollama path comes
first and needs no API key of any kind**; the Gemini path is second and optional.

## 3 · How it works

TBD — architecture diagram and one paragraph per stage.

The design claim, stated up front: *it decides whether it has enough evidence, acts
again when it does not, asks when the question is ambiguous, refuses when the corpus
is silent, and checks its own output before returning it — that is what makes it an
agent rather than a RAG script.*

## 4 · The citation contract

TBD — the JSON schema, the `turn_citations` table, and why a structured
sentence→chunk mapping is verifiable and queryable where inline prose markers are
neither.

This is the technical thesis of the project. My previous system's README listed
*"passage-level citations require manual parsing of the synthesiser's inline
markers"* as a known limitation. Here citations are a structured contract emitted by
the model, persisted relationally, and independently verified — never regex'd out of
prose. The second consequence is that citations survive the conversation: because
every sentence→chunk link is a database row, a follow-up turn can say "expand on the
second source" and the agent resolves it with a query.

## 5 · Retrieval approach

TBD — hybrid dense + sparse rationale, RRF over weighted score fusion, FTS5 BM25 and
its real tokenisation tradeoff, the cross-encoder, the **measured** thresholds with
the score histogram that produced them, and the five-configuration ablation table.

## 6 · Conversation design

TBD — condensation under a vocabulary constraint and the specific retrieval-drift
failure it prevents, the programmatic drift guard and its measured rate, the turn-1
skip, the history double-cap, and three-way routing with its confusion matrix.

## 7 · Persistence

TBD — the full schema, and the argument for why the quota ledger in particular *must*
be durable: a daily rate limit cannot be enforced from a process that exits between
every request.

## 8 · Evaluation

TBD — metrics, the question and conversation taxonomy, which columns are
LLM-dependent, both ablation tables, and a note on synthesis-model variance across
ladder rungs.

## 9 · Design decisions and tradeoffs

TBD — assembled from [`decisions.md`](decisions.md), which is written as the build
proceeds rather than reconstructed at the end. Includes why Qdrant, Milvus, Chroma,
Redis, Postgres and LangGraph were **not** used despite production experience with all
of them.

## 10 · Free-tier quota engineering

TBD — the two ladders, the `gemini-2.5-flash-lite` trap, the per-turn cost model, and
both cache layers.

## 11 · Hardware notes

TBD — VRAM budget table, the two offload modes, and the silent-CPU-torch fix.

## 12 · Limitations and known failure cases

Stated up front rather than discovered by a reviewer. This list grows as the build
measures things; it does not shrink.

- **No OCR, ever.** The corpus is born-digital arXiv PDFs. A document that does not
  extract as text gets replaced, not OCR'd. This is a permanent scope exclusion, not
  a deferred feature.
- Equations, figures, and multi-page tables extract poorly. Documented limitation,
  not a bug being chased.
- English only.
- Cross-encoder groundedness verification is a **relevance proxy, not trained NLI** —
  and it shares a model family with the reranker, so verifier and retriever are
  correlated by construction. The strength of that correlation will be measured and
  reported rather than glossed.
- Abstention thresholds are tuned, not learned.
- Small golden set. Confidence intervals accompany every reported mean.
- Flat vector scan: fine to roughly 10k chunks, degrades past it.
- Single-user, single-process.

### Permanently out of scope

Listed deliberately — silent omission reads as a gap, stated exclusion reads as
discipline.

- OCR of scanned PDFs
- Docker, Postgres, Redis, Qdrant, Milvus, Chroma, FastAPI, auth, message queues
- Any hosted deployment or public endpoint
- Fine-tuning or training of any model

## 13 · With more time

TBD — Qdrant + HNSW at corpus scale, a trained NLI verifier, learned thresholds,
span-level offsets within a chunk, table-aware extraction, multi-user sessions.

## 14 · Sample inputs and outputs

TBD — index into `outputs/answers/` and `outputs/conversations/`, both committed so
results are readable without running anything.

---

## Troubleshooting

TBD — the silent-CPU-torch fix, Ollama model pulls, quota-exhaustion messages, and
corpus re-fetch. One entry is already confirmed and worth stating early:

**`pip install torch` gives you a CPU build and says nothing.** It reports success,
`torch.cuda.is_available()` returns `False`, and every encoder runs roughly 5× slower
with no error anywhere. If a CPU wheel is already installed, `pip install torch`
prints *"already satisfied"* and leaves it in place. RTX 50-series cards are compute
capability sm_120 and need CUDA 12.8 or newer, so the cu126 wheels load but carry no
kernels for them. This project pins the cu130 build; `doctor --gpu` hard-fails rather
than letting a CPU fallback pass silently.
