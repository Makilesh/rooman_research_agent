"""The agentic loop: judge the evidence, act again if it is thin, then stop.

This is what makes the system an agent rather than a retrieval script. It decides
whether it has enough evidence, acts again when it does not, and gives up in a bounded
way when acting again has not helped.

Two pieces carry most of the value, and both come from failures in my previous work:

**Sub-questions are reranked against themselves, not against the parent query.** In
the M&A engine this was worth roughly +20pp fact coverage. Retrieving "how does QLoRA
extend LoRA" as one query gets passages that look like the *question*; retrieving
"what memory optimisations does QLoRA introduce" separately gets passages that answer
one *part* of it. Scoring the second set against the parent query throws that away.

**Progressive constraint relaxation.** Any filter applied on the first attempt is
dropped on retry. In the M&A engine a wrong first-pass category guess removed whole
documents from the search space, and no amount of query rewriting could recover the
answer — the evidence was never a candidate. Tight first for precision, loose on retry
for recall.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import Any, Callable, Sequence

from . import db, prompts, rerank as rerank_mod, retrieve, router as router_mod
from .config import Config
from .retrieve import Hit


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One node of the loop, recorded so `ask --trace` can print what happened."""

    loop: int
    node: str
    detail: str
    ladder: str | None = None
    n_hits: int = 0
    top_score: float | None = None
    ms: int = 0


@dataclass
class Trace:
    steps: list[TraceStep] = field(default_factory=list)

    def add(self, **kwargs: Any) -> None:
        self.steps.append(TraceStep(**kwargs))

    def render(self) -> list[str]:
        out = []
        for s in self.steps:
            bits = [f"  [{s.loop}] {s.node:<22}"]
            if s.ladder:
                bits.append(f"({s.ladder})")
            bits.append(s.detail)
            if s.n_hits:
                bits.append(f"· {s.n_hits} hits")
            if s.top_score is not None:
                bits.append(f"· top {s.top_score:.3f}")
            if s.ms:
                bits.append(f"· {s.ms}ms")
            out.append(" ".join(bits))
        return out


@dataclass(frozen=True, slots=True)
class Judgement:
    sufficient: bool
    missing: str
    is_multi_part: bool
    used_llm: bool = True


@dataclass(frozen=True, slots=True)
class LoopResult:
    hits: list[Hit]
    query: str
    loops: int
    exhausted: bool          # ran out of attempts with evidence still judged thin
    sub_questions: list[str] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)


# ---------------------------------------------------------------------------
#  Document scoping, and relaxing it
# ---------------------------------------------------------------------------
def infer_document_scope(conn: Connection, query: str) -> set[str]:
    """Which papers does the question name explicitly?

    Scoping the first attempt to a named paper is a precision win, and it is exactly
    the filter that progressive relaxation must drop on retry -- a question that names
    the wrong paper (the NF4-in-LoRA case) can only be answered once the scope is
    gone.
    """
    rows = db.all_rows(conn, "SELECT doc_id, title FROM documents")
    scope: set[str] = set()
    lowered = query.lower()
    for r in rows:
        doc_id = r["doc_id"]
        # The doc_id is the short handle a person actually types: "lora", "bert".
        if re.search(rf"\b{re.escape(doc_id)}\b", lowered):
            scope.add(doc_id)
            continue
        # Otherwise look for a distinctive run of words from the real title.
        head = r["title"].split(":")[0].strip().lower()
        if len(head) > 8 and head in lowered:
            scope.add(doc_id)
    return scope


def _retrieve(conn: Connection, cfg: Config, models, query: str,
              scope: set[str] | None) -> list[Hit]:
    embedder, vectors, reranker = models
    dense = retrieve.dense_search(conn, retrieve.embed_query(embedder, query),
                                  vectors, cfg)
    sparse = retrieve.sparse_search(conn, query, cfg)
    fused = retrieve.reciprocal_rank_fusion(dense, sparse, cfg)
    if scope:
        scoped = [h for h in fused if h.doc_id in scope]
        # Never let a scope empty the candidate set: an empty slate cannot be
        # judged, only refused, and that would hide the reason.
        fused = scoped or fused
    return rerank_mod.rerank(reranker, query, fused, cfg)


# ---------------------------------------------------------------------------
#  Judging
# ---------------------------------------------------------------------------
def judge(cfg: Config, client, query: str, hits: Sequence[Hit],
          provider: str | None = None) -> Judgement:
    """Is this evidence enough? Volume ladder, never synthesis."""
    if not hits:
        return Judgement(False, "retrieval returned nothing", False, used_llm=False)
    try:
        completion = client.complete(
            prompts.judge_prompt(query, hits), purpose="judge", ladder="volume",
            schema=prompts.JUDGE_SCHEMA, provider=provider,
        )
        data = completion.data or {}
        return Judgement(bool(data.get("sufficient", True)),
                         (data.get("missing") or "").strip(),
                         bool(data.get("is_multi_part", False)))
    except Exception:
        # A judge failure must not cost the turn. Assuming sufficiency is the
        # conservative default here: it proceeds to synthesis, which has its own
        # abstention path, rather than refusing on the judge's absence.
        return Judgement(True, "", False, used_llm=False)


def decompose(cfg: Config, client, conn: Connection, query: str,
              provider: str | None = None) -> list[str]:
    titles = "; ".join(
        r["title"] for r in db.all_rows(conn, "SELECT title FROM documents"))
    try:
        completion = client.complete(
            prompts.decompose_prompt(query, cfg.max_subquestions, titles),
            purpose="decompose", ladder="volume",
            schema=prompts.DECOMPOSE_SCHEMA, provider=provider,
        )
        subs = [s.strip() for s in ((completion.data or {}).get("sub_questions") or [])
                if s and s.strip()]
        return subs[: cfg.max_subquestions]
    except Exception:
        return []


def rewrite(cfg: Config, client, conn: Connection, query: str, missing: str,
            provider: str | None = None) -> str | None:
    titles = "; ".join(
        r["title"] for r in db.all_rows(conn, "SELECT title FROM documents"))
    try:
        completion = client.complete(
            prompts.rewrite_prompt(query, missing, titles), purpose="rewrite",
            ladder="volume", schema=prompts.REWRITE_SCHEMA, provider=provider,
        )
        out = ((completion.data or {}).get("rewritten") or "").strip()
        return out or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  The loop
# ---------------------------------------------------------------------------
def gather_evidence(
    cfg: Config,
    conn: Connection,
    client,
    models,
    query: str,
    provider: str | None = None,
    on_event: Callable[[str], None] | None = None,
) -> LoopResult:
    """Retrieve, judge, and act again at most `max_retrieval_loops` times.

    Termination is structural, not conditional: the loop is a bounded `for`, so it
    cannot run away regardless of what any model returns.
    """
    trace = Trace()
    scope = infer_document_scope(conn, query)
    current_query = query
    sub_questions: list[str] = []
    hits: list[Hit] = []
    log = on_event or (lambda _m: None)

    for loop in range(cfg.max_retrieval_loops + 1):
        # Relaxation: the scope is a first-attempt precision aid only.
        active_scope = scope if loop == 0 else None
        if loop > 0 and scope:
            trace.add(loop=loop, node="relax-constraints",
                      detail=f"dropped document scope {sorted(scope)} — a wrong "
                             f"first-pass guess removes the answer from the search "
                             f"space entirely")

        started = time.monotonic()
        if sub_questions:
            hits = _retrieve_sub_questions(conn, cfg, models, sub_questions, trace, loop)
        else:
            hits = _retrieve(conn, cfg, models, current_query, active_scope)
        hits = router_mod.apply_diversity_guard(cfg, hits)
        trace.add(loop=loop, node="retrieve",
                  detail=f"query={current_query[:70]!r}"
                         + (f" scope={sorted(active_scope)}" if active_scope else ""),
                  n_hits=len(hits),
                  top_score=max((h.rerank_score or 0.0 for h in hits), default=None),
                  ms=int((time.monotonic() - started) * 1000))

        if loop == cfg.max_retrieval_loops:
            trace.add(loop=loop, node="loop-cap",
                      detail=f"reached the {cfg.max_retrieval_loops}-loop cap; "
                             f"proceeding with what was found")
            return LoopResult(hits, current_query, loop, False, sub_questions, trace)

        started = time.monotonic()
        verdict = judge(cfg, client, current_query, hits, provider)
        trace.add(loop=loop, node="sufficiency-judge", ladder="volume",
                  detail=("sufficient" if verdict.sufficient
                          else f"INSUFFICIENT — {verdict.missing[:80]}")
                         + (" · multi-part" if verdict.is_multi_part else ""),
                  ms=int((time.monotonic() - started) * 1000))

        if verdict.sufficient:
            return LoopResult(hits, current_query, loop, False, sub_questions, trace)

        # Insufficient. Decompose a multi-part question; otherwise rewrite.
        if verdict.is_multi_part and not sub_questions:
            subs = decompose(cfg, client, conn, current_query, provider)
            if len(subs) > 1:
                sub_questions = subs
                trace.add(loop=loop, node="decompose", ladder="volume",
                          detail=" | ".join(s[:46] for s in subs))
                continue
            trace.add(loop=loop, node="decompose", ladder="volume",
                      detail="returned a single part; falling through to rewrite")

        rewritten = rewrite(cfg, client, conn, current_query, verdict.missing, provider)
        if rewritten and rewritten != current_query:
            trace.add(loop=loop, node="rewrite", ladder="volume",
                      detail=f"{current_query[:40]!r} -> {rewritten[:60]!r}")
            current_query = rewritten
            continue

        trace.add(loop=loop, node="exhausted",
                  detail="evidence judged insufficient and no further action available")
        return LoopResult(hits, current_query, loop, True, sub_questions, trace)

    return LoopResult(hits, current_query, cfg.max_retrieval_loops, True,
                      sub_questions, trace)


def _retrieve_sub_questions(conn: Connection, cfg: Config, models,
                            sub_questions: Sequence[str], trace: Trace,
                            loop: int) -> list[Hit]:
    """Retrieve each sub-question independently, reranked AGAINST ITSELF.

    The reranking target is the load-bearing detail. Scoring sub-question results
    against the parent query would undo the decomposition: the whole point is that a
    passage answering one part need not resemble the composite question at all.
    """
    _, _, reranker = models
    merged: dict[str, Hit] = {}

    for sub in sub_questions:
        sub_hits = _retrieve(conn, cfg, models, sub, None)
        trace.add(loop=loop, node="sub-question",
                  detail=f"{sub[:60]!r} reranked against itself",
                  n_hits=len(sub_hits),
                  top_score=max((h.rerank_score or 0.0 for h in sub_hits),
                                default=None))
        for hit in sub_hits:
            prior = merged.get(hit.chunk_id)
            if prior is None or (hit.rerank_score or 0) > (prior.rerank_score or 0):
                merged[hit.chunk_id] = hit

    ordered = sorted(merged.values(), key=lambda h: -(h.rerank_score or 0.0))
    from dataclasses import replace as _replace

    return [_replace(h, rank=i + 1) for i, h in enumerate(ordered)]
