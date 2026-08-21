"""The citation contract: schema-constrained synthesis, id validation, abstention.

This is the technical thesis of the project. Citations are not markers parsed out of
prose — they are a structured sentence-to-chunk mapping the model is required to emit,
validated before anything is rendered, and persisted as relational rows.

The validation is the part that matters. Constrained decoding guarantees the *shape*
of the output; it cannot guarantee the ids are real, because a model can emit a
perfectly well-formed lie. Every returned id is checked against the set actually
retrieved for this turn, and an invented id is a hard failure that is never rendered.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import Sequence

from . import db, prompts
from .config import Config
from .llm import LLMClient, MalformedResponse
from .retrieve import Hit


class InventedCitation(RuntimeError):
    """The model cited a chunk id that was not in the retrieved set.

    Deliberately fatal rather than filtered. A fabricated citation is the single
    failure this whole system exists to prevent, so it surfaces loudly instead of
    being silently dropped into a partially-correct answer.
    """


@dataclass(frozen=True, slots=True)
class CitedSentence:
    idx: int
    text: str
    chunk_ids: list[str]
    verify_scores: dict[str, float] = field(default_factory=dict)
    status: str = "unverified"


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    insufficient_evidence: bool
    refusal_reason: str | None
    sentences: list[CitedSentence]
    hits: list[Hit]
    context: str          # byte-identical to what the synthesiser saw
    provider: str
    model: str
    latency_ms: int
    turn_id: str | None = None
    top_rerank_score: float | None = None
    # True when this Answer was rebuilt from a cache record rather than
    # synthesised. Without it `latency_ms=0` reads as a broken metric in the
    # committed artifacts -- the reader sees a provider and model named, and a
    # zero next to them, and reasonably concludes the number is invented.
    cached: bool = False

    @property
    def is_refusal(self) -> bool:
        return self.insufficient_evidence

    @property
    def cited_chunk_ids(self) -> set[str]:
        return {c for s in self.sentences for c in s.chunk_ids}

    def hit_by_id(self, chunk_id: str) -> Hit | None:
        return next((h for h in self.hits if h.chunk_id == chunk_id), None)


def synthesise(
    cfg: Config,
    client: LLMClient,
    question: str,
    hits: Sequence[Hit],
    provider: str | None = None,
) -> Answer:
    """Produce a cited answer or an explicit refusal.

    The context string is built once and carried on the result, so verification can
    score against byte-identical text rather than rebuilding it and drifting.
    """
    hits = list(hits)
    context = prompts.format_sources(hits)
    prompt = prompts.synthesis_prompt(question, hits)

    started = time.monotonic()
    completion = client.complete(
        prompt, purpose="synthesise", ladder="synthesis",
        schema=prompts.ANSWER_SCHEMA, provider=provider,
    )
    latency_ms = int((time.monotonic() - started) * 1000)

    data = completion.data
    if data is None:
        raise MalformedResponse("Synthesiser returned no parseable JSON payload.")

    valid_ids = {h.chunk_id for h in hits}
    sentences: list[CitedSentence] = []
    insufficient = bool(data.get("insufficient_evidence", False))

    for idx, raw in enumerate(data.get("sentences") or []):
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        cites = [c for c in (raw.get("cite") or []) if c]
        invented = [c for c in cites if c not in valid_ids]
        if invented:
            raise InventedCitation(
                f"The model cited {invented} for the sentence {text!r}, but only "
                f"{sorted(valid_ids)} were retrieved for this turn. A fabricated "
                f"citation is never rendered."
            )
        if not cites:
            # An uncited sentence violates the contract as surely as a bad id does.
            # It is dropped rather than fatal, because the rest of the answer may be
            # sound and the omission is visible in the sentence count.
            continue
        sentences.append(CitedSentence(idx=len(sentences), text=text, chunk_ids=cites))

    # A refusal that still carries citations is contradictory. Trust the flag and
    # drop the citations, so the "zero fabricated citations" property of the control
    # questions holds structurally rather than by the model's good behaviour.
    if insufficient:
        sentences = []
    elif not sentences:
        # No usable sentences and no refusal flag: treat as a refusal rather than
        # rendering an empty answer that looks like a bug.
        insufficient = True

    return Answer(
        question=question,
        insufficient_evidence=insufficient,
        refusal_reason=(data.get("refusal_reason") or None) if insufficient else None,
        sentences=sentences,
        hits=hits,
        context=context,
        provider=completion.provider,
        model=completion.model,
        latency_ms=latency_ms,
        top_rerank_score=max((h.rerank_score or 0.0 for h in hits), default=None),
    )


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------
def persist(conn: Connection, answer: Answer, session_id: str, ord_: int,
            route: str, n_loops: int = 0) -> str:
    """Write the turn, its citations, and every retrieval score.

    `turn_retrievals` records what was retrieved *and* what actually reached the
    context, at every scoring stage, so any ranking decision can be audited after the
    fact rather than re-derived by guesswork.
    """
    turn_id = f"t_{uuid.uuid4().hex[:12]}"
    db.insert(conn, "turns", {
        "turn_id": turn_id, "session_id": session_id, "ord": ord_, "role": "agent",
        "raw_text": answer.question, "condensed_query": None, "route": route,
        "top_rerank_score": answer.top_rerank_score, "n_retrieval_loops": n_loops,
        "latency_ms": answer.latency_ms, "provider": answer.provider,
        "model": answer.model,
    })

    cited = answer.cited_chunk_ids
    db.insert_many(conn, "turn_retrievals", [
        {"turn_id": turn_id, "chunk_id": h.chunk_id, "rank": h.rank,
         "dense_score": h.dense_score, "sparse_score": h.sparse_score,
         "rrf_score": h.rrf_score, "rerank_score": h.rerank_score,
         "used_in_context": 1 if h.chunk_id in cited else 0}
        for h in answer.hits
    ])

    rows = []
    for s in answer.sentences:
        for cid in s.chunk_ids:
            rows.append({
                "turn_id": turn_id, "sentence_idx": s.idx, "sentence_text": s.text,
                "chunk_id": cid, "verify_score": s.verify_scores.get(cid),
                "status": s.status,
            })
    db.insert_many(conn, "turn_citations", rows)
    conn.commit()
    return turn_id


def ensure_session(conn: Connection, session_id: str, fingerprint: str | None) -> str:
    if db.one(conn, "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)) is None:
        db.insert(conn, "sessions", {
            "session_id": session_id, "corpus_fingerprint": fingerprint,
            "meta_json": json.dumps({}),
        })
        conn.commit()
    return session_id


def from_cached(payload: dict) -> Answer:
    """Rebuild an Answer from a cached JSON record.

    Without this a semantic cache hit returned `answer=None` and the stored answer was
    silently discarded -- the turn reported zero sentences and zero citations in ~14ms
    and looked like a catastrophic failure rather than a cache hit. A cache that loses
    the thing it cached is worse than no cache.
    """
    hits = [
        Hit(s["chunk_id"], s["doc_id"], s["title"], s["page_start"], s["page_end"],
            s.get("section"), s.get("text", ""), rerank_score=s.get("rerank_score"),
            rank=i + 1)
        for i, s in enumerate(payload.get("sources") or [])
    ]
    sentences = [
        CitedSentence(idx=s["idx"], text=s["text"], chunk_ids=list(s["cite"]),
                      verify_scores=dict(s.get("verify_scores") or {}),
                      status=s.get("status", "verified"))
        for s in (payload.get("sentences") or [])
    ]
    return Answer(
        question=payload.get("question", ""),
        insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
        refusal_reason=payload.get("refusal_reason"),
        sentences=sentences, hits=hits,
        context="",  # not persisted; verification already ran before caching
        provider=payload.get("provider", "cache"),
        model=payload.get("model", "cache"),
        latency_ms=0, turn_id=payload.get("turn_id"), cached=True,
    )
