"""The turn state machine: condense, route, retrieve, synthesise, verify, persist.

A plain Python state machine, not LangGraph. I have shipped LangGraph in production
and chose against it here for one reason: this is the logic a reviewer most wants to
read, and a framework would hide it behind a graph definition. The whole flow is one
function you can follow top to bottom.

Step 11 adds the sufficiency loop on top of this; the shape does not change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from sqlite3 import Connection
from typing import Any, Sequence

from . import answer as answer_mod
from . import conversation, db, index as index_mod, prompts, rerank as rerank_mod
from . import retrieve, router as router_mod, sufficiency, verify as verify_mod
from .config import Config
from .retrieve import Hit


@dataclass(frozen=True, slots=True)
class TurnResult:
    session_id: str
    ord: int
    raw_text: str
    condensed: conversation.Condensation
    route: router_mod.Route
    answer: answer_mod.Answer | None
    clarification: str | None
    clarify_options: list[str] = field(default_factory=list)
    resolved_refs: list[str] = field(default_factory=list)
    cache_hit: bool = False
    latency_ms: int = 0
    trace: sufficiency.Trace | None = None
    n_loops: int = 0
    sub_questions: list[str] = field(default_factory=list)
    exhausted: bool = False
    # The query retrieval finally ran on. NOT the condensation: the
    # sufficiency loop may rewrite it again afterwards, and it is this
    # text that determines the rerank score the router sees.
    final_query: str = ""

    @property
    def decision(self) -> str:
        if self.answer is not None and self.answer.is_refusal:
            # The router let it through but the synthesiser found the evidence did not
            # actually support an answer. That is abstention, not refusal-by-score,
            # and the confusion matrix must be able to tell them apart.
            return "abstain"
        return self.route.decision


def run_turn(
    cfg: Config,
    conn: Connection,
    client,
    models: tuple[Any, Any, Any],
    session_id: str,
    raw_text: str,
    provider: str | None = None,
    is_clarification_reply: bool = False,
) -> TurnResult:
    embedder, vectors, reranker = models
    started = time.monotonic()
    fingerprint = index_mod.load_fingerprint(conn)

    turns = conversation.load_turns(conn, session_id)
    ord_ = len(turns) + 1
    window = conversation.history_window(turns, cfg)

    # -- coreference -------------------------------------------------------
    _, refs, unresolved_refs = conversation.resolve_source_references(
        conn, session_id, raw_text)
    ref_vocab = conversation.describe_references(conn, refs)

    if unresolved_refs and not refs:
        # "the second source" when the previous answer had one. Retrieving on that
        # phrase scores near zero and refuses for the wrong reason, so say what is
        # actually wrong.
        word, available = unresolved_refs[0]
        msg = (f"You asked about the {word} source, but the previous answer cited "
               f"{available if available else 'no'} source"
               f"{'' if available == 1 else 's'}. "
               f"Which one did you mean?")
        _persist_clarify(conn, session_id, ord_ + 1, raw_text,
                         conversation.Condensation(raw_text, False, False,
                                                   reason="ordinal reference"),
                         router_mod.Route(router_mod.CLARIFY, 0.0,
                                          "source reference did not resolve", []),
                         msg)
        return TurnResult(
            session_id, ord_, raw_text,
            conversation.Condensation(raw_text, False, False,
                                      reason="ordinal reference did not resolve"),
            router_mod.Route(router_mod.CLARIFY, 0.0,
                             f"the previous answer cited {available} source(s), so "
                             f"'the {word} source' does not exist", []),
            None, msg, [], [], False, int((time.monotonic() - started) * 1000))

    # -- clarification reply ----------------------------------------------
    query_seed = raw_text
    if is_clarification_reply or _last_was_clarify(turns):
        prior = _last_user_question(turns)
        if prior:
            query_seed = router_mod.merge_clarification(prior, raw_text)

    # -- condensation ------------------------------------------------------
    condensed = conversation.condense(cfg, client, window, query_seed, ref_vocab)
    query = condensed.query
    if refs:
        # An ordinal or explicit id points at a specific passage. Adding its title and
        # section to the query is what makes "expand on the second source" retrievable
        # at all -- the phrase itself contains no topical signal.
        query = f"{query} ({ref_vocab})"

    conversation.record_user_turn(conn, session_id, ord_, raw_text,
                                  condensed.query if condensed.used_llm else None)

    # -- semantic cache ----------------------------------------------------
    qvec = retrieve.embed_query(embedder, query)
    if fingerprint:
        cached = conversation.cache_lookup(conn, cfg, fingerprint, qvec)
        if cached is not None:
            restored = answer_mod.from_cached(cached["answer"])
            return TurnResult(
                session_id, ord_, raw_text, condensed,
                router_mod.Route(router_mod.ANSWER, 1.0,
                                 f"semantic cache hit (cos {cached['score']:.4f})", []),
                restored, None, cache_hit=True,
                latency_ms=int((time.monotonic() - started) * 1000),
            )

    # -- retrieve, judge, act again if the evidence is thin -----------------
    loop_result = sufficiency.gather_evidence(
        cfg, conn, client, models, query, provider=provider)
    reranked = loop_result.hits

    # A passage the user explicitly pointed at belongs in context whatever the
    # reranker thinks of it -- they have already told us it is relevant.
    reranked = _pin_referenced(conn, reranked, refs, cfg)
    reranked = router_mod.apply_diversity_guard(cfg, reranked)

    if loop_result.exhausted:
        # The loop tried everything available and the evidence is still thin.
        # Abstaining here is the honest outcome, and it is distinguishable in the
        # ledger from a refusal made on the score alone.
        _persist_refuse(conn, session_id, ord_ + 1, raw_text, condensed,
                        router_mod.Route(router_mod.REFUSE,
                                         max((h.rerank_score or 0.0 for h in reranked),
                                             default=0.0),
                                         "the retrieval loop was exhausted without "
                                         "finding sufficient evidence", []))
        return TurnResult(
            session_id, ord_, raw_text, condensed,
            router_mod.Route(router_mod.REFUSE, 0.0,
                             "retrieval loop exhausted without sufficient evidence",
                             []),
            None, None, [], refs, False,
            int((time.monotonic() - started) * 1000),
            loop_result.trace, loop_result.loops, loop_result.sub_questions, True)

    # -- route -------------------------------------------------------------
    route = router_mod.route(cfg, reranked)

    if route.decision == router_mod.CLARIFY:
        question, options = _ask_clarifying(cfg, client, query, route.competing,
                                            provider)
        _persist_clarify(conn, session_id, ord_ + 1, raw_text, condensed, route,
                         question)
        return TurnResult(session_id, ord_, raw_text, condensed, route, None,
                          question, options, refs, False,
                          int((time.monotonic() - started) * 1000),
                          loop_result.trace, loop_result.loops,
                          loop_result.sub_questions, False, loop_result.query)

    if route.decision == router_mod.REFUSE:
        _persist_refuse(conn, session_id, ord_ + 1, raw_text, condensed, route)
        return TurnResult(session_id, ord_, raw_text, condensed, route, None, None,
                          [], refs, False,
                          int((time.monotonic() - started) * 1000),
                          loop_result.trace, loop_result.loops,
                          loop_result.sub_questions, False, loop_result.query)

    # -- answer ------------------------------------------------------------
    context = retrieve.fit_context_budget(
        retrieve.expand_to_parents(conn, reranked), cfg,
        count_tokens=index_mod.token_counter(embedder),
    )
    result = answer_mod.synthesise(cfg, client, query, context, provider=provider)
    result = verify_mod.verify_answer(cfg, reranker, result)

    decision = "abstain" if result.is_refusal else "answer"
    turn_id = answer_mod.persist(conn, result, session_id, ord_ + 1, route=decision,
                                 n_loops=loop_result.loops)
    result = replace(result, turn_id=turn_id)

    if fingerprint and not result.is_refusal:
        from . import report

        conversation.cache_store(conn, fingerprint, query, qvec,
                                 report.answer_to_dict(result))

    return TurnResult(session_id, ord_, raw_text, condensed, route, result, None,
                      [], refs, False, int((time.monotonic() - started) * 1000),
                      loop_result.trace, loop_result.loops,
                      loop_result.sub_questions, False, loop_result.query)


# ---------------------------------------------------------------------------
def _pin_referenced(conn: Connection, hits: Sequence[Hit], refs: Sequence[str],
                    cfg: Config) -> list[Hit]:
    if not refs:
        return list(hits)
    present = {h.chunk_id for h in hits}
    missing = [r for r in refs if r not in present]
    if not missing:
        return list(hits)
    rows = db.all_rows(conn, f"""
        SELECT c.chunk_id, c.doc_id, d.title, c.page_start, c.page_end, c.section, c.text
        FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.chunk_id IN ({','.join('?' * len(missing))})
    """, missing)
    pinned = [
        Hit(r["chunk_id"], r["doc_id"], r["title"], r["page_start"], r["page_end"],
            r["section"], r["text"], rerank_score=1.0, rank=0)
        for r in rows
    ]
    return pinned + list(hits)


def _last_was_clarify(turns: Sequence[conversation.Turn]) -> bool:
    for turn in reversed(turns):
        if turn.role == "agent":
            return turn.route == router_mod.CLARIFY
    return False


def _last_user_question(turns: Sequence[conversation.Turn]) -> str | None:
    for turn in reversed(turns):
        if turn.role == "user":
            return turn.raw_text
    return None


def _ask_clarifying(cfg: Config, client, question: str, competing: Sequence[Hit],
                    provider: str | None) -> tuple[str, list[str]]:
    if not competing:
        return ("Which of the sources did you mean?", [])
    try:
        completion = client.complete(
            prompts.clarify_prompt(question, competing),
            purpose="clarify", ladder="volume", schema=prompts.CLARIFY_SCHEMA,
            provider=provider,
        )
        data = completion.data or {}
        text = (data.get("question") or "").strip()
        options = [o for o in (data.get("options") or []) if o]
        if text:
            return text, options
    except Exception:
        pass
    # Deterministic fallback that still names the alternatives, because a generic
    # "please clarify" wastes the user's turn.
    options = [f"{h.title}" + (f" ({h.section})" if h.section else "")
               for h in competing]
    return ("Which did you mean: " + ", or ".join(options) + "?", options)


def _persist_clarify(conn: Connection, session_id: str, ord_: int, raw: str,
                     condensed: conversation.Condensation,
                     route: router_mod.Route, question: str) -> str:
    """A clarifying question is a real turn, not a UI event.

    Stored with route='clarify' so the confusion matrix can score it and so the next
    turn can recognise that it is answering one.
    """
    turn_id = f"t_{__import__('uuid').uuid4().hex[:12]}"
    db.insert(conn, "turns", {
        "turn_id": turn_id, "session_id": session_id, "ord": ord_, "role": "agent",
        "raw_text": question, "condensed_query": condensed.query,
        "route": router_mod.CLARIFY, "top_rerank_score": route.top_score,
        "n_retrieval_loops": 0,
    })
    conn.commit()
    return turn_id


def _persist_refuse(conn: Connection, session_id: str, ord_: int, raw: str,
                    condensed: conversation.Condensation,
                    route: router_mod.Route) -> str:
    turn_id = f"t_{__import__('uuid').uuid4().hex[:12]}"
    db.insert(conn, "turns", {
        "turn_id": turn_id, "session_id": session_id, "ord": ord_, "role": "agent",
        "raw_text": route.reason, "condensed_query": condensed.query,
        "route": router_mod.REFUSE, "top_rerank_score": route.top_score,
        "n_retrieval_loops": 0,
    })
    conn.commit()
    return turn_id
