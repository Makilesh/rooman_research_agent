"""Multi-turn conversation: sessions, condensation, coreference, history window.

Condensation is where multi-turn RAG usually breaks, and it breaks quietly. In my
Opkey build the follow-up *"what statuses can it have during approval?"* was condensed
into a query containing an invented content word — *"workflow"* — which steered
retrieval into entirely the wrong chapters. Nothing errored. The answer was fluent and
wrong.

The fix here is two-layered, and the second layer is the one that matters:

1. **A prompt constraint.** The condenser may only reuse vocabulary already present in
   the history or the raw follow-up. It paraphrases and resolves pronouns; it does not
   introduce nouns.
2. **A programmatic drift guard.** After condensation, content words are stemmed and
   diffed against `history ∪ raw`. Any novel content word means the constraint was
   violated, and the raw query is used instead. A prompt instruction is a request; a
   diff is an enforcement, and only the second one is testable.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import Sequence

import numpy as np

from . import db
from .config import Config

# Words that carry no retrieval signal. A condensation that adds "the" is not drift;
# one that adds "workflow" is.
_STOPWORDS = frozenset("""
a an the of for to in on at by with from is are was were be been being do does did
what which who whom whose how why when where and or but if then than that this these
those it its as into about can could should would may might will shall not no
i you he she we they them his her their our your me him us my mine yours ours
there here also very much many more most some any all both each other another same
such only just even still yet again once between through during before after above
below up down out off over under further too so than because while although
does say says said tell asks ask answer question use used using make makes made
one two three four five first second third last next previous
""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")

# Ordinal references to a previous answer's footnotes.
_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "last": -1,
}
_ORDINAL_REF = re.compile(
    r"\b(" + "|".join(_ORDINALS) + r")\s+(?:source|citation|reference|paper|passage)\b",
    re.IGNORECASE,
)
_CHUNK_REF = re.compile(r"\[?([cp]_[a-z0-9]+_\d{4})\]?")


def stem(word: str) -> str:
    """Crude, deterministic suffix stripping.

    Not linguistically correct and does not need to be. Its only job is to stop the
    drift guard firing on "quantise" vs "quantised", and a real stemmer would be
    another dependency and another download for that.
    """
    w = word.lower()
    for suffix in ("ations", "ation", "ingly", "edly", "ings", "ing", "ies", "ied",
                   "es", "ed", "ly", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            w = w[: -len(suffix)]
            break
    # Strip a trailing "e" as well, always. Without this the stemmer is INCONSISTENT
    # rather than merely crude: "compared" loses "ed" and becomes "compar", while
    # "compare" matches no suffix and stays "compare". The drift guard then reports
    # a novel content word for a rephrasing that used a word already in the
    # conversation -- a false positive, observed on scenario B turn 3.
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def content_words(text: str) -> set[str]:
    return {stem(w) for w in _WORD.findall(text)
            if w.lower() not in _STOPWORDS and len(w) > 2}


@dataclass(frozen=True, slots=True)
class Turn:
    turn_id: str
    ord: int
    role: str
    raw_text: str
    condensed_query: str | None = None
    route: str | None = None
    answer_text: str = ""


@dataclass(frozen=True, slots=True)
class Condensation:
    query: str
    used_llm: bool
    drifted: bool
    novel_words: set[str] = field(default_factory=set)
    fell_back: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
#  Sessions
# ---------------------------------------------------------------------------
def new_session(conn: Connection, fingerprint: str | None) -> str:
    session_id = f"s_{uuid.uuid4().hex[:12]}"
    db.insert(conn, "sessions", {
        "session_id": session_id, "corpus_fingerprint": fingerprint,
        "meta_json": json.dumps({}),
    })
    conn.commit()
    return session_id


def fingerprint_warning(conn: Connection, session_id: str,
                        current: str | None) -> str | None:
    """A session outlives the corpus it was built against.

    Its stored citations point at chunk ids that may now resolve to different text,
    so the mismatch is surfaced rather than silently tolerated.
    """
    row = db.one(conn, "SELECT corpus_fingerprint FROM sessions WHERE session_id = ?",
                 (session_id,))
    if row is None or row["corpus_fingerprint"] is None or current is None:
        return None
    if row["corpus_fingerprint"] != current:
        return (f"This session was started against corpus {row['corpus_fingerprint']} "
                f"but the index is now {current}. Citations from earlier turns may no "
                f"longer point at the text they did. Start a new session with /new.")
    return None


def load_turns(conn: Connection, session_id: str) -> list[Turn]:
    rows = db.all_rows(conn, """
        SELECT turn_id, ord, role, raw_text, condensed_query, route
        FROM turns WHERE session_id = ? ORDER BY ord
    """, (session_id,))
    out = []
    for r in rows:
        text = ""
        if r["role"] == "agent":
            sents = db.all_rows(
                conn,
                "SELECT DISTINCT sentence_idx, sentence_text FROM turn_citations "
                "WHERE turn_id = ? ORDER BY sentence_idx", (r["turn_id"],))
            text = " ".join(s["sentence_text"] for s in sents)
        out.append(Turn(r["turn_id"], r["ord"], r["role"], r["raw_text"],
                        r["condensed_query"], r["route"], text))
    return out


def history_window(turns: Sequence[Turn], cfg: Config) -> list[Turn]:
    """Most recent turns, capped by count AND token budget, whichever binds first.

    Two caps rather than one because they fail differently: a turn cap alone lets six
    long turns blow the condenser's context, and a token cap alone lets twenty
    one-word turns through, which is more history than a condenser can use coherently.
    Full history stays in SQLite for `/history` regardless.
    """
    kept: list[Turn] = []
    tokens = 0
    for turn in reversed(turns):
        cost = (len(turn.raw_text) + len(turn.answer_text)) // 4
        if len(kept) >= cfg.history_max_turns:
            break
        if kept and tokens + cost > cfg.history_max_tokens:
            break
        kept.insert(0, turn)
        tokens += cost
    return kept


def render_history(turns: Sequence[Turn]) -> str:
    lines = []
    for turn in turns:
        who = "User" if turn.role == "user" else "Assistant"
        body = turn.raw_text if turn.role == "user" else turn.answer_text
        if body:
            lines.append(f"{who}: {body}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Coreference
# ---------------------------------------------------------------------------
def resolve_source_references(
    conn: Connection, session_id: str, text: str
) -> tuple[str, list[str], list[tuple[str, int]]]:
    """Resolve "the second source" and explicit chunk ids against `turn_citations`.

    This is the concrete payoff of the structured citation contract, and the thing
    inline prose markers cannot do. Because every sentence-to-chunk link is a row, a
    follow-up can point at one by position and the agent answers it with a query
    rather than by re-parsing its own previous prose.

    Ordinal position is defined as the order footnotes appeared in the rendered
    answer, which is the order the user actually saw.
    """
    referenced: list[str] = []
    unresolved: list[tuple[str, int]] = []

    for match in _CHUNK_REF.finditer(text):
        referenced.append(match.group(1))

    ordinals = _ORDINAL_REF.findall(text)
    if ordinals:
        pass
    if ordinals:
        last = db.one(conn, """
            SELECT turn_id FROM turns
            WHERE session_id = ? AND role = 'agent' AND route = 'answer'
            ORDER BY ord DESC LIMIT 1
        """, (session_id,))
        if last is not None:
            rows = db.all_rows(conn, """
                SELECT chunk_id, MIN(sentence_idx) AS first_seen
                FROM turn_citations WHERE turn_id = ?
                GROUP BY chunk_id ORDER BY first_seen
            """, (last["turn_id"],))
            order = [r["chunk_id"] for r in rows]
            for word in ordinals:
                idx = _ORDINALS[word.lower()]
                if not order:
                    unresolved.append((word, 0))
                    continue
                target = order[-1] if idx == -1 else (
                    order[idx - 1] if 0 < idx <= len(order) else None)
                if target:
                    referenced.append(target)
                else:
                    # The user pointed at a source that does not exist -- the previous
                    # answer had fewer. Retrieving on the bare phrase would score near
                    # zero and produce a confusing refusal, so the caller is told
                    # exactly what went wrong instead.
                    unresolved.append((word, len(order)))

    return text, list(dict.fromkeys(referenced)), unresolved


def describe_references(conn: Connection, chunk_ids: Sequence[str]) -> str:
    """Turn resolved chunk ids into vocabulary the condenser may legitimately use.

    Words drawn from a cited passage are not "novel" -- the user pointed at that
    passage, so its content is part of the conversation. Supplying them explicitly is
    what lets the drift guard stay strict everywhere else.
    """
    if not chunk_ids:
        return ""
    rows = db.all_rows(conn, f"""
        SELECT c.chunk_id, d.title, c.section, c.page_start
        FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.chunk_id IN ({','.join('?' * len(chunk_ids))})
    """, list(chunk_ids))
    return "; ".join(
        f"{r['chunk_id']} = {r['title']}"
        + (f", section {r['section']}" if r["section"] else "")
        + f", page {r['page_start']}"
        for r in rows
    )


# ---------------------------------------------------------------------------
#  Condensation
# ---------------------------------------------------------------------------
def condense(cfg: Config, client, history: Sequence[Turn], raw: str,
             extra_vocabulary: str = "") -> Condensation:
    """Rewrite a follow-up into a standalone query, or fall back to the raw text.

    Turn 1 never reaches here: with no history there is nothing to condense, and
    calling the model anyway spends quota and risks distorting a perfectly good query.
    """
    from . import prompts

    if not history:
        return Condensation(raw, used_llm=False, drifted=False,
                            reason="turn 1 — nothing to condense")

    rendered = render_history(history)
    prompt = prompts.condense_prompt(rendered, raw, extra_vocabulary)

    try:
        completion = client.complete(
            prompt, purpose="condense", ladder="volume",
            schema=prompts.CONDENSE_SCHEMA,
        )
        candidate = ((completion.data or {}).get("standalone_query") or "").strip()
    except Exception as exc:
        # A condenser failure must never cost the turn. The raw text is always a
        # usable query -- just a worse one.
        return Condensation(raw, used_llm=False, drifted=False, fell_back=True,
                            reason=f"condenser unavailable ({type(exc).__name__})")

    if not candidate:
        return Condensation(raw, used_llm=True, drifted=False, fell_back=True,
                            reason="condenser returned nothing")

    allowed = content_words(rendered) | content_words(raw) | content_words(extra_vocabulary)
    novel = content_words(candidate) - allowed

    if novel:
        # The drift guard. Retrieving on a hallucinated term is worse than retrieving
        # on an under-specified one, so the raw query wins.
        return Condensation(raw, used_llm=True, drifted=True, novel_words=novel,
                            fell_back=True,
                            reason=f"introduced content words absent from the "
                                   f"conversation: {sorted(novel)}")

    return Condensation(candidate, used_llm=True, drifted=False,
                        reason="condensed within the conversation's vocabulary")


# ---------------------------------------------------------------------------
#  Semantic answer cache
# ---------------------------------------------------------------------------
def cache_lookup(conn: Connection, cfg: Config, fingerprint: str,
                 query_vec: np.ndarray) -> dict | None:
    """Return a stored answer for a near-paraphrase of this query, or None.

    The condensed query is already embedded for retrieval, so the cache key is free.
    Scoped to one corpus fingerprint: any re-ingest invalidates every entry, because
    the same question over different text is a different question.
    """
    rows = db.all_rows(
        conn, "SELECT cache_id, embedding, answer_json FROM answer_cache "
              "WHERE corpus_fingerprint = ?", (fingerprint,))
    if not rows:
        return None
    best, best_score = None, -1.0
    for row in rows:
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        if vec.shape != query_vec.shape:
            continue
        score = float(vec @ query_vec)
        if score > best_score:
            best, best_score = row, score
    if best is not None and best_score >= cfg.semantic_cache_threshold:
        return {"answer": json.loads(best["answer_json"]), "score": best_score}
    return None


def cache_store(conn: Connection, fingerprint: str, condensed: str,
                query_vec: np.ndarray, answer_json: dict) -> None:
    db.insert(conn, "answer_cache", {
        "cache_id": f"ac_{uuid.uuid4().hex[:12]}",
        "corpus_fingerprint": fingerprint, "condensed_query": condensed,
        "embedding": query_vec.astype(np.float32).tobytes(),
        "answer_json": json.dumps(answer_json),
    })
    conn.commit()


def record_user_turn(conn: Connection, session_id: str, ord_: int, raw: str,
                     condensed: str | None) -> str:
    turn_id = f"t_{uuid.uuid4().hex[:12]}"
    db.insert(conn, "turns", {
        "turn_id": turn_id, "session_id": session_id, "ord": ord_, "role": "user",
        "raw_text": raw, "condensed_query": condensed, "route": None,
        "n_retrieval_loops": 0,
    })
    conn.commit()
    return turn_id
