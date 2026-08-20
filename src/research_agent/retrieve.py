"""Dense and sparse retrieval, and the fusion between them.

Step 4 delivers the two retrievers side by side so their behaviour can be judged by
eye before any LLM is in the loop. Step 5 adds RRF and reranking on top.

The FTS5 half carries a real hazard the dense half does not: user text is not a query
language. An unescaped quote, asterisk, caret or a bare `NEAR` raises
`sqlite3.OperationalError` in the middle of a turn. Every query string is therefore
tokenised and rebuilt rather than passed through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from sqlite3 import Connection
from typing import Sequence

import numpy as np

from . import db
from .config import Config

# FTS5 reserves these as query syntax. A question containing any of them is ordinary
# English, not an operator, so they are stripped rather than escaped.
_FTS_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")

# unicode61 splits on non-alphanumerics, so an identifier like GPT-3 or bge-m3 becomes
# separate tokens and matches any stray "3". Emitting those as phrases restores the
# precision the tokenizer throws away (decisions.md D-103).
_COMPOUND = re.compile(r"^[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)+$")

_STOPWORDS = frozenset("""
a an the of for to in on at by with from is are was were be been being do does did
what which who whom whose how why when where and or but if then than that this these
those it its as into about can could should would may might will shall
""".split())


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    page_start: int
    page_end: int
    section: str | None
    text: str
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    rank: int = 0

    @property
    def source_label(self) -> str:
        """How a citation renders: "Paper Title · p.N"."""
        pages = (f"p.{self.page_start}" if self.page_start == self.page_end
                 else f"pp.{self.page_start}-{self.page_end}")
        return f"{self.title} · {pages}"


# ---------------------------------------------------------------------------
#  Sparse
# ---------------------------------------------------------------------------
def build_fts_query(text: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    Returns an OR of terms, with hyphenated identifiers emitted as quoted phrases so
    "GPT-3" cannot match a bare "3".
    """
    terms: list[str] = []
    for token in _FTS_TOKEN.findall(text):
        if token.lower() in _STOPWORDS or len(token) < 2:
            continue
        if _COMPOUND.match(token):
            # Quote the parts as a phrase: unicode61 will split them anyway, and a
            # phrase query requires them adjacent and in order.
            parts = re.split(r"[-_.]", token)
            terms.append('"' + " ".join(p for p in parts if p) + '"')
        else:
            terms.append(f'"{token}"')
    return " OR ".join(dict.fromkeys(terms))


def sparse_search(conn: Connection, query: str, cfg: Config, k: int | None = None) -> list[Hit]:
    """FTS5 bm25(). Lower is better in SQLite, so the sign is flipped on the way out."""
    match = build_fts_query(query)
    if not match:
        return []
    limit = k or cfg.sparse_top_k
    rows = db.all_rows(conn, """
        SELECT c.chunk_id, c.doc_id, d.title, c.page_start, c.page_end, c.section,
               c.text, bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE chunks_fts MATCH ? AND c.level = 0
        ORDER BY score
        LIMIT ?
    """, (match, limit))
    return [
        Hit(r["chunk_id"], r["doc_id"], r["title"], r["page_start"], r["page_end"],
            r["section"], r["text"], sparse_score=-float(r["score"]), rank=i + 1)
        for i, r in enumerate(rows)
    ]


# ---------------------------------------------------------------------------
#  Dense
# ---------------------------------------------------------------------------
def dense_search(
    conn: Connection, query_vec: np.ndarray, vectors: np.ndarray,
    cfg: Config, k: int | None = None,
) -> list[Hit]:
    """Exact cosine over the whole corpus.

    Vectors are already L2-normalised, so a dot product is the cosine. At this size the
    full scan is milliseconds -- and it is exact, where an ANN index would not be.
    """
    limit = k or cfg.dense_top_k
    if vectors.size == 0:
        return []
    scores = vectors @ query_vec.astype(np.float32)
    top = np.argsort(-scores)[:limit]

    order = {r["row_idx"]: r["chunk_id"] for r in
             db.all_rows(conn, "SELECT row_idx, chunk_id FROM chunk_order")}
    ids = [order[int(i)] for i in top if int(i) in order]
    if not ids:
        return []

    rows = {r["chunk_id"]: r for r in db.all_rows(conn, f"""
        SELECT c.chunk_id, c.doc_id, d.title, c.page_start, c.page_end, c.section, c.text
        FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.chunk_id IN ({','.join('?' * len(ids))})
    """, ids)}
    return [
        Hit(cid, rows[cid]["doc_id"], rows[cid]["title"], rows[cid]["page_start"],
            rows[cid]["page_end"], rows[cid]["section"], rows[cid]["text"],
            dense_score=float(scores[int(idx)]), rank=i + 1)
        for i, (idx, cid) in enumerate(zip(top, ids)) if cid in rows
    ]


def embed_query(model, text: str) -> np.ndarray:
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)
    return np.asarray(vec[0], dtype=np.float32)


# ---------------------------------------------------------------------------
#  Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    dense: Sequence[Hit], sparse: Sequence[Hit], cfg: Config
) -> list[Hit]:
    """Fuse two ranked lists by rank position alone.

    Dense cosine and BM25 live on incomparable scales. Normalising them makes the
    fusion weight corpus-dependent -- retune per corpus or it silently degrades. RRF
    uses only rank, so there is no normalisation and one stable hyperparameter. The
    cost is that score magnitude is discarded, which is acceptable precisely because
    the cross-encoder re-scores the fused candidates anyway.
    """
    merged: dict[str, Hit] = {}
    scores: dict[str, float] = {}

    for hits in (dense, sparse):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (cfg.rrf_k + rank)
            if hit.chunk_id in merged:
                prev = merged[hit.chunk_id]
                merged[hit.chunk_id] = replace(
                    prev,
                    dense_score=prev.dense_score if prev.dense_score is not None else hit.dense_score,
                    sparse_score=prev.sparse_score if prev.sparse_score is not None else hit.sparse_score,
                )
            else:
                merged[hit.chunk_id] = hit

    ordered = sorted(merged.values(), key=lambda h: -scores[h.chunk_id])
    return [replace(h, rrf_score=scores[h.chunk_id], rank=i + 1)
            for i, h in enumerate(ordered)]


def expand_to_parents(conn: Connection, hits: Sequence[Hit]) -> list[Hit]:
    """Swap each child for the parent passage it came from, de-duplicated.

    Retrieval wants precision; synthesis wants context. The reranker scores a ~700-token
    child, and the model then answers over the ~2000-token passage containing it.
    """
    out: list[Hit] = []
    seen: set[str] = set()
    for hit in hits:
        row = db.one(conn, """
            SELECT p.chunk_id, p.doc_id, d.title, p.page_start, p.page_end, p.section, p.text
            FROM chunks c JOIN chunks p ON p.chunk_id = c.parent_id
            JOIN documents d ON d.doc_id = p.doc_id
            WHERE c.chunk_id = ?
        """, (hit.chunk_id,))
        if row is None:
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                out.append(hit)
            continue
        if row["chunk_id"] in seen:
            continue
        seen.add(row["chunk_id"])
        # Scores stay attached to the child that earned them; the parent inherits them
        # for reporting, so an audit can always trace which child won the slot.
        out.append(replace(hit, chunk_id=row["chunk_id"], text=row["text"],
                           page_start=row["page_start"], page_end=row["page_end"],
                           section=row["section"]))
    return out
