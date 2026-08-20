"""Build both retrieval indexes: bge-m3 dense vectors and SQLite FTS5.

Vectors live in a numpy sidecar rather than a SQLite extension. `sqlite-vec` needs a
loadable binary extension that is disabled on some Python builds -- and an install
failure costs far more points than a linear scan costs milliseconds. At this corpus
size the scan is exact, and it is HNSW that would be the approximation.

The FTS5 index needs no work here at all: triggers on `chunks` mirror every insert,
update and delete into `chunks_fts` inside the same transaction, so the two indexes
can never disagree about what the corpus contains.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Callable, Sequence

import numpy as np

from . import corpus, db
from .chunking import Chunk, chunk_document
from .config import Config
from .ingest import Page

FINGERPRINT_KEY = "corpus_fingerprint"
EMBED_MODEL_KEY = "embed_model"
DIMS_KEY = "dims"
BUILT_AT_KEY = "built_at"


@dataclass(frozen=True, slots=True)
class IndexStats:
    n_documents: int
    n_children: int
    n_parents: int
    dims: int
    seconds: float
    device: str
    peak_vram_gb: float
    fingerprint: str

    @property
    def chunks_per_second(self) -> float:
        return self.n_children / self.seconds if self.seconds else 0.0


# ---------------------------------------------------------------------------
#  Encoders
# ---------------------------------------------------------------------------
def load_embedder(cfg: Config):
    """Load bge-m3, and refuse to silently fall back to CPU.

    A CPU fallback here is the expensive failure: everything still works, roughly five
    times slower, with no error anywhere. `require_cuda` makes that loud.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if cfg.require_cuda and cfg.embed_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "EMBED_DEVICE is cuda but torch.cuda.is_available() is False. This is "
            "almost always a CPU-only torch wheel reporting success. Run "
            "`research-agent doctor --gpu` for the fix, or set REQUIRE_CUDA=false to "
            "proceed on CPU knowingly."
        )
    dtype = torch.float16 if cfg.embed_dtype == "float16" else torch.float32
    model = SentenceTransformer(
        cfg.embed_model,
        device=cfg.embed_device,
        model_kwargs={"torch_dtype": dtype},
    )
    model.max_seq_length = cfg.embed_max_tokens
    return model


def token_counter(model) -> Callable[[str], int]:
    """Count tokens with the embedding model's own tokenizer.

    Approximating by character count would let chunks silently exceed the encoder's
    window and be truncated -- losing exactly the evidence a citation points at.
    """
    tok = model.tokenizer

    def count(text: str) -> int:
        # `encode` warns when the result exceeds the model's window. Here we are
        # counting, not running inference, so that warning is noise -- scoped rather
        # than silenced globally, which would hide real ones.
        import logging

        logger = logging.getLogger("transformers.tokenization_utils_base")
        previous = logger.level
        logger.setLevel(logging.ERROR)
        try:
            return len(tok.encode(text, add_special_tokens=False))
        finally:
            logger.setLevel(previous)

    return count


def embed_texts(model, texts: Sequence[str], cfg: Config,
                on_progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Encode in batches, normalised, returned as float32."""
    out: list[np.ndarray] = []
    total = len(texts)
    for start in range(0, total, cfg.embed_batch_size):
        batch = list(texts[start:start + cfg.embed_batch_size])
        vecs = model.encode(
            batch, batch_size=len(batch), convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        out.append(np.asarray(vecs, dtype=np.float32))
        if on_progress:
            on_progress(min(start + len(batch), total), total)
    return np.vstack(out) if out else np.zeros((0, cfg.embed_dims), dtype=np.float32)


# ---------------------------------------------------------------------------
#  Fingerprint
# ---------------------------------------------------------------------------
def extracted_text_hashes(conn: Connection) -> list[str]:
    """One hash per document over the text the pipeline actually extracted.

    Hashing the PDF bytes instead is not enough, and this was caught the hard way:
    improving the extractor changed every chunk's text while leaving the source bytes
    identical, so the fingerprint did not move and nothing downstream knew the corpus
    had changed. Hashing the extracted text catches both a revised source and a
    revised parser.
    """
    rows = db.all_rows(
        conn, "SELECT doc_id, text FROM pages ORDER BY doc_id, page_no"
    )
    per_doc: dict[str, hashlib._Hash] = {}
    for r in rows:
        per_doc.setdefault(r["doc_id"], hashlib.sha256()).update(r["text"].encode("utf-8"))
    return [h.hexdigest() for h in per_doc.values()]


def compute_fingerprint(cfg: Config, doc_hashes: Sequence[str]) -> str:
    """Identity of "the corpus, extracted, chunked and embedded this particular way".

    Any change to the sources, the extraction, the chunk geometry, or the embedding
    model produces a different fingerprint. That is what invalidates the semantic
    answer cache and what tells a session its stored citations may no longer mean what
    they meant.
    """
    payload = json.dumps({
        "docs": sorted(doc_hashes),
        "child_tokens": cfg.child_tokens,
        "child_overlap_ratio": cfg.child_overlap_ratio,
        "parent_tokens": cfg.parent_tokens,
        "min_chunk_tokens": cfg.min_chunk_tokens,
        "embed_model": cfg.embed_model,
        "embed_max_tokens": cfg.embed_max_tokens,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_fingerprint(conn: Connection) -> str | None:
    return db.get_state(conn, FINGERPRINT_KEY)


def load_pages(conn: Connection, doc_id: str) -> list[Page]:
    rows = db.all_rows(
        conn,
        "SELECT page_no, n_columns, n_blocks, text FROM pages WHERE doc_id = ? "
        "ORDER BY page_no", (doc_id,),
    )
    return [Page(r["page_no"], r["text"], r["n_columns"], r["n_blocks"]) for r in rows]


# ---------------------------------------------------------------------------
#  Build
# ---------------------------------------------------------------------------
def build(
    cfg: Config,
    conn: Connection,
    on_event: Callable[[str], None] | None = None,
) -> IndexStats:
    """Chunk every ingested document, embed the children, write both indexes."""
    import torch

    log = on_event or (lambda _m: None)
    docs = db.all_rows(conn, "SELECT doc_id, title, sha256 FROM documents ORDER BY doc_id")
    if not docs:
        raise RuntimeError("No documents ingested. Run `ingest` first.")

    log(f"loading {cfg.embed_model} onto {cfg.embed_device} ({cfg.embed_dtype})")
    model = load_embedder(cfg)
    count = token_counter(model)
    device = str(next(model.parameters()).device)
    log(f"resolved device: {device}")

    # Rebuilding is a replace, not an append: leaving stale chunks behind would let a
    # gold label resolve to a chunk from a previous chunking scheme.
    conn.execute("DELETE FROM chunk_order")
    conn.execute("DELETE FROM chunks")
    conn.commit()

    started = time.monotonic()
    all_children: list[Chunk] = []
    n_parents = 0

    for row in docs:
        pages = load_pages(conn, row["doc_id"])
        chunks = chunk_document(row["doc_id"], pages, cfg, count)
        children = [c for c in chunks if c.level == 0]
        parents = [c for c in chunks if c.level == 1]

        # Parents first: children carry a parent_id foreign key.
        for c in parents + children:
            db.insert(conn, "chunks", {
                "chunk_id": c.chunk_id, "doc_id": c.doc_id, "parent_id": c.parent_id,
                "level": c.level, "ord": c.ord, "page_start": c.page_start,
                "page_end": c.page_end, "section": c.section,
                "token_count": c.token_count, "text": c.text,
            })
        conn.commit()
        all_children.extend(children)
        n_parents += len(parents)
        log(f"  {row['doc_id']:<12} {len(children):>4} children  {len(parents):>3} parents")

    log(f"embedding {len(all_children)} children, batch {cfg.embed_batch_size}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    vectors = embed_texts(
        model, [c.text for c in all_children], cfg,
        on_progress=lambda done, total: (
            log(f"  {done}/{total}") if done % (cfg.embed_batch_size * 50) == 0 else None
        ),
    )
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0

    # Row i of the .npy is chunk_order.row_idx i. Nothing else defines that mapping,
    # so it is written in the same transaction as the vectors are saved.
    cfg.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.embeddings_path, vectors)
    db.insert_many(conn, "chunk_order", [
        {"row_idx": i, "chunk_id": c.chunk_id} for i, c in enumerate(all_children)
    ])

    fingerprint = compute_fingerprint(cfg, extracted_text_hashes(conn))
    db.set_state(conn, FINGERPRINT_KEY, fingerprint)
    db.set_state(conn, EMBED_MODEL_KEY, cfg.embed_model)
    db.set_state(conn, DIMS_KEY, str(vectors.shape[1] if vectors.size else cfg.embed_dims))
    db.set_state(conn, BUILT_AT_KEY, time.strftime("%Y-%m-%dT%H:%M:%S"))
    conn.commit()

    return IndexStats(
        n_documents=len(docs), n_children=len(all_children), n_parents=n_parents,
        dims=int(vectors.shape[1]) if vectors.size else 0, seconds=elapsed,
        device=device, peak_vram_gb=peak, fingerprint=fingerprint,
    )


def load_vectors(cfg: Config) -> np.ndarray:
    if not cfg.embeddings_path.exists():
        raise RuntimeError(f"{cfg.embeddings_path} missing. Run `index` first.")
    return np.load(cfg.embeddings_path)


def check_staleness(cfg: Config, conn: Connection) -> str | None:
    """Return a human-readable reason the index is stale, or None."""
    stored = load_fingerprint(conn)
    if stored is None:
        return "no index has been built yet"
    current = compute_fingerprint(cfg, extracted_text_hashes(conn))
    if current != stored:
        return (f"corpus fingerprint changed ({stored} -> {current}): the sources, the "
                f"chunk geometry, or the embedding model moved since the index was built")
    n_rows = db.scalar(conn, "SELECT COUNT(*) FROM chunk_order") or 0
    if cfg.embeddings_path.exists():
        n_vecs = int(np.load(cfg.embeddings_path, mmap_mode="r").shape[0])
        if n_vecs != n_rows:
            return f"{n_vecs} vectors but {n_rows} chunk_order rows"
    else:
        return f"{cfg.embeddings_path.name} is missing"
    return None
