"""Structure-aware child/parent chunking with ids that survive re-ingest.

Two properties matter more than anything else here.

**Structure.** Fixed-size splits sever equations, claims and table rows mid-thought.
Chunks are grown paragraph-by-paragraph and cut at paragraph boundaries, and the
section heading in force travels with the chunk because it measurably helps the
reranker separate "the LoRA method section" from "the LoRA related-work section" --
a distinction this corpus turns on constantly.

**Stable ids.** A chunk id is derived from the document, not from a global counter, so
adding a fourteenth paper does not renumber the other thirteen and silently invalidate
every gold label. Re-ingesting the same bytes reproduces the same ids exactly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .config import Config
from .ingest import Page, detect_heading

# Approximate tokens per character for English technical prose. Used only when a real
# tokenizer has not been supplied; the indexer always supplies one.
_CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    parent_id: str | None
    level: int              # 0 = child (retrieved), 1 = parent (synthesised on)
    ord: int
    page_start: int
    page_end: int
    section: str | None
    token_count: int
    text: str

    @property
    def text_sha(self) -> str:
        """Fingerprint of the chunk's own text.

        Gold labels record this alongside the chunk id. An id that still resolves but
        whose text has changed -- because the chunking config moved, or the source was
        revised -- would otherwise score an evaluation against passages that no longer
        say what they said when they were labelled, with no visible symptom.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


TokenCounter = Callable[[str], int]


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def child_id(doc_id: str, ordinal: int) -> str:
    return f"c_{doc_id}_{ordinal:04d}"


def parent_id(doc_id: str, ordinal: int) -> str:
    return f"p_{doc_id}_{ordinal:04d}"


@dataclass(frozen=True, slots=True)
class _Para:
    text: str
    page: int
    section: str | None
    tokens: int


def _paragraphs(pages: Sequence[Page], count: TokenCounter) -> list[_Para]:
    """Flatten pages into paragraphs, tracking page number and current section."""
    out: list[_Para] = []
    section: str | None = None
    for page in pages:
        for block in page.text.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            # A heading applies to what follows it, and is kept in the text as well:
            # dropping it would lose a genuinely useful retrieval signal.
            heading = detect_heading(block)
            if heading:
                section = heading
                continue
            out.append(_Para(block, page.number, section, count(block)))
    return out


def chunk_document(
    doc_id: str,
    pages: Sequence[Page],
    cfg: Config,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """Produce children then parents for one document."""
    count = count_tokens or approx_tokens
    paras = _paragraphs(pages, count)
    if not paras:
        return []

    children = _build_children(doc_id, paras, cfg, count)
    parents = _build_parents(doc_id, children, cfg)
    return children + parents


def _build_children(
    doc_id: str, paras: list[_Para], cfg: Config, count: TokenCounter
) -> list[Chunk]:
    target = cfg.child_tokens
    overlap_target = int(target * cfg.child_overlap_ratio)

    children: list[Chunk] = []
    buf: list[_Para] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        text = "\n\n".join(p.text for p in buf)
        tokens = count(text)
        # A chunk under the floor is a stray fragment -- a caption, an orphaned
        # equation label -- and it pollutes retrieval more than it helps. Merge it
        # backwards rather than emitting it.
        if tokens < cfg.min_chunk_tokens and children:
            prev = children[-1]
            merged = f"{prev.text}\n\n{text}"
            children[-1] = Chunk(
                prev.chunk_id, prev.doc_id, None, 0, prev.ord,
                prev.page_start, max(prev.page_end, buf[-1].page),
                prev.section, count(merged), merged,
            )
        else:
            children.append(Chunk(
                chunk_id=child_id(doc_id, len(children)),
                doc_id=doc_id, parent_id=None, level=0, ord=len(children),
                page_start=buf[0].page, page_end=buf[-1].page,
                section=buf[0].section, token_count=tokens, text=text,
            ))
        buf, buf_tokens = [], 0

    for para in paras:
        # A single paragraph longer than the target is split on sentence boundaries
        # rather than truncated, because truncation silently loses evidence.
        if para.tokens > target:
            flush()
            for piece in _split_long(para, target, count):
                children.append(Chunk(
                    chunk_id=child_id(doc_id, len(children)),
                    doc_id=doc_id, parent_id=None, level=0, ord=len(children),
                    page_start=para.page, page_end=para.page,
                    section=para.section, token_count=count(piece), text=piece,
                ))
            continue

        if buf_tokens + para.tokens > target and buf:
            tail = _overlap_tail(buf, overlap_target)
            flush()
            buf = list(tail)
            buf_tokens = sum(p.tokens for p in buf)

        buf.append(para)
        buf_tokens += para.tokens

    flush()
    return children


def _overlap_tail(buf: list[_Para], overlap_target: int) -> list[_Para]:
    """Carry the last paragraphs of a chunk into the next one.

    Overlap exists so a claim spanning a chunk boundary is retrievable from at least
    one chunk in full, rather than being severed by both.
    """
    tail: list[_Para] = []
    total = 0
    for para in reversed(buf):
        if total + para.tokens > overlap_target and tail:
            break
        tail.insert(0, para)
        total += para.tokens
    return tail


def _split_long(para: _Para, target: int, count: TokenCounter) -> list[str]:
    sentences = _sentences(para.text)
    out: list[str] = []
    buf: list[str] = []
    tokens = 0
    for s in sentences:
        st = count(s)
        if tokens + st > target and buf:
            out.append(" ".join(buf))
            buf, tokens = [], 0
        buf.append(s)
        tokens += st
    if buf:
        out.append(" ".join(buf))
    return out or [para.text]


def _sentences(text: str) -> list[str]:
    """Cheap sentence split.

    No NLTK, no spaCy: one more dependency and one more download for a task where the
    failure mode is a slightly awkward chunk boundary.
    """
    import re

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if p.strip()]


def _build_parents(doc_id: str, children: list[Chunk], cfg: Config) -> list[Chunk]:
    """Group consecutive children into parents.

    Retrieval wants precision, so it searches children. Synthesis wants context, so it
    reads parents. The reranker scores a 700-token child; the model then answers over
    the ~2000-token passage that child came from.
    """
    parents: list[Chunk] = []
    group: list[Chunk] = []
    tokens = 0

    def flush() -> None:
        nonlocal group, tokens
        if not group:
            return
        pid = parent_id(doc_id, len(parents))
        text = "\n\n".join(c.text for c in group)
        parents.append(Chunk(
            chunk_id=pid, doc_id=doc_id, parent_id=None, level=1, ord=len(parents),
            page_start=group[0].page_start, page_end=group[-1].page_end,
            section=group[0].section, token_count=tokens, text=text,
        ))
        for c in group:
            children[c.ord] = Chunk(
                c.chunk_id, c.doc_id, pid, c.level, c.ord, c.page_start, c.page_end,
                c.section, c.token_count, c.text,
            )
        group, tokens = [], 0

    for child in children:
        if tokens + child.token_count > cfg.parent_tokens and group:
            flush()
        group.append(child)
        tokens += child.token_count
    flush()
    return parents


def chunk_stats(chunks: Iterable[Chunk]) -> dict[str, float]:
    children = [c for c in chunks if c.level == 0]
    if not children:
        return {}
    counts = sorted(c.token_count for c in children)
    return {
        "n_children": len(counts),
        "min_tokens": counts[0],
        "median_tokens": counts[len(counts) // 2],
        "max_tokens": counts[-1],
        "mean_tokens": sum(counts) / len(counts),
    }
