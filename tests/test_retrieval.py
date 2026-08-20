"""Ingestion, chunking and retrieval. No network, no model downloads."""

from __future__ import annotations

import pytest

from research_agent import db
from research_agent.chunking import chunk_document, child_id, chunk_document as _cd
from research_agent.config import Config
from research_agent.ingest import (
    Block, Page, clean_text, detect_columns, detect_heading, reading_order,
)
from research_agent.retrieve import build_fts_query, reciprocal_rank_fusion, Hit


class FakeRect:
    """Stands in for pymupdf.Rect so layout logic is testable without a PDF."""

    def __init__(self, width=612.0, height=792.0):
        self.width, self.height = width, height
        self.x0, self.y0 = 0.0, 0.0
        self.x1, self.y1 = width, height


def _col_block(side: str, y: float, chars: int = 400, rect=None) -> Block:
    rect = rect or FakeRect()
    x0 = 60.0 if side == "L" else 320.0
    return Block(x0, y, x0 + 230.0, y + 90.0, "word " * (chars // 5))


# ---------------------------------------------------------------------------
#  Text repair
# ---------------------------------------------------------------------------
def test_hyphen_break_is_repaired_but_real_compounds_survive():
    assert "measuring" in clean_text("measur-\ning the similarity")
    # This corpus is full of genuine hyphenated compounds; joining unconditionally
    # would corrupt every one of them.
    assert "low-rank" in clean_text("we use low-rank decomposition")
    assert "fine-tuning" in clean_text("full fine-tuning is costly")


def test_caps_hyphen_break_is_repaired():
    """Titles are typeset in caps, so the lowercase rule alone misses them."""
    assert clean_text("LARGE LAN-\nGUAGE MODELS") == "LARGE LANGUAGE MODELS"


def test_ligatures_are_expanded():
    assert clean_text("ﬁne-tuning the ﬂow") == "fine-tuning the flow"


# ---------------------------------------------------------------------------
#  Column detection
# ---------------------------------------------------------------------------
def test_two_parallel_prose_columns_are_detected():
    rect = FakeRect()
    blocks = [_col_block(s, y) for y in (100, 220, 340) for s in ("L", "R")]
    assert detect_columns(blocks, rect) == 2


def test_single_column_page_with_a_scattered_equation_is_not_two_columns():
    """The bug this heuristic exists to prevent.

    A displayed equation shatters into a dozen tiny fragments either side of the
    midline. Counting blocks reads that as a two-column layout and reorders the page;
    weighting by character mass ignores it.
    """
    rect = FakeRect()
    body = [Block(108, y, 504, y + 60, "word " * 100) for y in (100, 200, 300, 400)]
    fragments = [Block(x, 250, x + 18, 268, s) for x, s in
                 [(233, "max"), (240, "Φ"), (261, "X"), (284, "|y|"), (298, "t=1"),
                  (320, "log"), (350, "PΦ"), (380, "yt|x")]]
    assert detect_columns(body + fragments, rect) == 1


def test_a_single_dense_sidebar_is_not_a_column():
    rect = FakeRect()
    blocks = [_col_block("L", y) for y in (100, 220, 340)]
    blocks.append(Block(320, 100, 550, 130, "note " * 20))
    assert detect_columns(blocks, rect) == 1


def test_reading_order_reads_left_column_fully_before_right():
    rect = FakeRect()
    left = [Block(60, y, 290, y + 80, f"L{y}") for y in (100, 200, 300)]
    right = [Block(320, y, 550, y + 80, f"R{y}") for y in (100, 200, 300)]
    ordered = reading_order(left + right, rect, n_columns=2)
    labels = [b.text for b in ordered]
    assert labels == ["L100", "L200", "L300", "R100", "R200", "R300"], (
        "sorting by (y, x) instead is exactly the bug that interleaves columns"
    )


def test_single_column_reading_order_is_top_to_bottom():
    rect = FakeRect()
    blocks = [Block(108, y, 504, y + 60, f"B{y}") for y in (300, 100, 200)]
    assert [b.text for b in reading_order(blocks, rect, 1)] == ["B100", "B200", "B300"]


# ---------------------------------------------------------------------------
#  Headings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("line,expected", [
    ("5.5 SCALING UP TO GPT-3", "5.5 SCALING UP TO GPT-3"),
    ("D.4 GPT-3", "D.4 GPT-3"),
    ("A.1 Training Details", "A.1 Training Details"),
    ("References", "References"),
    ("B Additional Results", "B Additional Results"),
])
def test_headings_are_detected(line, expected):
    assert detect_heading(line) == expected


@pytest.mark.parametrize("line", [
    "A model trained on FLAN v2 with a batch size of one produces the following",
    "We describe the simple design of LoRA and its practical benefits.",
    "",
])
def test_prose_is_not_mistaken_for_a_heading(line):
    assert detect_heading(line) is None


# ---------------------------------------------------------------------------
#  Chunking
# ---------------------------------------------------------------------------
def _pages(n_paras: int, words: int = 200) -> list[Page]:
    para = " ".join(f"w{i}" for i in range(words))
    return [Page(1, "\n\n".join([para] * n_paras), 1, n_paras)]


def test_chunk_ids_are_document_local_and_stable(cfg: Config):
    pages = _pages(12)
    a = chunk_document("lora", pages, cfg)
    b = chunk_document("lora", pages, cfg)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b], "re-ingest must reproduce ids"
    assert all(c.chunk_id.startswith(("c_lora_", "p_lora_")) for c in a)

    # Adding another document must not renumber this one -- that is what would
    # silently invalidate every gold label.
    other = chunk_document("bert", pages, cfg)
    assert not ({c.chunk_id for c in a} & {c.chunk_id for c in other})


def test_children_respect_the_token_target(cfg: Config):
    children = [c for c in chunk_document("d", _pages(20), cfg) if c.level == 0]
    assert children
    assert all(c.token_count <= cfg.child_tokens * 1.3 for c in children)


def test_every_child_is_linked_to_a_parent(cfg: Config):
    chunks = chunk_document("d", _pages(20), cfg)
    children = [c for c in chunks if c.level == 0]
    parents = {c.chunk_id for c in chunks if c.level == 1}
    assert parents
    assert all(c.parent_id in parents for c in children)


def test_text_sha_changes_when_the_text_changes(cfg: Config):
    """Gold labels record this so a chunk id that still resolves but whose text has
    moved cannot silently score an evaluation against the wrong passage."""
    a = chunk_document("d", _pages(6), cfg)[0]
    b = chunk_document("d", _pages(6, words=210), cfg)[0]
    assert a.text_sha != b.text_sha


def test_pages_are_tracked_onto_chunks(cfg: Config):
    pages = [Page(1, "alpha " * 400, 1, 1), Page(2, "beta " * 400, 1, 1)]
    chunks = chunk_document("d", pages, cfg)
    assert all(c.page_start >= 1 for c in chunks)
    assert max(c.page_end for c in chunks) == 2


# ---------------------------------------------------------------------------
#  FTS5 query construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hostile", [
    'what is "attention"?',
    "rank * from chunks",
    "LoRA NEAR BERT",
    "what about (a) or (b)?",
    "^caret and -minus",
    'unbalanced " quote',
])
def test_hostile_input_produces_a_valid_fts_query(hostile, conn):
    """An unescaped quote, asterisk or a bare NEAR raises OperationalError mid-turn.

    This is a live crash path, not a theoretical one, so it is exercised against a
    real FTS5 table rather than merely inspected.
    """
    match = build_fts_query(hostile)
    if match:
        conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (match,)).fetchall()


def test_compound_identifiers_become_phrase_queries():
    """unicode61 splits GPT-3 into `gpt` + `3`, so a bare term would match any 3."""
    q = build_fts_query("What rank does LoRA use for GPT-3?")
    assert '"GPT 3"' in q
    assert '"LoRA"' in q


def test_stopwords_are_dropped():
    q = build_fts_query("what is the of for a an")
    assert q == ""


# ---------------------------------------------------------------------------
#  Fusion
# ---------------------------------------------------------------------------
def _hit(cid: str, rank: int) -> Hit:
    return Hit(cid, "d", "T", 1, 1, None, "text", rank=rank)


def test_rrf_rewards_agreement_between_the_two_retrievers(cfg: Config):
    dense = [_hit("a", 1), _hit("b", 2), _hit("c", 3)]
    sparse = [_hit("c", 1), _hit("b", 2), _hit("z", 3)]
    fused = reciprocal_rank_fusion(dense, sparse, cfg)
    ids = [h.chunk_id for h in fused]
    # b and c appear in both lists; z appears in one, at rank 3.
    assert ids[0] in {"b", "c"}
    assert ids.index("z") > ids.index("b")
    assert set(ids) == {"a", "b", "c", "z"}


def test_rrf_uses_rank_only_and_ignores_score_magnitude(cfg: Config):
    """The reason RRF was chosen: no normalisation between incomparable scorers."""
    huge = [Hit("a", "d", "T", 1, 1, None, "t", dense_score=999.0, rank=1)]
    tiny = [Hit("b", "d", "T", 1, 1, None, "t", sparse_score=0.0001, rank=1)]
    fused = reciprocal_rank_fusion(huge, tiny, cfg)
    assert abs(fused[0].rrf_score - fused[1].rrf_score) < 1e-12
