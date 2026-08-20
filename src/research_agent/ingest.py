"""PDF -> ordered text with page numbers citations can rely on.

The whole project rests on this file being right. Naive `page.get_text()` on a
two-column arXiv paper reads across the gutter and interleaves the columns: the result
is fluent-looking prose that is semantically destroyed, passes every automated check,
and quietly wrecks retrieval. So blocks are extracted with geometry, assigned to
columns, and emitted in true reading order.

Deliberately not handled, and documented rather than chased: equations, figures, and
multi-page tables extract poorly. There is no OCR here and never will be -- a document
that does not extract as text gets replaced, not OCR'd.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pymupdf

from .config import Config

# Fraction of page height treated as running header / footer. arXiv page furniture
# (page numbers, conference banners, preprint stamps) lives here.
HEADER_BAND = 0.06
FOOTER_BAND = 0.94

# A block must be at least this wide, relative to the page's text width, to count as
# spanning the gutter. Guards against a wide-ish equation reading as full-width.
FULL_WIDTH_RATIO = 0.62

# Minimum share of *character mass* that must sit cleanly on one side of the midline
# before a page is called two-column. Character mass rather than block count is
# load-bearing: a single-column page carrying a displayed equation shatters into a
# dozen tiny fragments scattered either side of the midline, and counting blocks reads
# that as a two-column layout. Those fragments carry almost no text, so weighting by
# characters ignores them.
TWO_COLUMN_CONFIDENCE = 0.60

# A block only votes on layout if it holds at least this much text. Figure labels,
# axis ticks and equation fragments are evidence of nothing.
MIN_VOTING_CHARS = 80

# A genuine column block is roughly half the text width. Anything far outside this
# band is a fragment, an inset, or a full-width paragraph.
COLUMN_WIDTH_BAND = (0.30, 0.62)


@dataclass(frozen=True, slots=True)
class Page:
    number: int          # 1-based, matches what a reader sees and what we cite
    text: str
    n_columns: int
    n_blocks: int


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    path: Path
    sha256: str
    pages: list[Page]

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def extracted_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


@dataclass(frozen=True, slots=True)
class Block:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0


# ---------------------------------------------------------------------------
#  Text repair
# ---------------------------------------------------------------------------
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}
# Only join a hyphen that ends a line and is followed by a lowercase continuation.
# Joining unconditionally would corrupt genuine compounds split across lines, and
# this corpus is full of them ("low-rank", "fine-tuning", "state-of-the-art").
_HYPHEN_BREAK = re.compile(r"(\w)-\n([a-z])")
_SOFT_BREAK = re.compile(r"(?<![.!?:;])\n(?=[a-z(])")
_MULTI_BLANK = re.compile(r"\n{3,}")
_PAGE_NUMBER_ONLY = re.compile(r"^\s*\d{1,3}\s*$")
_LINE_NUMBER_ONLY = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)


def clean_text(raw: str) -> str:
    """Repair the artifacts PDF extraction reliably introduces."""
    text = unicodedata.normalize("NFKC", raw)
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    text = text.replace("­", "")          # soft hyphen
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    # Rejoin lines the PDF broke mid-sentence, so chunks contain real sentences
    # rather than typesetting artifacts.
    text = _SOFT_BREAK.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
#  Layout
# ---------------------------------------------------------------------------
def _blocks(page: "pymupdf.Page") -> list[Block]:
    out: list[Block] = []
    for x0, y0, x1, y1, text, _no, btype in page.get_text("blocks"):
        if btype != 0:          # 1 == image
            continue
        if not text.strip():
            continue
        out.append(Block(x0, y0, x1, y1, text))
    return out


def _strip_furniture(blocks: list[Block], page_rect: "pymupdf.Rect") -> list[Block]:
    """Drop running headers, footers, page numbers, and the arXiv margin stamp."""
    h, w = page_rect.height, page_rect.width
    kept: list[Block] = []
    for b in blocks:
        body = b.text.strip()

        # The vertical "arXiv:xxxx [cs.CL] date" stamp on page 1: a very narrow,
        # very tall block pinned to the left margin.
        if b.width < w * 0.08 and (b.y1 - b.y0) > h * 0.25:
            continue
        if b.y1 < h * HEADER_BAND and len(body) < 120:
            continue
        if b.y0 > h * FOOTER_BAND and len(body) < 120:
            continue
        if _PAGE_NUMBER_ONLY.match(body):
            continue
        kept.append(b)
    return kept


def text_span(blocks: list[Block], page_rect: "pymupdf.Rect") -> tuple[float, float]:
    """The horizontal extent actually occupied by text, ignoring page margins.

    Measuring against the physical page width would make every threshold depend on
    the margin size, which differs between conference templates.
    """
    if not blocks:
        return page_rect.x0, page_rect.x1
    return min(b.x0 for b in blocks), max(b.x1 for b in blocks)


def detect_columns(blocks: list[Block], page_rect: "pymupdf.Rect") -> int:
    """Return 1 or 2, weighting the evidence by character mass.

    Deliberately conservative: when the evidence is weak the answer is 1. Both errors
    hurt, but they hurt differently -- wrongly merging a real two-column page produces
    fluent word salad that passes every automated check, while wrongly splitting a
    single-column page mostly reshuffles figure fragments. The bar for calling a page
    two-column is therefore evidence of *sustained prose* in two parallel columns.
    """
    x_left, x_right = text_span(blocks, page_rect)
    text_width = x_right - x_left
    if text_width <= 0:
        return 1
    mid = (x_left + x_right) / 2
    lo, hi = COLUMN_WIDTH_BAND

    left_mass = right_mass = other_mass = 0
    for b in blocks:
        mass = len(b.text.strip())
        if mass < MIN_VOTING_CHARS:
            continue
        ratio = b.width / text_width
        if not (lo <= ratio <= hi):
            other_mass += mass          # full-width paragraph, or a narrow inset
            continue
        if b.x1 <= mid + text_width * 0.03:
            left_mass += mass
        elif b.x0 >= mid - text_width * 0.03:
            right_mass += mass
        else:
            other_mass += mass          # straddles the gutter, so there isn't one

    sided = left_mass + right_mass
    total = sided + other_mass
    if total == 0 or left_mass == 0 or right_mass == 0:
        return 1
    # Both columns must carry real weight; one dense sidebar is not a column.
    balance = min(left_mass, right_mass) / max(left_mass, right_mass)
    if balance < 0.25:
        return 1
    return 2 if sided / total >= TWO_COLUMN_CONFIDENCE else 1


def reading_order(blocks: list[Block], page_rect: "pymupdf.Rect", n_columns: int) -> list[Block]:
    """Sort blocks the way a human reads them.

    One column: plain top-to-bottom. Two columns: full-width blocks (title, abstract,
    wide figures and tables) split the page into horizontal bands, and within each
    band the entire left column is read before the right. Sorting by (y, x) instead --
    which is what `sort=True` does -- is exactly the bug that produces readable-looking
    word salad.
    """
    if n_columns == 1:
        return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))


    x_left, x_right = text_span(blocks, page_rect)
    text_width = max(x_right - x_left, 1.0)
    mid = (x_left + x_right) / 2
    full = [b for b in blocks if b.width > text_width * FULL_WIDTH_RATIO]
    columnar = [b for b in blocks if b not in full]

    # Band boundaries are the vertical extents of the full-width blocks.
    boundaries = sorted((b.y0, b.y1) for b in full)
    ordered: list[Block] = []
    cursor = page_rect.y0

    for y0, y1 in boundaries:
        band = [b for b in columnar if cursor <= b.y0 < y0]
        ordered.extend(_order_band(band, mid))
        ordered.extend(b for b in full if b.y0 == y0 and b.y1 == y1)
        cursor = max(cursor, y1)

    ordered.extend(_order_band([b for b in columnar if b.y0 >= cursor], mid))
    return ordered


def _order_band(band: list[Block], mid: float) -> list[Block]:
    left = sorted((b for b in band if b.cx < mid), key=lambda b: b.y0)
    right = sorted((b for b in band if b.cx >= mid), key=lambda b: b.y0)
    return left + right


# ---------------------------------------------------------------------------
#  Section headings
# ---------------------------------------------------------------------------
_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,80})\s*$")
_NAMED_HEADING = re.compile(
    r"^\s*(Abstract|Introduction|Related Work|Background|Method|Methods|Approach|"
    r"Experiments|Experimental Setup|Results|Discussion|Conclusion|Conclusions|"
    r"Limitations|References|Appendix)\s*$",
    re.IGNORECASE,
)


def detect_heading(line: str) -> str | None:
    """Return a normalised section heading, or None.

    Section context is carried into chunk metadata because it measurably helps the
    reranker distinguish "the LoRA method section" from "the LoRA related-work
    section" -- a distinction this corpus turns on constantly.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None
    m = _NUMBERED_HEADING.match(stripped)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    m = _NAMED_HEADING.match(stripped)
    if m:
        return m.group(1).title()
    return None


# ---------------------------------------------------------------------------
#  Extraction
# ---------------------------------------------------------------------------
def extract_pdf(path: Path) -> list[Page]:
    """Page-streamed extraction. Memory stays flat regardless of document size."""
    pages: list[Page] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            raw_blocks = _blocks(page)
            body = _strip_furniture(raw_blocks, page.rect)
            n_cols = detect_columns(body, page.rect)
            ordered = reading_order(body, page.rect, n_cols)
            text = clean_text("\n".join(b.text for b in ordered))
            pages.append(Page(number=i, text=text, n_columns=n_cols, n_blocks=len(ordered)))
    return pages


def extract_text_file(path: Path) -> list[Page]:
    """MD and TXT sources. One page, because there is no pagination to cite."""
    return [Page(number=1, text=clean_text(path.read_text(encoding="utf-8", errors="replace")),
                 n_columns=1, n_blocks=1)]


def extract(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".md", ".txt"}:
        return extract_text_file(path)
    raise ValueError(f"Unsupported source type {suffix!r} for {path.name}. "
                     f"Supported: .pdf, .md, .txt. There is no OCR path.")


def column_profile(pages: Iterable[Page]) -> dict[int, int]:
    """How many pages were read as one column vs two -- printed at the Step 3 gate."""
    profile: dict[int, int] = {}
    for p in pages:
        profile[p.n_columns] = profile.get(p.n_columns, 0) + 1
    return profile
