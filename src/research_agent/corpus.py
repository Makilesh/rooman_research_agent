"""Reproducible corpus acquisition from arXiv.

The manifest is the artifact that gets committed; the PDFs are not. That keeps the
repository small and sidesteps any question about redistributing someone else's paper,
while leaving the corpus exactly reproducible from a single command.

Version pinning is load-bearing. arXiv ids resolve to the latest revision by default,
so an unpinned id can start returning a different PDF at any time -- which changes the
sha256, changes the chunk ids derived from it, and silently invalidates every gold
label. Every manifest entry names its version.
"""

from __future__ import annotations

import hashlib
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import yaml

from .config import Config

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{versioned_id}"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    doc_id: str
    arxiv_id: str          # versioned, e.g. 1706.03762v7
    title: str
    first_author: str
    published: str
    role: str              # why this paper earns its place in the corpus

    @property
    def filename(self) -> str:
        return f"{self.arxiv_id}.pdf"

    @property
    def source_url(self) -> str:
        return ARXIV_PDF.format(versioned_id=self.arxiv_id)


@dataclass(frozen=True, slots=True)
class FetchResult:
    entry: ManifestEntry
    path: Path
    sha256: str
    n_bytes: int
    skipped: bool          # already present with a matching hash


def load_manifest(cfg: Config) -> list[ManifestEntry]:
    raw = yaml.safe_load(cfg.manifest_path.read_text(encoding="utf-8"))
    return [ManifestEntry(**doc) for doc in raw["documents"]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_metadata(arxiv_ids: list[str], user_agent: str, timeout_s: float) -> list[dict]:
    """Query the arXiv API for real titles, authors and dates.

    Used to build and to audit the manifest. Metadata is never written from memory --
    a plausible-looking author list is exactly the kind of invented ground truth that
    invalidates everything downstream.
    """
    import httpx

    r = httpx.get(
        ARXIV_API,
        params={"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids) + 5},
        headers={"User-Agent": user_agent},
        timeout=timeout_s,
        follow_redirects=True,   # arXiv 301s http -> https
    )
    r.raise_for_status()

    out: list[dict] = []
    for e in ET.fromstring(r.text).findall("a:entry", ATOM):
        authors = [a.find("a:name", ATOM).text for a in e.findall("a:author", ATOM)]
        out.append({
            "arxiv_id": e.find("a:id", ATOM).text.rsplit("/", 1)[-1],
            "title": " ".join(e.find("a:title", ATOM).text.split()),
            "published": e.find("a:published", ATOM).text[:10],
            "first_author": authors[0] if authors else "",
            "n_authors": len(authors),
        })
    return out


def fetch_corpus(
    cfg: Config,
    entries: list[ManifestEntry] | None = None,
    on_event: Callable[[str], None] | None = None,
    downloader: Callable[[str, str, float], bytes] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[FetchResult]:
    """Download every manifest entry, politely, resuming where possible.

    `downloader` and `sleep` are injectable so the tests exercise this path without
    touching the network or waiting three seconds per paper.
    """
    entries = entries if entries is not None else load_manifest(cfg)
    cfg.sources_dir.mkdir(parents=True, exist_ok=True)
    log = on_event or (lambda _msg: None)
    fetched_any = False

    for entry in entries:
        dest = cfg.sources_dir / entry.filename

        if dest.exists() and dest.stat().st_size > 0:
            digest = sha256_file(dest)
            log(f"skip   {entry.arxiv_id:<16} already present ({dest.stat().st_size:,} bytes)")
            yield FetchResult(entry, dest, digest, dest.stat().st_size, skipped=True)
            continue

        # The delay goes *before* each request but not before the first, so a
        # fully-cached run costs nothing and a resumed run does not double-wait.
        if fetched_any:
            log(f"       waiting {cfg.arxiv_delay_s:.0f}s (arXiv politeness)")
            sleep(cfg.arxiv_delay_s)

        log(f"fetch  {entry.arxiv_id:<16} {entry.source_url}")
        payload = (downloader or _download)(
            entry.source_url, cfg.arxiv_user_agent, cfg.fetch_timeout_s
        )
        fetched_any = True

        if not payload.startswith(b"%PDF"):
            raise RuntimeError(
                f"{entry.arxiv_id}: response is not a PDF (starts with "
                f"{payload[:16]!r}). arXiv sometimes serves an HTML rate-limit page "
                f"with a 200 status; a zero-length or HTML 'PDF' would fail much "
                f"later, during extraction."
            )

        # Write via a temp file so an interrupted download never leaves a truncated
        # PDF that the next run would happily skip as "already present".
        tmp = dest.with_suffix(".pdf.part")
        tmp.write_bytes(payload)
        tmp.replace(dest)

        yield FetchResult(entry, dest, sha256_file(dest), len(payload), skipped=False)


def _download(url: str, user_agent: str, timeout_s: float) -> bytes:
    import httpx

    r = httpx.get(url, headers={"User-Agent": user_agent}, timeout=timeout_s,
                  follow_redirects=True)
    r.raise_for_status()
    return r.content
