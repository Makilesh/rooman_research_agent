"""Live web search as an ADDITIONAL retriever, behind `--web`. Default off.

The rules this module exists under, all of them load-bearing:

1. **Supplementary, never a replacement.** Local corpus results are always retrieved;
   web results are appended to the same candidate pool and compete on the same
   reranker score.
2. **Same pipeline, no shortcuts.** A web passage is chunked, embedded, reranked,
   cited and groundedness-verified exactly like a corpus passage. There is no path by
   which unverified web text reaches an answer.
3. **Visibly distinguishable.** Web sources render with their URL and a `web:` id
   prefix, so a reader can never mistake one for a peer-reviewed corpus paper.
4. **Abstention must not regress.** With the flag off, behaviour is byte-identical to
   the corpus-only path -- the control questions must still refuse exactly as before.
   The README's headline numbers stay corpus-only so the evaluation stays honest.

The provider is DuckDuckGo's HTML endpoint: no API key, no account, no quota to
account for. That keeps the extension consistent with the project's "works with no
key at all" property rather than adding a second credential to the quickstart.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html import unescape
from typing import Sequence
from urllib.parse import parse_qs, unquote, urlparse

from .config import Config
from .retrieve import Hit

WEB_PREFIX = "web_"
DDG_HTML = "https://html.duckduckgo.com/html/"

_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_TAGS = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class WebResult:
    title: str
    url: str
    snippet: str

    @property
    def chunk_id(self) -> str:
        """Stable id derived from the URL, so the same page keeps the same id.

        Prefixed `web_` so an id is self-describing wherever it appears -- in a
        citation, in `turn_citations`, or in an error message.
        """
        return WEB_PREFIX + hashlib.sha256(self.url.encode()).hexdigest()[:12]


def _clean(fragment: str) -> str:
    return unescape(_TAGS.sub("", fragment)).strip()


def _real_url(href: str) -> str:
    """DuckDuckGo wraps results in a redirect; recover the destination."""
    if "duckduckgo.com/l/" in href or href.startswith("//duckduckgo.com/l/"):
        query = parse_qs(urlparse("https:" + href if href.startswith("//") else href).query)
        if "uddg" in query:
            return unquote(query["uddg"][0])
    return href


def search(query: str, cfg: Config, limit: int = 5,
           transport=None) -> list[WebResult]:
    """Return web results, or an empty list. Never raises into the agent path.

    A search outage must degrade to corpus-only behaviour rather than failing the
    turn -- the web is the optional half.
    """
    try:
        if transport is not None:
            html = transport(query)
        else:
            import httpx

            response = httpx.post(
                DDG_HTML, data={"q": query},
                headers={"User-Agent": cfg.arxiv_user_agent},
                timeout=cfg.web_timeout_s, follow_redirects=True,
            )
            response.raise_for_status()
            html = response.text
    except Exception:
        return []

    out: list[WebResult] = []
    seen: set[str] = set()
    for match in _RESULT.finditer(html):
        url = _real_url(match.group("href"))
        if not url.startswith("http") or url in seen:
            continue
        snippet = _clean(match.group("snippet"))
        if len(snippet) < 80:
            # Too short to cite from, and too short for the verifier to score.
            continue
        seen.add(url)
        out.append(WebResult(_clean(match.group("title")), url, snippet))
        if len(out) >= limit:
            break
    return out


def to_hits(results: Sequence[WebResult]) -> list[Hit]:
    """Adapt web results into the same `Hit` the corpus pipeline uses.

    Deliberately the same type. A separate WebHit would let a web passage take a
    different path through reranking, citation and verification, and the whole point
    is that it takes the identical one.

    `page_start`/`page_end` are 0: a web page has no page number, and inventing one
    would put a false precision into a citation.
    """
    return [
        Hit(chunk_id=r.chunk_id, doc_id="web", title=r.title or r.url,
            page_start=0, page_end=0, section=r.url, text=r.snippet)
        for r in results
    ]


def is_web(chunk_id: str) -> bool:
    return chunk_id.startswith(WEB_PREFIX)


def source_label(hit: Hit) -> str:
    """How a web citation renders. Never looks like a corpus paper."""
    return f"[web] {hit.title} — {hit.section}"
