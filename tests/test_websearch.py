"""Web search as a supplementary retriever. No network — the transport is injected."""

from __future__ import annotations

from dataclasses import replace

import pytest

from research_agent import websearch
from research_agent.config import Config
from research_agent.retrieve import Hit

FAKE_HTML = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnf4">
    NF4 explained</a>
  <a class="result__snippet">NormalFloat is an information-theoretically optimal
  data type for zero-centred normally distributed weights, used by QLoRA.</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.org/lora">LoRA overview</a>
  <a class="result__snippet">Low-rank adaptation freezes the pretrained weights and
  injects trainable rank decomposition matrices into each transformer layer.</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.net/tiny">Too short</a>
  <a class="result__snippet">Short.</a>
</div>
"""


def test_results_are_parsed_and_redirects_unwrapped(cfg):
    results = websearch.search("nf4", cfg, transport=lambda q: FAKE_HTML)
    assert [r.url for r in results] == ["https://example.com/nf4",
                                        "https://example.org/lora"]
    assert results[0].title == "NF4 explained"


def test_snippets_too_short_to_cite_are_dropped(cfg):
    """A fragment the verifier cannot score is not evidence."""
    results = websearch.search("x", cfg, transport=lambda q: FAKE_HTML)
    assert all("Short." not in r.snippet for r in results)


def test_a_search_outage_degrades_to_corpus_only(cfg):
    """The web is the optional half; losing it must not fail the turn."""
    def boom(_q):
        raise ConnectionError("no network")

    assert websearch.search("x", cfg, transport=boom) == []


def test_web_chunk_ids_are_stable_and_self_describing(cfg):
    a = websearch.search("x", cfg, transport=lambda q: FAKE_HTML)
    b = websearch.search("x", cfg, transport=lambda q: FAKE_HTML)
    assert [r.chunk_id for r in a] == [r.chunk_id for r in b]
    assert all(r.chunk_id.startswith("web_") for r in a)
    assert all(websearch.is_web(r.chunk_id) for r in a)


def test_a_web_source_never_looks_like_a_corpus_paper(cfg):
    """A reader must never have to work out which kind of source they are reading."""
    hits = websearch.to_hits(websearch.search("x", cfg, transport=lambda q: FAKE_HTML))
    label = hits[0].source_label
    assert label.startswith("[web] ")
    assert "https://example.com/nf4" in label
    assert "p." not in label, "a web page has no page number; inventing one is a lie"


def test_a_corpus_source_is_unchanged_by_the_web_path():
    corpus_hit = Hit("c_lora_0001", "lora", "LoRA", 4, 4, None, "text")
    assert corpus_hit.source_label == "LoRA · p.4"
    assert not corpus_hit.is_web
    assert not websearch.is_web(corpus_hit.chunk_id)


def test_web_hits_use_the_same_type_as_corpus_hits(cfg):
    """Deliberately the same `Hit`. A separate type would let a web passage take a
    different path through reranking, citation and verification -- and taking the
    identical path is the entire point."""
    hits = websearch.to_hits(websearch.search("x", cfg, transport=lambda q: FAKE_HTML))
    assert all(isinstance(h, Hit) for h in hits)
    assert all(h.page_start == 0 for h in hits), "no invented page numbers"


def test_the_flag_defaults_to_off(cfg):
    """The headline numbers are corpus-only, so the corpus-only path is the default."""
    assert Config().web_enabled is False
    assert cfg.web_enabled is False


def test_enabling_web_does_not_remove_corpus_candidates(cfg, conn, monkeypatch):
    """Supplementary, never a replacement: the corpus candidates all survive."""
    from research_agent import sufficiency

    corpus_hits = [Hit(f"c_lora_{i:04d}", "lora", "LoRA", 1, 1, None, f"corpus {i}",
                       rerank_score=0.9) for i in range(3)]

    monkeypatch.setattr(sufficiency.retrieve, "dense_search",
                        lambda *a, **k: corpus_hits)
    monkeypatch.setattr(sufficiency.retrieve, "sparse_search", lambda *a, **k: [])
    monkeypatch.setattr(sufficiency.retrieve, "embed_query", lambda *a, **k: None)
    monkeypatch.setattr(sufficiency.retrieve, "reciprocal_rank_fusion",
                        lambda d, s, c: list(d))
    monkeypatch.setattr(sufficiency.websearch, "search",
                        lambda *a, **k: [websearch.WebResult(
                            "W", "https://example.com/w", "a web snippet " * 10)])
    seen: dict = {}

    def fake_rerank(model, query, hits, config):
        seen["hits"] = list(hits)
        return list(hits)

    monkeypatch.setattr(sufficiency.rerank_mod, "rerank", fake_rerank)

    web_cfg = replace(cfg, web_enabled=True)
    sufficiency._retrieve(conn, web_cfg, (None, None, None), "q", None)

    ids = [h.chunk_id for h in seen["hits"]]
    for h in corpus_hits:
        assert h.chunk_id in ids, "a corpus candidate was displaced by the web"
    assert any(websearch.is_web(i) for i in ids), "the web result should be appended"


def test_with_the_flag_off_no_search_is_attempted(cfg, conn, monkeypatch):
    """The regression guard: corpus-only behaviour must be untouched."""
    from research_agent import sufficiency

    monkeypatch.setattr(sufficiency.retrieve, "dense_search", lambda *a, **k: [])
    monkeypatch.setattr(sufficiency.retrieve, "sparse_search", lambda *a, **k: [])
    monkeypatch.setattr(sufficiency.retrieve, "embed_query", lambda *a, **k: None)
    monkeypatch.setattr(sufficiency.retrieve, "reciprocal_rank_fusion",
                        lambda d, s, c: [])
    monkeypatch.setattr(sufficiency.rerank_mod, "rerank", lambda *a, **k: [])

    called = {"n": 0}

    def counting_search(*a, **k):
        called["n"] += 1
        return []

    monkeypatch.setattr(sufficiency.websearch, "search", counting_search)
    sufficiency._retrieve(conn, cfg, (None, None, None), "q", None)
    assert called["n"] == 0, "the corpus-only path must not touch the network"
