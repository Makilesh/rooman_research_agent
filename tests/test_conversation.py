"""Condensation, the drift guard, coreference, and routing. No network."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from research_agent import conversation, db, router
from research_agent.config import Config
from research_agent.conversation import Turn, content_words, stem
from research_agent.llm import LLMClient, OllamaProvider
from research_agent.retrieve import Hit


def _client(cfg, conn, payload):
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return LLMClient(cfg=cfg, conn=conn, api_keys={},
                     ollama=OllamaProvider(cfg.ollama_host,
                                           transport=lambda *a: {"response": body}))


HISTORY = [
    Turn("t1", 1, "user", "What problem does LoRA solve?"),
    Turn("t2", 2, "agent", "What problem does LoRA solve?", route="answer",
         answer_text="LoRA freezes the pretrained weights and injects trainable "
                     "rank decomposition matrices, reducing trainable parameters."),
]


# ---------------------------------------------------------------------------
#  Stemming — the drift guard is only as good as this
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ("compare", "compared"), ("compare", "compares"),
    ("quantise", "quantised"), ("reduce", "reduces"), ("adapt", "adapting"),
])
def test_stemmer_is_consistent_across_inflections(a, b):
    """An inconsistent stemmer makes the drift guard fire on legitimate rephrasing.

    Observed for real: "compared" lost its suffix and became "compar" while
    "compare" matched no suffix and stayed "compare", so a condensation that reused
    a word already in the conversation was reported as drift.
    """
    assert stem(a) == stem(b)


def test_stopwords_do_not_count_as_content():
    assert content_words("the of and a an it") == set()


# ---------------------------------------------------------------------------
#  Condensation
# ---------------------------------------------------------------------------
def test_turn_one_skips_condensation_entirely(cfg, conn):
    """No history means nothing to condense. Calling the model anyway spends quota
    and risks distorting a perfectly good query."""
    before = db.scalar(conn, "SELECT COUNT(*) FROM llm_calls")
    out = conversation.condense(cfg, _client(cfg, conn, {}), [], "What is LoRA?")
    assert out.query == "What is LoRA?"
    assert out.used_llm is False
    assert db.scalar(conn, "SELECT COUNT(*) FROM llm_calls") == before


def test_a_pronoun_follow_up_condenses_to_a_standalone_query(cfg, conn):
    client = _client(cfg, conn, {
        "standalone_query": "How does the quantised version of LoRA reduce memory?"})
    out = conversation.condense(cfg, client, HISTORY,
                                "How does the quantised version reduce memory?")
    assert out.used_llm and not out.drifted
    assert "LoRA" in out.query


def test_the_drift_guard_catches_an_injected_novel_content_word(cfg, conn):
    """The Opkey failure, reproduced deliberately.

    A follow-up about statuses was condensed into a query containing "workflow" -- a
    word nobody had used -- which steered retrieval into the wrong chapters. Nothing
    errored; the answer was fluent and wrong.
    """
    client = _client(cfg, conn, {
        "standalone_query": "How does the LoRA quantisation workflow reduce memory?"})
    out = conversation.condense(cfg, client, HISTORY,
                                "How does the quantised version reduce memory?")
    assert out.drifted is True
    assert stem("workflow") in out.novel_words
    assert out.fell_back is True
    assert out.query == "How does the quantised version reduce memory?", (
        "on drift the RAW query must be used, not the hallucinated one"
    )


def test_condenser_failure_never_costs_the_turn(cfg, conn):
    def boom(*a, **k):
        raise TimeoutError("condenser timed out")

    client = LLMClient(cfg=cfg, conn=conn, api_keys={},
                       ollama=OllamaProvider(cfg.ollama_host, transport=boom))
    out = conversation.condense(cfg, client, HISTORY, "and after that?")
    assert out.query == "and after that?"
    assert out.fell_back is True


def test_reusing_conversation_vocabulary_is_not_drift(cfg, conn):
    client = _client(cfg, conn, {
        "standalone_query": "What trainable parameters does LoRA reduce?"})
    out = conversation.condense(cfg, client, HISTORY, "how many does it reduce?")
    assert out.drifted is False


# ---------------------------------------------------------------------------
#  History window
# ---------------------------------------------------------------------------
def test_window_respects_the_turn_cap(cfg):
    turns = [Turn(f"t{i}", i, "user", "hi") for i in range(20)]
    assert len(conversation.history_window(turns, cfg)) == cfg.history_max_turns


def test_window_respects_the_token_cap(cfg):
    """Two caps because they fail differently: a turn cap alone lets six long turns
    blow the condenser's context."""
    long_turn = "word " * 4000
    turns = [Turn(f"t{i}", i, "user", long_turn) for i in range(6)]
    kept = conversation.history_window(turns, cfg)
    assert len(kept) < cfg.history_max_turns


def test_window_keeps_the_most_recent_turns(cfg):
    turns = [Turn(f"t{i}", i, "user", f"turn {i}") for i in range(10)]
    kept = conversation.history_window(turns, cfg)
    assert kept[-1].raw_text == "turn 9"


# ---------------------------------------------------------------------------
#  Coreference
# ---------------------------------------------------------------------------
def _seed_answer_turn(conn, chunk_ids):
    db.insert(conn, "documents", {"doc_id": "rag", "title": "RAG", "path": "p",
                                  "sha256": "x"})
    for i, cid in enumerate(chunk_ids):
        db.insert(conn, "chunks", {
            "chunk_id": cid, "doc_id": "rag", "parent_id": None, "level": 0,
            "ord": i, "page_start": i + 1, "page_end": i + 1, "section": None,
            "token_count": 10, "text": f"text {i}"})
    db.insert(conn, "sessions", {"session_id": "s1", "corpus_fingerprint": "fp"})
    db.insert(conn, "turns", {"turn_id": "ta", "session_id": "s1", "ord": 2,
                              "role": "agent", "raw_text": "q", "route": "answer"})
    for i, cid in enumerate(chunk_ids):
        db.insert(conn, "turn_citations", {
            "turn_id": "ta", "sentence_idx": i, "sentence_text": f"s{i}",
            "chunk_id": cid, "verify_score": 0.9, "status": "verified"})
    conn.commit()


def test_an_ordinal_reference_resolves_through_turn_citations(conn):
    """The payoff of the structured citation contract.

    "The second source" is answerable only because every sentence-to-chunk link is a
    row. Inline prose markers give nothing to query.
    """
    _seed_answer_turn(conn, ["c_rag_0001", "c_rag_0002", "c_rag_0003"])
    _, refs, unresolved = conversation.resolve_source_references(
        conn, "s1", "Expand on the second source.")
    assert refs == ["c_rag_0002"]
    assert unresolved == []


def test_last_source_resolves_to_the_final_citation(conn):
    _seed_answer_turn(conn, ["c_rag_0001", "c_rag_0002"])
    _, refs, _ = conversation.resolve_source_references(
        conn, "s1", "tell me about the last source")
    assert refs == ["c_rag_0002"]


def test_an_unresolvable_ordinal_is_reported_not_silently_ignored(conn):
    """Observed for real: the previous answer cited one source, the user asked for
    the second, and the bare phrase retrieved nothing -- producing a confusing
    refusal instead of a useful question."""
    _seed_answer_turn(conn, ["c_rag_0001"])
    _, refs, unresolved = conversation.resolve_source_references(
        conn, "s1", "Expand on the second source.")
    assert refs == []
    assert unresolved == [("second", 1)]


def test_an_explicit_chunk_id_resolves(conn):
    _seed_answer_turn(conn, ["c_rag_0001"])
    _, refs, _ = conversation.resolve_source_references(
        conn, "s1", "say more about [c_rag_0001]")
    assert refs == ["c_rag_0001"]


# ---------------------------------------------------------------------------
#  Routing
# ---------------------------------------------------------------------------
def _hit(cid, doc, score):
    return Hit(cid, doc, doc.upper(), 1, 1, None, "text", rerank_score=score)


def test_each_band_routes_correctly(cfg):
    assert router.route(cfg, [_hit("a", "lora", 0.99)]).decision == router.ANSWER
    mid = (cfg.tau_low + cfg.tau_high) / 2
    assert router.route(cfg, [_hit("a", "lora", mid)]).decision == router.CLARIFY
    assert router.route(cfg, [_hit("a", "lora", 0.01)]).decision == router.REFUSE


def test_routing_uses_the_top_score_not_the_average(cfg):
    """Averaging makes retrieving MORE results look WORSE and penalises recall."""
    hits = [_hit("a", "lora", 0.99)] + [_hit(f"n{i}", "bert", 0.01) for i in range(9)]
    assert router.route(cfg, hits).decision == router.ANSWER


def test_empty_retrieval_refuses(cfg):
    assert router.route(cfg, []).decision == router.REFUSE


def test_clarification_names_one_option_per_document(cfg):
    hits = [_hit("a", "lora", 0.9), _hit("b", "lora", 0.88), _hit("c", "qlora", 0.85)]
    competing = router.competing_candidates(cfg, hits)
    assert [h.doc_id for h in competing] == ["lora", "qlora"], (
        "repeating one paper three times describes one option three times"
    )


def test_diversity_guard_admits_a_second_paper(cfg):
    """Measured at Step 7: multi-hop answers came from one paper because the second
    never reached the context slate."""
    cfg = replace(cfg, context_top_n=3)
    hits = [_hit(f"l{i}", "lora", 0.95 - i * 0.01) for i in range(5)]
    hits.append(_hit("q1", "qlora", 0.90))
    kept = router.apply_diversity_guard(cfg, hits)
    assert len(kept) == 3
    assert {h.doc_id for h in kept} == {"lora", "qlora"}


def test_diversity_guard_does_not_invent_diversity(cfg):
    """If only one paper is relevant, nothing changes."""
    cfg = replace(cfg, context_top_n=3)
    hits = [_hit(f"l{i}", "lora", 0.95) for i in range(5)]
    hits.append(_hit("q1", "qlora", 0.01))  # below tau_low
    kept = router.apply_diversity_guard(cfg, hits)
    assert {h.doc_id for h in kept} == {"lora"}


def test_clarification_reply_merges_with_the_original_question():
    merged = router.merge_clarification("What rank is used?", "In LoRA, for GPT-3.")
    assert "rank" in merged and "LoRA" in merged


def test_routing_refuses_to_run_on_unmeasured_thresholds(cfg):
    unmeasured = replace(cfg, thresholds_are_measured=False, tau_low=0.0, tau_high=0.0)
    with pytest.raises(RuntimeError, match="Step 5"):
        router.route(unmeasured, [_hit("a", "lora", 0.5)])


# ---------------------------------------------------------------------------
#  Semantic cache
# ---------------------------------------------------------------------------
def test_cache_returns_a_near_paraphrase_and_misses_a_different_query(cfg, conn):
    import numpy as np

    vec = np.ones(8, dtype=np.float32) / np.sqrt(8)
    conversation.cache_store(conn, "fp1", "what is lora", vec, {"answer": "x"})
    assert conversation.cache_lookup(conn, cfg, "fp1", vec) is not None

    other = np.zeros(8, dtype=np.float32)
    other[0] = 1.0
    assert conversation.cache_lookup(conn, cfg, "fp1", other) is None


def test_a_fingerprint_change_invalidates_the_cache(cfg, conn):
    """The same question over different text is a different question."""
    import numpy as np

    vec = np.ones(8, dtype=np.float32) / np.sqrt(8)
    conversation.cache_store(conn, "fp1", "q", vec, {"answer": "x"})
    assert conversation.cache_lookup(conn, cfg, "fp2", vec) is None


def test_session_warns_when_the_corpus_moved(conn):
    db.insert(conn, "sessions", {"session_id": "s1", "corpus_fingerprint": "old"})
    conn.commit()
    assert conversation.fingerprint_warning(conn, "s1", "old") is None
    warning = conversation.fingerprint_warning(conn, "s1", "new")
    assert warning and "no longer point at the text they did" in warning
