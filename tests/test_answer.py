"""The citation contract and groundedness verification. Mocked LLM, no network."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from research_agent import db, prompts, verify
from research_agent.answer import Answer, CitedSentence, InventedCitation, persist, synthesise
from research_agent.config import Config
from research_agent.llm import LLMClient, MalformedResponse, OllamaProvider
from research_agent.retrieve import Hit

HITS = [
    Hit("c_lora_0020", "lora", "LoRA", 10, 10, "7 Understanding",
        "We set a parameter budget of 18M on GPT-3 175B, which corresponds to r = 8 "
        "if we adapt one type of attention weights or r = 4 if we adapt two types.",
        rerank_score=0.99, rank=1),
    Hit("c_qlora_0007", "qlora", "QLoRA", 5, 5, "3 QLoRA Finetuning",
        "We term the resulting data type k-bit NormalFloat (NFk), since the data type "
        "is information-theoretically optimal for zero-centered normally distributed data.",
        rerank_score=0.95, rank=2),
]


def _client(cfg, conn, payload):
    """An LLMClient whose transport returns exactly `payload`, with no network."""
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return LLMClient(
        cfg=cfg, conn=conn, api_keys={},
        ollama=OllamaProvider(cfg.ollama_host,
                              transport=lambda *a: {"response": body}),
    )


# ---------------------------------------------------------------------------
#  The contract
# ---------------------------------------------------------------------------
def test_a_supported_answer_produces_cited_sentences(cfg, conn):
    client = _client(cfg, conn, {
        "insufficient_evidence": False, "refusal_reason": None,
        "sentences": [
            {"text": "LoRA uses r = 4 when adapting two attention weight types.",
             "cite": ["c_lora_0020"]},
        ],
    })
    ans = synthesise(cfg, client, "What rank?", HITS)
    assert not ans.is_refusal
    assert len(ans.sentences) == 1
    assert ans.sentences[0].chunk_ids == ["c_lora_0020"]


def test_an_invented_chunk_id_raises_rather_than_rendering(cfg, conn):
    """A fabricated citation is the one failure this whole system exists to prevent.

    Constrained decoding guarantees the *shape* of the output; it cannot stop a model
    emitting a well-formed lie. So the ids are checked against what was actually
    retrieved, and a bad one is fatal rather than quietly filtered out of an
    otherwise plausible answer.
    """
    client = _client(cfg, conn, {
        "insufficient_evidence": False, "refusal_reason": None,
        "sentences": [{"text": "Something plausible.", "cite": ["c_lora_9999"]}],
    })
    with pytest.raises(InventedCitation, match="c_lora_9999"):
        synthesise(cfg, client, "What rank?", HITS)


def test_the_refusal_path_produces_zero_citations(cfg, conn):
    """A refusal carrying citations is contradictory, so the flag wins structurally
    rather than depending on the model behaving."""
    client = _client(cfg, conn, {
        "insufficient_evidence": True,
        "refusal_reason": "The passages do not mention MMLU.",
        "sentences": [{"text": "Sneaky leftover.", "cite": ["c_lora_0020"]}],
    })
    ans = synthesise(cfg, client, "MMLU results?", HITS)
    assert ans.is_refusal
    assert ans.sentences == []
    assert ans.cited_chunk_ids == set()
    assert "MMLU" in ans.refusal_reason


def test_an_empty_answer_becomes_a_refusal_not_a_blank_page(cfg, conn):
    client = _client(cfg, conn, {"insufficient_evidence": False,
                                 "refusal_reason": None, "sentences": []})
    assert synthesise(cfg, client, "q", HITS).is_refusal


def test_uncited_sentences_are_dropped(cfg, conn):
    client = _client(cfg, conn, {
        "insufficient_evidence": False, "refusal_reason": None,
        "sentences": [
            {"text": "Cited properly.", "cite": ["c_lora_0020"]},
            {"text": "No citation at all.", "cite": []},
        ],
    })
    ans = synthesise(cfg, client, "q", HITS)
    assert [s.text for s in ans.sentences] == ["Cited properly."]


def test_malformed_json_raises_after_the_single_retry(cfg, conn):
    client = _client(cfg, conn, "this is not json")
    with pytest.raises(MalformedResponse):
        synthesise(cfg, client, "q", HITS)


def test_the_schema_reaches_the_provider(cfg, conn):
    seen = {}

    def transport(path, payload, timeout):
        seen["format"] = payload.get("format")
        return {"response": json.dumps(
            {"insufficient_evidence": True, "refusal_reason": "x", "sentences": []})}

    client = LLMClient(cfg=cfg, conn=conn, api_keys={},
                       ollama=OllamaProvider(cfg.ollama_host, transport=transport))
    synthesise(cfg, client, "q", HITS)
    assert seen["format"] == prompts.ANSWER_SCHEMA


# ---------------------------------------------------------------------------
#  Context identity -- the trap this project was built around
# ---------------------------------------------------------------------------
def test_the_verifier_sees_byte_identical_text_to_the_synthesiser(cfg, conn):
    """Asserted, not merely commented.

    In my previous system the validator got 500-character truncations while the
    synthesiser got full chunks plus parents, which produced false unsupported-claim
    flags on correctly-sourced figures. The context is built once and both stages
    read the same objects.
    """
    client = _client(cfg, conn, {
        "insufficient_evidence": False, "refusal_reason": None,
        "sentences": [{"text": "LoRA uses r = 4.", "cite": ["c_lora_0020"]}],
    })
    ans = synthesise(cfg, client, "What rank?", HITS)

    # What the synthesiser was shown.
    assert ans.context == prompts.format_sources(HITS)
    # What the verifier will score against, for every cited chunk.
    for s in ans.sentences:
        for cid in s.chunk_ids:
            hit = ans.hit_by_id(cid)
            assert hit is not None
            assert hit.text in ans.context, (
                "the verifier must score against text the synthesiser actually saw"
            )
    assert verify.context_fingerprint(ans.hits) == verify.context_fingerprint(HITS)


def test_context_fingerprint_detects_truncation(cfg):
    truncated = [replace(h, text=h.text[:120]) for h in HITS]
    assert verify.context_fingerprint(truncated) != verify.context_fingerprint(HITS)


# ---------------------------------------------------------------------------
#  Verification
# ---------------------------------------------------------------------------
class FakeReranker:
    """Returns a fixed score per (sentence, passage) pair. No model download."""

    def __init__(self, scores):
        import torch

        self.activation_fn = torch.nn.Sigmoid()
        self._scores = scores
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs, **kwargs):
        self.seen.extend(pairs)
        return [self._scores.get(p[0], 0.9) for p in pairs]


def _measured(cfg: Config) -> Config:
    return replace(cfg, thresholds_are_measured=True,
                   tau_low=0.8, tau_high=0.99, tau_verify=0.378)


def _answer(sentences) -> Answer:
    return Answer(question="q", insufficient_evidence=False, refusal_reason=None,
                  sentences=sentences, hits=HITS,
                  context=prompts.format_sources(HITS), provider="ollama",
                  model="test", latency_ms=1)


def test_a_supported_sentence_verifies(cfg):
    cfg = _measured(cfg)
    ans = _answer([CitedSentence(0, "LoRA uses r = 4.", ["c_lora_0020"])])
    out = verify.verify_answer(cfg, FakeReranker({"LoRA uses r = 4.": 0.94}), ans)
    assert out.sentences[0].status == verify.VERIFIED
    assert out.sentences[0].verify_scores["c_lora_0020"] == pytest.approx(0.94)


def test_a_plausible_but_unsupported_sentence_is_flagged(cfg):
    """The injected-claim test: fluent, correctly-shaped, and not in the evidence."""
    cfg = _measured(cfg)
    claim = "LoRA reduces inference latency by 40% on all benchmarks."
    ans = _answer([CitedSentence(0, claim, ["c_lora_0020"])])
    out = verify.verify_answer(cfg, FakeReranker({claim: 0.05}), ans)
    assert out.sentences[0].status == verify.UNVERIFIED


def test_all_cited_chunks_must_clear_the_floor_not_just_the_best(cfg):
    """A sentence citing two passages claims both support it. Letting one strong
    citation carry a weak one is how an unsupported clause survives review."""
    cfg = _measured(cfg)
    sent = CitedSentence(0, "Mixed claim.", ["c_lora_0020", "c_qlora_0007"])

    class PerChunk(FakeReranker):
        def predict(self, pairs, **kwargs):
            return [0.95 if "18M" in p[1] else 0.02 for p in pairs]

    out = verify.verify_answer(cfg, PerChunk({}), _answer([sent]))
    assert out.sentences[0].status == verify.UNVERIFIED


def test_verification_makes_zero_llm_calls(cfg, conn):
    cfg = _measured(cfg)
    ans = _answer([CitedSentence(0, "LoRA uses r = 4.", ["c_lora_0020"])])
    before = db.scalar(conn, "SELECT COUNT(*) FROM llm_calls")
    verify.verify_answer(cfg, FakeReranker({}), ans)
    assert db.scalar(conn, "SELECT COUNT(*) FROM llm_calls") == before


def test_a_refusal_needs_no_verification(cfg):
    cfg = _measured(cfg)
    ans = replace(_answer([]), insufficient_evidence=True, refusal_reason="none")
    assert verify.verify_answer(cfg, FakeReranker({}), ans).is_refusal


def test_verification_refuses_to_run_on_unmeasured_thresholds(cfg):
    """A zero floor marks everything verified, which is worse than not verifying."""
    ans = _answer([CitedSentence(0, "x", ["c_lora_0020"])])
    unmeasured = replace(cfg, thresholds_are_measured=False, tau_verify=0.0)
    with pytest.raises(RuntimeError, match="Step 5"):
        verify.verify_answer(unmeasured, FakeReranker({}), ans)


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------
def _seed_corpus(conn):
    db.insert(conn, "documents", {"doc_id": "lora", "title": "LoRA",
                                  "path": "p", "sha256": "x"})
    db.insert(conn, "documents", {"doc_id": "qlora", "title": "QLoRA",
                                  "path": "p", "sha256": "y"})
    for h in HITS:
        db.insert(conn, "chunks", {
            "chunk_id": h.chunk_id, "doc_id": h.doc_id, "parent_id": None, "level": 0,
            "ord": 0, "page_start": h.page_start, "page_end": h.page_end,
            "section": h.section, "token_count": 100, "text": h.text})
    conn.commit()


def test_persist_writes_citations_and_every_retrieval_score(cfg, conn):
    _seed_corpus(conn)
    db.insert(conn, "sessions", {"session_id": "s1", "corpus_fingerprint": "fp"})
    conn.commit()
    ans = _answer([CitedSentence(0, "LoRA uses r = 4.", ["c_lora_0020"],
                                 {"c_lora_0020": 0.94}, verify.VERIFIED)])
    turn_id = persist(conn, ans, "s1", 1, route="answer")

    cites = db.all_rows(conn, "SELECT * FROM turn_citations WHERE turn_id = ?", (turn_id,))
    assert len(cites) == 1
    assert cites[0]["chunk_id"] == "c_lora_0020"
    assert cites[0]["status"] == verify.VERIFIED
    assert cites[0]["verify_score"] == pytest.approx(0.94)

    # Every retrieved chunk is recorded, with a flag for the ones that reached context.
    rets = db.all_rows(conn, "SELECT * FROM turn_retrievals WHERE turn_id = ?", (turn_id,))
    assert len(rets) == len(HITS)
    used = {r["chunk_id"]: r["used_in_context"] for r in rets}
    assert used["c_lora_0020"] == 1
    assert used["c_qlora_0007"] == 0


def test_rendered_page_numbers_match_the_stored_citations(cfg, conn):
    from research_agent.report import render_answer_markdown

    ans = _answer([CitedSentence(0, "LoRA uses r = 4.", ["c_lora_0020"],
                                 {"c_lora_0020": 0.94}, verify.VERIFIED)])
    md = render_answer_markdown(ans)
    assert "LoRA · p.10" in md
    assert "c_lora_0020" in md
    assert "[^1]" in md


def test_a_refusal_renders_without_a_sources_section(cfg):
    from research_agent.report import render_answer_markdown

    ans = replace(_answer([]), insufficient_evidence=True,
                  refusal_reason="Nothing in the corpus mentions MMLU.")
    md = render_answer_markdown(ans)
    assert "do not contain an answer" in md
    assert "## Sources" not in md
    assert "MMLU" in md


def test_unverified_sentences_are_marked_not_dropped(cfg):
    """Transparency over polish: silently removing a claim hides that it was made."""
    from research_agent.report import render_answer_markdown

    ans = _answer([CitedSentence(0, "Shaky claim.", ["c_lora_0020"],
                                 {"c_lora_0020": 0.02}, verify.UNVERIFIED)])
    md = render_answer_markdown(ans)
    assert "Shaky claim." in md
    assert "[unverified]" in md
