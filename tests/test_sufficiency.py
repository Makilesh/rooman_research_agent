"""The agentic loop: termination, decomposition, relaxation, ladder routing."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from research_agent import db, sufficiency
from research_agent.config import Config
from research_agent.llm import LLMClient, OllamaProvider
from research_agent.retrieve import Hit


class ScriptedOllama(OllamaProvider):
    """Returns a canned payload per `purpose`, inferred from the prompt. No network."""

    def __init__(self, host, *, sufficient=True, multi_part=False, subs=None,
                 rewritten=None):
        super().__init__(host, transport=self._respond)
        self.calls: list[str] = []
        self._sufficient = sufficient
        self._multi = multi_part
        self._subs = subs or []
        self._rewritten = rewritten

    def _respond(self, path, payload, timeout):
        prompt = payload["prompt"]
        if "Decide whether these passages" in prompt:
            self.calls.append("judge")
            return {"response": json.dumps({
                "sufficient": self._sufficient, "missing": "the second paper",
                "is_multi_part": self._multi})}
        if "Split this question" in prompt:
            self.calls.append("decompose")
            return {"response": json.dumps({"sub_questions": self._subs})}
        if "did not retrieve enough evidence" in prompt:
            self.calls.append("rewrite")
            return {"response": json.dumps({"rewritten": self._rewritten or ""})}
        self.calls.append("other")
        return {"response": "{}"}


def _client(cfg, conn, provider):
    return LLMClient(cfg=cfg, conn=conn, api_keys={}, ollama=provider)


def _hit(cid, doc, score=0.9):
    return Hit(cid, doc, doc.upper(), 1, 1, None, f"text of {cid}",
               rerank_score=score)


class FakeModels:
    """Stands in for (embedder, vectors, reranker); records what it was asked."""

    def __init__(self, per_query):
        self.per_query = per_query
        self.rerank_queries: list[str] = []


def _patch_retrieval(monkeypatch, models: FakeModels):
    def fake_retrieve(conn, cfg, _models, query, scope):
        models.rerank_queries.append(query)
        hits = models.per_query.get(query, models.per_query.get("*", []))
        if scope:
            scoped = [h for h in hits if h.doc_id in scope]
            hits = scoped or hits
        return list(hits)

    monkeypatch.setattr(sufficiency, "_retrieve", fake_retrieve)


# ---------------------------------------------------------------------------
#  Termination — the property that must hold on every input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sufficient,multi,subs,rewritten", [
    (False, False, [], None),                 # nothing works
    (False, True, ["a", "b"], "rewritten q"),  # everything fires
    (False, False, [], "rewritten q"),        # rewrite loops
    (True, False, [], None),                  # immediate success
])
def test_the_loop_always_terminates(cfg, conn, monkeypatch, sufficient, multi,
                                    subs, rewritten):
    """Termination is structural: a bounded `for`, not a condition a model controls.

    No mock, however pathological, can make this run away.
    """
    models = FakeModels({"*": [_hit("c1", "lora")]})
    _patch_retrieval(monkeypatch, models)
    provider = ScriptedOllama(cfg.ollama_host, sufficient=sufficient,
                              multi_part=multi, subs=subs, rewritten=rewritten)
    out = sufficiency.gather_evidence(cfg, conn, _client(cfg, conn, provider),
                                      models, "a question")
    assert out.loops <= cfg.max_retrieval_loops


def test_a_satisfied_judge_stops_after_one_pass(cfg, conn, monkeypatch):
    models = FakeModels({"*": [_hit("c1", "lora")]})
    _patch_retrieval(monkeypatch, models)
    provider = ScriptedOllama(cfg.ollama_host, sufficient=True)
    out = sufficiency.gather_evidence(cfg, conn, _client(cfg, conn, provider),
                                      models, "q")
    assert out.loops == 0
    assert provider.calls.count("judge") == 1
    assert "decompose" not in provider.calls


# ---------------------------------------------------------------------------
#  Decomposition
# ---------------------------------------------------------------------------
def test_decomposition_fires_on_a_multi_part_question(cfg, conn, monkeypatch):
    models = FakeModels({"*": [_hit("c1", "lora")]})
    _patch_retrieval(monkeypatch, models)
    provider = ScriptedOllama(cfg.ollama_host, sufficient=False, multi_part=True,
                              subs=["What memory does QLoRA use?",
                                    "What does LoRA reduce?"])
    out = sufficiency.gather_evidence(cfg, conn, _client(cfg, conn, provider),
                                      models, "How does QLoRA extend LoRA?")
    assert "decompose" in provider.calls
    assert len(out.sub_questions) == 2


def test_decomposition_does_not_fire_on_a_single_part_question(cfg, conn, monkeypatch):
    models = FakeModels({"*": [_hit("c1", "lora")]})
    _patch_retrieval(monkeypatch, models)
    provider = ScriptedOllama(cfg.ollama_host, sufficient=False, multi_part=False,
                              rewritten="a better query")
    sufficiency.gather_evidence(cfg, conn, _client(cfg, conn, provider), models,
                                "What rank does LoRA use?")
    assert "decompose" not in provider.calls
    assert "rewrite" in provider.calls


def test_sub_questions_are_reranked_against_themselves(cfg, conn, monkeypatch):
    """The single highest-value detail in this step.

    Scoring sub-question results against the PARENT query undoes the decomposition:
    a passage answering one part need not resemble the composite question at all.
    """
    subs = ["What memory does QLoRA use?", "What does LoRA reduce?"]
    models = FakeModels({
        subs[0]: [_hit("c_q", "qlora")],
        subs[1]: [_hit("c_l", "lora")],
        "*": [_hit("c1", "lora")],
    })
    _patch_retrieval(monkeypatch, models)
    provider = ScriptedOllama(cfg.ollama_host, sufficient=False, multi_part=True,
                              subs=subs)
    out = sufficiency.gather_evidence(cfg, conn, _client(cfg, conn, provider),
                                      models, "How does QLoRA extend LoRA?")

    for sub in subs:
        assert sub in models.rerank_queries, (
            f"{sub!r} must be retrieved and reranked as its own query"
        )
    # Evidence from both sub-questions survives into the merged slate.
    assert {h.doc_id for h in out.hits} == {"qlora", "lora"}


# ---------------------------------------------------------------------------
#  Progressive relaxation
# ---------------------------------------------------------------------------
def test_document_scope_is_inferred_from_the_question(conn):
    db.insert(conn, "documents", {"doc_id": "lora", "title": "LoRA: Low-Rank "
              "Adaptation of Large Language Models", "path": "p", "sha256": "x"})
    db.insert(conn, "documents", {"doc_id": "qlora", "title": "QLoRA: Efficient "
              "Finetuning of Quantized LLMs", "path": "p", "sha256": "y"})
    conn.commit()
    assert sufficiency.infer_document_scope(
        conn, "Which section of the lora paper covers NF4?") == {"lora"}
    assert sufficiency.infer_document_scope(conn, "what rank is used?") == set()


def test_relaxation_drops_the_scope_on_retry(cfg, conn, monkeypatch):
    """The M&A failure this exists to prevent: a wrong first-pass scope removes the
    answer from the search space, and no amount of rewriting recovers it."""
    db.insert(conn, "documents", {"doc_id": "lora", "title": "LoRA", "path": "p",
                                  "sha256": "x"})
    conn.commit()
    seen_scopes: list[object] = []

    def fake_retrieve(conn_, cfg_, _models, query, scope):
        seen_scopes.append(scope)
        return [_hit("c1", "lora")]

    monkeypatch.setattr(sufficiency, "_retrieve", fake_retrieve)
    provider = ScriptedOllama(cfg.ollama_host, sufficient=False,
                              rewritten="a better query")
    sufficiency.gather_evidence(cfg, conn, _client(cfg, conn, provider),
                                FakeModels({}), "what does the lora paper say?")
    assert seen_scopes[0] == {"lora"}, "first attempt is scoped, for precision"
    assert all(s is None for s in seen_scopes[1:]), "every retry drops the scope"


def test_a_scope_never_empties_the_candidate_set(cfg, conn, monkeypatch):
    """An empty slate can only be refused, which would hide why."""
    models = FakeModels({"*": [_hit("c1", "bert")]})
    _patch_retrieval(monkeypatch, models)
    provider = ScriptedOllama(cfg.ollama_host, sufficient=True)
    out = sufficiency.gather_evidence(
        cfg, conn, _client(cfg, conn, provider), models, "about lora")
    assert out.hits, "scoping must not remove every candidate"


# ---------------------------------------------------------------------------
#  Ladder routing
# ---------------------------------------------------------------------------
def test_every_loop_call_lands_on_the_volume_ladder(cfg, conn, monkeypatch):
    """Routing judge/rewrite/decompose through synthesis models would exhaust
    reasoning quota in about four turns, and the failure would land on answer
    generation -- the one thing a reviewer sees."""
    models = FakeModels({"*": [_hit("c1", "lora")]})
    _patch_retrieval(monkeypatch, models)
    provider = ScriptedOllama(cfg.ollama_host, sufficient=False, multi_part=True,
                              subs=["a", "b"], rewritten="better")
    sufficiency.gather_evidence(cfg, conn, _client(cfg, conn, provider), models, "q")

    rows = db.all_rows(conn, "SELECT DISTINCT purpose, ladder FROM llm_calls")
    assert rows
    for r in rows:
        assert r["ladder"] == "volume", (
            f"{r['purpose']} must not consume synthesis quota"
        )


def test_a_judge_failure_does_not_cost_the_turn(cfg, conn, monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("judge unavailable")

    models = FakeModels({"*": [_hit("c1", "lora")]})
    _patch_retrieval(monkeypatch, models)
    provider = OllamaProvider(cfg.ollama_host, transport=boom)
    out = sufficiency.gather_evidence(cfg, conn, _client(cfg, conn, provider),
                                      models, "q")
    assert out.hits, "a judge outage must fall through to synthesis, not refuse"


def test_the_trace_records_every_node(cfg, conn, monkeypatch):
    models = FakeModels({"*": [_hit("c1", "lora")]})
    _patch_retrieval(monkeypatch, models)
    provider = ScriptedOllama(cfg.ollama_host, sufficient=False,
                              rewritten="better query")
    out = sufficiency.gather_evidence(cfg, conn, _client(cfg, conn, provider),
                                      models, "q")
    nodes = {s.node for s in out.trace.steps}
    assert {"retrieve", "sufficiency-judge"} <= nodes
    assert out.trace.render(), "the trace must be printable for `ask --trace`"
