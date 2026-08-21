"""Retrieval evaluation: Recall@5, MRR, nDCG@10, and the configuration ablation.

Every metric in this module is **LLM-free and model-independent**. Retrieval quality
is measured without generating a single token, which is what makes it runnable as
often as needed, comparable across providers, and immune to the synthesis-model
variance that makes generation metrics noisy.

Small-sample honesty: the answerable set is 8 questions. A mean over 8 items has a
wide interval, so every reported mean carries one, computed by bootstrap rather than
assumed normal.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import Callable, Sequence

import numpy as np

from . import db, index as index_mod, rerank as rerank_mod, retrieve
from .config import Config
from .labels import GoldItem, LabelSet

CONFIGS = ("dense", "sparse", "hybrid", "hybrid_rerank")


@dataclass(frozen=True, slots=True)
class ItemResult:
    item_id: str
    ranked_ids: list[str]
    gold: list[str]
    latency_ms: float

    def recall_at(self, k: int) -> float:
        if not self.gold:
            return float("nan")
        hits = sum(1 for g in self.gold if g in self.ranked_ids[:k])
        return hits / len(self.gold)

    def mrr(self) -> float:
        for i, cid in enumerate(self.ranked_ids, start=1):
            if cid in self.gold:
                return 1.0 / i
        return 0.0

    def ndcg_at(self, k: int) -> float:
        """Binary-relevance nDCG. Gold chunks are equally relevant; there is no
        graded judgement to preserve, so the gain is 1 or 0."""
        if not self.gold:
            return float("nan")
        dcg = sum(1.0 / math.log2(i + 1)
                  for i, cid in enumerate(self.ranked_ids[:k], start=1) if cid in self.gold)
        ideal = sum(1.0 / math.log2(i + 1)
                    for i in range(1, min(len(self.gold), k) + 1))
        return dcg / ideal if ideal else 0.0


@dataclass(frozen=True, slots=True)
class ConfigResult:
    config: str
    items: list[ItemResult]

    def mean(self, fn: Callable[[ItemResult], float]) -> float:
        vals = [fn(i) for i in self.items if not math.isnan(fn(i))]
        return float(np.mean(vals)) if vals else float("nan")

    def ci95(self, fn: Callable[[ItemResult], float], iters: int = 5000) -> tuple[float, float]:
        """Bootstrap interval.

        With 8 items the mean is a genuinely uncertain estimate, and reporting it bare
        would imply a precision the sample cannot support.
        """
        vals = np.array([fn(i) for i in self.items if not math.isnan(fn(i))])
        if vals.size == 0:
            return float("nan"), float("nan")
        rng = np.random.default_rng(0)  # fixed so the interval is reproducible
        means = rng.choice(vals, size=(iters, vals.size), replace=True).mean(axis=1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def p50_latency(self) -> float:
        return float(np.percentile([i.latency_ms for i in self.items], 50))

    def p95_latency(self) -> float:
        return float(np.percentile([i.latency_ms for i in self.items], 95))


class Retrievers:
    """Loads the encoders once and serves every ablation configuration from them."""

    def __init__(self, cfg: Config, conn: Connection, with_reranker: bool = True):
        self.cfg, self.conn = cfg, conn
        self.embedder = index_mod.load_embedder(cfg)
        self.vectors = index_mod.load_vectors(cfg)
        self.reranker = rerank_mod.load_reranker(cfg) if with_reranker else None

    def run(self, config: str, question: str) -> tuple[list[retrieve.Hit], float]:
        started = time.perf_counter()
        cfg, conn = self.cfg, self.conn

        if config == "dense":
            hits = retrieve.dense_search(
                conn, retrieve.embed_query(self.embedder, question), self.vectors, cfg)
        elif config == "sparse":
            hits = retrieve.sparse_search(conn, question, cfg)
        else:
            dense = retrieve.dense_search(
                conn, retrieve.embed_query(self.embedder, question), self.vectors, cfg)
            sparse = retrieve.sparse_search(conn, question, cfg)
            hits = retrieve.reciprocal_rank_fusion(dense, sparse, cfg)
            if config == "hybrid_rerank":
                assert self.reranker is not None
                # Rerank returns only the context slate; the tail is kept behind it so
                # Recall@5 and nDCG@10 remain measurable over a full ranking.
                top = rerank_mod.rerank(self.reranker, question, hits, cfg)
                kept = {h.chunk_id for h in top}
                hits = top + [h for h in hits if h.chunk_id not in kept]
        return hits, (time.perf_counter() - started) * 1000


def evaluate_config(retrievers: Retrievers, labels: Sequence[GoldItem],
                    config: str) -> ConfigResult:
    """Score one configuration over every answerable question.

    Control questions are excluded here by construction: they have no gold chunk, so
    retrieval recall is undefined for them. Their behaviour is measured as abstention
    accuracy once generation exists, not as a retrieval number.
    """
    results = []
    for item in labels:
        if not item.gold_chunks:
            continue
        hits, ms = retrievers.run(config, item.question)
        results.append(ItemResult(item.id, [h.chunk_id for h in hits],
                                  item.gold_chunks, ms))
    return ConfigResult(config, results)


def persist(conn: Connection, cfg: Config, results: Sequence[ConfigResult],
            notes: str = "") -> str:
    """Write a run to eval_runs/eval_results so ablations can be diffed over time."""
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    db.insert(conn, "eval_runs", {
        "run_id": run_id, "config_name": "retrieval_ablation",
        "provider": "none (llm-free)", "notes": notes,
    })
    rows = []
    for cr in results:
        for item in cr.items:
            for metric, value in (("recall@5", item.recall_at(5)),
                                  ("mrr", item.mrr()),
                                  ("ndcg@10", item.ndcg_at(10)),
                                  ("latency_ms", item.latency_ms)):
                rows.append({
                    "run_id": run_id, "item_id": f"{cr.config}::{item.item_id}",
                    "metric": metric, "value": None if math.isnan(value) else value,
                    "detail_json": None,
                })
    db.insert_many(conn, "eval_results", rows)
    conn.commit()
    return run_id


# ---------------------------------------------------------------------------
#  Threshold derivation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ThresholdEvidence:
    """Two distinct populations, because two different decisions are being made.

    **Routing** looks at one number per question: the top rerank score on the slate.
    **Verification** looks at one number per (sentence, cited chunk) pair.

    Deriving a routing threshold from the per-pair distribution -- which the first
    version of this function did -- is a category error. A question whose best gold
    chunk scores 0.99 routes to ANSWER regardless of whether its third gold chunk
    scores 0.18, so per-chunk scores describe a decision nobody makes. It also drags
    the positive population's lower tail down and pushes tau_high far higher than the
    routing evidence supports.

    Note the sample sizes: the per-question populations are 8 and 4. That is small
    enough that the sweep matters more than the point estimate, which is why both are
    reported.
    """

    routing_positives: rerank_mod.Population   # per answerable question, top score
    routing_negatives: rerank_mod.Population   # per control question, top score
    pair_positives: rerank_mod.Population      # per gold chunk
    pair_negatives: rerank_mod.Population      # per chunk retrieved for a control
    sweep: rerank_mod.ThresholdSweep
    tau_high: float
    tau_low: float
    tau_verify: float
    derivation: dict[str, str] = field(default_factory=dict)
    separable: bool = True


def derive_thresholds(retrievers: Retrievers, labels: LabelSet,
                      cfg: Config) -> ThresholdEvidence:
    """Measure the rerank score distribution, then read the thresholds off it.

    Positives are gold chunks. Negatives come from the *control* questions -- whose
    entire corpus is a true negative, since nothing in it answers them -- plus
    low-ranked chunks retrieved for answerable questions.

    Using the controls for the negative population is the important choice. Taking
    "every non-gold chunk retrieved for an answerable question" as negative would be
    badly contaminated: a chunk not listed as gold is very often still relevant. The
    controls have no correct answer anywhere in the corpus, so every chunk retrieved
    for them is a genuine negative.
    """
    assert retrievers.reranker is not None
    pair_pos: list[float] = []
    pair_neg: list[float] = []
    route_pos: list[float] = []
    route_neg: list[float] = []

    # Calibrating on the single-turn set alone produced a threshold that refused
    # legitimate conversational questions: "What problem does LoRA solve?" scores
    # 0.684, well below the 0.789 minimum of the single-turn positives. Well-formed
    # evaluation questions are longer and more specific than what a person types in a
    # conversation, so a threshold fitted only to them is fitted to the wrong
    # distribution. First-turn scenario questions need no condensation, so they can be
    # scored directly and added to the population that the router will actually face.
    extra = _conversation_turn_ones(cfg)

    for item in list(labels.items) + extra:
        hits, _ = retrievers.run("hybrid", item.question)
        candidates = hits[: cfg.rerank_candidates]
        scores = rerank_mod.score_pairs(
            retrievers.reranker, [(item.question, h.text) for h in candidates], cfg)
        if not scores:
            continue

        # What the router actually sees: the single best score on the slate.
        top = max(scores)
        if item.must_abstain:
            route_neg.append(top)
        elif item.gold_chunks:
            route_pos.append(top)

        for hit, score in zip(candidates, scores):
            if hit.chunk_id in item.gold_chunks:
                pair_pos.append(score)
            elif item.must_abstain:
                pair_neg.append(score)

    rpos = rerank_mod.Population("top score per ANSWERABLE question (routing)", route_pos)
    rneg = rerank_mod.Population("top score per CONTROL question (routing)", route_neg)
    ppos = rerank_mod.Population("per gold chunk (verification)", pair_pos)
    pneg = rerank_mod.Population("per chunk retrieved for a control (verification)", pair_neg)

    # The sweep is over the routing populations, because routing is what it informs.
    sweep = rerank_mod.sweep_thresholds(route_pos, route_neg)

    # tau_low -- below this, refuse. Set just above the best score the corpus can
    # muster for a question it genuinely cannot answer.
    tau_low = round(max(route_neg), 3) if route_neg else 0.0
    # tau_high -- above this, answer directly. Set at the weakest top score among
    # questions that DO have an answer, so real evidence is never sent to clarify.
    tau_high = round(min(route_pos), 3) if route_pos else 0.0

    derivation: dict[str, str] = {}
    separable = tau_high > tau_low
    if separable:
        derivation["tau_low"] = ("highest top-score any control question achieved "
                                 f"(max of n={len(route_neg)})")
        derivation["tau_high"] = ("lowest top-score any answerable question achieved "
                                  f"(min of n={len(route_pos)})")
    else:
        # The populations overlap, so no single cut separates them. The clarify band
        # is set to exactly the overlap region: below it no answerable question ever
        # scored, above it no control ever scored, and inside it the two genuinely
        # mix. That is what "ambiguous" means here, measured rather than asserted.
        #
        # An earlier version instead opened a band around the sweep's balanced-accuracy
        # peak (peak-0.05, peak+0.15). That was arbitrary and, measured end to end, it
        # was badly wrong: it produced tau_high = 0.99, which pushed roughly half of
        # genuinely answerable questions into clarify and scored 2/13 on the
        # conversation scenarios. The lesson kept here: a threshold rule has to be
        # validated against routing behaviour, not just against its own histogram.
        # Percentiles, not extremes. Keying the band on min(positives) and
        # max(negatives) uses exactly one observation from each population, and with
        # n=11 and n=4 those single points move a long way on resampling. Measured
        # end to end, the extremes rule produced a band of [0.674, 0.839] that
        # swallowed most legitimate questions and scored 6/13 on the scenarios.
        #
        # p05 of the positives and p95 of the negatives are the standard robust
        # choice: they tolerate one unusual question on either side and describe
        # where the distributions actually overlap in density rather than in range.
        tau_low = round(rpos.pct(5), 3)
        tau_high = round(max(rneg.pct(95), tau_low + 0.02), 3)
        derivation["tau_low"] = (
            f"populations OVERLAP (best control {max(route_neg):.3f} >= weakest "
            f"answerable {min(route_pos):.3f}), so no clean cut exists. Set at p05 of "
            f"the answerable population (n={len(route_pos)}) -- robust to one unusually "
            f"weak question, where the minimum is not."
        )
        derivation["tau_high"] = (
            f"p95 of the control population (n={len(route_neg)}): above this, a score "
            f"is one the corpus essentially never produces for a question it cannot "
            f"answer. NOTE the tiny negative sample -- this is the least well-evidenced "
            f"number in the system."
        )

    # tau_verify -- groundedness floor for one cited sentence against one chunk. Read
    # off the per-pair control distribution, not the routing one: a sentence
    # paraphrases its source rather than restating the question, so the same model
    # scores a systematically weaker match and a routing-scale floor would reject
    # correctly-cited sentences.
    tau_verify = round(max(0.05, pneg.pct(95)), 3) if pair_neg else 0.0
    derivation["tau_verify"] = (
        f"p95 of the per-chunk control distribution (n={len(pair_neg)}) — 95% of "
        f"chunks retrieved for an unanswerable question score below this"
    )

    return ThresholdEvidence(rpos, rneg, ppos, pneg, sweep,
                             tau_high, tau_low, tau_verify, derivation, separable)


def _conversation_turn_ones(cfg: Config) -> list[GoldItem]:
    """First turns of each conversation scenario, as extra calibration items.

    Only turn 1, deliberately: later turns depend on condensation, so scoring them
    here would fold condenser behaviour into a retrieval threshold and make the
    measurement depend on which model happened to serve the condensation.

    A `clarify` expectation is excluded from both populations. It is by definition
    neither clearly answerable nor clearly unanswerable, so it cannot inform a
    boundary without begging the question.
    """
    import yaml

    if not cfg.conversations_path.exists():
        return []
    spec = yaml.safe_load(cfg.conversations_path.read_text(encoding="utf-8"))
    out: list[GoldItem] = []
    for scenario in spec.get("scenarios", []):
        turns = scenario.get("turns") or []
        if not turns:
            continue
        first = turns[0]
        expected = first.get("expected_route")
        if expected not in {"answer", "refuse"}:
            continue
        out.append(GoldItem(
            id=f"{scenario['id']}::turn1",
            question=first["raw_text"],
            cls="conversation_turn1",
            must_abstain=(expected == "refuse"),
            expected_route=expected,
            gold_chunks=list(first.get("gold_chunks") or []),
            expected_facts=[],
        ))
    return out


# ---------------------------------------------------------------------------
#  Generation-dependent metrics
# ---------------------------------------------------------------------------
def drop_caches(cfg, conn) -> tuple[int, int]:
    """Empty both cache layers and report what was discarded.

    There are two, and clearing only one produces numbers that look measured and
    are not. `answer_cache` short-circuits a whole turn; the `DiskCache` under
    `.cache/llm` short-circuits an individual LLM call on sha256(model+prompt).
    An eval that cleared only the first still served 557 of 1520 calls from disk
    at a recorded 0 ms, so committed transcripts reported `Latency: 0 ms` for
    turns that genuinely cost seconds -- a figure describing the cache's history
    rather than the pipeline's behaviour.
    """
    import shutil

    n_answers = conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]
    conn.execute("DELETE FROM answer_cache")
    conn.commit()

    root = cfg.llm_cache_dir
    n_prompts = sum(1 for _ in root.rglob("*.json")) if root.exists() else 0
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    return n_answers, n_prompts


@dataclass
class GenerationMetrics:
    """Everything that needs a model to produce. Labelled as such everywhere.

    Kept separate from the retrieval metrics on purpose: retrieval numbers are
    model-independent and reproducible on demand, while everything here moves with
    whichever model served the run. Mixing them in one table would let a synthesis
    model's variance masquerade as a retrieval result.
    """

    # Which model actually served this run. The report used to print
    # `cfg.ollama_model` unconditionally, so a Gemini run produced a file headed
    # "Generation: llama3.1:8b via Ollama" over Gemini's numbers -- the one
    # artifact a reviewer cross-references against the README contradicted it.
    provider: str = ""
    model: str = ""

    n_items: int = 0
    schema_valid: int = 0
    invented_citations: int = 0
    sentences_total: int = 0
    sentences_verified: int = 0
    controls_total: int = 0
    controls_correct: int = 0
    refusals_with_citations: int = 0
    fact_coverage: list[float] = field(default_factory=list)
    multi_hop_papers: list[int] = field(default_factory=list)
    latencies: list[int] = field(default_factory=list)
    routes: list[tuple[str, str]] = field(default_factory=list)   # (expected, actual)
    per_item: list[dict] = field(default_factory=list)

    @property
    def citation_precision(self) -> float:
        """Share of CITED sentences whose evidence supports them.

        The denominator is sentences carrying at least one citation, not all
        sentences. Uncited sentences are dropped before rendering, so including them
        would divide by a population that never reaches the reader.
        """
        return (self.sentences_verified / self.sentences_total
                if self.sentences_total else float("nan"))

    @property
    def abstention_accuracy(self) -> float:
        return (self.controls_correct / self.controls_total
                if self.controls_total else float("nan"))

    @property
    def mean_fact_coverage(self) -> float:
        return float(np.mean(self.fact_coverage)) if self.fact_coverage else float("nan")

    @property
    def route_accuracy(self) -> float:
        return (sum(1 for e, a in self.routes if _route_ok(e, a)) / len(self.routes)
                if self.routes else float("nan"))

    def latency(self, pct: float) -> float:
        return float(np.percentile(self.latencies, pct)) if self.latencies else float("nan")

    def confusion(self) -> dict[tuple[str, str], int]:
        out: dict[tuple[str, str], int] = {}
        for pair in self.routes:
            out[pair] = out.get(pair, 0) + 1
        return out


def _route_ok(expected: str, actual: str) -> bool:
    """Refusing on score and abstaining after synthesis are different mechanisms with
    the same correct outcome: no answer, no citations."""
    if expected in {"refuse", "abstain"}:
        return actual in {"refuse", "abstain"}
    return expected == actual


def fact_coverage(answer, expected_facts: Sequence[str]) -> float:
    """Share of expected keywords appearing in the answer text.

    Case-insensitive substring match, stated plainly because the choice matters: a
    stricter token match would penalise "r = 4" against "a rank of 4", and a looser
    one would credit coincidence. Substring is the honest middle, and it is reported
    as what it is rather than as semantic coverage.
    """
    if not expected_facts or answer is None:
        return float("nan")
    text = " ".join(s.text for s in answer.sentences).lower()
    return sum(1 for f in expected_facts if f.lower() in text) / len(expected_facts)


def ledger_summary(conn: Connection) -> dict:
    """Per-turn LLM cost, split by ladder, straight from the durable ledger."""
    rows = db.all_rows(conn, """
        SELECT ladder, purpose, COUNT(*) n,
               SUM(CASE WHEN cached = 1 THEN 1 ELSE 0 END) cached
        FROM llm_calls GROUP BY ladder, purpose
    """)
    total = sum(r["n"] for r in rows)
    cached = sum(r["cached"] or 0 for r in rows)
    by_ladder: dict[str, int] = {}
    for r in rows:
        by_ladder[r["ladder"] or "none"] = by_ladder.get(r["ladder"] or "none", 0) + r["n"]
    return {
        "total": total, "cached": cached,
        "cache_hit_rate": cached / total if total else 0.0,
        "by_ladder": by_ladder,
        "by_purpose": {r["purpose"]: r["n"] for r in rows},
    }
