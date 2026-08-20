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
    positives: rerank_mod.Population
    negatives: rerank_mod.Population
    sweep: rerank_mod.ThresholdSweep
    tau_high: float
    tau_low: float
    tau_verify: float


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
    positives: list[float] = []
    negatives: list[float] = []

    for item in labels.items:
        hits, _ = retrievers.run("hybrid", item.question)
        pairs = [(item.question, h.text) for h in hits[: cfg.rerank_candidates]]
        scores = rerank_mod.score_pairs(retrievers.reranker, pairs, cfg)
        for hit, score in zip(hits, scores):
            if hit.chunk_id in item.gold_chunks:
                positives.append(score)
            elif item.must_abstain:
                negatives.append(score)

    pos = rerank_mod.Population("gold chunks (answerable questions)", positives)
    neg = rerank_mod.Population("all chunks retrieved for control questions", negatives)
    sweep = rerank_mod.sweep_thresholds(positives, negatives)

    # tau_low: the floor below which refusing is right. Set from the negative
    # population's upper tail -- above almost every score the corpus produces for a
    # question it cannot answer.
    tau_low = round(neg.pct(95), 3) if negatives else 0.0
    # tau_high: answering directly is safe. Set from the positive population's lower
    # tail, so genuine evidence is not routed to clarification.
    tau_high = round(pos.pct(25), 3) if positives else 0.0
    # Keep the band non-degenerate even if the populations overlap more than expected.
    if tau_high <= tau_low:
        tau_high = round(min(0.99, tau_low + 0.05), 3)
    # tau_verify: groundedness floor for sentence-to-chunk support at Step 8. Softer
    # than tau_low, because a sentence paraphrases its source rather than restating
    # the question, and the same model scores a systematically weaker match.
    tau_verify = round(max(0.05, neg.pct(75)), 3) if negatives else 0.0

    return ThresholdEvidence(pos, neg, sweep, tau_high, tau_low, tau_verify)
