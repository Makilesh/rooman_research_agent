"""Cross-encoder reranking, and the thresholds derived from its score distribution.

A bi-encoder embeds query and passage separately, so it never sees the interaction
between their tokens. The cross-encoder does, on the top-25 only -- full-corpus
cross-encoding would be quadratic and pointless.

On thresholds: cross-encoder scores are sharply bimodal, and the naive move -- average
the scores of everything retrieved -- makes retrieving *more* results look *worse* and
silently penalises recall. Usable evidence is scored against a measured floor instead,
and the floor comes from the observed distribution rather than from another project.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .config import Config
from .retrieve import Hit


def load_reranker(cfg: Config):
    """Load bge-reranker-v2-m3, refusing a silent CPU fallback for the same reason."""
    import torch
    from sentence_transformers import CrossEncoder

    if cfg.require_cuda and cfg.embed_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "EMBED_DEVICE is cuda but CUDA is unavailable. See `doctor --gpu`."
        )
    dtype = torch.float16 if cfg.embed_dtype == "float16" else torch.float32
    return CrossEncoder(
        cfg.rerank_model,
        device=cfg.embed_device,
        max_length=cfg.rerank_max_tokens,
        model_kwargs={"torch_dtype": dtype},
    )


def _assert_squashed(model) -> None:
    """Confirm the model applies its own sigmoid, and never squash twice.

    Caught the hard way. `CrossEncoder.predict()` applies `Sigmoid()` by default, so
    squashing the result again maps the true range [0.0, 0.998] onto [0.5, 0.73].
    Nothing errors -- the scores still sort, the pipeline still runs -- but every
    passage looks equally relevant, the bimodal separation vanishes, and any threshold
    derived from that distribution is meaningless. This is checked rather than assumed
    because a library default is not a contract.
    """
    import torch

    activation = getattr(model, "activation_fn", None)
    if not isinstance(activation, torch.nn.Sigmoid):
        raise RuntimeError(
            f"Expected CrossEncoder.activation_fn to be Sigmoid, found {activation!r}. "
            f"Scores would be raw logits rather than [0,1], and every measured "
            f"threshold would be on the wrong scale."
        )


def rerank(model, query: str, hits: Sequence[Hit], cfg: Config) -> list[Hit]:
    """Score (query, chunk) pairs and return the top-N in [0,1]."""
    candidates = list(hits[: cfg.rerank_candidates])
    if not candidates:
        return []

    scores = score_pairs(model, [(query, h.text) for h in candidates], cfg)
    scored = [replace(h, rerank_score=s) for h, s in zip(candidates, scores)]
    scored.sort(key=lambda h: -(h.rerank_score or 0.0))
    return [replace(h, rank=i + 1) for i, h in enumerate(scored[: cfg.context_top_n])]


def score_pairs(model, pairs: Sequence[tuple[str, str]], cfg: Config) -> list[float]:
    """Relevance in [0,1] for arbitrary (query, text) pairs.

    Used by the threshold derivation and, at Step 8, by groundedness verification --
    where the "query" is an answer sentence rather than a user question.
    """
    if not pairs:
        return []
    _assert_squashed(model)
    raw = model.predict(list(pairs), batch_size=cfg.rerank_batch_size,
                        show_progress_bar=False)
    return [float(s) for s in np.asarray(raw).reshape(-1)]


# ---------------------------------------------------------------------------
#  Threshold derivation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Population:
    """Summary of one score population, printed as the evidence for a threshold."""

    name: str
    scores: list[float]

    @property
    def n(self) -> int:
        return len(self.scores)

    def pct(self, p: float) -> float:
        return float(np.percentile(self.scores, p)) if self.scores else float("nan")

    def summary(self) -> dict[str, float]:
        if not self.scores:
            return {}
        a = np.asarray(self.scores)
        return {
            "n": len(a), "min": float(a.min()), "p05": self.pct(5),
            "p25": self.pct(25), "median": self.pct(50), "p75": self.pct(75),
            "p95": self.pct(95), "max": float(a.max()), "mean": float(a.mean()),
        }


def histogram(pop: Population, bins: int = 20, width: int = 46) -> list[str]:
    """A text histogram. The thresholds have to be justified by a visible
    distribution, not asserted."""
    if not pop.scores:
        return ["(no scores)"]
    counts, edges = np.histogram(pop.scores, bins=bins, range=(0.0, 1.0))
    peak = max(counts.max(), 1)
    out = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "█" * int(round(width * c / peak))
        out.append(f"  {lo:4.2f}-{hi:4.2f} | {bar:<{width}} {c}")
    return out


@dataclass(frozen=True, slots=True)
class ThresholdSweep:
    """Route accuracy as a function of tau, rather than a single asserted number.

    A point estimate derived from ~18 positive examples would be a number with no
    error bars presented as if it had them. A curve shows how sensitive the routing
    actually is, which is both more honest and more useful.
    """

    taus: list[float]
    positive_retention: list[float]   # share of gold chunks scoring >= tau
    negative_rejection: list[float]   # share of non-gold chunks scoring < tau

    def best_f1(self) -> tuple[float, float]:
        best_tau, best = self.taus[0], -1.0
        for tau, keep, reject in zip(self.taus, self.positive_retention,
                                     self.negative_rejection):
            # Balanced accuracy rather than F1: the negative pool is thousands of
            # times larger than the positive one, so precision is meaningless here.
            balanced = (keep + reject) / 2
            if balanced > best:
                best_tau, best = tau, balanced
        return best_tau, best


def sweep_thresholds(
    positives: Sequence[float], negatives: Sequence[float], steps: int = 41
) -> ThresholdSweep:
    taus = [i / (steps - 1) for i in range(steps)]
    pos = np.asarray(positives) if len(positives) else np.zeros(0)
    neg = np.asarray(negatives) if len(negatives) else np.zeros(0)
    return ThresholdSweep(
        taus=taus,
        positive_retention=[float((pos >= t).mean()) if pos.size else 0.0 for t in taus],
        negative_rejection=[float((neg < t).mean()) if neg.size else 0.0 for t in taus],
    )
