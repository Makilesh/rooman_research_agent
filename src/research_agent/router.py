"""Three-way confidence routing: answer, ask, or refuse.

Binary answer/refuse forces a bad choice on an ambiguous question. "What rank is
used?" over a corpus containing LoRA and QLoRA is not unanswerable — it is
*underspecified*, and the useful response is to ask which one rather than to guess
confidently or refuse unhelpfully.

Routing reads the **top** rerank score on the slate, never an average. Averaging
across retrieved chunks makes retrieving *more* results look *worse* and silently
penalises recall — the tail of any slate is noise by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import Config
from .retrieve import Hit

ANSWER = "answer"
CLARIFY = "clarify"
REFUSE = "refuse"


@dataclass(frozen=True, slots=True)
class Route:
    decision: str
    top_score: float
    reason: str
    competing: list[Hit]

    @property
    def is_answer(self) -> bool:
        return self.decision == ANSWER


def route(cfg: Config, hits: Sequence[Hit]) -> Route:
    """Decide what to do with this slate."""
    cfg.require_measured_thresholds()

    if not hits:
        return Route(REFUSE, 0.0, "retrieval returned nothing", [])

    scored = [h for h in hits if h.rerank_score is not None]
    if not scored:
        # Un-reranked slates cannot be routed on a rerank threshold. Answering anyway
        # would apply a measured threshold to an unmeasured quantity.
        return Route(ANSWER, 0.0, "no rerank scores available; routing skipped", list(hits))

    top = max(h.rerank_score or 0.0 for h in scored)

    if top >= cfg.tau_high:
        return Route(ANSWER, top,
                     f"top rerank {top:.3f} >= tau_high {cfg.tau_high}", list(hits))
    if top < cfg.tau_low:
        return Route(REFUSE, top,
                     f"top rerank {top:.3f} < tau_low {cfg.tau_low}: nothing in the "
                     f"corpus is a strong enough match to answer from", list(hits))

    competing = competing_candidates(cfg, scored)
    return Route(CLARIFY, top,
                 f"top rerank {top:.3f} falls between tau_low {cfg.tau_low} and "
                 f"tau_high {cfg.tau_high}: the evidence is suggestive but not "
                 f"decisive", competing)


def competing_candidates(cfg: Config, hits: Sequence[Hit], limit: int = 4) -> list[Hit]:
    """The distinct alternatives a clarifying question should name.

    One per document: repeating three chunks from the same paper describes one option
    three times, which is not a choice.
    """
    seen: set[str] = set()
    out: list[Hit] = []
    for hit in sorted(hits, key=lambda h: -(h.rerank_score or 0.0)):
        if (hit.rerank_score or 0.0) < cfg.tau_low:
            continue
        if hit.doc_id in seen:
            continue
        seen.add(hit.doc_id)
        out.append(hit)
        if len(out) >= limit:
            break
    return out


def apply_diversity_guard(cfg: Config, hits: Sequence[Hit],
                          top_n: int | None = None) -> list[Hit]:
    """Ensure a second paper survives into context when it has earned a place.

    Measured at Step 7: all three multi-hop questions answered from a single paper,
    and in two of them the second paper never reached the context slate at all.
    Reranking is per-passage and paper-blind, so one paper's chunks can take every
    slot even when another paper clears the floor comfortably.

    This reserves the last slot for the best passage from a document not already
    represented, provided it clears `tau_low`. It cannot manufacture evidence -- if
    only one paper is relevant, nothing changes -- but it stops a monopoly that is an
    artefact of ranking rather than of the corpus.
    """
    limit = top_n or cfg.context_top_n
    ranked = sorted(hits, key=lambda h: -(h.rerank_score or 0.0))
    if len(ranked) <= 1:
        return list(ranked[:limit])

    kept = list(ranked[:limit])
    represented = {h.doc_id for h in kept}
    if len(represented) > 1:
        return kept  # already diverse; nothing to do

    for hit in ranked[limit:]:
        if hit.doc_id in represented:
            continue
        if (hit.rerank_score or 0.0) < cfg.tau_low:
            break  # ranked descending, so nothing further can qualify
        # Displace the weakest passage from the monopolising paper, not the strongest.
        kept[-1] = hit
        break
    return kept


def merge_clarification(original: str, reply: str) -> str:
    """Fold the user's answer to a clarifying question back into the query.

    Both halves are kept. The reply alone ("In LoRA, for GPT-3.") is not a question,
    and the original alone is what was ambiguous in the first place.
    """
    reply = reply.strip().rstrip(".")
    return f"{original.rstrip('?').strip()}? Specifically: {reply}."
