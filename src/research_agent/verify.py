"""Groundedness verification with zero LLM calls.

Each answer sentence is scored against each chunk it cites, using the cross-encoder
that is already loaded. Free, deterministic, repeatable across evaluation runs, and it
consumes no quota — which is what makes it affordable to run on every single turn
rather than only when someone remembers to.

**The trap this file is built around.** In my previous system the validator received
500-character truncations while the synthesiser saw full chunks plus their parents.
The result was false unsupported-claim flags on figures that were, in fact, correctly
sourced — the validator was marking the synthesiser wrong for using evidence the
validator had never been shown. Here the context string is built once in `prompts.py`,
carried on the `Answer`, and the verifier scores against the *same passage objects*.
`tests/test_verify.py` asserts that identity rather than trusting this paragraph.

**Honest caveat, stated in the README too.** This is a relevance proxy, not a trained
NLI model. It answers "does this passage look relevant to this sentence?", not "does
this passage entail this sentence?" It shares a model family with the reranker, so
verifier and retriever are correlated by construction: a passage the reranker liked
will tend to score well here too. That correlation is measured and reported rather
than glossed over.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .answer import Answer, CitedSentence
from .config import Config
from .rerank import score_pairs

VERIFIED = "verified"
REPAIRED = "repaired"
UNVERIFIED = "unverified"


def verify_answer(cfg: Config, reranker, answer: Answer) -> Answer:
    """Score every (sentence, cited chunk) pair and label each sentence.

    A sentence is `verified` when every chunk it cites clears the floor. Requiring
    *all* of them rather than the best one is deliberate: a sentence citing two
    passages is claiming both support it, and letting one strong citation carry a
    weak one is exactly how a plausible-but-unsupported clause survives review.
    """
    if answer.is_refusal or not answer.sentences:
        # A refusal has no citations, so there is nothing to ground. Verification is
        # not "skipped" here so much as trivially satisfied.
        return answer

    cfg.require_measured_thresholds()

    pairs: list[tuple[str, str]] = []
    index: list[tuple[int, str]] = []
    for s in answer.sentences:
        for cid in s.chunk_ids:
            hit = answer.hit_by_id(cid)
            if hit is None:
                continue
            # The verifier is handed the hit's own text -- the identical object the
            # synthesiser's context was rendered from -- never a re-fetch or a
            # truncation.
            pairs.append((s.text, hit.text))
            index.append((s.idx, cid))

    scores = score_pairs(reranker, pairs, cfg)

    by_sentence: dict[int, dict[str, float]] = {}
    for (sidx, cid), score in zip(index, scores):
        by_sentence.setdefault(sidx, {})[cid] = score

    verified: list[CitedSentence] = []
    for s in answer.sentences:
        found = by_sentence.get(s.idx, {})
        status = (VERIFIED
                  if found and all(v >= cfg.tau_verify for v in found.values())
                  else UNVERIFIED)
        verified.append(replace(s, verify_scores=found, status=status))

    return replace(answer, sentences=verified)


def repair(cfg: Config, client, reranker, answer: Answer) -> Answer:
    """One repair pass over the sentences that failed, then stop.

    Exactly one. An unbounded repair loop spends quota re-litigating a claim the
    evidence does not support, and the honest outcome for such a claim is to mark it
    rather than to keep rewriting until something scores above the floor.

    A sentence that still fails is rendered with `[unverified]` rather than dropped.
    Silently removing it would leave an answer that looks clean while having quietly
    discarded a claim the reader never learns was made — transparency beats polish.
    """
    from . import prompts

    failing = [s for s in answer.sentences if s.status == UNVERIFIED]
    if not failing:
        return answer

    repaired: dict[int, CitedSentence] = {}
    for s in failing:
        try:
            completion = client.complete(
                prompts.repair_prompt(s.text, answer.hits),
                purpose="repair", ladder="synthesis",
                schema=prompts.REPAIR_SCHEMA,
            )
        except Exception:
            # A failed repair leaves the sentence flagged. It never escalates into
            # losing the answer.
            continue

        data = completion.data or {}
        if data.get("unsupported") or not (data.get("text") or "").strip():
            continue

        valid = {h.chunk_id for h in answer.hits}
        cites = [c for c in (data.get("cite") or []) if c in valid]
        if not cites:
            continue

        candidate = replace(s, text=data["text"].strip(), chunk_ids=cites)
        rescored = score_pairs(
            reranker,
            [(candidate.text, h.text) for h in
             (answer.hit_by_id(c) for c in cites) if h is not None],
            cfg,
        )
        if rescored and all(v >= cfg.tau_verify for v in rescored):
            repaired[s.idx] = replace(
                candidate, status=REPAIRED,
                verify_scores=dict(zip(cites, rescored)),
            )

    if not repaired:
        return answer
    return replace(answer, sentences=[repaired.get(s.idx, s) for s in answer.sentences])


def verification_summary(answer: Answer) -> dict[str, int]:
    out: dict[str, int] = {VERIFIED: 0, REPAIRED: 0, UNVERIFIED: 0}
    for s in answer.sentences:
        out[s.status] = out.get(s.status, 0) + 1
    return out


def context_fingerprint(hits: Sequence) -> str:
    """Hash of the exact passage text used, so the identity claim is testable."""
    import hashlib

    h = hashlib.sha256()
    for hit in hits:
        h.update(hit.chunk_id.encode("utf-8"))
        h.update(hit.text.encode("utf-8"))
    return h.hexdigest()[:16]
