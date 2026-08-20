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

# Content tokens shared between a sentence and its passage, as a fraction of the
# sentence. A grounded sentence either quotes its source or paraphrases it closely,
# and both leave heavy lexical traces.
LEXICAL_FLOOR = 0.70
# Longest run of consecutive sentence tokens found verbatim in the passage. Five is
# long enough that matching by coincidence is implausible for technical prose.
NGRAM_FLOOR = 5

_STOP = frozenset("""
a an the of for to in on at by with from is are was were be been being do does did
and or but if then than that this these those it its as into about we our us they
their can could should would may might will shall not no other more most such using
use used than which who whom whose how why when where each any all both same so
""".split())


def _content_tokens(text: str) -> list[str]:
    import re

    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def lexical_support(sentence: str, passage: str) -> float:
    """Deterministic containment score in [0, 1]. No model, no quota, no randomness.

    This exists because measurement showed the cross-encoder cannot be trusted for
    this job on its own. A sentence lifted **verbatim** from its cited passage scored
    0.3438, while a passage that did *not* contain it scored 0.6218. That is not a
    tuning problem: the cross-encoder scores topical relevance between a *query* and a
    *document*, and "does this long passage contain this short statement" is a
    different question that it answers badly.

    Lexical overlap answers exactly that question, and it answers it perfectly for the
    case that dominates in practice -- a grounded model quotes or closely paraphrases
    its source. The cross-encoder is kept for genuine paraphrase, where lexical
    overlap is weak but the claim is still supported. A sentence passes on either.
    """
    s_tokens = _content_tokens(sentence)
    if not s_tokens:
        return 0.0
    p_tokens = _content_tokens(passage)
    p_set = set(p_tokens)

    coverage = sum(1 for t in s_tokens if t in p_set) / len(s_tokens)

    # Longest contiguous run of sentence tokens appearing in order in the passage.
    longest = run = 0
    positions = {t: i for i, t in enumerate(p_tokens)}
    prev = None
    for t in s_tokens:
        idx = positions.get(t)
        if idx is not None and prev is not None and idx == prev + 1:
            run += 1
        elif idx is not None:
            run = 1
        else:
            run = 0
        longest = max(longest, run)
        prev = idx

    # A long verbatim run is decisive on its own; otherwise fall back to coverage.
    return 1.0 if longest >= NGRAM_FLOOR else coverage


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
    index: list[tuple[int, str, str, str]] = []
    for s in answer.sentences:
        for cid in s.chunk_ids:
            hit = answer.hit_by_id(cid)
            if hit is None:
                continue
            # The verifier is handed the hit's own text -- the identical object the
            # synthesiser's context was rendered from -- never a re-fetch or a
            # truncation.
            pairs.append((s.text, hit.text))
            index.append((s.idx, cid, hit.text, s.text))

    scores = score_pairs(reranker, pairs, cfg)

    by_sentence: dict[int, dict[str, float]] = {}
    for (sidx, cid, passage, sentence), ce_score in zip(index, scores):
        # A sentence counts as supported if EITHER signal says so. Lexical containment
        # catches quotation and close paraphrase, which the cross-encoder scores
        # badly; the cross-encoder catches genuine paraphrase, where lexical overlap
        # is weak. Requiring both would fail almost everything that is actually true.
        lex = lexical_support(sentence, passage)
        by_sentence.setdefault(sidx, {})[cid] = max(
            ce_score, lex if lex >= LEXICAL_FLOOR else 0.0
        )

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
