"""Every prompt in the system, in one readable file.

Kept together deliberately. Prompts scattered through the modules that call them are
impossible to review as a set, and the interaction between them — what the synthesiser
is told versus what the judge is told — is where grounded systems actually fail.

The citation contract is enforced twice over: the schema makes a malformed shape hard
to emit, and `answer.py` rejects any chunk id that was not in the retrieved set. The
prompt's job is only to make the right answer the easy one.
"""

from __future__ import annotations

from typing import Sequence

from .retrieve import Hit

# ---------------------------------------------------------------------------
#  The citation contract
# ---------------------------------------------------------------------------
# Constrained decoding on both providers is what turns the contract from a request
# into a guarantee about shape. It cannot guarantee the ids are real -- that is
# validated separately, because a model can emit a well-formed lie.
ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "insufficient_evidence": {
            "type": "boolean",
            "description": "True if the sources do not contain the answer.",
        },
        "refusal_reason": {
            "type": ["string", "null"],
            "description": "If insufficient_evidence, why the sources fall short.",
        },
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "cite": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Chunk ids supporting this sentence.",
                    },
                },
                "required": ["text", "cite"],
            },
        },
    },
    "required": ["insufficient_evidence", "refusal_reason", "sentences"],
}


SYSTEM = """You are a research assistant answering strictly from a fixed set of source passages.

RULES, in order of priority:

1. Use ONLY the passages provided. You have background knowledge about these papers.
   Do not use it. If a fact is not in the passages, it does not exist for this answer.

2. Every sentence you write must cite at least one passage id, exactly as given in
   the SOURCES block (for example c_lora_0020). Never invent an id. Never cite an id
   that is not in the SOURCES block. If you cannot support a sentence with a passage,
   delete the sentence.

3. CHECK THE ATTRIBUTION BEFORE YOU ANSWER. Each passage is labelled with the paper
   it comes from. If the question asks what a SPECIFIC paper, model, or system says,
   and the passages that mention the topic come from a DIFFERENT paper, then the
   sources do not answer the question. Say so. A passage about the right topic from
   the wrong paper is not evidence — it is the most convincing way to be wrong, and
   it will pass every other check because the sentence really is in the passage you
   cited.

4. If the passages do not contain the answer, set insufficient_evidence to true, give
   a refusal_reason naming what is missing, and return an empty sentences list. This
   is a correct and valued outcome, not a failure. Refusing when the sources are
   silent is more useful than a plausible guess.

5. If the question assumes something the passages contradict, do NOT play along and do
   NOT simply refuse. Say plainly that the premise is wrong, then cite what the
   passages actually say. Set insufficient_evidence to false for this case.

6. IF THE QUESTION COMPARES TWO THINGS, ANSWER FROM BOTH. When the question asks how
   one method differs from, extends, or compares with another, and the passages
   include material from more than one paper, your answer must draw on both and cite
   both. Do not answer a comparative question from the stronger paper alone — an
   answer that describes only one side of a comparison is not an answer to the
   question that was asked, however well cited it is.

   If the passages genuinely only cover one side, say which side is missing rather
   than presenting a one-sided answer as complete.

7. Be specific. Prefer the exact figure, name, or setting from the passage over a
   paraphrase. Do not add caveats the passages do not support."""


FEW_SHOT = """EXAMPLE — a supported answer:

SOURCES:
[c_demo_0001] (Demo Paper · p.4) We train for 100k steps with a batch size of 32.

QUESTION: What batch size is used?

{"insufficient_evidence": false, "refusal_reason": null, "sentences": [
  {"text": "The model is trained with a batch size of 32.", "cite": ["c_demo_0001"]},
  {"text": "Training runs for 100k steps.", "cite": ["c_demo_0001"]}]}

EXAMPLE — the sources are silent:

SOURCES:
[c_demo_0002] (Demo Paper · p.4) We train for 100k steps with a batch size of 32.

QUESTION: What learning rate schedule is used?

{"insufficient_evidence": true, "refusal_reason": "The passages give the batch size
and step count but say nothing about a learning rate schedule.", "sentences": []}

EXAMPLE — a comparative question, answered from BOTH papers:

SOURCES:
[c_demo_0010] (Alpha Paper · p.3) Alpha freezes the base weights and trains a small
adapter, cutting trainable parameters by 99%.
[c_demo_0011] (Beta Paper · p.5) Beta additionally quantises the frozen base to 4-bit,
reducing memory a further 3x over Alpha.

QUESTION: How does Beta reduce memory beyond what Alpha achieves?

{"insufficient_evidence": false, "refusal_reason": null, "sentences": [
  {"text": "Alpha freezes the base weights and trains a small adapter, cutting
trainable parameters by 99%.", "cite": ["c_demo_0010"]},
  {"text": "Beta goes further by quantising that frozen base to 4-bit, reducing
memory a further 3x over Alpha.", "cite": ["c_demo_0011"]}]}

Both papers are cited, because the question asked about both.

EXAMPLE — right topic, WRONG PAPER (the passages must be refused):

SOURCES:
[c_demo_0004] (Beta Paper · p.9) We provide evaluations on MMLU.

QUESTION: How does the Alpha paper evaluate on MMLU?

{"insufficient_evidence": true, "refusal_reason": "The only passage mentioning MMLU
is from the Beta paper. Nothing here shows the Alpha paper evaluating on MMLU.",
"sentences": []}

EXAMPLE — the question's premise is wrong:

SOURCES:
[c_demo_0003] (Beta Paper · p.2) Beta introduces 4-bit quantisation.

QUESTION: Which section of the Alpha paper describes its 4-bit quantisation?

{"insufficient_evidence": false, "refusal_reason": null, "sentences": [
  {"text": "The premise is incorrect: 4-bit quantisation is not introduced by the
Alpha paper.", "cite": ["c_demo_0003"]},
  {"text": "It is introduced by the Beta paper instead.", "cite": ["c_demo_0003"]}]}"""


def format_sources(hits: Sequence[Hit]) -> str:
    """Render the context block.

    This exact string is what the synthesiser sees, and `verify.py` must score
    against the very same passage text. In my previous system the validator received
    500-character truncations while the synthesiser got full chunks plus parents,
    which produced false unsupported-claim flags on correctly-sourced figures. The
    context is therefore built once, here, and passed to both.
    """
    blocks = []
    for hit in hits:
        blocks.append(f"[{hit.chunk_id}] ({hit.source_label})\n{hit.text.strip()}")
    return "\n\n".join(blocks)


def synthesis_prompt(question: str, hits: Sequence[Hit]) -> str:
    return (
        f"{SYSTEM}\n\n{FEW_SHOT}\n\n"
        f"--- BEGIN SOURCES ---\n{format_sources(hits)}\n--- END SOURCES ---\n\n"
        f"QUESTION: {question}\n\n"
        f"Valid citation ids for this question, and no others: "
        f"{', '.join(h.chunk_id for h in hits)}\n\n"
        f"Respond with JSON matching the required schema."
    )


REPAIR_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "cite": {"type": "array", "items": {"type": "string"}},
        "unsupported": {"type": "boolean"},
    },
    "required": ["text", "cite", "unsupported"],
}


def repair_prompt(sentence: str, hits: Sequence[Hit]) -> str:
    """One rewrite attempt for a sentence its own evidence does not support."""
    return (
        "A sentence you wrote is not supported by the passage it cited.\n\n"
        f"--- BEGIN SOURCES ---\n{format_sources(hits)}\n--- END SOURCES ---\n\n"
        f"UNSUPPORTED SENTENCE: {sentence}\n\n"
        "Rewrite it so it states only what the passages actually support, and cite the "
        "passage ids that support it. If no passage supports any version of this "
        "claim, set unsupported to true and leave text empty — dropping a claim is "
        "better than dressing it up.\n\n"
        "Respond with JSON matching the required schema."
    )


# ---------------------------------------------------------------------------
#  Condensation
# ---------------------------------------------------------------------------
CONDENSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "standalone_query": {
            "type": "string",
            "description": "The follow-up rewritten so it stands alone.",
        },
    },
    "required": ["standalone_query"],
}

CONDENSE_SYSTEM = """Rewrite a follow-up question so it stands alone, without the conversation.

THE ONE UNBREAKABLE RULE: use only words that already appear in the conversation
above or in the follow-up itself. Do not introduce any new noun, technical term, or
topic word — not even one that seems obviously right or more precise.

This matters more than fluency. A rewritten query that adds an invented term steers
retrieval into documents the conversation was never about, and the resulting answer
is confident, well-formed, and wrong.

WHAT YOU SHOULD DO:
- Replace pronouns ("it", "that", "they", "this method") with the specific thing they
  refer to, using the words the conversation already used for it.
- Carry forward context the follow-up assumes.
- Keep it short. A stilted query that retrieves correctly beats an elegant one that
  does not.

WHAT YOU MUST NOT DO:
- Invent a category, framework, or process word to make the query sound complete.
- Add a synonym you prefer over the word the conversation used.
- Answer the question. You are only rewriting it."""

CONDENSE_EXAMPLES = """EXAMPLE:

CONVERSATION:
User: What problem does LoRA solve?
Assistant: LoRA freezes the pretrained weights and injects trainable rank
decomposition matrices, reducing the number of trainable parameters.

FOLLOW-UP: How does the quantised version reduce memory further?

{"standalone_query": "How does the quantised version of LoRA reduce memory further?"}

EXAMPLE — resisting the temptation to add a word:

CONVERSATION:
User: What statuses can a request have?
Assistant: A request can be draft, submitted, or closed.

FOLLOW-UP: what about during approval?

{"standalone_query": "What statuses can a request have during approval?"}

Note what was NOT added: no "workflow", no "lifecycle", no "process". Those words are
not in the conversation, so they may not appear in the query."""


def condense_prompt(history: str, raw: str, extra_vocabulary: str = "") -> str:
    extra = (f"\nThe user referred to these sources, so their titles and sections are "
             f"also available vocabulary:\n{extra_vocabulary}\n"
             if extra_vocabulary else "")
    return (
        f"{CONDENSE_SYSTEM}\n\n{CONDENSE_EXAMPLES}\n\n"
        f"CONVERSATION:\n{history}\n{extra}\n"
        f"FOLLOW-UP: {raw}\n\n"
        f"Respond with JSON matching the required schema."
    )


# ---------------------------------------------------------------------------
#  Clarification
# ---------------------------------------------------------------------------
CLARIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "options"],
}


def clarify_prompt(question: str, candidates: Sequence[Hit]) -> str:
    """Build a clarifying question from the passages that are actually competing.

    A generic "could you clarify?" wastes the user's turn. The retrieved candidates
    already say what the ambiguity IS -- two papers, two sections, two settings -- so
    the question names them.
    """
    listing = "\n".join(
        f"- {h.title}"
        + (f", section {h.section}" if h.section else "")
        + f" (p.{h.page_start})"
        for h in candidates
    )
    return (
        "A user's question is ambiguous: the sources contain several plausible but "
        "different answers, and answering from one of them would be a guess.\n\n"
        f"QUESTION: {question}\n\n"
        f"COMPETING SOURCES:\n{listing}\n\n"
        "Write one short clarifying question that names the specific alternatives, so "
        "the user can pick. Do not ask 'could you clarify' or 'can you be more "
        "specific' -- name the actual options. List them in `options` too.\n\n"
        "Respond with JSON matching the required schema."
    )


# ---------------------------------------------------------------------------
#  Sufficiency judging, decomposition, rewriting
# ---------------------------------------------------------------------------
JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "missing": {"type": "string"},
        "is_multi_part": {"type": "boolean"},
    },
    "required": ["sufficient", "missing", "is_multi_part"],
}


def judge_prompt(question: str, hits: Sequence[Hit]) -> str:
    """Ask whether the retrieved passages can actually answer the question.

    Runs on the VOLUME ladder. A judgement is a cheap call and there are two to four
    of them per turn; routing them through synthesis models would exhaust reasoning
    quota in about four turns and the failure would land on answer generation.
    """
    listing = "\n".join(
        f"[{h.chunk_id}] ({h.source_label})\n{h.text.strip()[:900]}"
        for h in hits
    )
    return (
        "Decide whether these passages contain enough to answer the question. Do not "
        "answer it.\n\n"
        "Judge only what is present. You have background knowledge about these "
        "papers; ignore it entirely.\n\n"
        "TWO WAYS THE PASSAGES CAN LOOK RIGHT AND BE INSUFFICIENT. Check both:\n"
        "  (a) RIGHT TOPIC, WRONG PAPER. The passages discuss what was asked about, "
        "but come from a different paper than the question names. Insufficient.\n"
        "  (b) RIGHT PAPER, WRONG TOPIC. The passages come from the paper the "
        "question names, but do not contain the specific thing asked about. This is "
        "the easier one to miss: the passages look authoritative and on-brand for the "
        "question, and they simply do not answer it. Insufficient.\n\n"
        "Case (b) often means the question's premise is wrong -- it attributes "
        "something to a paper that does not contain it. Say so in `missing`, naming "
        "the thing that is absent, so the next search can look elsewhere.\n\n"
        "Set `is_multi_part` true if the question asks about two or more distinct "
        "things that would need separate evidence -- for example comparing two "
        "papers, or asking how one method extends another.\n\n"
        "EXAMPLE of (b): the question asks which section of the Alpha paper describes "
        "its 4-bit quantisation, and every passage is from the Alpha paper but about "
        "low-rank decomposition. Verdict: insufficient. Missing: \"the Alpha passages "
        "contain no quantisation scheme; the premise may be misattributed\".\n\n"
        f"QUESTION: {question}\n\n"
        f"--- PASSAGES ---\n{listing}\n--- END PASSAGES ---\n\n"
        "If insufficient, say concisely in `missing` what evidence is absent.\n\n"
        "Respond with JSON matching the required schema."
    )


DECOMPOSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "sub_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sub_questions"],
}


def decompose_prompt(question: str, max_parts: int, vocabulary: str) -> str:
    """Split a multi-part question into independently retrievable sub-questions.

    The same vocabulary constraint as condensation applies, and for the same reason:
    a sub-question containing an invented term retrieves the wrong documents, and the
    error is invisible because the sub-answer still looks plausible.
    """
    return (
        f"Split this question into at most {max_parts} sub-questions, each of which "
        f"could be looked up on its own.\n\n"
        f"Each sub-question must name its subject explicitly -- it will be retrieved "
        f"in isolation, with no access to the original question. 'How does it "
        f"compare?' is useless as a sub-question; 'What memory does QLoRA use?' is "
        f"not.\n\n"
        f"Use only words from the question itself or from this list of document "
        f"titles:\n{vocabulary}\n\n"
        f"Do not introduce a technical term that appears in neither. If the question "
        f"really only asks one thing, return it unchanged as a single sub-question.\n\n"
        f"QUESTION: {question}\n\n"
        f"Respond with JSON matching the required schema."
    )


REWRITE_SCHEMA: dict = {
    "type": "object",
    "properties": {"rewritten": {"type": "string"}},
    "required": ["rewritten"],
}


def rewrite_prompt(question: str, missing: str, vocabulary: str) -> str:
    return (
        "A search for this question did not retrieve enough evidence. Rewrite it to "
        "retrieve better.\n\n"
        f"QUESTION: {question}\n"
        f"WHAT WAS MISSING: {missing}\n\n"
        f"Use only words from the question or from these document titles:\n"
        f"{vocabulary}\n\n"
        "Prefer the concrete terms a paper would actually use over general ones. Do "
        "not add a term that appears in neither list.\n\n"
        "Respond with JSON matching the required schema."
    )
