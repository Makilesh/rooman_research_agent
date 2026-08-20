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

3. If the passages do not contain the answer, set insufficient_evidence to true, give
   a refusal_reason naming what is missing, and return an empty sentences list. This
   is a correct and valued outcome, not a failure. Refusing when the sources are
   silent is more useful than a plausible guess.

4. If the question assumes something the passages contradict, do NOT play along and do
   NOT simply refuse. Say plainly that the premise is wrong, then cite what the
   passages actually say. Set insufficient_evidence to false for this case.

5. Be specific. Prefer the exact figure, name, or setting from the passage over a
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
