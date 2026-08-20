"""Gold-label loading, validation, and staleness detection.

Fabricated ground truth is the worst failure available in this project: it
invalidates every number downstream while looking entirely healthy. So a label is
never trusted on the strength of having been written down. Three things are checked
on every evaluation run:

1. the chunk id still exists;
2. the chunk's text still hashes to what it hashed to when the label was made;
3. the corpus fingerprint at labelling time matches the current index.

Any of those failing is a hard error, not a warning. A label that silently drifts is
worse than no label, because the resulting numbers still look plausible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import Any

import yaml

from . import db
from .config import Config

ANSWERABLE_CLASSES = {"single_hop", "multi_hop"}
ABSTAIN_CLASSES = {"unanswerable"}


@dataclass(frozen=True, slots=True)
class GoldItem:
    id: str
    question: str
    cls: str
    must_abstain: bool
    expected_route: str
    gold_chunks: list[str]
    expected_facts: list[str]
    text_shas: dict[str, str] = field(default_factory=dict)
    label_quote: str | None = None
    requires_documents: list[str] = field(default_factory=list)
    why_unanswerable: str | None = None
    false_premise_is: str | None = None


@dataclass(frozen=True, slots=True)
class LabelSet:
    items: list[GoldItem]
    fingerprint_at_labelling: str | None
    labelled_on: str | None

    @property
    def answerable(self) -> list[GoldItem]:
        return [i for i in self.items if i.gold_chunks]

    @property
    def controls(self) -> list[GoldItem]:
        return [i for i in self.items if i.must_abstain]


def chunk_text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load(cfg: Config) -> LabelSet:
    raw = yaml.safe_load(cfg.questions_path.read_text(encoding="utf-8"))
    items = [
        GoldItem(
            id=q["id"], question=q["question"], cls=q["class"],
            must_abstain=bool(q.get("must_abstain", False)),
            expected_route=q["expected_route"],
            gold_chunks=list(q.get("gold_chunks") or []),
            expected_facts=list(q.get("expected_facts") or []),
            text_shas=dict(q.get("text_shas") or {}),
            label_quote=q.get("label_quote"),
            requires_documents=list(q.get("requires_documents") or []),
            why_unanswerable=q.get("why_unanswerable"),
            false_premise_is=q.get("false_premise_is"),
        )
        for q in raw["questions"]
    ]
    return LabelSet(items, raw.get("corpus_fingerprint_at_labelling"),
                    raw.get("labelled_on"))


def fetch_chunks(conn: Connection, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}
    rows = db.all_rows(conn, f"""
        SELECT c.chunk_id, c.doc_id, d.title, c.page_start, c.page_end, c.section, c.text
        FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.chunk_id IN ({','.join('?' * len(chunk_ids))})
    """, chunk_ids)
    return {r["chunk_id"]: dict(r) for r in rows}


def validate(cfg: Config, conn: Connection, labels: LabelSet,
             current_fingerprint: str | None) -> list[str]:
    """Return a list of problems. Empty means every label is sound."""
    problems: list[str] = []

    if (labels.fingerprint_at_labelling and current_fingerprint
            and labels.fingerprint_at_labelling != current_fingerprint):
        problems.append(
            f"labels were made against corpus {labels.fingerprint_at_labelling} but the "
            f"index is now {current_fingerprint}. Re-validate before trusting any number."
        )

    all_ids = [cid for item in labels.items for cid in item.gold_chunks]
    chunks = fetch_chunks(conn, all_ids)

    for item in labels.items:
        # Class and behaviour must agree, or the confusion matrix means nothing.
        if item.cls in ABSTAIN_CLASSES and not item.must_abstain:
            problems.append(f"{item.id}: class {item.cls} must set must_abstain: true")
        if item.must_abstain and item.gold_chunks:
            problems.append(
                f"{item.id}: must_abstain is true but gold_chunks is non-empty. "
                f"A control question cannot have a correct citation."
            )
        if item.cls in ANSWERABLE_CLASSES and not item.gold_chunks:
            problems.append(f"{item.id}: answerable class with no gold_chunks")
        if item.cls == "multi_hop" and len({chunks[c]["doc_id"] for c in item.gold_chunks
                                            if c in chunks}) < 2:
            problems.append(
                f"{item.id}: labelled multi_hop but its gold chunks come from one paper. "
                f"It is a single-hop question wearing a multi-hop label."
            )

        for cid in item.gold_chunks:
            row = chunks.get(cid)
            if row is None:
                problems.append(f"{item.id}: gold chunk {cid} does not exist")
                continue
            recorded = item.text_shas.get(cid)
            actual = chunk_text_sha(row["text"])
            if recorded and recorded != actual:
                problems.append(
                    f"{item.id}: {cid} still resolves but its text has changed "
                    f"({recorded} -> {actual}). The label now points at a different "
                    f"passage than the one it was checked against."
                )
            elif not recorded:
                problems.append(f"{item.id}: {cid} has no recorded text_sha — run "
                                f"`labels --stamp`")

        for doc in item.requires_documents:
            if not any(chunks.get(c, {}).get("doc_id") == doc for c in item.gold_chunks):
                problems.append(
                    f"{item.id}: requires_documents lists {doc} but no gold chunk is from it"
                )
    return problems


def stamp(cfg: Config, conn: Connection) -> int:
    """Record the current text_sha for every gold chunk. Run after labelling.

    Edits the file line by line rather than round-tripping through the YAML dumper.
    A dump-and-rewrite would silently strip every comment, and in this file the
    comments carry the label quotes and the reasoning a reviewer checks the labels
    against -- which is most of their value.
    """
    raw = yaml.safe_load(cfg.questions_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: (q.get("gold_chunks") or []) for q in raw["questions"]}
    chunks = fetch_chunks(conn, [c for ids in by_id.values() for c in ids])

    lines = cfg.questions_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    current: str | None = None
    stamped = 0

    for line in lines:
        stripped = line.strip()
        # Drop any previous stamp so re-running is idempotent rather than additive.
        if stripped.startswith("text_shas:"):
            continue
        if stripped.startswith("- id:"):
            current = stripped.split(":", 1)[1].strip()
        out.append(line)

        if current and stripped.startswith("gold_chunks:"):
            ids = by_id.get(current, [])
            if not ids:
                continue
            indent = " " * (len(line) - len(line.lstrip()))
            pairs = []
            for cid in ids:
                if cid not in chunks:
                    raise RuntimeError(
                        f"{current}: gold chunk {cid} does not exist; cannot stamp"
                    )
                pairs.append(f"{cid}: {chunk_text_sha(chunks[cid]['text'])}")
                stamped += 1
            out.append(f"{indent}text_shas: {{{', '.join(pairs)}}}")

    cfg.questions_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return stamped
