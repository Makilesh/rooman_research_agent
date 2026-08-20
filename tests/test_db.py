"""Schema, migrations, and the constraints that make a fabricated citation impossible."""

from __future__ import annotations

import sqlite3

import pytest

from research_agent import db
from research_agent.config import Config

# Every table the data model specifies. If one goes missing, the failure should name
# it rather than surfacing three steps later as a mysterious OperationalError.
EXPECTED_TABLES = {
    "documents", "chunks", "chunks_fts", "chunk_order", "corpus_state",
    "sessions", "turns", "turn_citations", "turn_retrievals", "answer_cache",
    "llm_calls", "eval_runs", "eval_results",
}


def test_every_specified_table_exists(conn):
    present = set(db.table_names(conn))
    assert EXPECTED_TABLES <= present, f"missing: {sorted(EXPECTED_TABLES - present)}"


def test_migrations_are_idempotent(cfg: Config):
    c = db.connect(cfg)
    first = db.migrate(c)
    second = db.migrate(c)
    assert first, "the first migration should create something"
    assert second == [], "a second run must be a genuine no-op, not a silent rebuild"
    assert EXPECTED_TABLES <= set(db.table_names(c))


def test_wal_and_foreign_keys_are_on(conn):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def _seed_doc_and_chunk(conn, chunk_id="c_0001", text="Low-rank adaptation freezes the base weights."):
    db.insert(conn, "documents", {
        "doc_id": "d_lora", "title": "LoRA", "arxiv_id": "2106.09685",
        "path": "data/sources/2106.09685.pdf", "sha256": "deadbeef", "n_pages": 26,
    })
    db.insert(conn, "chunks", {
        "chunk_id": chunk_id, "doc_id": "d_lora", "parent_id": None, "level": 0,
        "ord": 1, "page_start": 4, "page_end": 4, "section": "3 Method",
        "token_count": 120, "text": text,
    })
    conn.commit()


def test_fts_index_is_populated_by_trigger(conn):
    _seed_doc_and_chunk(conn)
    hits = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'adaptation'"
    ).fetchall()
    assert len(hits) == 1, "the AFTER INSERT trigger must mirror chunks into chunks_fts"


def test_fts_porter_stemming_is_active(conn):
    _seed_doc_and_chunk(conn, text="The adapter freezes pretrained weights")
    # 'freeze' only matches 'freezes' if the porter tokenizer is in effect.
    assert conn.execute(
        "SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'freeze'"
    ).fetchone() is not None


def test_fts_stays_in_sync_on_update_and_delete(conn):
    _seed_doc_and_chunk(conn)
    conn.execute("UPDATE chunks SET text = 'quantised nf4 storage' WHERE chunk_id = 'c_0001'")
    conn.commit()
    assert conn.execute("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'adaptation'").fetchone() is None
    assert conn.execute("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'nf4'").fetchone() is not None

    conn.execute("DELETE FROM chunks WHERE chunk_id = 'c_0001'")
    conn.commit()
    assert conn.execute("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'nf4'").fetchone() is None


def test_a_citation_cannot_point_at_a_nonexistent_chunk(conn):
    """The database refuses a fabricated chunk id.

    This is the last line of defence behind `answer.py`'s validation: even if an
    invented id somehow reached persistence, it does not get stored.
    """
    _seed_doc_and_chunk(conn)
    db.insert(conn, "sessions", {"session_id": "s1", "corpus_fingerprint": "fp1"})
    db.insert(conn, "turns", {
        "turn_id": "t1", "session_id": "s1", "ord": 1, "role": "agent",
        "raw_text": "q", "route": "answer",
    })
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        db.insert(conn, "turn_citations", {
            "turn_id": "t1", "sentence_idx": 0, "sentence_text": "Invented.",
            "chunk_id": "c_9999_does_not_exist", "verify_score": 0.9, "status": "verified",
        })
        conn.commit()


def test_route_and_status_values_are_constrained(conn):
    db.insert(conn, "sessions", {"session_id": "s1", "corpus_fingerprint": "fp1"})
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.insert(conn, "turns", {
            "turn_id": "t_bad", "session_id": "s1", "ord": 1, "role": "agent",
            "raw_text": "q", "route": "maybe",  # not one of the four routes
        })
        conn.commit()


def test_corpus_state_roundtrip_and_overwrite(conn):
    db.set_state(conn, "corpus_fingerprint", "fp_aaa")
    assert db.get_state(conn, "corpus_fingerprint") == "fp_aaa"
    db.set_state(conn, "corpus_fingerprint", "fp_bbb")
    assert db.get_state(conn, "corpus_fingerprint") == "fp_bbb"
    assert db.get_state(conn, "never_set") is None


def test_reset_refuses_without_confirmation(cfg: Config):
    c = db.connect(cfg)
    db.migrate(c)
    assert cfg.db_path.exists()
    with pytest.raises(RuntimeError, match="confirm=True"):
        db.reset(cfg)
    assert cfg.db_path.exists(), "a refused reset must not have deleted anything"
    c.close()
    db.reset(cfg, confirm=True)
    assert not cfg.db_path.exists()


def test_reset_on_an_open_database_explains_itself(cfg: Config):
    """Windows will not unlink an open file, and WinError 32 tells the user nothing."""
    c = db.connect(cfg)
    db.migrate(c)
    try:
        with pytest.raises(RuntimeError, match="still open"):
            db.reset(cfg, confirm=True)
    finally:
        c.close()


def test_iso_cutoff_matches_the_ledger_timestamp_format(conn):
    """A same-day mixed-format comparison silently matches rows it should exclude.

    `llm_calls.ts` is ISO-8601 with a 'T' separator; `datetime('now')` uses a space.
    They are compared as strings, and at index 10 'T' (84) sorts after ' ' (32) -- so
    for any row from TODAY, `ts >= datetime('now','-5 minutes')` is true regardless of
    the actual time. The year saves you across days, which is exactly why the bug
    hides: it only appears while diagnosing a run in progress, which is when it did.
    """
    from datetime import datetime, timedelta, timezone

    from research_agent import db as _db

    # Today, but two hours ago -- outside a five-minute window by any honest reading.
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    conn.execute(
        "INSERT INTO llm_calls (ts, provider, model, key_alias, purpose, prompt_sha) "
        "VALUES (?, 'gemini', 'm', 'k', 'p', 's')", (stale,))
    conn.commit()

    correct = _db.scalar(
        conn, "SELECT COUNT(*) FROM llm_calls WHERE ts >= ?",
        (_db.iso_cutoff(minutes=5),))
    assert correct == 0, "a row from two hours ago is not within five minutes"

    naive = _db.scalar(
        conn, "SELECT COUNT(*) FROM llm_calls WHERE ts >= datetime('now','-5 minutes')")
    assert naive == 1, (
        "the trap: the naive comparison wrongly includes a two-hour-old row, because "
        "'T' sorts after the space that datetime() uses"
    )
