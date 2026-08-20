"""SQLite connection, migrations, and typed row helpers.

Every table in the data model is created here by an idempotent migration. Nothing
creates a table by hand; nothing writes DDL outside this file.

WAL is on because the CLI reads while it writes and WAL is the only mode where that
is not a lock fight. Foreign keys are on because a citation pointing at a chunk that
does not exist is precisely the failure this project is built to prevent, and the
database should say so rather than storing it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Config

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
#  DDL. Split into one statement per element so a partial failure is legible.
# ---------------------------------------------------------------------------
_MIGRATIONS: tuple[tuple[int, str, str], ...] = (
    # ---- corpus ----------------------------------------------------------
    (1, "documents", """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id          TEXT PRIMARY KEY,
            title           TEXT NOT NULL,
            arxiv_id        TEXT,
            path            TEXT NOT NULL,
            source_url      TEXT,
            sha256          TEXT NOT NULL,
            n_pages         INTEGER,
            extracted_chars INTEGER,
            ingested_at     TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),
    (1, "chunks", """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT PRIMARY KEY,
            doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            parent_id   TEXT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
            level       INTEGER NOT NULL,        -- 0 = child, 1 = parent
            ord         INTEGER NOT NULL,
            page_start  INTEGER NOT NULL,
            page_end    INTEGER NOT NULL,
            section     TEXT,
            token_count INTEGER NOT NULL,
            text        TEXT NOT NULL
        )
    """),
    # Extracted page text, kept so `ingest` and `index` are separate commands rather
    # than one pass that re-parses every PDF whenever the chunk config changes. It is
    # also what the gold-label validation tool reads, so a label is always checked
    # against the text the pipeline actually saw -- not against the PDF, and never
    # against anyone's memory of the paper.
    (1, "pages", """
        CREATE TABLE IF NOT EXISTS pages (
            doc_id    TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            page_no   INTEGER NOT NULL,
            n_columns INTEGER NOT NULL,
            n_blocks  INTEGER NOT NULL,
            text      TEXT NOT NULL,
            PRIMARY KEY (doc_id, page_no)
        )
    """),
    (1, "idx_chunks_doc", "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, level, ord)"),
    (1, "idx_chunks_parent", "CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id)"),

    # External-content FTS5: the index stores no copy of the text, it points back
    # into `chunks`. Keeps one source of truth and halves the on-disk size.
    # `porter unicode61` gives real stemming; its cost is that unicode61 splits
    # identifiers like GPT-3 into `gpt` + `3` (decisions.md D-103).
    (1, "chunks_fts", """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='rowid',
            tokenize='porter unicode61'
        )
    """),
    (1, "chunks_fts_ai", """
        CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
        END
    """),
    (1, "chunks_fts_ad", """
        CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.rowid, old.text);
        END
    """),
    (1, "chunks_fts_au", """
        CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.rowid, old.text);
            INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
        END
    """),
    (1, "chunk_order", """
        CREATE TABLE IF NOT EXISTS chunk_order (
            row_idx  INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE REFERENCES chunks(chunk_id) ON DELETE CASCADE
        )
    """),
    (1, "corpus_state", """
        CREATE TABLE IF NOT EXISTS corpus_state (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        )
    """),

    # ---- conversation ----------------------------------------------------
    (1, "sessions", """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id         TEXT PRIMARY KEY,
            created_at         TEXT NOT NULL DEFAULT (datetime('now')),
            corpus_fingerprint TEXT,
            meta_json          TEXT
        )
    """),
    (1, "turns", """
        CREATE TABLE IF NOT EXISTS turns (
            turn_id           TEXT PRIMARY KEY,
            session_id        TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            ord               INTEGER NOT NULL,
            role              TEXT NOT NULL CHECK (role IN ('user', 'agent')),
            raw_text          TEXT NOT NULL,
            condensed_query   TEXT,
            route             TEXT CHECK (route IN ('answer','clarify','refuse','abstain')),
            top_rerank_score  REAL,
            n_retrieval_loops INTEGER NOT NULL DEFAULT 0,
            latency_ms        INTEGER,
            provider          TEXT,
            model             TEXT,
            created_at        TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),
    (1, "idx_turns_session", "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, ord)"),
    (1, "turn_citations", """
        CREATE TABLE IF NOT EXISTS turn_citations (
            turn_id       TEXT NOT NULL REFERENCES turns(turn_id) ON DELETE CASCADE,
            sentence_idx  INTEGER NOT NULL,
            sentence_text TEXT NOT NULL,
            chunk_id      TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE RESTRICT,
            verify_score  REAL,
            status        TEXT CHECK (status IN ('verified','repaired','unverified')),
            PRIMARY KEY (turn_id, sentence_idx, chunk_id)
        )
    """),
    (1, "idx_citations_chunk", "CREATE INDEX IF NOT EXISTS idx_citations_chunk ON turn_citations(chunk_id)"),
    (1, "turn_retrievals", """
        CREATE TABLE IF NOT EXISTS turn_retrievals (
            turn_id        TEXT NOT NULL REFERENCES turns(turn_id) ON DELETE CASCADE,
            chunk_id       TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            rank           INTEGER NOT NULL,
            dense_score    REAL,
            sparse_score   REAL,
            rrf_score      REAL,
            rerank_score   REAL,
            used_in_context INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (turn_id, chunk_id)
        )
    """),
    (1, "answer_cache", """
        CREATE TABLE IF NOT EXISTS answer_cache (
            cache_id           TEXT PRIMARY KEY,
            corpus_fingerprint TEXT NOT NULL,
            condensed_query    TEXT NOT NULL,
            embedding          BLOB NOT NULL,
            answer_json        TEXT NOT NULL,
            created_at         TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """),
    (1, "idx_answer_cache_fp", "CREATE INDEX IF NOT EXISTS idx_answer_cache_fp ON answer_cache(corpus_fingerprint)"),

    # ---- ops / quota -----------------------------------------------------
    # The load-bearing table. RPM and RPD are derived from it by windowed counts,
    # which is the only way a daily limit survives a process that exits between
    # every request (decisions.md D-002).
    (1, "llm_calls", """
        CREATE TABLE IF NOT EXISTS llm_calls (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ts                TEXT NOT NULL,
            provider          TEXT NOT NULL,
            model             TEXT NOT NULL,
            key_alias         TEXT NOT NULL,
            ladder            TEXT,
            purpose           TEXT NOT NULL,
            prompt_sha        TEXT NOT NULL,
            cached            INTEGER NOT NULL DEFAULT 0,
            ok                INTEGER,
            error             TEXT,
            latency_ms        INTEGER,
            prompt_tokens     INTEGER,
            completion_tokens INTEGER
        )
    """),
    (1, "idx_llm_calls_window",
     "CREATE INDEX IF NOT EXISTS idx_llm_calls_window ON llm_calls(model, key_alias, ts)"),

    # ---- evaluation ------------------------------------------------------
    (1, "eval_runs", """
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id      TEXT PRIMARY KEY,
            started_at  TEXT NOT NULL DEFAULT (datetime('now')),
            config_name TEXT NOT NULL,
            provider    TEXT,
            notes       TEXT
        )
    """),
    (1, "eval_results", """
        CREATE TABLE IF NOT EXISTS eval_results (
            run_id      TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
            item_id     TEXT NOT NULL,
            metric      TEXT NOT NULL,
            value       REAL,
            detail_json TEXT
        )
    """),
    (1, "idx_eval_results", "CREATE INDEX IF NOT EXISTS idx_eval_results ON eval_results(run_id, metric)"),
)


# ---------------------------------------------------------------------------
def connect(cfg: Config, path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection with the pragmas this project depends on."""
    target = Path(path) if path is not None else cfg.db_path
    if str(target) != ":memory:":
        target.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target), timeout=cfg.sqlite_busy_timeout_ms / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={cfg.sqlite_busy_timeout_ms}")
    # Durable enough for a single-user CLI, and materially faster than FULL.
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Create every missing table, index, trigger and virtual table.

    Idempotent by construction: every statement is `IF NOT EXISTS`, and the applied
    set is recorded so a second run is a genuine no-op rather than a silent rebuild.
    Returns the names applied on this call.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name       TEXT PRIMARY KEY,
            version    INTEGER NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    already = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations")}

    applied: list[str] = []
    for version, name, ddl in _MIGRATIONS:
        if name in already:
            continue
        conn.execute(ddl)
        conn.execute(
            "INSERT INTO schema_migrations(name, version) VALUES (?, ?)", (name, version)
        )
        applied.append(name)
    conn.commit()
    return applied


def table_names(conn: sqlite3.Connection) -> list[str]:
    """Every table and virtual table, for `doctor` and for the schema gate."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    return [r["name"] for r in rows]


def reset(cfg: Config, confirm: bool = False) -> None:
    """Delete the database file. Development only; refuses without an explicit flag."""
    if not confirm:
        raise RuntimeError(
            "reset() destroys the database, including the quota ledger. Pass "
            "confirm=True (CLI: --confirm) if that is what you intend."
        )
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(cfg.db_path) + suffix)
        if not p.exists():
            continue
        try:
            p.unlink()
        except PermissionError as exc:
            # Windows refuses to unlink a file that any process still has open, and
            # a raw WinError 32 tells the caller nothing useful.
            raise RuntimeError(
                f"Cannot delete {p}: a connection to it is still open. Close every "
                f"connection (or exit any other running `research-agent` process) "
                f"before resetting."
            ) from exc


# ---------------------------------------------------------------------------
#  Row helpers. Thin on purpose -- an ORM would hide the SQL that reviewers
#  most want to read, which is the same reason there is no LangGraph here.
# ---------------------------------------------------------------------------
def one(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
    return conn.execute(sql, params).fetchone()


def all_rows(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def scalar(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return None if row is None else row[0]


def insert(conn: sqlite3.Connection, table: str, values: dict[str, Any]) -> None:
    cols = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(values.values()))


def insert_many(conn: sqlite3.Connection, table: str, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0])
    marks = ", ".join("?" for _ in cols)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({marks})",
        [tuple(r[c] for c in cols) for r in rows],
    )
    return len(rows)


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    return scalar(conn, "SELECT v FROM corpus_state WHERE k = ?", (key,))


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO corpus_state(k, v) VALUES (?, ?) "
        "ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        (key, value),
    )
    conn.commit()
