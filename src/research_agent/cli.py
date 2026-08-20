"""Command-line entry point.

Step 1 ships `doctor` and `budget`. Later steps add `fetch`, `ingest`, `index`,
`ask`, `chat`, `eval` and `history`; they are not stubbed here, because a command
that exists and does nothing is worse than one that does not exist yet.
"""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import replace

import typer
from rich.console import Console
from rich.table import Table

from . import agent
from . import answer as answer_mod
from . import conversation
from . import router as router_mod
from . import corpus, db, evaluate, report
from . import verify as verify_mod
from . import ingest as ingest_mod
from . import labels as labels_mod
from . import index as index_mod
from . import rerank as rerank_mod
from . import retrieve
from .config import Config
from .llm import LLMClient, OllamaProvider, QuotaExhausted

# Windows consoles default to cp1252, which cannot encode box-drawing characters,
# em dashes, or the accented author names and mathematical symbols that arXiv text
# is full of. Piping output then dies with a UnicodeEncodeError several layers deep
# in the renderer. Reconfiguring here fixes every command at once.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False, help="Cited research agent.")
console = Console()


def _ok(label: str, detail: str = "") -> None:
    console.print(f"  [green]OK[/green]   {label}" + (f"  [dim]{detail}[/dim]" if detail else ""))


def _warn(label: str, detail: str = "") -> None:
    console.print(f"  [yellow]WARN[/yellow] {label}" + (f"  [dim]{detail}[/dim]" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    console.print(f"  [red]FAIL[/red] {label}" + (f"  [dim]{detail}[/dim]" if detail else ""))


@app.command()
def doctor(
    gpu: bool = typer.Option(False, "--gpu", help="Also load torch and verify CUDA."),
) -> None:
    """Probe the environment: SQLite, FTS5, Ollama, Gemini keys, ladders, GPU."""
    cfg = Config.load()
    failures = 0

    console.rule("[bold]Configuration")
    console.print(f"  provider      {cfg.provider}")
    console.print(f"  offload_mode  {cfg.offload_mode}")
    console.print(f"  embed_device  {cfg.embed_device}")
    console.print(f"  db            {cfg.db_path}")

    # -- SQLite ------------------------------------------------------------
    console.rule("[bold]SQLite")
    import sqlite3

    conn = db.connect(cfg)
    applied = db.migrate(conn)
    _ok(f"sqlite {sqlite3.sqlite_version}", f"journal_mode=WAL, foreign_keys=ON")
    _ok(f"migrations: {len(applied)} applied this run", "idempotent; 0 means already current")

    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS _fts_probe USING fts5(x, tokenize='porter unicode61')"
        )
        conn.execute("INSERT INTO _fts_probe VALUES ('running quickly')")
        stemmed = bool(conn.execute("SELECT 1 FROM _fts_probe WHERE _fts_probe MATCH 'run'").fetchone())
        conn.execute("DROP TABLE _fts_probe")
        conn.commit()
        if stemmed:
            _ok("FTS5 with porter stemming", "sparse retrieval is available")
        else:
            _fail("FTS5 present but porter stemming did not match")
            failures += 1
    except sqlite3.OperationalError as exc:
        _fail("FTS5 unavailable", str(exc))
        console.print("       Sparse retrieval cannot be built without it.")
        failures += 1

    tables = db.table_names(conn)
    _ok(f"{len(tables)} tables", ", ".join(tables))

    # -- Ollama ------------------------------------------------------------
    console.rule("[bold]Ollama")
    provider = OllamaProvider(cfg.ollama_host)
    try:
        models = provider.tags(timeout_s=5.0)
        names = [m.get("name", "?") for m in models]
        _ok(f"reachable at {cfg.ollama_host}", f"{len(names)} models")
        for m in models:
            size_gb = m.get("size", 0) / 1e9
            marker = ""
            if m.get("name") == cfg.ollama_model:
                marker = "  <- OLLAMA_MODEL (coresident)"
            elif m.get("name") == cfg.ollama_model_large:
                marker = "  <- OLLAMA_MODEL_LARGE (sequential)"
            console.print(f"       {m.get('name','?'):<28} {size_gb:>6.2f} GB{marker}")
        if cfg.ollama_model not in names:
            _warn(f"{cfg.ollama_model} not pulled", f"run: ollama pull {cfg.ollama_model}")
    except Exception as exc:
        _warn(f"not reachable at {cfg.ollama_host}", f"{type(exc).__name__}: {exc}")
        console.print("       The agent needs either Ollama or a Gemini key to generate.")

    # -- Gemini ------------------------------------------------------------
    console.rule("[bold]Gemini")
    keys = LLMClient.load_keys(cfg, dict(os.environ))
    if keys:
        _ok(f"{len(keys)} key(s) configured", ", ".join(keys))
        console.print(
            "       [dim]Free-tier quota is per project. Keys from one project share"
            " one bucket;\n       rotation only multiplies capacity across separate"
            " projects.[/dim]"
        )
    else:
        _ok("no keys configured", "expected — the Ollama path requires none")

    # -- Ladders -----------------------------------------------------------
    console.rule("[bold]Ladders")
    for name in ("synthesis", "volume"):
        t = Table(title=f"{name} ladder", title_justify="left", show_edge=False)
        t.add_column("#", justify="right")
        t.add_column("model")
        t.add_column("RPM", justify="right")
        t.add_column("RPD", justify="right")
        t.add_column("note")
        for i, rung in enumerate(cfg.ladder(name), 1):  # type: ignore[arg-type]
            note = ""
            if rung.model == "gemini-2.5-flash-lite":
                note = "named like a volume model, capped at 20 RPD"
            t.add_row(str(i), rung.model, str(rung.rpm), str(rung.rpd), note)
        console.print(t)
    _ok("ladder placement validated", "Config.validate() enforces the 2.5-flash-lite rule")

    # -- GPU ---------------------------------------------------------------
    if gpu:
        console.rule("[bold]GPU")
        try:
            import torch
        except ImportError as exc:
            _fail("torch not importable", str(exc))
            raise typer.Exit(1)

        console.print(f"  torch              {torch.__version__}")
        console.print(f"  torch.version.cuda {torch.version.cuda}")
        available = torch.cuda.is_available()
        console.print(f"  is_available()     {available}")
        if not available:
            _fail("CUDA is not available")
            console.print(
                "\n  [red]`pip install torch` resolves to a CPU-only wheel and reports"
                " success.[/red]\n"
                "  Every encoder then runs roughly 5x slower with no error anywhere.\n"
                "  Fix:\n"
                "    pip install --force-reinstall --no-deps torch==2.12.1+cu130 \\\n"
                "        --index-url https://download.pytorch.org/whl/cu130\n"
            )
            if cfg.require_cuda:
                raise typer.Exit(1)
        else:
            props = torch.cuda.get_device_properties(0)
            cap = torch.cuda.get_device_capability(0)
            console.print(f"  device             {torch.cuda.get_device_name(0)}")
            console.print(f"  capability         sm_{cap[0]}{cap[1]}")
            console.print(f"  total VRAM         {props.total_memory / 2**30:.2f} GiB")
            console.print(f"  built arch list    {', '.join(torch.cuda.get_arch_list())}")
            if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
                _fail(
                    f"this torch has no kernels for sm_{cap[0]}{cap[1]}",
                    "the wheel loads but the GPU is unusable",
                )
                failures += 1
            else:
                # A real kernel launch. A device query alone does not prove the
                # build actually carries kernels for this architecture.
                dim = cfg.gpu_probe_dim
                a = torch.randn(dim, dim, device="cuda", dtype=torch.float16)
                (a @ a).float().sum().item()
                torch.cuda.synchronize()
                _ok(
                    "fp16 matmul executed on device",
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                )

    console.rule()
    if failures:
        console.print(f"[red]{failures} check(s) failed.[/red]")
        raise typer.Exit(1)
    console.print("[green]All checks passed.[/green]")


@app.command()
def budget() -> None:
    """Remaining RPM/RPD per model per key, derived from the durable ledger."""
    cfg = Config.load()
    conn = db.connect(cfg)
    db.migrate(conn)
    client = LLMClient(cfg=cfg, conn=conn, api_keys=LLMClient.load_keys(cfg, dict(os.environ)))

    total = db.scalar(conn, "SELECT COUNT(*) FROM llm_calls") or 0
    billed = db.scalar(conn, "SELECT COUNT(*) FROM llm_calls WHERE cached = 0") or 0
    cached = total - billed
    console.print(
        f"Ledger: {total} call(s) recorded, {billed} counted against quota, "
        f"{cached} served from cache."
    )

    t = Table(show_edge=False)
    for col in ("ladder", "model", "key", "RPM", "RPD"):
        t.add_column(col, justify="right" if col in ("RPM", "RPD") else "left")
    for row in client.budget():
        rpm = f"{row['rpm_left']}/{row['rpm_limit']}"
        rpd = f"{row['rpd_left']}/{row['rpd_limit']}"
        t.add_row(row["ladder"], row["model"], row["key"], rpm, rpd)
    console.print(t)

    if not client.api_keys:
        console.print(
            "\n[dim]No Gemini key configured, so the figures above are limits rather"
            " than live balances.\nThe Ollama path is unlimited and needs no key.[/dim]"
        )


@app.command()
def fetch(
    audit: bool = typer.Option(
        False, "--audit", help="Re-check manifest titles and authors against the arXiv API."
    ),
) -> None:
    """Download every manifest paper from arXiv, politely and reproducibly."""
    cfg = Config.load()
    entries = corpus.load_manifest(cfg)
    console.print(f"Manifest: {len(entries)} documents -> {cfg.sources_dir}\n")

    if audit:
        console.rule("[bold]Manifest audit against the live arXiv API")
        live = {m["arxiv_id"]: m for m in corpus.fetch_metadata(
            [e.arxiv_id for e in entries], cfg.arxiv_user_agent, cfg.fetch_timeout_s
        )}
        mismatches = 0
        for e in entries:
            m = live.get(e.arxiv_id)
            if m is None:
                _fail(e.arxiv_id, "not returned by the API")
                mismatches += 1
            elif m["title"] != e.title or m["first_author"] != e.first_author:
                _fail(e.arxiv_id, f"manifest says {e.title!r} / {e.first_author!r}; "
                                  f"API says {m['title']!r} / {m['first_author']!r}")
                mismatches += 1
            else:
                _ok(e.arxiv_id, f"{e.first_author} — {e.title[:52]}")
        console.print()
        if mismatches:
            raise typer.Exit(1)

    results = list(corpus.fetch_corpus(cfg, entries, on_event=console.print))

    console.print()
    t = Table(show_edge=False)
    for col, just in (("doc_id", "left"), ("arxiv id", "left"), ("bytes", "right"),
                      ("sha256 (first 16)", "left"), ("state", "left")):
        t.add_column(col, justify=just)  # type: ignore[arg-type]
    for r in results:
        t.add_row(r.entry.doc_id, r.entry.arxiv_id, f"{r.n_bytes:,}",
                  r.sha256[:16], "cached" if r.skipped else "downloaded")
    console.print(t)

    empty = [r for r in results if r.n_bytes == 0]
    if empty:
        _fail(f"{len(empty)} zero-byte file(s)", ", ".join(r.entry.arxiv_id for r in empty))
        raise typer.Exit(1)
    _ok(f"{len(results)} document(s) present", "re-running this command is a no-op")


@app.command()
def ingest(
    page: int = typer.Option(3, "--page", help="Which page to dump for the reading-order gate."),
    chars: int = typer.Option(600, "--chars", help="How much of that page to print."),
    quiet: bool = typer.Option(False, "--quiet", help="Skip the text dump."),
) -> None:
    """Extract text and page metadata from every fetched source.

    Prints the first N characters of one page per paper. That dump is the gate, and
    it cannot be automated away: interleaved columns produce text that passes every
    automated check while being semantically destroyed. A human has to read it.
    """
    cfg = Config.load()
    conn = db.connect(cfg)
    db.migrate(conn)
    entries = corpus.load_manifest(cfg)

    # A document dropped from the manifest must leave the database too. Otherwise the
    # index keeps serving a paper the corpus no longer claims to contain, and a
    # citation can point at a source that is no longer part of the deliverable.
    keep = {e.doc_id for e in entries}
    stale = [r["doc_id"] for r in db.all_rows(conn, "SELECT doc_id FROM documents")
             if r["doc_id"] not in keep]
    for doc_id in stale:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        _warn(f"pruned {doc_id}", "no longer in the manifest; its chunks are gone too")
    if stale:
        conn.commit()

    stats = []
    for entry in entries:
        path = cfg.sources_dir / entry.filename
        if not path.exists():
            _fail(entry.doc_id, f"{path.name} missing — run `fetch` first")
            raise typer.Exit(1)

        pages = ingest_mod.extract(path)
        digest = corpus.sha256_file(path)
        profile = ingest_mod.column_profile(pages)

        conn.execute("DELETE FROM documents WHERE doc_id = ?", (entry.doc_id,))
        db.insert(conn, "documents", {
            "doc_id": entry.doc_id, "title": entry.title, "arxiv_id": entry.arxiv_id,
            "path": str(path), "source_url": entry.source_url, "sha256": digest,
            "n_pages": len(pages),
            "extracted_chars": sum(len(p.text) for p in pages),
        })
        db.insert_many(conn, "pages", [
            {"doc_id": entry.doc_id, "page_no": p.number, "n_columns": p.n_columns,
             "n_blocks": p.n_blocks, "text": p.text}
            for p in pages
        ])
        conn.commit()
        stats.append((entry, pages, profile, digest))

    t = Table(show_edge=False)
    for col, just in (("doc_id", "left"), ("pages", "right"), ("chars", "right"),
                      ("chars/page", "right"), ("layout", "left"), ("empty pages", "right")):
        t.add_column(col, justify=just)  # type: ignore[arg-type]
    for entry, pages, profile, _ in stats:
        n_two = profile.get(2, 0)
        layout = (f"TWO-COLUMN ({n_two}/{len(pages)})" if n_two > len(pages) / 2
                  else f"single ({profile.get(1, 0)}/{len(pages)})"
                       + (f", {n_two} mixed" if n_two else ""))
        total = sum(len(p.text) for p in pages)
        empty = sum(1 for p in pages if len(p.text) < 40)
        t.add_row(entry.doc_id, str(len(pages)), f"{total:,}",
                  f"{total // max(len(pages), 1):,}", layout, str(empty) if empty else "-")
    console.print(t)

    # -- automated assertions ------------------------------------------------
    console.print()
    problems = 0
    for entry, pages, _, _ in stats:
        blank = [p.number for p in pages if len(p.text) < 40]
        if len(blank) > len(pages) * 0.2:
            _fail(entry.doc_id, f"{len(blank)} near-empty pages — extraction may have failed")
            problems += 1
    if problems == 0:
        _ok("every document extracted text on the large majority of its pages")

    if quiet:
        return

    console.print()
    console.rule(f"[bold]READING-ORDER GATE — page {page}, first {chars} characters")
    console.print(
        "[dim]Read these. Interleaved columns look fluent and are semantically "
        "destroyed.\nWhat you are checking: does each passage continue into the next, "
        "or does it jump mid-sentence?[/dim]\n"
    )
    for entry, pages, _, _ in stats:
        target = next((p for p in pages if p.number == page), None)
        if target is None:
            _warn(entry.doc_id, f"has no page {page}")
            continue
        console.print(f"[bold cyan]── {entry.doc_id} · {entry.arxiv_id} · "
                      f"page {page} · read as {target.n_columns} column(s) ──[/bold cyan]")
        console.print(target.text[:chars].replace("\n", " ⏎ "))
        console.print()


@app.command("index")
def index_cmd() -> None:
    """Chunk every ingested document, embed the children, build both indexes."""
    cfg = Config.load()
    conn = db.connect(cfg)
    db.migrate(conn)

    stats = index_mod.build(cfg, conn, on_event=console.print)

    console.print()
    console.print(f"  documents        {stats.n_documents}")
    console.print(f"  child chunks     {stats.n_children}")
    console.print(f"  parent chunks    {stats.n_parents}")
    console.print(f"  dimensions       {stats.dims}")
    console.print(f"  device           {stats.device}")
    console.print(f"  peak VRAM        {stats.peak_vram_gb:.2f} GiB")
    console.print(f"  elapsed          {stats.seconds:.1f}s "
                  f"({stats.chunks_per_second:.1f} chunks/sec)")
    console.print(f"  fingerprint      {stats.fingerprint}")

    rows = db.all_rows(conn, "SELECT token_count, page_start FROM chunks WHERE level = 0")
    counts = sorted(r["token_count"] for r in rows)
    console.print()
    console.print(f"  child tokens     min {counts[0]}  median {counts[len(counts)//2]}  "
                  f"max {counts[-1]}  mean {sum(counts)/len(counts):.0f}")

    # Assertions the spec requires of every chunk.
    problems = 0
    null_pages = db.scalar(conn, "SELECT COUNT(*) FROM chunks WHERE page_start IS NULL")
    if null_pages:
        _fail(f"{null_pages} chunks have no page_start", "citations depend on it")
        problems += 1
    tiny = db.scalar(conn, "SELECT COUNT(*) FROM chunks WHERE level = 0 AND token_count < ?",
                     (cfg.min_chunk_tokens,))
    over = db.scalar(conn, "SELECT COUNT(*) FROM chunks WHERE level = 0 AND token_count > ?",
                     (cfg.embed_max_tokens,))
    console.print()
    if null_pages == 0:
        _ok("every chunk has a page_start")
    if tiny:
        _warn(f"{tiny} child chunk(s) below the {cfg.min_chunk_tokens}-token floor")
    else:
        _ok(f"no child chunk below the {cfg.min_chunk_tokens}-token floor")
    if over:
        _fail(f"{over} child chunk(s) exceed the encoder window ({cfg.embed_max_tokens})",
              "they would be silently truncated, losing cited evidence")
        problems += 1
    else:
        _ok(f"no child chunk exceeds the {cfg.embed_max_tokens}-token encoder window")

    if problems:
        raise typer.Exit(1)


_MODELS: dict = {}


def _load_models(cfg: Config):
    """Load the encoders once per process. A batch of twelve questions would
    otherwise pay the load cost twelve times."""
    if not _MODELS:
        _MODELS['embedder'] = index_mod.load_embedder(cfg)
        _MODELS['vectors'] = index_mod.load_vectors(cfg)
        _MODELS['reranker'] = rerank_mod.load_reranker(cfg)
    return _MODELS['embedder'], _MODELS['vectors'], _MODELS['reranker']


def _answer_once(cfg: Config, question: str, provider: str | None,
                 save: str | None, show_retrieval: bool,
                 quiet: bool = False, trace: bool = False):
    """One single-turn question, through the same state machine `chat` uses.

    Deliberately not a second implementation. An earlier version duplicated the
    retrieve/synthesise/verify pipeline here, which meant `ask` silently missed every
    capability added to the conversational path -- the sufficiency loop among them.
    A single-turn question is just a session with one turn.
    """
    conn = db.connect(cfg)
    db.migrate(conn)

    stale = index_mod.check_staleness(cfg, conn)
    if stale:
        _fail('index is stale', stale)
        raise typer.Exit(1)

    models = _load_models(cfg)
    client = LLMClient(cfg=cfg, conn=conn,
                       api_keys=LLMClient.load_keys(cfg, dict(os.environ)))
    session_id = conversation.new_session(conn, index_mod.load_fingerprint(conn))

    try:
        result = agent.run_turn(cfg, conn, client, models, session_id, question,
                                provider=provider)
    except answer_mod.InventedCitation as exc:
        _fail('fabricated citation', str(exc))
        raise typer.Exit(1)
    except QuotaExhausted as exc:
        _fail('quota exhausted', str(exc))
        raise typer.Exit(1)

    if trace and result.trace is not None:
        console.rule('[bold]AGENT TRACE')
        for line in result.trace.render():
            console.print(line, markup=False, highlight=False)
        console.print()

    if show_retrieval and result.answer is not None:
        t = Table(show_edge=False, title_justify='left', title='CONTEXT')
        for c, j in (('#', 'right'), ('rerank', 'right'), ('chunk', 'left'),
                     ('paper', 'left'), ('page', 'right')):
            t.add_column(c, justify=j)
        for h in result.answer.hits:
            t.add_row(str(h.rank),
                      f'{h.rerank_score:.4f}' if h.rerank_score else '-',
                      h.chunk_id, h.doc_id, str(h.page_start))
        console.print(t)
        console.print()

    if not quiet:
        if result.clarification:
            console.print(result.clarification, markup=False)
        elif result.answer is None:
            console.print(f'**Refused.** {result.route.reason}', markup=False)
        else:
            console.print(report.render_answer_markdown(result.answer),
                          markup=False, highlight=False)

    if save and result.answer is not None:
        md, js = report.write_answer(cfg, result.answer, save)
        if not quiet:
            _ok(f'wrote {md.relative_to(cfg.repo_root)} and {js.name}')
    return result


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to retrieve for."),
    retrieve_only: bool = typer.Option(
        False, "--retrieve-only", help="Show retrieval only; no generation."
    ),
    k: int = typer.Option(10, "--k", help="How many results to show per retriever."),
    provider: str = typer.Option(None, "--provider", help="ollama | gemini | auto."),
    save: str = typer.Option(None, "--save", help="Write outputs/answers/<id>.{md,json}."),
    show_retrieval: bool = typer.Option(
        False, "--show-retrieval", help="Also print the retrieval tables."
    ),
    trace: bool = typer.Option(False, "--trace", help="Print the agent loop trace."),
) -> None:
    """Ask a question and get a cited answer, or an explicit refusal."""
    cfg = Config.load()
    if not retrieve_only:
        _answer_once(cfg, question, provider, save, show_retrieval, trace=trace)
        return
    conn = db.connect(cfg)
    db.migrate(conn)

    stale = index_mod.check_staleness(cfg, conn)
    if stale:
        _fail("index is stale", stale)
        raise typer.Exit(1)

    model = index_mod.load_embedder(cfg)
    vectors = index_mod.load_vectors(cfg)
    qvec = retrieve.embed_query(model, question)

    dense = retrieve.dense_search(conn, qvec, vectors, cfg, k=k)
    sparse = retrieve.sparse_search(conn, question, cfg, k=k)

    console.print(f"\n[bold]{question}[/bold]")
    console.print(f"[dim]FTS5 MATCH: {retrieve.build_fts_query(question)}[/dim]\n")

    t = Table(show_edge=False, title="DENSE — bge-m3 cosine", title_justify="left")
    for c, j in (("#", "right"), ("score", "right"), ("paper", "left"),
                 ("page", "right"), ("section", "left"), ("text", "left")):
        t.add_column(c, justify=j)  # type: ignore[arg-type]
    for h in dense:
        t.add_row(str(h.rank), f"{h.dense_score:.4f}", h.doc_id, str(h.page_start),
                  (h.section or "")[:24], " ".join(h.text.split())[:70])
    console.print(t)
    console.print()

    t2 = Table(show_edge=False, title="SPARSE — SQLite FTS5 bm25()", title_justify="left")
    for c, j in (("#", "right"), ("score", "right"), ("paper", "left"),
                 ("page", "right"), ("section", "left"), ("text", "left")):
        t2.add_column(c, justify=j)  # type: ignore[arg-type]
    for h in sparse:
        t2.add_row(str(h.rank), f"{h.sparse_score:.3f}", h.doc_id, str(h.page_start),
                   (h.section or "")[:24], " ".join(h.text.split())[:70])
    console.print(t2)

    overlap = {h.chunk_id for h in dense} & {h.chunk_id for h in sparse}
    console.print(f"\n[dim]{len(overlap)}/{k} chunks appear in both lists — "
                  f"the disagreement is what hybrid retrieval exists to exploit.[/dim]")

    # -- fusion and reranking ------------------------------------------------
    wide_dense = retrieve.dense_search(conn, qvec, vectors, cfg)
    wide_sparse = retrieve.sparse_search(conn, question, cfg)
    fused = retrieve.reciprocal_rank_fusion(wide_dense, wide_sparse, cfg)

    console.print()
    t3 = Table(show_edge=False, title_justify="left",
               title=f"HYBRID — RRF (k={cfg.rrf_k}), top 6 of {len(fused)} fused")
    for c, j in (("#", "right"), ("rrf", "right"), ("paper", "left"),
                 ("page", "right"), ("section", "left"), ("text", "left")):
        t3.add_column(c, justify=j)  # type: ignore[arg-type]
    for h in fused[:6]:
        t3.add_row(str(h.rank), f"{h.rrf_score:.5f}", h.doc_id, str(h.page_start),
                   (h.section or "")[:22], " ".join(h.text.split())[:64])
    console.print(t3)

    model_rr = rerank_mod.load_reranker(cfg)
    reranked = rerank_mod.rerank(model_rr, question, fused, cfg)

    console.print()
    t4 = Table(show_edge=False, title_justify="left",
               title="HYBRID + CROSS-ENCODER RERANK")
    for c, j in (("#", "right"), ("score", "right"), ("was", "right"),
                 ("paper", "left"), ("page", "right"), ("section", "left"),
                 ("text", "left")):
        t4.add_column(c, justify=j)  # type: ignore[arg-type]
    fused_rank = {h.chunk_id: h.rank for h in fused}
    for h in reranked:
        t4.add_row(str(h.rank), f"{h.rerank_score:.4f}",
                   f"#{fused_rank.get(h.chunk_id, 0)}", h.doc_id, str(h.page_start),
                   (h.section or "")[:22], " ".join(h.text.split())[:60])
    console.print(t4)


@app.command()
def labels(
    stamp: bool = typer.Option(False, "--stamp", help="Record each gold chunk's text_sha."),
    show: bool = typer.Option(False, "--show", help="Print every label beside its chunk text."),
    chars: int = typer.Option(700, "--chars", help="How much chunk text to print."),
) -> None:
    """Validate the gold labels against the text the pipeline actually extracted.

    Fabricated ground truth invalidates every number downstream while looking
    perfectly healthy, so labels are checked rather than trusted.
    """
    cfg = Config.load()
    conn = db.connect(cfg)
    db.migrate(conn)
    fingerprint = index_mod.load_fingerprint(conn)

    if stamp:
        n = labels_mod.stamp(cfg, conn)
        _ok(f"stamped text_sha for {n} gold chunk(s)")

    label_set = labels_mod.load(cfg)
    console.print(f"{len(label_set.items)} questions · "
                  f"{len(label_set.answerable)} answerable · "
                  f"{len(label_set.controls)} controls")
    console.print(f"labelled against corpus {label_set.fingerprint_at_labelling} "
                  f"on {label_set.labelled_on}; index is now {fingerprint}\n")

    counts: dict[str, int] = {}
    for item in label_set.items:
        counts[item.cls] = counts.get(item.cls, 0) + 1
    t = Table(show_edge=False)
    t.add_column("class"); t.add_column("n", justify="right"); t.add_column("behaviour")
    for cls, n in counts.items():
        behaviour = {
            "single_hop": "answer, 1-2 citations, correct paper",
            "multi_hop": "answer citing >= 2 papers",
            "unanswerable": "explicit refusal, zero citations",
            "false_premise": "reject the premise, cite what the sources do say",
        }.get(cls, "")
        t.add_row(cls, str(n), behaviour)
    console.print(t)

    if show:
        chunks = labels_mod.fetch_chunks(
            conn, [c for i in label_set.items for c in i.gold_chunks]
        )
        for item in label_set.items:
            console.print()
            console.rule(f"[bold cyan]{item.id} · {item.cls}")
            console.print(f"[bold]{item.question}[/bold]")
            if item.false_premise_is:
                console.print(f"[yellow]false premise:[/yellow] {item.false_premise_is}")
            if item.why_unanswerable:
                console.print(f"[yellow]unanswerable:[/yellow] {item.why_unanswerable}")
            if not item.gold_chunks:
                console.print("[dim]no gold chunks — a control question has no correct "
                              "citation by construction.[/dim]")
                continue
            for cid in item.gold_chunks:
                row = chunks.get(cid)
                if row is None:
                    _fail(cid, "does not exist")
                    continue
                console.print(f"\n  [green]{cid}[/green]  {row['doc_id']} "
                              f"p.{row['page_start']} §{row['section'] or '-'}")
                console.print("  " + " ".join(row["text"].split())[:chars])

    problems = labels_mod.validate(cfg, conn, label_set, fingerprint)
    console.print()
    if problems:
        for p in problems:
            _fail(p)
        raise typer.Exit(1)
    _ok(f"all {len(label_set.items)} labels validate",
        "ids exist, text hashes match, classes agree with expected behaviour")


@app.command("eval-retrieval")
def eval_retrieval(
    thresholds: bool = typer.Option(
        True, "--thresholds/--no-thresholds", help="Also derive the routing thresholds."
    ),
    write: bool = typer.Option(True, "--write/--no-write", help="Write outputs/eval_report.md."),
) -> None:
    """LLM-free retrieval ablation and threshold derivation. Consumes zero quota."""
    cfg = Config.load()
    conn = db.connect(cfg)
    db.migrate(conn)

    stale = index_mod.check_staleness(cfg, conn)
    if stale:
        _fail("index is stale", stale)
        raise typer.Exit(1)

    label_set = labels_mod.load(cfg)
    problems = labels_mod.validate(cfg, conn, label_set, index_mod.load_fingerprint(conn))
    if problems:
        for p in problems:
            _fail(p)
        console.print("\n[red]Refusing to evaluate against unvalidated labels.[/red]")
        raise typer.Exit(1)
    _ok(f"{len(label_set.items)} labels validated against the current index")

    calls_before = db.scalar(conn, "SELECT COUNT(*) FROM llm_calls") or 0
    retrievers = evaluate.Retrievers(cfg, conn)

    results = []
    for config in evaluate.CONFIGS:
        res = evaluate.evaluate_config(retrievers, label_set.items, config)
        results.append(res)
        console.print(f"  {config:<16} {len(res.items)} questions scored")

    t = Table(show_edge=False, title_justify="left",
              title=f"RETRIEVAL ABLATION — {len(results[0].items)} answerable questions, "
                    f"zero LLM calls")
    t.add_column("config"); t.add_column("Recall@5", justify="right")
    t.add_column("95% CI", justify="right"); t.add_column("MRR", justify="right")
    t.add_column("nDCG@10", justify="right"); t.add_column("p50 ms", justify="right")
    t.add_column("p95 ms", justify="right")
    for res in results:
        lo, hi = res.ci95(lambda i: i.recall_at(5))
        t.add_row(res.config, f"{res.mean(lambda i: i.recall_at(5)):.3f}",
                  f"[{lo:.2f}, {hi:.2f}]",
                  f"{res.mean(lambda i: i.mrr()):.3f}",
                  f"{res.mean(lambda i: i.ndcg_at(10)):.3f}",
                  f"{res.p50_latency():.0f}", f"{res.p95_latency():.0f}")
    console.print()
    console.print(t)

    by_name = {r.config: r for r in results}
    hybrid = by_name["hybrid"].mean(lambda i: i.recall_at(5))
    dense = by_name["dense"].mean(lambda i: i.recall_at(5))
    sparse = by_name["sparse"].mean(lambda i: i.recall_at(5))
    console.print()
    if hybrid >= max(dense, sparse):
        _ok("hybrid >= both single-retriever baselines on Recall@5",
            f"hybrid {hybrid:.3f} vs dense {dense:.3f}, sparse {sparse:.3f}")
    else:
        _fail("hybrid does NOT beat both baselines on Recall@5",
              f"hybrid {hybrid:.3f} vs dense {dense:.3f}, sparse {sparse:.3f} — investigate")

    run_id = evaluate.persist(conn, cfg, results, notes="llm-free retrieval ablation")
    _ok(f"persisted as {run_id}", "eval_runs / eval_results")

    evidence = None
    if thresholds:
        console.print()
        console.rule("[bold]THRESHOLD DERIVATION")
        evidence = evaluate.derive_thresholds(retrievers, label_set, cfg)
        for pop in (evidence.routing_positives, evidence.routing_negatives,
                    evidence.pair_positives, evidence.pair_negatives):
            console.print(f"\n[bold]{pop.name}[/bold]  (n={pop.n})")
            s = pop.summary()
            console.print("  " + "  ".join(f"{k}={v:.3f}" if k != "n" else f"n={int(v)}"
                                           for k, v in s.items()))
            for line in rerank_mod.histogram(pop):
                console.print(line)

        console.print("\n[bold]Sensitivity sweep[/bold]  "
                      "(what each candidate tau would actually do)")
        console.print("   tau  | gold kept | control rejected | balanced")
        for tau, keep, rej in zip(evidence.sweep.taus,
                                  evidence.sweep.positive_retention,
                                  evidence.sweep.negative_rejection):
            if round(tau * 40) % 4:
                continue
            console.print(f"  {tau:.2f} |   {keep:6.1%}  |     {rej:6.1%}       "
                          f"|  {(keep + rej) / 2:6.1%}")
        best_tau, best_bal = evidence.sweep.best_f1()
        console.print(f"\n  balanced-accuracy peak at tau={best_tau:.2f} ({best_bal:.1%})")
        if not evidence.separable:
            console.print()
            _warn("the two routing populations OVERLAP",
                  "no single threshold separates them cleanly")
        for name, value in (("TAU_LOW", evidence.tau_low),
                            ("TAU_HIGH", evidence.tau_high),
                            ("TAU_VERIFY", evidence.tau_verify)):
            console.print(f"\n  [bold]{name} = {value}[/bold]")
            console.print(f"    [dim]{evidence.derivation.get(name.lower(), '')}[/dim]")
        console.print(f"\n  Between {evidence.tau_low} and {evidence.tau_high} the "
                      f"agent asks rather than guesses.")

    calls_after = db.scalar(conn, "SELECT COUNT(*) FROM llm_calls") or 0
    console.print()
    if calls_after == calls_before:
        _ok(f"zero LLM calls consumed", f"ledger unchanged at {calls_after} rows")
    else:
        _fail(f"{calls_after - calls_before} LLM calls consumed",
              "retrieval evaluation must be model-independent")

    if write:
        path = report.write_eval_report(cfg, results, label_set, evidence)
        _ok(f"wrote {path.relative_to(cfg.repo_root)}")


@app.command("answer-all")
def answer_all(
    provider: str = typer.Option(None, "--provider", help="ollama | gemini | auto."),
    only: str = typer.Option(None, "--only", help="Comma-separated question ids."),
) -> None:
    """Run the whole single-turn question set and report the contract properties."""
    cfg = Config.load()
    conn = db.connect(cfg)
    db.migrate(conn)
    label_set = labels_mod.load(cfg)

    # Same reason as chat-eval: a cache that persists between runs makes the run
    # measure its own history rather than the pipeline (decisions.md D-125).
    conn.execute("DELETE FROM answer_cache")
    conn.commit()

    wanted = set(only.split(",")) if only else None
    items = [i for i in label_set.items if not wanted or i.id in wanted]

    rows = []
    for item in items:
        console.print(f"[dim]-> {item.id}[/dim]")
        try:
            turn = _answer_once(cfg, item.question, provider, item.id,
                                show_retrieval=False, quiet=True)
            ans = turn.answer
            invented = (ans.cited_chunk_ids - {h.chunk_id for h in ans.hits}
                        if ans else set())
            papers = ({h.doc_id for h in ans.hits if h.chunk_id in ans.cited_chunk_ids}
                      if ans else set())
            rows.append({
                "id": item.id, "class": item.cls,
                "expected": "abstain" if item.must_abstain else "answer",
                "actual": turn.decision,
                "schema_ok": True, "invented": len(invented),
                "n_sent": len(ans.sentences) if ans else 0,
                "n_cites": len(ans.cited_chunk_ids) if ans else 0,
                "papers": len(papers),
                "verified": (sum(1 for s in ans.sentences if s.status == "verified")
                             if ans else 0),
                "loops": turn.n_loops,
                "subs": len(turn.sub_questions),
                "ms": turn.latency_ms,
            })
        except Exception as exc:
            rows.append({"id": item.id, "class": item.cls,
                         "expected": "abstain" if item.must_abstain else "answer",
                         "actual": f"ERROR {type(exc).__name__}", "schema_ok": False,
                         "invented": 0, "n_sent": 0, "n_cites": 0, "papers": 0,
                         "verified": 0, "loops": 0, "subs": 0, "ms": 0})
            _fail(item.id, f"{type(exc).__name__}: {exc}")

    t = Table(show_edge=False)
    numeric = ("sent", "cites", "papers", "verified", "loops", "subs", "ms")
    for c in ("id", "class", "expected", "actual") + numeric:
        t.add_column(c, justify="right" if c in numeric else "left")
    for r in rows:
        ok = (r["actual"] in {"refuse", "abstain"} if r["expected"] == "abstain"
              else r["actual"] == r["expected"])
        mark = "" if ok else "  <-- MISMATCH"
        t.add_row(r["id"], r["class"], r["expected"], r["actual"] + mark,
                  str(r["n_sent"]), str(r["n_cites"]), str(r["papers"]),
                  str(r["verified"]), str(r["loops"]), str(r["subs"]), str(r["ms"]))
    console.print()
    console.print(t)

    n = len(rows)
    schema_ok = sum(1 for r in rows if r["schema_ok"])
    no_invented = sum(1 for r in rows if r["invented"] == 0)
    controls = [r for r in rows if r["expected"] == "abstain"]
    # Refusing on the score and abstaining after synthesis are different mechanisms
    # with the same correct outcome: no answer, no citations.
    controls_ok = sum(1 for r in controls if r["actual"] in {"refuse", "abstain"})
    refusals_clean = all(r["n_cites"] == 0 for r in rows if r["actual"] == "abstain")

    console.print()
    console.print(f"  schema-valid                         {schema_ok}/{n}")
    console.print(f"  zero invented citation ids           {no_invented}/{n}")
    console.print(f"  controls that abstained              {controls_ok}/{len(controls)}")
    console.print(f"  refusals carrying zero citations     {'yes' if refusals_clean else 'NO'}")
    total_cites = db.scalar(conn, "SELECT COUNT(*) FROM turn_citations") or 0
    console.print(f"  turn_citations rows in the database  {total_cites}")


def _render_turn(result, cfg: Config, show_trace: bool = True) -> None:
    """Print one turn: what the condenser did, how it routed, and the answer."""
    c = result.condensed
    if show_trace:
        if c.used_llm or c.fell_back:
            flag = "[red]DRIFT[/red]" if c.drifted else "ok"
            console.print(f"[dim]  raw       :[/dim] {result.raw_text}")
            console.print(f"[dim]  condensed :[/dim] {c.query}")
            console.print(f"[dim]  drift     :[/dim] {flag} — {c.reason}")
        else:
            console.print(f"[dim]  condense  : skipped ({c.reason})[/dim]")
        if result.resolved_refs:
            console.print(f"[dim]  resolved  :[/dim] {', '.join(result.resolved_refs)}")
        console.print(f"[dim]  route     :[/dim] {result.decision} — {result.route.reason}")

    if result.cache_hit:
        console.print("\n[green]Served from the semantic answer cache.[/green]\n")
        return
    if result.clarification:
        console.print(f"\n[yellow]{result.clarification}[/yellow]\n", markup=False)
        return
    if result.answer is None:
        console.print(f"\n**Refused.** {result.route.reason}\n", markup=False)
        return
    console.print()
    console.print(report.render_answer_markdown(result.answer), markup=False,
                  highlight=False)


@app.command()
def chat(
    provider: str = typer.Option(None, "--provider", help="ollama | gemini | auto."),
    session: str = typer.Option(None, "--session", help="Resume an existing session."),
) -> None:
    """Multi-turn REPL. /history /sources /new /quit"""
    cfg = Config.load()
    conn = db.connect(cfg)
    db.migrate(conn)
    stale = index_mod.check_staleness(cfg, conn)
    if stale:
        _fail("index is stale", stale)
        raise typer.Exit(1)

    fingerprint = index_mod.load_fingerprint(conn)
    session_id = session or conversation.new_session(conn, fingerprint)
    warning = conversation.fingerprint_warning(conn, session_id, fingerprint)
    if warning:
        _warn("corpus changed since this session started", warning)

    models = _load_models(cfg)
    client = LLMClient(cfg=cfg, conn=conn,
                       api_keys=LLMClient.load_keys(cfg, dict(os.environ)))
    console.print(f"Session [bold]{session_id}[/bold] · corpus {fingerprint}")
    console.print("[dim]/history  /sources  /new  /quit[/dim]\n")

    while True:
        try:
            raw = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not raw:
            continue
        if raw in {"/quit", "/exit"}:
            return
        if raw == "/new":
            session_id = conversation.new_session(conn, fingerprint)
            console.print(f"[dim]new session {session_id}[/dim]\n")
            continue
        if raw == "/history":
            for t in conversation.load_turns(conn, session_id):
                who = "you" if t.role == "user" else "agent"
                body = t.raw_text if t.role == "user" else (t.answer_text or t.raw_text)
                console.print(f"  {t.ord:>2}. [{who}] {body[:110]}")
            console.print()
            continue
        if raw == "/sources":
            rows = db.all_rows(conn, """
                SELECT DISTINCT tc.chunk_id, d.title, c.page_start
                FROM turn_citations tc
                JOIN turns t ON t.turn_id = tc.turn_id
                JOIN chunks c ON c.chunk_id = tc.chunk_id
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE t.session_id = ? ORDER BY tc.chunk_id
            """, (session_id,))
            for r in rows:
                console.print(f"  {r['chunk_id']}  {r['title']} · p.{r['page_start']}")
            console.print()
            continue

        try:
            result = agent.run_turn(cfg, conn, client, models, session_id, raw,
                                    provider=provider)
            _render_turn(result, cfg)
        except Exception as exc:
            _fail(f"{type(exc).__name__}", str(exc))


@app.command()
def history(
    session: str = typer.Option(..., "--session", help="Session id."),
) -> None:
    """Full stored history for a session, including citations."""
    cfg = Config.load()
    conn = db.connect(cfg)
    db.migrate(conn)
    for t in conversation.load_turns(conn, session):
        who = "user" if t.role == "user" else "agent"
        console.print(f"[bold]{t.ord:>2}. {who}[/bold] "
                      f"{'(' + t.route + ')' if t.route else ''}")
        console.print(f"    {t.raw_text}")
        if t.condensed_query and t.condensed_query != t.raw_text:
            console.print(f"    [dim]condensed: {t.condensed_query}[/dim]")
        if t.answer_text:
            console.print(f"    {t.answer_text[:200]}")


@app.command("chat-eval")
def chat_eval(
    provider: str = typer.Option(None, "--provider", help="ollama | gemini | auto."),
    only: str = typer.Option(None, "--only", help="Comma-separated scenario ids."),
) -> None:
    """Run every conversation scenario and report route + drift behaviour."""
    import yaml

    cfg = Config.load()
    conn = db.connect(cfg)
    db.migrate(conn)
    fingerprint = index_mod.load_fingerprint(conn)
    models = _load_models(cfg)
    client = LLMClient(cfg=cfg, conn=conn,
                       api_keys=LLMClient.load_keys(cfg, dict(os.environ)))

    # The semantic answer cache accumulates across runs and short-circuits turns,
    # which makes two eval runs incomparable. Evaluation starts from an empty one so
    # the numbers describe the pipeline rather than the cache's history.
    conn.execute("DELETE FROM answer_cache")
    conn.commit()

    spec = yaml.safe_load(cfg.conversations_path.read_text(encoding="utf-8"))
    wanted = set(only.split(",")) if only else None
    rows: list[dict] = []
    transcripts: dict[str, list[str]] = {}

    for scenario in spec["scenarios"]:
        if wanted and scenario["id"] not in wanted:
            continue
        console.rule(f"[bold cyan]{scenario['id']} — {scenario['title']}")
        session_id = conversation.new_session(conn, fingerprint)
        lines = [f"# {scenario['id']} — {scenario['title']}", "",
                 f"Session `{session_id}` · corpus `{fingerprint}`", ""]

        for i, turn in enumerate(scenario["turns"], start=1):
            raw = turn["raw_text"]
            console.print(f"\n[bold]turn {i}:[/bold] {raw}")
            result = agent.run_turn(
                cfg, conn, client, models, session_id, raw, provider=provider,
                is_clarification_reply=bool(turn.get("is_clarification_reply")),
            )
            _render_turn(result, cfg)

            expected = turn.get("expected_route")
            actual = result.decision
            contains = turn.get("expected_condensation_contains") or []
            forbidden = turn.get("must_not_contain") or []
            cq = result.condensed.query
            missing = [w for w in contains if w.lower() not in cq.lower()]
            leaked = [w for w in forbidden if w.lower() in cq.lower()]

            rows.append({
                "scenario": scenario["id"], "turn": i,
                "expected": expected, "actual": actual,
                "match": expected == actual,
                "drifted": result.condensed.drifted,
                "novel": sorted(result.condensed.novel_words),
                "missing": missing, "leaked": leaked,
                "refs": result.resolved_refs,
            })

            lines += [f"## Turn {i}", "", f"**User:** {raw}", ""]
            if result.condensed.used_llm:
                lines += [f"*Condensed to:* `{cq}`",
                          f"*Drift guard:* {'FELL BACK' if result.condensed.drifted else 'passed'}"
                          f" — {result.condensed.reason}", ""]
            else:
                lines += [f"*Condensation:* skipped — {result.condensed.reason}", ""]
            if result.resolved_refs:
                lines += [f"*Resolved source references:* `{', '.join(result.resolved_refs)}`", ""]
            lines += [f"*Route:* `{actual}` (expected `{expected}`) — {result.route.reason}", ""]
            if result.clarification:
                lines += [f"**Agent asks:** {result.clarification}", ""]
            elif result.answer is not None:
                lines += [report.render_answer_markdown(result.answer), ""]
            else:
                lines += [f"**Agent refuses.** {result.route.reason}", ""]

        transcripts[scenario["id"]] = lines

    out_dir = cfg.outputs_dir / "conversations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid, lines in transcripts.items():
        (out_dir / f"{sid}.md").write_text("\n".join(lines), encoding="utf-8")

    t = Table(show_edge=False)
    for c in ("scenario", "turn", "expected", "actual", "drift", "condensation"):
        t.add_column(c)
    for r in rows:
        note = ""
        if r["missing"]:
            note += f"missing {r['missing']} "
        if r["leaked"]:
            note += f"[red]LEAKED {r['leaked']}[/red]"
        t.add_row(r["scenario"][:22], str(r["turn"]), str(r["expected"]),
                  r["actual"] + ("" if r["match"] else "  <-- MISMATCH"),
                  "[red]DRIFT[/red]" if r["drifted"] else "-", note or "ok")
    console.print()
    console.print(t)

    n = len(rows)
    correct = sum(1 for r in rows if r["match"])
    drifted = sum(1 for r in rows if r["drifted"])
    console.print()
    console.print(f"  route accuracy            {correct}/{n}")
    console.print(f"  condensation drift rate   {drifted}/{n}"
                  f"  (target 0)")
    console.print(f"  transcripts               outputs/conversations/ "
                  f"({len(transcripts)} files)")


@app.command("db")
def db_cmd(
    tables: bool = typer.Option(False, "--tables", help="List every table."),
    reset: bool = typer.Option(False, "--reset", help="Delete the database file."),
    confirm: bool = typer.Option(False, "--confirm", help="Required by --reset."),
) -> None:
    """Inspect or reset the database. Avoids requiring the `sqlite3` CLI, which is
    not installed on Windows by default."""
    cfg = Config.load()
    if reset:
        db.reset(cfg, confirm=confirm)
        console.print(f"Deleted {cfg.db_path} and its WAL sidecars.")
        return
    conn = db.connect(cfg)
    db.migrate(conn)
    if tables:
        for name in db.table_names(conn):
            n = db.scalar(conn, f"SELECT COUNT(*) FROM {name}")
            console.print(f"  {name:<22} {n:>8} row(s)")
    else:
        console.print(f"{cfg.db_path}  ({len(db.table_names(conn))} tables)")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(app())
