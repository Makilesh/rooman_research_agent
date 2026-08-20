"""Command-line entry point.

Step 1 ships `doctor` and `budget`. Later steps add `fetch`, `ingest`, `index`,
`ask`, `chat`, `eval` and `history`; they are not stubbed here, because a command
that exists and does nothing is worse than one that does not exist yet.
"""

from __future__ import annotations

import os
import sys

import typer
from rich.console import Console
from rich.table import Table

from . import db
from .config import Config
from .llm import LLMClient, OllamaProvider

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
                a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
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
