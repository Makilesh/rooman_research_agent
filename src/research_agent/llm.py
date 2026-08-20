"""Provider layer: two ladders, a durable quota limiter, and a disk cache.

The design claim this file has to earn: **a daily rate limit cannot be enforced from
a process that exits between every request.** An in-memory limiter resets on each CLI
invocation, so it enforces nothing and you burn a 20-request daily cap without ever
seeing a warning. Here RPM and RPD are derived by windowed counts over durable
`llm_calls` rows, so the limiter is correct across restarts.

Two ladders, because a conversational turn costs roughly one synthesis call plus two
to four volume calls. Routing volume work through synthesis models exhausts reasoning
quota in about four turns; separating them means answer quality degrades last.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sqlite3 import Connection
from typing import Any, Callable, Protocol

from .config import Config, Ladder, Rung

# Windows over which the ledger counts. RPD is a rolling 24h window rather than a
# calendar day: Google resets on Pacific midnight, and a rolling window is the
# conservative reading -- it can refuse slightly early, never slightly late.
RPM_WINDOW = timedelta(minutes=1)
RPD_WINDOW = timedelta(hours=24)

# Pseudo-key for Ollama, which has no quota. Present so every ledger row has a
# key_alias and `budget` needs no special case.
LOCAL_KEY = "local"

# Local inference has no provider-imposed cap. A sentinel keeps the ledger path
# identical for both providers instead of branching around the limiter.
UNLIMITED = 1_000_000_000


class QuotaExhausted(RuntimeError):
    """Every rung on a ladder is saturated. Names the rung; never a stack trace."""

    def __init__(self, ladder: Ladder, tried: list[tuple[str, str]]) -> None:
        self.ladder = ladder
        self.tried = tried
        detail = ", ".join(f"{m}/{k}" for m, k in tried) or "no models configured"
        super().__init__(
            f"The {ladder} ladder is exhausted for today. Tried: {detail}. "
            f"Free-tier quota resets on a rolling 24h window; "
            f"`research-agent budget` shows when capacity returns. "
            f"The Ollama path is unlimited and needs no key."
        )


class MalformedResponse(RuntimeError):
    """The model returned something that is not valid JSON for the given schema."""


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    data: dict[str, Any] | None
    provider: str
    model: str
    key_alias: str
    ladder: Ladder | None
    cached: bool
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# ---------------------------------------------------------------------------
#  Providers
# ---------------------------------------------------------------------------
class Provider(Protocol):
    name: str

    def generate(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any] | None,
        api_key: str | None,
        timeout_s: float,
    ) -> tuple[str, int | None, int | None]:
        """Return (raw_text, prompt_tokens, completion_tokens)."""


class OllamaProvider:
    """Local, unlimited, and the default. Driven over HTTP, never by shelling out.

    `ollama list` was observed to hang past 120s on this machine while
    `GET /api/tags` returned instantly, so every interaction here is an HTTP call
    with a real timeout (decisions.md D-105).
    """

    name = "ollama"

    def __init__(self, host: str, transport: Callable[..., Any] | None = None) -> None:
        self.host = host.rstrip("/")
        # Injectable so tests exercise the full path with zero network.
        self._transport = transport

    def _post(self, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(path, payload, timeout_s)
        import httpx

        r = httpx.post(f"{self.host}{path}", json=payload, timeout=timeout_s)
        r.raise_for_status()
        return r.json()

    def tags(self, timeout_s: float = 5.0) -> list[dict[str, Any]]:
        if self._transport is not None:
            return self._transport("/api/tags", {}, timeout_s).get("models", [])
        import httpx

        r = httpx.get(f"{self.host}/api/tags", timeout=timeout_s)
        r.raise_for_status()
        return r.json().get("models", [])

    def generate(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any] | None,
        api_key: str | None,
        timeout_s: float,
    ) -> tuple[str, int | None, int | None]:
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if schema is not None:
            # Ollama constrains generation to a JSON schema via `format`. This is
            # what makes the citation contract structurally hard to violate rather
            # than something a retry loop has to clean up after.
            payload["format"] = schema
        body = self._post("/api/generate", payload, timeout_s)
        return (
            body.get("response", ""),
            body.get("prompt_eval_count"),
            body.get("eval_count"),
        )


class GeminiProvider:
    """Cloud, quota-limited, optional. Never required for the agent to run."""

    name = "gemini"

    def __init__(self, transport: Callable[..., Any] | None = None) -> None:
        self._transport = transport
        self._clients: dict[str, Any] = {}

    def _client(self, api_key: str) -> Any:
        if api_key not in self._clients:
            from google import genai

            self._clients[api_key] = genai.Client(api_key=api_key)
        return self._clients[api_key]

    def generate(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any] | None,
        api_key: str | None,
        timeout_s: float,
    ) -> tuple[str, int | None, int | None]:
        if self._transport is not None:
            return self._transport(model, prompt, schema, api_key, timeout_s)
        if not api_key:
            raise RuntimeError("Gemini called with no API key.")

        cfg: dict[str, Any] = {}
        if schema is not None:
            # Gemini's equivalent of Ollama's `format`: the same contract enforced
            # on both providers, so a malformed payload is an edge case on either.
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema

        resp = self._client(api_key).models.generate_content(
            model=model, contents=prompt, config=cfg or None
        )
        usage = getattr(resp, "usage_metadata", None)
        return (
            resp.text or "",
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )


# ---------------------------------------------------------------------------
#  The ledger
# ---------------------------------------------------------------------------
class Ledger:
    """Sliding-window RPM/RPD accounting over durable `llm_calls` rows.

    A row is written at *acquire* time, before the call is made, and updated with the
    outcome afterwards. Counting attempts rather than successes is deliberate: the
    provider counts a failed request against quota too, and a crash mid-call must not
    leave the day's usage understated.

    Cached rows are written with `cached=1` and excluded from every window, because a
    cache hit consumes no quota.
    """

    def __init__(self, conn: Connection, now: Callable[[], datetime] | None = None) -> None:
        self.conn = conn
        # Injectable clock: the limiter tests use a fake one and never sleep.
        self.now = now or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).isoformat()

    def used(self, model: str, key_alias: str) -> tuple[int, int]:
        """(requests in the last minute, requests in the last 24h)."""
        now = self.now()
        sql = (
            "SELECT COUNT(*) FROM llm_calls "
            "WHERE model = ? AND key_alias = ? AND cached = 0 AND ts >= ?"
        )
        rpm = self.conn.execute(sql, (model, key_alias, self._iso(now - RPM_WINDOW))).fetchone()[0]
        rpd = self.conn.execute(sql, (model, key_alias, self._iso(now - RPD_WINDOW))).fetchone()[0]
        return rpm, rpd

    def try_acquire(
        self,
        model: str,
        key_alias: str,
        rpm: int,
        rpd: int,
        purpose: str,
        prompt_sha: str,
        provider: str,
        ladder: Ladder | None,
    ) -> int | None:
        """Reserve one request slot, or return None immediately if saturated.

        Non-blocking on purpose. Sleeping on a saturated key while another key sits
        idle would make rotation add no real capacity at all.
        """
        used_rpm, used_rpd = self.used(model, key_alias)
        if used_rpm >= rpm or used_rpd >= rpd:
            return None
        cur = self.conn.execute(
            "INSERT INTO llm_calls (ts, provider, model, key_alias, ladder, purpose, "
            "prompt_sha, cached, ok) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)",
            (self._iso(self.now()), provider, model, key_alias, ladder, purpose, prompt_sha),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def record_cached(
        self, provider: str, model: str, key_alias: str, ladder: Ladder | None,
        purpose: str, prompt_sha: str,
    ) -> int:
        """A cache hit still gets a ledger row -- it just does not count against quota."""
        cur = self.conn.execute(
            "INSERT INTO llm_calls (ts, provider, model, key_alias, ladder, purpose, "
            "prompt_sha, cached, ok, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, 0)",
            (self._iso(self.now()), provider, model, key_alias, ladder, purpose, prompt_sha),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish(
        self, row_id: int, ok: bool, latency_ms: int,
        error: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        self.conn.execute(
            "UPDATE llm_calls SET ok = ?, error = ?, latency_ms = ?, prompt_tokens = ?, "
            "completion_tokens = ? WHERE id = ?",
            (int(ok), error, latency_ms, prompt_tokens, completion_tokens, row_id),
        )
        self.conn.commit()

    def remaining(self, model: str, key_alias: str, rpm: int, rpd: int) -> dict[str, int]:
        used_rpm, used_rpd = self.used(model, key_alias)
        return {
            "rpm_used": used_rpm, "rpm_limit": rpm, "rpm_left": max(0, rpm - used_rpm),
            "rpd_used": used_rpd, "rpd_limit": rpd, "rpd_left": max(0, rpd - used_rpd),
        }


# ---------------------------------------------------------------------------
#  Disk cache
# ---------------------------------------------------------------------------
class DiskCache:
    """sha256(model + prompt + params) -> response text.

    Non-negotiable: without it one debugging session eats the day's synthesis quota.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(model: str, prompt: str, params: dict[str, Any] | None) -> str:
        blob = json.dumps(
            {"model": model, "prompt": prompt, "params": params or {}},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        # Two-level fan-out: a flat directory of thousands of files is slow to list
        # on Windows and unpleasant to inspect by hand.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))["text"]
        except (json.JSONDecodeError, KeyError, OSError):
            # A corrupt cache entry is a cache miss, never a crash.
            return None

    def put(self, key: str, text: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
#  The client
# ---------------------------------------------------------------------------
@dataclass
class LLMClient:
    cfg: Config
    conn: Connection
    api_keys: dict[str, str] = field(default_factory=dict)
    ollama: OllamaProvider | None = None
    gemini: GeminiProvider | None = None
    now: Callable[[], datetime] | None = None
    monotonic: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.ledger = Ledger(self.conn, self.now)
        self.cache = DiskCache(self.cfg.llm_cache_dir)
        if self.ollama is None:
            self.ollama = OllamaProvider(self.cfg.ollama_host)
        if self.gemini is None:
            self.gemini = GeminiProvider()

    # -- keys ---------------------------------------------------------------
    @staticmethod
    def load_keys(cfg: Config, environ: dict[str, str]) -> dict[str, str]:
        """Read GEMINI_API_KEY_1..N. Absent keys are simply absent -- not an error.

        The Ollama path must work with no key at all, so this never raises.
        """
        keys: dict[str, str] = {}
        for i in range(1, cfg.gemini_max_keys + 1):
            value = (environ.get(f"{cfg.gemini_key_env_prefix}{i}") or "").strip()
            if value:
                keys[f"key_{i}"] = value
        return keys

    # -- the call -----------------------------------------------------------
    def complete(
        self,
        prompt: str,
        purpose: str,
        ladder: Ladder = "volume",
        schema: dict[str, Any] | None = None,
        provider: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Completion:
        """One call signature for both providers, with schema-constrained JSON on both."""
        chosen = provider or self.cfg.provider
        if chosen == "ollama":
            return self._complete_ollama(prompt, purpose, ladder, schema, params)
        if chosen == "gemini":
            return self._complete_gemini(prompt, purpose, ladder, schema, params)
        # auto: local first, cloud only when local is unavailable. Keeps the default
        # path free and makes cloud spend an explicit fallback rather than a habit.
        try:
            return self._complete_ollama(prompt, purpose, ladder, schema, params)
        except Exception:
            return self._complete_gemini(prompt, purpose, ladder, schema, params)

    def _parse(self, raw: str, schema: dict[str, Any] | None) -> dict[str, Any] | None:
        if schema is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MalformedResponse(f"Response was not valid JSON: {exc}") from exc

    def _complete_ollama(
        self, prompt: str, purpose: str, ladder: Ladder,
        schema: dict[str, Any] | None, params: dict[str, Any] | None,
    ) -> Completion:
        model = (
            self.cfg.ollama_model_large
            if self.cfg.offload_mode == "sequential"
            else self.cfg.ollama_model
        )
        sha = DiskCache.key(model, prompt, params)

        hit = self.cache.get(sha)
        if hit is not None:
            self.ledger.record_cached("ollama", model, LOCAL_KEY, ladder, purpose, sha)
            return Completion(hit, self._parse(hit, schema), "ollama", model,
                              LOCAL_KEY, ladder, True, 0)

        # Local inference has no quota, but it still gets a ledger row so that
        # "LLM calls per turn" is measurable on the free path too.
        row_id = self.ledger.try_acquire(
            model, LOCAL_KEY, rpm=UNLIMITED, rpd=UNLIMITED, purpose=purpose,
            prompt_sha=sha, provider="ollama", ladder=ladder,
        )
        assert row_id is not None  # local limits are effectively infinite

        return self._invoke(
            provider_obj=self.ollama, provider_name="ollama", model=model,
            api_key=None, timeout_s=self.cfg.ollama_timeout_s, prompt=prompt,
            schema=schema, sha=sha, row_id=row_id, key_alias=LOCAL_KEY, ladder=ladder,
            purpose=purpose, rpm=UNLIMITED, rpd=UNLIMITED,
        )

    def _complete_gemini(
        self, prompt: str, purpose: str, ladder: Ladder,
        schema: dict[str, Any] | None, params: dict[str, Any] | None,
    ) -> Completion:
        rungs: tuple[Rung, ...] = self.cfg.ladder(ladder)
        if not self.api_keys:
            raise QuotaExhausted(ladder, [])

        tried: list[tuple[str, str]] = []
        for rung in rungs:
            sha = DiskCache.key(rung.model, prompt, params)
            hit = self.cache.get(sha)
            if hit is not None:
                alias = next(iter(self.api_keys))
                self.ledger.record_cached("gemini", rung.model, alias, ladder, purpose, sha)
                return Completion(hit, self._parse(hit, schema), "gemini", rung.model,
                                  alias, ladder, True, 0)

            # Drain every key on this rung before stepping down. Stepping down while
            # another key still has quota on the current rung would throw away
            # capability for no reason.
            for alias, api_key in self.api_keys.items():
                tried.append((rung.model, alias))
                row_id = self.ledger.try_acquire(
                    rung.model, alias, rung.rpm, rung.rpd, purpose,
                    sha, "gemini", ladder,
                )
                if row_id is None:
                    continue  # saturated -- next key, without sleeping
                return self._invoke(
                    provider_obj=self.gemini, provider_name="gemini", model=rung.model,
                    api_key=api_key, timeout_s=self.cfg.gemini_timeout_s, prompt=prompt,
                    schema=schema, sha=sha, row_id=row_id, key_alias=alias, ladder=ladder,
                    purpose=purpose, rpm=rung.rpm, rpd=rung.rpd,
                )
        raise QuotaExhausted(ladder, tried)

    def _invoke(
        self, provider_obj: Any, provider_name: str, model: str, api_key: str | None,
        timeout_s: float, prompt: str, schema: dict[str, Any] | None, sha: str,
        row_id: int, key_alias: str, ladder: Ladder | None, purpose: str,
        rpm: int, rpd: int,
    ) -> Completion:
        """Call, time, parse, retry once on malformed JSON, then give up loudly.

        `row_id` is a slot already reserved by the caller. A retry is a second real
        request against the provider, so it reserves its own slot -- charging one
        request for two is how a ledger quietly drifts out of step with the quota it
        is supposed to model.
        """
        attempts = self.cfg.schema_retry_attempts + 1
        last_error: Exception | None = None
        current_row: int | None = row_id

        for attempt in range(attempts):
            if current_row is None:
                # No quota left to fund the retry.
                break
            started = self.monotonic()
            try:
                raw, ptok, ctok = provider_obj.generate(
                    model, prompt, schema, api_key, timeout_s
                )
                data = self._parse(raw, schema)
            except MalformedResponse as exc:
                last_error = exc
                self.ledger.finish(
                    current_row, ok=False,
                    latency_ms=int((self.monotonic() - started) * 1000), error=str(exc),
                )
                current_row = (
                    self.ledger.try_acquire(
                        model, key_alias, rpm, rpd, f"{purpose}:retry",
                        sha, provider_name, ladder,
                    )
                    if attempt + 1 < attempts
                    else None
                )
                continue
            except Exception as exc:
                # Transport, timeout, auth: the attempt still consumed a slot, and
                # the caller needs the real error rather than a retry loop.
                self.ledger.finish(
                    current_row, ok=False,
                    latency_ms=int((self.monotonic() - started) * 1000), error=str(exc),
                )
                raise

            latency_ms = int((self.monotonic() - started) * 1000)
            self.ledger.finish(current_row, ok=True, latency_ms=latency_ms,
                               prompt_tokens=ptok, completion_tokens=ctok)
            self.cache.put(sha, raw)
            return Completion(raw, data, provider_name, model, key_alias, ladder,
                              False, latency_ms, ptok, ctok)

        raise MalformedResponse(
            f"{model} returned unparseable JSON on {attempts} attempt(s). "
            f"Last error: {last_error}"
        )

    # -- reporting ----------------------------------------------------------
    def budget(self) -> list[dict[str, Any]]:
        """Remaining RPM/RPD per model per key, read from the ledger."""
        out: list[dict[str, Any]] = []
        aliases = list(self.api_keys) or ["(no key configured)"]
        for ladder_name in ("synthesis", "volume"):
            for rung in self.cfg.ladder(ladder_name):  # type: ignore[arg-type]
                for alias in aliases:
                    row = {"ladder": ladder_name, "model": rung.model, "key": alias}
                    row.update(self.ledger.remaining(rung.model, alias, rung.rpm, rung.rpd))
                    out.append(row)
        return out
