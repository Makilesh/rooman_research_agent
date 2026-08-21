"""Ladders, the durable limiter, and caching. Fake clock, mocked transport, no network."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from research_agent import db
from research_agent.config import TRAP_MODEL, Config, Rung
from research_agent.llm import (
    DiskCache,
    GeminiProvider,
    LLMClient,
    MalformedResponse,
    OllamaProvider,
    QuotaExhausted,
)

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}


class RecordingGemini(GeminiProvider):
    """Counts transport calls so a test can assert "zero network" literally."""

    def __init__(self, responses=None):
        super().__init__(transport=self._respond)
        self.calls: list[tuple[str, str]] = []
        self._responses = list(responses or [])

    def _respond(self, model, prompt, schema, api_key, timeout_s):
        self.calls.append((model, api_key))
        if self._responses:
            return self._responses.pop(0), 10, 5
        return json.dumps({"answer": f"from {model}"}), 10, 5


def _client(cfg, conn, clock, keys=None, gemini=None, ollama=None) -> LLMClient:
    return LLMClient(
        cfg=cfg, conn=conn, api_keys=keys if keys is not None else {"key_1": "k1"},
        gemini=gemini or RecordingGemini(), ollama=ollama, now=clock,
    )


# ---------------------------------------------------------------------------
#  Ladder configuration
# ---------------------------------------------------------------------------
def test_trap_model_on_the_volume_ladder_fails_config_validation():
    """gemini-2.5-flash-lite is named like a volume model and capped at 20 RPD.

    On the volume ladder it exhausts silently and cascades failures into the agent
    path. The config must refuse to load if anyone moves it.
    """
    bad = Config(volume_ladder=(Rung(TRAP_MODEL, rpm=10, rpd=20),))
    with pytest.raises(ValueError, match="20 RPD"):
        bad.validate()


def test_trap_model_must_remain_on_the_synthesis_ladder():
    bad = Config(synthesis_ladder=(Rung("gemini-3.7-flash", 5, 20),))
    with pytest.raises(ValueError, match="synthesis ladder"):
        bad.validate()


def test_a_model_may_not_sit_on_both_ladders():
    shared = Rung("gemini-3.5-flash-lite", 15, 500)
    bad = Config(
        synthesis_ladder=(shared, Rung(TRAP_MODEL, 10, 20)),
        volume_ladder=(shared,),
    )
    with pytest.raises(ValueError, match="both ladders"):
        bad.validate()


def test_default_ladders_validate():
    Config().validate()  # must not raise


def test_thresholds_guard_blocks_use_before_measurement():
    """The shipped defaults ARE measured (Step 5), but the guard must still bite for
    anyone who resets them -- a zero floor routes everything to ANSWER silently."""
    assert Config().thresholds_are_measured is True
    unmeasured = Config(thresholds_are_measured=False, tau_low=0.0, tau_high=0.0)
    with pytest.raises(RuntimeError, match="Step 5"):
        unmeasured.require_measured_thresholds()


def test_shipped_thresholds_are_internally_consistent():
    cfg = Config()
    cfg.require_measured_thresholds()
    assert 0.0 < cfg.tau_low < cfg.tau_high <= 1.0
    assert 0.0 < cfg.tau_verify < 1.0


def test_measured_thresholds_must_be_ordered():
    with pytest.raises(ValueError, match="tau_low < tau_high"):
        Config(thresholds_are_measured=True, tau_low=0.8, tau_high=0.3).validate()


# ---------------------------------------------------------------------------
#  The limiter
# ---------------------------------------------------------------------------
def test_rpd_exhaustion_steps_down_a_rung_rather_than_failing(cfg, conn, clock):
    """The whole point of the ladder: capacity multiplies across models."""
    gem = RecordingGemini()
    client = _client(cfg, conn, clock, gemini=gem)
    top = cfg.synthesis_ladder[0]

    for i in range(top.rpd):
        clock.advance(minutes=1)  # stay under RPM; RPD is what we are draining
        client.complete(f"prompt {i}", purpose="synthesise", ladder="synthesis",
                        schema=SCHEMA, provider="gemini")
    assert all(m == top.model for m, _ in gem.calls)

    clock.advance(minutes=1)
    out = client.complete("one more", purpose="synthesise", ladder="synthesis",
                          schema=SCHEMA, provider="gemini")
    assert out.model == cfg.synthesis_ladder[1].model, "should step DOWN a rung, not fail"


def test_a_saturated_key_does_not_block_a_free_key(cfg, conn, clock):
    """Rotation must add real capacity, which means never sleeping on a busy key."""
    gem = RecordingGemini()
    client = _client(cfg, conn, clock, keys={"key_1": "k1", "key_2": "k2"}, gemini=gem)
    top = cfg.synthesis_ladder[0]

    for i in range(top.rpm):
        client.complete(f"p{i}", purpose="synthesise", ladder="synthesis",
                        schema=SCHEMA, provider="gemini")
    assert {k for _, k in gem.calls} == {"k1"}, "key_1 should be drained first"

    # No clock advance: key_1 is still saturated on RPM this very second.
    out = client.complete("next", purpose="synthesise", ladder="synthesis",
                          schema=SCHEMA, provider="gemini")
    assert out.key_alias == "key_2"
    assert out.model == top.model, "must stay on the top rung while a key has capacity"


def test_rpm_window_recovers_after_a_minute(cfg, conn, clock):
    gem = RecordingGemini()
    client = _client(cfg, conn, clock, gemini=gem)
    top = cfg.synthesis_ladder[0]
    for i in range(top.rpm):
        client.complete(f"p{i}", purpose="synthesise", ladder="synthesis",
                        schema=SCHEMA, provider="gemini")
    clock.advance(seconds=61)
    out = client.complete("after the window", purpose="synthesise", ladder="synthesis",
                          schema=SCHEMA, provider="gemini")
    assert out.model == top.model


def test_consumed_rpd_survives_a_process_restart(cfg, clock):
    """The test that proves the design.

    A fresh connection to the same file -- which is what a new CLI invocation is --
    must still see today's consumed quota. An in-memory limiter resets here and
    silently enforces nothing.
    """
    conn_a = db.connect(cfg)
    db.migrate(conn_a)
    gem_a = RecordingGemini()
    client_a = _client(cfg, conn_a, clock, gemini=gem_a)
    top = cfg.synthesis_ladder[0]

    for i in range(top.rpd):
        clock.advance(minutes=1)
        client_a.complete(f"p{i}", purpose="synthesise", ladder="synthesis",
                          schema=SCHEMA, provider="gemini")
    conn_a.close()

    # Simulate the process exiting and a new one starting.
    conn_b = db.connect(cfg)
    gem_b = RecordingGemini()
    client_b = _client(cfg, conn_b, clock, gemini=gem_b)
    clock.advance(minutes=1)
    out = client_b.complete("brand new process", purpose="synthesise",
                            ladder="synthesis", schema=SCHEMA, provider="gemini")
    assert out.model != top.model, (
        "a new process must still see the RPD already consumed on the top rung"
    )
    conn_b.close()


def test_rpd_window_rolls_off_after_24h(cfg, conn, clock):
    gem = RecordingGemini()
    client = _client(cfg, conn, clock, gemini=gem)
    top = cfg.synthesis_ladder[0]
    for i in range(top.rpd):
        clock.advance(minutes=1)
        client.complete(f"p{i}", purpose="synthesise", ladder="synthesis",
                        schema=SCHEMA, provider="gemini")
    clock.advance(hours=25)
    out = client.complete("next day", purpose="synthesise", ladder="synthesis",
                          schema=SCHEMA, provider="gemini")
    assert out.model == top.model


def test_full_ladder_exhaustion_raises_a_named_error_not_a_stack_trace(cfg, conn, clock):
    """With the fallback off, a drained ladder raises a named error."""
    tiny = replace(
        cfg,
        synthesis_ladder=(Rung("only-model", rpm=1, rpd=1), Rung(TRAP_MODEL, rpm=1, rpd=1)),
        synthesis_falls_back_to_volume=False,
    )
    client = _client(tiny, conn, clock, gemini=RecordingGemini())
    for p in ("a", "b"):
        client.complete(p, purpose="synthesise", ladder="synthesis",
                        schema=SCHEMA, provider="gemini")

    with pytest.raises(QuotaExhausted) as exc:
        client.complete("c", purpose="synthesise", ladder="synthesis",
                        schema=SCHEMA, provider="gemini")
    assert "synthesis ladder is exhausted" in str(exc.value)
    assert "Ollama path is unlimited" in str(exc.value)


def test_exhausted_synthesis_falls_back_to_the_volume_ladder(cfg, conn, clock):
    """Answer quality degrades last: a weaker cited answer beats no answer.

    When every synthesis rung is drained the call is served by a volume model rather
    than failing the turn.
    """
    tiny = replace(
        cfg,
        synthesis_ladder=(Rung("only-model", rpm=1, rpd=1), Rung(TRAP_MODEL, rpm=1, rpd=1)),
        volume_ladder=(Rung("volume-model", rpm=99, rpd=99),),
    )
    gem = RecordingGemini()
    client = _client(tiny, conn, clock, gemini=gem)
    for p in ("a", "b"):
        client.complete(p, purpose="synthesise", ladder="synthesis",
                        schema=SCHEMA, provider="gemini")

    out = client.complete("c", purpose="synthesise", ladder="synthesis",
                          schema=SCHEMA, provider="gemini")
    assert out.model == "volume-model"
    assert client.degraded_calls == 1

    # The ledger must show the degradation rather than an unexplained volume-model
    # synthesis appearing in the counts.
    purposes = {r["purpose"] for r in
                conn.execute("SELECT purpose FROM llm_calls").fetchall()}
    assert "synthesise:degraded" in purposes


def test_volume_never_falls_back_to_synthesis(cfg, conn, clock):
    """The asymmetry is the whole point of splitting the ladders.

    Judging and condensing on reasoning quota drains the answer budget in about four
    turns, and the failure then lands on answer generation. Volume work fails instead.
    """
    tiny = replace(
        cfg,
        synthesis_ladder=(Rung("big-model", rpm=99, rpd=99), Rung(TRAP_MODEL, 10, 20)),
        volume_ladder=(Rung("small-model", rpm=1, rpd=1),),
    )
    gem = RecordingGemini()
    client = _client(tiny, conn, clock, gemini=gem)
    client.complete("a", purpose="judge", ladder="volume", schema=SCHEMA,
                    provider="gemini")

    with pytest.raises(QuotaExhausted):
        client.complete("b", purpose="judge", ladder="volume", schema=SCHEMA,
                        provider="gemini")
    assert all(m != "big-model" for m, _ in gem.calls), (
        "volume work must never touch a synthesis model"
    )


def test_no_keys_configured_raises_quota_exhausted_not_an_auth_crash(cfg, conn, clock):
    client = _client(cfg, conn, clock, keys={})
    with pytest.raises(QuotaExhausted):
        client.complete("x", purpose="synthesise", ladder="synthesis",
                        schema=SCHEMA, provider="gemini")


def test_volume_and_synthesis_land_on_their_own_ladders(cfg, conn, clock):
    """Judge/condense work must never consume reasoning quota."""
    client = _client(cfg, conn, clock, gemini=RecordingGemini())
    client.complete("judge this", purpose="judge", ladder="volume",
                    schema=SCHEMA, provider="gemini")
    client.complete("synthesise this", purpose="synthesise", ladder="synthesis",
                    schema=SCHEMA, provider="gemini")

    rows = {r["purpose"]: (r["ladder"], r["model"]) for r in
            conn.execute("SELECT purpose, ladder, model FROM llm_calls")}
    assert rows["judge"][0] == "volume"
    assert rows["judge"][1] in {r.model for r in cfg.volume_ladder}
    assert rows["synthesise"][0] == "synthesis"
    assert rows["synthesise"][1] in {r.model for r in cfg.synthesis_ladder}


# ---------------------------------------------------------------------------
#  Caching
# ---------------------------------------------------------------------------
def test_cache_hit_makes_zero_transport_calls_but_still_writes_a_ledger_row(cfg, conn, clock):
    gem = RecordingGemini()
    client = _client(cfg, conn, clock, gemini=gem)
    client.complete("same prompt", purpose="synthesise", ladder="synthesis",
                    schema=SCHEMA, provider="gemini")
    assert len(gem.calls) == 1

    out = client.complete("same prompt", purpose="synthesise", ladder="synthesis",
                          schema=SCHEMA, provider="gemini")
    assert out.cached is True
    assert len(gem.calls) == 1, "a cache hit must not reach the transport"

    cached_rows = conn.execute("SELECT COUNT(*) FROM llm_calls WHERE cached = 1").fetchone()[0]
    assert cached_rows == 1, "a cache hit is still an event worth recording"


def test_cached_calls_do_not_consume_quota(cfg, conn, clock):
    gem = RecordingGemini()
    client = _client(cfg, conn, clock, gemini=gem)
    top = cfg.synthesis_ladder[0]

    client.complete("p", purpose="synthesise", ladder="synthesis", schema=SCHEMA, provider="gemini")
    for _ in range(top.rpd * 2):
        client.complete("p", purpose="synthesise", ladder="synthesis",
                        schema=SCHEMA, provider="gemini")

    used_rpm, used_rpd = client.ledger.used(top.model, "key_1")
    assert used_rpd == 1, "only the single uncached call may count against quota"


def test_cache_key_separates_models_and_params():
    a = DiskCache.key("m1", "p", None)
    assert a != DiskCache.key("m2", "p", None)
    assert a != DiskCache.key("m1", "p2", None)
    assert a != DiskCache.key("m1", "p", {"temperature": 0.1})
    assert a == DiskCache.key("m1", "p", {})


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(cfg, conn, clock):
    cache = DiskCache(cfg.llm_cache_dir)
    key = DiskCache.key("m", "p", None)
    path = cache._path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert cache.get(key) is None


# ---------------------------------------------------------------------------
#  Schema-constrained output
# ---------------------------------------------------------------------------
def test_malformed_json_retries_exactly_once_then_raises(cfg, conn, clock):
    gem = RecordingGemini(responses=["not json at all", "still not json"])
    client = _client(cfg, conn, clock, gemini=gem)
    with pytest.raises(MalformedResponse):
        client.complete("p", purpose="synthesise", ladder="synthesis",
                        schema=SCHEMA, provider="gemini")
    assert len(gem.calls) == 2, "one attempt plus exactly one retry"

    rows = conn.execute("SELECT purpose, ok FROM llm_calls ORDER BY id").fetchall()
    assert [r["purpose"] for r in rows] == ["synthesise", "synthesise:retry"]
    assert all(r["ok"] == 0 for r in rows), "both attempts are recorded as failures"


def test_a_retry_that_succeeds_returns_parsed_data(cfg, conn, clock):
    gem = RecordingGemini(responses=["oops", json.dumps({"answer": "recovered"})])
    client = _client(cfg, conn, clock, gemini=gem)
    out = client.complete("p", purpose="synthesise", ladder="synthesis",
                          schema=SCHEMA, provider="gemini")
    assert out.data == {"answer": "recovered"}
    assert len(gem.calls) == 2


def test_schema_is_passed_through_to_the_provider(cfg, conn, clock):
    seen: dict = {}

    def transport(model, prompt, schema, api_key, timeout_s):
        seen["schema"] = schema
        return json.dumps({"answer": "ok"}), 1, 1

    client = _client(cfg, conn, clock, gemini=GeminiProvider(transport=transport))
    client.complete("p", purpose="synthesise", ladder="synthesis",
                    schema=SCHEMA, provider="gemini")
    assert seen["schema"] == SCHEMA, (
        "the citation contract is enforced by constrained decoding, not by hope"
    )


def test_ollama_passes_schema_as_format_and_needs_no_key(cfg, conn, clock):
    seen: dict = {}

    def transport(path, payload, timeout_s):
        seen.update(path=path, payload=payload)
        return {"response": json.dumps({"answer": "local"}), "prompt_eval_count": 7,
                "eval_count": 3}

    client = LLMClient(cfg=cfg, conn=conn, api_keys={},
                       ollama=OllamaProvider(cfg.ollama_host, transport=transport), now=clock)
    out = client.complete("p", purpose="synthesise", ladder="synthesis", schema=SCHEMA)

    assert seen["path"] == "/api/generate"
    assert seen["payload"]["format"] == SCHEMA
    assert seen["payload"]["stream"] is False
    assert out.data == {"answer": "local"}
    assert out.provider == "ollama"
    assert out.prompt_tokens == 7


def test_ollama_calls_are_ledgered_so_calls_per_turn_is_measurable_locally(cfg, conn, clock):
    def transport(path, payload, timeout_s):
        return {"response": json.dumps({"answer": "x"})}

    client = LLMClient(cfg=cfg, conn=conn, api_keys={},
                       ollama=OllamaProvider(cfg.ollama_host, transport=transport), now=clock)
    client.complete("a", purpose="condense", ladder="volume", schema=SCHEMA)
    client.complete("b", purpose="synthesise", ladder="synthesis", schema=SCHEMA)

    rows = conn.execute("SELECT provider, purpose, ladder FROM llm_calls ORDER BY id").fetchall()
    assert [r["purpose"] for r in rows] == ["condense", "synthesise"]
    assert all(r["provider"] == "ollama" for r in rows)


def test_offload_mode_selects_the_larger_local_model(cfg, conn, clock):
    seen: dict = {}

    def transport(path, payload, timeout_s):
        seen["model"] = payload["model"]
        return {"response": json.dumps({"answer": "x"})}

    seq = replace(cfg, offload_mode="sequential")
    client = LLMClient(cfg=seq, conn=conn, api_keys={},
                       ollama=OllamaProvider(seq.ollama_host, transport=transport), now=clock)
    client.complete("p", purpose="synthesise", ladder="synthesis", schema=SCHEMA)
    assert seen["model"] == seq.ollama_model_large


# ---------------------------------------------------------------------------
#  Keys and reporting
# ---------------------------------------------------------------------------
def test_missing_and_blank_keys_are_absent_not_errors(cfg):
    keys = LLMClient.load_keys(cfg, {"GEMINI_API_KEY_1": "abc", "GEMINI_API_KEY_2": "   "})
    assert list(keys.values()) == ["abc"], (
        "a blank key is not a key; the Ollama path needs none")


def test_key_alias_follows_the_key_not_its_position(cfg):
    """Quota is accounted per (model, key_alias). A positional alias hands a key's
    consumption to whatever later occupies its slot -- observed live, where two fresh
    keys inherited 20/20 and 18/20 RPD from their predecessors."""
    first = LLMClient.load_keys(cfg, {"GEMINI_API_KEY_1": "AAA",
                                      "GEMINI_API_KEY_2": "BBB"})
    swapped = LLMClient.load_keys(cfg, {"GEMINI_API_KEY_1": "BBB",
                                        "GEMINI_API_KEY_2": "AAA"})
    assert set(first) == set(swapped), "moving a key between slots must not rename it"

    replaced = LLMClient.load_keys(cfg, {"GEMINI_API_KEY_1": "AAA",
                                         "GEMINI_API_KEY_2": "CCC"})
    assert set(replaced) != set(first), "a genuinely new key must get a new identity"


def test_key_alias_never_contains_the_key(cfg):
    secret = "AQ.Ab8-super-secret-value"
    alias = LLMClient.key_alias(secret)
    assert secret not in alias
    assert alias.startswith("k_") and len(alias) == 12


def test_duplicate_keys_collapse_to_one_bucket(cfg):
    """The same key in two slots shares one quota bucket; counting it twice would
    double the apparent capacity."""
    keys = LLMClient.load_keys(cfg, {"GEMINI_API_KEY_1": "same",
                                     "GEMINI_API_KEY_2": "same"})
    assert len(keys) == 1


def test_a_rate_limited_key_rotates_rather_than_aborting(cfg, conn, clock):
    """A 429 means the provider disagrees with the local ledger. Trust the provider
    and rotate; aborting would waste the keys that still have capacity."""
    def transport(model, prompt, schema, api_key, timeout_s):
        if api_key == "spent":
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
        return json.dumps({"answer": "ok"}), 5, 5

    client = LLMClient(cfg=cfg, conn=conn,
                       api_keys={"k_spent": "spent", "k_fresh": "fresh"},
                       gemini=GeminiProvider(transport=transport), now=clock)
    out = client.complete("p", purpose="synthesise", ladder="synthesis",
                          schema=SCHEMA, provider="gemini")
    assert out.key_alias == "k_fresh"
    assert "k_spent" not in client.disabled_keys, (
        "rate limiting is temporary; the key must not be permanently disabled")


def test_budget_reports_every_rung_and_reflects_consumption(cfg, conn, clock):
    client = _client(cfg, conn, clock, gemini=RecordingGemini())
    rows = client.budget()
    expected = len(cfg.synthesis_ladder) + len(cfg.volume_ladder)
    assert len(rows) == expected

    top = cfg.synthesis_ladder[0]
    client.complete("p", purpose="synthesise", ladder="synthesis", schema=SCHEMA, provider="gemini")
    after = {(r["model"], r["key"]): r for r in client.budget()}[(top.model, "key_1")]
    assert after["rpd_used"] == 1
    assert after["rpd_left"] == top.rpd - 1


# ---------------------------------------------------------------------------
#  Permanently rejected keys
# ---------------------------------------------------------------------------
def test_a_permanently_rejected_key_is_skipped_not_fatal(cfg, conn, clock):
    """Observed live: two of four real keys returned 403 PERMISSION_DENIED.

    A 403 is not quota exhaustion -- waiting never fixes it. Re-raising would abort
    the turn while working keys sat idle; retrying would loop forever. The key is
    disabled for the process and rotation continues.
    """
    def transport(model, prompt, schema, api_key, timeout_s):
        if api_key in {"dead1", "dead2"}:
            raise RuntimeError("403 PERMISSION_DENIED. Your project has been "
                               "denied access. Please contact support.")
        return json.dumps({"answer": "from the working key"}), 5, 5

    client = LLMClient(
        cfg=cfg, conn=conn,
        api_keys={"key_1": "dead1", "key_2": "dead2", "key_3": "alive"},
        gemini=GeminiProvider(transport=transport), now=clock,
    )
    out = client.complete("p", purpose="synthesise", ladder="synthesis",
                          schema=SCHEMA, provider="gemini")
    assert out.key_alias == "key_3"
    assert set(client.disabled_keys) == {"key_1", "key_2"}


def test_a_disabled_key_is_not_retried_on_later_calls(cfg, conn, clock):
    attempts: list[str] = []

    def transport(model, prompt, schema, api_key, timeout_s):
        attempts.append(api_key)
        if api_key == "dead1":
            raise RuntimeError("API_KEY_INVALID")
        return json.dumps({"answer": "ok"}), 5, 5

    client = LLMClient(cfg=cfg, conn=conn,
                       api_keys={"key_1": "dead1", "key_2": "alive"},
                       gemini=GeminiProvider(transport=transport), now=clock)
    for i in range(3):
        client.complete(f"p{i}", purpose="synthesise", ladder="synthesis",
                        schema=SCHEMA, provider="gemini")
    assert attempts.count("dead1") == 1, "a dead key must be tried once, not every time"


def test_quota_exhaustion_and_key_rejection_are_different(cfg, conn, clock):
    """A saturated key recovers on its own; a rejected one never does. Conflating
    them would either retry forever or give up while capacity remains."""
    from research_agent.llm import is_permanent_key_error

    assert is_permanent_key_error(RuntimeError("403 PERMISSION_DENIED"))
    assert is_permanent_key_error(RuntimeError("API key not valid"))
    assert not is_permanent_key_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not is_permanent_key_error(RuntimeError("503 Service Unavailable"))


# ---------------------------------------------------------------------------
#  Tokens per minute — the third published limit
# ---------------------------------------------------------------------------
def test_tpm_blocks_a_call_that_would_exceed_the_token_ceiling(cfg, conn, clock):
    """RPM usually binds first, which is exactly why TPM is easy to forget.

    Five requests a minute of 8k-token prompts is 40k against a 250k ceiling. It stops
    being irrelevant the moment prompts grow or a 15 RPM volume model carries long
    context.
    """
    tiny = replace(cfg, volume_ladder=(Rung("m", rpm=99, rpd=99, tpm=1000),),
                   synthesis_falls_back_to_volume=False)
    client = _client(tiny, conn, clock, gemini=RecordingGemini())

    # A first call that reports 900 tokens spent.
    row = client.ledger.try_acquire("m", "key_1", 99, 99, "p", "sha", "gemini",
                                    "volume", tpm=1000, est_tokens=10)
    client.ledger.finish(row, ok=True, latency_ms=1, prompt_tokens=800,
                         completion_tokens=100)
    assert client.ledger.tokens_used("m", "key_1") == 900

    # A second call estimated at 200 tokens would exceed 1000 and is refused.
    assert client.ledger.try_acquire("m", "key_1", 99, 99, "p", "sha2", "gemini",
                                     "volume", tpm=1000, est_tokens=200) is None
    # A small one still fits.
    assert client.ledger.try_acquire("m", "key_1", 99, 99, "p", "sha3", "gemini",
                                     "volume", tpm=1000, est_tokens=50) is not None


def test_cached_calls_do_not_consume_tokens(cfg, conn, clock):
    client = _client(cfg, conn, clock, gemini=RecordingGemini())
    client.ledger.record_cached("gemini", "m", "key_1", "volume", "p", "sha")
    assert client.ledger.tokens_used("m", "key_1") == 0


def test_tpm_window_rolls_off(cfg, conn, clock):
    client = _client(cfg, conn, clock, gemini=RecordingGemini())
    row = client.ledger.try_acquire("m", "key_1", 99, 99, "p", "sha", "gemini",
                                    "volume", tpm=1000, est_tokens=10)
    client.ledger.finish(row, ok=True, latency_ms=1, prompt_tokens=900,
                         completion_tokens=0)
    assert client.ledger.tokens_used("m", "key_1") == 900
    clock.advance(seconds=61)
    assert client.ledger.tokens_used("m", "key_1") == 0


def test_every_ladder_rung_declares_all_three_limits():
    """A rung missing a limit is a rung the limiter cannot fully enforce."""
    cfg = Config()
    for name in ("synthesis", "volume"):
        for rung in cfg.ladder(name):
            assert rung.rpm > 0 and rung.rpd > 0 and rung.tpm > 0, rung.model


def test_no_access_models_are_absent_from_both_ladders():
    """Pro models are published at 0/0/0 on this tier. A rung the account cannot use
    is not a fallback, it is a guaranteed failed attempt on the way down."""
    from research_agent.config import NO_ACCESS_MODELS

    cfg = Config()
    listed = {r.model for name in ("synthesis", "volume") for r in cfg.ladder(name)}
    assert not (listed & NO_ACCESS_MODELS)
