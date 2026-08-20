"""The only place in this codebase where a number is allowed to live.

Every threshold, top-k, chunk size, timeout, model name and ladder rung is a named
field here. If a magic number appears anywhere else, that is a bug -- Step 0's VERIFY
greps for exactly that.

Config is frozen: nothing mutates it at runtime. Values come from the environment
(via `.env` when present) so a reviewer can retune without editing source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Everything is anchored to the repo root so the CLI behaves identically from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
#  Gemini ladders (spec section 7).
#
#  RPD is per-model, so rotating the rung multiplies daily capacity. Two ladders
#  exist because a conversational turn costs ~1 synthesis call plus 2-4 volume
#  calls (condensation, sufficiency judge, optional rewrite/decomposition).
#  Routing volume work through synthesis models exhausts reasoning quota in about
#  four turns; separating them means answer quality degrades last.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Rung:
    """One model on one ladder, with its published free-tier limits."""

    model: str
    rpm: int
    rpd: int


Ladder = Literal["synthesis", "volume"]

# Capability-ordered. Drained top to bottom, across every key on a rung before
# stepping down (spec section 7.2a).
SYNTHESIS_LADDER: tuple[Rung, ...] = (
    Rung("gemini-3.7-flash", rpm=5, rpd=20),
    Rung("gemini-3.6-flash", rpm=5, rpd=20),
    Rung("gemini-3.5-flash", rpm=5, rpd=20),
    Rung("gemini-3-flash-preview", rpm=5, rpd=20),
    Rung("gemini-2.5-flash", rpm=5, rpd=20),
    # Named like a volume model, capped at 20 RPD. It belongs at the BOTTOM of the
    # synthesis ladder and nowhere else: on the volume ladder it exhausts silently
    # and cascades failures into the agent path (spec section 7.2b). Config.validate()
    # enforces this; tests/test_llm.py fails if anyone moves it.
    Rung("gemini-2.5-flash-lite", rpm=10, rpd=20),
)

# Capacity-ordered, not capability-ordered. Serves condensation, sufficiency
# judging, query rewriting and decomposition.
VOLUME_LADDER: tuple[Rung, ...] = (
    Rung("gemini-3.5-flash-lite", rpm=15, rpd=500),
    Rung("gemini-3.1-flash-lite", rpm=15, rpd=500),
)

# Named so the assertion below reads as intent rather than a string literal
# floating inside a guard clause.
TRAP_MODEL = "gemini-2.5-flash-lite"


@dataclass(frozen=True, slots=True)
class Config:
    # -- Providers ----------------------------------------------------------
    provider: Literal["ollama", "gemini", "auto"] = "ollama"
    ollama_host: str = "http://127.0.0.1:11434"
    # Coresident 8B-class: fits the ~7 GB left after both encoders (section 6.1).
    ollama_model: str = "llama3.1:8b"
    # Sequential 14B-class: only loadable once the encoders are freed.
    ollama_model_large: str = "qwen2.5:14b"
    ollama_timeout_s: float = 300.0
    # Ollama's default context window is 4096 tokens and it TRUNCATES SILENTLY above
    # it -- no error, no warning, just an answer written from half the sources. The
    # context here is ~6 parent passages, so this must be set explicitly and the
    # actual prompt_eval_count checked against it on every call.
    ollama_num_ctx: int = 8192
    # Keep the model resident between calls. With the encoders holding ~2.2 GB,
    # Ollama's scheduler otherwise evicts and reloads a 5.5 GB model repeatedly, and
    # a reload costs ~30s of pure disk I/O per turn.
    ollama_keep_alive: str = "15m"
    gemini_key_env_prefix: str = "GEMINI_API_KEY_"
    gemini_max_keys: int = 3
    gemini_timeout_s: float = 60.0

    # -- Hardware (section 6.1) ---------------------------------------------
    # coresident: encoders stay hot on GPU, local LLM must fit in what is left.
    # sequential: free encoders after retrieval, then call a 14B-class model.
    offload_mode: Literal["coresident", "sequential"] = "coresident"
    embed_device: str = "cuda"
    # Square matrix edge for doctor's fp16 smoke test. Large enough to be a
    # genuine kernel launch, small enough to never pressure VRAM.
    gpu_probe_dim: int = 2048
    embed_dtype: Literal["float16", "float32"] = "float16"
    require_cuda: bool = True

    # -- Encoders -----------------------------------------------------------
    embed_model: str = "BAAI/bge-m3"
    embed_dims: int = 1024
    embed_batch_size: int = 4
    embed_max_tokens: int = 1024
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_batch_size: int = 16
    rerank_max_tokens: int = 1024

    # -- Chunking (section 4.2) ---------------------------------------------
    child_tokens: int = 700
    child_overlap_ratio: float = 0.15
    parent_tokens: int = 2000
    min_chunk_tokens: int = 50

    # -- Retrieval ----------------------------------------------------------
    dense_top_k: int = 25
    sparse_top_k: int = 25
    rrf_k: int = 60
    rerank_candidates: int = 25
    context_top_n: int = 6
    # Token budget for the SOURCES block. Parent passages run ~2000 tokens each, so
    # six of them overflow an 8192 window before the prompt template is even added.
    # Passages are dropped whole from the bottom of the ranking rather than being
    # text-truncated: a half-present passage would break the verifier's guarantee
    # that it scores against exactly what the synthesiser saw, and could let the
    # model cite text it never received.
    context_max_tokens: int = 5200

    # -- Thresholds ---------------------------------------------------------
    # MEASURED 2026-08-20 against corpus e62c6925a03ad297 by `eval-retrieval`.
    # Full evidence -- both score histograms, all four population summaries, and the
    # sensitivity sweep -- is in outputs/eval_report.md.
    #
    # These are defaults rather than env-only values so a clean clone answers a
    # question with no .env file at all. They are facts measured on this corpus, and
    # this file is where measured numbers belong. Re-measure if the corpus changes:
    # `require_measured_thresholds()` still guards every caller, and the flag below
    # is what a re-measurement would flip.
    #
    # Honest caveat, repeated in the README: these are tuned on the same question set
    # the routing is later scored against, not on held-out data (decisions.md D-113).
    thresholds_are_measured: bool = True
    tau_low: float = 0.8       # below this, refuse
    tau_high: float = 0.99     # above this, answer directly; between the two, ask
    tau_verify: float = 0.378  # groundedness floor for one cited sentence

    # -- Conversation (section 4.1) -----------------------------------------
    history_max_turns: int = 6
    history_max_tokens: int = 2000
    condense_timeout_s: float = 3.0
    condense_skip_first_turn: bool = True
    semantic_cache_threshold: float = 0.97

    # -- Agency (Step 11) ---------------------------------------------------
    max_retrieval_loops: int = 2
    max_subquestions: int = 4
    schema_retry_attempts: int = 1

    # -- Paths --------------------------------------------------------------
    repo_root: Path = field(default=REPO_ROOT)
    db_path: Path = field(default=REPO_ROOT / "storage" / "agent.db")
    embeddings_path: Path = field(default=REPO_ROOT / "storage" / "embeddings.npy")
    llm_cache_dir: Path = field(default=REPO_ROOT / ".cache" / "llm")
    sources_dir: Path = field(default=REPO_ROOT / "data" / "sources")
    manifest_path: Path = field(default=REPO_ROOT / "data" / "manifest.yaml")
    questions_path: Path = field(default=REPO_ROOT / "data" / "questions.yaml")
    conversations_path: Path = field(default=REPO_ROOT / "data" / "conversations.yaml")
    outputs_dir: Path = field(default=REPO_ROOT / "outputs")

    # -- Corpus fetch (Step 2) ----------------------------------------------
    arxiv_delay_s: float = 3.0
    arxiv_user_agent: str = (
        "cited-research-agent/0.1 (research prototype; contact via repository issues)"
    )
    fetch_timeout_s: float = 120.0

    # -- Ops ----------------------------------------------------------------
    log_level: str = "INFO"
    sqlite_busy_timeout_ms: int = 5000

    # -- Ladders as data, not code (Step 1) ---------------------------------
    synthesis_ladder: tuple[Rung, ...] = field(default=SYNTHESIS_LADDER)
    volume_ladder: tuple[Rung, ...] = field(default=VOLUME_LADDER)

    # -----------------------------------------------------------------------
    @classmethod
    def load(cls, **overrides: object) -> "Config":
        """Build a Config from the environment, then validate its invariants."""
        load_dotenv(REPO_ROOT / ".env", override=False)

        # `slots=True` replaces class attributes with slot descriptors, so `cls.field`
        # is NOT the default value. Read defaults off the dataclass metadata instead.
        defaults = {f.name: f.default for f in fields(cls)}

        def env_str(env_name: str, field_name: str) -> str:
            return os.getenv(env_name, defaults[field_name])  # type: ignore[arg-type]

        def env_float(env_name: str, field_name: str) -> float:
            raw = os.getenv(env_name)
            return defaults[field_name] if raw is None else float(raw)  # type: ignore[return-value]

        def env_int(env_name: str, field_name: str) -> int:
            raw = os.getenv(env_name)
            return defaults[field_name] if raw is None else int(raw)  # type: ignore[return-value]

        def env_bool(env_name: str, field_name: str) -> bool:
            raw = os.getenv(env_name)
            if raw is None:
                return defaults[field_name]  # type: ignore[return-value]
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        cfg = cls(
            provider=env_str("PROVIDER", "provider"),  # type: ignore[arg-type]
            ollama_host=env_str("OLLAMA_HOST", "ollama_host"),
            ollama_model=env_str("OLLAMA_MODEL", "ollama_model"),
            ollama_model_large=env_str("OLLAMA_MODEL_LARGE", "ollama_model_large"),
            offload_mode=env_str("OFFLOAD_MODE", "offload_mode"),  # type: ignore[arg-type]
            embed_device=env_str("EMBED_DEVICE", "embed_device"),
            require_cuda=env_bool("REQUIRE_CUDA", "require_cuda"),
            tau_high=env_float("TAU_HIGH", "tau_high"),
            tau_low=env_float("TAU_LOW", "tau_low"),
            tau_verify=env_float("TAU_VERIFY", "tau_verify"),
            thresholds_are_measured=env_bool(
                "THRESHOLDS_ARE_MEASURED", "thresholds_are_measured"
            ),
            log_level=env_str("LOG_LEVEL", "log_level"),
            gemini_max_keys=env_int("GEMINI_MAX_KEYS", "gemini_max_keys"),
            **overrides,  # type: ignore[arg-type]
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Invariants that must hold for the design to be what it claims to be."""
        volume_models = {r.model for r in self.volume_ladder}
        synthesis_models = {r.model for r in self.synthesis_ladder}

        # Section 7.2b -- the assertion the spec asks for by name.
        if TRAP_MODEL in volume_models:
            raise ValueError(
                f"{TRAP_MODEL} is capped at 20 RPD despite its name. It belongs at the "
                "bottom of the SYNTHESIS ladder. On the volume ladder it exhausts "
                "silently and cascades failures into the agent path (spec 7.2b)."
            )
        if TRAP_MODEL not in synthesis_models:
            raise ValueError(f"{TRAP_MODEL} must remain on the synthesis ladder.")
        overlap = volume_models & synthesis_models
        if overlap:
            raise ValueError(
                f"A model may not sit on both ladders: {sorted(overlap)}. Shared rungs "
                "make the two quota pools indistinguishable in the ledger."
            )
        if self.offload_mode not in {"coresident", "sequential"}:
            raise ValueError(
                f"offload_mode must be coresident|sequential, got {self.offload_mode!r}"
            )
        if self.provider not in {"ollama", "gemini", "auto"}:
            raise ValueError(f"provider must be ollama|gemini|auto, got {self.provider!r}")
        if self.thresholds_are_measured and not (0.0 < self.tau_low < self.tau_high <= 1.0):
            raise ValueError(
                "Measured thresholds must satisfy 0 < tau_low < tau_high <= 1; got "
                f"tau_low={self.tau_low}, tau_high={self.tau_high}"
            )

    def require_measured_thresholds(self) -> None:
        """Guard for any caller that routes or verifies against a threshold."""
        if not self.thresholds_are_measured:
            raise RuntimeError(
                "Thresholds have not been measured yet (Step 5). Routing against the "
                "placeholder values would send every query to ANSWER. Derive them from "
                "the rerank score distribution, then set TAU_LOW / TAU_HIGH / "
                "TAU_VERIFY and THRESHOLDS_ARE_MEASURED=true."
            )

    def ladder(self, which: Ladder) -> tuple[Rung, ...]:
        return self.synthesis_ladder if which == "synthesis" else self.volume_ladder

    def as_dict(self) -> dict[str, object]:
        """Serialisable view, for `doctor` and for the Step 0 VERIFY dump."""
        out: dict[str, object] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Path):
                out[f.name] = str(v)
            elif isinstance(v, tuple) and v and isinstance(v[0], Rung):
                out[f.name] = [{"model": r.model, "rpm": r.rpm, "rpd": r.rpd} for r in v]
            else:
                out[f.name] = v
        return out
