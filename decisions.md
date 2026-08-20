# Design decisions

A running ADR log, appended at the moment each choice is made rather than
reconstructed at the end. Reconstructed rationales are always thinner than real
ones, and reviewers can tell.

Every entry carries a **README line** field. The README's tradeoff section is
assembled from those lines at Step 13, so it writes itself instead of being
invented after the fact.

**Numbering.** `D-0xx` are the decisions the build spec anticipated and are kept at
the numbers it assigns them. `D-1xx` are decisions that came up during the build
which the spec did not foresee. Both are chronological within their block.

---

## D-001 · No vector database, no service stack
**Date:** 2026-08-20
**Context:** I have shipped Qdrant (hybrid dense+sparse, M&A due-diligence engine),
Milvus, Chroma and FAISS in production, plus Redis and Postgres as supporting
services. The obvious move here is to reach for the same stack.
**Options:** (a) Qdrant + Postgres + Docker, as in the M&A engine; (b) Chroma +
Redis + FastAPI, as in the Opkey assistant; (c) in-process retrieval over a single
SQLite file.
**Decision:** (c). One file, no daemon, no container, no compose.
**Why:** Two independent reasons.
  1. *A reviewer with ten minutes cannot stand up a service stack.* 30 of the 100
     points are "working end-to-end agent," and every install step is a chance to
     lose that block. Setup simplicity beats architectural sophistication here.
  2. *At this corpus size in-process search is not an approximation of Qdrant --
     it is exact, and HNSW is the approximation.* 12-13 papers is on the order of
     3-6k chunks. A float32 dot product over 6k x 1024 is a few milliseconds. An
     ANN index would trade exactness for a speedup we cannot perceive.
**Consequence:** No horizontal scale, no concurrent writers, no filtered-ANN. The
flat scan starts to hurt somewhere north of ~10k chunks; that ceiling goes in the
README's limitations section rather than being hidden.
**Evidence:** To be measured -- retrieval latency lands in `outputs/eval_report.md`
at Step 6. Until then, TBD.
**Revisit if:** the corpus grows past ~10k chunks, or more than one process needs
concurrent write access.
**README line:** "I have shipped Qdrant, Milvus, Chroma, Redis and Postgres in
production and deliberately chose a single SQLite file here: at 3-6k chunks exact
search is faster than the reviewer's time cost of standing up a service."

---

## D-002 · SQLite as the persistence layer, not an implementation detail
**Date:** 2026-08-20
**Context:** A CLI could hold sessions, citations and rate-limit counters in
memory and write nothing. Most submissions will.
**Options:** (a) stateless, in-memory only; (b) JSON files on disk; (c) SQLite
with a real relational schema.
**Decision:** (c), with a migrated schema in `db.py` covering corpus, conversation,
ops and evaluation.
**Why:** Three reasons, in order of how load-bearing they are.
  1. **The quota ledger is the strongest one.** Gemini's RPD is a *daily* limit but
     the CLI is a short-lived process. An in-memory limiter resets on every single
     invocation, which means it enforces nothing -- you burn a 20-request daily cap
     without ever seeing a warning. Deriving RPM and RPD by windowed `COUNT(*)`
     over durable `llm_calls` rows makes the limiter correct across restarts.
     Without persistence the entire two-ladder design is theatre.
  2. **Citations must outlive the turn.** A follow-up like "expand on the second
     source" is answerable only because `turn_citations` is a queryable relation.
     This is exactly what inline prose markers cannot do, and it is the concrete
     payoff of the structured citation contract.
  3. **Evaluation runs must be comparable over time.** `eval_runs` / `eval_results`
     let the ablation table be regenerated and diffed rather than re-derived by hand.
**Consequence:** A migration path to maintain, and single-writer semantics. WAL mode
gives concurrent readers, which is all a single-user CLI needs.
**Evidence:** The restart-persistence test in `tests/test_llm.py` (Step 1) is the
one that proves reason 1 -- a new connection to the same file still sees today's
consumed RPD.
**Revisit if:** the agent ever becomes multi-user or multi-process, at which point
Postgres is the right answer and the schema ports largely unchanged.
**README line:** "The quota ledger is why there is a database: a daily rate limit
cannot be enforced from a process that exits between every request."

---

## D-101 · `pyproject.toml` added, so `python -m research_agent` actually works
**Date:** 2026-08-20
**Context:** The specified layout puts the package at `src/research_agent/` but
gives no packaging file. `src/` is not on `sys.path`, so every documented command
in the spec -- starting with Step 0's own VERIFY -- fails in a fresh venv with
`ModuleNotFoundError`.
**Options:** (a) set `PYTHONPATH=src` in the Makefile and in every README command;
(b) move the package to the repo root, abandoning the src layout; (c) add a minimal
`pyproject.toml` and `pip install -e .`.
**Decision:** (c). One file outside the specified layout, flagged to the user.
**Why:** (a) leaks an environment variable into every copy-pasteable README command
and breaks the moment a reviewer opens a new shell -- a guaranteed clean-clone-test
failure, and the clean-clone test is the Step 13 gate. (b) discards the src layout's
main benefit: tests import the installed package rather than accidentally importing
the working directory. (c) costs one small file and makes `pip install -e .` the
single documented setup step.
**Consequence:** Setup is two commands (`pip install -r requirements.txt` then
`pip install -e .`) rather than one. Both are in the README quickstart and in
`make setup`. Runtime dependencies stay in `requirements.txt` only -- `pyproject`
declares none, because two sources of truth for a version is how a pin drifts.
**Evidence:** Step 0 VERIFY: the config dump runs from a fresh venv with no
`PYTHONPATH` set.
**Revisit if:** never, realistically.
**README line:** "Installed as a package (`pip install -e .`) so every documented
command works from any directory with no PYTHONPATH juggling."

---

## D-102 · Thresholds ship as unusable placeholders behind a hard guard
**Date:** 2026-08-20
**Context:** `tau_high`, `tau_low` and `tau_verify` are measured from the rerank
score distribution in Step 5. Steps 0-4 exist before that measurement.
**Options:** (a) seed them with plausible-looking defaults; (b) carry over the
values from my M&A engine; (c) ship zeros behind a guard that refuses to run.
**Decision:** (c). `Config.thresholds_are_measured` defaults false, and
`require_measured_thresholds()` raises for any caller that routes or verifies.
**Why:** (a) and (b) both fail *silently and confidently*, which is the worst
failure shape in this project. A zero or borrowed `tau_low` routes every query to
ANSWER, including the ones that must refuse -- and the abstention controls are a
scored deliverable. Different corpus means a different score distribution; the spec
is explicit that numbers must not be carried over. A guard that raises turns an
invisible wrong answer into a visible error.
**Consequence:** Nothing in the routing or verification path can be exercised before
Step 5 completes. That is the intended coupling, not an inconvenience.
**Evidence:** The measured values and the histogram that produced them land in
`decisions.md` D-015 and `outputs/eval_report.md` at Step 5. Currently TBD.
**Revisit if:** the corpus changes, in which case the measurement is redone rather
than the numbers being adjusted by hand.
**README line:** "Routing thresholds are measured from the observed rerank score
distribution, not guessed and not carried over from a previous corpus -- the code
refuses to route until they have been measured."

---

## D-103 · FTS5 query text must be escaped; tokenisation is the real BM25 tradeoff
**Date:** 2026-08-20
**Context:** The spec records FTS5's fixed `k1=1.2 / b=0.75` as the tradeoff against
a Python BM25 library. Verified locally that FTS5 with `porter unicode61` is present
in this Python's bundled SQLite (3.49.1) and that `bm25()` and porter stemming both
work. But the fixed k1/b is not the tradeoff that will actually cost accuracy here.
**Options:** (a) `bm25s`; (b) `rank_bm25`; (c) SQLite FTS5.
**Decision:** (c), keeping the spec's choice -- but for the dependency and
transactionality reasons, not the k1/b one, and with two mitigations attached.
**Why:** k1/b would never have been tuned on a 3-6k-chunk corpus, so giving them up
costs nothing real. The tradeoff that does bite is **tokenisation**: `unicode61`
splits on non-alphanumerics, so `GPT-3` becomes `gpt` + `3`, `NF4` becomes `nf` +
`4`, `4-bit` becomes `4` + `bit`. On a corpus chosen precisely for checkable
specifics, that erodes exactly the terms the single-hop questions turn on -- and
`bm25s` tokenises the same way, so switching would not fix it. Two consequences are
therefore mandatory rather than optional:
  1. Multi-token technical terms are emitted as FTS5 phrase queries.
  2. User text is escaped before it reaches FTS5. An unescaped `"`, `*`, `-` or a
     bare `NEAR` raises `sqlite3.OperationalError` mid-turn -- a live crash path,
     not a theoretical one.
**Consequence:** Sparse retrieval carries a known precision ceiling on hyphenated
and alphanumeric identifiers. Step 6's ablation measures it rather than assuming it.
**Evidence:** Local probe, 2026-08-20: SQLite 3.49.1, FTS5 available, `porter`
stemming confirmed (`MATCH 'run'` retrieved "running quickly"). Ablation numbers TBD
at Step 6.
**Revisit if:** the sparse-only ablation row underperforms enough to matter, in
which case a custom tokeniser or a trigram fallback is the next move -- not `bm25s`.
**README line:** "Sparse retrieval is SQLite FTS5 BM25: real BM25 in the same file
and transaction as the chunk metadata, with zero extra dependencies. Its real cost
is not the fixed k1/b but unicode61 tokenisation splitting identifiers like GPT-3."

---

## D-104 · Exact transitive pins, and what upstream forced
**Date:** 2026-08-20
**Context:** The spec requires every package pinned to an exact version.
**Decision:** Direct *and* load-bearing transitive dependencies (`tokenizers`,
`safetensors`, `huggingface-hub`) are pinned exactly, and the PyTorch CUDA wheel
index is declared in `requirements.txt` as `--extra-index-url` so the whole install
is one command.
**Why:** Pinning transitives is what makes the install genuinely reproducible rather
than reproducible-until-upstream-releases. Declaring the index inline means a
reviewer runs `pip install -r requirements.txt` and gets a CUDA build, instead of
following prose and getting a CPU build.
**Consequence, discovered during Step 0:** `transformers==5.15.1` caps
`tokenizers<=0.23.0`, but no `0.23.0` final was ever released -- only `0.23.0rc0`
and `0.23.1`. The cap therefore admits only the `0.22.x` line, so the pin is
`tokenizers==0.22.2`. This cost two failed installs and is exactly the class of
breakage that exact pins exist to freeze in place once solved.
Second consequence: `--extra-index-url` lets pip consider both indexes for every
package. With everything exactly pinned the version resolved is unambiguous, though
a wheel may be served from either host.
**Evidence:** Two pasted `ResolutionImpossible` / `No matching distribution` failures
in the Step 0 report, then a clean install.
**Revisit if:** pip's index priority ever produces a surprising wheel source, in
which case torch moves to a separate documented install line.
**README line:** "Every dependency is pinned exactly, including the transitive ones
that actually break -- `transformers` 5.15.1 caps `tokenizers` below a version that
was never released, so the working pin is not the obvious one."

---

## D-105 · Ollama is driven over its HTTP API, never the CLI
**Date:** 2026-08-20
**Context:** `doctor` must report which local models are available.
**Decision:** All Ollama interaction -- capability probing included -- goes through
`http://127.0.0.1:11434`, never through the `ollama` executable.
**Why:** Empirically, on this machine `ollama list` hung past a 120-second timeout
while `GET /api/tags` returned the full model list in well under a second. Shelling
out to a CLI that can block indefinitely inside `doctor` would make the first
command a reviewer runs appear to hang. The HTTP path also takes an explicit
timeout, which a subprocess does not without extra machinery.
**Consequence:** `doctor` cannot report anything the API does not expose. It does
expose everything needed: model list, sizes, and load state.
**Evidence:** Step 0 probe, 2026-08-20 -- `ollama list` backgrounded at 120s with no
output; `GET /api/tags` returned six models immediately.
**Revisit if:** a future capability is only reachable through the CLI.
**README line:** "Ollama is driven over its HTTP API rather than by shelling out, so
every probe has a real timeout and `doctor` can never hang."

---

## D-003 · A SQLite-backed sliding-window limiter, not an in-memory one
**Date:** 2026-08-20
**Context:** Gemini's free tier caps requests per minute *and* per day. The agent is
a CLI: a short-lived process that starts, answers, and exits.
**Options:** (a) an in-process token bucket or sliding window; (b) a lock file with
counters; (c) windowed `COUNT(*)` over durable `llm_calls` rows.
**Decision:** (c). Every attempt writes a row; RPM and RPD are derived by counting
rows inside a time window.
**Why:** An in-memory limiter resets on every invocation, so against a *daily* cap it
enforces exactly nothing — you exhaust a 20-request budget across twenty separate CLI
runs and the limiter reports full capacity the whole way. It is not a weak limiter, it
is a decorative one. Deriving from durable rows makes the limit correct across
restarts, which is the only property that matters here.
**Consequence:** One row per attempt, and a `(model, key_alias, ts)` index to keep the
windowed counts cheap. The ledger doubles as the source for "LLM calls per turn, split
by ladder" in the evaluation.
**Evidence:** Measured, 2026-08-20. Two separate OS processes against one database
file, top rung RPD=3: process 1 served `[model-A, model-A, model-A]`; process 2, a
fresh process, was correctly stepped down and served `[model-B]`. The same script with
a fresh database per process — which is what an in-memory limiter amounts to — served
`model-A` again from process 2, over an already-exhausted cap. Unit test:
`test_consumed_rpd_survives_a_process_restart`.
**Revisit if:** the agent ever becomes a long-lived service, where an in-memory window
in front of the ledger would cut read volume. The ledger stays either way.
**README line:** "A daily rate limit cannot be enforced from a process that exits
between every request, so the limiter counts durable rows rather than in-memory state
— demonstrated by draining a rung in one process and watching a second process
correctly step down."

---

## D-004 · Ladder ordering, and where gemini-2.5-flash-lite belongs
**Date:** 2026-08-20
**Context:** A conversational turn costs roughly one synthesis call plus two to four
volume calls: condensation, sufficiency judging, optional rewriting, optional
decomposition.
**Options:** (a) one ladder ordered purely by capability; (b) two ladders, split by
what the call is for.
**Decision:** (b). A **synthesis** ladder ordered by capability, and a **volume**
ladder ordered by daily capacity. Judge and condense calls may never touch synthesis
rungs. Within a rung, every key is drained before stepping down.
**Why:** Routing volume work through synthesis models exhausts reasoning quota in
about four turns, and the failure lands on answer generation — the one thing a
reviewer sees. Splitting the ladders means capacity is spent where it is cheap and
**answer quality degrades last**. Draining keys before rungs preserves capability:
stepping down while another key still has quota on the current rung throws away a
better model for nothing.
**The trap:** `gemini-2.5-flash-lite` is named like a volume model and is capped at
**20 RPD**, not 500. Placed on the volume ladder it exhausts silently after twenty
condensation calls and cascades failures into the agent path, where the cause is
almost unfindable. It sits at the **bottom of the synthesis ladder**.
`Config.validate()` raises if anyone moves it, and three tests cover the placement.
**Consequence:** Two ladders to keep current as model names change. They are data in
`config.py`, not code, and `doctor` prints both so drift is visible.
**Evidence:** `test_trap_model_on_the_volume_ladder_fails_config_validation`,
`test_volume_and_synthesis_land_on_their_own_ladders`,
`test_rpd_exhaustion_steps_down_a_rung_rather_than_failing`.
**Revisit if:** published free-tier limits change — which is why `doctor` prints the
ladders rather than leaving them implicit.
**README line:** "Two ladders, split by purpose rather than capability alone:
gemini-2.5-flash-lite is named like a volume model but capped at 20 RPD, and the
config refuses to start if anyone puts it on the volume ladder."

---

## D-005 · Two cache layers, exact before semantic
**Date:** 2026-08-20
**Context:** Development and evaluation re-run the same prompts constantly.
**Options:** (a) no cache; (b) exact prompt-hash cache only; (c) exact plus a semantic
cache over near-paraphrases.
**Decision:** (c), but with the layers doing clearly different jobs. Layer one is
`sha256(model + prompt + params)` on disk in `.cache/llm/`. Layer two is the semantic
answer cache keyed on the condensed-query embedding, cosine ≥ 0.97 within one corpus
fingerprint.
**Why:** Layer one is non-negotiable — without it a single debugging session eats the
day's synthesis quota, and eval re-runs become unaffordable. Layer two costs almost
nothing because the condensed query is already embedded for retrieval, so the cache
key is free.
**Honest prediction, recorded now so it cannot be quietly revised later:** layer two
will show a hit rate at or near zero on this workload. Twelve deliberately distinct
questions and thirteen conversation turns give it almost nothing to match, and layer
one already absorbs every exact re-run. It is built because it is specified and cheap,
and its measured hit rate will be reported as measured — including if that number is
zero.
**Consequence:** Cache hits still write a ledger row, with `cached=1`, and are excluded
from every quota window. Cache correctness therefore cannot silently corrupt quota
accounting. A corrupt cache file is treated as a miss rather than an exception.
**Evidence:** `test_cache_hit_makes_zero_transport_calls_but_still_writes_a_ledger_row`,
`test_cached_calls_do_not_consume_quota`,
`test_a_corrupt_cache_entry_is_a_miss_not_a_crash`. Measured semantic hit rate: TBD at
Step 12.
**Revisit if:** the measured semantic hit rate is zero across the full evaluation, in
which case the README reports it as a feature that did not pay for itself rather than
pretending otherwise.
**README line:** "Two cache layers: an exact prompt-hash cache that makes evaluation
re-runs free, and a semantic cache whose measured hit rate is reported honestly even
where it is zero."

---

## D-106 · Ledger rows are written at attempt time, not at completion
**Date:** 2026-08-20
**Context:** The limiter could record a call before making it or after it returns.
**Options:** (a) write on success; (b) write on completion, success or failure;
(c) reserve a row before the request, then update it with the outcome.
**Decision:** (c).
**Why:** The provider counts a failed request against quota too, so recording only
successes understates usage and lets the limiter authorise calls the provider will
reject. Worse, (a) and (b) both lose the record entirely if the process is killed
mid-call — the one situation where accurate accounting matters most. Reserving first
means a crash leaves an `ok IS NULL` row that still counts, which is the conservative
direction to be wrong in.
**Consequence:** `ok IS NULL` rows are real and expected; they mean "attempted, outcome
unknown". A malformed-JSON retry reserves its own slot and is logged as
`<purpose>:retry`, because charging one request for two is how a ledger drifts out of
step with the quota it models.
**Evidence:** `test_malformed_json_retries_exactly_once_then_raises` asserts both the
retry count and that both attempts appear in the ledger as failures.
**Revisit if:** never — being wrong toward over-counting is the correct bias for a free
tier.
**README line:** "Quota is charged at attempt time, not on success, so a failed or
crashed request still counts — the same way the provider counts it."

---

## D-107 · RPD is a rolling 24-hour window, not a calendar day
**Date:** 2026-08-20
**Context:** Google's free-tier RPD resets at midnight Pacific. The ledger could model
either a calendar day in some timezone or a rolling window.
**Decision:** Rolling 24 hours.
**Why:** A calendar day requires the local process to agree with Google about the reset
boundary and the timezone. Getting it wrong in the permissive direction means issuing
requests the provider rejects — which then fail inside the agent path. A rolling window
can only ever be *conservative*: it may refuse slightly early, never slightly late.
**Consequence:** After a heavy run, capacity returns gradually over the following day
rather than all at once at a known instant. Acceptable for a tool run interactively.
**Evidence:** `test_rpd_window_rolls_off_after_24h`.
**Revisit if:** the reset boundary ever needs to be exact, e.g. for a scheduled batch.
**README line:** "The daily window is rolling rather than calendar-based, so the limiter
can only ever refuse early, never issue a request the provider will reject."

---

## D-108 · Console output is forced to UTF-8 at the entry point
**Date:** 2026-08-20
**Context:** `doctor` crashed with `UnicodeEncodeError: 'charmap' codec can't encode
characters` from inside the renderer, several frames deep, on a plain Windows shell.
**Decision:** `sys.stdout` and `sys.stderr` are reconfigured to UTF-8 with
`errors="replace"` in `cli.py`, before anything prints.
**Why:** Windows consoles default to cp1252, which cannot encode box-drawing
characters, em dashes, or the accented author names and mathematical symbols that arXiv
text is full of. This is not cosmetic: the crash happens at print time, deep in the
rendering stack, and it would have surfaced later as "the agent dies on some papers and
not others". Fixing it once at the entry point covers every command, including rendered
answers containing citation text.
**Consequence:** A character the terminal font cannot draw appears as a replacement
glyph instead of terminating the process. Visibly degraded beats dead.
**Evidence:** Reproduced then fixed during Step 1 — `doctor` failed on a bare
`console.rule()` before the change and completed cleanly after it.
**Revisit if:** never.
**README line:** "Console output is forced to UTF-8, because arXiv text is full of
characters that make a default Windows console raise mid-print."
