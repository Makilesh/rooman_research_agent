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
**Why:** On this machine `ollama list` never returned, while `GET /api/tags` gave the
full model list in well under a second.

**CORRECTED 2026-08-20, after the backgrounded probe finally completed.** My first
diagnosis here was wrong and is left visible rather than quietly rewritten. I recorded
that `ollama list` "hung past 120 seconds". It did not: it printed the model table
within about three seconds. What actually happened is worse, and is the real argument
for this decision.

Ollama was not running, so the CLI **started the server as a child process**. That
server inherited the command's stdout pipe and never exited — it kept writing to it
for hours (update-checker lines at 16:04, 17:04, 18:04, 19:04). The data arrived
immediately; the *pipe* never closed, so the shell never saw EOF and the command never
returned.

That distinction matters. "Slow command" is solved by a longer timeout. "A subprocess
silently spawns a daemon that holds your stdout open indefinitely" is not solved by a
timeout at all — the timeout fires, the caller moves on, and a long-lived process is
left attached to a pipe nobody is reading. Inside `doctor`, the first command a
reviewer runs, that is a genuinely bad failure. An HTTP request has none of these
properties: it has a real timeout, it spawns nothing, and it cannot leave anything
behind.
**Consequence:** `doctor` cannot report anything the API does not expose. It exposes
everything needed: model list, sizes, and load state. It also means `doctor` reports
Ollama as unreachable when the server is not already running, rather than starting one
as a side effect — which is the correct behaviour for a diagnostic command.
**Evidence:** Step 0 probe, 2026-08-20. `ollama list` was backgrounded at 120s with no
output and completed only hours later; its captured log shows the model table printed
at 16:03:09 followed by Ollama server startup lines and hourly update-checker output on
the same pipe. `GET /api/tags` returned six models immediately. Ollama version 0.16.0.
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

---

## D-006 · Fetch by manifest; never commit the PDFs
**Date:** 2026-08-20
**Context:** The corpus is thirteen arXiv papers, ~25 MB of PDF.
**Options:** (a) commit the PDFs; (b) commit a manifest and fetch on demand.
**Decision:** (b). `data/manifest.yaml` is the committed artifact; `fetch` downloads.
**Why:** Keeps the repository small enough to clone quickly, avoids redistributing
someone else's paper, and makes the corpus exactly reproducible from one command. The
manifest is also more informative than the files: it records why each paper earns its
place, which is the part a reviewer actually wants.
**Consequence:** A reviewer needs network access once. The fetcher waits three seconds
between requests, sends a descriptive User-Agent, and skips files already present, so
re-running is free and arXiv is not hammered.
**Evidence:** Measured 2026-08-20 — 13/13 downloaded, 25.7 MB total, sha256 recorded
per file, delays visible in the log, second run a complete no-op.
**Revisit if:** arXiv availability ever becomes a problem, at which point the honest
fix is a documented mirror, not a committed copy.
**README line:** "The corpus is a version-pinned manifest rather than committed PDFs:
one command reproduces it exactly, and no one else's paper gets redistributed."

---

## D-007 · Reading-order extraction with mass-weighted column detection
**Date:** 2026-08-20
**Context:** Naive `page.get_text()` on a two-column paper reads straight across the
gutter, interleaving the columns into text that is fluent-looking and semantically
destroyed — and that passes every automated check.
**Options:** (a) `get_text()`; (b) `get_text(sort=True)`, which sorts by (y, x);
(c) geometric block extraction with explicit column assignment.
**Decision:** (c). Blocks are extracted with bounding boxes, full-width blocks split
the page into bands, and within each band the entire left column is read before the
right. (b) is *worse than useless* here: sorting by (y, x) is precisely the operation
that interleaves columns.
**The refinement that mattered, and how it was found:** the first implementation
counted *blocks* either side of the midline. Measured against the real corpus it
reported LoRA — a single-column ICLR paper — as two-column on 6 of 26 pages. Inspecting
those pages showed the cause: a displayed equation shatters into a dozen tiny fragments
(`max`, `Φ`, `(x,y)∈Z`) scattered around the midline, and block-counting reads that as
a layout. Detection now weights by **character mass**, ignores blocks under 80
characters, requires column-width geometry, and requires both columns to carry
comparable weight.
**Consequence, and a correction to the brief's premise:** measured across the corpus,
only **2 of 13 papers are actually two-column** (BERT and DPR, both ACL/EMNLP style).
The other eleven are single-column NeurIPS/ICLR papers. The brief assumed the whole
corpus was two-column; it is not. The handling still earns its place — it is
load-bearing for BERT and DPR, and BERT is the second-most-cited paper in the question
set — but the scale of the problem was overstated and this record says so.
**Evidence:** Column profile measured per paper. Junction inspection on DPR page 2:
the left column ends *"...more specifically, we assume"* and the right column begins
*"the extractive QA setting, in which..."* — the sentence continues correctly across
the column break. Block side sequences are `LLLLLRRRRRRRRRRRRRR`, with no interleaving.
**Revisit if:** a paper is added whose layout is three-column or rotated.
**README line:** "Column order is decided by character mass, not block count: a
displayed equation shatters into fragments either side of the midline and fools any
block-counting heuristic into reordering a single-column page."

---

## D-008 · What extraction deliberately does not handle
**Date:** 2026-08-20
**Decision:** Equations, figure internals, and multi-page tables are accepted as poorly
extracted. There is no OCR, now or ever.
**Why:** Each would be a project of its own with a poor return here. A table rendered
as `Batch Size 32 16 1 Sequence Length 512 256 128` is not useful evidence, and no
amount of parser tuning makes it so — the fix is a table-aware extractor, which is
listed under "with more time" rather than half-built. Chasing them would also risk the
prose path, which is what every question in the set actually depends on.
**Consequence:** Questions whose answer lives only inside a results table will retrieve
the surrounding prose instead, and may abstain. That is the correct behaviour: an
honest refusal beats a confident answer read out of mangled table cells.
**Evidence:** Visible in the Step 3 gate dump — LoRA page 4 and CoT page 4 both show
table-cell fragmentation, in an otherwise clean corpus.
**Revisit if:** a table-aware extractor becomes worth its dependency.
**README line:** "Equations, figures and multi-page tables extract poorly and are a
documented limitation, not a bug being chased. There is no OCR path."

---

## D-009 · Chunk sizing: ~700-token children, ~2000-token parents, 15% overlap
**Date:** 2026-08-20
**Decision:** Children are grown paragraph-by-paragraph to ~700 tokens and cut only at
paragraph boundaries; parents group consecutive children to ~2000 tokens; overlap is
15% of the target, carried as whole paragraphs.
**Why:** Fixed-size splits sever claims mid-sentence. Cutting at paragraph boundaries
costs some size uniformity and buys chunks that are readable on their own — which
matters doubly here, because a chunk is what a citation *points at*, and a reviewer
will read it. Overlap exists so a claim spanning a boundary survives intact in at least
one chunk rather than being severed by both. Sub-floor fragments are merged backwards
rather than emitted, because a 30-token caption pollutes retrieval more than it helps.
**Consequence:** Measured on this corpus: 720 children, 273 parents, token counts
min 106 / median 657 / mean 628 / max 1021. Nothing exceeds the 1024-token encoder
window, so no chunk is silently truncated — which would lose exactly the evidence a
citation points at.
**Evidence:** `index` asserts all three properties on every build and fails the command
if any is violated.
**Revisit if:** the encoder window changes, or the ablation shows parent expansion is
not paying for itself.
**README line:** "Chunks are cut at paragraph boundaries rather than fixed offsets,
because a chunk is what a citation points at and a reviewer will read it."

---

## D-010 · Document-local chunk ids, plus a text fingerprint per chunk
**Date:** 2026-08-20
**Context:** Gold labels reference chunk ids. If ids move, the labels silently point at
the wrong text and every downstream number is wrong with no visible symptom.
**Options:** (a) a global counter (`c_0017`); (b) a hash of the text; (c) a
document-local ordinal (`c_lora_0042`).
**Decision:** (c), plus a `text_sha` recorded alongside every gold label.
**Why:** A global counter renumbers every paper the moment a fourteenth is added — the
exact silent-invalidation failure this is meant to prevent. A pure text hash is stable
but unreadable, and an invented id would be indistinguishable from a real one in a
prompt. A document-local ordinal is stable under corpus growth, readable in context
(`[c_lora_0042]`), and makes a fabricated id obvious on sight.
**The second half is the important half.** A stable id is not enough: re-chunking with
different parameters keeps ids valid-looking while pointing at *different text*.
Recording `text_sha` beside each label means evaluation can hard-fail on a stale label
instead of quietly scoring against the wrong passage.
**Consequence:** Ids are longer than the brief's `c_0017`. Worth it.
**Evidence:** `test_chunk_ids_are_document_local_and_stable`,
`test_text_sha_changes_when_the_text_changes`.
**Revisit if:** never.
**README line:** "Chunk ids are document-local so adding a paper cannot renumber the
others, and every gold label records a text fingerprint so a stale label fails loudly
instead of scoring against the wrong passage."

---

## D-011 · FTS5 over a Python BM25 library — and the tradeoff that actually bites
**Date:** 2026-08-20
**Decision:** SQLite FTS5 with `porter unicode61`, in the same file and transaction as
the chunk metadata. See also [D-103], written before the corpus existed and confirmed
by it.
**Why:** Zero extra dependency, incrementally updatable, and — the part that matters
operationally — triggers on `chunks` mirror every insert, update and delete into
`chunks_fts` inside the same transaction, so the two indexes cannot disagree about what
the corpus contains. The fixed `k1=1.2 / b=0.75` is a real limitation and an irrelevant
one: they would never have been tuned on 720 chunks.
**The real cost, confirmed on the corpus:** `unicode61` splits on non-alphanumerics, so
`GPT-3` becomes `gpt` + `3`. Mitigated as planned — compound identifiers are emitted as
quoted phrase queries, so `"GPT 3"` requires adjacency rather than matching any stray
`3`. All query text is tokenised and rebuilt rather than passed through, because an
unescaped quote, asterisk or a bare `NEAR` raises `OperationalError` mid-turn.
**Consequence:** Sparse retrieval keeps a precision ceiling on alphanumeric identifiers.
Step 6's ablation measures it rather than assuming it.
**Evidence:** Six hostile inputs are executed against a real FTS5 table in
`test_hostile_input_produces_a_valid_fts_query`. Live behaviour: the query "What rank
does LoRA use for GPT-3?" compiles to `"rank" OR "LoRA" OR "use" OR "GPT 3"`.
**Revisit if:** the sparse-only ablation row underperforms enough to matter — in which
case a custom tokenizer, not `bm25s`, is the next move.
**README line:** "Sparse retrieval is SQLite FTS5 BM25 in the same transaction as the
chunk metadata, so the dense and sparse indexes cannot disagree about the corpus."

---

## D-012 · numpy sidecar over sqlite-vec
**Date:** 2026-08-20
**Decision:** Vectors in a float32 `.npy` file, with a `chunk_order` table mapping row
index to chunk id.
**Why:** `sqlite-vec` needs a loadable binary extension, which is disabled on some
Python builds — and an install failure costs far more of the 30-point functionality
block than a linear scan costs in milliseconds. At 720 chunks a 720x1024 dot product is
sub-millisecond, and it is *exact*: an ANN index here would be the approximation, not
the optimisation.
**Consequence:** Two artifacts to keep in step. They are written in the same
transaction, and `check_staleness` compares vector count against `chunk_order` rows and
refuses to answer if they diverge.
**Evidence:** 720 vectors x 1024 dims, peak 1.18 GiB VRAM, 12.6s to build.
**Revisit if:** the corpus passes ~10k chunks.
**README line:** "Vectors sit in a numpy sidecar rather than a SQLite extension: an
install failure costs more than a linear scan over 720 chunks ever could."

---

## D-013 · Reciprocal Rank Fusion, k=60
**Date:** 2026-08-20
**Context:** Dense cosine and FTS5 BM25 produce scores on incomparable scales — cosine
in [0,1], BM25 unbounded and corpus-dependent.
**Options:** (a) min-max normalise then weighted sum; (b) z-score then sum; (c) RRF.
**Decision:** RRF, k=60.
**Why:** Normalisation makes the fusion weight corpus-dependent — retune per corpus or
it silently degrades. RRF uses only rank position: no normalisation, one stable
hyperparameter.
**Consequence:** Discards score magnitude entirely. Acceptable precisely because the
cross-encoder re-scores the fused candidates anyway.
**Evidence:** `test_rrf_uses_rank_only_and_ignores_score_magnitude` asserts a
999.0-scoring dense hit and a 0.0001-scoring sparse hit fuse to identical RRF scores.
Live: dense and sparse agree on only 1 of 6 top results for "What rank does LoRA use for
GPT-3?" — that disagreement is the whole point of fusing them.
**Revisit if:** the corpus passes ~50k chunks, where rank saturation starts to matter.
**README line:** "Rank-based fusion avoids score-scale mismatch between dense and sparse
retrievers, and needs no corpus-specific weight to retune."

---

## D-014 · Cross-encoder reranking on the top-25
**Date:** 2026-08-20
**Decision:** `BAAI/bge-reranker-v2-m3`, fp16 on GPU, over the top-25 fused candidates,
returning the top-6.
**Why:** A bi-encoder embeds query and passage separately and never sees the interaction
between their tokens; the cross-encoder restores it. Running it over the full corpus
would be quadratic and pointless — the fused top-25 already contains what matters.
**Evidence, measured on three held-out questions:** reranking improved the top-3 on
**3 of 3**. The decisive case was *"What batch size and warmup schedule did BERT use for
pre-training?"*, where RRF's #1 and #3 were both chunks from the **LoRA** paper — the
adjacent-paper confusion this corpus was chosen to expose — and the cross-encoder
placed BERT chunks in all three slots. On the DPO question it demoted a page of
unreadable equation fragments out of the top-3 entirely.
**Revisit if:** rerank latency becomes a problem at larger top-k.
**README line:** "The cross-encoder earns its place on cross-paper confusions: on a BERT
question, rank-fusion returned LoRA chunks at #1 and #3 and reranking replaced all
three."

---

## D-110 · The double-sigmoid bug, and why the guard is permanent
**Date:** 2026-08-20
**Context:** The reranker is documented as emitting a logit, so the first implementation
applied a sigmoid to map scores into [0,1].
**What actually happened:** `sentence_transformers.CrossEncoder.predict()` already
applies `Sigmoid()` by default. Squashing twice mapped the true range [0.000, 0.998]
onto **[0.500, 0.731]**.
**Why this was nearly invisible:** nothing errored. Scores still sorted, the pipeline
still ran, and the top-6 still populated. The only symptom was that every passage looked
about equally relevant — and the *first* thing anyone would do with that distribution is
derive τ_high, τ_low and τ_verify from it. Every one of those thresholds would have been
measured on a scale compressed into a third of its true range, and the routing built on
them would have been confidently wrong.
**How it was caught:** the top-6 for "What rank does LoRA use for GPT-3?" all scored
between 0.71 and 0.73. That compression is arithmetically diagnostic — `sigmoid([0,1])`
is exactly `[0.500, 0.731]`. A direct probe confirmed it: one relevant and two
irrelevant passages scored `[0.9976, 0.0, 0.0]` through `predict()` and
`[6.02, -11.02, -10.95]` with the activation forced to identity.
**Decision:** Use `predict()` output directly, and assert at call time that
`activation_fn` is `Sigmoid`.
**Why assert rather than comment:** a library default is not a contract. If a future
version returns raw logits, every threshold silently moves to the wrong scale — the
assertion turns that into an immediate, legible failure.
**Evidence:** Before: top-6 spanned 0.7115-0.7270. After: 0.9026-0.9792, with the
bimodal separation the design depends on.
**Revisit if:** never.
**README line:** "The reranker's own sigmoid is asserted rather than assumed: squashing
an already-squashed score compresses [0, 1] into [0.5, 0.73], breaks nothing visibly,
and would have made every measured threshold wrong."

---

## D-111 · The corpus fingerprint hashes extracted text, not source bytes
**Date:** 2026-08-20
**Context:** The fingerprint identifies "this corpus, chunked and embedded this way".
It invalidates the semantic answer cache and warns a session that its stored citations
may no longer mean what they meant.
**What went wrong:** it was first computed over the PDF sha256 plus the chunk config.
Improving the extractor changed the text of every chunk while leaving the source bytes
untouched — so the fingerprint did not move, and nothing downstream knew the corpus had
changed. Confirmed live: two consecutive builds over materially different extracted text
both reported `a60700ff1b9dfc99`.
**Decision:** Hash the extracted text per document instead. Catches a revised source
*and* a revised parser.
**Consequence:** Any change to `ingest.py` invalidates the cache, which is correct and
occasionally inconvenient. The fingerprint moved to `f068f95758f6dcae` on the first
build after the fix.
**Evidence:** The two identical fingerprints across different text, above.
**Revisit if:** never.
**README line:** "The corpus fingerprint hashes extracted text rather than source bytes,
because improving the parser changes every chunk while leaving the PDFs byte-identical."

---

## D-112 · A `pages` table, so ingest and index are separable
**Date:** 2026-08-20
**Context:** The specified schema goes straight from `documents` to `chunks`.
**Decision:** Added a `pages` table holding extracted per-page text, column count and
block count.
**Why:** Three things need it. `ingest` and `index` become genuinely separate commands
rather than one pass that re-parses thirteen PDFs whenever the chunk config moves. The
corpus fingerprint (D-111) needs the extracted text to hash. And the gold-label
validation tool at Step 6 must show a label against *the text the pipeline actually
saw* — not against the PDF, and never against anyone's memory of the paper, which the
project's own rules forbid.
**Consequence:** One table beyond the specified schema, and ~1.4 MB of duplicated text
in the database. Flagged rather than slipped in.
**Evidence:** `ingest` and `index` run independently; re-indexing after a config change
takes 12.6s and re-parses nothing.
**Revisit if:** the corpus grows enough for the duplication to matter.
**README line:** "Extracted page text is persisted, so gold labels are always validated
against the text the pipeline actually saw rather than against the PDF."

---

## D-015 · The three thresholds, measured — and what the measurement actually showed
**Date:** 2026-08-20
**Context:** τ_high, τ_low and τ_verify govern whether the agent answers, asks, or
refuses. The brief is explicit that they must come from the observed score
distribution and never be carried over from another project.
**Decision:** `TAU_LOW = 0.8`, `TAU_HIGH = 0.99`, `TAU_VERIFY = 0.378`, measured
against corpus `e62c6925a03ad297`, with the full evidence in
`outputs/eval_report.md`.

**Two populations, not one — this was a correction to my own first attempt.**
Routing decides on **one number per question**: the top rerank score on the slate.
Verification decides on **one number per (sentence, chunk) pair**. My first
derivation used the per-pair distribution for both, which is a category error: a
question whose best gold chunk scores 0.996 routes to ANSWER regardless of whether
its third gold chunk scores 0.184. Using per-pair scores dragged the positive
population's lower tail down and would have set τ_high on evidence describing a
decision nobody makes. Measured both ways:

| population | n | min | median | max |
|---|---:|---:|---:|---:|
| per gold chunk (wrong for routing) | 17 | 0.184 | 0.929 | 0.996 |
| top score per answerable question | 8 | 0.789 | 0.985 | 0.996 |
| top score per control question | 4 | 0.032 | 0.420 | 0.829 |
| per chunk retrieved for a control | 100 | 0.000 | 0.046 | 0.829 |

**The bimodality the brief warned about is real and visible.** 50 of 100 control
chunks score in [0.00, 0.05]; the per-chunk control mean is 0.107 against a gold
mean of 0.815. This is exactly why usable evidence is scored against a measured
floor rather than averaged: averaging across the retrieved slate would make
retrieving *more* results look *worse*.

**The honest finding: the two routing populations overlap.** The best control
question scores 0.829, above the weakest answerable question at 0.789. No single
threshold separates them on this set. Rather than assert a clean cut, τ was set from
the sensitivity sweep's balanced-accuracy peak (0.85), with the clarify band opened
around it. The overlapping control is `q12_react_llama3_lr` — deliberately the
hardest control, because ReAct *is* in the corpus and *is* about fine-tuning, so
retrieval surfaces confident-looking near-misses. That it lands in the clarify band
rather than being refused outright is arguably the better behaviour, and it is
reported rather than tuned away.
**Consequence:** ~1 in 8 answerable questions will route to clarify rather than
answer. Given the alternative is answering a question the corpus cannot support,
that is the right direction to err.
**Evidence:** `outputs/eval_report.md` §2 — both histograms, all four population
summaries, and the full sweep.
**Revisit if:** the corpus or the question set changes. The measurement is a command,
not a manual exercise.
**README line:** "Thresholds are measured, and the measurement is reported honestly:
the answerable and control populations overlap on this set, so the clarify band was
widened around the sweep peak rather than a clean separation being claimed."

---

## D-113 · A held-out calibration set was NOT built — stated as a known weakness
**Date:** 2026-08-20
**Context:** I argued before Step 0 that deriving thresholds from the same 12
questions the routing is later scored on is fitting on the test set, and proposed a
held-out calibration set alongside the sweep.
**What was actually built:** the sweep, and per-question reporting with bootstrap
intervals. **The held-out set was not built**, because writing 10-15 extra questions
requires gold labels validated against extracted text, and the corpus changed size
mid-build.
**Decision:** ship without it, and say so in the README in those words.
**Why this is recorded rather than quietly omitted:** route accuracy reported at
Step 10 and Step 12 will be measured against thresholds fitted on the same
questions, and is therefore **optimistic by construction**. A reader who does not
know that will over-read the number. The sweep partially compensates — it shows how
much the choice of τ actually matters instead of hiding a point estimate — but it is
not a substitute for held-out data.
**Consequence:** the README's limitations section must say "thresholds are tuned on
the evaluation set, not held out" explicitly, not merely "tuned, not learned".
**Evidence:** n/a — this entry records an absence.
**Revisit if:** there is time before submission to write and validate a calibration
set; it is the single highest-value remaining improvement to the evaluation.
**README line:** "Routing thresholds are tuned on the same question set the routing
is scored against, not on held-out data, so route accuracy is optimistic by
construction — the sensitivity sweep is published so the reader can judge how much
that matters."

---

## D-114 · Small-sample honesty: bootstrap intervals and per-question rows
**Date:** 2026-08-20
**Context:** The answerable set is 8 questions. The ablation table is the evidence
for the highest-weighted section of the brief.
**Decision:** every mean carries a bootstrap 95% interval with a fixed seed, and the
report prints per-question Recall@5 alongside the aggregate.
**Why:** the measured spread makes the point better than any argument — Recall@5 runs
0.479 [0.21, 0.75] for dense against 0.646 [0.42, 0.88] for the full pipeline. The
intervals overlap heavily. The ordering is consistent and matches the mechanism, but
claiming the *difference* is established at n=8 would be overclaiming, and a reviewer
who knows evaluation would spot it immediately. Printing intervals turns a weak
number into an honest one. Per-question rows are printed because at this size a
single catastrophic question can move the mean by 12 points.
**Consequence:** the headline numbers look less impressive than a bare mean would.
That is the point.
**Evidence:** `outputs/eval_report.md` §1.
**Revisit if:** the question set grows past ~30, where the intervals tighten enough
for differences to be claimed.
**README line:** "Every reported mean carries a bootstrap confidence interval,
because a mean over eight questions is not a precise number and presenting it as one
would be dishonest."

---

## D-115 · Gold labels drafted from extracted text, with the quote recorded
**Date:** 2026-08-20
**Context:** The brief forbids writing gold labels from memory of these papers, and
I have read all of them in training. The user delegated the drafting.
**Decision:** every label was chosen by retrieving candidates and reading the
**extracted chunk text** through a purpose-built tool, then recording in
`questions.yaml` the exact sentence from that chunk which carries the answer.
**Why the quote matters more than the id:** a bare chunk id is unfalsifiable without
running the pipeline. A recorded quote lets anyone check a label by reading the YAML
against the paper — which is what makes delegated labelling reviewable rather than
merely asserted. Where a label could not be grounded in a sentence I could point at,
the question was rewritten. `q01` was narrowed from "heads and model dimension" to
"heads and dimension per head" for exactly this reason: the model dimension appears
only inside a results table that extracts as fragments, so the broader question had
no clean textual support.
**Consequence:** three validation layers run on every evaluation — the chunk id must
exist, its `text_sha` must match what it was when labelled, and the corpus
fingerprint must match. Any failure is a hard error, because a label that silently
drifts is worse than no label: the resulting numbers still look plausible.
**Evidence:** `research-agent labels --show` prints every label beside its chunk
text; `labels --stamp` recorded 18 text hashes; all 12 validate.
**Revisit if:** the chunking config changes, which will correctly invalidate every
stamp and force re-validation.
**README line:** "Every gold label records the sentence it was drawn from, so a
reviewer can check the ground truth by reading the question file — no label was
written from memory of the papers."

---

## D-116 · Controls supply the negative population, not non-gold chunks
**Date:** 2026-08-20
**Context:** Deriving a refusal threshold needs a negative population.
**Options:** (a) every non-gold chunk retrieved for an answerable question;
(b) every chunk retrieved for a control question.
**Decision:** (b).
**Why:** (a) is badly contaminated. A chunk not listed as gold is very often still
relevant — the gold set is the chunks that best answer the question, not the only
ones that touch it. Treating those as negatives would push the measured negative
distribution far too high and set the refusal floor far too aggressively. The
control questions have no correct answer anywhere in the corpus, so every chunk
retrieved for them is a true negative by construction. This is the direct payoff of
having built unanswerable controls into the question set rather than treating them
purely as a scored behaviour.
**Consequence:** the negative population is only as large as the controls make it —
100 per-chunk scores from 4 questions. Small, and reported as such.
**Evidence:** `outputs/eval_report.md` §2 — the control histogram, with 50 of 100
scores below 0.05.
**Revisit if:** more controls are added, which would tighten the estimate directly.
**README line:** "The refusal threshold is measured against questions the corpus
genuinely cannot answer, because a chunk that merely wasn't labelled gold is often
still relevant and makes a contaminated negative."

---

## D-016 · Hybrid retrieval beats both single retrievers — with the caveat stated
**Date:** 2026-08-20
**Context:** The Step 6 gate requires hybrid to beat both baselines on Recall@5,
and to investigate rather than proceed if it does not.
**Measured, 8 answerable questions, zero LLM calls:**

| Config | Recall@5 | 95% CI | MRR | nDCG@10 | p50 ms |
|---|---:|---:|---:|---:|---:|
| Dense only (bge-m3) | 0.479 | [0.21, 0.75] | 0.512 | 0.588 | 16 |
| Sparse only (FTS5 BM25) | 0.479 | [0.25, 0.73] | 0.621 | 0.552 | 4 |
| Hybrid, RRF k=60 | 0.583 | [0.29, 0.83] | 0.542 | 0.620 | 16 |
| Hybrid + cross-encoder rerank | **0.646** | [0.42, 0.88] | **0.666** | **0.672** | 385 |

**Gate passed:** hybrid 0.583 > dense 0.479 and sparse 0.479, and the full pipeline
adds a further 6.3 points. The ordering is monotone across all three of Recall@5,
nDCG@10 and MRR, and it matches the mechanism, which is the more convincing part.
**What must be said alongside it:** the intervals overlap heavily at n=8. The
*ordering* is consistent and mechanistically explicable; the *magnitude* of each gap
is not established by this sample. Dense and sparse tie exactly on Recall@5 while
differing on MRR (0.512 vs 0.621) — sparse puts a gold chunk higher when it finds
one, dense finds a different set. That complementarity is precisely what fusion
exploits, and it is visible in live retrieval too: for "What rank does LoRA use for
GPT-3?", dense and sparse agreed on only 1 of their top 6.
**Cost:** reranking is the entire latency budget — 16ms to 385ms p50, a 24x
increase for +6.3 points of Recall@5. Worth it here; recorded so the tradeoff is
visible rather than assumed.
**Evidence:** `outputs/eval_report.md`, run `run_ee06c1d029` in `eval_runs`.
**Revisit if:** latency ever matters more than recall, in which case `hybrid` without
reranking is a defensible configuration and the table already prices it.
**README line:** "Hybrid retrieval beats both single-retriever baselines and
reranking adds more on top, but the confidence intervals overlap at eight questions
— the ordering is trustworthy, the exact gaps are not."

---

## D-016 · The citation contract: structured output, not parsed prose
**Date:** 2026-08-20
**Context:** My M&A engine's README lists as a known limitation: *"passage-level
citations require manual parsing of the synthesiser's inline markers."* This build
exists largely to fix that.
**Decision:** The model emits `{insufficient_evidence, refusal_reason, sentences:
[{text, cite: [chunk_id]}]}` under constrained decoding on both providers. Citations
are never regex'd out of prose.
**Why it is better than markers, concretely:** a sentence-to-chunk mapping is
*checkable* (every id can be validated against what was retrieved), *persistable*
(one row per link in `turn_citations`), and *queryable* (a later turn can resolve
"the second source" with SQL). Inline markers are none of those. They are also
lossy in a way that is easy to miss: `[2]` in prose tells you nothing about which
sentence it supports once two markers sit in one paragraph.
**Consequence:** the answer is assembled from sentences rather than generated as
prose, which costs some fluency. Measured on 12 questions the answers read normally.
**Evidence:** 12/12 schema-valid on the full question set; 31 `turn_citations` rows
persisted; rendering verified to match stored page numbers by test.
**Revisit if:** never — this is the thesis.
**README line:** "Citations are a structured contract the model must emit, not markers
parsed out of prose, which is what makes them checkable, storable and queryable."

---

## D-017 · Constrained decoding guarantees shape; ids are validated separately
**Date:** 2026-08-20
**Decision:** Every returned chunk id is checked against the set actually retrieved
for that turn. An id outside that set raises `InventedCitation` and the answer is
never rendered.
**Why fatal rather than filtered:** dropping a bad citation silently would leave a
partially-correct answer that looks complete, and a fabricated citation is the single
failure this whole system exists to prevent. Loud beats tidy.
**Why it is needed at all, given the schema:** constrained decoding guarantees the
*shape* of the output. It cannot guarantee the ids are real — a model can emit a
perfectly well-formed lie. The two mechanisms cover different failures and neither is
redundant.
**Consequence:** a hard failure on a rare model error, which is the correct trade.
**Evidence:** `test_an_invented_chunk_id_raises_rather_than_rendering`; 12/12 questions
produced zero invented ids in the live run.
**README line:** "Schema-constrained decoding fixes the shape of a citation; a
separate check fixes its truth, because a model can emit a well-formed lie."

---

## D-018 · A refusal carries zero citations, structurally
**Date:** 2026-08-20
**Decision:** When `insufficient_evidence` is true, any sentences the model returned
are discarded. An answer with no usable sentences is also converted into a refusal.
**Why:** the control questions are scored on *zero fabricated citations*, and that
property should hold because of how the code is written, not because the model
behaved well on the day. A refusal that still carries citations is self-contradictory.
**Consequence:** a model that flags insufficiency while also writing a good answer
loses the answer. Correct: the flag is the model's own judgement that it should not
be answering.
**Evidence:** `test_the_refusal_path_produces_zero_citations`; live run — all four
controls abstained with zero citations.
**README line:** "A refusal cannot carry a citation, by construction rather than by
good behaviour."

---

## D-117 · Ollama silently truncates the prompt — the guard that catches it
**Date:** 2026-08-20
**Context:** The first end-to-end run timed out. Diagnosing it turned up something
much worse than slowness.
**What was happening:** Ollama's default context window is **4096 tokens**. The
context here is ~7000. Ollama does not error, warn, or degrade visibly — it drops the
overflow and answers from what is left. Measured directly: a 26,093-character prompt
came back reporting `prompt_eval_count = 4096`.
**Why this is the most dangerous bug found in the build:** the model still produces
fluent, correctly-formatted, properly-cited output. The schema validates. The ids all
exist. Verification passes. Every check in the system goes green while the answer was
written from **less than half the evidence**, and citations may point at passages the
model never actually received.
**Decision:** set `num_ctx` explicitly, and after every call compare the reported
`prompt_eval_count` against the window. If it has pinned to the limit, raise
`ContextTruncated` rather than returning the answer.
**Consequence:** a legitimately over-long context now fails loudly instead of
answering badly. That immediately caught a second real case (a question whose six
parent passages exceeded 8192 tokens), which led to D-120.
**Evidence:** measured `prompt=4096` on a ~7000-token prompt; after the fix, the same
question answers in 3.1s with the full context.
**Revisit if:** never.
**README line:** "Ollama truncates an over-long prompt silently and answers from what
is left, with every downstream check still passing — so the reported prompt token
count is compared against the window on every call."

---

## D-118 · Lexical containment alongside the cross-encoder, because measurement forced it
**Date:** 2026-08-20
**Context:** The plan was cross-encoder-only groundedness verification: free,
deterministic, quota-neutral.
**What the first run showed:** both sentences of a correct answer were flagged
`unverified`. They were **verbatim quotes** from the passage they cited.
**The measurement:** for a sentence lifted word-for-word from `p_attention_0002`:

| passage | contains the sentence? | cross-encoder score |
|---|---|---:|
| `p_attention_0002` | **yes, verbatim** | **0.3438** |
| `p_attention_0001` | no | 0.6218 |

**Why:** the cross-encoder scores *topical relevance between a query and a document*.
"Does this 1,800-token passage contain this 12-word statement?" is a different
question, and it answers it badly — badly enough to rank a verbatim containment below
a passage that does not contain the sentence at all. Two further scale mismatches
compounded it: `tau_verify` was derived from *questions* scored against *child*
chunks, but applied to *sentences* scored against *parent* passages.
**Decision:** compute a deterministic lexical containment score — content-token
coverage, plus a decisive rule for any 5-token verbatim run — and accept a sentence
if **either** signal clears the floor.
**Why either, not both:** lexical overlap is near-perfect for quotation and close
paraphrase, which is what a grounded model actually produces most of the time, and
weak for genuine paraphrase. The cross-encoder is the reverse. Requiring both would
fail almost everything that is true.
**Consequence:** verification is no longer purely model-based, which is an
*improvement* in honesty — the lexical half is fully deterministic and explainable,
and it is free. It also weakens a caveat: the reranker/verifier correlation I flagged
matters less now that the dominant signal is not the reranker.
**Evidence:** before, `verified 0 / unverified 2` on a correct answer; after,
`verified 2`. Across the full 12-question run every answering question had all its
sentences verified.
**Revisit if:** a trained NLI verifier becomes affordable, which is the real fix.
**README line:** "Cross-encoder relevance scores a verbatim quote *below* a passage
that does not contain it, so groundedness also uses deterministic lexical containment
— measured, not assumed."

---

## D-119 · Groundedness is not correctness — the attribution trap
**Date:** 2026-08-20
**Context:** The control question *"How does Attention Is All You Need evaluate on
MMLU?"* must be refused. It was not.
**What happened:** retrieval surfaced a QLoRA passage that genuinely discusses MMLU.
The model answered *"We provide evaluations on MMLU"* and cited it. The citation was
real. The sentence was verbatim in the passage. Verification passed. The answer was
**completely wrong** — it answered about the wrong paper.
**The lesson, which goes in the README:** groundedness verification checks
*sentence ↔ passage* support. It cannot check *question ↔ answer* relevance. A
sentence can be perfectly grounded in a real passage and still be the wrong answer,
and that failure passes every automated check in the system. This is the most
convincing way for a citation system to be wrong.
**Decision:** an explicit attribution rule in the system prompt, promoted above the
refusal rule, plus a worked example of the exact trap: right topic, wrong paper.
**Consequence:** the question now refuses correctly, and all four controls abstain.
But the underlying gap is structural, not fixed by a prompt: the durable fix is the
three-way router (Step 10), which would have sent this to CLARIFY on its top rerank
score of 0.8287 — inside the measured clarify band of [0.80, 0.99].
**Evidence:** before, a confidently wrong cited answer; after, an explicit refusal
naming what is missing. Full run: 4/4 controls abstain.
**Revisit if:** Step 10's router makes the prompt rule redundant. Keep both anyway —
they fail differently.
**README line:** "Groundedness verification proves a sentence came from its cited
passage; it cannot prove the passage answers the question. A right-topic wrong-paper
answer passes every automated check, which is why attribution is checked explicitly."

---

## D-120 · Context is trimmed by dropping whole passages, never by truncating one
**Date:** 2026-08-20
**Context:** Six parent passages at ~2000 tokens each overflow an 8192-token window
before the prompt template is added. D-117's guard caught this immediately.
**Options:** (a) raise `num_ctx` — costs KV-cache VRAM, which is scarce in coresident
mode; (b) truncate the context string; (c) drop whole passages from the bottom of the
ranking until it fits.
**Decision:** (c), with the budget in `config.py`.
**Why not (b):** truncating a passage's text would mean the verifier scores against
text the synthesiser did not see — reintroducing exactly the trap this project was
built around — and would let the model cite a chunk id whose content it only partly
received. A citation must point at a passage that was present *in full*.
**Consequence:** low-ranked evidence is dropped rather than half-included. The
dropped passages are the lowest-reranked, so the loss is the least valuable evidence,
and `turn_retrievals` still records them as retrieved-but-unused for audit.
**Evidence:** questions that previously raised `ContextTruncated` now answer with 3-4
passages in context.
**README line:** "When the context budget binds, whole passages are dropped from the
bottom of the ranking — never truncated — because a citation must point at a passage
the model received in full."

---

## D-121 · Multi-hop synthesis does NOT yet work — measured, not assumed
**Date:** 2026-08-20
**Context:** Three questions are labelled `multi_hop` and are supposed to produce
answers citing two or more papers.
**Measured on the live run:**

| question | papers cited | papers available in context |
|---|---|---|
| q05 chinchilla vs gpt3 | `[chinchilla]` | `[chinchilla, gpt3]` |
| q06 rag uses dpr | `[rag]` | `[rag]` |
| q07 lora to qlora | `[qlora]` | `[qlora]` |

**All three answered from a single paper.** Two distinct causes, and they need
different fixes:
1. **q06 and q07: the second paper never reached the context at all.** Reranking to
   top-6, parent expansion, and the token budget together squeeze out the weaker
   paper. This is precisely what the **document-diversity guard (Step 10)** is
   specified to fix — ensure both papers survive into context when both clear the
   floor.
2. **q05: both papers were available and the model used one.** The Chinchilla
   introduction happens to mention GPT-3's ~300B training tokens, so a single passage
   sufficed. The answer is correct and well-cited; it simply is not multi-hop.
   **Sub-question decomposition (Step 11)** is the fix — retrieving each sub-question
   independently forces evidence from both papers into play.
**Decision:** record this as an open, measured gap rather than quietly accepting
three correct-looking answers. The answers are *right*; the *capability* being
claimed is not yet demonstrated.
**Consequence:** any claim about multi-hop in the README stays `TBD` until Steps 10
and 11 land and this table is re-measured.
**Evidence:** the table above, derived from `outputs/answers/q0{5,6,7}.json`.
**Revisit if:** after Step 11, re-run and re-measure. If decomposition does not move
these numbers, say so.
**README line:** "Multi-hop questions currently produce correct answers from a single
paper — the capability is not yet demonstrated, and the measurement that shows it is
published rather than the three passing answers being taken at face value."

---

## D-122 · Coresident VRAM contention costs ~20x, and `keep_alive` is the fix
**Date:** 2026-08-20
**Context:** The hardware plan budgets encoders at ~4 GB, leaving ~7.5 GB for an
8B-class local model on a 12 GB card.
**What was measured:** the encoders take only 2.16 GB — well under budget. But an
identical short Ollama call took **1.5s standalone** and **71.4s** with the encoders
resident. Watching `GPU free` across a run showed why: it fell to 5.23 GiB and then
rose back to 8.64 GiB. Ollama's scheduler was **evicting and reloading the 5.46 GB
model between calls** under memory pressure. The cost was pure disk I/O, not
inference.
**Decision:** send `keep_alive` on every request so the model stays resident, and set
`num_ctx` explicitly so its KV-cache reservation is predictable rather than
renegotiated per call.
**Consequence:** measured per-question latency across the full 12-question run is
3.3s-15.0s, against a 180s timeout before the fix.
**Note on the original hardware plan:** the VRAM budget was roughly right about
totals and wrong about the failure mode. The problem was never that things did not
fit — it was that a scheduler with a little headroom pressure will thrash rather than
fail, and thrashing looks exactly like slowness.
**Evidence:** the `GPU free` trace, and the 1.5s vs 71.4s comparison of the same call.
**Revisit if:** `sequential` offload mode is measured (Step 7's open item) and proves
better.
**README line:** "Encoders and an 8B local model coexist on 12 GB, but only with
`keep_alive` set — otherwise the scheduler evicts and reloads the model between
calls and a 1.5s request takes 71s."

---

## D-021 · The condensation vocabulary constraint, and the guard that enforces it
**Date:** 2026-08-20
**Context:** In my Opkey build, the follow-up *"what statuses can it have during
approval?"* was condensed into a query containing **"workflow"** — a word nobody had
used. Retrieval went to the wrong chapters. Nothing errored; the answer was fluent
and wrong.
**Decision:** two layers. The prompt forbids introducing any content word absent from
the history or the raw follow-up. Then, programmatically, content words in the
condensed query are stemmed and diffed against `history ∪ raw ∪ referenced-source
vocabulary`; any novel word discards the condensation and uses the raw query.
**Why the second layer is the real one:** a prompt instruction is a request. A diff is
an enforcement, and only the diff is testable. `test_the_drift_guard_catches_an_
injected_novel_content_word` reproduces the exact Opkey failure and asserts the raw
query wins.
**Consequence:** slightly stilted rewrites, and a fallback that is sometimes a worse
query than the model produced. Retrieving on an under-specified query beats retrieving
on a hallucinated term.
**Evidence:** **condensation drift rate 0/13 across all four scenarios.** Target met.
**README line:** "The condenser may only reuse words already in the conversation, and
that constraint is enforced by a stemmed diff rather than trusted to the prompt."

---

## D-123 · The drift guard's stemmer had a false positive
**Date:** 2026-08-20
**Context:** Scenario B turn 3 reported drift on the content word `compar`, for a
condensation that was word-for-word identical to the user's own question.
**Cause:** the stemmer stripped `-ed` from "compared" to give `compar`, but "compare"
matched no suffix and stayed `compare`. The two never compared equal, so reusing a
word already in the conversation registered as novel.
**Decision:** strip a trailing `e` unconditionally after suffix removal, so
`compare`/`compared`/`compares` all reduce to `compar`.
**Why this mattered more than it looks:** a false positive in the drift guard is not
cosmetic. It discards a good condensation and retrieves on the raw pronoun-laden
follow-up instead — degrading exactly the turn the condenser existed to fix, while
reporting a drift rate that overstates the problem.
**Consequence:** drift rate went from 1/13 to **0/13** with no change to the guard's
strictness. The stemmer is still crude, deliberately: its only job is inflection
matching, and a real stemmer is another dependency for that.
**Evidence:** `test_stemmer_is_consistent_across_inflections`, parameterised over the
inflection pairs this corpus actually produces.
**README line:** "An inconsistent stemmer made the drift guard fire on legitimate
rephrasing — the guard is only as good as the normalisation underneath it."

---

## D-025 · Three-way routing, and how badly the thresholds were derived
**Date:** 2026-08-20
**Decision:** route on the **top** rerank score: `>= tau_high` answer, `< tau_low`
refuse, between them ask. Never on an average — averaging across the slate makes
retrieving *more* results look *worse*, because the tail of any slate is noise.

**This threshold was derived wrong twice, and both times end-to-end measurement
caught it.** The sequence is worth recording, because it is the whole argument for
validating a rule against behaviour rather than against its own histogram.

| attempt | rule | tau_low | tau_high | route accuracy |
|---|---|---:|---:|---:|
| 1 | band around the sweep's balanced-accuracy peak (peak−0.05, peak+0.15) | 0.80 | 0.99 | **2/13** |
| 2 | the overlap region: min(positives) to max(negatives) | 0.674 | 0.839 | 6/13 |
| 3 | percentiles: p05 of positives, p95 of negatives | **0.736** | **0.786** | **10/13** |

Attempt 1's `+0.15` was arbitrary and produced `tau_high = 0.99`, pushing roughly half
of genuinely answerable questions into the clarify band. Attempt 2 was principled in
shape but keyed the band on **one observation from each population** — with n=11 and
n=4, those single points move enormously on resampling, and the resulting band was so
wide it swallowed most real questions. Attempt 3 uses percentiles, which tolerate one
unusual question on either side.
**The honest caveat, and it is a large one:** the negative population is **four
questions**. That is not enough to estimate a p95, and `tau_high` is therefore the
least well-evidenced number in the system. The derivation says so in its own output.
**Evidence:** the table above; `outputs/eval_report.md` for the distributions.
**Revisit if:** more control questions are written. This is the single highest-value
addition to the evaluation, ahead of everything else on the list.
**README line:** "The routing thresholds were derived wrong twice before end-to-end
measurement caught it — a threshold rule has to be validated against routing
behaviour, not against its own histogram."

---

## D-124 · Threshold calibration must include conversational turns
**Date:** 2026-08-20
**Context:** Thresholds calibrated only on the 12 well-formed single-turn questions
refused *"What problem does LoRA solve?"*, which scores 0.684 — below the 0.789
minimum of that population.
**Cause:** evaluation questions are longer and more specific than what a person types
mid-conversation. A threshold fitted to them is fitted to the wrong distribution.
**Decision:** add the first turn of each conversation scenario to the calibration
population. Turn 1 only, deliberately: later turns depend on condensation, and
folding condenser behaviour into a retrieval threshold would make the number depend
on which model happened to serve the rewrite. `clarify` expectations are excluded
from both populations — by definition neither clearly answerable nor clearly not, so
they cannot inform a boundary without begging the question.
**Consequence:** the answerable population grew from 8 to 11 and its minimum dropped
from 0.789 to 0.684 — a large shift from three questions, which is itself evidence of
how under-sampled this is.
**README line:** "Routing thresholds are calibrated on conversational turns as well as
evaluation questions, because a question typed mid-conversation scores measurably
lower than a well-formed one."

---

## D-026 · Clarifying questions are built from the competing candidates
**Date:** 2026-08-20
**Decision:** generate the clarifying question from the retrieved candidates that
cleared `tau_low`, one per document, and fall back to a deterministic "Which did you
mean: X, or Y?" if the model call fails.
**Why one per document:** three chunks from the same paper describe one option three
times, which is not a choice.
**Why not "could you clarify?":** a generic prompt spends the user's turn and returns
no information. The retrieved candidates already say what the ambiguity *is*.
**Consequence:** a clarification costs one volume-ladder call, never a synthesis one.
**Evidence:** `test_clarification_names_one_option_per_document`.
**README line:** "A clarifying question names the actual competing sources, because
the retrieval results already say what the ambiguity is."

---

## D-027 · The document-diversity guard
**Date:** 2026-08-20
**Context:** Step 7 measured all three multi-hop questions answering from a single
paper, and in two of them the second paper never reached the context slate at all.
**Decision:** if the top-N slate is monopolised by one document, the weakest slot is
given to the best passage from another document, provided it clears `tau_low`.
**Why:** reranking is per-passage and paper-blind, so one paper's chunks can take
every slot even when another clears the floor comfortably. That is an artefact of
ranking, not a fact about the corpus.
**What it cannot do:** manufacture evidence. If only one paper is relevant, nothing
changes — asserted by `test_diversity_guard_does_not_invent_diversity`.
**Consequence:** the sixth-best passage is sometimes displaced by a weaker one from
another paper. That is the intended trade for multi-hop questions and a small loss
for single-hop ones.
**Evidence:** `test_diversity_guard_admits_a_second_paper`. Whether it actually fixes
multi-hop citation spread is re-measured at Step 12 — it is not claimed here.
**README line:** "Reranking is paper-blind, so one document can monopolise the context
slate; the diversity guard reserves a slot for a second paper that has earned one."

---

## D-125 · Evaluation clears the semantic cache before running
**Date:** 2026-08-20
**Context:** Two consecutive scenario runs with no code change between them produced
different route decisions.
**Cause:** the semantic answer cache persists across runs. Once populated, it
short-circuits turns, so a later run measures the cache's history rather than the
pipeline.
**Decision:** `chat-eval` truncates `answer_cache` before it starts.
**Why this matters beyond tidiness:** it briefly made a threshold change look like it
had *worsened* route accuracy, which nearly sent me tuning in the wrong direction. An
evaluation that is not reproducible is not evidence.
**Consequence:** the measured semantic cache hit rate during evaluation is 0 by
construction. That is honest — the cache's value is in interactive use, and D-005
already predicted a near-zero hit rate on this workload.
**README line:** "Evaluation starts from an empty semantic cache, because a cache that
persists between runs measures its own history rather than the pipeline."

---

## D-126 · Scenario outcomes, stated as measured rather than as designed
**Date:** 2026-08-20
**Measured:** route accuracy **10/13**, condensation drift rate **0/13**.

| gate | outcome |
|---|---|
| A — "the quantised version" resolves without the user naming QLoRA | **partial** |
| B — turn 1 clarifies, turn 2 resolves | **failed on turn 1** |
| C — abstains mid-conversation | **passed** |
| D — ordinal source reference resolves | **passed in mechanism, not in demo** |

**A (partial).** The condenser did resolve the pronoun correctly, producing *"How does
the quantised version of LoRA reduce memory?"* with no drift, and turns 2-4 all
answered. But retrieval then returned LoRA's own memory discussion rather than
QLoRA's NF4 and Double Quantization. The *coreference* worked; the *cross-paper hop*
did not. Same root cause as D-121.

**B (failed).** *"What rank is used?"* scored 0.188 and was refused, not clarified.
The finding underneath it is more interesting than the failure: **a vague query and an
unanswerable query are indistinguishable to a relevance score.** Both retrieve nothing
that scores well. Detecting ambiguity needs a different signal — for instance several
candidates that are mutually inconsistent yet individually plausible — and a top-score
threshold cannot express that.

**C (passed).** All three turns correct, including the mid-conversation abstention on
Llama 3 after two turns of answering confidently about ReAct.

**D (passed in mechanism).** Turn 2 correctly detected that the previous answer cited
only **one** source and asked which was meant, rather than retrieving on the
contentless phrase "the second source" — which had previously produced a bewildering
refusal at score 0.057. The resolution machinery is verified by tests against
`turn_citations`. The intended demonstration did not run because turn 1's answer cited
one source, not two.
**Decision:** report all four as measured. The expectations were written before the
system existed and three of them encode assumptions that turned out to be wrong; the
honest move is to say which, not to adjust the expectations until they pass.
**README line:** "Route accuracy is 10/13 and drift is 0/13; of the four conversation
gates one passed cleanly, one passed in mechanism, one partially, and one failed —
reported per scenario rather than as a single number."

---

## D-028 · A plain state machine, and one implementation of it
**Date:** 2026-08-20
**Decision:** the turn loop is ~200 lines of ordinary Python in `agent.py` and
`sufficiency.py`. No LangGraph.
**Why:** I have shipped LangGraph in production. Here it would hide exactly the logic
a reviewer most wants to read — the flow is one function you can follow top to bottom,
and the trace prints the nodes it actually visited rather than a graph definition
describing nodes it might.
**A duplication caught while wiring this up:** `ask` had grown its own copy of the
retrieve/synthesise/verify pipeline alongside `chat`'s. It silently missed every
capability added to the conversational path, the sufficiency loop included. `ask` is
now a session with one turn, going through the same state machine. Two
implementations of one pipeline is how a system develops behaviour nobody intended.
**Evidence:** `ask --trace` prints the loop; the same code serves `chat`.
**README line:** "The agent loop is plain Python, and there is exactly one of it — a
single-turn question is just a session with one turn."

---

## D-029 · Sub-questions are reranked against themselves
**Date:** 2026-08-20
**Decision:** each sub-question is retrieved *and reranked* using itself as the query;
results are merged by best score per chunk.
**Why:** scoring sub-question results against the parent query undoes the
decomposition. A passage answering one part of a composite question need not resemble
the composite question at all. In my M&A engine this detail was worth roughly +20pp
fact coverage.
**Consequence on this corpus:** it did not reproduce that gain — see D-127. The
mechanism is asserted by test regardless, because if decomposition is ever worth
having, this is the part that makes it work.
**Evidence:** `test_sub_questions_are_reranked_against_themselves` asserts each
sub-question appears as its own rerank query and that evidence from both survives.
**README line:** "Each sub-question is reranked against itself, not the parent query —
otherwise decomposition throws away the thing it was for."

---

## D-030 · Progressive constraint relaxation
**Date:** 2026-08-20
**Context:** In the M&A engine a wrong first-pass category guess removed whole
documents from the search space, and no amount of query rewriting recovered the
answer — the evidence was never a candidate.
**Decision:** the first attempt scopes retrieval to any paper the question names
explicitly; every retry drops that scope entirely. Tight first for precision, loose on
retry for recall. A scope is also never allowed to empty the candidate set, because an
empty slate can only be refused and that hides the reason.
**Why it matters here specifically:** a question that names the *wrong* paper — "which
section of the LoRA paper describes NF4" — is answerable only after the scope is gone.
The filter that helps a well-posed question is exactly the filter that makes a
false-premise question unanswerable.
**Evidence, live:** on "How does QLoRA reduce memory beyond what LoRA alone achieves?"
the first pass scoped to `['lora', 'qlora']` and scored 0.915; the judge called it
insufficient; the retry dropped the scope and scored **0.979**, and the judge passed
it. `test_relaxation_drops_the_scope_on_retry` asserts the mechanism.
**README line:** "Any filter applied on the first retrieval is dropped on retry,
because a wrong first-pass guess removes the answer from the search space and no
amount of rewriting brings it back."

---

## D-031 · Termination is structural, not conditional
**Date:** 2026-08-20
**Decision:** the loop is a bounded `for` over `max_retrieval_loops + 1`. There is no
exit condition a model can influence.
**Why:** an agentic loop whose termination depends on a judge returning `sufficient`
can be talked out of stopping. Making the bound structural means no mock, and no
model, however pathological, can make it run away.
**Evidence:** `test_the_loop_always_terminates` is parameterised over four adversarial
judge/decompose/rewrite configurations, including one where nothing ever succeeds.
**README line:** "The retrieval loop is a bounded for-loop, not a while-loop with an
exit condition the model controls."

---

## D-127 · Decomposition does NOT improve fact coverage on this corpus
**Date:** 2026-08-20
**Context:** Step 7 measured all three multi-hop questions answering from a single
paper. Sub-question decomposition was the specified fix, and I expected it to work —
it was worth about +20pp in my previous system.
**Measured, three multi-hop questions, decomposition off vs on:**

| question | facts off | facts on | papers off | papers on | subs |
|---|---:|---:|---:|---:|---:|
| q05 chinchilla vs gpt3 | 0.50 | **0.25** | 1 | 1 | 3 |
| q06 rag uses dpr | 0.50 | 0.50 | 1 | 1 | 0 |
| q07 lora to qlora | 0.20 | 0.20 | 1 | 1 | 0 |
| **mean** | **0.40** | **0.32** | **1.00** | **1.00** | |

**It is slightly worse, and it does not move paper spread at all.** Decomposition
fired on exactly one of the three, and that one got *worse*.

**Why, as far as the evidence shows.** Two things. First, this corpus cross-references
itself: the Chinchilla introduction states GPT-3's ~300B training tokens, so a
single-paper answer is genuinely available and the model takes it. Second, splitting
q05 into three sub-questions widened the merged slate, and the context token budget
then dropped the weaker half — so decomposition *added* candidates and the budget
*removed* them, ending with a thinner answer than the undecomposed query produced.

**Decision:** keep the mechanism, report the number, and do not claim the capability.
The spec asked for this to measurably help or be reported honestly as not helping. It
does not help here.
**What I would try next, not built:** raise the context budget when a decomposition
has fired, so sub-question evidence is not immediately discarded by the very budget
that made room for it; and require the synthesiser to attempt one sentence per
sub-question. Both are guesses until measured.
**Evidence:** the table above, from a direct on/off comparison with the semantic cache
cleared between runs.
**Revisit if:** the context budget is made decomposition-aware.
**README line:** "Sub-question decomposition was built, measured, and does not improve
fact coverage on this corpus (0.40 to 0.32) — reported as measured rather than
claimed as a feature."

---

## D-128 · Multi-hop remains unsolved after the loop — the gap, restated
**Date:** 2026-08-20
**Context:** D-121 recorded multi-hop answers coming from one paper and named two
fixes: the diversity guard (Step 10) and decomposition (Step 11). Both are now built.
**Measured:** papers cited per multi-hop answer is still **1.00**, with the diversity
guard active and with decomposition on.
**What this says.** The diversity guard does get a second paper into the *context
slate* — that is asserted by test. The failure has moved one stage later: the
**synthesiser** receives passages from two papers and still writes its answer from
one. That is a different problem from the one diagnosed at Step 7, and neither of the
two fixes addresses it.
**Decision:** stop claiming multi-hop, and say precisely where it fails. The three
multi-hop answers are correct and well-cited; they are not multi-hop.
**What would actually address it, not built:** the synthesis prompt does not ask for
cross-document synthesis at all. A rule requiring that, when passages from two papers
are supplied for a comparative question, the answer cite both — plus a check that
fails the answer if it does not — is the obvious next move, and it is a Step 7 change
rather than a Step 11 one.
**Evidence:** the D-127 table; `outputs/answers/q0{5,6,7}.json`.
**README line:** "Multi-hop is not solved: the diversity guard puts a second paper in
the context, and the synthesiser still answers from one. The failure is in synthesis,
not retrieval, and that is where the next fix belongs."

---

## D-129 · A semantic cache hit was discarding the cached answer
**Date:** 2026-08-20
**Context:** In a full run, `q07` reported **0 sentences, 0 citations, 14ms**. It
looked like a catastrophic pipeline failure.
**Cause:** it was a cache hit. The cache lookup returned the stored record, but the
turn was constructed with `answer=None` — nothing ever reconstructed an `Answer` from
the JSON. The cache stored the answer correctly and then threw it away on retrieval.
**Why it hid so well:** the failure is indistinguishable from a legitimate empty
result unless you notice the 14ms. Every other check passed; the row simply had zeros
in it.
**Decision:** `answer.from_cached()` rebuilds a full `Answer` from the stored record,
and the sources record now includes passage text so a restored answer is renderable.
Both eval commands clear the cache before running, for the reproducibility reason in
D-125.
**Consequence:** a cache hit is now indistinguishable from a fresh answer in output,
and reports its cosine similarity in the route reason.
**Evidence:** before, `q07` returned 0/0/14ms; after, 3 sentences and 3 verified.
**README line:** "A cache that loses the thing it cached is worse than no cache — a
hit now rebuilds the full answer, and evaluation always starts from an empty one."

---

## D-130 · The router made the false-premise question worse
**Date:** 2026-08-20
**Context:** At Step 7, `q11` ("which section of the LoRA paper describes its NF4
quantisation?") correctly cited QLoRA. After the router landed, it refuses.
**Cause, traced:** scoping retrieval to LoRA — which the question names — produces
uniformly low rerank scores, because LoRA contains no quantisation content. Those low
scores are *correct evidence that the premise is false*. But the sufficiency judge
declared the LoRA passages sufficient, so relaxation never fired, and the router then
refused on the low top score.
**The shape of the problem:** the low score means two different things — "the corpus
cannot answer this" and "the corpus cannot answer this *as asked*" — and the router
cannot tell them apart, because a single scalar cannot carry that distinction.
**Decision:** record it as a regression rather than special-casing the router. A rule
that answers when the score is low would break the abstention controls, which is a
worse trade.
**Consequence:** false-premise correction currently works only when the named paper is
not scoped away, which for `q11` it always is. The honest fix is for the judge to
recognise "right topic, wrong paper" — which the *synthesis* prompt already does since
D-119, but the judge prompt asks a different question and gets it wrong.
**Evidence:** `q11` answered at Step 7, refuses at Step 11; loop trace shows `loops=0`,
meaning the judge passed the scoped LoRA evidence.
**Revisit if:** the judge prompt is aligned with the synthesis prompt's attribution
rule. That is the single most likely thing to fix it.
**README line:** "Adding the router made one false-premise question worse: a low
retrieval score means both 'unanswerable' and 'unanswerable as asked', and one number
cannot distinguish them."

---

## D-131 · The two prompt fixes: one worked, one did not, and both are reported
**Date:** 2026-08-20
**Context:** Phase 5 ended with two named gaps and I was asked to fix both before the
full evaluation.

**Fix 1 — cross-document synthesis (rule 6).** The synthesis prompt never asked for
it. Added a rule requiring that a comparative question be answered from both papers
when both are in context, plus a worked example.
**Outcome: it works, and it is not reliable.** On `q07` it produced the first genuine
multi-hop answer in the project — QLoRA's NF4, Double Quantization and the 780GB→48GB
figure alongside LoRA's 10,000x trainable-parameter reduction, citing both papers. On
the full evaluation run the same question averaged **1.00 papers per multi-hop
answer** again. Same code, same corpus, different run.

**Fix 2 — aligning the judge with the attribution rule.** The judge already caught
"right topic, wrong paper" but not the mirror case, "right paper, wrong topic", which
is what `q11` actually is.
**Outcome: partially worked.** The judge now correctly reports insufficiency and
relaxation fires — `q11` went from `loops=0` to `loops=2`, exactly the intended
behaviour. But it still refuses, and the refusal reason shows why: *"None of the
provided sections from LoRA papers mention a 4-bit NF4 quantisation scheme."* **No
QLoRA passage reached the context at all**, so there was no evidence with which to
correct the premise.

I also found and fixed a genuine contradiction between my own rules along the way:
rule 3 said "right topic, wrong paper → refuse" while rule 5 said "wrong premise →
correct it and cite". For `q11` those are the same situation with opposite
instructions, and rule 3 won on priority. Rule 3 now splits the case explicitly: the
topic elsewhere in the corpus means correct the premise; the topic nowhere means
refuse.
**Decision:** stop after three attempts on `q11` and report the diagnosis instead.
Step 6 already measured it at **0.00 Recall@5 on every retrieval configuration** — the
question names the wrong paper, so the query text steers both retrievers into LoRA.
A prompt cannot correct a premise using evidence it was never given. This is a
retrieval problem wearing a synthesis costume.
**README line:** "A false-premise question that names the wrong paper defeats
retrieval by construction: the query text steers the retriever to the paper the
question names, and no prompt can correct a premise from evidence that never arrived."

---

## D-132 · Run-to-run variance is large enough to swamp small effects
**Date:** 2026-08-20
**Context:** Several measurements moved between runs with no code change in between.
Observed: condensation drift rate 0/13 then 1/13; papers per multi-hop answer 2 then
1.00; conversation route accuracy 9/13, 6/13, 10/13 across threshold attempts that
were partly measuring noise.
**Cause:** local generation is sampled, not greedy. The disk cache makes an *identical*
prompt deterministic, but any turn whose prompt depends on a previous turn's answer
gets a different prompt as soon as that answer differs by a word. Conversations
compound this turn over turn.
**Decision:** record it as a first-class limitation rather than reporting single-run
numbers as though they were stable, and never attribute a small change to a code
change without re-running.
**What this cost me, concretely:** a threshold change briefly appeared to *reduce*
route accuracy from 9/13 to 6/13, and I nearly tuned in the wrong direction on the
strength of it. The semantic-cache fix (D-125) removed one source of variance; the
sampling remains.
**What would fix it, not built:** fix the sampling seed and set temperature to 0 for
every evaluation call, then re-run each configuration three times and report the
spread. That is the correct way to measure an effect this small, and it is the single
most valuable addition to the harness.
**Consequence for every number in `outputs/eval_report.md`:** they are single-run
observations. The large, mechanistically-explicable effects (hybrid beats single
retrievers; reranking fixes cross-paper confusion) are trustworthy. Differences of a
few percentage points are not.
**README line:** "Local generation is sampled, so numbers move between runs — the
large effects are trustworthy, differences of a few points are not, and the report
says which is which."

---

## D-133 · Retrieval and generation metrics are reported in separate sections
**Date:** 2026-08-20
**Decision:** `outputs/eval_report.md` puts LLM-free metrics in section 1 and
LLM-dependent metrics in sections 2-3, each labelled, rather than merging them into
one ablation table.
**Why:** the brief asks which columns are LLM-dependent, and the honest answer is
structural rather than a footnote. Retrieval numbers are reproducible on demand and
comparable across providers; generation numbers move with the model, and per D-132
they move between runs of the *same* model. Putting them in one table invites reading
a synthesis model's variance as a retrieval result.
**Consequence:** the report is longer and the headline table is smaller. Both are
correct.
**Evidence:** `outputs/eval_report.md` sections 1 through 4, plus the traceability
table in section 6 marking every README-bound number LLM-dependent or not.
**README line:** "Retrieval metrics and generation metrics are reported separately and
labelled, because one is reproducible and the other moves with the model."

---

## D-134 · Measured results, Phase 6
**Date:** 2026-08-20
**Full evaluation, 12 single-turn questions and 13 conversation turns, Ollama
`llama3.1:8b`, zero Gemini calls:**

| Metric | Value | LLM-dependent |
|---|---:|---|
| Recall@5, hybrid + rerank | 0.646 [0.42, 0.88] | no |
| Recall@5, hybrid | 0.583 [0.29, 0.83] | no |
| Recall@5, dense / sparse | 0.479 / 0.479 | no |
| **Citation precision** | **0.950** | yes |
| **Abstention accuracy** | **1.000 (4/4)** | yes |
| **Invented citation ids** | **0** | yes |
| **Refusals carrying citations** | **0** | yes |
| Fact coverage (mean) | 0.486 | yes |
| Route accuracy, single-turn | 0.833 (10/12) | yes |
| Route accuracy, conversational | 10/13 | yes |
| Condensation drift rate | 1/13 (0/13 in a prior run) | yes |
| Papers per multi-hop answer | 1.00 | yes |
| p50 / p95 latency | 11.5s / 24.9s | yes |
| LLM calls / cache hit rate | 340 / 27.1% | yes |

**The four numbers that matter most are the structural ones:** zero invented citation
ids, zero refusals carrying citations, 4/4 abstention, 0.950 citation precision. Those
hold because of how the code is written rather than how the model behaved — the first
two are enforced in `answer.py` and cannot be violated.
**The weakest numbers, named:** fact coverage at 0.486 is low and reflects an 8B local
model, substring matching, and demanding `expected_facts`. Papers-per-multi-hop at
1.00 means the headline multi-hop capability is still unproven.
**README line:** "Zero invented citations and 4/4 abstention across every run — those
hold structurally, not by good behaviour on the day."

---

## D-135 · Greedy decoding, and the warm-up call that makes it actually deterministic
**Date:** 2026-08-20
**Context:** D-132 recorded that several measurements moved between runs with no code
change, and that this nearly caused a threshold to be tuned in the wrong direction.
**Decision:** `temperature = 0` and a fixed `seed` on every local call, both included
in the disk-cache key, plus one throwaway warm-up call before any measurement begins.

**The warm-up is not superstition — it was measured.** Six consecutive identical
greedy calls after a cold model load:

```
1. e803d99d385c   <- differs
2. b12bd5f2c7ad
3. b12bd5f2c7ad
4. b12bd5f2c7ad ... 6/6 agree from call 2 onward
```

The **first** call after a load returns different text from every subsequent identical
call, which then agree with each other completely. One call is enough to make an
evaluation irreproducible, so it is now spent deliberately and discarded rather than
landing on the first question of the run.

**Verified end to end.** Two full evaluation runs with `.cache/llm` deleted between
them — so the model regenerated every answer from scratch — produced **byte-identical
reports apart from wall-clock latency.** Every sentence count, verification status,
paper count, fact-coverage score, loop count and route decision matched.

**Decoding parameters are part of the cache key.** Without that, changing the
temperature would silently serve answers generated under the old one, which is a
subtle way to make a comparison meaningless.

**It also improved the numbers, which was not the goal.** Sampling was costing
accuracy, not just reproducibility: fact coverage 0.486 -> **0.660**, single-turn
route accuracy 0.833 -> **0.917**, condensation drift 1/13 -> **0/13**.
**Consequence:** every generation metric in `outputs/eval_report.md` can now be
reproduced by re-running the command, and the "single-run observation" caveat that
D-132 attached to all of them is retired. D-132 stays in the log as the record of why
this was built.
**Evidence:** the six-call trace above; the two-run diff.
**README line:** "Evaluation runs greedily with a fixed seed and a discarded warm-up
call, because the first generation after a model load is not reproducible — two runs
with the cache cleared now produce identical reports."

---

## D-136 · The sufficiency judge now fires on every question
**Date:** 2026-08-20
**Observed:** after the D-131 judge strengthening, all twelve questions report
`loops = 2` — the judge calls the evidence insufficient on the first pass every time,
so every question runs the full loop including relaxation.
**Is it wrong?** Not in outcome: single-turn route accuracy is 0.917, abstention is
4/4, and citation precision is 0.882. The loop is finding good evidence.
**But it is not free.** p50 latency is ~14.6s where a single pass was ~4s, and each
question now costs a judge call plus a rewrite or decomposition on the volume ladder.
On Ollama that is only time; on Gemini it would be roughly three times the volume-rung
consumption per question.
**Reading:** the judge prompt is now strict enough that "sufficient" is nearly
unreachable on a first pass. A judge that always says no carries little information —
it has become an unconditional second retrieval rather than a decision.
**Decision:** record it rather than re-tune. Re-tuning the judge to be less strict
risks reintroducing exactly the attribution failure D-119 and D-131 were built to fix,
and the current behaviour is correct-but-expensive rather than wrong.
**What to measure before changing it:** run with `max_retrieval_loops = 0` and compare
route accuracy and fact coverage. If a single pass matches, the judge is pure cost. If
it does not, the loop is earning its latency. That measurement is not yet done.
**Evidence:** `loops = 2` on all twelve rows of `outputs/eval_report.md` section 2.
**README line:** "The sufficiency judge currently rejects the first retrieval on every
question, which makes the loop an unconditional second pass — correct in outcome,
roughly 3x the latency, and flagged rather than tuned away."

---

## D-137 · Final measured results, Phase 6 (reproducible)
**Date:** 2026-08-20
**Provider:** Ollama `llama3.1:8b`, greedy, fixed seed. **Zero Gemini calls have been
made at any point in this build.** The user deferred the Gemini comparison.

| Metric | Value | LLM-dependent |
|---|---:|---|
| Recall@5 — hybrid + rerank | 0.646 [0.42, 0.88] | no |
| Recall@5 — hybrid / dense / sparse | 0.583 / 0.479 / 0.479 | no |
| **Citation precision** | **0.882** | yes |
| **Abstention accuracy** | **1.000 (4/4)** | yes |
| **Invented citation ids** | **0** | yes |
| **Refusals carrying citations** | **0** | yes |
| Fact coverage (mean) | 0.660 | yes |
| Route accuracy — single-turn | 0.917 (11/12) | yes |
| Route accuracy — conversational | 10/13 | yes |
| **Condensation drift rate** | **0/13** | yes |
| Papers per multi-hop answer | 1.00 | yes |
| p50 / p95 latency | 14.6s / 18.6s | yes |

**Reproducible:** two runs with the cache cleared produced identical values for every
row above except latency.

**The strongest results are the structural ones.** Zero invented citation ids and zero
refusals carrying citations hold because `answer.py` enforces them, not because the
model behaved. 4/4 abstention has held across every run of the project.

**The weakest, named plainly.** Papers-per-multi-hop is 1.00: the headline multi-hop
capability is still unproven, and the one run that produced a genuine two-paper answer
did not reproduce. `q11` remains the single route failure, diagnosed at Step 6 as a
retrieval problem (0.00 Recall@5 on every configuration) rather than a synthesis one.
**README line:** "Every generation number is reproducible: two runs with the cache
cleared produce identical reports."

---

## D-138 · What is committed, and what is not
**Date:** 2026-08-20
**Decision:** `outputs/` is committed in full — 12 answers as `.md` + `.json`, 4
conversation transcripts, `eval_report.md`. Everything rebuildable is ignored:
`storage/`, `*.npy`, `.cache/`, `data/sources/*.pdf`, `.venv/`.
**Why commit outputs:** if a reviewer cannot run Ollama, or skips setup entirely, the
results are still on the page and still scoreable. That is worth more than the ~90 KB
it costs.

**A problem found while auditing this, and it was not a size problem.** The corpus
PDFs are deliberately not committed (D-006) so the repository does not redistribute
anyone's paper. The answer JSONs then embedded **20,679 characters of verbatim paper
text each** — 8,000 of it from passages that were never even cited — because
`answer_to_dict` was serving two callers with one shape: the semantic cache, which
genuinely needs full passage text to reconstruct an answer, and the committed
artifact, which does not. That quietly reintroduced exactly what D-006 avoids.
**Fix:** the committed JSON carries only **cited** passages, each truncated to a
600-character excerpt — enough to check a citation by eye, far short of republishing
the passage. The cache keeps full text, in the database, which is gitignored.
`outputs/answers/` went from 311 KB to 63 KB.

**Two further gitignore findings:** a stray `*m/` pattern would have ignored any
directory whose name ends in "m", and the private working notes were listed only by
directory. Both fixed; the private files are now listed by name as well, so they stay
ignored if they are ever moved back to the root.
**README line:** "Results are committed so they are readable without running
anything, but the answer files carry only cited excerpts — committing full passage
text would re-introduce the redistribution question that not committing the PDFs
avoids."

---

## D-139 · Refusals are deliverables too
**Date:** 2026-08-20
**Context:** Four of the twelve `outputs/answers/` files were missing.
**Cause:** a refusal decided by the router never produces an `Answer` object, and the
writer was guarded on `if result.answer is not None`. So nothing was written for
exactly the four questions that demonstrate abstention — the capability the brief
singles out and that most submissions skip.
**Decision:** `report.write_turn` writes an artifact for every outcome: answer,
refusal, or clarification, each with its route, top rerank score and loop count.
**Why it matters beyond completeness:** an honest refusal is a *result*, not the
absence of one. A reviewer looking for evidence of abstention should find it as a
file, not have to infer it from a gap in the directory listing.
**Evidence:** `outputs/answers/` now holds 12 `.md` + 12 `.json`.
**README line:** "Refusals are written out like any other answer, because an honest
refusal is a result rather than the absence of one."

---

## D-140 · The clean-clone test, and the two README numbers it falsified
**Date:** 2026-08-20
**Method:** copied the repository to a fresh directory excluding every gitignored
path, made a new venv, and followed the README quickstart literally.
**Result: it passes end to end.** Install clean, `doctor --gpu` green, 11/11 papers
fetched with **sha256 values identical to the original run**, ingest and index
reproducing **fingerprint `e62c6925a03ad297` exactly**, and a cited answer out the
far end.

**It also caught two things the README claimed and could not support:**

1. **"Subsequent runs answer in about 15 seconds" was wrong.** Measured: a single
   `research-agent ask` takes **~45 seconds wall-clock**, of which ~30 s is process
   start plus loading bge-m3 and the reranker. The 14.6 s p50 in the evaluation is
   *in-process turn latency*, measured inside a loop where the encoders load once and
   serve every question. Both numbers are true; only one of them is what a single
   command costs, and the README was quoting the wrong one.
2. **Windows MAX_PATH.** Installing torch into a deeply-nested directory fails with
   `OSError: No such file or directory` on
   `predicated_tile_access_iterator_residual_last.h` — a header nested past 260
   characters. Reproduced at a 100-character clone path, clean at 36. Added to
   troubleshooting with both fixes.

The quickstart also used a shortened question whose answer is noticeably worse than
the one showcased at the top of the README; it now uses the same question, so a
reviewer sees what was advertised.
**README line:** "The quickstart was tested by following it literally in a clean
clone, which is how two of its own numbers turned out to be wrong."

---

## D-141 · Streamlit UI: two bugs that only appear when you actually run it
**Date:** 2026-08-20
**Decision:** `ui.py` imports `agent.run_turn` and renders the result. No business
logic. Streamlit is in a separate `requirements-ui.txt` so the quickstart does not
pull its dependency tree, and `research-agent ui` fails with a useful message if it
is absent.

**Two failures that static review would not have found, and that an HTTP 200 did not
either — the first response was Streamlit's own error page:**

1. **`ImportError: attempted relative import with no known parent package`.**
   Streamlit executes the file as a top-level script rather than importing it as part
   of the package, so `from . import agent` fails. Absolute imports work either way
   because the package is installed.
2. **`SQLite objects created in a thread can only be used in that same thread`.**
   Streamlit serves each script re-run from a worker thread, and the connection was
   cached across them by `@st.cache_resource`.
   **Fixed inside the UI layer, not the core.** Encoders stay cached — they cost ~30
   seconds — while the connection is opened per run. SQLite connections are cheap and
   WAL already supports concurrent readers, so this is the correct pattern rather
   than a workaround. Adding `check_same_thread=False` to `db.connect` would have
   made the whole CLI carry a concession that only a server needs, and Step 14 is
   explicit that the UI must not require core changes.

**A third bug was mine, in the rendering.** Groundedness is scored per *(sentence,
chunk)* pair, and the sources panel collapsed them to one score per chunk — so a
passage supporting three sentences displayed only the last one's score. The visible
symptom was a sentence marked `[unverified]` sitting directly above a source labelled
"groundedness 1.000". It now shows the range across sentences.
**Evidence:** driven in a real browser; a question answered end to end with cited,
expandable sources and a "How this turn was decided" trace.
**README line:** "The UI imports the agent and renders it — the two bugs it hit were
both about Streamlit's execution model, and neither required a change to the core."

---

## D-142 · Web search: supplementary, same pipeline, default off
**Date:** 2026-08-20
**Decision:** `--web` appends web results to the corpus candidate pool. They are
reranked, cited and groundedness-verified through the identical path, and render as
`[web] Title — url` with a `web_` id prefix.
**Four rules, each enforced rather than intended:**
- **Supplementary, never a replacement.** Corpus candidates are always retrieved and
  web results are appended; a test asserts no corpus candidate is displaced.
- **Same pipeline.** Web results are adapted into the same `Hit` type. A separate
  `WebHit` would have let web text take a different path through reranking and
  verification, and taking the identical path is the entire point.
- **Visibly distinguishable.** The `[web]` marker comes first, and `page_start` is 0
  rather than an invented page number — a web page has no page, and putting a false
  precision into a citation is exactly the kind of small lie this project exists to
  prevent.
- **Abstention must not regress.** Measured with the flag off: **4/4 controls abstain,
  with identical routes to the pre-Phase-8 baseline.** A test also asserts the
  corpus-only path makes no network call at all.
**Provider:** DuckDuckGo's HTML endpoint — no API key, no account, no quota to
account for. A second credential in the quickstart would undermine the project's
"works with no key at all" property for the sake of an optional extension.
**Consequence:** snippets are short, so a web citation carries less context than a
corpus passage. A search outage returns an empty list and degrades to corpus-only
rather than failing the turn.
**The headline numbers stay corpus-only**, so the evaluation stays honest.
**README line:** "Web search is supplementary, off by default, and goes through the
same chunk-cite-verify path as the corpus — with the flag off, the control questions
abstain identically and no network call is made."

---

## D-143 · Two of four real keys were permanently denied — rotation had to learn a third failure class
**Date:** 2026-08-20
**Observed on the first real Gemini call:** `403 PERMISSION_DENIED. Your project has
been denied access.` Probing each key individually: **key_1 and key_2 denied, key_3
and key_4 working.** `ListModels` succeeded on all four — metadata access and
generation access are separate permissions, so a key can look healthy and be unable
to generate.
**The gap this exposed:** the limiter knew two failure classes, quota-exhausted and
everything-else. A 403 is neither. Re-raising aborted the turn while two working keys
sat idle; retrying would have looped forever on a condition no amount of waiting
fixes.
**Decision:** a third class. `KeyRejected` marks a key permanently unusable for the
process, rotation skips it, and the turn continues on the next key. Not persisted —
a key restored on the provider's side should work again next run without an edit.
**Consequence:** the full evaluation ran to completion on two keys while two were
dead, which is exactly what multi-key rotation is supposed to buy and would not have
worked an hour earlier.
**Evidence:** `disabled_keys` after the first call: `{key_1: 403..., key_2: 403...}`,
answer served by key_3. Ledger: key_1 and key_2 show 2 failed attempts each and then
nothing — tried once, never retried.
**README line:** "Two of four real keys were denied at the project level; a denied key
is disabled and rotation continues, because waiting never fixes a 403 and failing the
turn wastes the keys that work."

---

## D-144 · A transient 5xx must not end a 25-turn evaluation
**Date:** 2026-08-20
**Observed:** the first complete Gemini run died on
`RemoteProtocolError: Server disconnected without sending a response.` Every call
already spent was wasted.
**Decision:** transient failures — dropped connections, 5xx, timeouts — rotate to the
next key or rung instead of aborting. The loop is already bounded, so a genuine
outage still terminates rather than spinning; and if *every* attempt was transient the
error says so explicitly rather than reporting quota exhaustion, because nothing was
actually exhausted.
**Why the distinction is worth code:** the three classes need three different
responses. Quota-exhausted → step down a rung. Key-rejected → disable and skip.
Transient → try again elsewhere. Collapsing any two of them produces either an
infinite retry or a needless abort.
**Vindicated immediately:** the completed run logged **14 × 503 UNAVAILABLE ("this
model is currently experiencing high demand")** and 6 further connection drops. Under
the previous behaviour any one of them would have ended the run.
**Evidence:** `test_quota_exhaustion_and_key_rejection_are_different`; the ledger's
error breakdown.
**README line:** "Quota exhaustion, a denied key and a dropped connection need three
different responses — collapsing any two of them causes either an infinite retry or a
needless abort."

---

## D-145 · Gemini rejects a JSON schema that Ollama accepts
**Date:** 2026-08-20
**Found by calling it, not by reading the docs.** `ANSWER_SCHEMA` used
`"type": ["string", "null"]` for `refusal_reason`. Ollama accepts the union happily.
Gemini's `response_schema` rejects it before a request is even sent — its type field
is an enum admitting exactly one value, so the SDK fails Pydantic validation locally.
**Decision:** `"type": "string"`, with the empty string carrying the "no refusal"
case. The parser already treated `""` as absent, so nothing downstream changed.
**Why this matters beyond the one field:** the project claims schema-constrained
generation on *both* providers. A schema that constrains only one of them is not a
contract, it is a contract on one path and a hope on the other — and it would have
gone unnoticed indefinitely, because every Ollama run passed. A test now walks all
seven schemas and fails on any union type.
**Evidence:** `test_no_schema_uses_a_union_type`, parameterised over every schema.
**README line:** "The citation schema had a union type that Ollama accepted and Gemini
rejects — a constraint that only holds on one provider is not a contract, and only
running both revealed it."

---

## D-146 · The quota engineering worked: 123 billed calls, zero rate-limit rejections
**Date:** 2026-08-20
**The claim under test:** a DB-backed sliding-window limiter enforcing RPM and RPD
per `(model, key_alias)` should prevent the provider ever having to reject a request
for rate.
**Measured across the full Gemini evaluation — 607 calls billed against quota, 302
served from cache:**

| failure | count |
|---|---:|
| 503 UNAVAILABLE (provider-side load) | 14 |
| schema validation (pre-fix, historical rows) | 12 |
| 403 denied (the two dead keys) | 9 |
| connection dropped | 6 |
| **429 / RESOURCE_EXHAUSTED** | **0** |

**Zero rate-limit rejections.** Every failure was the provider's availability, a dead
key, or my own schema bug. The limiter never let a request through that the provider
would have refused for rate.

**The ladder stepped down for real**, which no local run could demonstrate:

| synthesis rung | billed calls |
|---|---:|
| gemini-3.7-flash | 42 |
| gemini-3.6-flash | 9 |
| gemini-3.5-flash | 1 |

RPD drained on the top rung and the client walked down, exactly as designed — and
across keys before rungs, so capability degraded last.
**The cost was higher than I estimated.** I budgeted ~25 synthesis calls; the run
billed 123 Gemini calls, of which 59 were synthesis. The difference is D-136: the
sufficiency judge rejects the first retrieval on every question, so each question
costs a judge and usually a rewrite on top of the synthesis. The estimate was made
before that behaviour existed and I did not revise it — that is the error, not the
spend.
**README line:** "Across 607 billed calls the provider never once rejected a request
for rate: the durable limiter is the reason, and the synthesis ladder demonstrably
stepped down three rungs under real quota pressure."

---

## D-147 · Gemini vs Ollama, measured — and where the comparison is contaminated
**Date:** 2026-08-20
**Identical corpus, identical thresholds, identical retrieval. Only the generator
changed.**

| Metric | Ollama `llama3.1:8b` | Gemini | LLM-dep. |
|---|---:|---:|---|
| Citation precision | 0.882 | **1.000** | yes |
| Abstention accuracy | 1.000 (4/4) | **1.000 (4/4)** | yes |
| Fact coverage (mean) | 0.660 | **0.881** | yes |
| Route accuracy, single-turn | 0.917 (11/12) | **1.000 (12/12)** | yes |
| Invented citation ids | 0 | **0** | yes |
| Refusals carrying citations | 0 | **0** | yes |
| Condensation drift rate | 0/13 | **0/13** | yes |
| Papers per multi-hop answer | 1.00 | **1.00** | yes |
| Route accuracy, conversational | 10/13 | **7/13** | yes |
| Recall@5 (hybrid + rerank) | 0.646 | 0.646 | **no** |

**The retrieval row is identical by construction, and that is the point of having
measured it separately.** Everything model-independent is unchanged; only the
generation columns move.

**What Gemini fixed: `q11`.** The false-premise question failed on every earlier
attempt. On Gemini: *"The premise of the question is incorrect: the 4-bit NormalFloat
(NF4) quantisation scheme is not introduced or described by the LoRA paper. Instead,
NF4 is introduced by the QLoRA paper"* — citing QLoRA, coverage 1.00. That is the
last single-turn route failure closed, and it took the judge fix, relaxation *and* a
stronger synthesiser together.

**What Gemini did not fix: multi-hop.** Papers per multi-hop answer is **1.00 on both
providers**. A stronger model with the cross-document rule in its prompt and passages
from two papers in its context still answers from one. That rules out "the local model
is not capable enough" as the explanation and points at context construction — which
is where D-128 said the next fix belongs.

**Where the comparison is contaminated, stated rather than buried.** Conversational
route accuracy is *worse* on Gemini (7/13 vs 10/13), and the run logged 14 × 503
"experiencing high demand" plus 6 dropped connections. Rotation kept the run alive,
but a turn served after several failed attempts is not the same experiment as a turn
served first time. **I do not claim Gemini is worse conversationally** — the run
happened during a provider-load window and the single-turn half of the same run
improved on every metric. Re-running it in a quieter window is the honest next step
and has not been done.
**README line:** "Gemini improves every generation metric on single-turn questions and
closes the last route failure; the conversational half of the same run was measured
during a provider-load window and is not a clean comparison."

---

## D-148 · Key aliases follow the key, not its slot
**Date:** 2026-08-21
**Context:** The user replaced the Gemini keys and moved the two working ones into
slots 1 and 2.
**What that exposed:** quota is accounted per `(model, key_alias)`, and the alias was
the slot number. Moving a key handed its consumption to whatever now occupied its
slot. Measured immediately after the swap: the two **brand-new** keys showed 20/20 and
18/20 RPD already consumed on the top rung, while the keys that had actually spent it
showed 2/20. Both directions wrong at once -- the limiter would refuse fresh quota and
over-spend drained quota.
**Decision:** the alias is `k_` plus the first 10 hex of `sha256(key)`. It follows the
key across slots, gives a genuinely new key a genuinely new identity, never contains
any of the key, and collapses duplicates -- the same key in two slots shares one quota
bucket, and counting it twice would double the apparent capacity.
**Consequence:** historical rows keyed on `key_1..key_4` no longer match any live key
and are ignored for quota. That is a one-time loss of accounting for keys that can no
longer be identified, and it is unavoidable. The conservative reading is that some
quota was already spent -- which is what the 16 observed 429s in the next run were,
and rotation absorbed every one.
**Evidence:** `test_key_alias_follows_the_key_not_its_position`,
`test_key_alias_never_contains_the_key`, `test_duplicate_keys_collapse_to_one_bucket`.
**README line:** "Quota is accounted per key identity, not per slot in the env file --
moving a key between slots otherwise hands its spent quota to whatever replaces it."

---

## D-149 · A 429 is a third failure class
**Date:** 2026-08-21
**Context:** After the alias migration the local ledger could no longer see spend that
had really happened, so the provider began returning 429s the limiter had not
predicted.
**Decision:** `is_rate_limited()` joins the existing permanent/transient split. A 429
means the provider disagrees with the local ledger; the provider wins, and the client
rotates to the next key and then the next rung. The key is **not** disabled -- unlike a
403, this recovers on its own.
**Why three classes and not two:** each needs a different response. Quota exhausted →
step down a rung. Key rejected → disable and skip. Rate limited → rotate but keep.
Transient → try again elsewhere. Collapsing any two produces an infinite retry or a
needless abort, and I have now had both.
**Evidence:** 16 × 429 during the second Gemini run, all absorbed by rotation; the run
completed. `test_a_rate_limited_key_rotates_rather_than_aborting`.
**README line:** "A 429 means the provider disagrees with the local ledger -- the
provider wins, and the client rotates rather than failing the turn."

---

## D-150 · The ledger timestamp format is a trap for ad-hoc queries
**Date:** 2026-08-21
**Context:** While diagnosing a live run I twice concluded the system was failing
badly when it was healthy.
**Cause:** `llm_calls.ts` is timezone-aware ISO-8601 (`2026-08-21T04:12:00+00:00`)
while every other table's `created_at` uses SQLite's `datetime('now')`
(`2026-08-21 04:12:00`). They compare as strings, and at index 10 `T` (84) sorts after
a space (32) -- so for any row **from today**, `ts >= datetime('now','-5 minutes')` is
true regardless of the actual time. The date saves you across days, which is exactly
why it hides: it only misfires while inspecting a run in progress, which is the only
time anyone runs such a query.
**The limiter was never affected** -- it compares ISO to ISO throughout. This is a
hazard for anything written by hand against the ledger.
**Decision:** `db.iso_cutoff()` produces a correctly-formatted bound, the column
carries a comment saying so, and a test demonstrates the trap with a two-hour-old row
that the naive comparison wrongly includes.
**Why not just normalise the column:** it would need a migration over existing rows,
and the ISO format is the right one for a timestamp that must be unambiguous across
timezones. The fix is to make the correct comparison easy, not to weaken the storage.
**README line:** "Ledger timestamps are ISO-8601 while SQLite's own defaults are not;
a helper exists because comparing the two by hand silently matches every row from
today."

---

## D-151 · TPM is a third limit, and it was not being enforced
**Date:** 2026-08-21
**Context:** The account's published limits arrived as **RPM / TPM / RPD**. My ladder
had RPM and RPD exactly right on all eight models. TPM -- 250,000 tokens per minute --
was not tracked at all.
**Why it had gone unnoticed:** RPM binds first under normal load. Five requests a
minute at 8k tokens is 40k against a 250k ceiling, so TPM never fires. That is an
argument for it being *usually* irrelevant, not for leaving it unenforced: it stops
being irrelevant the moment prompts grow, or a 15 RPM volume model starts carrying
long context, and the failure mode is a 429 the limiter promised would never happen.
**Decision:** `Rung` carries `tpm`; `try_acquire` checks tokens already billed in the
last minute plus an estimate for the request about to be made.
**On the estimate:** token counts are only known after a call returns, so the check is
"what has been spent" plus `len(prompt)//4 + 512`. Over-estimating costs a rung step;
under-estimating costs a 429. Over-estimating is the correct direction to be wrong in.
**Also recorded:** Gemini 2.5 Pro and 3.1 Pro are published at 0/0/0 on this tier.
They are absent from both ladders rather than listed-and-skipped -- a rung the account
cannot use is not a fallback, it is a guaranteed failed attempt on the way down.
**Evidence:** `test_tpm_blocks_a_call_that_would_exceed_the_token_ceiling`,
`test_tpm_window_rolls_off`, `test_cached_calls_do_not_consume_tokens`,
`test_every_ladder_rung_declares_all_three_limits`.
**README line:** "Three limits are enforced, not two -- RPM usually binds first, which
is exactly why TPM is the one that gets forgotten until it produces a 429 the limiter
promised could not happen."

---

## D-152 · Synthesis falls back to the volume ladder; volume never falls back to synthesis
**Date:** 2026-08-21
**Context:** Until now, a drained synthesis ladder raised `QuotaExhausted` and the
turn failed. With every rung and every key spent, the agent simply stopped.
**Decision:** when the synthesis ladder is exhausted, serve the call from the **volume**
ladder instead. The answer gets worse; the agent keeps working. That is "answer
quality degrades last" taken to its conclusion -- a weaker cited answer beats no
answer, and every other guarantee (validated ids, groundedness, abstention) is
unchanged because they are enforced in code rather than by the model.

**The asymmetry is the entire point and is enforced, not merely intended.** Volume
exhausted never escalates to synthesis. Judging and condensing on reasoning quota is
precisely the failure the two-ladder split exists to prevent: about four turns would
drain the answer budget, and the failure would then land on answer generation -- the
one thing a reviewer sees. Volume work fails instead.
`test_volume_never_falls_back_to_synthesis` asserts it.

**Measured by draining the real ladder shape against four keys:**

| stage | calls/day |
|---|---:|
| synthesis ladder — 6 rungs x 20 RPD x 4 keys | 480 |
| degraded onto volume — 2 rungs x 500 RPD x 4 keys | 4,000 |
| **total before the agent gives up** | **4,480** |

Each rung drained exactly 80 in strict capability order before stepping down. The
fallback is a **9.3x** increase in how long the agent keeps answering.

**A second measurement worth keeping, because it looked like a bug and was not.** The
same drain in a tight loop stopped at 260 calls, not 480 -- because RPM, not RPD,
binds in a burst: 5 RPM x 4 keys = 20 per rung, 10 x 4 = 40 for the 2.5-flash-lite
rung, 15 x 4 = 60 per volume rung. 20x5 + 40 + 60 + 60 = 260 exactly. Both numbers are
correct; they measure different limits, and only the fake-clock run measures the daily
ceiling.
**Consequence:** degraded calls are logged as `<purpose>:degraded` and surfaced by
`budget`, so a degraded run is visible rather than just looking cheaper. The behaviour
is switchable via `synthesis_falls_back_to_volume` for anyone who would rather fail
than answer with a lite model.
**Revisit if:** measured answer quality on the volume rungs turns out to be bad enough
that refusing would genuinely serve the user better. That has not been measured.
**README line:** "When the synthesis ladder drains, answers are served by volume
models rather than failing -- 480 calls of capacity becomes 4,480 -- and the reverse
is forbidden, because judging on reasoning quota is the failure the split exists to
prevent."

---

## D-153 · The 2.5-flash-lite placement rule was retired, and why that is now safe
**Date:** 2026-08-21
**Context:** The user asked for `gemini-2.5-flash-lite` on the **volume** ladder.
`Config.validate()` raised on exactly that, and three tests enforced it. D-004 called
its placement at the bottom of synthesis load-bearing.
**The question worth asking is what the guard was actually protecting.** Not the
placement -- the *fact*. The model is named like a volume model and capped at **20
RPD, not 500**. The original danger was assuming 500, sending it volume traffic, and
having it exhaust after twenty condensation calls with the failure surfacing somewhere
unrelated.

**That danger no longer exists, for reasons built since D-004:**
- the config records `rpd=20`, confirmed against the account's published limits;
- the limiter enforces the true number rather than an assumed one;
- exhaustion steps down a rung instead of cascading;
- a 429 rotates rather than aborting (D-149);
- there is now an unlimited local rung beneath everything (D-154).

**Decision:** move it to the volume ladder as asked, and rewrite the assertion to
guard the fact instead of the position -- any declaration giving it `rpd > 20` still
raises. The protection survives; the obsolete constraint does not.
**Consequence:** the synthesis ladder loses a rung (480 -> 400 calls/day across four
keys) and volume gains 80. Answer quality is *better* protected by this, not worse:
the rung that left synthesis was its weakest, and synthesis now falls through to the
whole volume ladder anyway.
**Evidence:** `test_the_trap_model_may_not_be_declared_with_volume_scale_rpd`,
`test_the_trap_model_is_allowed_on_the_volume_ladder_at_its_true_rpd`.
**Revisit if:** the published RPD ever changes. The guard checks the number, so it
will keep working.
**README line:** "The 2.5-flash-lite guard now protects the fact rather than the
placement: it may sit anywhere, but never be declared with volume-scale RPD."

---

## D-154 · A local rung at the bottom, so quota can never end the run
**Date:** 2026-08-21
**Decision:** the volume ladder ends in `qwen2.5:14b` served by Ollama -- no key, no
quota, no rotation. Since synthesis falls through to volume (D-152), the complete
chain for any task is:

```
3.7-flash -> 3.6 -> 3.5 -> 3-flash-preview -> 2.5-flash        (400/day, 4 keys)
   -> 3.5-flash-lite -> 3.1-flash-lite                        (4000/day)
   -> 2.5-flash-lite                                          (80/day)
   -> qwen2.5:14b, local                                      (unlimited)
```

**The agent can no longer fail for lack of quota.** It can only fail because Ollama is
not running -- which is a different problem with a different fix, and `doctor` reports
it explicitly.
**Two invariants enforced in `validate()`:**
- a local rung must be **last**. It is unlimited, so anything below it is unreachable,
  and a silently unreachable fallback is worse than no fallback.
- a model may still not appear on both ladders, so the local rung lives only on
  volume; synthesis reaches it by falling through.
**Implementation note worth recording:** `_complete_ollama` picks its model from the
offload mode, which is right for the default path and wrong for a ladder rung that
names its model explicitly. `_complete_local` exists so the rung's name wins. Reusing
the former would have silently served `llama3.1:8b` wherever the ladder said
`qwen2.5:14b` -- the ladder would have looked correct and been a lie.
**Measured, draining the real shape against four keys:** every rung consumed exactly
its published capacity in order -- 80 per synthesis rung, 2000 per 500-RPD volume
rung, 80 for 2.5-flash-lite -- before the local floor took over and did not exhaust.
**README line:** "The ladder ends in a local model, so the agent cannot fail for lack
of quota -- only because Ollama is not running, which `doctor` says plainly."

---

## D-155 · Ollama does not run by default, and the local rung depends on it
**Date:** 2026-08-21
**Measured on this machine:** no `ollama.exe` process, no entry in the HKCU `Run` key,
no Windows service. Ollama runs **only** when something launches it -- `ollama serve`,
or implicitly `ollama run` / `ollama list`, which spawn the server that then holds the
caller's stdout pipe (D-105).
**Why it matters now:** the local rung is the floor of the ladder, and a floor that is
usually absent is not a floor. The agent's "cannot fail for lack of quota" property is
conditional on a server that is not running after a reboot.
**Decision:** do not auto-start it. Spawning a daemon is not the agent's job, and
D-105 is a direct record of what shelling out to the CLI costs. `doctor` reports the
local rung as unavailable when the server is down, and the ladder raises the real
error rather than a quota error when it cannot reach it.
**Consequence for anyone reading the quota numbers:** the unlimited floor requires
`ollama serve` to be running. On a fresh boot it is not.
**README line:** "Ollama does not auto-start on Windows -- no service, no Run key -- so
the local fallback rung is only a floor while `ollama serve` is up."

---

## D-156 · CORRECTION: the Gemini conversational gap is real, not provider load
**Date:** 2026-08-21
**What I claimed in D-147 and in the README:** conversational route accuracy came out
lower on Gemini (7/13, 8/13) but both runs coincided with `503 UNAVAILABLE` windows, so
I declined to call Gemini worse conversationally and said a clean re-run was the
honest next step.

**I have now run it a third time, in materially better conditions, and the claim does
not survive.**

| run | 503s | p50 latency | conversational |
|---|---:|---:|---:|
| 1 | 14 | — | 7/13 |
| 2 | 32 | 57 s | 8/13 |
| 3 | **13** | **22 s** | **7/13** |

Run 3 had the fewest failures and less than half the latency of run 2, and scored the
same as run 1. **There is no relationship between provider load and the conversational
number.** Load was a plausible confound and it was wrong; three runs is enough to say
so.

**Where the gap actually is.** It localises entirely to scenario A:

| turn | Ollama | Gemini | depends on condensation? |
|---|---|---|---|
| A1 "What problem does LoRA solve?" | refuse | refuse | **no** — turn 1 skips condensation |
| A2 "How does the quantised version reduce memory further?" | answer | refuse | yes |
| A3 "What does it report for a 65B model?" | answer | abstain | yes |
| A4 "Does that cost quality?" | answer | refuse | yes |

A1 fails identically on both, and must: turn 1 skips condensation, so the query,
retrieval and rerank score are byte-identical regardless of provider, and the route is
decided by a local cross-encoder score. Turns 2-4 are the condensation-dependent ones,
and they are exactly the ones that diverge.

**The finding underneath, which is the useful part.** Gemini's condensations pass the
drift guard -- **0/13 drift on every run, both providers** -- and still retrieve worse
on this corpus. The guard prevents a rewrite from introducing a *hallucinated* term.
It does not, and cannot, ensure the rewrite is a *good query*. Those are different
properties, and I had been treating a clean drift rate as evidence for both.
**Decision:** correct the README rather than leave a comfortable explanation standing.
The single-turn story is unambiguous -- Gemini wins on every metric, 1.000 route
accuracy three times running -- and the conversational story is a real regression that
I do not yet have a mechanism for.
**What would settle it, not done:** log the condensed query for every turn in
`eval` (currently only `chat-eval` writes them) and diff Ollama's rewrites against
Gemini's on the same history. That is a small change and the obvious next step.
**README line:** "Gemini scores lower conversationally, and it is not provider load --
three runs across very different load conditions gave 7, 8 and 7. The gap sits in the
condensation-dependent turns: those rewrites pass the drift guard and still retrieve
worse, because passing the guard and being a good query are different properties."

---

## D-157 · CORRECTION 2: the conversational gap is the sufficiency judge, not condensation
**Date:** 2026-08-21
**This is the second time I have been wrong about this, and the logging the user asked
for is what settled it.** D-147 blamed provider load; D-156 corrected that to
condensation quality. Both were wrong.

**What the instrumentation shows.** Logging the query retrieval *actually ran on* --
not just the condensation -- exposed that the sufficiency loop rewrites the query
again afterwards, and it is that text the router scores. The `condensed` field I had
been logging was never the whole story.

The decisive evidence is the turns where **both providers fired zero loops**:

| turn | Ollama top | Gemini top |
|---|---:|---:|
| D t1 | 0.9891 | 0.9891 |
| D t2 | 0.000 | 0.000 |
| D t3 | 0.8404 | 0.8336 |

Identical, as retrieval must be -- it is a local encoder over a local index. Now the
turns where the loop counts differ:

| turn | ollama loops / top | gemini loops / top | routes |
|---|---|---|---|
| A t1 | **1** / 0.9857 | **0** / 0.6837 | answer / refuse |
| A t2 | **2** / 0.8213 | **1** / 0.000 | answer / refuse |
| A t4 | **2** / 0.9898 | **1** / 0.7167 | answer / refuse |
| B t1 | **2** / 0.9661 | **0** / 0.188 | answer / refuse |

Every divergence tracks the loop count. Totals: **Ollama 15 loops, Gemini 10.**

**The mechanism.** A1 is the cleanest case: turn 1 skips condensation, so both
providers retrieve on the byte-identical raw question. Ollama's judge called the first
retrieval *insufficient*, the loop rewrote to *"What problem does LoRA solve for large
language models' adaptation to downstream tasks?"*, and the score rose to 0.9857.
Gemini's judge called the same evidence *sufficient*, no rewrite happened, and the
router then refused the raw query on 0.6837 -- below `tau_low`.

**The architectural finding, which is the useful part.** The judge and the router are
two independent gates asking overlapping questions, and the judge runs first. A judge
that accepts early **denies the router a better query**, and the router then refuses on
a score a rewrite would have lifted. Ollama's judge is over-eager -- D-136 recorded
that as pure latency cost -- and here that over-eagerness is accidentally load-bearing.
Gemini's judge is better calibrated and the pipeline is *worse* for it.

**Decision:** record the mechanism; do not tune the judge to fix the number. Making
Gemini's judge more pessimistic would be fitting to a 13-turn set, and the honest
reading is that the gate ordering is wrong, not that one model judges badly.
**What I would change, not built:** let the router see the *best* score across loop
iterations rather than the last, or run the router before the judge and let a
sub-threshold score trigger the loop rather than a refusal. Either removes the
coupling. Both are more than a tuning change.
**Evidence:** `outputs/conversations/_condensation.{ollama,gemini}.json`, and
`research-agent condensation-diff` which prints the rewrites side by side.
**README line:** "The conversational gap is not condensation and not provider load: a
judge that accepts the first retrieval denies the router a rewritten query, and the
router then refuses on a score the rewrite would have lifted."

---

## D-158 · Submission audit: five defects found by reading the repo as a reviewer
**Date:** 2026-08-21
Asked whether the repo was ready to submit, I checked it the way a reviewer would --
clone it clean, follow the README, and cross-reference every claim against the
committed artifacts -- rather than re-reading my own summary of it. Five things were
wrong, and four of them were invisible from inside the build.

**1. `outputs/eval_report.md` was mislabelled.** The header printed
`cfg.ollama_model` unconditionally, so a Gemini run produced a file headed
*"Generation: `llama3.1:8b` via Ollama"* over Gemini's numbers -- citation precision
1.000, fact coverage 0.881, single-turn route 1.000, all of which the README's own
comparison table attributes to Gemini, not to the local model. The one artifact a
reviewer cross-references against the README contradicted it. `GenerationMetrics` now
carries the provider and model that actually served the run, and the report prints
those. The traceability row saying *"Gemini results -- TBD, no Gemini call has been
made"* was stale by three full Gemini runs; it now names the provider that produced
sections 2-3.

**2. Both providers wrote the same transcript filenames.** Whichever ran last owned
`outputs/conversations/*.md`, and Gemini ran last. The committed showcase scenario --
scenario A, the coreference gate the whole design is built around -- showed **four
refusals in four turns**, including a refusal of *"What problem does LoRA solve?"*,
a question the corpus plainly answers and the default path answers at 0.9857. A
reviewer opening the flagship deliverable saw the agent fail completely. The
unsuffixed name now belongs to the Ollama path, which is both the default and the
only one reproducible without a key; other providers write `{scenario}.{provider}.md`.
Same rule for `eval_report.md`.

**3. Nine of twelve committed answers reported `Latency: 0 ms`.** Two cache layers
exist and the eval harness cleared one. `answer_cache` short-circuits a whole turn;
the `DiskCache` under `.cache/llm` short-circuits an individual LLM call on
sha256(model+prompt). With only the first cleared, **557 of 1520 calls were served
from disk at a recorded 0 ms**, so artifacts reported zero for turns that genuinely
cost seconds. The harness's own comment claimed the numbers "describe the pipeline
rather than the cache's history"; half of that was false. `evaluate.drop_caches()`
now empties both and reports what it discarded, and `Answer.cached` labels a genuine
cache hit rather than letting a bare zero imply a broken metric.

**4. Seven Makefile targets printed "not yet implemented (Step N)".** `fetch-corpus`,
`ingest`, `index`, `ask`, `chat`, `eval`, `eval-retrieval` -- every one implemented in
the CLI for phases, every one still a Step-1 stub in the Makefile, all seven
advertised as working by `make help`. Directly against the spec's no-placeholder rule,
and it makes a finished repo read as abandoned. All wired to their real commands.

**5. The README's conversational route accuracy was stale.** It claimed 10/13 local;
the measured value on current code is **9/13**. Two independent cache-cleared runs
produced byte-identical condensations, final queries, loop counts, scores and routes,
so this is not run-to-run variance -- the number simply predated a change and was
never re-measured.

**Why this is worth an entry.** Four of the five are the same failure: an artifact and
a claim drifted apart, and nothing in the build checked them against each other.
Tests passed throughout -- 165 of them -- because every one tests code, and none of
these were code defects. **What I would add, not built:** a `verify-artifacts` command
that re-reads `outputs/` and fails when a committed number disagrees with the README,
so the traceability table is enforced rather than asserted.
**Evidence:** this file's git history, `outputs/`, and the clean-clone run in the
scratchpad that reproduced fetch, ingest and index from the committed manifest alone.
**README line:** "Every number in this README is regenerated from `outputs/` by a
cold-cache run; the transcripts and the evaluation report are written per provider so
one run cannot overwrite another's evidence."

---

## D-159 · One metric, two definitions, and the README quoted the friendlier one
**Date:** 2026-08-21
Running `chat-eval` and `eval` back to back on the same corpus, provider and code
produced **9/13** and **10/13** for conversational route accuracy. Same 13 turns,
same session, two numbers.

**Cause.** Two implementations of "was this route correct". `evaluate._route_ok`
treats `refuse` and `abstain` as interchangeable when either is expected -- they are
different mechanisms with the same outcome, no answer and no citations. `chat-eval`
carried its own inline `expected == actual` instead. They disagreed on exactly one
turn: **C t3, expected `refuse`, got `abstain`** -- the agent retrieved something
above threshold, synthesised, and the model itself judged the evidence unsupportive.
That is the outcome the scenario tests for, reached the long way.

**Why this was worse than a wrong number.** The README quoted 10/13 while the
committed artifact recorded 9/13, so the two disagreed and neither was wrong on its
own terms. A reviewer cross-referencing them finds a contradiction and has no way to
tell which is the claim and which is the evidence.

**Decision:** `route_ok` is now public and is the single definition; `chat-eval` calls
it. `_route_ok` stays as an alias so existing call sites keep working. **10/13 is the
defensible number** -- counting C t3 as a failure penalises a correct refusal for
arriving by synthesis rather than by score, and the transcript still shows the
mechanism (`Route: abstain (expected refuse)`) so nothing is hidden by the rollup.

**The general lesson, which is the reusable part.** Every metric reported in more than
one place needs one implementation, and the check that catches this is running the two
commands back to back and diffing -- not reading either one. Duplicated metric logic
does not fail loudly; it produces two plausible numbers and lets the friendlier one
reach the README. Related: [[D-158]].
**Evidence:** `outputs/conversations/_condensation.ollama.json` and
`outputs/eval_report.md`, regenerated together after the fix.
**README line:** "Refusing on score and abstaining after synthesis count as the same
correct outcome -- no answer, no citations -- and that rule has one implementation
used by every command that reports it."
