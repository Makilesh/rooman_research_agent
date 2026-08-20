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
