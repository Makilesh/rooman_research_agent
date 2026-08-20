"""Streamlit chat UI — a thin read-only view over the existing agent.

**No business logic lives here.** This module imports `agent.run_turn` and renders
what it returns. If this file ever needs a change in core code to work, that is a
signal the core API is wrong and should be reported rather than worked around.

Optional. Nothing in Phases 0-7 depends on it, and `pip install -r requirements.txt`
does not install Streamlit — see `requirements-ui.txt`.
"""

from __future__ import annotations

import os

import streamlit as st

# ABSOLUTE imports, not relative. Streamlit executes this file as a top-level
# script rather than importing it as part of the package, so `from . import ...`
# raises "attempted relative import with no known parent package". The package is
# installed (`pip install -e .`), so absolute imports work either way.
from research_agent import agent, conversation, db, index as index_mod
from research_agent import rerank as rerank_mod, report
from research_agent.config import Config
from research_agent.llm import LLMClient


@st.cache_resource(show_spinner="Loading encoders (once per session)...")
def _models(_cfg: Config):
    """Load both encoders exactly once per server process.

    Streamlit re-runs the whole script on every interaction, so without this the
    encoders would reload on every keystroke -- about 30 seconds each time.
    """
    return (index_mod.load_embedder(_cfg), index_mod.load_vectors(_cfg),
            rerank_mod.load_reranker(_cfg))


def _boot():
    """Config and encoders cached; the DB connection deliberately is NOT.

    SQLite connections are bound to the thread that created them, and Streamlit
    serves each script re-run from a worker thread that is not necessarily the one
    that ran the last. Caching the connection raises
    `SQLite objects created in a thread can only be used in that same thread`
    on the second interaction.

    Opening a connection per run is the correct pattern rather than a workaround:
    SQLite connections are cheap, WAL mode already supports concurrent readers, and
    it keeps the threading concern inside the UI layer where it arises. The core
    `db.connect` is unchanged -- the CLI is single-threaded and should not carry a
    `check_same_thread=False` that only a server needs.
    """
    cfg = Config.load()
    conn = db.connect(cfg)
    db.migrate(conn)
    return cfg, conn, _models(cfg)


def _sources_panel(answer) -> None:
    """Every cited passage, expandable, with its verification scores attached.

    Groundedness is scored per (sentence, chunk) PAIR, not per chunk. An earlier
    version collapsed them into one score per chunk, so a passage supporting three
    sentences displayed only the last one's score -- which produced the visibly
    contradictory combination of a sentence marked `[unverified]` sitting above a
    source labelled "groundedness 1.000". The weakest score is what determines
    whether a sentence is trusted, so the range is shown.
    """
    cited = answer.cited_chunk_ids
    per_chunk: dict[str, list[float]] = {}
    for sentence in answer.sentences:
        for cid, score in sentence.verify_scores.items():
            per_chunk.setdefault(cid, []).append(score)

    for hit in answer.hits:
        if hit.chunk_id not in cited:
            continue
        scores = per_chunk.get(hit.chunk_id, [])
        label = f"{hit.source_label} — `{hit.chunk_id}`"
        if len(scores) == 1:
            label += f"  ·  groundedness {scores[0]:.3f}"
        elif scores:
            label += (f"  ·  groundedness {min(scores):.3f}–{max(scores):.3f} "
                      f"across {len(scores)} sentences")
        with st.expander(label):
            st.write(" ".join(hit.text.split()))


def main() -> None:
    st.set_page_config(page_title="Cited Research Agent", page_icon="📄",
                       layout="centered")
    cfg, conn, models = _boot()
    fingerprint = index_mod.load_fingerprint(conn)

    stale = index_mod.check_staleness(cfg, conn)
    if stale:
        st.error(f"Index is stale: {stale}\n\nRun `research-agent ingest && "
                 f"research-agent index`.")
        st.stop()

    if "session_id" not in st.session_state:
        st.session_state.session_id = conversation.new_session(conn, fingerprint)
        st.session_state.history = []

    with st.sidebar:
        st.subheader("Session")
        st.code(st.session_state.session_id, language=None)
        st.caption(f"corpus `{fingerprint}` · {cfg.ollama_model}")
        st.caption(f"tau_low {cfg.tau_low} · tau_high {cfg.tau_high} "
                   f"· tau_verify {cfg.tau_verify}")
        if st.button("New session", use_container_width=True):
            st.session_state.session_id = conversation.new_session(conn, fingerprint)
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.subheader("Corpus")
        for r in db.all_rows(conn, "SELECT title, n_pages FROM documents ORDER BY title"):
            st.caption(f"· {r['title'][:52]} ({r['n_pages']}p)")

    st.title("Cited Research Agent")
    st.caption("Every sentence is cited to a chunk and page. When the corpus cannot "
               "answer, it says so.")

    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    prompt = st.chat_input("Ask about the corpus...")
    if not prompt:
        return

    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, judging, synthesising, verifying..."):
            client = LLMClient(cfg=cfg, conn=conn,
                               api_keys=LLMClient.load_keys(cfg, dict(os.environ)))
            try:
                turn = agent.run_turn(cfg, conn, client, models,
                                      st.session_state.session_id, prompt)
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                return

        if turn.clarification:
            body = f"**{turn.clarification}**"
            st.warning(turn.clarification)
        elif turn.answer is None or turn.answer.is_refusal:
            reason = (turn.answer.refusal_reason if turn.answer
                      else turn.route.reason)
            body = f"**The sources do not contain an answer.**\n\n{reason}"
            st.info(body)
        else:
            body = report.render_answer_markdown(turn.answer)
            st.markdown(body.split("## Sources")[0].split("\n", 1)[-1])
            st.divider()
            st.caption("Sources")
            _sources_panel(turn.answer)

        with st.expander("How this turn was decided"):
            st.write(f"**Route** `{turn.decision}` — {turn.route.reason}")
            if turn.condensed.used_llm:
                st.write(f"**Condensed to** `{turn.condensed.query}`")
                st.write(f"**Drift guard** "
                         f"{'FELL BACK' if turn.condensed.drifted else 'passed'} — "
                         f"{turn.condensed.reason}")
            else:
                st.write(f"**Condensation** skipped — {turn.condensed.reason}")
            if turn.resolved_refs:
                st.write(f"**Resolved source references** `{turn.resolved_refs}`")
            st.write(f"**Retrieval loops** {turn.n_loops} · "
                     f"**latency** {turn.latency_ms} ms")
            if turn.trace is not None:
                st.code("\n".join(turn.trace.render()), language=None)

    st.session_state.history.append({"role": "assistant", "content": body})


if __name__ == "__main__":
    main()
