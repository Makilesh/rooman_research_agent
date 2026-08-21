# D_source_carry_forward — Referring to a previous answer's source by position

Session `s_2ebe2d571caf` · corpus `e62c6925a03ad297`

## Turn 1

**User:** How does RAG combine retrieval with generation?

*Condensation:* skipped — turn 1 — nothing to condense

*Route:* `answer` (expected `answer`) — top rerank 0.989 >= tau_high 0.786

# How does RAG combine retrieval with generation?

RAG combines a pre-trained seq2seq model as its parametric memory with a dense vector index of Wikipedia accessed via a pre-trained neural retriever as its non-parametric memory.[^1] For an input query, the retriever uses Maximum Inner Product Search (MIPS) to find the top-K latent documents conditioned on the input.[^1] The seq2seq generator model then conditions on both these latent documents and the input to generate the output.[^1] The model marginalizes over the latent documents with a top-K approximation, either on a per-output basis where the same document conditions all tokens or a per-token basis where different documents can condition different tokens.[^1]

## Sources

[^1]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.1-2 — `p_rag_0000`

---

- Provider: `gemini` · model: `gemini-3.6-flash`
- Latency: 8250 ms
- Passages in context: 2
- Top rerank score: 0.9891
- Sentence verification: verified 4


## Turn 2

**User:** Expand on the second source.

*Condensation:* skipped — ordinal reference did not resolve

*Route:* `clarify` (expected `answer`) — the previous answer cited 1 source(s), so 'the second source' does not exist

**Agent asks:** You asked about the second source, but the previous answer cited 1 source. Which one did you mean?

## Turn 3

**User:** Which paper was that from?

*Condensed to:* `Which paper was RAG combining retrieval with generation from?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.846 >= tau_high 0.786

# Which paper was RAG combining retrieval with generation from?

RAG is presented in the paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks".[^1] It combines a pre-trained neural retriever, Dense Passage Retriever (DPR), with a pre-trained sequence-to-sequence generator, BART.[^1]

## Sources

[^1]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.1-2 — `p_rag_0000`

---

- Provider: `gemini` · model: `gemini-3.6-flash`
- Latency: 30875 ms
- Passages in context: 3
- Top rerank score: 0.8459
- Sentence verification: verified 2

