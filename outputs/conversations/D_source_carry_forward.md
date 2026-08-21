# D_source_carry_forward — Referring to a previous answer's source by position

Session `s_f6c8774d7d96` · corpus `e62c6925a03ad297`

## Turn 1

**User:** How does RAG combine retrieval with generation?

*Condensation:* skipped — turn 1 — nothing to condense

*Route:* `answer` (expected `answer`) — top rerank 0.989 >= tau_high 0.786

# How does RAG combine retrieval with generation?

RAG combines parametric memory from a pre-trained sequence-to-sequence transformer (BART) with non-parametric memory from a dense vector index of Wikipedia accessed via a neural retriever (DPR) in an end-to-end trained probabilistic model.[^1] For an input query, the retriever uses Maximum Inner Product Search (MIPS) to identify the top-K latent documents.[^1] The seq2seq generator then conditions on these retrieved latent documents together with the input query to generate the output.[^1] Finally, RAG marginalizes over the latent documents using a top-K approximation, either on a per-output basis where the same document conditions the whole sequence, or on a per-token basis where different documents can condition different tokens.[^1]

## Sources

[^1]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.1-2 — `p_rag_0000`

---

- Provider: `gemini` · model: `gemini-3.6-flash`
- Latency: 0 ms
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

*Condensed to:* `Which paper was RAG combined retrieval with generation from?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.834 >= tau_high 0.786

# Which paper was RAG combined retrieval with generation from?

Retrieval-Augmented Generation (RAG) is introduced in the paper 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks' by Patrick Lewis et al.[^1] It combines retrieval and generation by pairing a pre-trained neural retriever (Dense Passage Retriever, or DPR) with a pre-trained sequence-to-sequence model (BART) as the generator.[^1]

## Sources

[^1]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.1-2 — `p_rag_0000`

---

- Provider: `gemini` · model: `gemini-3.6-flash`
- Latency: 0 ms
- Passages in context: 2
- Top rerank score: 0.8336
- Sentence verification: verified 2

