# D_source_carry_forward — Referring to a previous answer's source by position

Session `s_b2c6f7e8e22d` · corpus `e62c6925a03ad297`

## Turn 1

**User:** How does RAG combine retrieval with generation?

*Condensation:* skipped — turn 1 — nothing to condense

*Route:* `answer` (expected `answer`) — top rerank 0.989 >= tau_high 0.786

# How does RAG combine retrieval with generation?

We combine a pre-trained retriever (Query Encoder + Document Index) with a pre-trained seq2seq model (Generator) and fine-tune end-to-end.[^1] For query x, we use Maximum Inner Product Search (MIPS) to find the top-K documents zi. For final prediction y, we treat z as a latent variable and marginalize over seq2seq predictions given different documents.[^1]

## Sources

[^1]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.1-2 — `p_rag_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 5530 ms
- Passages in context: 2
- Top rerank score: 0.9891
- Sentence verification: verified 2


## Turn 2

**User:** Expand on the second source.

*Condensation:* skipped — ordinal reference did not resolve

*Route:* `clarify` (expected `answer`) — the previous answer cited 1 source(s), so 'the second source' does not exist

**Agent asks:** You asked about the second source, but the previous answer cited 1 source. Which one did you mean?

## Turn 3

**User:** Which paper was that from?

*Condensed to:* `Which paper was RAG combine retrieval with generation from?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.840 >= tau_high 0.786

# Which paper was RAG combine retrieval with generation from?

We endow pre-trained, parametric-memory generation models with a non-parametric memory through a general-purpose fine-tuning approach which we refer to as retrieval-augmented generation (RAG).[^1] We build RAG models where the parametric memory is a pre-trained seq2seq transformer, and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever.[^2]

## Sources

[^1]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.1-2 — `p_rag_0000`
[^2]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.5-6 — `p_rag_0003`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 6203 ms
- Passages in context: 2
- Top rerank score: 0.8404
- Sentence verification: verified 2

