# D_source_carry_forward — Referring to a previous answer's source by position

Session `s_5dd8ddda0a9d` · corpus `e62c6925a03ad297`

## Turn 1

**User:** How does RAG combine retrieval with generation?

*Condensation:* skipped — turn 1 — nothing to condense

*Route:* `answer` (expected `answer`) — top rerank 0.989 >= tau_high 0.786

# How does RAG combine retrieval with generation?

We combine these components in a probabilistic model trained end-to-end (Fig. 1).[^1] The retriever (Dense Passage Retriever [26], henceforth DPR) provides latent documents conditioned on the input, and the seq2seq model (BART [32]) then conditions on these latent documents together with the input to generate the output.[^1] We marginalize the latent documents with a top-K approximation, either on a per-output basis (assuming the same document is responsible for all tokens) or a per-token basis (where different documents are responsible for different tokens).[^1]

## Sources

[^1]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.1-2 — `p_rag_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 2
- Top rerank score: 0.9891
- Sentence verification: verified 3


## Turn 2

**User:** Expand on the second source.

*Condensation:* skipped — ordinal reference did not resolve

*Route:* `clarify` (expected `answer`) — the previous answer cited 1 source(s), so 'the second source' does not exist

**Agent asks:** You asked about the second source, but the previous answer cited 1 source. Which one did you mean?

## Turn 3

**User:** Which paper was that from?

*Condensed to:* `Which paper is the model in We marginalize the latent documents with a top-K approximation trained end-to-end (Fig. 1) from?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.983 >= tau_high 0.786

# Which paper is the model in We marginalize the latent documents with a top-K approximation trained end-to-end (Fig. 1) from?

Our models leverage two components: (i) a retriever pη(z|x) with parameters η that returns (top-K truncated) distributions over text passages given a query x and (ii) a generator pθ(yi|x, z, y1:i−1) parametrized by θ that generates a current token based on a context of the previous i −1 tokens y1:i−1, the original input x and a retrieved passage z.[^1] The models we use are trained end-to-end as shown in Figure 1.[^1]

## Sources

[^1]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.1-2 — `p_rag_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 2
- Top rerank score: 0.9826
- Sentence verification: verified 2

