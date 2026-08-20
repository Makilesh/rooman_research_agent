# How does RAG combine a dense retriever with a generator, and which retriever does it build on?

We build RAG models where the parametric memory is a pre-trained seq2seq transformer, and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever.[^1] The retriever (Dense Passage Retriever [26], henceforth DPR) provides latent documents conditioned on the input, and the seq2seq model (BART [32]) then conditions on these latent documents together with the input to generate the output.[^1] We combine a pre-trained retriever (Query Encoder + Document Index) with a pre-trained seq2seq model (Generator) and fine-tune end-to-end.[^2] `[unverified]`

## Sources

[^1]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.1-2 — `p_rag_0000`
[^2]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.5-6 — `p_rag_0003`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.9835
- Sentence verification: unverified 1, verified 2
