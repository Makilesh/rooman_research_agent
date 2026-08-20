# How does RAG combine a dense retriever with a generator, and which retriever does it build on?

We build RAG models where the parametric memory is a pre-trained seq2seq transformer, and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever.[^1] The retrieval component pη(z|x) is based on DPR [26].[^2]

## Sources

[^1]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.1-2 — `p_rag_0000`
[^2]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · pp.7-8 — `p_rag_0005`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.9849
- Sentence verification: verified 2
