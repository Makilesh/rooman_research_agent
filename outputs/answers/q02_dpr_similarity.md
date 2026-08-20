# What similarity function does Dense Passage Retrieval use to compare a question and a passage, and why was it chosen?

Our experiments show that using triplet loss does not affect the results much.[^1] Although more expressive model forms for measuring the similarity between a question and a passage do exist, such as networks consisting of multiple layers of cross attentions, the similarity function needs to be decomposable so that the representations of the collection of passages can be precomputed.[^2] We thus choose the simpler inner product function and improve the dense passage retriever by learning better encoders.[^2]

## Sources

[^1]: Dense Passage Retrieval for Open-Domain Question Answering · pp.11-13 — `p_dpr_0009`
[^2]: Dense Passage Retrieval for Open-Domain Question Answering · pp.2-3 — `p_dpr_0001`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.9905
- Sentence verification: verified 3
