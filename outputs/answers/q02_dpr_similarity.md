# What similarity function does Dense Passage Retrieval use to compare a question and a passage, and why was it chosen?

Although more expressive model forms for measuring the similarity between a question and a passage do exist, such as networks consisting of multiple layers of cross attentions, the similarity function needs to be decomposable so that the representations of the collection of passages can be precomputed.[^1] Inner product search has been widely used and studied, as well as its connection to cosine similarity and L2 distance (Mussmann and Ermon, 2016; Ram and Gray, 2012).[^1] As our ablation study finds other similarity functions perform comparably (Section 5.2; Appendix B), we thus choose the simpler inner product function and improve the dense passage retriever by learning better encoders.[^1]

## Sources

[^1]: Dense Passage Retrieval for Open-Domain Question Answering · pp.2-3 — `p_dpr_0001`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 8561 ms
- Passages in context: 2
- Top rerank score: 0.9916
- Sentence verification: verified 3
