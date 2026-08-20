# What similarity function does Dense Passage Retrieval use to compare a question and a passage, and why was it chosen?

We define the similarity between the question and the passage using the dot product of their vectors: sim(q, p) = EQ(q)⊺EP (p).[^1] Although more expressive model forms for measuring the similarity between a question and a passage do exist, such as networks consisting of multiple layers of cross attentions, the similarity function needs to be decomposable so that the representations of the collection of passages can be precomputed.[^1] Most decomposable similarity functions are some transformations of Euclidean distance (L2).[^2]

## Sources

[^1]: Dense Passage Retrieval for Open-Domain Question Answering · pp.2-3 — `p_dpr_0001`
[^2]: Dense Passage Retrieval for Open-Domain Question Answering · pp.6-7 — `p_dpr_0004`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 8171 ms
- Passages in context: 3
- Top rerank score: 0.9905
- Sentence verification: verified 3
