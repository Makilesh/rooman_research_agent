# How many attention heads does the base Transformer use, and what is the dimension of each head?

In this work we employ h = 8 parallel attention layers, or heads.[^1] We found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values.[^1] For each of these we use dk = dv = dmodel/h = 64.[^2][^3]

## Sources

[^1]: Attention Is All You Need · pp.3-5 — `p_attention_0001`
[^2]: Attention Is All You Need · pp.5-7 — `p_attention_0002`
[^3]: Attention Is All You Need · pp.9-10 — `p_attention_0004`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 7546 ms
- Passages in context: 3
- Top rerank score: 0.9612
- Sentence verification: verified 3
