# How many attention heads does the base Transformer use, and what is the dimension of each head?

In this work we employ h = 8 parallel attention layers, or heads.[^1] For each of these we use dk = dv = dmodel/h = 64.[^2][^1]

## Sources

[^1]: Attention Is All You Need · pp.5-7 — `p_attention_0002`
[^2]: Attention Is All You Need · pp.3-5 — `p_attention_0001`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 5030 ms
- Passages in context: 3
- Top rerank score: 0.8626
- Sentence verification: verified 2
