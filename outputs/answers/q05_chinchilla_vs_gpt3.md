# According to the compute-optimal scaling work, how should model size and training tokens be scaled together, and how does that compare with how GPT-3 was trained?

We find that model size and the number of training tokens should be scaled in equal proportions.[^1] Unlike Kaplan et al. (2020), our estimate is that for every doubling of model size, the number of training tokens should also double.[^1]

## Sources

[^1]: Training Compute-Optimal Large Language Models · pp.1-3 — `p_chinchilla_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 4750 ms
- Passages in context: 2
- Top rerank score: 0.9967
- Sentence verification: verified 2
