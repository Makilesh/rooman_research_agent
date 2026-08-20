# According to the compute-optimal scaling work, how should model size and training tokens be scaled together, and how does that compare with how GPT-3 was trained?

We find that model size and the number of training tokens should be scaled in equal proportions.[^1] Following Kaplan et al. (2020) and the training setup of GPT-3, many of the recently trained large models have been trained for approximately 300 billion tokens, in line with the approach of predominantly increasing model size when increasing compute.[^1]

## Sources

[^1]: Training Compute-Optimal Large Language Models · pp.1-3 — `p_chinchilla_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 5780 ms
- Passages in context: 2
- Top rerank score: 0.9962
- Sentence verification: verified 2
