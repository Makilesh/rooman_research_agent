# According to the compute-optimal scaling work, how should model size and training tokens be scaled together, and how does that compare with how GPT-3 was trained?

For every doubling of model size the number of training tokens should also be doubled.[^1] By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally.[^1] Based on our estimated compute-optimal frontier, we predict that for the compute budget used to train Gopher, an optimal model should be 4 times smaller, while being training on 4 times more tokens.[^1]

## Sources

[^1]: Training Compute-Optimal Large Language Models · pp.1-3 — `p_chinchilla_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 2
- Top rerank score: 0.9941
- Sentence verification: verified 3
