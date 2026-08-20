# What rank does LoRA use in its GPT-3 175B experiments, and which weight matrices does it adapt?

LoRA imposes a low-rank constraint on the weight updates.[^1] The key functional difference is that our learned weights can be merged with the main weights during inference, thus not introducing any latency.[^1]

## Sources

[^1]: LoRA: Low-Rank Adaptation of Large Language Models · pp.5-6 — `p_lora_0003`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 4937 ms
- Passages in context: 2
- Top rerank score: 0.9804
- Sentence verification: verified 2
