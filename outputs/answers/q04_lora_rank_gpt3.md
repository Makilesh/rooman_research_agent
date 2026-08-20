# What rank does LoRA use in its GPT-3 175B experiments, and which weight matrices does it adapt?

We set a parameter budget of 18M (roughly 35MB if stored in FP16) on GPT-3 175B, which corresponds to r = 8 if we adapt one type of attention weights or r = 4 if we adapt two types.[^1] In principle, we can apply LoRA to any subset of weight matrices in a neural network to reduce the number of trainable parameters. We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity and parameter-efficiency.[^2]

## Sources

[^1]: LoRA: Low-Rank Adaptation of Large Language Models · pp.10-12 — `p_lora_0007`
[^2]: LoRA: Low-Rank Adaptation of Large Language Models · pp.4-5 — `p_lora_0002`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 7921 ms
- Passages in context: 3
- Top rerank score: 0.9849
- Sentence verification: verified 2
