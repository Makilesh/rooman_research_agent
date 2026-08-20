# What rank does LoRA use in its GPT-3 175B experiments, and which weight matrices does it adapt?

We set a parameter budget of 18M (roughly 35MB if stored in FP16) on GPT-3 175B, which corresponds to r = 8 if we adapt one type of attention weights or r = 4 if we adapt two types.[^1] In the Transformer architecture, there are four weight matrices in the self-attention module (Wq, Wk, Wv, Wo) and two in the MLP module. We limit our study to only adapting the attention weights for downstream tasks.[^2]

## Sources

[^1]: LoRA: Low-Rank Adaptation of Large Language Models · pp.10-12 — `p_lora_0007`
[^2]: LoRA: Low-Rank Adaptation of Large Language Models · pp.4-5 — `p_lora_0002`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.9842
- Sentence verification: verified 2
