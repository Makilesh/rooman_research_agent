# How does QLoRA reduce memory beyond what LoRA alone achieves?

QLORA introduces multiple innovations designed to reduce memory use without sacrificing performance: (2) Double Quantization, a method that quantizes the quantization constants, saving an average of about 0.37 bits per parameter (approximately 3 GB for a 65B model).[^1] Additionally, we introduce Paged Optimizers, to prevent memory spikes during gradient checkpointing from causing out-of-memory errors that have traditionally made finetuning on a single machine difficult for large models.[^2]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.2-3 — `p_qlora_0001`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 6796 ms
- Passages in context: 3
- Top rerank score: 0.9147
- Sentence verification: verified 2
