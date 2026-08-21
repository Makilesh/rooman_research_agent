# How does QLoRA reduce memory beyond what LoRA alone achieves?

QLORA reduces the average memory requirements of finetuning a 65B parameter model from >780GB of GPU memory to <48GB without degrading the runtime or predictive performance compared to a 16bit fully finetuned baseline.[^1] QLORA achieves this by introducing Paged Optimizers, to prevent memory spikes during gradient checkpointing from causing out-of-memory errors that have traditionally made finetuning on a single machine difficult for large models.[^2]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.3-5 — `p_qlora_0002`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 7217 ms
- Passages in context: 3
- Top rerank score: 0.9675
- Sentence verification: verified 2
