# How does QLoRA reduce memory beyond what LoRA alone achieves?

QLORA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimizers to manage memory spikes.[^1] LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times compared to GPT-3 175B fine-tuned with Adam.[^2] QLORA reduces the average memory requirements of finetuning a 65B parameter model from >780GB of GPU memory to <48GB without degrading the runtime or predictive performance compared to a 16bit fully finetuned baseline.[^1] Using QLORA, we train the Guanaco family of models, with the second best model reaching 97.8% of the performance level of ChatGPT on the Vicuna [10] benchmark, while being trainable in less than 12 hours on a single consumer GPU; using a single professional GPU over 24 hours we achieve 99.3% with our largest model, essentially closing the gap to ChatGPT on the Vicuna benchmark.[^1] We propose Low-Rank Adaptation (LoRA) approach which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.[^2]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`
[^2]: LoRA: Low-Rank Adaptation of Large Language Models · pp.1-3 — `p_lora_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 14344 ms
- Passages in context: 3
- Top rerank score: 0.9809
- Sentence verification: verified 5
