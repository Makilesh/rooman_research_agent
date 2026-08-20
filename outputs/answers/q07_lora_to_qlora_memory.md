# How does QLoRA reduce memory beyond what LoRA alone achieves?

Double Quantization helps reduce the memory footprint of quantization constants.[^1] Double Quantization treats quantization constants cFP32 as inputs to a second quantization. This second step yields the quantized quantization constants cFP8 and the second level of quantization constants cFP32[^1] On average, for a blocksize of 64, this quantization reduces the memory footprint per parameter from 32/64 = 0.5 bits, to 8/64 + 32/(64 · 256) = 0.127 bits, a reduction of 0.373 bits per parameter.[^1]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.5-6 — `p_qlora_0003`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.9791
- Sentence verification: verified 3
