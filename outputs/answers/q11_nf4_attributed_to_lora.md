# Which section of the LoRA paper describes its 4-bit NF4 quantisation scheme?

The NormalFloat (NF) data type builds on Quantile Quantization which is an information-theoretically optimal data type that ensures each quantization bin has an equal number of values assigned from the input tensor.[^1] 4-bit NormalFloat (NF4) quantization and Double Quantization are used in QLORA to reduce memory footprint without degrading performance.[^2] The 4-bit NormalFloat (NF4) data type is information-theoretically optimal, it still needs to be determined if this property translates to empirical advantages.[^3]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.3-5 — `p_qlora_0002`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`
[^3]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.6-8 — `p_qlora_0004`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 8110 ms
- Passages in context: 3
- Top rerank score: 0.7886
- Sentence verification: verified 3
