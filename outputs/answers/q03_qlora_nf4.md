# What is the NF4 NormalFloat data type in QLoRA, and in what sense is it optimal?

The NormalFloat (NF) data type builds on Quantile Quantization [15] which is an information-theoretically optimal data type that ensures each quantization bin has an equal number of values assigned from the input tensor.[^1] `[unverified]` The information theoretically optimal data type for zero-mean normal distributions with arbitrary standard deviations σ in the range [−1, 1] is computed as follows: (1) estimate the 2k + 1 quantiles of a theoretical N(0, 1) distribution to obtain a k-bit quantile quantization data type for normal distributions, (2) take this data type and normalize its values into the [−1, 1] range, (3) quantize an input weight tensor by normalizing it into the [−1, 1] range through absolute maximum rescaling.[^2] The data type is information-theoretically optimal for zero-centered normally distributed data.[^1]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.5-6 — `p_qlora_0003`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.3-5 — `p_qlora_0002`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.9828
- Sentence verification: unverified 1, verified 2
