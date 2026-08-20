# What is the NF4 NormalFloat data type in QLoRA, and in what sense is it optimal?

The NormalFloat (NF) data type builds on Quantile Quantization which is an information-theoretically optimal data type that ensures each quantization bin has an equal number of values assigned from the input tensor.[^1] For our data type, we set the arbitrary range [−1, 1]. As such, both the quantiles for the data type and the neural network weights need to be normalized into this range.[^1] The information theoretically optimal data type for zero-mean normal distributions with arbitrary standard deviations σ in the range [−1, 1] is computed as follows: (1) estimate the 2k + 1 quantiles of a theoretical N(0, 1) distribution to obtain a k-bit quantile quantization data type for normal distributions[^1] We present QLORA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance.[^2] The NormalFloat data type significantly improves the bit-for-bit accuracy gains compared to regular 4-bit Floats. While Double Quantization (DQ) only leads to minor gains, it allows for a more fine-grained control over the memory footprint[^3]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.3-5 — `p_qlora_0002`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`
[^3]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.6-8 — `p_qlora_0004`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.9895
- Sentence verification: verified 5
