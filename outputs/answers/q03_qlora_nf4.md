# What is the NF4 NormalFloat data type in QLoRA, and in what sense is it optimal?

The NormalFloat (NF) data type builds on Quantile Quantization [15] which is an information-theoretically optimal data type that ensures each quantization bin has an equal number of values assigned from the input tensor.[^1] For our data type, we set the arbitrary range [−1, 1]. As such, both the quantiles for the data type and the neural network weights need to be normalized into this range.[^1] Since pretrained neural network weights usually have a zero-centered normal distribution with standard deviation σ (see Appendix F), we can transform all weights to a single fixed distribution by scaling σ such that the distribution fits exactly into the range of our data type.[^1] The information theoretically optimal data type for zero-mean normal distributions with arbitrary standard deviations σ in the range [−1, 1] is computed as follows: (1) estimate the 2k + 1 quantiles of a theoretical N(0, 1) distribution to obtain a k-bit quantile quantization data type for normal distributions[^1] We term the resulting data type that has equal expected number of values in each quantization bin k-bit NormalFloat (NFk), since the data type is information-theoretically optimal for zero-centered normally distributed data.[^1]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.3-5 — `p_qlora_0002`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 12375 ms
- Passages in context: 3
- Top rerank score: 0.9895
- Sentence verification: verified 5
