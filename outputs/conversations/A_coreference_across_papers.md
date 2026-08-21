# A_coreference_across_papers — Coreference across papers (LoRA -> QLoRA without naming it)

Session `s_b6b0c5f5401e` · corpus `e62c6925a03ad297`

## Turn 1

**User:** What problem does LoRA solve?

*Condensation:* skipped — turn 1 — nothing to condense

*Route:* `answer` (expected `answer`) — top rerank 0.986 >= tau_high 0.786

# What problem does LoRA solve?

Fine-tuning enormous language models is prohibitively expensive in terms of the hardware required and the storage/switching cost for hosting independent instances for different tasks.[^1] LoRA, an efficient adaptation strategy that neither introduces inference latency nor reduces input sequence length while retaining high model quality.[^2]

## Sources

[^1]: LoRA: Low-Rank Adaptation of Large Language Models · pp.1-3 — `p_lora_0000`
[^2]: LoRA: Low-Rank Adaptation of Large Language Models · pp.12-13 — `p_lora_0008`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.9857
- Sentence verification: verified 2


## Turn 2

**User:** How does the quantised version reduce memory further?

*Condensed to:* `How does the quantised version reduce hardware cost and storage/switching cost?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.821 >= tau_high 0.786

# How does the quantised version reduce hardware cost and storage/switching cost?

Double Quantization helps reduce the memory footprint of quantization constants.[^1] Double Quantization treats quantization constants cFP32 as inputs to a second quantization, yielding the quantized quantization constants cFP8 and reducing the average memory footprint per parameter from 0.5 bits to 0.127 bits.[^1] QLORA reduces the average memory requirements of finetuning a 65B parameter model from >780GB of GPU memory to <48GB without degrading the runtime or predictive performance compared to a 16bit fully finetuned baseline.[^2]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.5-6 — `p_qlora_0003`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.8213
- Sentence verification: verified 3


## Turn 3

**User:** What does it report for a 65B model?

*Condensed to:* `What does QLORA report for a 65B parameter model?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.957 >= tau_high 0.786

# What does QLORA report for a 65B parameter model?

QLORA reduces the average memory requirements of finetuning a 65B parameter model from >780GB of GPU memory to <48GB without degrading the runtime or predictive performance compared to a 16bit fully finetuned baseline.[^1] Using QLORA, we train the Guanaco family of models, with the second best model reaching 97.8% of the performance level of ChatGPT on the Vicuna [10] benchmark, while being trainable in less than 12 hours on a single consumer GPU;[^1] When deployed, our smallest Guanaco model (7B parameters) requires just 5 GB of memory and outperforms a 26 GB Alpaca model by more than 20 percentage points on the Vicuna benchmark.[^2]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.9-10 — `p_qlora_0006`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.9569
- Sentence verification: verified 3


## Turn 4

**User:** Does that cost quality?

*Condensed to:* `Does QLORA reduce quality?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.990 >= tau_high 0.786

# Does QLORA reduce quality?

QLoRA has one storage data type (usually 4-bit NormalFloat) and a computation data type (16-bit BrainFloat).[^1] We have discussed how QLoRA works and how it can significantly reduce the required memory for finetuning models.[^1]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.5-6 — `p_qlora_0003`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.9898
- Sentence verification: verified 2

