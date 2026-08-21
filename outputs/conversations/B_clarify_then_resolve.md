# B_clarify_then_resolve — Ambiguous question -> clarify -> resolve

Session `s_aff01e7aefeb` · corpus `e62c6925a03ad297`

## Turn 1

**User:** What rank is used?

*Condensation:* skipped — turn 1 — nothing to condense

*Route:* `answer` (expected `clarify`) — top rerank 0.966 >= tau_high 0.786

# What rank is used?

Low Rank Adapters (LoRA) are used.[^1] We add a small set of learnable Low-rank Adapter weights [28] that are tuned by backpropagating gradients through the quantized weights.[^2]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.3-5 — `p_qlora_0002`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 4750 ms
- Passages in context: 3
- Top rerank score: 0.9661
- Sentence verification: verified 2


## Turn 2

**User:** In LoRA, for GPT-3.

*Condensed to:* `What rank is used in LoRA for GPT-3?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.959 >= tau_high 0.786

# What rank is used in LoRA for GPT-3?

We focus our study on GPT-3 175B, where we achieved the largest reduction of trainable parameters (up to 10,000×) without adversely affecting task performances.[^1] Given the empirical advantage of LoRA, we hope to further explain the properties of the low-rank adaptation learned from downstream tasks.[^1]

## Sources

[^1]: LoRA: Low-Rank Adaptation of Large Language Models · pp.9-10 — `p_lora_0006`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 5469 ms
- Passages in context: 3
- Top rerank score: 0.9586
- Sentence verification: verified 2


## Turn 3

**User:** How does that compare to full fine-tuning?

*Condensed to:* `How does LoRA for GPT-3 compare to full fine-tuning?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.944 >= tau_high 0.786

# How does LoRA for GPT-3 compare to full fine-tuning?

Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.[^1] We observe a 25% speedup during training on GPT-3 175B compared to full fine-tuning as we do not need to calculate the gradient for the vast majority of the parameters.[^2]

## Sources

[^1]: LoRA: Low-Rank Adaptation of Large Language Models · pp.1-3 — `p_lora_0000`
[^2]: LoRA: Low-Rank Adaptation of Large Language Models · pp.4-5 — `p_lora_0002`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 5485 ms
- Passages in context: 2
- Top rerank score: 0.9445
- Sentence verification: verified 2

