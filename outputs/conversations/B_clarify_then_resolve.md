# B_clarify_then_resolve — Ambiguous question -> clarify -> resolve

Session `s_1c025d89ed0a` · corpus `e62c6925a03ad297`

## Turn 1

**User:** What rank is used?

*Condensation:* skipped — turn 1 — nothing to condense

*Route:* `refuse` (expected `clarify`) — top rerank 0.188 < tau_low 0.736: nothing in the corpus is a strong enough match to answer from

**Agent refuses.** top rerank 0.188 < tau_low 0.736: nothing in the corpus is a strong enough match to answer from

## Turn 2

**User:** In LoRA, for GPT-3.

*Condensed to:* `What rank is used in LoRA for GPT-3?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.972 >= tau_high 0.786

# What rank is used in LoRA for GPT-3?

Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.[^1] We know that the intrinsic rank needed to represent the "task-specific directions" (thus for model adaptation) is low. In contrast, those directions in the rank-4 version of ∆W (corresponding to r = 4) are amplified by a much larger factor 20.[^2] One should expect a very different set of feature directions to be amplified for each different downstream task. In particular, for r = 4, this amplification factor is as large as 20. In other words, there are (generally speaking) four feature directions in each layer (out of the entire feature space from the pre-trained model W), that need to be amplified by a very large factor 20, in order to achieve our reported accuracy for the downstream specific task.[^2]

## Sources

[^1]: LoRA: Low-Rank Adaptation of Large Language Models · pp.1-3 — `p_lora_0000`
[^2]: LoRA: Low-Rank Adaptation of Large Language Models · pp.23-26 — `p_lora_0015`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.9718
- Sentence verification: verified 3


## Turn 3

**User:** How does that compare to full fine-tuning?

*Condensed to:* `How does LoRA compared to full fine-tuning?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.981 >= tau_high 0.786

# How does LoRA compared to full fine-tuning?

LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times compared to GPT-3 175B fine-tuned with Adam.[^1] LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters.[^1]

## Sources

[^1]: LoRA: Low-Rank Adaptation of Large Language Models · pp.1-3 — `p_lora_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 5405 ms
- Passages in context: 2
- Top rerank score: 0.9815
- Sentence verification: verified 2

