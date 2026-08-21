# A_coreference_across_papers — Coreference across papers (LoRA -> QLoRA without naming it)

Session `s_234bd5630bbf` · corpus `e62c6925a03ad297`

## Turn 1

**User:** What problem does LoRA solve?

*Condensation:* skipped — turn 1 — nothing to condense

*Route:* `refuse` (expected `answer`) — top rerank 0.684 < tau_low 0.736: nothing in the corpus is a strong enough match to answer from

**Agent refuses.** top rerank 0.684 < tau_low 0.736: nothing in the corpus is a strong enough match to answer from

## Turn 2

**User:** How does the quantised version reduce memory further?

*Condensed to:* `How does the quantised version of LoRA reduce memory further?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.990 >= tau_high 0.786

# How does the quantised version of LoRA reduce memory further?

Standard LoRA lowers memory requirements by freezing pretrained model weights and only optimizing smaller, injected low-rank matrices.[^1] QLoRA, the quantized version of LoRA, reduces memory further by backpropagating gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters.[^2] To save additional memory, QLoRA introduces 4-bit NormalFloat (NF4), Double Quantization to quantize the quantization constants, and Paged Optimizers to manage memory spikes.[^2]

## Sources

[^1]: LoRA: Low-Rank Adaptation of Large Language Models · pp.1-3 — `p_lora_0000`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`

---

- Provider: `gemini` · model: `gemini-3.6-flash`
- Latency: 21140 ms
- Passages in context: 3
- Top rerank score: 0.9905
- Sentence verification: verified 3


## Turn 3

**User:** What does it report for a 65B model?

*Condensed to:* `What does QLoRA report for a 65B model?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.952 >= tau_high 0.786

# What does QLoRA report for a 65B model?

Regular 16-bit finetuning of a 65B parameter LLaMA model requires more than 780 GB of GPU memory, but QLoRA reduces the average memory requirement to less than 48 GB.[^1] The Guanaco 65B model was finetuned on a single GPU over 24 hours using a variant of OASST1.[^1][^2] Guanaco 65B achieved 99.3% performance relative to ChatGPT on the Vicuna benchmark with an Elo rating of 1022 ± 1.[^1][^2] When compared to GPT-4, Guanaco 65B has an expected win probability of 30% based on Elo ratings from human annotators.[^2]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.9-10 — `p_qlora_0006`

---

- Provider: `gemini` · model: `gemini-3.6-flash`
- Latency: 45656 ms
- Passages in context: 3
- Top rerank score: 0.9520
- Sentence verification: verified 4


## Turn 4

**User:** Does that cost quality?

*Condensed to:* `What does QLoRA report for a 65B model?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — semantic cache hit (cos 1.0002)

# What does QLoRA report for a 65B model?

Regular 16-bit finetuning of a 65B parameter LLaMA model requires more than 780 GB of GPU memory, but QLoRA reduces the average memory requirement to less than 48 GB.[^1] The Guanaco 65B model was finetuned on a single GPU over 24 hours using a variant of OASST1.[^1][^2] Guanaco 65B achieved 99.3% performance relative to ChatGPT on the Vicuna benchmark with an Elo rating of 1022 ± 1.[^1][^2] When compared to GPT-4, Guanaco 65B has an expected win probability of 30% based on Elo ratings from human annotators.[^2]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.9-10 — `p_qlora_0006`

---

- Provider: `gemini` · model: `gemini-3.6-flash`
- Latency: 0 ms *(cache hit — not a synthesis timing)*
- Passages in context: 3
- Sentence verification: verified 4

