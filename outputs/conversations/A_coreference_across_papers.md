# A_coreference_across_papers — Coreference across papers (LoRA -> QLoRA without naming it)

Session `s_92f79edad1af` · corpus `e62c6925a03ad297`

## Turn 1

**User:** What problem does LoRA solve?

*Condensation:* skipped — turn 1 — nothing to condense

*Route:* `refuse` (expected `answer`) — top rerank 0.684 < tau_low 0.736: nothing in the corpus is a strong enough match to answer from

**Agent refuses.** top rerank 0.684 < tau_low 0.736: nothing in the corpus is a strong enough match to answer from

## Turn 2

**User:** How does the quantised version reduce memory further?

*Condensed to:* `How does the quantised version of LoRA reduce memory?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.853 >= tau_high 0.786

# How does the quantised version of LoRA reduce memory?

Since the memory footprint of LoRA is so minimal, we can use more adapters to improve performance without significantly increasing the total memory used.[^1] Aggressively reducing the amount of LoRA parameter yields only minor memory benefits.[^1]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.3-5 — `p_qlora_0002`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.8529
- Sentence verification: verified 2


## Turn 3

**User:** What does it report for a 65B model?

*Condensed to:* `What does LoRA report for a 65B model?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.856 >= tau_high 0.786

# What does LoRA report for a 65B model?

QLORA reduces the average memory requirements of finetuning a 65B parameter model from >780GB of GPU memory to <48GB without degrading the runtime or predictive performance compared to a 16bit fully finetuned baseline.[^1] Using QLORA, we train the Guanaco family of models, with the second best model reaching 97.8% of the performance level of ChatGPT on the Vicuna [10] benchmark, while being trainable in less than 12 hours on a single consumer GPU; using a single professional GPU over 24 hours we achieve 99.3% with our largest model, essentially closing the gap to ChatGPT on the Vicuna benchmark.[^1]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.8563
- Sentence verification: verified 2


## Turn 4

**User:** Does that cost quality?

*Condensed to:* `What does it report for a 65B model?`
*Drift guard:* passed — condensed within the conversation's vocabulary

*Route:* `answer` (expected `answer`) — top rerank 0.890 >= tau_high 0.786

# What does it report for a 65B model?

QLORA reduces the average memory requirements of finetuning a 65B parameter model from >780GB of GPU memory to <48GB without degrading the runtime or predictive performance compared to a 16bit fully finetuned baseline.[^1] Using QLORA, we train the Guanaco family of models, with the second best model reaching 97.8% of the performance level of ChatGPT on the Vicuna [10] benchmark, while being trainable in less than 12 hours on a single consumer GPU;[^1] When deployed, our smallest Guanaco model (7B parameters) requires just 5 GB of memory and outperforms a 26 GB Alpaca model by more than 20 percentage points on the Vicuna benchmark (Table 6).[^1] Based on our automated and human evaluations, we find that the top QLORA tuned model, Guanaco 65B, which we finetune on a variant of OASST1, is the best-performing open-source chatbot model and offers performance competitive to ChatGPT.[^2] When compared to GPT-4, Guanaco 65B and 33B have an expected win probability of 30%, based on Elo rating from human annotators system-level pairwise comparisons - the highest reported to date.[^2]

## Sources

[^1]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.1-2 — `p_qlora_0000`
[^2]: QLoRA: Efficient Finetuning of Quantized LLMs · pp.9-10 — `p_qlora_0006`

---

- Provider: `ollama` · model: `llama3.1:8b`
- Latency: 0 ms
- Passages in context: 3
- Top rerank score: 0.8895
- Sentence verification: verified 5

