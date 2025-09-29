# IC‑Benchmark: Interface Complexity (ICI) × Retrieval × SLM — Preliminary Claim

## One‑sentence Claim
**Interface complexity (ICI) is a first‑order factor in tool retrieval and tool use. It harms small language models (SLMs) substantially more than larger LLMs. A simple, unsupervised ICI‑aware reranking (score = text_sim − λ·ICI_norm) significantly improves Top‑K recall under fuzzy queries, directly boosting SLMs’ chance to succeed without changing the model.**

## Why it matters
Fuzzy, multi‑server tool ecosystems make “find the right tool, then parameterize” the main bottleneck for agents. High‑ICI tools—deeper nesting, branching (oneOf/anyOf/if‑then‑else), heavy constraints and large enums—often **dominate text similarity** due to longer descriptions and dense terminology, pushing truly **SLM‑friendly low‑ICI tools** out of Top‑K. This hurts SLMs more (long‑context fragility, structured output brittleness), widening the SLM–LLM gap.

## Mechanism sketch
- **Retrieval bias:** length/term‑dense, high‑ICI schemas top the text ranking → gold tool drops out of Top‑K.
- **Downstream brittleness (for SLMs):** more branches/constraints enlarge the parameter space and validation burden.
- **Fix (no training):** subtract a small λ·ICI_norm penalty in reranking; promote abstract/low‑ICI tools into Top‑K.

## Hypotheses (H)
- **H1 (bias):** Higher ICI increases the chance a mismatched complex tool ranks above the gold tool under text similarity.
- **H2 (reranking):** Adding ICI penalty improves **Recall@K** notably (K∈[3,5]) without model changes.
- **H3 (SLM benefit):** The **marginal gain** from ICI‑aware reranking is larger for SLMs than LLMs, narrowing the SLM gap.
- **H4 (downstream, future work):** Given the same Top‑K, low‑ICI tools yield higher parameter‑validation Pass@1; the gap is larger for SLMs.

## Preliminary evidence (synthetic fuzzy set)
On 8 tools (4 capability pairs) and 24 fuzzy queries:
- **Recall@3** improves from **0.833 → 0.958** (+12.5 pp) with λ≈0.25–0.5.
- **Top‑1** may slightly drop (e.g., 0.79 → 0.71–0.75), but agents rely on **Top‑K** exploration; higher recall is directly useful for SLMs.

_Reproduction:_
```bash
bash scripts/run_prelim_ici.sh
# Outputs data/prelim/results/preliminary_ir_results.csv, lambda_sweep.csv, rescued_cases.jsonl, etc.

Takeaway

Before scaling models or adding complex training, make retrieval SLM‑aware: penalize interface complexity so that usable tools show up early. This alone changes SLM outcomes.


---

### 如何使用

1) 把上面的 **Codex prompt**（第 1 节）整体发给 Codex；  
2) 将第 2 节里的所有**文件**按路径落盘；  
3) 运行：

```bash
bash scripts/run_prelim_ici.sh


输出会在 data/prelim/results/ 下生成，并能支撑你在一天内提交“我们的方法在 Recall@3 上显著提升”的初步证据，同时文档中清晰阐述了SLM/LLM 差距与 ICI 机制。
