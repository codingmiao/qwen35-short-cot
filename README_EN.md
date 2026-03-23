# Qwen3.5 Chain-of-Thought Optimization Study

This project investigates fine-tuning the Qwen3.5 model using distilled data to address the issue of excessively long chain-of-thought reasoning.

---

## 📋 Overview

The Qwen3.5 series is strong overall, but its *thinking (CoT)* exhibits issues such as overly long reasoning chains and mixed Chinese-English outputs. These problems have been widely discussed in the community, e.g., in this issue:
[https://github.com/QwenLM/Qwen3.5/issues/35](https://github.com/QwenLM/Qwen3.5/issues/35)

We observe that its reasoning contains a large amount of ineffective information, such as:

```
Typical pattern:

"Here's a thinking process that leads to the solution:

1. **Analyze the Request**
2. **Recall/Identify the Concept**
3. **Step-by-step Reasoning**
4. **Final Answer**"
```

**Characteristics:**

* ✗ Excessive meta-cognition ("I need to analyze...", "Let me recall...")
* ✗ Repetition of problem statements (20–30%)
* ✗ Over-explaining basic concepts (30–40%)
* ✗ Repeated self-verification and hesitation (20–30%)
* ✗ Heavy formatting (Markdown, bold, lists)

---

To address this, prior work proposed distilling reasoning from other models. For example:
[https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled](https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled)

This approach reduces redundant reasoning loops and improves efficiency.

However, it has notable limitations:

* **Inconsistent dataset quality**: includes shallow reasoning or refusal responses
* **Lack of systematic evaluation**: unclear trade-offs between brevity and reasoning ability

---

## Our Approach

We propose a cleaner and more rigorously evaluated solution:

* Use **DeepSeek R1** data for distillation (instead of Claude)
* Apply **strict filtering** (25,840 high-quality samples)
* Perform **systematic comparison** across three models:

| Model      | Description                     |
| ---------- | ------------------------------- |
| sft_org    | Original Qwen3.5-9B             |
| sft_claude | Claude-distilled version        |
| sft_r1     | Our DeepSeek R1 distilled model |

---

## Core Findings

| Model       | Truncation Rate | Overall Accuracy | **True Accuracy** | Avg Tokens | Compression |
| ----------- | --------------- | ---------------- | ----------------- | ---------- | ----------- |
| **sft_org** | **82.1%**       | 28.8%            | **95.0%**         | 7753       | 1.0x        |
| **sft_r1**  | **22.4% ↓**     | **73.3% ↑**      | **93.9%**         | **3071 ↓** | **2.52x**   |
| sft_claude  | 30.1%           | 60.0%            | 82.3%             | 4141       | 1.87x       |

---

### Key Conclusions

1. ✅ **Truncation drastically reduced**
   82.1% → 22.4% (−59.7%)

2. ✅ **Minimal accuracy loss**
   95.0% → 93.9% (−1.1%)

3. ✅ **2.52× reasoning compression**

4. ✅ **Outperforms Claude distillation across all metrics**

---

## 1. Methodology

### 1.1 Model Construction (sft_r1)

#### Motivation

Why not directly compare with the Claude-distilled model?

**Issues:**

1. **Data Quality Problems**

   * Empty or shallow reasoning
   * Refusal responses
   * Cross-contamination between problems
   * Lack of filtering standards

2. **Poor Chinese reasoning quality**

   * English-trained reasoning fails in Chinese contexts

---

### R1 Reasoning Style Example

```
"Alright, this is a vector parallelism problem.

If two vectors are parallel, their components must be proportional.

Given a = (-1,2), b = (1,-2y):
-1/1 = 2/(-2y)

Solve:
-1 = -1/y → y = 1

Check:
b = (1,-2), which equals -1 × a → correct.

Answer: D"
```

---

### Dataset Filtering

**Source:**
Chinese-DeepSeek-R1-Distill-data-110k-SFT

**Criteria:**

* Domain: STEM / Math
* Length ≤ 1536 tokens
* Score > 9
* Complete reasoning

**Result:**

* Original: 110,000 samples
* Filtered: **25,840 samples (23.5%)**

---

### Training Setup

* Base model: Qwen3.5-9B
* Method: LoRA

**Hyperparameters:**

```yaml
BATCH_SIZE: 4
ACCUMULATION: 4
LR: 5e-5
EPOCHS: 2

MAX_LEN: 2048
LORA_RANK: 8
LORA_ALPHA: 32
```

---

## 2. Evaluation

### Metrics

* **Truncation Rate**
* **True Accuracy** (excluding truncated samples)
* **Overall Accuracy**
* **Average Tokens**
* **Compression Ratio**

---

## 3. Results

### 3.1 Truncation Analysis

| Model      | Truncation Rate | True Accuracy | Overall Accuracy |
| ---------- | --------------- | ------------- | ---------------- |
| sft_org    | 82.1%           | 95.0%         | 28.8%            |
| sft_r1     | 22.4%           | 93.9%         | 73.3%            |
| sft_claude | 30.1%           | 82.3%         | 60.0%            |

**Insight:**

* sft_org is highly accurate *if not truncated*
* Truncation is the primary cause of failure

---

### 3.2 Token Efficiency

| Model      | Avg Tokens | Median |
| ---------- | ---------- | ------ |
| sft_org    | 7753       | 8192   |
| sft_r1     | 3071       | 1683   |
| sft_claude | 4141       | 2844   |

---

### 3.3 Distillation Effectiveness

| Model      | Compression | Accuracy Retention | Score    |
| ---------- | ----------- | ------------------ | -------- |
| **sft_r1** | **2.52x**   | **98.9%**          | **1.53** |
| sft_claude | 1.87x       | 86.6%              | 1.20     |

---

## 4. Reasoning Style Comparison

### sft_org: *Performative reasoning*

* Heavy meta-cognition
* Redundant explanations
* Excessive formatting

### sft_r1: *Efficient reasoning*

* Direct problem-solving
* Minimal explanation
* Natural language

---

### Example Comparison

| Metric      | sft_org | sft_r1   |
| ----------- | ------- | -------- |
| Tokens      | 8093    | 848      |
| Compression | 1.0x    | **9.5x** |
| Accuracy    | ✓       | ✓        |

---

## 5. Limitations

### 5.1 Data Quality

* Residual noise remains
* Some reasoning inconsistencies

### 5.2 Domain Coverage

* Only STEM/math tested
* Unknown performance in:

   * Coding
   * Writing
   * General reasoning

### 5.3 Model Scale

* Only tested on 9B
* Needs validation on 27B+

### 5.4 Evaluation Issues

* `\boxed{}` extraction limitations
* Weak LaTeX parsing
* Judge model cost

---

## 6. Conclusion

### Key Takeaways

1. ✅ **R1 distillation significantly outperforms Claude**
2. ✅ **Strict data filtering is critical**
3. ✅ **Accuracy loss is minimal and controllable**
4. ✅ **Reasoning style fundamentally changes**

---

## Practical Impact

| Metric     | Improvement |
| ---------- | ----------- |
| Accuracy   | +154%       |
| Latency    | −56%        |
| Token cost | −60%        |

---

## Future Work

* Higher-quality dataset cleaning
* 27B model validation
* Improved evaluation pipeline
