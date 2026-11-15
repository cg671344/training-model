# Hands-On LLM Post-Training Pipeline (`NEXUS-DSL v1`)

English | [简体中文](./readme_zh.md)

This project demonstrates the complete end-to-end **Post-Training** procedure on a Base Foundation Model (`Qwen/Qwen2.5-0.5B`) running locally on an **NVIDIA GTX 1650 (4 GB VRAM)**.

## Why This Task? ("Before vs. After" Capability)

A raw base model has **zero knowledge** of our custom Smart-Home Compiler protocol (`NEXUS-DSL v1`):
```text
@NEXUS_V1
CMD lum.kitchen.main := ON;
CMD hvac.kitchen.temp := 22C;
@END
```
- **Before Post-Training:** `0%` valid syntax, `0%` exact match (the base model rambles or autocompletes unrelated text).
- **After Post-Training (LoRA SFT):** Translates single-intent requests, multi-intent compound requests, and out-of-scope refusals (`ERR_SCOPE := OUT_OF_DOMAIN;`) with **90.0% valid syntax** and **85.0% exact match**.

---

## 1. Quantitative Results: Before vs. After Post-Training

| Metric (on 20 Held-Out Unseen Prompts) | Before Training (Base `Qwen/Qwen2.5-0.5B`) | After Post-Training (`Base + LoRA SFT`) | Change |
| :--- | :---: | :---: | :---: |
| **Valid `NEXUS-DSL v1` Syntax** | `0 / 20` (**0.0%**) | `18 / 20` (**90.0%**) | **+90.0%** |
| **Exact Command Match** | `0 / 20` (**0.0%**) | `17 / 20` (**85.0%**) | **+85.0%** |
| **Out-of-Scope Refusal Accuracy** | `0 / 4` (**0.0%**) | `4 / 4` (**100.0%**) | **+100.0%** |
| **Multi-Intent Composition** | `0 / 4` (**0.0%**) | `3 / 4` (**75.0%**) | **+75.0%** |

### Training Loss Progression ([`results/training_log.json`](./results/training_log.json))
- **Base Model:** `Qwen/Qwen2.5-0.5B`
- **Hardware:** NVIDIA GeForce GTX 1650 (4 GB VRAM, `fp16`)
- **Trainable Parameters:** `8,798,208` / `502,830,976` (**1.75%** via LoRA `r=16, alpha=32`)
- **Before Training Validation Loss:** `3.7875`
- **Epoch 1:** Train Loss `1.1684` | Val Loss `0.4108`
- **Epoch 2:** Train Loss `0.3576` | Val Loss `0.3772`
- **Epoch 3:** Train Loss `0.3566` | Val Loss `0.3700`
- **Epoch 4:** Train Loss `0.3525` | Val Loss `0.3687`

---

## 2. Side-by-Side Output Comparison ([`before_training.json`](./results/before_training.json) vs. [`after_training.json`](./results/after_training.json))

| Category | Unseen User Prompt | Before Training (Raw Base Model) | After Training (LoRA SFT Adapter) |
| :--- | :--- | :--- | :--- |
| **Single Action** | *"Could you switch on the kitchen lights please?"* | `'Yes, I can switch on the kitchen lights.erable erable Could you switch...'` | `@NEXUS_V1`<br>`CMD lum.kitchen.main := ON;`<br>`@END` |
| **Synonym Mapping** (*office → study*) | *"Set the thermostat in my office to 21 degrees."* | `'Set the thermostat in my office to 21 degrees.})(); Comey Set the...'` | `@NEXUS_V1`<br>`CMD hvac.study.temp := 21C;`<br>`@END` |
| **Synonym Mapping** (*curtains → blinds*) | *"Close the living room curtains."* | `"I'm sorry, but I don't understand the request. Can you please provide..."` | `@NEXUS_V1`<br>`CMD shade.living.blinds := CLOSED;`<br>`@END` |
| **Multi-Intent Composition** | *"Turn on the lights in the study and set the temperature in the study to 23 degrees."* | `'What is the temperature in the study? ViewById: 1 ViewById: 2...'` | `@NEXUS_V1`<br>`CMD lum.study.main := ON;`<br>`CMD hvac.study.temp := 23C;`<br>`@END` |
| **Multi-Domain Composition** | *"Close the blinds in the bedroom and switch off the television in the bedroom."* | `'I want to turn on the lights in the bedroom and turn off the...'` | `@NEXUS_V1`<br>`CMD shade.bedroom.blinds := CLOSED;`<br>`CMD media.bedroom.tv := OFF;`<br>`@END` |
| **Out-of-Scope Trap** | *"How do I install PyTorch on Windows?"* | `'How do I install PyTorch on Windows?forgettable forgettable...'` | `@NEXUS_V1`<br>`ERR_SCOPE := OUT_OF_DOMAIN;`<br>`@END` |
| **Out-of-Scope Trap** | *"Write a haiku about autumn leaves."* | `'Write a haiku about autumn leaves.})(); })(); })(); })();...'` | `@NEXUS_V1`<br>`ERR_SCOPE := OUT_OF_DOMAIN;`<br>`@END` |

---

## 3. Three Key Post-Training Lessons from This Run

1. **Completion-Only Loss Masking ([`CompletionOnlySFTDataset`](./03_train_sft.py)):**
   In Sample #0 (107 total tokens), the 85 system and user prompt tokens were masked with `label = -100`, so 100% of gradient updates focused on the 22 target tokens (`@NEXUS_V1\nCMD hvac.study.temp := 22C;\n@END`).
2. **Untrained Special Tokens in Raw Base Models ([`04_eval_after.py`](./04_eval_after.py)):**
   In the raw `Qwen/Qwen2.5-0.5B` base model, `tokenizer.eos_token` is `<|endoftext|>` (`151643`), whereas ChatML uses `<|im_end|>` (`151645`). Because LoRA freezes `embed_tokens` and `lm_head` by default, `<|im_end|>` remains uninitialized in the output head unless you either add `stop_strings=["@END"]` at inference or include `modules_to_save=["embed_tokens", "lm_head"]` in `LoraConfig`.
3. **Diagnosing Data Imbalance via Error Analysis:**
   The 2 syntax failures on the test set (#08 *"Arm the security alarm for the entire house"* → `CMD sec.house.alarm := ARMED;` instead of `sec.all.alarm`) happened because the `all` zone only had 6 examples in [`data/sft_train.jsonl`](./data/sft_train.jsonl) compared to ~40 per named room—so the model copied the noun `house` as the zone identifier. Adding 15–20 more whole-house examples fixes this immediately.

---

## 4. Workspace Files Reference

| File | Stage / Role | Description |
| :--- | :--- | :--- |
| [`dsl_validator.py`](./dsl_validator.py) | Verifier | Deterministic parser & evaluator for `NEXUS-DSL v1` syntax and semantic equivalence. |
| [`01_prepare_data.py`](./01_prepare_data.py) | Step 1: Data | Generates [`data/sft_train.jsonl`](./data/sft_train.jsonl) (258 train), [`data/sft_val.jsonl`](./data/sft_val.jsonl) (30 val), [`data/test_heldout.jsonl`](./data/test_heldout.jsonl) (20 unseen test), and [`data/dpo_train.jsonl`](./data/dpo_train.jsonl) (60 DPO pairs). |
| [`02_eval_before.py`](./02_eval_before.py) | Step 2: Baseline | Evaluates the raw Base Model (`Qwen/Qwen2.5-0.5B`) before training and saves [`results/before_training.json`](./results/before_training.json). |
| [`03_train_sft.py`](./03_train_sft.py) | Step 3: SFT + LoRA | Runs Supervised Fine-Tuning with **Completion-Only Loss Masking** (`label = -100` on prompts) and saves the LoRA adapter to `checkpoints/nexus-sft-lora` and [`results/training_log.json`](./results/training_log.json). |
| [`04_eval_after.py`](./04_eval_after.py) | Step 4: Evaluation | Evaluates the Post-Trained model on the held-out test set, prints a side-by-side comparison against Step 2, and saves [`results/after_training.json`](./results/after_training.json). |
| [`05_train_dpo.py`](./05_train_dpo.py) | Step 5: DPO Alignment | Demonstrates Stage 2 Direct Preference Optimization (`chosen` vs `rejected` responses) on 4 GB VRAM using `model.disable_adapter()`. |

---

## Quick Reproduction Commands

```powershell
# 1. Generate datasets
.venv\Scripts\python.exe 01_prepare_data.py

# 2. Evaluate base model before training
.venv\Scripts\python.exe 02_eval_before.py

# 3. Run LoRA Supervised Fine-Tuning (SFT)
.venv\Scripts\python.exe 03_train_sft.py

# 4. Evaluate post-trained model & compare Before vs. After
.venv\Scripts\python.exe 04_eval_after.py

# 5. (Optional) Run Stage 2 DPO preference alignment
.venv\Scripts\python.exe 05_train_dpo.py
```

---

## 5. Deep Dive: Understanding the Three Key Lessons

### 5.1 Why Mask Prompt Tokens with `label = -100`? (Completion-Only Loss Masking)

In Sample #0 (107 total tokens), the first 85 system prompt and user input tokens are masked with `label = -100`, focusing 100% of gradient updates on the final 22 target tokens (`@NEXUS_V1\nCMD hvac.study.temp := 22C;\n@END`).

- **How Cross-Entropy Loss Works:** During causal LM training, we feed the full sequence of token IDs alongside a matching `labels` array. In PyTorch's `CrossEntropyLoss`, any position with `label = -100` is **ignored** and contributes zero loss and zero gradient.
- **Why Masking Matters:** Each training example consists of two parts:
  - **First 85 tokens (Prompt / Question):** System prompt + User message + `<|im_start|>assistant`
  - **Last 22 tokens (Completion / Answer):** `@NEXUS_V1\nCMD hvac.study.temp := 22C;\n@END<|im_end|>`
- Without masking, the model would spend ~80% of its capacity memorizing the system prompt and user phrasing instead of learning the `NEXUS-DSL v1` syntax. In [`03_train_sft.py`](./03_train_sft.py) (`CompletionOnlySFTDataset`), we set:
  ```python
  labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
  ```
  so the model still reads the full prompt as context, while loss is computed **exclusively on the assistant's 22 answer tokens**.

### 5.2 Why Didn't the Model Stop After `@END`? (Untrained Special Tokens in Base Models)

Language models do not automatically know when to stop generating; they must emit a special **End-of-Sequence (EOS) token** that `model.generate()` watches for.

In the raw `Qwen/Qwen2.5-0.5B` base model:
- **Pre-training EOS token:** `<|endoftext|>` (ID `151643`), used to separate raw web documents.
- **ChatML turn-end token:** `<|im_end|>` (ID `151645`), used to end an assistant turn. In an unaligned base model, `<|im_end|>` has rarely been trained as an assistant stop signal.
- **Why LoRA Didn't Fix It Automatically:** A transformer consists of `embed_tokens` (input vocabulary) → middle Transformer layers → `lm_head` (output vocabulary projection). By default, LoRA only adapts the middle attention/MLP layers and **freezes `embed_tokens` and `lm_head`**. So even after the middle layers learned to stop after `@END`, the frozen `lm_head` still assigned near-zero probability to `<|im_end|>`, causing the model to keep rambling past `@END`.
- **Two Ways to Fix This:**
  1. **Inference-time stop string (used in [`04_eval_after.py`](./04_eval_after.py)):** Pass `stop_strings=["@END"]` to `model.generate(...)` to halt generation cleanly as soon as `@END` is emitted.
  2. **Training-time unfreezing:** Set `modules_to_save=["embed_tokens", "lm_head"]` in `LoraConfig` so the output head learns to emit `<|im_end|>` natively (at the cost of slightly higher VRAM and trainable parameter count).

### 5.3 Diagnosing Data Imbalance via Error Analysis (`sec.house.alarm` vs `sec.all.alarm`)

On the 20 held-out test cases, the only 2 syntax errors—Test #08 (*"Arm the security alarm for the entire house."*) and Test #12 (*"Turn off every light in the whole house."*)—failed identically: the model emitted `sec.house.alarm` and `lum.house.main` instead of `sec.all.alarm` and `lum.all.main`.

- **Root Cause:** Inspecting [`01_prepare_data.py`](./01_prepare_data.py) shows that each of the 6 named rooms (`kitchen`, `bedroom`, `living`, `study`, `garage`, `porch`) appears in **~40 training examples**, whereas the whole-house `all` zone appears in **only 6 examples**.
- **Fallback Behavior:** Because `all` was severely underrepresented, when the model encountered unseen phrasings like *"entire house"* or *"whole house"*, it fell back to copying the English noun `house` directly into the zone slot.
- **Key Takeaway:** In post-training, **fixing data coverage via error analysis is far more effective than tuning learning rates or adding epochs**. Adding 15–20 varied `all`-zone examples (e.g., *"entire house"*, *"whole home"*, *"all rooms"*) in [`01_prepare_data.py`](./01_prepare_data.py) directly resolves this failure mode.

