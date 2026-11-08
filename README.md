# Hands-On LLM Post-Training Pipeline (`NEXUS-DSL v1`)

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
- **After Post-Training (LoRA SFT):** Translates single-intent requests, multi-intent compound requests, and out-of-scope refusals (`ERR_SCOPE := OUT_OF_DOMAIN;`) with high accuracy.

---

## Project Structure

| File | Stage | Description |
| :--- | :--- | :--- |
| [`dsl_validator.py`](./dsl_validator.py) | Verifier | Deterministic parser & evaluator for `NEXUS-DSL v1` syntax and semantic equivalence. |
| [`01_prepare_data.py`](./01_prepare_data.py) | Step 1: Data | Generates `data/sft_train.jsonl`, `data/sft_val.jsonl`, `data/test_heldout.jsonl`, and `data/dpo_train.jsonl`. |
| [`02_eval_before.py`](./02_eval_before.py) | Step 2: Baseline | Evaluates the raw Base Model (`Qwen/Qwen2.5-0.5B`) before training and saves `results/before_training.json`. |
| [`03_train_sft.py`](./03_train_sft.py) | Step 3: SFT + LoRA | Runs Supervised Fine-Tuning with **Completion-Only Loss Masking** (`label = -100` on prompts) and saves the LoRA adapter to `checkpoints/nexus-sft-lora`. |
| [`04_eval_after.py`](./04_eval_after.py) | Step 4: Evaluation | Evaluates the Post-Trained model on the held-out test set and prints a side-by-side comparison against Step 2. |
| [`05_train_dpo.py`](./05_train_dpo.py) | Step 5: DPO Alignment | Demonstrates Stage 2 Direct Preference Optimization (`chosen` vs `rejected` responses). |
