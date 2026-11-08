"""
05_train_dpo.py
===============
Step 5 of Post-Training (Optional Stage 2): Direct Preference Optimization (DPO)
& Merging LoRA Weights into a Standalone Model.

Why Stage 2 (DPO) exists after Stage 1 (SFT):
---------------------------------------------
- SFT teaches the model *what* to generate (maximizing log-probability of `chosen`).
- DPO explicitly contrasts `chosen` (clean NEXUS-DSL) against `rejected` (chatty markdown
  wrappers or extra explanations), pushing down the probability of bad habits while
  anchoring to the SFT reference policy (`beta` KL penalty).

DPO Loss Formula:
  L_DPO = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                               - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
SFT_ADAPTER_DIR = "checkpoints/nexus-sft-lora"
DPO_ADAPTER_DIR = "checkpoints/nexus-dpo-lora"


def compute_response_logprob(model, tokenizer, prompt_msgs, response_msgs, device):
    """Computes sum of log-probabilities over ONLY the assistant response tokens."""
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        prompt_msgs + response_msgs, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_tensor).logits  # [1, seq_len, vocab]

    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, len(prompt_ids) - 1 : -1, :]
    shift_labels = input_tensor[:, len(prompt_ids) :]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum()


def run_dpo_stage():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("STEP 5: Stage 2 Preference Alignment with DPO (Direct Preference Optimization)")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    # Load SFT adapter with `is_trainable=True`
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_DIR, is_trainable=True)

    with open("data/dpo_train.jsonl", "r", encoding="utf-8") as f:
        dpo_pairs = [json.loads(line) for line in f][:30]  # Fast 30-pair demonstration

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    beta = 0.1

    model.train()
    total_loss = 0.0
    reward_accuracies = 0

    for idx, pair in enumerate(dpo_pairs, start=1):
        prompt_msgs = pair["prompt"]
        chosen_msgs = pair["chosen"]
        rejected_msgs = pair["rejected"]

        # 1. Compute reference log-probs using base model (adapter disabled) or initial SFT
        with torch.no_grad():
            with model.disable_adapter():
                with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                    ref_chosen_lp = compute_response_logprob(model, tokenizer, prompt_msgs, chosen_msgs, device)
                    ref_rejected_lp = compute_response_logprob(model, tokenizer, prompt_msgs, rejected_msgs, device)

        # 2. Compute policy log-probs with active trainable adapter
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
            pi_chosen_lp = compute_response_logprob(model, tokenizer, prompt_msgs, chosen_msgs, device)
            pi_rejected_lp = compute_response_logprob(model, tokenizer, prompt_msgs, rejected_msgs, device)

            chosen_reward = beta * (pi_chosen_lp - ref_chosen_lp)
            rejected_reward = beta * (pi_rejected_lp - ref_rejected_lp)
            loss = -F.logsigmoid(chosen_reward - rejected_reward)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if chosen_reward.item() > rejected_reward.item():
            reward_accuracies += 1

        if idx % 10 == 0:
            print(
                f"  Step {idx:02d}/{len(dpo_pairs)} | DPO Loss: {total_loss / idx:.4f} | "
                f"Preference Accuracy (Chosen > Rejected): {reward_accuracies / idx * 100:.1f}%"
            )

    Path(DPO_ADAPTER_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(DPO_ADAPTER_DIR)
    tokenizer.save_pretrained(DPO_ADAPTER_DIR)
    print(f"\nSaved DPO-aligned LoRA adapter to '{DPO_ADAPTER_DIR}'.")


if __name__ == "__main__":
    run_dpo_stage()
