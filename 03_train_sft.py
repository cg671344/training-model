"""
03_train_sft.py
===============
Step 3 of Post-Training: Supervised Fine-Tuning (SFT) with LoRA.

What happens inside this script:
1. Loads the Base Foundation Model (`Qwen/Qwen2.5-0.5B`).
2. Attaches Low-Rank Adaptation (LoRA) matrices (`r=16, lora_alpha=32`) to all
   attention & MLP linear projections (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
   Only ~1.7% of parameters are trained; 98.3% remain frozen.
3. Tokenizes each conversation using the ChatML template and applies
   **Completion-Only Loss Masking**:
   - System & User prompt tokens get `label = -100` (ignored by CrossEntropyLoss).
   - Assistant response + `<|im_end|>` EOS token get their true token IDs as labels.
4. Trains for 4 epochs on your GTX 1650 GPU using Mixed Precision (`fp16`),
   tracking both training and validation loss.
5. Saves the trained LoRA adapter to `checkpoints/nexus-sft-lora`.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "checkpoints/nexus-sft-lora"


class CompletionOnlySFTDataset(Dataset):
    """
    Tokenizes `messages` into input_ids and labels where prompt tokens
    (system + user) are masked with -100 so loss is computed ONLY on the
    assistant's response and its closing <|im_end|> token.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_rows = [json.loads(line) for line in f]

        for row in raw_rows:
            messages = row["messages"]
            prompt_msgs = messages[:-1]

            # 1. Tokenize prompt alone (system + user + "<|im_start|>assistant\n")
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            # 2. Tokenize full conversation (including assistant response + "<|im_end|>\n")
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(
                full_text, add_special_tokens=False, truncation=True, max_length=max_length
            )["input_ids"]

            # Mask prompt tokens with -100 so the model only learns to generate the assistant turn
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            labels = labels[: len(full_ids)]

            self.samples.append({
                "input_ids": full_ids,
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_batch(batch: List[Dict[str, List[int]]], pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(x["input_ids"]) + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def evaluate_val_loss(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                total_steps += 1
    model.train()
    return total_loss / max(total_steps, 1)


def train_sft():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("STEP 3: Supervised Fine-Tuning (SFT) with LoRA")
    print(f"Base Model : {BASE_MODEL_ID}")
    print(f"Device     : {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base foundation model in fp16 to fit comfortably in 4GB VRAM
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Configure LoRA adapter across all attention and feed-forward linear layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    train_ds = CompletionOnlySFTDataset("data/sft_train.jsonl", tokenizer)
    val_ds = CompletionOnlySFTDataset("data/sft_val.jsonl", tokenizer)

    # Inspect one masked sample so the user can see completion-only loss masking
    sample = train_ds[0]
    unmasked_tokens = [t for t, l in zip(sample["input_ids"], sample["labels"]) if l != -100]
    print("\n--- Completion-Only Loss Masking Inspection ---")
    print(f"Total tokens in sample #0       : {len(sample['input_ids'])}")
    print(f"Masked prompt tokens (label=-100): {sample['labels'].count(-100)}")
    print(f"Trained target tokens            : {len(unmasked_tokens)}")
    print(f"Decoded trained target           :\n{tokenizer.decode(unmasked_tokens)!r}")
    print("-----------------------------------------------\n")

    batch_size = 4
    grad_accum_steps = 2  # Effective batch size = 8
    epochs = 4
    lr = 3e-4

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    total_update_steps = (len(train_loader) * epochs) // grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(5, int(total_update_steps * 0.1)),
        num_training_steps=total_update_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    history = []
    initial_val_loss = evaluate_val_loss(model, val_loader, device)
    print(f"[Before Training] Initial Validation Loss: {initial_val_loss:.4f}")
    history.append({"epoch": 0, "train_loss": None, "val_loss": round(initial_val_loss, 4)})

    start_time = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += outputs.loss.item()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        val_loss = evaluate_val_loss(model, val_loader, device)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Elapsed: {elapsed:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
        })

    # Save LoRA adapter & tokenizer
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    Path("results").mkdir(exist_ok=True)
    with open("results/training_log.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": BASE_MODEL_ID,
                "adapter_dir": OUTPUT_DIR,
                "epochs": epochs,
                "learning_rate": lr,
                "lora_r": 16,
                "lora_alpha": 32,
                "training_time_sec": round(time.time() - start_time, 1),
                "loss_history": history,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 65)
    print(f"STEP 3 COMPLETE: LoRA Adapter saved to '{OUTPUT_DIR}'")
    print(f"Validation Loss dropped from {initial_val_loss:.4f} -> {history[-1]['val_loss']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    train_sft()
