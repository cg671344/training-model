"""
04_eval_after.py
================
Step 4 of Post-Training: Post-Training Evaluation & Side-by-Side Comparison.

Loads:
- Base Foundation Model (`Qwen/Qwen2.5-0.5B`)
- Trained LoRA Adapter (`checkpoints/nexus-sft-lora`)

Evaluates on the exact same 20 unseen prompts in `data/test_heldout.jsonl` and
compares the results side-by-side against `results/before_training.json`.
"""

import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dsl_validator import evaluate_prediction

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
ADAPTER_DIR = "checkpoints/nexus-sft-lora"


def run_after_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 65)
    print("STEP 4: Evaluating Post-Trained Model (Base + LoRA SFT Adapter)")
    print(f"Base Model   : {BASE_MODEL_ID}")
    print(f"LoRA Adapter : {ADAPTER_DIR}")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    with open("data/test_heldout.jsonl", "r", encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f]

    before_data = None
    if Path("results/before_training.json").exists():
        with open("results/before_training.json", "r", encoding="utf-8") as f:
            before_data = json.load(f)

    results = []
    valid_count = 0
    exact_count = 0

    for idx, item in enumerate(test_cases, start=1):
        messages = item["messages"]
        prompt_messages = messages[:-1]
        ground_truth = messages[-1]["content"]
        user_query = messages[1]["content"]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")],
                stop_strings=["@END"],
                tokenizer=tokenizer,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        eval_res = evaluate_prediction(prediction, ground_truth)
        if eval_res["syntax_valid"]:
            valid_count += 1
        if eval_res["exact_match"]:
            exact_count += 1

        before_out = (
            before_data["predictions"][idx - 1]["base_model_output"]
            if before_data
            else "N/A"
        )

        results.append({
            "id": idx,
            "user_query": user_query,
            "ground_truth": ground_truth,
            "before_output": before_out,
            "after_output": prediction,
            "syntax_valid": eval_res["syntax_valid"],
            "exact_match": eval_res["exact_match"],
            "reason": eval_res["reason"],
        })

        status_icon = "PASS" if eval_res["exact_match"] else ("VALID_SYNTAX" if eval_res["syntax_valid"] else "FAIL")
        print(f"\n[Test #{idx:02d} - {status_icon}] User: {user_query}")
        print(f"  Expected     : {ground_truth.replace(chr(10), ' | ')}")
        print(f"  BEFORE (Base): {before_out[:85].replace(chr(10), ' ')!r}")
        print(f"  AFTER  (LoRA): {prediction.replace(chr(10), ' | ')}")

    total = len(test_cases)
    summary = {
        "stage": "after_post_training",
        "base_model": BASE_MODEL_ID,
        "adapter_dir": ADAPTER_DIR,
        "total_test_cases": total,
        "syntax_valid_count": valid_count,
        "syntax_valid_rate": round(valid_count / total * 100, 1),
        "exact_match_count": exact_count,
        "exact_match_rate": round(exact_count / total * 100, 1),
        "predictions": results,
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/after_training.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("BEFORE vs. AFTER POST-TRAINING SUMMARY")
    print("=" * 65)
    if before_data:
        print(
            f"  Syntax Validity : {before_data['syntax_valid_rate']:5.1f}% ({before_data['syntax_valid_count']}/{total})  --->  "
            f"{summary['syntax_valid_rate']:5.1f}% ({valid_count}/{total})"
        )
        print(
            f"  Exact DSL Match : {before_data['exact_match_rate']:5.1f}% ({before_data['exact_match_count']}/{total})  --->  "
            f"{summary['exact_match_rate']:5.1f}% ({exact_count}/{total})"
        )
    else:
        print(f"  Syntax Validity : {summary['syntax_valid_rate']}% ({valid_count}/{total})")
        print(f"  Exact DSL Match : {summary['exact_match_rate']}% ({exact_count}/{total})")
    print("=" * 65)


if __name__ == "__main__":
    run_after_eval()
