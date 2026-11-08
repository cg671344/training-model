"""
02_eval_before.py
=================
Step 2 of Post-Training: Baseline Evaluation ("Before Training").

Loads the raw Base Foundation Model (`Qwen/Qwen2.5-0.5B`) BEFORE any SFT/LoRA
post-training and tests it on the 20 unseen prompts in `data/test_heldout.jsonl`.

Saves detailed results to `results/before_training.json`.
"""

import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dsl_validator import evaluate_prediction

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"


def run_baseline_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 65)
    print(f"STEP 2: Evaluating Base Foundation Model BEFORE Post-Training")
    print(f"Model : {BASE_MODEL_ID} | Device: {device} ({dtype})")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    with open("data/test_heldout.jsonl", "r", encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f]

    results = []
    valid_count = 0
    exact_count = 0

    for idx, item in enumerate(test_cases, start=1):
        messages = item["messages"]
        prompt_messages = messages[:-1]
        ground_truth = messages[-1]["content"]
        user_query = messages[1]["content"]

        # Format using ChatML template
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

        results.append({
            "id": idx,
            "user_query": user_query,
            "ground_truth": ground_truth,
            "base_model_output": prediction,
            "syntax_valid": eval_res["syntax_valid"],
            "exact_match": eval_res["exact_match"],
            "reason": eval_res["reason"],
        })

        if idx <= 5:
            print(f"\n[Test #{idx}] User: {user_query}")
            print(f"  Expected : {ground_truth.replace(chr(10), ' | ')}")
            print(f"  Base Out : {prediction[:100].replace(chr(10), ' ')!r}")
            print(f"  Status   : Valid={eval_res['syntax_valid']} | ExactMatch={eval_res['exact_match']} ({eval_res['reason']})")

    total = len(test_cases)
    summary = {
        "stage": "before_post_training",
        "model": BASE_MODEL_ID,
        "total_test_cases": total,
        "syntax_valid_count": valid_count,
        "syntax_valid_rate": round(valid_count / total * 100, 1),
        "exact_match_count": exact_count,
        "exact_match_rate": round(exact_count / total * 100, 1),
        "predictions": results,
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/before_training.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"BEFORE TRAINING SUMMARY ({BASE_MODEL_ID}):")
    print(f"  - Valid NEXUS-DSL Syntax : {valid_count}/{total} ({summary['syntax_valid_rate']}%)")
    print(f"  - Exact Command Match    : {exact_count}/{total} ({summary['exact_match_rate']}%)")
    print("=" * 65)


if __name__ == "__main__":
    run_baseline_eval()
