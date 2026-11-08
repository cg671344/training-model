"""
01_prepare_data.py
==================
Step 1 of Post-Training: Organizing & Validating Training Data.

This script generates:
1. `data/sft_train.jsonl`    — Supervised Fine-Tuning (SFT) training dataset (messages format)
2. `data/sft_val.jsonl`      — Validation split to monitor overfitting during training
3. `data/test_heldout.jsonl` — Held-out evaluation set with UNSEEN phrasings & combinations
4. `data/dpo_train.jsonl`    — Preference dataset (prompt / chosen / rejected) for Stage 2 DPO

Data Format Used (Standard Hugging Face / OpenAI ChatML JSONL):
---------------------------------------------------------------
Each line in `sft_train.jsonl` is a JSON object:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "@NEXUS_V1\nCMD ...\n@END"}
  ]
}
"""

import json
import os
import random
from pathlib import Path
from dsl_validator import validate_nexus_dsl

SYSTEM_PROMPT = (
    "You are the NEXUS-DSL v1 Smart Home Compiler. "
    "Translate the user's request into a strict NEXUS-DSL v1 program (@NEXUS_V1 ... @END). "
    "If the request is unrelated to smart home device control, emit ERR_SCOPE := OUT_OF_DOMAIN;."
)


def wrap_dsl(*statements: str) -> str:
    body = "\n".join(statements)
    return f"@NEXUS_V1\n{body}\n@END"


def build_datasets():
    random.seed(42)

    # 1. Define rich templates for single-intent commands across 5 domains & 7 zones
    zones = [
        ("kitchen", ["in the kitchen", "for the kitchen", "kitchen"]),
        ("bedroom", ["in the bedroom", "for the master bedroom", "bedroom"]),
        ("living", ["in the living room", "for the lounge", "living room"]),
        ("study", ["in the study", "in my office", "for the study room"]),
        ("garage", ["in the garage", "for the garage", "garage"]),
        ("porch", ["on the front porch", "for the porch", "porch"]),
    ]

    single_examples = []

    # Domain 1: Lighting (lum)
    for zone_id, zone_phrases in zones:
        for zp in zone_phrases:
            single_examples.extend([
                (f"Turn on the lights {zp}.", f"CMD lum.{zone_id}.main := ON;"),
                (f"Switch off the light {zp}.", f"CMD lum.{zone_id}.main := OFF;"),
                (f"Please kill the lights {zp}.", f"CMD lum.{zone_id}.main := OFF;"),
                (f"Power up the main lamp {zp}.", f"CMD lum.{zone_id}.main := ON;"),
            ])
        for level in [20, 30, 50, 75, 80, 100]:
            zp = random.choice(zone_phrases)
            single_examples.append(
                (f"Dim the lights {zp} to {level}%.", f"CMD lum.{zone_id}.dimmer := {level}%;")
            )
            zp2 = random.choice(zone_phrases)
            single_examples.append(
                (f"Set brightness {zp2} to {level} percent.", f"CMD lum.{zone_id}.dimmer := {level}%;")
            )

    # Domain 2: Climate / HVAC (hvac)
    for zone_id, zone_phrases in zones[:4]:  # indoor rooms
        for temp in [18, 19, 20, 21, 22, 23, 24, 25, 26]:
            zp = random.choice(zone_phrases)
            single_examples.append(
                (f"Set the temperature {zp} to {temp} degrees.", f"CMD hvac.{zone_id}.temp := {temp}C;")
            )
            zp2 = random.choice(zone_phrases)
            single_examples.append(
                (f"Change the thermostat {zp2} to {temp}C.", f"CMD hvac.{zone_id}.temp := {temp}C;")
            )
        for zp in zone_phrases[:2]:
            single_examples.extend([
                (f"Turn on the AC {zp}.", f"CMD hvac.{zone_id}.ac := ON;"),
                (f"Turn off the air conditioner {zp}.", f"CMD hvac.{zone_id}.ac := OFF;"),
                (f"Switch the heater on {zp}.", f"CMD hvac.{zone_id}.heater := ON;"),
                (f"Turn off the heating {zp}.", f"CMD hvac.{zone_id}.heater := OFF;"),
            ])

    # Domain 3: Security (sec)
    for zone_id, zone_phrases in zones:
        for zp in zone_phrases[:2]:
            single_examples.extend([
                (f"Lock the door {zp}.", f"CMD sec.{zone_id}.lock := LOCKED;"),
                (f"Unlock the door {zp}.", f"CMD sec.{zone_id}.lock := UNLOCKED;"),
            ])
    single_examples.extend([
        ("Arm the whole house security system.", "CMD sec.all.alarm := ARMED;"),
        ("Activate home alarm across all zones.", "CMD sec.all.alarm := ARMED;"),
        ("Disarm the house alarm.", "CMD sec.all.alarm := DISARMED;"),
        ("Turn off all security alarms.", "CMD sec.all.alarm := DISARMED;"),
        ("Shut off every light in the house.", "CMD lum.all.main := OFF;"),
        ("Turn on all lights in the house.", "CMD lum.all.main := ON;"),
    ])

    # Domain 4: Shades / Blinds (shade)
    for zone_id, zone_phrases in zones[:4]:
        for zp in zone_phrases[:2]:
            single_examples.extend([
                (f"Open the blinds {zp}.", f"CMD shade.{zone_id}.blinds := OPEN;"),
                (f"Close the curtains {zp}.", f"CMD shade.{zone_id}.blinds := CLOSED;"),
                (f"Lower the shades {zp}.", f"CMD shade.{zone_id}.blinds := CLOSED;"),
                (f"Raise the window blinds {zp}.", f"CMD shade.{zone_id}.blinds := OPEN;"),
            ])

    # Domain 5: Media (media)
    for zone_id, zone_phrases in [("living", ["in the living room", "living room"]), ("bedroom", ["in the bedroom", "bedroom"]), ("study", ["in the study", "in my office"])]:
        for zp in zone_phrases:
            single_examples.extend([
                (f"Turn on the TV {zp}.", f"CMD media.{zone_id}.tv := ON;"),
                (f"Switch off the television {zp}.", f"CMD media.{zone_id}.tv := OFF;"),
                (f"Mute the speaker {zp}.", f"CMD media.{zone_id}.speaker := MUTE;"),
                (f"Pause the music {zp}.", f"CMD media.{zone_id}.speaker := PAUSE;"),
                (f"Play music on the speaker {zp}.", f"CMD media.{zone_id}.speaker := PLAY;"),
            ])

    random.shuffle(single_examples)

    # 2. Build Multi-Intent (2 commands in one request) examples
    multi_examples = []
    connectors = [" and ", ", and also ", ". Also, ", " then "]
    for _ in range(90):
        (u1, c1), (u2, c2) = random.sample(single_examples, 2)
        if c1 == c2:
            continue
        conn = random.choice(connectors)
        u2_clean = u2[0].lower() + u2[1:] if conn != ". Also, " else u2
        combined_user = u1.rstrip(".") + conn + u2_clean
        multi_examples.append((combined_user, f"{c1}\n{c2}"))

    # 3. Build Out-of-Scope / Refusal examples (teaches the model WHEN NOT to hallucinate commands)
    out_of_scope_prompts = [
        "What is the capital of France?",
        "Write a Python function to reverse a linked list.",
        "Can you order me a pepperoni pizza from Domino's?",
        "Tell me a funny joke about robots.",
        "How do I boil an egg perfectly?",
        "Who won the FIFA World Cup in 2022?",
        "Explain quantum entanglement in simple terms.",
        "Translate 'good morning' into Japanese.",
        "What is 256 multiplied by 19?",
        "Write an email to my boss asking for Friday off.",
        "Book a flight to Tokyo next Monday.",
        "Recommend a good sci-fi movie to watch tonight.",
        "What is the stock price of NVIDIA today?",
        "Summarize the plot of Hamlet.",
        "How many calories are in an avocado?",
        "Create a workout plan for building muscle.",
        "Why is the sky blue?",
        "Fix the syntax error in my SQL query.",
        "Bake a chocolate cake for me.",
        "Walk my dog in the park.",
        "Solve x^2 - 5x + 6 = 0.",
        "Who is the CEO of Google?",
        "Draft a resume for a software engineer.",
        "Give me relationship advice.",
    ]
    refusal_examples = [(p, "ERR_SCOPE := OUT_OF_DOMAIN;") for p in out_of_scope_prompts]

    # Combine & format into SFT records
    all_raw = single_examples[:160] + multi_examples[:80] + refusal_examples * 2
    random.shuffle(all_raw)

    def to_sft_record(user_text: str, dsl_body: str):
        assistant_text = wrap_dsl(dsl_body)
        is_valid, reason, _ = validate_nexus_dsl(assistant_text)
        assert is_valid, f"Dataset bug: {reason} in {assistant_text}"
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        }

    sft_records = [to_sft_record(u, c) for u, c in all_raw]
    train_records = sft_records[:-30]
    val_records = sft_records[-30:]

    # 4. Construct a carefully curated HELD-OUT TEST SET (20 unseen test cases)
    # Includes single commands, multi-intent commands, and out-of-scope traps.
    heldout_raw = [
        # Single-intent unseen phrasings
        ("Could you switch on the kitchen lights please?", "CMD lum.kitchen.main := ON;"),
        ("Make it dark in the bedroom—turn off the main light.", "CMD lum.bedroom.main := OFF;"),
        ("Set the living room dimmer to 40%.", "CMD lum.living.dimmer := 40%;"),
        ("Set the thermostat in my office to 21 degrees.", "CMD hvac.study.temp := 21C;"),
        ("Turn off the AC in the bedroom right now.", "CMD hvac.bedroom.ac := OFF;"),
        ("Please lock the garage door.", "CMD sec.garage.lock := LOCKED;"),
        ("Unlock the front porch door.", "CMD sec.porch.lock := UNLOCKED;"),
        ("Arm the security alarm for the entire house.", "CMD sec.all.alarm := ARMED;"),
        ("Open the window blinds in the study.", "CMD shade.study.blinds := OPEN;"),
        ("Close the living room curtains.", "CMD shade.living.blinds := CLOSED;"),
        ("Mute the living room speaker.", "CMD media.living.speaker := MUTE;"),
        ("Turn off every light in the whole house.", "CMD lum.all.main := OFF;"),
        # Multi-intent unseen combinations
        (
            "Turn on the lights in the study and set the temperature in the study to 23 degrees.",
            "CMD lum.study.main := ON;\nCMD hvac.study.temp := 23C;",
        ),
        (
            "Close the blinds in the bedroom and switch off the television in the bedroom.",
            "CMD shade.bedroom.blinds := CLOSED;\nCMD media.bedroom.tv := OFF;",
        ),
        (
            "Lock the door in the kitchen and arm the whole house security system.",
            "CMD sec.kitchen.lock := LOCKED;\nCMD sec.all.alarm := ARMED;",
        ),
        (
            "Turn on the AC in the living room and dim the lights in the living room to 50%.",
            "CMD hvac.living.ac := ON;\nCMD lum.living.dimmer := 50%;",
        ),
        # Out-of-scope traps
        ("What is the weather forecast in New York tomorrow?", "ERR_SCOPE := OUT_OF_DOMAIN;"),
        ("Write a haiku about autumn leaves.", "ERR_SCOPE := OUT_OF_DOMAIN;"),
        ("How do I install PyTorch on Windows?", "ERR_SCOPE := OUT_OF_DOMAIN;"),
        ("Calculate the square root of 144.", "ERR_SCOPE := OUT_OF_DOMAIN;"),
    ]
    test_records = [to_sft_record(u, c) for u, c in heldout_raw]

    # 5. Construct Stage 2 DPO Preference Dataset (prompt, chosen, rejected)
    # Teaches the model to prefer pure NEXUS-DSL without chatty prose or markdown code fences.
    dpo_records = []
    for u, c in all_raw[:60]:
        chosen_dsl = wrap_dsl(c)
        # Common failure modes we want DPO to penalize:
        rejected_variants = [
            f"Sure! Here is your NEXUS-DSL command:\n```nexus\n{chosen_dsl}\n```\nLet me know if you need anything else!",
            f"{chosen_dsl}\nNote: Command executed successfully.",
            f"I have processed your request:\n{c}",
        ]
        dpo_records.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": u},
            ],
            "chosen": [{"role": "assistant", "content": chosen_dsl}],
            "rejected": [{"role": "assistant", "content": random.choice(rejected_variants)}],
        })

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    def write_jsonl(path: Path, rows: list):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "sft_train.jsonl", train_records)
    write_jsonl(out_dir / "sft_val.jsonl", val_records)
    write_jsonl(out_dir / "test_heldout.jsonl", test_records)
    write_jsonl(out_dir / "dpo_train.jsonl", dpo_records)

    print("=" * 65)
    print("STEP 1 COMPLETE: Post-Training Datasets Generated & Validated")
    print("=" * 65)
    print(f"  - data/sft_train.jsonl    : {len(train_records):3d} examples (SFT training split)")
    print(f"  - data/sft_val.jsonl      : {len(val_records):3d} examples (SFT validation split)")
    print(f"  - data/test_heldout.jsonl : {len(test_records):3d} examples (Held-out Before/After test set)")
    print(f"  - data/dpo_train.jsonl    : {len(dpo_records):3d} examples (Stage 2 DPO preference pairs)")
    print("\nSample SFT Training Record:")
    print(json.dumps(train_records[0], indent=2))


if __name__ == "__main__":
    build_datasets()
