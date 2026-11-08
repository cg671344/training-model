"""
dsl_validator.py
================
Deterministic parser and evaluator for our custom Smart-Home language: NEXUS-DSL v1.

Why this exists:
A base foundation model has NEVER seen NEXUS-DSL v1 on the internet.
Before post-training, its accuracy on this strict specification is 0%.
After Supervised Fine-Tuning (SFT) with LoRA, it will reliably emit valid,
executable NEXUS-DSL programs and reject out-of-scope requests using the exact
error protocol.

NEXUS-DSL v1 Specification:
---------------------------
Every valid response MUST start with `@NEXUS_V1` on the first line,
contain one or more `;`-terminated statements, and end with `@END`.

Valid statement forms:
1. Action command:
   CMD <domain>.<zone>.<device> := <value>;
   - domains: lum (lighting), hvac (climate), sec (security), media (audio/tv), shade (blinds)
   - zones: kitchen, bedroom, living, study, garage, porch, all
2. Out-of-scope / safety refusal:
   ERR_SCOPE := OUT_OF_DOMAIN;
"""

import re
from typing import Dict, List, Tuple

VALID_DOMAINS = {"lum", "hvac", "sec", "media", "shade"}
VALID_ZONES = {"kitchen", "bedroom", "living", "study", "garage", "porch", "all"}

CMD_PATTERN = re.compile(
    r"^CMD\s+([a-z]+)\.([a-z]+)\.([a-z_]+)\s*:=\s*([A-Z0-9_%+-]+);$"
)
ERR_PATTERN = re.compile(r"^ERR_SCOPE\s*:=\s*OUT_OF_DOMAIN;$")


def normalize_dsl(text: str) -> List[str]:
    """Extract non-empty stripped lines from model output."""
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def validate_nexus_dsl(output_text: str) -> Tuple[bool, str, List[str]]:
    """
    Checks whether `output_text` strictly conforms to NEXUS-DSL v1 syntax.
    Returns:
        (is_valid_syntax, reason, parsed_statements)
    """
    lines = normalize_dsl(output_text)
    if not lines:
        return False, "Empty output", []

    if lines[0] != "@NEXUS_V1":
        return False, f"Missing '@NEXUS_V1' header (got: {lines[0][:30]!r})", []

    if lines[-1] != "@END":
        return False, f"Missing '@END' footer (got: {lines[-1][:30]!r})", []

    body = lines[1:-1]
    if not body:
        return False, "Empty body between @NEXUS_V1 and @END", []

    statements = []
    for idx, stmt in enumerate(body, start=1):
        if ERR_PATTERN.match(stmt):
            statements.append(stmt)
            continue

        m = CMD_PATTERN.match(stmt)
        if not m:
            return False, f"Line {idx} invalid statement syntax: {stmt!r}", []

        domain, zone, _device, _val = m.groups()
        if domain not in VALID_DOMAINS:
            return False, f"Line {idx} unknown domain {domain!r}", []
        if zone not in VALID_ZONES:
            return False, f"Line {idx} unknown zone {zone!r}", []

        statements.append(stmt)

    return True, "Valid NEXUS-DSL v1", statements


def evaluate_prediction(prediction: str, ground_truth: str) -> Dict[str, object]:
    """
    Compares a model prediction against the ground-truth NEXUS-DSL program.
    Returns both syntax validity and exact semantic match.
    """
    is_valid, reason, pred_stmts = validate_nexus_dsl(prediction)
    _, _, gt_stmts = validate_nexus_dsl(ground_truth)

    # Order-independent set comparison of statements (for multi-command prompts)
    exact_match = is_valid and (set(pred_stmts) == set(gt_stmts)) and (len(pred_stmts) == len(gt_stmts))

    return {
        "syntax_valid": is_valid,
        "exact_match": exact_match,
        "reason": reason,
        "pred_statements": pred_stmts,
        "gt_statements": gt_stmts,
    }
