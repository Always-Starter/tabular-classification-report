#!/usr/bin/env python3
"""Freeze one selected development variant, all pipelines, and their evidence."""
import argparse
from pathlib import Path
from common import code_hashes, environment, read_json, sha, utcnow, write_json
from verify_results import verify_training


def freeze(results, model_dir, review_path, output, require_human_approval=False):
    results, model_dir, output = Path(results), Path(model_dir), Path(output)
    r, review = read_json(results), read_json(review_path)
    verify_training(results)
    required = {"selected_variant", "preferred_model", "rationale", "sensitivity_review", "warnings_review"}
    if r.get("schema_version", 2) >= 5:
        required |= {"selection_rule", "tie_breaker"}
    if set(review) != required or any(not isinstance(v, str) or not v.strip() for v in review.values()):
        raise ValueError(f"Review must contain nonempty text fields: {sorted(required)}")
    if review["selected_variant"] != "baseline":
        raise ValueError("Sensitivity variants are interpretive and cannot be locked. To adopt one, create and "
                         "validate a new independent plan/run with that configuration as its baseline.")
    selected = r["variants"][review["selected_variant"]]
    if selected.get("selection_eligible", True) is not True:
        raise ValueError("The selected development variant is not eligible for Model Lock")
    if review["preferred_model"] not in selected["models"]:
        raise ValueError("Preferred model is not in selected variant")
    if r["code_sha256"] != code_hashes() or r["environment"] != environment():
        raise ValueError("Code or environment changed since CV; rerun development before locking")
    artifacts = {}
    for name, details in selected["models"].items():
        artifact = details["artifact"]
        if sha(model_dir / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Model changed since development: {name}")
        artifacts[name] = artifact
    lock = {"schema_version": 2, "status": "frozen", "created_at": utcnow(),
            "human_approval_required": require_human_approval,
            "plan": selected["plan"], "class_order": r["class_order"], "review": review,
            "training_source": r["training_source"], "eligible_training_rows": r["eligible_rows"],
            "models": selected["models"], "artifacts": artifacts,
            "environment": r["environment"], "code_sha256": r["code_sha256"],
            "training_results_sha256": sha(results),
            "policy": "Evaluate the frozen plan at most once; never retune after held-out access."}
    write_json(output, lock, exclusive=True)
    lock_hash = sha(output)
    write_json(output.with_name(output.name + ".seal.json"),
               {"lock_sha256": lock_hash, "human_approval_required": require_human_approval,
                "sealed_at": utcnow()}, exclusive=True)
    print(f"Frozen Model Lock SHA-256: {lock_hash}")
    return lock


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-human-approval", action="store_true",
                   help="Require an explicit approval of this lock before held-out evaluation")
    a = p.parse_args()
    freeze(a.results, a.model_dir, a.review, a.output, a.require_human_approval)
