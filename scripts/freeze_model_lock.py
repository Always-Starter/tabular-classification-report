#!/usr/bin/env python3
"""Freeze one reviewed development variant, all pipelines, and their evidence."""
import argparse
from pathlib import Path
from common import code_hashes, environment, read_json, sha, utcnow, write_json
from verify_results import verify_training


def freeze(results, model_dir, review_path, output):
    results, model_dir, output = Path(results), Path(model_dir), Path(output)
    r, review = read_json(results), read_json(review_path)
    verify_training(results)
    required = {"selected_variant", "preferred_model", "rationale", "sensitivity_review", "warnings_review"}
    if set(review) != required or any(not isinstance(v, str) or not v.strip() for v in review.values()):
        raise ValueError(f"Review must contain nonempty text fields: {sorted(required)}")
    selected = r["variants"][review["selected_variant"]]
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
    lock = {"schema_version": 2, "status": "pending_human_approval", "created_at": utcnow(),
            "plan": selected["plan"], "class_order": r["class_order"], "review": review,
            "training_source": r["training_source"], "eligible_training_rows": r["eligible_rows"],
            "models": selected["models"], "artifacts": artifacts,
            "environment": r["environment"], "code_sha256": r["code_sha256"],
            "training_results_sha256": sha(results),
            "policy": "Approve exact lock digest before one holdout read; never retune after that read."}
    write_json(output, lock, exclusive=True)
    print(f"Lock SHA-256 for review: {sha(output)}")
    return lock


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    freeze(a.results, a.model_dir, a.review, a.output)
