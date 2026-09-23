#!/usr/bin/env python3
"""Evaluate all already-fitted, approved pipelines in one held-out data read."""
import argparse
import hashlib
from pathlib import Path
import joblib
import numpy as np
from common import (code_hashes, environment, features, load_table, predictions, read_json,
                    score_metrics, sha, utcnow, write_json)
from validate_plan import validate


def evaluate(test, lock_path, model_dir, output_dir, sheet="Data"):
    test, lock_path, model_dir, output_dir = map(Path, (test, lock_path, model_dir, output_dir))
    lock = read_json(lock_path)
    plan = validate(lock["plan"])
    lock_hash = sha(lock_path)
    approval_path = lock_path.with_name(lock_path.name + ".approval.json")
    if not approval_path.exists():
        raise ValueError("Human approval is required before opening the test file")
    approval = read_json(approval_path)
    if approval.get("status") != "approved" or approval.get("lock_sha256") != lock_hash:
        raise ValueError("Approval does not match this exact Model Lock")
    if code_hashes() != lock["code_sha256"] or environment() != lock["environment"]:
        raise ValueError("Code/environment changed since locking; test remains sealed")
    if test.resolve() == Path(lock["training_source"]["path"]).resolve():
        raise ValueError("The training file cannot also be the held-out file")
    order = lock["class_order"]
    models = {}
    for name, artifact in lock["artifacts"].items():
        path = (model_dir / artifact["path"]).resolve()
        if not path.is_relative_to(model_dir.resolve()) or sha(path) != artifact["sha256"]:
            raise ValueError(f"Frozen model hash/path mismatch: {name}")
        # Load only locally produced, trusted joblib files; hashes do not make arbitrary pickle safe.
        models[name] = joblib.load(path)
        if list(models[name].classes_) != order or list(models[name].feature_names_in_) != plan["features"]:
            raise ValueError(f"Frozen model schema mismatch: {name}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use an empty evaluation output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = lock_path.with_name(lock_path.name + ".holdout.json")
    receipt = {"status": "started", "lock_sha256": lock_hash, "started_at": utcnow(),
               "test_path": str(test.resolve()), "output_dir": str(output_dir.resolve())}
    # Atomic exclusive creation is next to the lock, not in the user-selected output directory.
    write_json(receipt_path, receipt, exclusive=True)
    try:
        raw = test.read_bytes()
        test_hash = hashlib.sha256(raw).hexdigest()
        if test_hash == lock["training_source"]["sha256"]:
            raise ValueError("Held-out file is a copy of the training file")
        frame = load_table(test, sheet, plan["target"], raw=raw)
        x = features(frame, plan)
        if plan["target"] in frame:
            present = frame[plan["target"]].notna().to_numpy()
            actual = [str(value) if ok else None for value, ok in zip(frame[plan["target"]], present, strict=True)]
        else:
            actual = [None] * len(frame)
        mask = np.asarray([v is not None for v in actual])
        y = np.asarray([v for v in actual if v is not None])
        if not set(y) <= set(order):
            raise ValueError("Test contains unseen target classes; no model changes are permitted")
        result = {"schema_version": 2, "lock_sha256": lock_hash, "approval": approval,
                  "test_sha256": test_hash, "test_rows": len(frame), "labelled_rows": int(mask.sum()),
                  "missing_labels": int((~mask).sum()), "class_order": order,
                  "plan": plan, "models": {}, "created_at": utcnow()}
        for name, model in models.items():
            pred, prob = predictions(model, x, order, plan)
            prediction_path = output_dir / f"{name}_predictions.json"
            write_json(prediction_path, {"class_order": order, "records": [
                {"source_row": i, "actual": actual[i], "prediction": str(pred[i]), "probabilities": prob[i].tolist()}
                for i in range(len(frame))]}, exclusive=True)
            scores = score_metrics(y, pred[mask], prob[mask], order, plan)
            scores["predictions"] = {"path": prediction_path.name, "sha256": sha(prediction_path)}
            result["models"][name] = scores
        write_json(output_dir / "test_results.json", result, exclusive=True)
        receipt.update(status="completed", completed_at=utcnow(), test_sha256=test_hash,
                       results_sha256=sha(output_dir / "test_results.json"))
        write_json(receipt_path, receipt)
        return result
    except Exception as exc:
        receipt.update(status="failed_after_access_reserved", error=str(exc), failed_at=utcnow())
        write_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--sheet", default="Data")
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    evaluate(a.test, a.lock, a.model_dir, a.output_dir, a.sheet)
