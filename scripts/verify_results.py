#!/usr/bin/env python3
"""Recompute saved metrics without reopening test data or refitting models."""
import argparse
from pathlib import Path
import numpy as np
from common import read_json, score_metrics, sha, write_json


def compare(saved, calculated):
    if saved["confusion_matrix"] != calculated["confusion_matrix"]:
        raise ValueError("Confusion matrix differs from saved predictions")
    if saved["class_order"] != calculated["class_order"] or saved["labelled_rows"] != calculated["labelled_rows"]:
        raise ValueError("Class order or row count mismatch")
    for metric, expected in calculated["metrics"].items():
        observed = saved["metrics"][metric]
        if (expected is None) != (observed is None) or expected is not None and abs(expected - observed) > 1e-10:
            raise ValueError(f"Metric mismatch: {metric}")
    if saved["undefined_metrics"] != calculated["undefined_metrics"]:
        raise ValueError("Undefined metric explanations differ")


def recompute(records, order, plan):
    labelled = [r for r in records if r["actual"] is not None]
    y = np.asarray([r["actual"] for r in labelled])
    pred = np.asarray([r["prediction"] for r in labelled])
    prob = np.asarray([r["probabilities"] for r in labelled]).reshape(len(labelled), len(order))
    # Validate decision rule on unlabelled rows too.
    for row in records:
        probabilities = np.asarray(row["probabilities"])
        if len(probabilities) != len(order) or not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1):
            raise ValueError("Invalid saved probabilities")
        if plan["task"] == "binary":
            pos = plan["positive_class"]
            expected = pos if probabilities[order.index(pos)] >= plan["threshold"] else next(c for c in order if c != pos)
        else:
            expected = order[int(np.argmax(probabilities))]
        if row["prediction"] != expected:
            raise ValueError("Saved prediction does not follow frozen decision rule")
    return score_metrics(y, pred, prob, order, plan)


def verify(path, lock_path):
    path, lock_path = Path(path), Path(lock_path)
    r, lock = read_json(path), read_json(lock_path)
    receipt = read_json(lock_path.with_name(lock_path.name + ".holdout.json"))
    approval = read_json(lock_path.with_name(lock_path.name + ".approval.json"))
    if (r["lock_sha256"] != sha(lock_path) or receipt["lock_sha256"] != sha(lock_path)
            or approval != r["approval"] or approval["lock_sha256"] != sha(lock_path)
            or approval["status"] != "approved" or receipt["status"] != "completed"
            or receipt["results_sha256"] != sha(path) or receipt["test_sha256"] != r["test_sha256"]):
        raise ValueError("Lock, approval, receipt or result integrity mismatch")
    if r["plan"] != lock["plan"] or r["class_order"] != lock["class_order"] or set(r["models"]) != set(lock["models"]):
        raise ValueError("Results differ from frozen plan")
    examples = {}
    for name, saved in r["models"].items():
        prediction_path = (path.parent / saved["predictions"]["path"]).resolve()
        if not prediction_path.is_relative_to(path.parent.resolve()) or sha(prediction_path) != saved["predictions"]["sha256"]:
            raise ValueError("Prediction file integrity mismatch")
        predictions = read_json(prediction_path)
        records = predictions["records"]
        if predictions["class_order"] != r["class_order"] or [v["source_row"] for v in records] != list(range(r["test_rows"])):
            raise ValueError("Prediction order/count mismatch")
        calculated = recompute(records, r["class_order"], r["plan"])
        compare(saved, calculated)
        if calculated["labelled_rows"] != r["labelled_rows"] or r["missing_labels"] + r["labelled_rows"] != r["test_rows"]:
            raise ValueError("Labelled/missing row counts do not reconcile")
        cm = np.asarray(calculated["confusion_matrix"])
        # Independent count arithmetic gives a concrete human-verification candidate.
        numerator, denominator = int(np.trace(cm)), int(cm.sum())
        examples[name] = {"metric": "accuracy", "correct": numerator, "labelled_rows": denominator,
                          "calculation": f"{numerator}/{denominator}",
                          "value": numerator / denominator if denominator else None,
                          "human_verified": False}
    return {"verified": True, "results_sha256": sha(path), "manual_check_candidates": examples,
            "note": "Automated checks are not evidence that a human performed the Reflection verification."}


def verify_training(path):
    path = Path(path)
    result = read_json(path)
    for variant in result["variants"].values():
        for name, model in variant["models"].items():
            source = (path.parent / model["oof"]["path"]).resolve()
            if not source.is_relative_to(path.parent.resolve()) or sha(source) != model["oof"]["sha256"]:
                raise ValueError(f"OOF integrity mismatch: {name}")
            payload = read_json(source)
            records = payload["records"]
            if payload["class_order"] != result["class_order"] or len(records) != model["oof_rows"]:
                raise ValueError("OOF class order or row count mismatch")
            for fold, split in zip(model["fold_results"], result["outer_splits"], strict=True):
                subset = [r for r in records if r["fold"] == fold["fold"]]
                if [r["source_row"] for r in subset] != split["valid_source_rows"]:
                    raise ValueError("OOF rows do not match saved validation split")
                compare(fold, recompute(subset, result["class_order"], variant["plan"]))
            for metric, summary in model["outer_summary"].items():
                values = [f["metrics"][metric] for f in model["fold_results"]]
                if values != summary["folds"] or summary["defined_folds"] != sum(v is not None for v in values):
                    raise ValueError("Outer summary mismatch")
                if any(v is None for v in values):
                    if summary["mean"] is not None or summary["std"] is not None:
                        raise ValueError("Undefined fold scores cannot become a numerical summary")
                elif (not np.isclose(np.mean(values), summary["mean"])
                      or not np.isclose(np.std(values, ddof=1), summary["std"])):
                    raise ValueError("Outer summary mismatch")
    return {"verified": True, "training_results_sha256": sha(path)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = verify(a.results, a.lock)
    if a.output:
        write_json(a.output, result)
    print(result)
