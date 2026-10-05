#!/usr/bin/env python3
"""Inspect training data and cautiously suggest a classification target."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from common import load_table

TARGET_NAMES = {"label", "target", "class", "outcome", "response", "dependent_variable"}


def target_candidates(frame):
    """Rank training-only candidates; the heuristic predicts a column, not its meaning."""
    candidates = []
    for column in frame.columns:
        values = frame[column].dropna()
        distinct = int(values.nunique())
        maximum = max(3, min(20, len(values) // 2))
        if not 2 <= distinct <= maximum:
            continue
        normalized = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        name_evidence = ("conventional_target_name" if normalized in TARGET_NAMES else
                         "target_like_suffix" if normalized.endswith(("_label", "_target")) else
                         "none")
        completeness = float(len(values) / len(frame))
        nonnumeric = not pd.api.types.is_numeric_dtype(frame[column])
        score = ({"conventional_target_name": 8, "target_like_suffix": 5, "none": 0}[name_evidence]
                 + (2 if distinct == 2 else 1 if distinct <= 5 else 0)
                 + (1 if nonnumeric else 0)
                 + (1 if completeness == 1 else 0))
        reasons = []
        if name_evidence != "none":
            reasons.append(name_evidence)
        reasons.append("binary_cardinality" if distinct == 2 else "low_class_cardinality")
        if nonnumeric:
            reasons.append("categorical_storage")
        if completeness == 1:
            reasons.append("no_missing_candidate_labels")
        candidates.append({"column": str(column), "distinct_nonmissing_values": distinct,
                           "nonmissing_rows": len(values), "name_evidence": name_evidence,
                           "completeness": completeness, "storage_evidence":
                           "categorical_or_text" if nonnumeric else "numeric",
                           "inference_score": score, "inference_reasons": reasons})
    return sorted(candidates, key=lambda item: (-item["inference_score"],
                                                 item["distinct_nonmissing_values"],
                                                 -item["completeness"], item["column"]))


def inspect_schema(train, sheet="Data", target=None):
    train = Path(train)
    raw = train.read_bytes()
    frame = load_table(train, sheet, raw=raw)
    columns = [str(column) for column in frame.columns]
    if target is not None and target not in frame.columns:
        raise ValueError(f"Specified target {target!r} is absent. Available columns: {columns}")
    candidates = target_candidates(frame) if target is None else []
    selected = candidates[0] if candidates else None
    suggested = selected["column"] if selected else None
    tied = ([item["column"] for item in candidates
             if item["inference_score"] == selected["inference_score"]]
            if selected else [])
    return {
        "training_source": {"path": str(train.resolve()), "sheet": sheet,
                            "sha256": hashlib.sha256(raw).hexdigest()},
        "test_data_accessed": False,
        "row_count": len(frame),
        "columns": columns,
        "dtypes": {str(column): str(frame[column].dtype) for column in frame.columns},
        "target": target if target is not None else suggested,
        "target_resolution": ("explicitly_supplied" if target is not None else
                              "ai_inferred" if suggested is not None else "unresolved"),
        "target_candidates": candidates,
        "target_inference": (None if target is not None or selected is None else {
            "method": "training_only_ranked_heuristic",
            "selected_column": suggested,
            "selected_score": selected["inference_score"],
            "selected_reasons": selected["inference_reasons"],
            "equally_scored_candidates": tied,
            "tie_breaker": ("lower class cardinality, then greater completeness, then lexical column name"
                            if len(tied) > 1 else None),
            "semantics_confirmed": False,
        }),
        "target_inference_note": ("The AI selected the highest-ranked classification-compatible column "
                                  "using the recorded training-only heuristic. This is a reproducible guess, "
                                  "not confirmation of task or label semantics."
                                  if suggested is not None else
                                  "No classification-compatible target candidate was found; ask the user "
                                  "or obtain authoritative metadata." if target is None else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--sheet", default="Data")
    parser.add_argument("--target", help="Target explicitly supplied by the user or authoritative task metadata")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_schema(args.train, args.sheet, args.target)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
