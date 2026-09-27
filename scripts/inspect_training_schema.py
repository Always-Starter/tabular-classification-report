#!/usr/bin/env python3
"""Inspect training data and cautiously suggest a classification target."""
import argparse
import hashlib
import json
from pathlib import Path

from common import load_table

TARGET_NAMES = {"label", "target", "class", "outcome", "response", "dependent_variable"}


def target_candidates(frame):
    """Return training-only candidates; values establish suitability, not meaning."""
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
        candidates.append({"column": str(column), "distinct_nonmissing_values": distinct,
                           "nonmissing_rows": len(values), "name_evidence": name_evidence})
    return candidates


def inspect_schema(train, sheet="Data", target=None):
    train = Path(train)
    raw = train.read_bytes()
    frame = load_table(train, sheet, raw=raw)
    columns = [str(column) for column in frame.columns]
    if target is not None and target not in frame.columns:
        raise ValueError(f"Specified target {target!r} is absent. Available columns: {columns}")
    candidates = target_candidates(frame) if target is None else []
    strong = [item for item in candidates if item["name_evidence"] == "conventional_target_name"]
    suggested = strong[0]["column"] if len(strong) == 1 else None
    return {
        "training_source": {"path": str(train.resolve()), "sheet": sheet,
                            "sha256": hashlib.sha256(raw).hexdigest()},
        "test_data_accessed": False,
        "row_count": len(frame),
        "columns": columns,
        "dtypes": {str(column): str(frame[column].dtype) for column in frame.columns},
        "target": target if target is not None else suggested,
        "target_resolution": ("explicitly_supplied" if target is not None else
                              "provisionally_inferred" if suggested is not None else "unresolved"),
        "target_candidates": candidates,
        "target_inference_note": ("Training-only naming and cardinality evidence suggests a possible "
                                  "classification target; this does not establish task semantics."
                                  if suggested is not None else
                                  "No unique conventional, classification-compatible target name; "
                                  "ask the user to identify the target." if target is None else None),
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
