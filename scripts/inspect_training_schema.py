#!/usr/bin/env python3
"""Inspect the training schema without inferring a classification target."""
import argparse
import hashlib
import json
from pathlib import Path

from common import load_table


def inspect_schema(train, sheet="Data", target=None):
    train = Path(train)
    raw = train.read_bytes()
    frame = load_table(train, sheet, raw=raw)
    columns = [str(column) for column in frame.columns]
    if target is not None and target not in frame.columns:
        raise ValueError(f"Specified target {target!r} is absent. Available columns: {columns}")
    return {
        "training_source": {"path": str(train.resolve()), "sheet": sheet,
                            "sha256": hashlib.sha256(raw).hexdigest()},
        "test_data_accessed": False,
        "row_count": len(frame),
        "columns": columns,
        "dtypes": {str(column): str(frame[column].dtype) for column in frame.columns},
        "target": target,
        "target_resolution": "explicitly_supplied" if target is not None else "unresolved",
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
