#!/usr/bin/env python3
"""Training-only diagnosis for Checkpoint 1 of tabular-classification-report."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr
from common import load_table


def json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def cramers_v(x, y):
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return None
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    rows, cols = table.shape
    corrected = max(0, phi2 - ((cols - 1) * (rows - 1)) / max(n - 1, 1))
    corrected_rows = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    corrected_cols = cols - ((cols - 1) ** 2) / max(n - 1, 1)
    denominator = min(corrected_cols - 1, corrected_rows - 1)
    return math.sqrt(corrected / denominator) if denominator > 0 else None


def diagnose(frame, target):
    if target not in frame.columns:
        raise KeyError(f"Target column {target!r} was not found")
    predictors = frame.drop(columns=[target])
    labels = frame[target]
    valid = labels.notna()
    valid_labels = labels[valid]
    classes = list(pd.unique(valid_labels))
    label_codes = pd.Series(pd.Categorical(valid_labels, categories=classes).codes, index=valid_labels.index)

    result = {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(frame[column].dtype) for column in frame.columns},
        "target": {
            "name": target,
            "dtype": str(labels.dtype),
            "unique_nonmissing": int(labels.nunique(dropna=True)),
            "missing_count": int(labels.isna().sum()),
            "counts": {str(key): int(value) for key, value in labels.value_counts(dropna=False).items()},
            "proportions_nonmissing": {
                str(key): float(value)
                for key, value in valid_labels.value_counts(normalize=True).items()
            },
        },
        "missing_predictors": {},
        "duplicates": {
            "exact_duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_predictor_rows": int(predictors.duplicated().sum()),
        },
        "identifier_candidates": [],
        "numeric": {},
        "categorical": {},
        "univariate_associations": {},
    }

    for column in predictors.columns:
        series = predictors[column]
        missing = int(series.isna().sum())
        result["missing_predictors"][str(column)] = {
            "count": missing,
            "percent": 100 * missing / len(frame) if len(frame) else None,
        }
        nonmissing = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        uniqueness = unique / nonmissing if nonmissing else 0
        name_signal = bool(re.search(r"(^|_)(id|uuid|key|number|account)($|_)", str(column).lower()))
        if uniqueness >= 0.98 or name_signal:
            result["identifier_candidates"].append({
                "column": str(column),
                "unique_nonmissing": unique,
                "nonmissing": nonmissing,
                "uniqueness_ratio": uniqueness,
                "name_signal": name_signal,
            })

    numeric_columns = list(predictors.select_dtypes(include=[np.number]).columns)
    for column in numeric_columns:
        series = predictors[column]
        clean = series.replace([np.inf, -np.inf], np.nan)
        quantiles = clean.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
        result["numeric"][str(column)] = {
            "count": int(series.notna().sum()),
            "nonfinite_count": int(np.isinf(series).sum()),
            "mean": json_value(clean.mean()),
            "std": json_value(clean.std()),
            "skew": json_value(clean.skew()),
            "quantiles": {str(key): json_value(value) for key, value in quantiles.items()},
            "negative_count": int((clean < 0).sum()),
            "zero_count": int((clean == 0).sum()),
        }

    categorical_columns = [column for column in predictors.columns if column not in numeric_columns]
    for column in categorical_columns:
        series = predictors[column]
        counts = series.value_counts(dropna=False)
        result["categorical"][str(column)] = {
            "cardinality_nonmissing": int(series.nunique(dropna=True)),
            "representative_levels": [
                {"value": str(key), "count": int(value), "percent": 100 * int(value) / len(frame)}
                for key, value in counts.head(10).items()
            ],
            "rare_levels_lt_1pct": int((series.value_counts(normalize=True, dropna=True) < .01).sum()),
        }

    binary = len(classes) == 2
    for column in predictors.columns:
        series = predictors.loc[valid, column]
        if pd.api.types.is_numeric_dtype(series):
            series = series.replace([np.inf, -np.inf], np.nan)
        complete = series.notna()
        item = {"n_complete": int(complete.sum()), "missing_excluded": int((~complete).sum())}
        if len(classes) < 2 or complete.sum() < 3 or series[complete].nunique() < 2:
            item.update(method="not_estimable", effect=None)
        elif binary and pd.api.types.is_numeric_dtype(series):
            effect, p_value = pointbiserialr(label_codes[complete], series[complete])
            item.update({
                "method": "point_biserial",
                "effect": json_value(effect),
                "p_value_descriptive": json_value(p_value),
                "coded_positive_class": str(classes[1]),
            })
        elif pd.api.types.is_numeric_dtype(series):
            values, groups = series[complete], label_codes[complete]
            total = float(((values - values.mean()) ** 2).sum())
            between = sum(len(v) * (v.mean() - values.mean()) ** 2 for _, v in values.groupby(groups))
            item.update(method="eta_squared_descriptive", effect=float(between / total) if total else None)
        else:
            categories = series[complete].astype("string")
            item.update({
                "method": "bias_corrected_cramers_v" if categories.nunique() <= 200 else "skipped_high_cardinality",
                "effect": json_value(cramers_v(categories, label_codes[complete])) if categories.nunique() <= 200 else None,
            })
        result["univariate_associations"][str(column)] = item

    result["univariate_associations_ranked"] = sorted(
        (
            {"column": column, **details}
            for column, details in result["univariate_associations"].items()
            if details.get("effect") is not None
        ),
        key=lambda item: abs(item["effect"]),
        reverse=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect training data without reading a held-out test set.")
    parser.add_argument("--train", required=True, type=Path, help="Training CSV, TSV, XLS, or XLSX file")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument("--sheet", default="Data", help="Worksheet for Excel input (default: Data)")
    parser.add_argument("--output", type=Path, help="Optional JSON output path; stdout is always written")
    args = parser.parse_args()

    raw = args.train.read_bytes()
    training_hash = hashlib.sha256(raw).hexdigest()
    frame = load_table(args.train, args.sheet, args.target, raw=raw)
    result = {
        "training_source": {"path": str(args.train.resolve()), "sheet": args.sheet, "sha256": training_hash},
        "test_data_accessed": False,
        **diagnose(frame, args.target),
    }
    rendered = json.dumps(result, indent=2, default=json_value)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
