#!/usr/bin/env python3
"""Validate an explicit, bounded experiment before any model is fit."""
import argparse
import copy
import re
from sklearn.model_selection import ParameterGrid
from common import ESTIMATORS, METRICS, pipeline, read_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(plan):
    required = {"schema_version", "target", "task", "features", "numeric_features", "categorical_features",
                "excluded_features", "seed", "cv", "metrics", "models", "decision_trace", "sensitivities"}
    require(required <= plan.keys(), f"Missing plan fields: {sorted(required - plan.keys())}")
    allowed = required | {"positive_class", "threshold"}
    require(set(plan) <= allowed, f"Unknown plan fields: {set(plan) - allowed}")
    require(plan["schema_version"] == 2, "Only plan schema_version 2 is supported")
    require(plan["task"] in {"binary", "multiclass"}, "task must be binary or multiclass")
    require(isinstance(plan["target"], str) and bool(plan["target"]), "target must be a column name")
    for key in ("features", "numeric_features", "categorical_features"):
        require(isinstance(plan[key], list) and all(isinstance(x, str) for x in plan[key]), f"Invalid {key}")
        require(len(plan[key]) == len(set(plan[key])), f"Duplicate {key}")
    features = set(plan["features"])
    require(bool(features) and plan["target"] not in features, "Features must be nonempty and exclude target")
    require(set(plan["numeric_features"]) | set(plan["categorical_features"]) == features, "Feature groups must cover features")
    require(not (set(plan["numeric_features"]) & set(plan["categorical_features"])), "Feature groups overlap")
    require(isinstance(plan["excluded_features"], dict) and all(isinstance(v, str) and v.strip() for v in plan["excluded_features"].values()), "Excluded features need reasons")
    require(not (features & set(plan["excluded_features"])), "Excluded features cannot also be selected")
    require(plan["target"] not in plan["excluded_features"], "Target is not an excluded predictor")
    require(type(plan["seed"]) is int and 0 <= plan["seed"] < 2**32 - 100, "Invalid seed")
    cv = plan["cv"]
    require(set(cv) <= {"strategy", "outer_splits", "inner_splits", "group_column", "time_column", "gap"}, "Unknown CV fields")
    require(cv.get("strategy") in {"stratified", "stratified_group", "time"}, "Unsupported CV strategy")
    for k in ("outer_splits", "inner_splits"):
        require(type(cv.get(k)) is int and 2 <= cv[k] <= 10, f"{k} must be 2..10")
    if cv["strategy"] != "stratified":
        column = cv.get("group_column" if cv["strategy"] == "stratified_group" else "time_column")
        require(isinstance(column, str) and column in plan["excluded_features"] and column != plan["target"], "Group/time column must be explicitly excluded from features")
    require(type(cv.get("gap", 0)) is int and cv.get("gap", 0) >= 0, "gap must be a nonnegative count of distinct timestamps")
    metrics = plan["metrics"]
    require(set(metrics) == {"primary", "secondary"} and isinstance(metrics["secondary"], list), "metrics needs primary and secondary")
    requested = [metrics["primary"], *metrics["secondary"]]
    require(all(isinstance(m, str) and m in METRICS for m in requested), "Unsupported metric")
    require(len(requested) == len(set(requested)), "Metrics must be unique")
    if plan["task"] == "binary":
        require(isinstance(plan.get("positive_class"), str), "positive_class must be a string")
        require(type(plan.get("threshold")) in {int, float} and 0 <= plan["threshold"] <= 1, "Declare threshold in [0,1]")
        require("roc_auc_ovr_macro" not in requested, "Use roc_auc for binary tasks")
    else:
        require(not ({"positive_class", "threshold"} & plan.keys()), "Multiclass uses argmax; no positive_class/threshold")
        require(not ({"f1", "precision", "recall", "roc_auc", "average_precision"} & set(requested)), "Use multiclass-compatible metrics")
    require(isinstance(plan["decision_trace"], list) and bool(plan["decision_trace"]), "Provide contemporaneous decision evidence")
    for decision in plan["decision_trace"]:
        require(set(decision) == {"observation", "decision", "rationale", "human_review_point"}, "Decision trace requires all four fields")
        require(all(isinstance(v, str) and v.strip() for v in decision.values()), "Decision evidence cannot be empty")
    require(isinstance(plan["models"], list) and len(plan["models"]) == 2, "Compare exactly two models")
    names = []
    for spec in plan["models"]:
        require(set(spec) == {"name", "type", "params", "grid", "preprocessing"}, "Each model needs name/type/params/grid/preprocessing")
        require(isinstance(spec["name"], str) and re.fullmatch(r"[a-z][a-z0-9_]{0,39}", spec["name"]), "Unsafe model name")
        names.append(spec["name"])
        require(spec["type"] in ESTIMATORS, f"Unsupported estimator: {spec['type']}; extend and test explicitly")
        cfg = spec["preprocessing"]
        fields = {"numeric_imputer", "missing_indicator", "numeric_transform", "scaler", "categorical_imputer", "categorical_encoder"}
        require(fields <= set(cfg) <= fields | {"numeric_fill_value", "min_frequency", "max_categories"}, "Invalid preprocessing fields")
        require(cfg["numeric_imputer"] in {"median", "mean", "most_frequent", "constant"}, "Unsupported numeric imputation")
        require(type(cfg["missing_indicator"]) is bool, "missing_indicator must be boolean")
        require(cfg["numeric_transform"] in {"none", "log1p"}, "Unsupported numeric transform")
        require(cfg["scaler"] in {"none", "standard", "robust"}, "Unsupported scaler")
        require(cfg["categorical_imputer"] in {"most_frequent", "constant"}, "Unsupported categorical imputation")
        require(cfg["categorical_encoder"] in {"onehot", "ordinal"}, "Unsupported encoder")
        require(isinstance(spec["grid"], dict) and isinstance(spec["params"], dict), "params/grid must be objects")
        require(all(k.startswith("model__") for k in spec["grid"]), "Grid tunes estimator parameters; declare preprocessing sensitivity separately")
        require(not ({"random_state", "n_jobs"} & set(spec["params"])), "Seeds/jobs are controlled by the runner")
        require(not ({"model__random_state", "model__n_jobs"} & set(spec["grid"])), "Do not tune random seeds/jobs")
        candidates = list(ParameterGrid(spec["grid"]))
        require(len(candidates) <= 16, "Limited tuning allows at most 16 candidates per model")
        model = pipeline(plan, spec)
        for candidate in candidates:
            model.set_params(**candidate)
            for component in model.get_params(deep=True).values():
                if hasattr(component, "_validate_params"):
                    component._validate_params()
    require(len(names) == len(set(names)), "Model names must be unique")
    checks = plan["sensitivities"]
    require(isinstance(checks, list) and len(checks) <= 3, "At most three declared sensitivities")
    seen = set()
    for check in checks:
        require(set(check) == {"name", "rationale", "overrides"}, "Sensitivity needs name/rationale/overrides")
        require(re.fullmatch(r"[a-z][a-z0-9_]{0,39}", check["name"]) and check["name"] not in seen, "Invalid/duplicate sensitivity name")
        seen.add(check["name"])
        require(bool(check["rationale"].strip()), "Sensitivity needs a rationale")
        require(set(check["overrides"]) <= {"features", "numeric_features", "categorical_features", "excluded_features", "models", "threshold"}, "Sensitivity cannot change target, metrics, labels or CV")
        variant = copy.deepcopy(plan)
        variant.update(check["overrides"])
        variant["sensitivities"] = []
        validate(variant)
        require([s["name"] for s in variant["models"]] == names, "Sensitivity must retain model names and order")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    validate(read_json(parser.parse_args().plan))
    print("Plan v2 is valid")
