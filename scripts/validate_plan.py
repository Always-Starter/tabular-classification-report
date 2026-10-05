#!/usr/bin/env python3
"""Validate an explicit, bounded experiment before any model is fit."""
import argparse
import copy
import re
from sklearn.model_selection import ParameterGrid
from common import ESTIMATORS, MATERIAL_FIXED_PARAMS, METRICS, pipeline, read_json

DECISION_BASES = {"invariant_methodological_rule", "dataset_specific_evidence",
                  "user_domain_constraint", "conventional_default", "unknown"}
THRESHOLD_METRICS = {"accuracy", "balanced_accuracy", "f1", "precision", "recall",
                     "f1_macro", "f1_weighted", "precision_macro", "recall_macro"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(plan):
    require(plan.get("schema_version") in {2, 3, 4, 5, 6},
            "Only plan schema_version 2, 3, 4, 5 or 6 is supported")
    schema = plan["schema_version"]
    required = {"schema_version", "target", "task", "features", "numeric_features", "categorical_features",
                "excluded_features", "seed", "cv", "metrics", "models", "decision_trace", "sensitivities"}
    if schema >= 3:
        required |= {"semantics", "feature_provenance", "candidate_selection", "model_count_rationale"}
    if schema >= 5:
        required.add("compute_budget")
    if schema >= 6:
        required.add("model_selection_policy")
    require(required <= plan.keys(), f"Missing plan fields: {sorted(required - plan.keys())}")
    allowed = required | {"positive_class", "threshold", "threshold_policy", "model_count_rationale"}
    require(set(plan) <= allowed, f"Unknown plan fields: {set(plan) - allowed}")
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
    if schema >= 3:
        semantics = plan["semantics"]
        semantic_fields = {"target_source", "target_meaning", "positive_class_meaning",
                           "false_positive_cost", "false_negative_cost", "row_dependence"}
        require(set(semantics) == semantic_fields, "Schema v3 semantics fields are incomplete")
        require(semantics["target_source"] in {"user", "authoritative_metadata", "provisionally_inferred",
                                                "ai_inferred"},
                "Invalid target_source")
        for key in ("target_meaning", "positive_class_meaning"):
            require(semantics[key] is None or isinstance(semantics[key], str) and semantics[key].strip(),
                    f"{key} must be a nonempty string or null")
        for key in ("false_positive_cost", "false_negative_cost"):
            require(semantics[key] is None or type(semantics[key]) in {int, float} and semantics[key] >= 0,
                    f"{key} must be a nonnegative number or null")
        require(semantics["row_dependence"] in {"independent", "grouped", "temporal", "unknown"},
                "Invalid row_dependence")
        provenance = plan["feature_provenance"]
        require(isinstance(provenance, dict), "feature_provenance must be an object")
        for name, record in provenance.items():
            require(name in features or name in plan["excluded_features"], "Provenance feature is not declared")
            require(set(record) == {"derivation", "uses_outcome_information", "prediction_time_available", "status"},
                    "Feature provenance fields are incomplete")
            require(record["derivation"] is None or isinstance(record["derivation"], str) and record["derivation"].strip(),
                    "Feature derivation must be a nonempty string or null")
            require(record["uses_outcome_information"] in {True, False, None}, "Invalid outcome-information status")
            require(record["prediction_time_available"] in {True, False, None}, "Invalid prediction-time status")
            require(record["status"] in {"confirmed", "unknown"}, "Invalid provenance status")
        selection = plan["candidate_selection"]
        require(set(selection) == {"registry_snapshot", "basis", "rationale", "considered_alternatives"},
                "candidate_selection fields are incomplete")
        require(selection["registry_snapshot"] == sorted(ESTIMATORS),
                "registry_snapshot must exactly record the available estimator families")
        require(isinstance(selection["basis"], list) and bool(selection["basis"])
                and set(selection["basis"]) <= {"training_diagnosis", "user", "rule_based_registry"},
                "Invalid candidate-selection basis")
        require(isinstance(selection["rationale"], str) and selection["rationale"].strip(),
                "Candidate selection needs a rationale")
        require(isinstance(selection["considered_alternatives"], dict)
                and all(k in ESTIMATORS and isinstance(v, str) and v.strip()
                        for k, v in selection["considered_alternatives"].items()),
                "Considered alternatives must map registry families to reasons")
    require(type(plan["seed"]) is int and 0 <= plan["seed"] < 2**32 - 100, "Invalid seed")
    cv = plan["cv"]
    cv_allowed = {"strategy", "outer_splits", "inner_splits", "group_column", "time_column", "gap"}
    if schema >= 5:
        cv_allowed |= {"rationale", "status", "selection_basis"}
        require({"rationale", "status", "selection_basis"} <= set(cv), "Schema v5+ CV needs rationale, status and selection_basis")
        require(isinstance(cv["rationale"], str) and cv["rationale"].strip(), "CV rationale is required")
        require(cv["status"] in {"confirmed", "provisional_unknown_dependence"}, "Invalid CV status")
        require(cv["selection_basis"] in DECISION_BASES, "Invalid CV selection_basis")
    require(set(cv) <= cv_allowed, "Unknown CV fields")
    require(cv.get("strategy") in {"stratified", "stratified_group", "time"}, "Unsupported CV strategy")
    for k in ("outer_splits", "inner_splits"):
        require(type(cv.get(k)) is int and 2 <= cv[k] <= 10, f"{k} must be 2..10")
    if cv["strategy"] != "stratified":
        column = cv.get("group_column" if cv["strategy"] == "stratified_group" else "time_column")
        require(isinstance(column, str) and column in plan["excluded_features"] and column != plan["target"], "Group/time column must be explicitly excluded from features")
    require(type(cv.get("gap", 0)) is int and cv.get("gap", 0) >= 0, "gap must be a nonnegative count of distinct timestamps")
    metrics = plan["metrics"]
    metric_fields = {"primary", "secondary"} if schema == 2 else {"primary", "secondary", "rationale", "status"}
    require(set(metrics) == metric_fields and isinstance(metrics["secondary"], list), "Invalid metrics fields")
    if schema >= 3:
        require(isinstance(metrics["rationale"], str) and metrics["rationale"].strip(), "Metric needs a rationale")
        require(metrics["status"] in {"confirmed", "provisional_unknown_semantics", "provisional_unknown_costs"},
                "Invalid metric status")
    requested = [metrics["primary"], *metrics["secondary"]]
    require(all(isinstance(m, str) and m in METRICS for m in requested), "Unsupported metric")
    require(len(requested) == len(set(requested)), "Metrics must be unique")
    if plan["task"] == "binary":
        require(isinstance(plan.get("positive_class"), str), "positive_class must be a string")
        if schema >= 5:
            require("threshold" not in plan, "Schema v5+ uses threshold_policy, not a scalar threshold")
            policy = plan.get("threshold_policy")
            fields = {"mode", "value", "search_values", "objective", "procedure", "value_source",
                      "value_rationale", "not_tuned_reason", "selection_scope"}
            require(isinstance(policy, dict) and set(policy) == fields, "Invalid threshold_policy fields")
            require(policy["mode"] in {"fixed", "tuned", "model_default"}, "Invalid threshold mode")
            require(isinstance(policy["search_values"], list), "Threshold search_values must be a list")
            if policy["mode"] == "fixed":
                require(type(policy["value"]) in {int, float} and 0 <= policy["value"] <= 1,
                        "Fixed threshold value must be in [0,1]")
                require(policy["search_values"] == [] and policy["objective"] is None and policy["procedure"] is None,
                        "A fixed threshold cannot declare a tuning search")
                require(policy["selection_scope"] == "predeclared_before_development", "Fixed threshold timing must be predeclared")
                require(isinstance(policy["not_tuned_reason"], str) and policy["not_tuned_reason"].strip(),
                        "Explain why the fixed threshold was not tuned")
                require(policy["value_source"] in {"training_diagnosis", "user_supplied", "authoritative_requirement",
                                                    "prior_independent_evidence", "conventional_default", "unknown"},
                        "Invalid fixed-threshold value source")
                if policy["value_source"] == "unknown":
                    require(policy["value_rationale"] is None, "Unknown threshold provenance requires null rationale")
                else:
                    require(isinstance(policy["value_rationale"], str) and policy["value_rationale"].strip(),
                            "Explain why this exact fixed threshold was chosen")
            elif policy["mode"] == "tuned":
                values = policy["search_values"]
                require(policy["value"] is None and 2 <= len(values) <= 21 and len(set(values)) == len(values)
                        and all(type(v) in {int, float} and 0 <= v <= 1 for v in values),
                        "Tuned threshold search_values need 2..21 distinct values in [0,1]")
                require(policy["objective"] == metrics["primary"] and policy["objective"] in THRESHOLD_METRICS,
                        "This implementation jointly tunes threshold and model parameters on a threshold-dependent primary metric")
                require(policy["procedure"] == "joint_inner_cv_grid" and policy["selection_scope"] == "inner_cv_only",
                        "Tuned thresholds must use the declared inner-CV-only procedure")
                require(policy["value_source"] == "inner_cv" and isinstance(policy["value_rationale"], str)
                        and policy["value_rationale"].strip() and policy["not_tuned_reason"] is None,
                        "Tuned threshold provenance must identify inner CV")
            else:
                require(policy["value"] is None and policy["search_values"] == [] and policy["objective"] is None,
                        "Model-default behavior has no numeric threshold search")
                require(policy["procedure"] == "estimator_default_class_decision"
                        and policy["value_source"] == "model_default"
                        and policy["selection_scope"] == "predeclared_before_development",
                        "Invalid model-default decision policy")
                require(isinstance(policy["value_rationale"], str) and policy["value_rationale"].strip()
                        and isinstance(policy["not_tuned_reason"], str) and policy["not_tuned_reason"].strip(),
                        "Model-default policy needs rationale and a reason it was not tuned")
        else:
            require(type(plan.get("threshold")) in {int, float} and 0 <= plan["threshold"] <= 1, "Declare threshold in [0,1]")
        require("roc_auc_ovr_macro" not in requested, "Use roc_auc for binary tasks")
    else:
        require(not ({"positive_class", "threshold", "threshold_policy"} & plan.keys()), "Multiclass uses argmax; no positive_class/threshold policy")
        require(not ({"f1", "precision", "recall", "roc_auc", "average_precision"} & set(requested)), "Use multiclass-compatible metrics")
    require(isinstance(plan["decision_trace"], list) and bool(plan["decision_trace"]), "Provide contemporaneous decision evidence")
    for decision in plan["decision_trace"]:
        fields = {"observation", "decision", "rationale", "human_review_point"}
        if schema >= 5:
            fields |= {"topic", "basis"}
        require(set(decision) == fields, "Decision trace fields are incomplete")
        require(all(isinstance(decision[k], str) and decision[k].strip() for k in fields), "Decision evidence cannot be empty")
        if schema >= 5:
            require(decision["basis"] in DECISION_BASES, "Invalid decision basis")
    if schema >= 5:
        topics = {item["topic"] for item in plan["decision_trace"]}
        needed = {"target_semantics", "feature_typing", "preprocessing", "imbalance", "model_shortlist",
                  "metric", "cv", "decision_rule", "compute_budget"}
        require(needed <= topics, f"Schema v5+ decision trace is missing topics: {sorted(needed - topics)}")
        budget = plan["compute_budget"]
        require(isinstance(budget, dict) and set(budget) == {"max_explicit_fits", "source", "rationale"},
                "Invalid compute_budget")
        require(type(budget["max_explicit_fits"]) is int and budget["max_explicit_fits"] > 0,
                "max_explicit_fits must be positive")
        require(budget["source"] in {"user_supplied", "environment_constraint", "conventional_default"}
                and isinstance(budget["rationale"], str) and budget["rationale"].strip(),
                "Compute budget needs a source and rationale")
    require(isinstance(plan["models"], list) and 2 <= len(plan["models"]) <= 3,
            "Compare two or three models; a third needs a documented reason")
    if len(plan["models"]) == 3 or schema >= 3:
        require(isinstance(plan.get("model_count_rationale"), str) and bool(plan["model_count_rationale"].strip()),
                "Model count needs a nonempty rationale grounded in training evidence")
    elif "model_count_rationale" in plan:
        require(isinstance(plan["model_count_rationale"], str) and bool(plan["model_count_rationale"].strip()),
                "model_count_rationale must be nonempty when supplied")
    names = []
    for spec in plan["models"]:
        model_fields = {"name", "type", "params", "grid", "preprocessing"}
        if schema >= 3:
            model_fields |= {"rationale", "grid_rationale", "stopping_rule"}
        if schema >= 4:
            model_fields.add("fixed_param_rationale")
        if schema >= 5:
            model_fields |= {"preprocessing_rationale", "imbalance_handling"}
        require(set(spec) == model_fields, "Invalid per-model fields")
        if schema >= 3:
            for key in ("rationale", "grid_rationale", "stopping_rule"):
                require(isinstance(spec[key], str) and spec[key].strip(), f"Model {key} must be nonempty")
        if schema >= 5:
            require(isinstance(spec["preprocessing_rationale"], str) and spec["preprocessing_rationale"].strip(),
                    "Each model needs a preprocessing rationale")
            imbalance = spec["imbalance_handling"]
            require(isinstance(imbalance, dict) and set(imbalance) == {"strategy", "basis", "rationale"},
                    "Invalid imbalance_handling")
            require(imbalance["strategy"] in {"none", "fixed_class_weight", "tuned_class_weight"}
                    and imbalance["basis"] in DECISION_BASES
                    and isinstance(imbalance["rationale"], str) and imbalance["rationale"].strip(),
                    "Imbalance handling needs strategy, basis and rationale")
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
        require(not ({f"model__{key}" for key in spec["params"]} & set(spec["grid"])),
                "A parameter cannot be both fixed in params and tuned in grid")
        if schema >= 5:
            has_fixed_weight = "class_weight" in spec["params"]
            has_tuned_weight = "model__class_weight" in spec["grid"]
            expected = {"none": (False, False), "fixed_class_weight": (True, False),
                        "tuned_class_weight": (False, True)}[spec["imbalance_handling"]["strategy"]]
            require((has_fixed_weight, has_tuned_weight) == expected,
                    "imbalance_handling must agree with fixed/tuned class_weight")
        if schema >= 4:
            fixed = spec["fixed_param_rationale"]
            require(isinstance(fixed, dict) and set(fixed) == set(spec["params"]),
                    "fixed_param_rationale must cover every fixed params key exactly")
            sources = {"training_diagnosis", "compute_budget", "convergence_requirement", "user_supplied",
                       "authoritative_requirement", "prior_independent_evidence", "implementation_constraint",
                       "predeclared_rule", "library_default", "literature", "domain_constraint", "heuristic",
                       "unknown"}
            for parameter, record in fixed.items():
                require(isinstance(record, dict)
                        and set(record) == {"not_tuned_reason", "value_source", "value_rationale"},
                        f"Fixed parameter {parameter} needs not_tuned_reason, value_source and value_rationale")
                require(isinstance(record["not_tuned_reason"], str) and record["not_tuned_reason"].strip(),
                        f"Fixed parameter {parameter} needs a reason it was not tuned")
                require(record["value_source"] in sources, f"Fixed parameter {parameter} has invalid value_source")
                if record["value_source"] == "unknown":
                    require(record["value_rationale"] is None,
                            f"Fixed parameter {parameter} with unknown value_source must use null value_rationale")
                else:
                    require(isinstance(record["value_rationale"], str) and record["value_rationale"].strip(),
                            f"Fixed parameter {parameter} needs a rationale for the exact fixed value")
        require(all(k.startswith("model__") for k in spec["grid"]), "Grid tunes estimator parameters; declare preprocessing sensitivity separately")
        require(not ({"random_state", "n_jobs"} & set(spec["params"])), "Seeds/jobs are controlled by the runner")
        require(not ({"model__random_state", "model__n_jobs"} & set(spec["grid"])), "Do not tune random seeds/jobs")
        if schema >= 6:
            for parameter in MATERIAL_FIXED_PARAMS.get(spec["type"], set()):
                require(parameter in spec["params"] or f"model__{parameter}" in spec["grid"],
                        f"Schema v6 requires {spec['type']} to declare material parameter {parameter} "
                        "as fixed with provenance or tuned in the bounded grid")
        if spec["type"] == "support_vector_classifier":
            require(cfg["scaler"] != "none", "SVM requires declared numeric scaling")
            require(plan["cv"]["strategy"] == "stratified",
                    "Calibrated SVM currently supports only stratified CV; extend calibration splits before grouped/time use")
            require("probability" not in spec["params"] and "model__probability" not in spec["grid"],
                    "SVM probability calibration is controlled by the runner")
            if schema >= 6:
                require("cache_size" not in spec["params"] and "model__cache_size" not in spec["grid"],
                        "SVM cache_size is a recorded runner execution control in schema v6")
            calibration_values = spec["grid"].get(
                "model__calibration_cv", [spec["params"].get("calibration_cv", 3)])
            require(all(type(value) is int and 2 <= value <= 10 for value in calibration_values),
                    "SVM calibration_cv must use integer values from 2 to 10")
        candidates = list(ParameterGrid(spec["grid"]))
        require(len(candidates) <= 16, "Limited tuning allows at most 16 candidates per model")
        if schema >= 5 and plan["task"] == "binary" and plan["threshold_policy"]["mode"] == "tuned":
            require(len(candidates) * len(plan["threshold_policy"]["search_values"]) <= 64,
                    "Joint model/threshold search allows at most 64 scored combinations per model")
        model = pipeline(plan, spec)
        for candidate in candidates:
            model.set_params(**candidate)
            for component in model.get_params(deep=True).values():
                if (component.__class__.__module__.startswith("sklearn.")
                        and hasattr(component, "_validate_params")):
                    component._validate_params()
    require(len(names) == len(set(names)), "Model names must be unique")
    if schema >= 6:
        policy = plan["model_selection_policy"]
        fields = {"comparison_source", "metric", "practical_tie_tolerance", "tolerance_source",
                  "tolerance_rationale", "tie_breakers", "preference_order", "final_refit_procedure"}
        require(isinstance(policy, dict) and set(policy) == fields,
                "Schema v6 model_selection_policy fields are incomplete")
        require(policy["comparison_source"] == "shared_outer_cv"
                and policy["metric"] == "primary"
                and policy["tie_breakers"] == ["lower_outer_std", "declared_preference_order"]
                and policy["final_refit_procedure"] == "final_inner_cv_then_full_training",
                "Invalid executable model-selection procedure")
        require(type(policy["practical_tie_tolerance"]) in {int, float}
                and policy["practical_tie_tolerance"] >= 0,
                "practical_tie_tolerance must be a nonnegative number")
        require(policy["tolerance_source"] in {"user_supplied", "authoritative_requirement",
                                               "prior_independent_evidence", "domain_constraint",
                                               "conventional_default", "unknown"},
                "Invalid practical-tie tolerance source")
        if policy["tolerance_source"] == "unknown":
            require(policy["tolerance_rationale"] is None,
                    "Unknown practical-tie tolerance provenance requires null rationale")
        else:
            require(isinstance(policy["tolerance_rationale"], str)
                    and policy["tolerance_rationale"].strip(),
                    "Explain the pre-fit basis for the practical-tie tolerance")
        require(policy["preference_order"] == names,
                "preference_order must list every model exactly once in plan order")
    checks = plan["sensitivities"]
    require(isinstance(checks, list) and len(checks) <= 3, "At most three declared sensitivities")
    seen = set()
    for check in checks:
        require(set(check) == {"name", "rationale", "overrides"}, "Sensitivity needs name/rationale/overrides")
        require(re.fullmatch(r"[a-z][a-z0-9_]{0,39}", check["name"]) and check["name"] not in seen, "Invalid/duplicate sensitivity name")
        seen.add(check["name"])
        require(bool(check["rationale"].strip()), "Sensitivity needs a rationale")
        sensitivity_fields = {"features", "numeric_features", "categorical_features", "excluded_features", "models"}
        sensitivity_fields.add("threshold_policy" if schema >= 5 else "threshold")
        require(set(check["overrides"]) <= sensitivity_fields, "Sensitivity cannot change target, metrics, labels or CV")
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
    print("Plan is valid")
