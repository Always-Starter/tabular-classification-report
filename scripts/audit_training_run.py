#!/usr/bin/env python3
"""Audit saved training-only development evidence without reading held-out data."""
import argparse
from pathlib import Path

from common import effective_fixed_params, execution_controls, read_json, sha, write_json
from verify_results import verify_training


def audit(results_path, lock_path=None):
    results_path = Path(results_path)
    results = read_json(results_path)
    verification = verify_training(results_path)
    lock = read_json(lock_path) if lock_path else None
    if lock:
        if lock.get("training_results_sha256") != sha(results_path):
            raise ValueError("Model Lock refers to different training results")
        variant_name = lock["review"]["selected_variant"]
        preferred_model = lock["review"]["preferred_model"]
        selection_review = lock["review"]
    else:
        variant_name, preferred_model, selection_review = "baseline", None, None
    variant = results["variants"][variant_name]
    if lock and variant_name != "baseline":
        raise ValueError("A Model Lock cannot select an interpretive sensitivity variant")
    plan = variant["plan"]
    models = {}
    for name, model in variant["models"].items():
        spec = next(spec for spec in plan["models"] if spec["name"] == name)
        effective = model.get("effective_fixed_params", effective_fixed_params(plan, spec))
        controls = model.get("execution_controls", execution_controls(plan, spec))
        models[name] = {
            "type": model["type"],
            "preprocessing_rationale": spec.get("preprocessing_rationale"),
            "imbalance_handling": spec.get("imbalance_handling"),
            "predefined_grid": spec["grid"],
            "fixed_params": spec["params"],
            "fixed_param_rationale": spec.get("fixed_param_rationale"),
            "effective_fixed_params": effective,
            "execution_controls": controls,
            "outer_folds": [{"fold": fold["fold"], "selected_params": fold["best_params"],
                             "selected_threshold": fold.get("selected_threshold"),
                             "inner_search": fold.get("inner_search"), "outer_metrics": fold["metrics"],
                             "tuning_boundary": fold["tuning_boundary"]}
                            for fold in model["fold_results"]],
            "outer_summary": model["outer_summary"],
            "final_selected_params": model["best_params"],
            "final_selected_threshold": model.get("selected_threshold"),
            "final_inner_search": model.get("final_inner_search"),
            "final_refit": model.get("final_refit"),
            "tuning_boundary": model["tuning_boundary"],
        }
    unresolved = []
    if selection_review is None:
        unresolved.append("final_model_selection_review")
    semantics = plan.get("semantics")
    if semantics is None:
        unresolved.append("Structured target semantics and FP/FN costs were not recorded in schema v2")
    else:
        for field in ("target_meaning", "positive_class_meaning", "false_positive_cost", "false_negative_cost"):
            if semantics[field] is None:
                unresolved.append(field)
    provenance = plan.get("feature_provenance", {})
    for feature, record in provenance.items():
        if record["status"] == "unknown":
            unresolved.append(f"feature_provenance:{feature}")
    policy = plan.get("threshold_policy")
    if plan["task"] == "binary" and (policy is None or policy.get("value_source") == "unknown"):
        unresolved.append("threshold_value_source")
    if any(fold["inner_search"] is None for model in models.values() for fold in model["outer_folds"]):
        unresolved.append("Per-outer-fold inner-CV candidate scores were not preserved")
    for name, model in models.items():
        if model["fixed_params"] and model["fixed_param_rationale"] is None:
            unresolved.extend(f"fixed_param_rationale:{name}:{parameter}"
                              for parameter in model["fixed_params"])
        elif model["fixed_param_rationale"]:
            unresolved.extend(f"fixed_param_value_source:{name}:{parameter}"
                              for parameter, record in model["fixed_param_rationale"].items()
                              if record["value_source"] == "unknown")
        unresolved.extend(f"legacy_runner_fixed_param_provenance:{name}:{parameter}"
                          for parameter in model["effective_fixed_params"]
                          if parameter not in (model["fixed_param_rationale"] or {}))
    return {
        "audit_scope": "training_only",
        "held_out_data_accessed_by_audit": False,
        "training_verification": verification,
        "training_source": results["training_source"],
        "variant": variant_name,
        "preferred_model": preferred_model,
        "selection_review": selection_review,
        "model_selection": results.get("model_selection"),
        "semantics": semantics,
        "feature_provenance": provenance,
        "candidate_selection": plan.get("candidate_selection"),
        "metric": plan["metrics"],
        "threshold_policy": policy or {"mode": "legacy_fixed", "value": plan.get("threshold"),
                                        "value_source": "unknown"},
        "cv_policy": plan.get("cv"),
        "compute_budget": plan.get("compute_budget"),
        "decision_trace": plan.get("decision_trace"),
        "sensitivity_evidence": {
            name: {"role": details.get("role", "legacy_unspecified"),
                   "selection_eligible": details.get("selection_eligible")}
            for name, details in results["variants"].items() if name != "baseline"
        },
        "sensitivity_policy": ("Interpretive only; adopting a sensitivity requires a new independent "
                               "baseline plan/run."),
        "models": models,
        "unresolved": unresolved,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-results", required=True, type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.training_results, args.lock)
    if args.output:
        write_json(args.output, result)
    print(result)
