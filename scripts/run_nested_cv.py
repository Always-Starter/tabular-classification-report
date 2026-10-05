#!/usr/bin/env python3
"""Train-only nested CV, declared sensitivity experiments, and final train refits."""
import argparse
import copy
import hashlib
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, ParameterGrid

from common import (PrimaryScorer, code_hashes, decisions_from_probabilities, effective_fixed_params,
                    environment, execution_controls, features, load_table, model_selection_evidence,
                    ordered_probabilities, pipeline, predictions, read_json, score_metrics, sha, splits,
                    threshold_policy, utcnow, write_json)
from validate_plan import validate


def svm_calibration_feasibility(y, plan, variants, outer, inner, final_inner):
    """Fail before fitting if any nested training subset is too small for SVM calibration."""
    subsets = [("full-training refit", np.arange(len(y)))]
    for outer_index, (outer_train, _) in enumerate(outer, 1):
        subsets.append((f"outer fold {outer_index} refit", outer_train))
        for inner_index, (inner_train, _) in enumerate(inner[outer_index - 1], 1):
            subsets.append((f"outer fold {outer_index}, inner fold {inner_index} fit",
                            outer_train[inner_train]))
    for inner_index, (inner_train, _) in enumerate(final_inner, 1):
        subsets.append((f"final inner fold {inner_index} fit", inner_train))
    for variant_name, variant in variants.items():
        for spec in variant["models"]:
            if spec["type"] != "support_vector_classifier":
                continue
            values = spec["grid"].get("model__calibration_cv",
                                      [spec["params"].get("calibration_cv", 3)])
            required = max(values)
            for subset_name, indices in subsets:
                counts = {label: int(np.sum(y[indices] == label)) for label in sorted(set(y))}
                if min(counts.values()) < required:
                    raise ValueError(
                        f"SVM calibration is infeasible before fitting: {variant_name}/{spec['name']} "
                        f"uses calibration_cv up to {required}, but {subset_name} has class counts {counts}. "
                        "Reduce nested folds/calibration_cv, obtain more training examples, or choose another "
                        "justified candidate in a new plan.")


def numeric_grid_boundaries(grid, selected):
    """Describe numeric endpoint selections without treating every endpoint as directional evidence."""
    edges = {}
    for parameter, values in grid.items():
        if len(values) < 2 or any(isinstance(v, (bool, np.bool_)) or
                                   not isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            continue
        ordered = sorted(set(values))
        if len(ordered) < 2:
            continue
        choice = selected[parameter]
        edge = "lower" if choice == ordered[0] else "upper" if choice == ordered[-1] else None
        if edge:
            interior = len(ordered) > 2
            edges[parameter] = {
                "selected": choice,
                "edge": edge,
                "evaluated_min": ordered[0],
                "evaluated_max": ordered[-1],
                "distinct_values_evaluated": len(ordered),
                "interior_candidates_evaluated": interior,
                "boundary_type": "edge_with_interior_candidates" if interior else "two_value_grid_endpoint",
                "supports_outside_range_question": interior,
                "interpretation": (
                    "Interior numeric candidates were evaluated; this edge selection supports an untested-direction "
                    "question for a future independent run, not adaptive expansion of the current run."
                    if interior else
                    "Every choice in a two-value numeric grid is an endpoint; this records coarse search coverage "
                    "and is not directional evidence that the optimum lies outside the evaluated range."
                ),
            }
    return edges


def search_evidence(search, metric):
    """Return JSON-safe candidate-level inner-CV evidence for an audit trail."""
    results = search.cv_results_
    split_columns = sorted((key for key in results if re.fullmatch(r"split\d+_test_score", key)),
                           key=lambda key: int(key[5:key.index("_")]))
    direction = -1 if metric == "log_loss" else 1
    candidates = []
    for index, params in enumerate(results["params"]):
        candidates.append({
            "params": params,
            "mean_score": float(direction * results["mean_test_score"][index]),
            "std_score": float(results["std_test_score"][index]),
            "rank": int(results["rank_test_score"][index]),
            "split_scores": [float(direction * results[key][index]) for key in split_columns],
        })
    return {"metric": metric, "inner_splits": len(split_columns), "best_index": int(search.best_index_),
            "best_score": float(direction * search.best_score_), "selected_threshold": None,
            "threshold_tuned": False, "candidates": candidates}


def tuned_threshold_search(x, y, plan, spec, cv, order):
    """Jointly select model parameters and a threshold using inner folds only."""
    policy = threshold_policy(plan)
    metric = policy["objective"]
    candidates = []
    for params in ParameterGrid(spec["grid"]):
        by_threshold = {value: [] for value in policy["search_values"]}
        for train_index, valid_index in cv:
            estimator = pipeline(plan, spec).set_params(**params)
            estimator.fit(x.iloc[train_index], y[train_index])
            prob = ordered_probabilities(estimator, x.iloc[valid_index], order)
            for value in policy["search_values"]:
                pred = decisions_from_probabilities(prob, order, plan, value)
                score = score_metrics(y[valid_index], pred, prob, order, plan)["metrics"][metric]
                if score is None:
                    raise ValueError(f"Threshold objective {metric} is undefined in an inner validation fold")
                by_threshold[value].append(score)
        for value, scores in by_threshold.items():
            candidates.append({"params": params, "threshold": value, "mean_score": float(np.mean(scores)),
                               "std_score": float(np.std(scores, ddof=1)), "split_scores": scores})
    order_by_score = sorted(range(len(candidates)), key=lambda i: (-candidates[i]["mean_score"], i))
    for rank, index in enumerate(order_by_score, 1):
        candidates[index]["rank"] = rank
    best_index = order_by_score[0]
    winner = candidates[best_index]
    estimator = pipeline(plan, spec).set_params(**winner["params"]).fit(x, y)
    evidence = {"metric": metric, "inner_splits": len(cv), "best_index": best_index,
                "best_score": winner["mean_score"], "selected_threshold": winner["threshold"],
                "threshold_tuned": True, "procedure": policy["procedure"], "candidates": candidates}
    return estimator, winner["params"], winner["threshold"], evidence


def fit_search(x, y, plan, spec, cv, order):
    policy = threshold_policy(plan)
    if policy and policy["mode"] == "tuned":
        return tuned_threshold_search(x, y, plan, spec, cv, order)
    scorer = PrimaryScorer(plan, order)
    search = GridSearchCV(pipeline(plan, spec), spec["grid"], scoring=scorer,
                          cv=cv, n_jobs=1, error_score="raise")
    search.fit(x, y)
    selected = policy["value"] if policy and policy["mode"] == "fixed" else None
    evidence = search_evidence(search, plan["metrics"]["primary"])
    evidence["selected_threshold"] = selected
    return search.best_estimator_, search.best_params_, selected, evidence


def run(train, plan_path, output_dir, sheet="Data", max_fits=None):
    plan = validate(read_json(plan_path))
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Training output directory must be empty; retain prior experiments")
    raw = Path(train).read_bytes()
    training_hash = hashlib.sha256(raw).hexdigest()
    frame = load_table(train, sheet, plan["target"], raw=raw)
    del raw
    if plan["target"] not in frame:
        raise ValueError("Training target is missing")
    valid = frame[plan["target"]].notna()
    source_rows = np.flatnonzero(valid)
    d = frame.loc[valid].reset_index(drop=True)
    y = d[plan["target"]].astype(str).to_numpy()
    order = sorted(set(y))
    if (plan["task"] == "binary" and (len(order) != 2 or plan["positive_class"] not in order)
            or plan["task"] == "multiclass" and len(order) < 3):
        raise ValueError("Task/positive class does not match training labels")
    variants = {"baseline": plan}
    for check in plan["sensitivities"]:
        if check["name"] == "baseline":
            raise ValueError("Sensitivity cannot be called baseline")
        variant = copy.deepcopy(plan)
        variant.update(check["overrides"])
        variant["sensitivities"] = []
        variants[check["name"]] = validate(variant)
    cv = plan["cv"]
    outer = splits(d, y, plan, cv["outer_splits"], plan["seed"])
    inner = [splits(d.iloc[a].reset_index(drop=True), y[a], plan, cv["inner_splits"], plan["seed"] + i + 1)
             for i, (a, _) in enumerate(outer)]
    final_inner = splits(d, y, plan, cv["inner_splits"], plan["seed"] + 99)
    svm_calibration_feasibility(y, plan, variants, outer, inner, final_inner)
    fit_count = sum((cv["outer_splits"] + 1) * (len(list(ParameterGrid(s["grid"]))) * cv["inner_splits"] + 1)
                    for p in variants.values() for s in p["models"])
    plan_budget = plan.get("compute_budget", {}).get("max_explicit_fits", 1200)
    fit_budget = min(plan_budget, max_fits) if max_fits is not None else plan_budget
    if fit_count > fit_budget:
        raise ValueError(f"Planned {fit_count} fits exceed budget {fit_budget}; reduce grid/folds/sensitivities")
    # The bundled estimators use dense output. Stop before an accidental large expansion.
    for p in variants.values():
        actual_predictors = set(frame.columns) - {p["target"]}
        declared_predictors = set(p["features"]) | set(p["excluded_features"])
        if actual_predictors != declared_predictors:
            raise ValueError(f"Every input predictor needs a selection/exclusion decision; "
                             f"unaccounted={sorted(actual_predictors - declared_predictors)}, "
                             f"absent={sorted(declared_predictors - actual_predictors)}")
        x = features(d, p)
        for spec in p["models"]:
            cfg = spec["preprocessing"]
            width = len(p["numeric_features"]) * (2 if cfg["missing_indicator"] else 1)
            width += sum(min(x[c].nunique(dropna=False), cfg.get("max_categories") or len(x))
                         if cfg["categorical_encoder"] == "onehot" else 1 for c in p["categorical_features"])
            if len(x) * max(width, 1) * 8 > 512 * 1024**2:
                raise ValueError("Estimated dense encoding exceeds 512 MiB; review high-cardinality encoding or extend sparse support")
            if spec["type"] == "support_vector_classifier":
                kernels = set(spec["grid"].get("model__kernel", [spec["params"].get("kernel", "rbf")]))
                if kernels - {"linear"} and len(x) ** 2 * 8 > 2 * 1024**3:
                    raise ValueError("Nonlinear SVM candidate exceeds the conservative 2 GiB pairwise-kernel guard; "
                                     "use a justified linear kernel, smaller training set, or an explicitly tested scalable method")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": plan["schema_version"], "created_at": utcnow(), "test_data_accessed": False,
              "training_source": {"path": str(Path(train).resolve()), "sha256": training_hash, "sheet": sheet},
              "input_rows": len(frame), "eligible_rows": len(d), "missing_targets": int((~valid).sum()),
              "class_order": order, "class_counts": {c: int(sum(y == c)) for c in order},
              "environment": environment(), "code_sha256": code_hashes(), "planned_fits": fit_count,
              "fit_budget": fit_budget,
              "outer_splits": [{"train_source_rows": source_rows[a].tolist(), "valid_source_rows": source_rows[b].tolist()}
                               for a, b in outer], "variants": {}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("error", category=ConvergenceWarning)
        for variant_name, p in variants.items():
            x = features(d, p)
            variant_dir = output_dir / variant_name
            variant_dir.mkdir()
            vr = {"plan": p, "models": {},
                  "role": "selection_eligible" if variant_name == "baseline" else "interpretive_sensitivity",
                  "selection_eligible": variant_name == "baseline"}
            result["variants"][variant_name] = vr
            for spec in p["models"]:
                print(f"Running {variant_name}/{spec['name']} ({fit_count} total planned fits)", flush=True)
                folds, records = [], []
                for i, (a, b) in enumerate(outer):
                    estimator, best_params, selected_threshold, evidence = fit_search(
                        x.iloc[a].reset_index(drop=True), y[a], p, spec, inner[i], order)
                    pred, prob = predictions(estimator, x.iloc[b], order, p, selected_threshold)
                    fold = score_metrics(y[b], pred, prob, order, p)
                    fold.update(fold=i + 1, best_params=best_params, selected_threshold=selected_threshold,
                                tuning_boundary=numeric_grid_boundaries(spec["grid"], best_params),
                                inner_search=evidence)
                    folds.append(fold)
                    records.extend({"source_row": int(source_rows[row]), "fold": i + 1,
                                    "actual": str(y[row]), "prediction": str(pred[j]), "probabilities": prob[j].tolist()}
                                   for j, row in enumerate(b))
                estimator, best_params, selected_threshold, final_evidence = fit_search(
                    x.reset_index(drop=True), y, p, spec, final_inner, order)
                model_path = variant_dir / f"{spec['name']}.joblib"
                joblib.dump(estimator, model_path)
                oof_path = variant_dir / f"{spec['name']}_oof.json"
                write_json(oof_path, {"class_order": order, "records": records})
                metric = p["metrics"]["primary"]
                summary = {}
                for m in [metric, *p["metrics"]["secondary"]]:
                    values = [f["metrics"][m] for f in folds]
                    complete = all(v is not None for v in values)
                    if m == metric and not complete:
                        raise ValueError(f"Undefined primary development metric {m}; revise plan")
                    summary[m] = {"folds": values, "mean": float(np.mean(values)) if complete else None,
                                  "std": float(np.std(values, ddof=1)) if complete else None,
                                  "defined_folds": sum(v is not None for v in values)}
                vr["models"][spec["name"]] = {
                    "type": spec["type"], "best_params": best_params, "selected_threshold": selected_threshold,
                    "effective_fixed_params": effective_fixed_params(p, spec),
                    "execution_controls": execution_controls(p, spec),
                    "tuning_boundary": numeric_grid_boundaries(spec["grid"], best_params),
                    "estimator_params": estimator.named_steps["model"].get_params(),
                    "final_inner_selection_score": final_evidence["best_score"],
                    "final_inner_search": final_evidence,
                    "final_refit": {"fit_rows": len(y), "refit": True, "selection_metric": final_evidence["metric"],
                                    "inner_splits": cv["inner_splits"], "split_seed": plan["seed"] + 99},
                    "outer_summary": summary, "fold_results": folds, "oof_rows": len(records),
                    "oof": {"path": str(oof_path.relative_to(output_dir)), "sha256": sha(oof_path)},
                    "artifact": {"path": str(model_path.relative_to(output_dir)), "sha256": sha(model_path)}}
        if plan["schema_version"] >= 6:
            result["model_selection"] = model_selection_evidence(plan, result["variants"]["baseline"]["models"])
        result["warnings"] = sorted({str(w.message) for w in caught})
    if sha(train) != result["training_source"]["sha256"]:
        raise ValueError("Training file changed during development")
    write_json(output_dir / "training_results.json", result, exclusive=True)
    write_json(output_dir / "environment.json", result["environment"], exclusive=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--sheet", default="Data")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-fits", type=int, help="Optional safety ceiling; schema-v5+ plans declare their own budget")
    args = parser.parse_args()
    run(args.train, args.plan, args.output_dir, args.sheet, args.max_fits)
