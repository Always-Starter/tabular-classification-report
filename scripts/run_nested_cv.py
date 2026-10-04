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

from common import (PrimaryScorer, code_hashes, environment, features, load_table, pipeline,
                    predictions, read_json, score_metrics, sha, splits, utcnow, write_json)
from validate_plan import validate


def numeric_grid_boundaries(grid, selected):
    """Flag selected edges of prespecified numeric grids without extending them."""
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
            edges[parameter] = {"selected": choice, "edge": edge,
                                "evaluated_min": ordered[0], "evaluated_max": ordered[-1]}
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
            "best_score": float(direction * search.best_score_), "candidates": candidates}


def run(train, plan_path, output_dir, sheet="Data", max_fits=1200):
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
    fit_count = sum((cv["outer_splits"] + 1) * (len(list(ParameterGrid(s["grid"]))) * cv["inner_splits"] + 1)
                    for p in variants.values() for s in p["models"])
    if fit_count > max_fits:
        raise ValueError(f"Planned {fit_count} fits exceed budget {max_fits}; reduce grid/folds/sensitivities")
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
              "outer_splits": [{"train_source_rows": source_rows[a].tolist(), "valid_source_rows": source_rows[b].tolist()}
                               for a, b in outer], "variants": {}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("error", category=ConvergenceWarning)
        for variant_name, p in variants.items():
            x = features(d, p)
            variant_dir = output_dir / variant_name
            variant_dir.mkdir()
            vr = {"plan": p, "models": {}}
            result["variants"][variant_name] = vr
            for spec in p["models"]:
                print(f"Running {variant_name}/{spec['name']} ({fit_count} total planned fits)", flush=True)
                folds, records = [], []
                scorer = PrimaryScorer(p, order)
                for i, (a, b) in enumerate(outer):
                    search = GridSearchCV(pipeline(p, spec), spec["grid"], scoring=scorer,
                                          cv=inner[i], n_jobs=1, error_score="raise")
                    search.fit(x.iloc[a], y[a])
                    pred, prob = predictions(search, x.iloc[b], order, p)
                    fold = score_metrics(y[b], pred, prob, order, p)
                    fold.update(fold=i + 1, best_params=search.best_params_,
                                tuning_boundary=numeric_grid_boundaries(spec["grid"], search.best_params_),
                                inner_search=search_evidence(search, p["metrics"]["primary"]))
                    folds.append(fold)
                    records.extend({"source_row": int(source_rows[row]), "fold": i + 1,
                                    "actual": str(y[row]), "prediction": str(pred[j]), "probabilities": prob[j].tolist()}
                                   for j, row in enumerate(b))
                search = GridSearchCV(pipeline(p, spec), spec["grid"], scoring=scorer,
                                      cv=final_inner, n_jobs=1, error_score="raise")
                search.fit(x, y)
                model_path = variant_dir / f"{spec['name']}.joblib"
                joblib.dump(search.best_estimator_, model_path)
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
                    "type": spec["type"], "best_params": search.best_params_,
                    "tuning_boundary": numeric_grid_boundaries(spec["grid"], search.best_params_),
                    "estimator_params": search.best_estimator_.named_steps["model"].get_params(),
                    "final_inner_selection_score": float(-search.best_score_ if metric == "log_loss" else search.best_score_),
                    "final_inner_search": search_evidence(search, metric),
                    "final_refit": {"fit_rows": len(y), "refit": True, "selection_metric": metric,
                                    "inner_splits": cv["inner_splits"], "split_seed": plan["seed"] + 99},
                    "outer_summary": summary, "fold_results": folds, "oof_rows": len(records),
                    "oof": {"path": str(oof_path.relative_to(output_dir)), "sha256": sha(oof_path)},
                    "artifact": {"path": str(model_path.relative_to(output_dir)), "sha256": sha(model_path)}}
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
    parser.add_argument("--max-fits", type=int, default=1200)
    args = parser.parse_args()
    run(args.train, args.plan, args.output_dir, args.sheet, args.max_fits)
