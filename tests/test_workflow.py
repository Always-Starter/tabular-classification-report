"""Behavioral tests using only generated data, never course train/test files."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sklearn.impute import SimpleImputer

import joblib
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import (ESTIMATORS, features, load_table, model_selection_evidence, pipeline,
                    predictions, read_json, score_metrics, sha, splits, write_json)
from validate_plan import validate
from run_nested_cv import numeric_grid_boundaries, run
from freeze_model_lock import freeze
from approve_model_lock import approve
from evaluate_holdout import evaluate
from verify_results import verify, verify_training
from diagnose_training import diagnose
from generate_report import decision_basis_text, generate, model_selection_text, validate_distribution_language
from audit_training_run import audit


def example():
    return read_json(ROOT / "tests/fixtures/example-plan.json")


def frame(n=120, multiclass=False):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(n, 2))
    labels = np.asarray(["alpha", "beta", "gamma"])[np.arange(n) % 3] if multiclass else np.where(x[:, 0] + x[:, 1] > 0, "yes", "no")
    d = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "category": np.where(x[:, 0] > 0, "a", "b"), "label": labels})
    d.loc[::11, "x1"] = np.nan
    d.loc[::13, "category"] = np.nan
    return d


class Contracts(unittest.TestCase):
    def test_schema_v6_policy_provenance_and_legacy_readability(self):
        p = example()
        self.assertEqual(decision_basis_text("conventional_default"), "conventional baseline")
        self.assertEqual(validate(p)["threshold_policy"]["value_source"], "conventional_default")
        self.assertEqual(p["cv"]["selection_basis"], "conventional_default")
        self.assertEqual({item["basis"] for item in p["decision_trace"]},
                         {"user_domain_constraint", "dataset_specific_evidence", "conventional_default"})
        legacy = copy.deepcopy(p)
        legacy["schema_version"] = 4
        legacy.pop("model_selection_policy")
        legacy["threshold"] = legacy.pop("threshold_policy")["value"]
        legacy.pop("compute_budget")
        for key in ("rationale", "status", "selection_basis"):
            legacy["cv"].pop(key)
        for item in legacy["decision_trace"]:
            item.pop("topic"); item.pop("basis")
        for spec in legacy["models"]:
            spec.pop("preprocessing_rationale"); spec.pop("imbalance_handling")
        self.assertEqual(validate(legacy)["threshold"], .5)

    def test_threshold_policy_modes_and_imbalance_contract(self):
        p = example()
        p["threshold_policy"] = {"mode": "model_default", "value": None, "search_values": [],
                                 "objective": None, "procedure": "estimator_default_class_decision",
                                 "value_source": "model_default",
                                 "value_rationale": "Use the estimator's declared class decision as a baseline.",
                                 "not_tuned_reason": "Operational error costs are unknown.",
                                 "selection_scope": "predeclared_before_development"}
        validate(p)
        weighted = example()
        weighted["models"][0]["params"]["class_weight"] = "balanced"
        weighted["models"][0]["fixed_param_rationale"]["class_weight"] = {
            "not_tuned_reason": "A user constraint requires fixed inverse-frequency weighting.",
            "value_source": "user_supplied",
            "value_rationale": "The user explicitly requested the library's balanced weighting rule."
        }
        weighted["models"][0]["imbalance_handling"] = {
            "strategy": "fixed_class_weight", "basis": "user_domain_constraint",
            "rationale": "The user requested fixed class weighting."
        }
        validate(weighted)
        weighted["models"][0]["imbalance_handling"]["strategy"] = "none"
        with self.assertRaisesRegex(ValueError, "imbalance_handling"):
            validate(weighted)

    def test_heterogeneous_diagnosis_records_evidence_without_prescribing_models(self):
        rng = np.random.default_rng(7)
        d = pd.DataFrame({"symmetric": rng.normal(size=200), "skewed": rng.lognormal(size=200),
                          "mostly_missing": np.where(np.arange(200) % 4, np.nan, rng.normal(size=200)),
                          "cat_a": np.tile(["a", "b", "c", "d"], 50),
                          "cat_b": np.tile(["u", "v"], 100),
                          "label": np.where(np.arange(200) < 190, "major", "minor")})
        result = diagnose(d, "label")
        self.assertLess(abs(result["numeric"]["symmetric"]["distribution_evidence"]["skewness"]["value"]), 1)
        self.assertTrue(result["numeric"]["skewed"]["distribution_evidence"]["skewness"]["material_flag"])
        self.assertGreater(result["missing_predictors"]["mostly_missing"]["percent"], 70)
        self.assertEqual(result["categorical"]["cat_a"]["cardinality_nonmissing"], 4)
        self.assertFalse(result["feature_type_evidence"]["symmetric"]["semantic_type_confirmed"])
        self.assertIn("do not establish", result["feature_type_evidence"]["symmetric"]["claim_limit"])
        self.assertNotIn("recommended_model", result)

    def test_distribution_evidence_and_claim_guardrails(self):
        d = frame(200)
        d["spiky"] = np.concatenate([np.linspace(-1, 1, 199), [50]])
        diagnosis = diagnose(d, "label")
        evidence = diagnosis["numeric"]["spiky"]["distribution_evidence"]
        self.assertTrue(evidence["skewness"]["material_flag"])
        self.assertTrue(evidence["tail_weight"]["high_signal_flag"])
        self.assertGreater(evidence["potential_outliers"]["flagged_count"], 0)
        self.assertIn("spiky", diagnosis["numeric_distribution_summary"]["material_skewness_columns"])

        narrative = {key: "No distribution claim is made."
                     for key in ("exploration", "preprocessing", "features", "model_rationale", "findings", "limitations")}
        narrative["preprocessing"] = "Robust scaling was used because distribution tails differ."
        with self.assertRaisesRegex(ValueError, "tails differ"):
            validate_distribution_language(narrative, diagnosis)
        narrative["preprocessing"] = "The data contain outliers."
        with self.assertRaisesRegex(ValueError, "potential/statistical"):
            validate_distribution_language(narrative, diagnosis)
        narrative["preprocessing"] = "The data have heavy tails."
        with self.assertRaisesRegex(ValueError, "excess-kurtosis"):
            validate_distribution_language(narrative, diagnosis)
        narrative["preprocessing"] = ("High excess kurtosis suggests a possible heavy-tail signal; Tukey IQR fences "
                                       "flagged potential statistical outlier candidates, not confirmed errors.")
        validate_distribution_language(narrative, diagnosis)

    def test_numeric_grid_boundaries_only_report_uncertainty(self):
        grid = {"model__C": [0.1, 1.0, 10.0], "model__criterion": ["gini", "entropy"],
                "model__single": [7]}
        upper = numeric_grid_boundaries(grid, {"model__C": 10.0,
                                               "model__criterion": "gini", "model__single": 7})
        self.assertEqual(upper["model__C"]["boundary_type"], "edge_with_interior_candidates")
        self.assertTrue(upper["model__C"]["interior_candidates_evaluated"])
        self.assertTrue(upper["model__C"]["supports_outside_range_question"])
        self.assertEqual(numeric_grid_boundaries(grid, {"model__C": 1.0,
                                                         "model__criterion": "gini", "model__single": 7}), {})
        coarse = numeric_grid_boundaries({"model__C": [0.1, 1.0]}, {"model__C": 0.1})["model__C"]
        self.assertEqual(coarse["boundary_type"], "two_value_grid_endpoint")
        self.assertFalse(coarse["interior_candidates_evaluated"])
        self.assertFalse(coarse["supports_outside_range_question"])
        self.assertIn("coarse search coverage", coarse["interpretation"])

    def test_invalid_plans_fail_before_fitting(self):
        cases = []
        p = example(); p["features"].append("label"); cases.append(p)
        p = example(); p["models"][1]["name"] = "linear"; cases.append(p)
        p = example(); p["models"][0]["name"] = "../../bad"; cases.append(p)
        p = example(); p["metrics"]["primary"] = "invented"; cases.append(p)
        p = example(); p["models"][0]["type"] = "unknown"; cases.append(p)
        p = example(); p["models"][0]["grid"] = {"model__C": [-1]}; cases.append(p)
        p = example(); p["models"][0]["grid"] = {"model__C": list(range(1, 20))}; cases.append(p)
        p = example(); p["models"][0]["grid"] = {"model__random_state": [1]}; cases.append(p)
        p = example(); p["models"][0]["grid"]["model__max_iter"] = [1000, 3000]; cases.append(p)
        p = example(); p["models"][0]["type"] = "support_vector_classifier"; p["models"][0]["preprocessing"]["scaler"] = "none"; cases.append(p)
        p = example(); p["models"] = p["models"][:1]; cases.append(p)
        p = example(); p["models"].extend([copy.deepcopy(p["models"][0]), copy.deepcopy(p["models"][1])]); cases.append(p)
        p = example(); p["models"].append(copy.deepcopy(p["models"][0])); p["models"][2]["name"] = "third"; p["model_count_rationale"] = ""; cases.append(p)
        p = example(); del p["models"][0]["fixed_param_rationale"]; cases.append(p)
        p = example(); p["models"][0]["fixed_param_rationale"] = {}; cases.append(p)
        p = example(); p["models"][0]["fixed_param_rationale"]["max_iter"]["value_source"] = "unknown"; cases.append(p)
        p = example(); p["models"][0]["fixed_param_rationale"]["max_iter"]["value_rationale"] = None; cases.append(p)
        for p in cases:
            with self.subTest(plan=p), self.assertRaises((ValueError, TypeError)):
                validate(p)

    def test_unknown_fixed_value_source_is_explicit(self):
        p = example()
        record = p["models"][1]["fixed_param_rationale"]["max_depth"]
        record["value_source"] = "unknown"
        record["value_rationale"] = None
        self.assertEqual(validate(p)["models"][1]["fixed_param_rationale"]["max_depth"]["value_source"],
                         "unknown")

    def test_three_models_require_a_reason(self):
        p = example()
        third = copy.deepcopy(p["models"][1])
        third["name"] = "forest"
        third["type"] = "random_forest"
        third["params"] = {"n_estimators": 10}
        third["fixed_param_rationale"] = {
            "n_estimators": {
                "not_tuned_reason": "The synthetic third-model check does not tune ensemble size.",
                "value_source": "predeclared_rule",
                "value_rationale": "Ten trees keep this software test fast; this is not a modelling recommendation."
            }
        }
        third["grid"] = {}
        p["models"].append(third)
        p["model_count_rationale"] = "Training diagnosis motivates an additional nonlinear ensemble comparison."
        p["model_selection_policy"]["preference_order"].append("forest")
        self.assertEqual(len(validate(p)["models"]), 3)

    def test_schema_v6_rejects_undeclared_runner_material_defaults(self):
        cases = []
        p = example(); del p["models"][0]["params"]["max_iter"]; del p["models"][0]["fixed_param_rationale"]["max_iter"]; cases.append((p, "max_iter"))
        p = example(); p["models"][1]["type"] = "random_forest"; p["models"][1]["params"] = {}; p["models"][1]["fixed_param_rationale"] = {}; p["models"][1]["grid"] = {}; cases.append((p, "n_estimators"))
        p = example(); p["models"][0]["type"] = "support_vector_classifier"; p["models"][0]["params"] = {}; p["models"][0]["fixed_param_rationale"] = {}; cases.append((p, "calibration_cv"))
        for plan, parameter in cases:
            with self.subTest(parameter=parameter), self.assertRaisesRegex(ValueError, parameter):
                validate(plan)

    def test_executable_model_selection_policy_handles_direction_and_ties(self):
        p = example()
        models = {
            "linear": {"outer_summary": {"f1": {"mean": .80, "std": .05}}},
            "tree": {"outer_summary": {"f1": {"mean": .81, "std": .08}}},
        }
        self.assertEqual(model_selection_evidence(p, models)["selected_model"], "tree")
        p["model_selection_policy"]["practical_tie_tolerance"] = .02
        tied = model_selection_evidence(p, models)
        self.assertEqual(tied["selected_model"], "linear")
        rendered = model_selection_text(tied, {"linear": "Logistic regression", "tree": "Decision tree"})
        self.assertIn("difference between Logistic regression and Decision tree was 0.01000", rendered)
        self.assertIn("practical-tie tolerance of 0.02", rendered)
        self.assertIn("lower outer-fold SD tie-breaker (0.05000 vs 0.08000)", rendered)
        three_way = copy.deepcopy(tied)
        three_way["candidates"].append({"model": "forest", "mean": .805, "std": .06,
                                         "preference_rank": 3, "gap_from_numerical_best": .005,
                                         "within_practical_tie": True})
        three_way["practical_tie_contenders"].append("forest")
        rendered = model_selection_text(
            three_way,
            {"linear": "Logistic regression", "tree": "Decision tree", "forest": "Random forest"},
        )
        self.assertIn("Random forest (mean 0.80500, SD 0.06000)", rendered)
        self.assertIn("Logistic regression was selected", rendered)
        p["metrics"]["primary"] = "log_loss"
        models = {
            "linear": {"outer_summary": {"log_loss": {"mean": .42, "std": .03}}},
            "tree": {"outer_summary": {"log_loss": {"mean": .45, "std": .02}}},
        }
        p["model_selection_policy"]["practical_tie_tolerance"] = 0
        evidence = model_selection_evidence(p, models)
        self.assertEqual(evidence["direction"], "minimize")
        self.assertEqual(evidence["selected_model"], "linear")
        rendered = model_selection_text(evidence, {"linear": "Logistic regression", "tree": "Decision tree"})
        self.assertIn("Logistic regression alone fell within", rendered)
        self.assertIn("selected without a tie-breaker", rendered)

    def test_nonlast_positive_class_metrics(self):
        p = example()
        y = np.array(["no", "yes", "no", "yes"])
        prob = np.array([[.9, .1], [.2, .8], [.8, .2], [.1, .9]])
        m = score_metrics(y, y, prob, ["no", "yes"], p)
        self.assertEqual(m["metrics"]["roc_auc"], 1)
        self.assertEqual(m["metrics"]["average_precision"], 1)
        self.assertEqual(m["metrics"]["precision"], 1)

    def test_zero_denominators_are_not_observed_zero_scores(self):
        p = example()
        y = np.array(["yes", "yes"])
        # 'no' is the frozen positive class; a false positive gives defined precision/F1=0.
        m = score_metrics(y, np.array(["no", "yes"]), np.array([[.9, .1], [.1, .9]]), ["no", "yes"], p)
        self.assertIsNone(m["metrics"]["recall"])
        self.assertEqual(m["metrics"]["precision"], 0)
        self.assertEqual(m["metrics"]["f1"], 0)
        # No actual or predicted positives makes all three denominators zero.
        m = score_metrics(y, y, np.array([[.1, .9], [.1, .9]]), ["no", "yes"], p)
        for metric in ("recall", "precision", "f1"):
            self.assertIsNone(m["metrics"][metric])
            self.assertIn(metric, m["undefined_metrics"])

    def test_all_estimators_and_configurable_preprocessing(self):
        p, d = example(), frame()
        x, y = features(d, p), d.label
        for kind in ESTIMATORS:
            with self.subTest(kind=kind):
                spec = copy.deepcopy(p["models"][0])
                spec.update(type=kind, params={}, grid={})
                spec["preprocessing"].update(numeric_imputer="mean", scaler="robust", categorical_encoder="ordinal")
                model = pipeline(p, spec).fit(x, y)
                pred, prob = predictions(model, x, ["no", "yes"], p)
                self.assertEqual(prob.shape, (len(d), 2))
                self.assertEqual(model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].strategy, "mean")
                self.assertEqual(len(pred), len(d))

    def test_group_and_time_splits_preserve_dependencies(self):
        p = example()
        y = np.tile(["no", "yes"], 60)
        d = frame(); d["entity"] = np.repeat(np.arange(30), 4)
        p["cv"] = {"strategy": "stratified_group", "outer_splits": 3, "inner_splits": 2,
                   "group_column": "entity", "rationale": "Repeated synthetic entities require grouped splits.",
                   "status": "confirmed", "selection_basis": "dataset_specific_evidence"}
        p["excluded_features"] = {"entity": "Repeated entity; group validation"}
        validate(p)
        for a, b in splits(d, y, p, 3, 1):
            self.assertFalse(set(d.iloc[a].entity) & set(d.iloc[b].entity))
        d["time"] = np.repeat(pd.date_range("2020-01-01", periods=30), 4)
        p["cv"] = {"strategy": "time", "outer_splits": 3, "inner_splits": 2, "time_column": "time", "gap": 1,
                   "rationale": "Synthetic timestamps require forward validation.", "status": "confirmed",
                   "selection_basis": "dataset_specific_evidence"}
        p["excluded_features"] = {"time": "Forward-only evaluation"}
        validate(p)
        for a, b in splits(d, y, p, 3, 1):
            self.assertLess(d.iloc[a].time.max(), d.iloc[b].time.min())
            self.assertFalse(set(d.iloc[a].time) & set(d.iloc[b].time))

    def test_label_strings_and_file_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = frame(20); d["label"] = ["01", "02"] * 10
            for suffix in (".csv", ".tsv", ".xlsx"):
                p = Path(tmp) / ("train" + suffix)
                if suffix == ".xlsx":
                    d.to_excel(p, sheet_name="Data", index=False)
                else:
                    d.to_csv(p, sep="\t" if suffix == ".tsv" else ",", index=False)
                self.assertEqual(set(load_table(p, "Data", "label").label), {"01", "02"})


class Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.train = cls.base / "training.csv"
        frame().to_csv(cls.train, index=False)
        cls.plan = example()
        threshold_check = copy.deepcopy(cls.plan["threshold_policy"])
        threshold_check.update(value=.65, value_rationale="A predeclared fixed alternative used only as sensitivity evidence.")
        cls.plan["sensitivities"] = [{"name": "threshold_check", "rationale": "Predeclared operating-point uncertainty",
                                      "overrides": {"threshold_policy": threshold_check}}]
        cls.plan_path = cls.base / "plan.json"
        write_json(cls.plan_path, cls.plan)
        cls.development = cls.base / "development"
        cls.results = run(cls.train, cls.plan_path, cls.development)
        cls.review = cls.base / "review.json"
        write_json(cls.review, {"selected_variant": "baseline", "preferred_model": cls.results["model_selection"]["selected_model"], "rationale": "Synthetic comparison for software testing only",
                               "selection_rule": "Use the fixture's primary outer-CV comparison, then plan order only if needed.",
                               "tie_breaker": "No substantive tie-breaker claim is made for the software fixture.",
                               "sensitivity_review": "Keep baseline; threshold alternative is a software fixture", "warnings_review": "Reviewed synthetic run warnings"})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def new_lock(self, directory, approval_required=False):
        path = Path(directory) / "model-lock.json"
        freeze(self.development / "training_results.json", self.development, self.review, path,
               require_human_approval=approval_required)
        return path

    def approve_fixture(self, lock):
        approve(lock, sha(lock), "automated synthetic test", "Fixture-only approval; not a student's approval")

    def holdout(self, directory, labelled=True):
        d = frame(30)
        d.loc[0, "category"] = "previously unseen"
        d.loc[1, "label"] = None
        if not labelled:
            d = d.drop(columns="label")
        test = Path(directory) / "heldout.tsv"
        d.to_csv(test, sep="\t", index=False)
        return test

    def test_nested_evidence_and_sensitivity(self):
        self.assertTrue(verify_training(self.development / "training_results.json")["verified"])
        self.assertEqual(set(self.results["variants"]), {"baseline", "threshold_check"})
        self.assertTrue(self.results["variants"]["baseline"]["selection_eligible"])
        self.assertEqual(self.results["variants"]["threshold_check"]["role"], "interpretive_sensitivity")
        self.assertFalse(self.results["variants"]["threshold_check"]["selection_eligible"])
        self.assertFalse(self.results["test_data_accessed"])
        for v in self.results["variants"].values():
            for model in v["models"].values():
                self.assertEqual(model["oof_rows"], 120)
                self.assertIn("tuning_boundary", model)
                self.assertTrue(all("tuning_boundary" in fold for fold in model["fold_results"]))
                self.assertIn("final_inner_search", model)
                self.assertEqual(model["final_refit"]["fit_rows"], 120)
                for fold in model["fold_results"]:
                    self.assertEqual(fold["inner_search"]["inner_splits"], 2)
                    self.assertTrue(all(len(candidate["split_scores"]) == 2
                                        for candidate in fold["inner_search"]["candidates"]))
        audited = audit(self.development / "training_results.json")
        self.assertEqual(audited["audit_scope"], "training_only")
        self.assertFalse(audited["held_out_data_accessed_by_audit"])
        self.assertIsNotNone(audited["models"]["linear"]["outer_folds"][0]["inner_search"])
        self.assertEqual(audited["models"]["linear"]["fixed_param_rationale"]["max_iter"]["value_source"],
                         "convergence_requirement")
        self.assertEqual(audited["models"]["linear"]["tuning_boundary"]["model__C"]["boundary_type"],
                         "two_value_grid_endpoint")
        m = joblib.load(self.development / "baseline/linear.joblib")
        learned = m.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_
        self.assertTrue(np.allclose(learned, frame()[["x1", "x2"]].median().to_numpy()))

    def test_sensitivity_variant_cannot_be_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            review = Path(tmp) / "review.json"
            payload = read_json(self.review)
            payload["selected_variant"] = "threshold_check"
            write_json(review, payload)
            with self.assertRaisesRegex(ValueError, "interpretive and cannot be locked"):
                freeze(self.development / "training_results.json", self.development, review,
                       Path(tmp) / "model-lock.json")

    def test_model_lock_rejects_preference_that_conflicts_with_predeclared_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            review = Path(tmp) / "review.json"
            payload = read_json(self.review)
            payload["preferred_model"] = next(
                name for name in self.results["variants"]["baseline"]["models"]
                if name != self.results["model_selection"]["selected_model"])
            write_json(review, payload)
            with self.assertRaisesRegex(ValueError, "conflicts with the predeclared executable selection policy"):
                freeze(self.development / "training_results.json", self.development, review,
                       Path(tmp) / "model-lock.json")

    def test_svm_calibration_feasibility_fails_before_any_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p = example()
            svm = p["models"][0]
            svm["type"] = "support_vector_classifier"
            svm["params"] = {"calibration_cv": 3}
            svm["fixed_param_rationale"] = {
                "calibration_cv": {
                    "not_tuned_reason": "This test predeclares the calibration fold count.",
                    "value_source": "predeclared_rule",
                    "value_rationale": "Three folds exercise the pre-fit feasibility guard."
                }
            }
            d = frame(26)
            d["label"] = ["no"] * 20 + ["yes"] * 6
            d.to_csv(tmp / "train.csv", index=False)
            write_json(tmp / "plan.json", p)
            with patch.object(SimpleImputer, "fit", side_effect=AssertionError("Must fail before fitting")):
                with self.assertRaisesRegex(ValueError, "SVM calibration is infeasible before fitting"):
                    run(tmp / "train.csv", tmp / "plan.json", tmp / "development")

    def test_fold_local_imputer_fits_and_missing_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); p = example()
            for spec in p["models"]:
                spec["grid"] = {}
            d = frame(); d.loc[0, "label"] = None
            d.to_csv(tmp / "train.csv", index=False); write_json(tmp / "plan.json", p)
            fit_rows = []
            original_fit = SimpleImputer.fit
            def observed_fit(estimator, x, *args, **kwargs):
                fit_rows.append(len(x))
                return original_fit(estimator, x, *args, **kwargs)
            with patch.object(SimpleImputer, "fit", observed_fit):
                r = run(tmp / "train.csv", tmp / "plan.json", tmp / "dev")
            self.assertEqual(r["missing_targets"], 1)
            self.assertEqual(r["eligible_rows"], 119)
            self.assertTrue(any(n < 40 for n in fit_rows))  # inner-training folds
            self.assertTrue(any(50 < n < 70 for n in fit_rows))  # outer-training/final-inner folds
            self.assertEqual(fit_rows.count(119), 4)  # numeric + categorical final refit, two models
            self.assertNotIn(120, fit_rows)

    def test_compute_budget_refuses_before_fitting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(SimpleImputer, "fit", side_effect=AssertionError("Must not fit")):
            with self.assertRaisesRegex(ValueError, "exceed budget"):
                run(self.train, self.plan_path, Path(tmp) / "out", max_fits=1)

    def test_tuned_threshold_is_selected_only_inside_each_inner_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p = example()
            p["sensitivities"] = []
            p["threshold_policy"] = {"mode": "tuned", "value": None, "search_values": [.35, .5, .65],
                                     "objective": "f1", "procedure": "joint_inner_cv_grid",
                                     "value_source": "inner_cv",
                                     "value_rationale": "Select the tested operating point within each inner CV search.",
                                     "not_tuned_reason": None, "selection_scope": "inner_cv_only"}
            for spec in p["models"]:
                spec["grid"] = {}
            train = tmp / "train.csv"
            frame().to_csv(train, index=False)
            write_json(tmp / "plan.json", p)
            result = run(train, tmp / "plan.json", tmp / "dev")
            self.assertTrue(verify_training(tmp / "dev/training_results.json")["verified"])
            for model in result["variants"]["baseline"]["models"].values():
                self.assertIn(model["selected_threshold"], [.35, .5, .65])
                self.assertTrue(model["final_inner_search"]["threshold_tuned"])
                self.assertEqual(len(model["final_inner_search"]["candidates"]), 3)
                for fold in model["fold_results"]:
                    self.assertIn(fold["selected_threshold"], [.35, .5, .65])
                    self.assertTrue(fold["inner_search"]["threshold_tuned"])
            audited = audit(tmp / "dev/training_results.json")
            self.assertEqual(audited["threshold_policy"]["selection_scope"], "inner_cv_only")

    def test_every_predictor_needs_a_selection_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); d = frame(); d["forgotten_column"] = np.arange(len(d))
            d.to_csv(tmp / "train.csv", index=False)
            with patch.object(SimpleImputer, "fit", side_effect=AssertionError("Must not fit")), self.assertRaisesRegex(ValueError, "unaccounted.*forgotten_column"):
                run(tmp / "train.csv", self.plan_path, tmp / "out")

    def test_unapproved_refuses_before_test_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = self.new_lock(tmp, approval_required=True)
            with patch("evaluate_holdout.load_table", side_effect=AssertionError("Must not read")), self.assertRaisesRegex(ValueError, "approval"):
                evaluate(Path(tmp) / "does-not-exist.csv", lock, self.development, Path(tmp) / "out")
            self.assertFalse(lock.with_name(lock.name + ".holdout.json").exists())

    def test_staged_lock_evaluates_after_explicit_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lock = self.new_lock(tmp, approval_required=True)
            self.approve_fixture(lock)
            result = evaluate(self.holdout(tmp), lock, self.development, tmp / "out")
            self.assertEqual(result["approval"]["lock_sha256"], sha(lock))
            self.assertTrue(verify(tmp / "out/test_results.json", lock)["verified"])

    def test_missing_lock_seal_refuses_before_test_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = self.new_lock(tmp)
            lock.with_name(lock.name + ".seal.json").unlink()
            with patch("evaluate_holdout.load_table", side_effect=AssertionError("Must not read")), self.assertRaises(FileNotFoundError):
                evaluate(Path(tmp) / "does-not-exist.csv", lock, self.development, Path(tmp) / "out")
            self.assertFalse(lock.with_name(lock.name + ".holdout.json").exists())

    def test_bad_digest_and_tampered_lock_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = self.new_lock(tmp, approval_required=True)
            with self.assertRaisesRegex(ValueError, "digest"):
                approve(lock, "0" * 64, "reviewer", "approved")
            self.approve_fixture(lock)
            payload = read_json(lock); payload["plan"]["threshold_policy"]["value"] = .9; write_json(lock, payload)
            with self.assertRaisesRegex(ValueError, "seal/digest"):
                evaluate(Path(tmp) / "absent.csv", lock, self.development, Path(tmp) / "out")

    def test_automatic_lock_evaluates_without_approval_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lock = self.new_lock(tmp)
            self.assertFalse(lock.with_name(lock.name + ".approval.json").exists())
            with self.assertRaisesRegex(ValueError, "does not require human approval"):
                approve(lock, sha(lock), "synthetic reviewer", "approved")
            result = evaluate(self.holdout(tmp), lock, self.development, tmp / "out")
            self.assertIsNone(result["approval"])
            self.assertTrue(verify(tmp / "out/test_results.json", lock)["verified"])
        with tempfile.TemporaryDirectory() as tmp:
            lock = self.new_lock(tmp)
            payload = read_json(lock); payload["plan"]["threshold_policy"]["value"] = .9; write_json(lock, payload)
            with patch("evaluate_holdout.load_table", side_effect=AssertionError("Must not read")), self.assertRaisesRegex(ValueError, "seal/digest"):
                evaluate(Path(tmp) / "absent.csv", lock, self.development, Path(tmp) / "out")

    def test_tampered_model_refuses_before_test_read(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            lock = self.new_lock(tmp)
            copied = Path(tmp) / "models"; shutil.copytree(self.development, copied)
            with (copied / "baseline/linear.joblib").open("ab") as stream:
                stream.write(b"tampering")
            with self.assertRaisesRegex(ValueError, "hash/path mismatch"):
                evaluate(Path(tmp) / "absent.csv", lock, copied, Path(tmp) / "out")

    def test_changed_code_refuses_before_test_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = self.new_lock(tmp)
            with patch("evaluate_holdout.code_hashes", return_value={}), self.assertRaisesRegex(ValueError, "Code/environment"):
                evaluate(Path(tmp) / "absent.csv", lock, self.development, Path(tmp) / "out")

    def test_corrupt_predictions_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); lock = self.new_lock(tmp)
            evaluate(self.holdout(tmp), lock, self.development, tmp / "out")
            path = tmp / "out/linear_predictions.json"
            content = read_json(path); content["records"][0]["prediction"] = "tampered"; write_json(path, content)
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify(tmp / "out/test_results.json", lock)

    def test_holdout_verify_repeat_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); lock = self.new_lock(tmp)
            test = self.holdout(tmp)
            result = evaluate(test, lock, self.development, tmp / "out")
            self.assertEqual(result["missing_labels"], 1)
            self.assertEqual(result["class_order"], ["no", "yes"])
            verified = verify(tmp / "out/test_results.json", lock)
            self.assertTrue(verified["verified"])
            with self.assertRaises(FileExistsError):
                evaluate(test, lock, self.development, tmp / "different-out")
            diagnosis = {"training_source": {"sha256": sha(self.train)}, **diagnose(frame(), "label")}
            write_json(tmp / "diagnosis.json", diagnosis)
            narrative = read_json(ROOT / "references/narrative-template.json")
            narrative.update(exploration="Synthetic data contain missing predictor values.", preprocessing="Fold-local imputation and encoding handle missing values.",
                             features="All three fixture predictors are retained.", model_rationale="Linear and tree boundaries provide a controlled contrast.",
                             findings="Software fixture only; do not interpret these numbers as course results.", limitations="Synthetic data do not establish real-world generalization.")
            narrative["metadata"] = {"full_name": "Example Student", "matric_number": "A1234567X",
                                     "llm_model_version": "test-model", "llm_interface": "test-interface",
                                     "repository_url": "https://example.com/example-skill"}
            write_json(tmp / "narrative.json", narrative)
            manifest = generate(self.development / "training_results.json", tmp / "diagnosis.json", tmp / "narrative.json", tmp / "report", test_results=tmp / "out/test_results.json", lock=lock)
            self.assertEqual(manifest["main_pages"], 2)
            self.assertTrue(manifest["draft"])
            self.assertFalse(manifest["renderer_changed_since_lock"])
            self.assertGreaterEqual(manifest["total_pages"], 3)
            report = (tmp / "report/report.md").read_text(encoding="utf-8")
            self.assertIn("outer folds did not trigger grid expansion", report)
            self.assertNotIn("SHA-256 evidence accompany this report", report)
            self.assertIn("Prediction errors (confusion matrix)", report)
            self.assertIn("Positive class: no. TN = true negatives", report)
            self.assertIn("| Logistic Regression | F1 (no) |", report)
            self.assertIn("|  | precision (no) |", report)
            self.assertIn("## Human in the Loop", report)
            self.assertIn("## Critical Evaluation", report)
            self.assertIn("## Trustworthiness", report)
            self.assertIn("endpoint of a two-value grid", report)
            self.assertIn("Fixed-parameter rationale", report)
            self.assertIn("Imbalance handling: none; basis: dataset-specific evidence", report)
            self.assertIn("FACT:", report)
            self.assertIn("LIMITATION/UNKNOWN:", report)
            self.assertIn("FUTURE WORK:", report)
            self.assertIn("Fixed threshold 0.5 was predeclared", report)
            self.assertIn("basis: conventional baseline", report)
            self.assertNotIn(".;", report)
            self.assertNotIn("..", report)
            self.assertIn("| Name | Example Student |", report)
            self.assertIn("| Matriculation number | A1234567X |", report)
            self.assertIn("| Skill repository | [https://example.com/example-skill](https://example.com/example-skill) |", report)
            links = [annotation.get_object().get("/A", {}).get("/URI")
                     for page in PdfReader(tmp / "report/report.pdf").pages
                     for annotation in (page.get("/Annots") or [])
                     if annotation.get_object().get("/Subtype") == "/Link"]
            self.assertIn("https://example.com/example-skill", links)
            pdf_text = "\n".join(page.extract_text() or ""
                                 for page in PdfReader(tmp / "report/report.pdf").pages)
            self.assertIn("https://example.com/example-skill", pdf_text)
            narrative["identification_cover_page"] = True
            write_json(tmp / "cover-narrative.json", narrative)
            cover_manifest = generate(
                self.development / "training_results.json", tmp / "diagnosis.json",
                tmp / "cover-narrative.json", tmp / "cover-report",
                test_results=tmp / "out/test_results.json", lock=lock)
            self.assertEqual(cover_manifest["cover_pages"], 1)
            self.assertEqual(cover_manifest["main_pages"], 2)
            cover_pages = [page.extract_text() or ""
                           for page in PdfReader(tmp / "cover-report/report.pdf").pages]
            self.assertIn("IN6227-Assignment-1", cover_pages[0])
            self.assertIn("Example Student", cover_pages[0])
            self.assertIn("A1234567X", cover_pages[0])
            self.assertIn("Tabular classification report", cover_pages[1])
            self.assertIn("Reflection", cover_pages[3])
            self.assertNotIn("Automated verification aid", report)
            self.assertNotIn('"numeric_imputer"', report)
            locked_hashes = read_json(lock)["code_sha256"]
            renderer_changed = {**locked_hashes, "generate_report.py": "0" * 64}
            with patch("generate_report.code_hashes", return_value=renderer_changed):
                with self.assertRaisesRegex(ValueError, "presentation-only-rerender"):
                    generate(self.development / "training_results.json", tmp / "diagnosis.json", tmp / "narrative.json",
                             tmp / "blocked", test_results=tmp / "out/test_results.json", lock=lock)
                derived = generate(self.development / "training_results.json", tmp / "diagnosis.json", tmp / "narrative.json",
                                   tmp / "presentation-only", test_results=tmp / "out/test_results.json", lock=lock,
                                   presentation_only_rerender=True)
                self.assertTrue(derived["renderer_changed_since_lock"])
            model_code_changed = {**locked_hashes, "common.py": "0" * 64}
            with patch("generate_report.code_hashes", return_value=model_code_changed):
                with self.assertRaisesRegex(ValueError, "Non-renderer code changed"):
                    generate(self.development / "training_results.json", tmp / "diagnosis.json", tmp / "narrative.json",
                             tmp / "blocked-model-code", test_results=tmp / "out/test_results.json", lock=lock,
                             presentation_only_rerender=True)
            # Verification uses stored predictions, even when the test file is no longer available.
            test.rename(tmp / "sealed-away.tsv")
            self.assertTrue(verify(tmp / "out/test_results.json", lock)["verified"])

    def test_unlabelled_and_single_class_holdout(self):
        for mode in ("unlabelled", "one-class"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp); lock = self.new_lock(tmp)
                test = self.holdout(tmp, labelled=mode != "unlabelled")
                if mode == "one-class":
                    d = load_table(test); d["label"] = "no"; d.to_csv(test, sep="\t", index=False)
                result = evaluate(test, lock, self.development, tmp / "out")
                self.assertIsNone(result["models"]["linear"]["metrics"]["roc_auc"])
                self.assertTrue(verify(tmp / "out/test_results.json", lock)["verified"])

    def test_missing_feature_consumes_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); lock = self.new_lock(tmp)
            test = tmp / "heldout.csv"; frame(30).drop(columns="x1").to_csv(test, index=False)
            with self.assertRaisesRegex(ValueError, "Missing required"):
                evaluate(test, lock, self.development, tmp / "out")
            self.assertEqual(read_json(lock.with_name(lock.name + ".holdout.json"))["status"], "failed_after_access_reserved")
            with self.assertRaises(FileExistsError):
                evaluate(test, lock, self.development, tmp / "out2")

    def test_multiclass_full_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); p = example(); p["task"] = "multiclass"
            del p["positive_class"], p["threshold_policy"]
            p["metrics"] = {"primary": "f1_macro", "secondary": ["accuracy", "log_loss", "roc_auc_ovr_macro"],
                            "rationale": "Exercise multiclass metrics in the synthetic workflow.", "status": "confirmed"}
            p["semantics"]["positive_class_meaning"] = None
            train = tmp / "train.csv"; frame(120, True).to_csv(train, index=False)
            write_json(tmp / "plan.json", p)
            run(train, tmp / "plan.json", tmp / "dev")
            lock = tmp / "model-lock.json"
            freeze(tmp / "dev/training_results.json", tmp / "dev", self.review, lock)
            test = tmp / "test.csv"; frame(30, True).to_csv(test, index=False)
            result = evaluate(test, lock, tmp / "dev", tmp / "out")
            self.assertEqual(result["class_order"], ["alpha", "beta", "gamma"])
            self.assertEqual(len(result["models"]["linear"]["confusion_matrix"]), 3)
            self.assertTrue(verify(tmp / "out/test_results.json", lock)["verified"])
            diagnosis = {"training_source": {"sha256": sha(train)}, **diagnose(frame(120, True), "label")}
            write_json(tmp / "diagnosis.json", diagnosis)
            narrative = read_json(ROOT / "references/narrative-template.json")
            narrative.update(exploration="Synthetic multiclass data.", preprocessing="Fold-local preparation.",
                             features="All fixture predictors retained.", model_rationale="Two software-test model families.",
                             findings="Software fixture only.", limitations="No real-world inference.")
            write_json(tmp / "narrative.json", narrative)
            manifest = generate(tmp / "dev/training_results.json", tmp / "diagnosis.json", tmp / "narrative.json",
                                tmp / "report", test_results=tmp / "out/test_results.json", lock=lock)
            self.assertEqual(manifest["main_pages"], 2)
            report = (tmp / "report/report.md").read_text(encoding="utf-8")
            self.assertIn("Actual to predicted counts: alpha", report)

    def test_three_model_full_workflow_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p = example()
            third = copy.deepcopy(p["models"][1])
            third.update(name="forest", type="random_forest", params={"n_estimators": 10}, grid={})
            third["fixed_param_rationale"] = {
                "n_estimators": {
                    "not_tuned_reason": "The synthetic third-model check does not tune ensemble size.",
                    "value_source": "predeclared_rule",
                    "value_rationale": "Ten trees keep this software test fast; this is not a modelling recommendation."
                }
            }
            p["models"].append(third)
            p["model_count_rationale"] = "Synthetic three-model software check; no course-data recommendation."
            p["model_selection_policy"]["preference_order"].append("forest")
            train = tmp / "train.csv"
            frame().to_csv(train, index=False)
            write_json(tmp / "plan.json", p)
            result = run(train, tmp / "plan.json", tmp / "dev")
            self.assertEqual(set(result["variants"]["baseline"]["models"]), {"linear", "tree", "forest"})
            review = tmp / "review.json"
            write_json(review, {"selected_variant": "baseline", "preferred_model": result["model_selection"]["selected_model"],
                                "rationale": "Synthetic test", "sensitivity_review": "None declared",
                                "selection_rule": "Use the prespecified primary outer-CV comparison.",
                                "tie_breaker": "No tie-breaker was needed in this software test.",
                                "warnings_review": "Reviewed synthetic warnings"})
            lock = tmp / "model-lock.json"
            freeze(tmp / "dev/training_results.json", tmp / "dev", review, lock)
            test = self.holdout(tmp)
            evaluated = evaluate(test, lock, tmp / "dev", tmp / "out")
            self.assertEqual(set(evaluated["models"]), {"linear", "tree", "forest"})
            self.assertTrue(verify(tmp / "out/test_results.json", lock)["verified"])
            diagnosis = {"training_source": {"sha256": sha(train)}, **diagnose(frame(), "label")}
            write_json(tmp / "diagnosis.json", diagnosis)
            narrative = read_json(ROOT / "references/narrative-template.json")
            narrative.update(exploration="Synthetic mixed-feature data.", preprocessing="Fold-local preparation.",
                             features="All fixture predictors retained.", model_rationale="Three software-test model families.",
                             findings="Software fixture only.", limitations="No real-world inference.")
            write_json(tmp / "narrative.json", narrative)
            manifest = generate(tmp / "dev/training_results.json", tmp / "diagnosis.json", tmp / "narrative.json",
                                tmp / "report", test_results=tmp / "out/test_results.json", lock=lock)
            self.assertEqual(manifest["main_pages"], 2)
            report = (tmp / "report/report.md").read_text(encoding="utf-8")
            for name in ("Logistic Regression", "Decision Tree", "Random Forest"):
                self.assertIn(f"| {name} | F1 (no) |", report)
            self.assertIn("Additional prespecified metrics", report)
            self.assertIn("|  | precision (no) |", report)


if __name__ == "__main__":
    unittest.main()
