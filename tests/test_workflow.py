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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import (ESTIMATORS, features, load_table, pipeline, predictions, read_json,
                    score_metrics, sha, splits, write_json)
from validate_plan import validate
from run_nested_cv import numeric_grid_boundaries, run
from freeze_model_lock import freeze
from approve_model_lock import approve
from evaluate_holdout import evaluate
from verify_results import verify, verify_training
from diagnose_training import diagnose
from generate_report import generate


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
    def test_numeric_grid_boundaries_only_report_uncertainty(self):
        grid = {"model__C": [0.1, 1.0, 10.0], "model__criterion": ["gini", "entropy"],
                "model__single": [7]}
        upper = numeric_grid_boundaries(grid, {"model__C": 10.0,
                                               "model__criterion": "gini", "model__single": 7})
        self.assertEqual(upper, {"model__C": {"selected": 10.0, "edge": "upper",
                                               "evaluated_min": 0.1, "evaluated_max": 10.0}})
        self.assertEqual(numeric_grid_boundaries(grid, {"model__C": 1.0,
                                                         "model__criterion": "gini", "model__single": 7}), {})

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
        p = example(); p["models"] = p["models"][:1]; cases.append(p)
        p = example(); p["models"].extend([copy.deepcopy(p["models"][0]), copy.deepcopy(p["models"][1])]); cases.append(p)
        p = example(); p["models"].append(copy.deepcopy(p["models"][0])); p["models"][2]["name"] = "third"; cases.append(p)
        for p in cases:
            with self.subTest(plan=p), self.assertRaises((ValueError, TypeError)):
                validate(p)

    def test_three_models_require_a_reason(self):
        p = example()
        third = copy.deepcopy(p["models"][1])
        third["name"] = "forest"
        third["type"] = "random_forest"
        third["params"] = {"n_estimators": 10}
        third["grid"] = {}
        p["models"].append(third)
        p["model_count_rationale"] = "Training diagnosis motivates an additional nonlinear ensemble comparison."
        self.assertEqual(len(validate(p)["models"]), 3)

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
        p["cv"] = {"strategy": "stratified_group", "outer_splits": 3, "inner_splits": 2, "group_column": "entity"}
        p["excluded_features"] = {"entity": "Repeated entity; group validation"}
        validate(p)
        for a, b in splits(d, y, p, 3, 1):
            self.assertFalse(set(d.iloc[a].entity) & set(d.iloc[b].entity))
        d["time"] = np.repeat(pd.date_range("2020-01-01", periods=30), 4)
        p["cv"] = {"strategy": "time", "outer_splits": 3, "inner_splits": 2, "time_column": "time", "gap": 1}
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
        cls.plan["sensitivities"] = [{"name": "threshold_check", "rationale": "Predeclared operating-point uncertainty", "overrides": {"threshold": .65}}]
        cls.plan_path = cls.base / "plan.json"
        write_json(cls.plan_path, cls.plan)
        cls.development = cls.base / "development"
        cls.results = run(cls.train, cls.plan_path, cls.development)
        cls.review = cls.base / "review.json"
        write_json(cls.review, {"selected_variant": "baseline", "preferred_model": "linear", "rationale": "Synthetic comparison for software testing only",
                               "sensitivity_review": "Keep baseline; threshold alternative is a software fixture", "warnings_review": "Reviewed synthetic run warnings"})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def new_lock(self, directory):
        path = Path(directory) / "model-lock.json"
        freeze(self.development / "training_results.json", self.development, self.review, path)
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
        self.assertFalse(self.results["test_data_accessed"])
        for v in self.results["variants"].values():
            for model in v["models"].values():
                self.assertEqual(model["oof_rows"], 120)
                self.assertIn("tuning_boundary", model)
                self.assertTrue(all("tuning_boundary" in fold for fold in model["fold_results"]))
        m = joblib.load(self.development / "baseline/linear.joblib")
        learned = m.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_
        self.assertTrue(np.allclose(learned, frame()[["x1", "x2"]].median().to_numpy()))

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

    def test_every_predictor_needs_a_selection_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); d = frame(); d["forgotten_column"] = np.arange(len(d))
            d.to_csv(tmp / "train.csv", index=False)
            with patch.object(SimpleImputer, "fit", side_effect=AssertionError("Must not fit")), self.assertRaisesRegex(ValueError, "unaccounted.*forgotten_column"):
                run(tmp / "train.csv", self.plan_path, tmp / "out")

    def test_unapproved_refuses_before_test_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = self.new_lock(tmp)
            with patch("evaluate_holdout.load_table", side_effect=AssertionError("Must not read")), self.assertRaisesRegex(ValueError, "approval"):
                evaluate(Path(tmp) / "does-not-exist.csv", lock, self.development, Path(tmp) / "out")
            self.assertFalse(lock.with_name(lock.name + ".holdout.json").exists())

    def test_bad_digest_and_tampered_lock_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = self.new_lock(tmp)
            with self.assertRaisesRegex(ValueError, "digest"):
                approve(lock, "0" * 64, "reviewer", "approved")
            self.approve_fixture(lock)
            payload = read_json(lock); payload["plan"]["threshold"] = .9; write_json(lock, payload)
            with self.assertRaisesRegex(ValueError, "Approval"):
                evaluate(Path(tmp) / "absent.csv", lock, self.development, Path(tmp) / "out")

    def test_tampered_model_refuses_before_test_read(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            lock = self.new_lock(tmp); self.approve_fixture(lock)
            copied = Path(tmp) / "models"; shutil.copytree(self.development, copied)
            with (copied / "baseline/linear.joblib").open("ab") as stream:
                stream.write(b"tampering")
            with self.assertRaisesRegex(ValueError, "hash/path mismatch"):
                evaluate(Path(tmp) / "absent.csv", lock, copied, Path(tmp) / "out")

    def test_changed_code_refuses_before_test_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = self.new_lock(tmp); self.approve_fixture(lock)
            with patch("evaluate_holdout.code_hashes", return_value={}), self.assertRaisesRegex(ValueError, "Code/environment"):
                evaluate(Path(tmp) / "absent.csv", lock, self.development, Path(tmp) / "out")

    def test_corrupt_predictions_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); lock = self.new_lock(tmp); self.approve_fixture(lock)
            evaluate(self.holdout(tmp), lock, self.development, tmp / "out")
            path = tmp / "out/linear_predictions.json"
            content = read_json(path); content["records"][0]["prediction"] = "tampered"; write_json(path, content)
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify(tmp / "out/test_results.json", lock)

    def test_holdout_verify_repeat_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); lock = self.new_lock(tmp); self.approve_fixture(lock)
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
            write_json(tmp / "narrative.json", narrative)
            manifest = generate(self.development / "training_results.json", tmp / "diagnosis.json", tmp / "narrative.json", tmp / "report", test_results=tmp / "out/test_results.json", lock=lock)
            self.assertEqual(manifest["main_pages"], 2)
            self.assertTrue(manifest["draft"])
            self.assertGreaterEqual(manifest["total_pages"], 3)
            report = (tmp / "report/report.md").read_text(encoding="utf-8")
            self.assertIn("Tuning stopped after the prespecified inner-CV search", report)
            self.assertNotIn("SHA-256 evidence accompany this report", report)
            # Verification uses stored predictions, even when the test file is no longer available.
            test.rename(tmp / "sealed-away.tsv")
            self.assertTrue(verify(tmp / "out/test_results.json", lock)["verified"])

    def test_unlabelled_and_single_class_holdout(self):
        for mode in ("unlabelled", "one-class"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp); lock = self.new_lock(tmp); self.approve_fixture(lock)
                test = self.holdout(tmp, labelled=mode != "unlabelled")
                if mode == "one-class":
                    d = load_table(test); d["label"] = "no"; d.to_csv(test, sep="\t", index=False)
                result = evaluate(test, lock, self.development, tmp / "out")
                self.assertIsNone(result["models"]["linear"]["metrics"]["roc_auc"])
                self.assertTrue(verify(tmp / "out/test_results.json", lock)["verified"])

    def test_missing_feature_consumes_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); lock = self.new_lock(tmp); self.approve_fixture(lock)
            test = tmp / "heldout.csv"; frame(30).drop(columns="x1").to_csv(test, index=False)
            with self.assertRaisesRegex(ValueError, "Missing required"):
                evaluate(test, lock, self.development, tmp / "out")
            self.assertEqual(read_json(lock.with_name(lock.name + ".holdout.json"))["status"], "failed_after_access_reserved")
            with self.assertRaises(FileExistsError):
                evaluate(test, lock, self.development, tmp / "out2")

    def test_multiclass_full_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); p = example(); p["task"] = "multiclass"
            del p["positive_class"], p["threshold"]
            p["metrics"] = {"primary": "f1_macro", "secondary": ["accuracy", "log_loss", "roc_auc_ovr_macro"]}
            train = tmp / "train.csv"; frame(120, True).to_csv(train, index=False)
            write_json(tmp / "plan.json", p)
            run(train, tmp / "plan.json", tmp / "dev")
            lock = tmp / "model-lock.json"
            freeze(tmp / "dev/training_results.json", tmp / "dev", self.review, lock)
            self.approve_fixture(lock)
            test = tmp / "test.csv"; frame(30, True).to_csv(test, index=False)
            result = evaluate(test, lock, tmp / "dev", tmp / "out")
            self.assertEqual(result["class_order"], ["alpha", "beta", "gamma"])
            self.assertEqual(len(result["models"]["linear"]["confusion_matrix"]), 3)
            self.assertTrue(verify(tmp / "out/test_results.json", lock)["verified"])

    def test_three_model_full_workflow_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p = example()
            third = copy.deepcopy(p["models"][1])
            third.update(name="forest", type="random_forest", params={"n_estimators": 10}, grid={})
            p["models"].append(third)
            p["model_count_rationale"] = "Synthetic three-model software check; no course-data recommendation."
            train = tmp / "train.csv"
            frame().to_csv(train, index=False)
            write_json(tmp / "plan.json", p)
            result = run(train, tmp / "plan.json", tmp / "dev")
            self.assertEqual(set(result["variants"]["baseline"]["models"]), {"linear", "tree", "forest"})
            review = tmp / "review.json"
            write_json(review, {"selected_variant": "baseline", "preferred_model": "linear",
                                "rationale": "Synthetic test", "sensitivity_review": "None declared",
                                "warnings_review": "Reviewed synthetic warnings"})
            lock = tmp / "model-lock.json"
            freeze(tmp / "dev/training_results.json", tmp / "dev", review, lock)
            self.approve_fixture(lock)
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
            for name in ("linear", "tree", "forest"):
                self.assertIn(f"{name} / f1", report)
            self.assertIn("Additional prespecified metrics", report)


if __name__ == "__main__":
    unittest.main()
