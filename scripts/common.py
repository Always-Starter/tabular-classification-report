"""Shared contracts for train-only development and frozen evaluation."""
import hashlib
import importlib.metadata
import json
from io import BytesIO
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

class CalibratedSVC(ClassifierMixin, BaseEstimator):
    """SVC with explicit fold-local probability calibration and a flat grid surface."""

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", class_weight=None,
                 random_state=None, calibration_cv=3, cache_size=512):
        self.C, self.kernel, self.gamma, self.class_weight = C, kernel, gamma, class_weight
        self.random_state, self.calibration_cv, self.cache_size = random_state, calibration_cv, cache_size

    def fit(self, x, y):
        estimator = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, class_weight=self.class_weight,
                        random_state=self.random_state, cache_size=self.cache_size)
        self.calibrated_ = CalibratedClassifierCV(estimator, cv=self.calibration_cv, ensemble=False).fit(x, y)
        self.classes_ = self.calibrated_.classes_
        self.n_features_in_ = self.calibrated_.n_features_in_
        return self

    def predict(self, x):
        return self.calibrated_.predict(x)

    def predict_proba(self, x):
        return self.calibrated_.predict_proba(x)


ESTIMATORS = {
    "logistic_regression": LogisticRegression, "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier, "decision_tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier, "gaussian_nb": GaussianNB,
    "support_vector_classifier": CalibratedSVC,
}
METRICS = {"accuracy", "balanced_accuracy", "f1", "precision", "recall", "f1_macro",
           "f1_weighted", "precision_macro", "recall_macro", "average_precision",
           "roc_auc", "roc_auc_ovr_macro", "log_loss"}
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "openpyxl", "xlrd", "reportlab", "pypdf")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=native)
        stream.write("\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, default=native).encode()).hexdigest()


def environment():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "packages": versions}


def code_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def load_table(path, sheet="Data", label_column=None, raw=None):
    path = Path(path)
    dtype = {label_column: "string"} if label_column else None
    source = BytesIO(raw) if raw is not None else path
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(source, sheet_name=sheet, dtype=dtype)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(source, sep="\t" if path.suffix.lower() == ".tsv" else ",", dtype=dtype)
    else:
        raise ValueError("Supported formats: CSV, TSV, XLSX, XLS")
    if not frame.columns.is_unique or frame.empty:
        raise ValueError("Input must have unique columns and at least one row")
    return frame


def features(frame, plan):
    absent = set(plan["features"]) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing required features: {sorted(absent)}")
    x = frame[plan["features"]].copy()
    for name in plan["numeric_features"]:
        x[name] = pd.to_numeric(x[name], errors="raise").astype(float)
        if np.isinf(x[name]).any():
            raise ValueError(f"Infinite values in {name}; declare a training-derived cleaning rule first")
    for name in plan["categorical_features"]:
        x[name] = x[name].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return x


def safe_log1p(x):
    if np.any(np.asarray(x) <= -1):
        raise ValueError("log1p requires all numeric values > -1")
    return np.log1p(x)


def pipeline(plan, spec):
    cfg = spec["preprocessing"]
    num = [("imputer", SimpleImputer(strategy=cfg["numeric_imputer"], fill_value=cfg.get("numeric_fill_value", 0),
                                   add_indicator=cfg["missing_indicator"], keep_empty_features=True))]
    if cfg["numeric_transform"] == "log1p":
        num.append(("transform", FunctionTransformer(safe_log1p, feature_names_out="one-to-one")))
    if cfg["scaler"] != "none":
        num.append(("scaler", StandardScaler() if cfg["scaler"] == "standard" else RobustScaler()))
    encoder = (OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                             min_frequency=cfg.get("min_frequency"), max_categories=cfg.get("max_categories"))
               if cfg["categorical_encoder"] == "onehot" else
               OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    cat = Pipeline([("imputer", SimpleImputer(strategy=cfg["categorical_imputer"],
                                             fill_value="<MISSING>", keep_empty_features=True)),
                    ("encoder", encoder)])
    prep = ColumnTransformer([("numeric", Pipeline(num), plan["numeric_features"]),
                              ("categorical", cat, plan["categorical_features"])])
    defaults = {}
    if spec["type"] in {"logistic_regression", "random_forest", "extra_trees", "decision_tree",
                        "support_vector_classifier"}:
        defaults["random_state"] = plan["seed"]
    if spec["type"] == "logistic_regression":
        defaults["max_iter"] = 3000
    if spec["type"] in {"random_forest", "extra_trees"}:
        defaults.update(n_estimators=150, n_jobs=1)
    if spec["type"] == "support_vector_classifier":
        defaults.update(calibration_cv=3, cache_size=512)
    defaults.update(spec.get("params", {}))
    return Pipeline([("preprocess", prep), ("model", ESTIMATORS[spec["type"]](**defaults))])


def predictions(model, x, order, plan):
    classes = list(model.classes_)
    if set(classes) != set(order):
        raise ValueError("Fitted model classes do not match frozen class order")
    prob = model.predict_proba(x)[:, [classes.index(c) for c in order]]
    if plan["task"] == "binary":
        pos = plan["positive_class"]
        neg = next(c for c in order if c != pos)
        pred = np.where(prob[:, order.index(pos)] >= plan["threshold"], pos, neg)
    else:
        pred = np.asarray(order)[np.argmax(prob, axis=1)]
    return pred, prob


def score_metrics(y, pred, prob, order, plan):
    y, pred, prob = np.asarray(y), np.asarray(pred), np.asarray(prob, dtype=float)
    if not set(y) <= set(order) or not set(pred) <= set(order):
        raise ValueError("Unexpected class labels")
    if prob.shape != (len(y), len(order)) or not np.isfinite(prob).all():
        raise ValueError("Invalid probabilities")
    if (prob < 0).any() or (prob > 1).any() or not np.allclose(prob.sum(axis=1), 1, atol=1e-7):
        raise ValueError("Probabilities must be in [0,1] and sum to one")
    scores, undefined = {}, {}
    requested = [plan["metrics"]["primary"], *plan["metrics"]["secondary"]]
    for metric in requested:
        if not len(y):
            scores[metric], undefined[metric] = None, "No labelled rows"
            continue
        if metric == "accuracy":
            value = accuracy_score(y, pred)
        elif metric == "balanced_accuracy":
            if set(y) != set(order):
                scores[metric], undefined[metric] = None, "A frozen class has no labelled support"
                continue
            value = balanced_accuracy_score(y, pred)
        elif metric in {"average_precision", "roc_auc"}:
            binary_y = y == plan["positive_class"]
            if len(np.unique(binary_y)) != 2:
                scores[metric], undefined[metric] = None, "Both positive and negative labels are required"
                continue
            p = prob[:, order.index(plan["positive_class"])]
            value = (average_precision_score if metric == "average_precision" else roc_auc_score)(binary_y, p)
        elif metric == "roc_auc_ovr_macro":
            if set(y) != set(order):
                scores[metric], undefined[metric] = None, "Every frozen class must be represented"
                continue
            value = roc_auc_score(y, prob, labels=order, multi_class="ovr", average="macro")
        elif metric == "log_loss":
            value = log_loss(y, prob, labels=order)
        else:
            base, _, average = metric.partition("_")
            fn = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[base]
            kwargs = {"average": average or "binary", "zero_division": 0}
            if not average:
                actual_positive = int(np.sum(y == plan["positive_class"]))
                predicted_positive = int(np.sum(pred == plan["positive_class"]))
                denominator = {"recall": actual_positive, "precision": predicted_positive,
                               "f1": actual_positive + predicted_positive}[base]
                if denominator == 0:
                    reason = {"recall": "No actual positive labels", "precision": "No predicted positive labels",
                              "f1": "No actual or predicted positive labels"}[base]
                    scores[metric], undefined[metric] = None, reason
                    continue
                kwargs["pos_label"] = plan["positive_class"]
            else:
                kwargs["labels"] = order
            value = fn(y, pred, **kwargs)
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite score: {metric}")
        scores[metric] = float(value)
    cm = confusion_matrix(y, pred, labels=order).tolist() if len(y) else [[0] * len(order) for _ in order]
    return {"metrics": scores, "undefined_metrics": undefined, "confusion_matrix": cm,
            "class_order": order, "labelled_rows": len(y)}


class PrimaryScorer:
    def __init__(self, plan, order):
        self.plan, self.order = plan, order

    def __call__(self, estimator, x, y):
        pred, prob = predictions(estimator, x, self.order, self.plan)
        metric = self.plan["metrics"]["primary"]
        value = score_metrics(y, pred, prob, self.order, self.plan)["metrics"][metric]
        if value is None:
            raise ValueError(f"Primary metric {metric} undefined in a training validation fold")
        return -value if metric == "log_loss" else value


def splits(frame, y, plan, n_splits, seed):
    cfg = plan["cv"]
    if cfg["strategy"] == "stratified":
        if pd.Series(y).value_counts().min() < n_splits:
            raise ValueError("Too few examples per class for the requested folds")
        pairs = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(frame, y)
    elif cfg["strategy"] == "stratified_group":
        groups = frame[cfg["group_column"]]
        if groups.isna().any() or groups.nunique() < n_splits:
            raise ValueError("Group splitting requires nonmissing groups and enough distinct groups")
        pairs = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(frame, y, groups)
    else:
        times = pd.to_datetime(frame[cfg["time_column"]], errors="raise", utc=True)
        if times.isna().any():
            raise ValueError("Time splitting requires nonmissing timestamps")
        unique = np.sort(times.unique())
        pairs = ((np.flatnonzero(times.isin(unique[a])), np.flatnonzero(times.isin(unique[b])))
                 for a, b in TimeSeriesSplit(n_splits=n_splits, gap=cfg.get("gap", 0)).split(unique))
    result = list(pairs)
    order = set(y)
    for train, valid in result:
        if set(np.asarray(y)[train]) != order or set(np.asarray(y)[valid]) != order:
            raise ValueError("Every inner/outer train and validation fold must contain all classes; revise CV using training data")
    return result
