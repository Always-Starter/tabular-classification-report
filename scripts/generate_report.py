#!/usr/bin/env python3
"""Generate traceable Markdown and a PDF with optional cover, two-page report and Reflection."""
import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape
from common import code_hashes, effective_fixed_params, execution_controls, read_json, sha, threshold_policy, write_json
from verify_results import verify, verify_training


def readable_value(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "not set"
    if isinstance(value, dict):
        return ", ".join(f"{key}: {readable_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(readable_value(item) for item in value)
    return str(value)


def settings_text(settings):
    return "; ".join(f"{key.removeprefix('model__').replace('_', ' ')}: {readable_value(value)}"
                     for key, value in settings.items()) or "default settings"


def sentence_fragment(value):
    """Return prose suitable for joining before renderer-supplied punctuation."""
    return str(value).strip().rstrip(".;")


def decision_basis_text(basis):
    """Render schema decision bases without implying a library default."""
    labels = {
        "conventional_default": "conventional baseline",
        "dataset_specific_evidence": "dataset-specific evidence",
        "user_domain_constraint": "user/domain constraint",
        "invariant_rule": "invariant methodological rule",
    }
    return labels.get(basis, basis.replace("_", " "))


def model_label(spec):
    labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "decision_tree": "Decision Tree",
        "knn": "k-Nearest Neighbors",
        "gaussian_nb": "Gaussian Naive Bayes",
        "support_vector_classifier": "Support Vector Classifier",
    }
    return labels.get(spec["type"], spec["type"].replace("_", " ").title())


def preprocessing_text(config):
    numeric = config["numeric_imputer"].replace("_", "-")
    categorical = config["categorical_imputer"].replace("_", "-")
    encoder = "one-hot" if config["categorical_encoder"] == "onehot" else "ordinal"
    parts = [f"{numeric} numeric imputation", f"{categorical} categorical imputation", f"{encoder} encoding"]
    if config["missing_indicator"]:
        parts.append("missing-value indicators")
    if config["numeric_transform"] != "none":
        parts.append(f"{config['numeric_transform']} numeric transform")
    if config["scaler"] != "none":
        parts.append(f"{config['scaler']} scaling")
    for key in ("numeric_fill_value", "min_frequency", "max_categories"):
        if key in config:
            parts.append(f"{key.replace('_', ' ')}: {readable_value(config[key])}")
    return "; ".join(parts)


def fixed_param_rationale_text(spec, effective=None):
    """Summarise why fixed parameters were not tuned and where exact values came from."""
    effective = spec["params"] if effective is None else effective
    if not effective:
        return "No estimator parameters were fixed"
    records = spec.get("fixed_param_rationale")
    if records is None:
        return "Fixed-parameter provenance was not recorded in this legacy plan"
    parts = []
    for parameter in effective:
        label = parameter.replace("_", " ")
        record = records.get(parameter)
        if record is None:
            parts.append(f"{label} - runner-fixed exact-value provenance missing in this legacy plan")
            continue
        if record["value_source"] == "unknown":
            value_basis = "exact-value source unknown"
        else:
            source = record["value_source"].replace("_", " ")
            value_basis = f"exact-value source {source}: {sentence_fragment(record['value_rationale'])}"
        parts.append(f"{label} - {value_basis}; not tuned: {sentence_fragment(record['not_tuned_reason'])}")
    return "Fixed-parameter rationale: " + " | ".join(parts)


def tuning_boundary_text(boundary, grid):
    """Distinguish inevitable two-value endpoints from edges selected over interior candidates."""
    parts = []
    for parameter, details in boundary.items():
        label = parameter.removeprefix("model__").replace("_", " ")
        count = details.get("distinct_values_evaluated")
        if count is None:
            values = grid.get(parameter, [])
            count = len(set(values)) if all(isinstance(value, (int, float)) and not isinstance(value, bool)
                                            for value in values) else None
        interior = details.get("interior_candidates_evaluated", bool(count and count > 2))
        selected = details["selected"]
        edge = details["edge"]
        if count == 2 and not interior:
            parts.append(f"{label}={selected} selected the {edge} endpoint of a two-value grid; endpoint selection "
                         "was inevitable and indicates coarse search coverage, not evidence that the optimum lies "
                         "outside the evaluated range")
        else:
            parts.append(f"{label}={selected} selected the {edge} edge after interior candidates were evaluated; "
                         "this supports an untested-direction question only for a future independent run")
    return ". ".join(parts)


def threshold_policy_text(plan, models):
    policy = threshold_policy(plan)
    if not policy:
        return "Multiclass prediction uses maximum predicted probability (argmax)."
    if policy["mode"] == "fixed":
        basis = decision_basis_text(policy["value_source"])
        rationale = sentence_fragment(policy["value_rationale"] or "exact-value rationale unknown")
        not_tuned = sentence_fragment(policy["not_tuned_reason"])
        return (f"Fixed threshold {policy['value']} was predeclared before development; basis: {basis}; "
                f"rationale: {rationale}; not tuned: {not_tuned}")
    if policy["mode"] == "model_default":
        return ("The predeclared model-default class decision was retained rather than threshold-tuned; "
                f"{policy['not_tuned_reason']}")
    selected = ", ".join(f"{name}: {details.get('selected_threshold')}" for name, details in models.items())
    return (f"Thresholds were jointly selected with model parameters inside inner CV only using "
            f"{policy['objective'].replace('_', ' ')}; final full-training selections: {selected}. "
            "Held-out data did not select or revise them")


def model_selection_text(selection, model_labels=None):
    if not selection:
        return "Legacy run: executable final-family selection evidence was not recorded."
    model_labels = model_labels or {}
    policy = selection["policy"]
    metric = selection["metric"].replace("_", "-")
    tolerance = policy["practical_tie_tolerance"]
    contenders = selection["practical_tie_contenders"]
    candidates = {item["model"]: item for item in selection["candidates"]}
    selected = selection["selected_model"]

    def label(name):
        return model_labels.get(name, name.replace("_", " ").capitalize())

    tie_breaker_labels = {
        "lower_outer_std": "lower outer-fold SD",
        "declared_preference_order": "declared model-preference order",
    }
    applied = [tie_breaker_labels.get(item, item.replace("_", " "))
               for item in selection["applied_tie_breakers"]]

    if len(contenders) == 1:
        item = candidates[selected]
        return (f"Under the predeclared outer-CV {metric} policy, {label(selected)} alone fell within the "
                f"practical-tie tolerance of {tolerance:g} (mean {item['mean']:.5f}, SD {item['std']:.5f}) "
                "and was selected without a tie-breaker.")

    if len(contenders) == 2:
        other = next(name for name in contenders if name != selected)
        selected_item, other_item = candidates[selected], candidates[other]
        difference = abs(selected_item["mean"] - other_item["mean"])
        if applied == ["lower outer-fold SD"]:
            selection_reason = (f"the predeclared lower outer-fold SD tie-breaker "
                                f"({selected_item['std']:.5f} vs {other_item['std']:.5f})")
        else:
            selection_reason = "the predeclared " + " and then ".join(applied) + " tie-breaker"
        return (f"The outer-CV mean {metric} difference between {label(selected)} and {label(other)} was "
                f"{difference:.5f}, within the predeclared practical-tie tolerance of {tolerance:g}. "
                f"{label(selected)} was therefore selected using {selection_reason}.")

    contender_text = ", ".join(
        f"{label(name)} (mean {candidates[name]['mean']:.5f}, SD {candidates[name]['std']:.5f})"
        for name in contenders
    )
    selection_reason = " and then ".join(applied) if applied else "no tie-breaker"
    return (f"Under the predeclared outer-CV {metric} policy, these models were within the practical-tie "
            f"tolerance of {tolerance:g}: {contender_text}. {label(selected)} was selected using the "
            f"predeclared {selection_reason} rule.")


def confusion_text(matrix, classes, positive_class=None):
    if len(classes) == 2 and positive_class in classes:
        pos = classes.index(positive_class)
        neg = 1 - pos
        return (f"Positive class: {positive_class}. True negatives: {matrix[neg][neg]:,}; "
                f"false positives: {matrix[neg][pos]:,}; false negatives: {matrix[pos][neg]:,}; "
                f"true positives: {matrix[pos][pos]:,}.")
    return "Actual to predicted counts: " + "; ".join(
        f"{actual} -> " + ", ".join(f"{predicted}: {matrix[row][col]:,}"
                                      for col, predicted in enumerate(classes))
        for row, actual in enumerate(classes)) + "."


def validate_distribution_language(narrative, diagnosis):
    """Reject distribution claims that overstate the saved training-only EDA."""
    fields = ("exploration", "preprocessing", "features", "model_rationale", "findings", "limitations")
    summary = diagnosis.get("numeric_distribution_summary", {})
    material_skew = set(summary.get("material_skewness_columns", []))
    if not material_skew:  # Backward-compatible evidence extraction for legacy diagnoses.
        material_skew = {
            column for column, details in diagnosis.get("numeric", {}).items()
            if details.get("skew") is not None and abs(details["skew"]) >= 1.0
        }
    high_kurtosis = set(summary.get("high_excess_kurtosis_columns", []))
    hedges = ("suggest", "consistent with", "descriptive", "signal", "may", "possible", "potential")
    outlier_qualifiers = ("potential", "candidate", "flagged", "statistical", "possible", "suspected",
                          "unverified", "not established", "not confirmed", "unknown")

    for field in fields:
        for sentence in re.split(r"(?<=[.!?])\s+", str(narrative.get(field, ""))):
            lowered = sentence.lower()
            if re.search(r"\btails? differ\b", lowered):
                raise ValueError("Narrative claim 'tails differ' is too vague; cite a saved skewness, excess-kurtosis, "
                                 "or IQR-fence result instead")
            if re.search(r"\b(?:heavy|thick)[ -]?tails?\b|\bheavy[ -]?tailed\b", lowered):
                if not high_kurtosis or "excess kurtosis" not in lowered or not any(term in lowered for term in hedges):
                    raise ValueError("Heavy-tail language requires saved high excess-kurtosis evidence and qualified "
                                     "wording; the diagnosis does not prove a heavy-tailed distribution")
            if re.search(r"\b(?:strong|pronounced|material|marked|significant) skew(?:ness|ed)?\b", lowered):
                if not material_skew:
                    raise ValueError("Material-skewness language requires a column meeting the declared skew threshold")
            if "outlier" in lowered:
                describes_method_robustness = any(term in lowered for term in ("sensitive to outlier", "robust to outlier"))
                if not describes_method_robustness and not any(term in lowered for term in outlier_qualifiers):
                    raise ValueError("Dataset outliers must be described as potential/statistical flags unless domain "
                                     "review has independently confirmed their status")


def generate(training, diagnosis, narrative, output_dir, variant="baseline", test_results=None, lock=None,
             font=None, presentation_only_rerender=False):
    import reportlab
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from pypdf import PdfReader

    training, diagnosis, narrative, output_dir = map(Path, (training, diagnosis, narrative, output_dir))
    tr, dg, n = read_json(training), read_json(diagnosis), read_json(narrative)
    verify_training(training)
    if dg["training_source"]["sha256"] != tr["training_source"]["sha256"]:
        raise ValueError("Diagnosis and modelling used different training files")
    if test_results:
        if not lock:
            raise ValueError("A lock is required for a held-out report")
        lk = read_json(lock)
        locked_hashes, current_hashes = lk["code_sha256"], code_hashes()
        changed_scripts = {name for name in locked_hashes.keys() | current_hashes.keys()
                           if locked_hashes.get(name) != current_hashes.get(name)}
        if changed_scripts - {Path(__file__).name}:
            raise ValueError(f"Non-renderer code changed since Model Lock: {sorted(changed_scripts - {Path(__file__).name})}")
        renderer_changed = Path(__file__).name in changed_scripts
        if renderer_changed and not presentation_only_rerender:
            raise ValueError("Report renderer changed since Model Lock; use --presentation-only-rerender "
                             "to make a clearly recorded report-only derivative from saved results")
        verification = verify(test_results, lock)
        if lk["training_results_sha256"] != sha(training):
            raise ValueError("Lock refers to different training results")
        variant = lk["review"]["selected_variant"]
        test = read_json(test_results)
    else:
        verification, test, renderer_changed = None, None, False
    if variant != "baseline" and tr["variants"][variant].get("selection_eligible") is False:
        raise ValueError("Sensitivity variants are interpretive only and cannot be selected for a report. "
                         "Adopt the configuration through a new independent baseline run.")
    vr = tr["variants"][variant]
    p = vr["plan"]
    text_keys = ("exploration", "preprocessing", "features", "model_rationale", "findings", "limitations")
    if any(not isinstance(n.get(k), str) or not n[k].strip() for k in text_keys):
        raise ValueError(f"Narrative must include nonempty text for {text_keys}")
    validate_distribution_language(n, dg)
    metadata_keys = ("full_name", "matric_number", "llm_model_version", "llm_interface", "repository_url")
    meta = n.get("metadata", {})
    missing = [k for k in metadata_keys if not str(meta.get(k, "")).strip()]
    if "reflection" not in n or not isinstance(n["reflection"], dict):
        raise ValueError("Reflection draft is required")
    reflection_keys = ("human_oversight", "challenged_decision", "manual_verification", "future_changes")
    if any(not str(n["reflection"].get(k, "")).strip() for k in reflection_keys):
        raise ValueError("Reflection needs oversight, challenge, manual verification and future changes; use explicit pending notes when unknown")
    if type(n.get("human_reflection_confirmed")) is not bool:
        raise ValueError("Declare human_reflection_confirmed true/false; never invent human verification")
    evidence_keys = ("fact", "interpretation", "limitation_unknown", "decision", "future_work")
    evidence_summary = n.get("evidence_summary")
    if p.get("schema_version", 2) >= 5:
        if not isinstance(evidence_summary, dict) or set(evidence_summary) != set(evidence_keys) or any(
                not isinstance(evidence_summary[key], str) or not evidence_summary[key].strip() for key in evidence_keys):
            raise ValueError("Schema-v5+ reports require fact/interpretation/limitation_unknown/decision/future_work evidence_summary")
    draft = bool(missing) or not n["human_reflection_confirmed"]
    metric = p["metrics"]["primary"]
    selection = tr.get("model_selection")
    model_labels = {spec["name"]: model_label(spec) for spec in p["models"]}
    if missing and not n["human_reflection_confirmed"]:
        draft_note = " | DRAFT - metadata and Reflection pending"
    elif missing:
        draft_note = " | DRAFT - metadata pending"
    elif not n["human_reflection_confirmed"]:
        draft_note = " | DRAFT - Reflection pending"
    else:
        draft_note = ""
    identification_cover = bool(n.get("identification_cover_page", False))
    report_first_page = 1 if identification_cover else 0
    report_second_page = report_first_page + 1
    reflection_group = report_second_page + 1
    sections = [[] for _ in range(reflection_group + 1)]
    def add(page, title, text):
        sections[page].append((title, str(text)))

    if identification_cover:
        identification_rows = [
            ["Full name", meta.get("full_name") or "[not supplied]"],
            ["Matriculation number", meta.get("matric_number") or "[not supplied]"],
        ]
        skill_rows = [
            ["LLM model/version", meta.get("llm_model_version") or "[not supplied]"],
            ["Interface", meta.get("llm_interface") or "[not supplied]"],
            ["Skill repository", meta.get("repository_url") or "[not supplied]"],
        ]
        add(0, "IN6227-Assignment-1", "Variant-2")
        add(0, "", "\n".join(
            f"{key}: {value}" for key, value in identification_rows))
        add(report_first_page, "Tabular classification report", "")
    else:
        identification_rows = []
        skill_rows = []
        add(report_first_page, "Tabular classification report", "IN6227-Assignment-1 | Variant-2" + draft_note)
    if test:
        preferred = lk["review"]["preferred_model"]
        if preferred not in vr["models"]:
            raise ValueError("Preferred Model Lock model is absent from selected variant")
        cv_score = vr["models"][preferred]["outer_summary"][metric]["mean"]
        held_score = test["models"][preferred]["metrics"][metric]
        cv_display = "N/A" if cv_score is None else f"{cv_score:.4f}"
        held_display = "N/A" if held_score is None else f"{held_score:.4f}"
        summary_emphasis = (f"Frozen pre-test choice: {model_labels[preferred]}. "
                            f"Nested-CV {metric.replace('_', ' ')}: {cv_display}; "
                            f"held-out: {held_display}.")
        summary = summary_emphasis + " The model choice was not revised after held-out evaluation."
    else:
        selected = selection["selected_model"] if selection else None
        summary_emphasis = (f"Development-only prespecified choice: {model_labels[selected]}. "
                            f"Primary measure: {metric.replace('_', ' ')}."
                            if selected else
                            f"Development-only comparison. Primary measure: {metric.replace('_', ' ')}.")
        summary = summary_emphasis + " No independent held-out result is claimed."
    add(report_first_page, "At a glance", summary)
    submission_rows = [
        ["Name", meta.get("full_name") or "[not supplied]",
         "Matriculation number", meta.get("matric_number") or "[not supplied]"],
        ["LLM model/version", meta.get("llm_model_version") or "[not supplied]",
         "Interface", meta.get("llm_interface") or "[not supplied]"],
        ["Skill repository", meta.get("repository_url") or "[not supplied]", "", ""],
    ]
    if identification_cover:
        add(report_first_page, "Skill information", "\n".join(f"{key}: {value}" for key, value in skill_rows))
    else:
        add(report_first_page, "Submission details", "\n".join(
            f"{row[0]}: {row[1]}" + (f" | {row[2]}: {row[3]}" if row[2] else "")
            for row in submission_rows))
    class_counts = "; ".join(f"{label}: {count:,}" for label, count in tr["class_counts"].items())
    add(report_first_page, "Data exploration and cleaning", f"Training: {tr['input_rows']:,} rows; {tr['eligible_rows']:,} labelled rows; "
        f"{tr['missing_targets']} missing targets excluded. Target: {p['target']}; class counts: {class_counts}. "
        f"Predictors retained: {len(p['features'])}. Exact duplicate rows: {dg['duplicates']['exact_duplicate_rows']}.\n" + n["exploration"])
    add(report_first_page, "Preprocessing and feature decisions", n["preprocessing"] + " " + n["features"])
    for spec in p["models"]:
        model = vr["models"][spec["name"]]
        effective = model.get("effective_fixed_params", effective_fixed_params(p, spec))
        boundary = model.get("tuning_boundary", {})
        imbalance = spec.get("imbalance_handling")
        imbalance_labels = {
            "fixed_class_weight": "fixed, predeclared class weighting",
            "tuned_class_weight": "tuned class-weighting strategy",
            "none": "none",
        }
        add(report_first_page, f"Model: {model_labels[spec['name']]}", "Preparation: " + preprocessing_text(spec["preprocessing"])
            + (f". Imbalance handling: {imbalance_labels.get(imbalance['strategy'], imbalance['strategy'].replace('_', ' '))}"
               + (f"; basis: {decision_basis_text(imbalance['basis'])}" if imbalance["strategy"] != "fixed_class_weight" else "")
               if imbalance else "")
            + ". Selected settings: " + settings_text(model["best_params"])
            + ". Fixed settings: " + settings_text(effective)
            + ". Execution controls: " + settings_text(
                model.get("execution_controls", execution_controls(p, spec))) + ". "
            + fixed_param_rationale_text(spec, effective) + "."
            + (" Search-boundary interpretation: " + tuning_boundary_text(boundary, spec["grid"]) + "."
               if boundary else ""))
    add(report_first_page if len(p["models"]) == 3 else report_second_page,
        "Model choice and stopping rules", n["model_rationale"]
        + f" The declared search used {tr['planned_fits']} planned fits across baseline and sensitivity analyses; "
        + "outer folds did not trigger grid expansion or a change of primary metric. Sensitivity variants were "
        + "interpreted as robustness evidence only and were not eligible for selection or locking. "
        + model_selection_text(selection, model_labels))
    add(report_second_page, "Evaluation and comparison", f"Primary metric: {metric.replace('_', ' ')} "
        f"({'lower' if metric == 'log_loss' else 'higher'} is better). "
        f"Nested CV: {p['cv']['strategy']}, {p['cv']['outer_splits']} outer / {p['cv']['inner_splits']} inner folds; seed {p['seed']}. "
        "All learned preprocessing is fitted within folds. All classifiers share validation splits. "
        + (f"Binary positive class: {p['positive_class']}. " if p["task"] == "binary" else "")
        + threshold_policy_text(p, vr["models"]) + ". "
        + (f"Held-out evaluation: {test['test_rows']:,} rows, {test['labelled_rows']:,} labelled, {test['missing_labels']} unlabelled. "
           if test else "No held-out evaluation performed; these are development estimates. ")
        + "Fold SD measures variability, not a confidence interval. The baseline alone was selection-eligible; "
        + "adopting a sensitivity would require a new independent plan/run.")
    # The printed metric set is fixed by the pre-test plan order, never selected from
    # held-out performance. Full scores for every declared metric remain in evidence JSON.
    displayed_metrics = [metric, *p["metrics"]["secondary"][:3]]
    omitted_metrics = p["metrics"]["secondary"][3:]
    if omitted_metrics:
        add(report_second_page, "Additional prespecified metrics", "Full fold and held-out results for "
            + ", ".join(omitted_metrics) + " are retained in the verified result files; "
            "the compact PDF table shows the primary and first three secondary metrics in plan order.")
    table = [["Model", "Metric", "Nested CV mean (SD)", "Held-out"]]
    primary_rows = []
    for name, model in vr["models"].items():
        for index, m in enumerate(displayed_metrics):
            summary = model["outer_summary"][m]
            value = test["models"][name]["metrics"][m] if test else None
            development_score = "N/A" if summary["mean"] is None else f"{summary['mean']:.3f} ({summary['std']:.3f})"
            label = {"f1": "F1", "roc_auc": "ROC AUC", "roc_auc_ovr_macro": "ROC AUC (OvR macro)",
                     "average_precision": "Average precision"}.get(m, m.replace("_", " "))
            if m in {"f1", "precision", "recall"} and p["task"] == "binary":
                label += f" ({p['positive_class']})"
            table.append([model_labels[name] if index == 0 else "", label,
                          development_score, "N/A" if value is None else f"{value:.3f}"])
            if index == 0:
                primary_rows.append(len(table) - 1)
    undefined_development = {name: sorted({reason for fold in model["fold_results"] for reason in fold["undefined_metrics"].values()})
                             for name, model in vr["models"].items() if any(f["undefined_metrics"] for f in model["fold_results"])}
    if undefined_development:
        reasons = "; ".join(f"{name}: {', '.join(values)}" for name, values in undefined_development.items())
        add(report_second_page, "Undefined development scores", reasons
            + ". A secondary-metric summary is N/A if any fold is undefined; no partial-fold average is substituted.")
    error_table = None
    if test:
        if len(test["class_order"]) == 2 and p.get("positive_class") in test["class_order"]:
            pos = test["class_order"].index(p["positive_class"])
            neg = 1 - pos
            add(report_second_page, "Prediction errors (confusion matrix)",
                f"Positive class: {p['positive_class']}. "
                "TN = true negatives, FP = false positives, FN = false negatives, TP = true positives.")
            error_table = [["Model", "TN", "FP", "FN", "TP"]]
            for name, model in test["models"].items():
                matrix = model["confusion_matrix"]
                error_table.append([model_labels[name], f"{matrix[neg][neg]:,}", f"{matrix[neg][pos]:,}",
                                    f"{matrix[pos][neg]:,}", f"{matrix[pos][pos]:,}"])
                if model["undefined_metrics"]:
                    add(report_second_page, f"Undefined held-out metrics: {model_labels[name]}", settings_text(model["undefined_metrics"]))
        else:
            for name, model in test["models"].items():
                add(report_second_page, f"Prediction errors (confusion matrix): {model_labels[name]}",
                    confusion_text(model["confusion_matrix"], test["class_order"], p.get("positive_class"))
                    + (" Undefined metrics: " + settings_text(model["undefined_metrics"]) + "."
                       if model["undefined_metrics"] else ""))
    if evidence_summary:
        add(report_second_page, "Findings and discussion", "FACT: " + evidence_summary["fact"]
            + " INTERPRETATION: " + evidence_summary["interpretation"]
            + " DECISION: " + evidence_summary["decision"])
        add(report_second_page, "Limitations", "LIMITATION/UNKNOWN: " + evidence_summary["limitation_unknown"]
            + " FUTURE WORK: " + evidence_summary["future_work"]
            + f" Selection-eligible development baseline: {variant}.")
    else:
        add(report_second_page, "Findings and discussion", n["findings"])
        add(report_second_page, "Limitations", n["limitations"] + f" Selection-eligible development baseline: {variant}.")
    reflection_title = n.get("reflection_title", "Reflection - outside the two-page report limit")
    reflection_subtitle = n.get(
        "reflection_subtitle",
        "Human review draft" if not n["human_reflection_confirmed"] else "Human-confirmed Reflection",
    )
    add(reflection_group, reflection_title, reflection_subtitle)
    reflection_titles = {
        "human_oversight": "Human in the Loop",
        "challenged_decision": "Critical Evaluation",
        "manual_verification": "Trustworthiness",
        "future_changes": "Future Changes",
    }
    for k in reflection_keys:
        add(reflection_group, reflection_titles[k], n["reflection"][k])
    # Arithmetic aids stay in the manifest; the student's Reflection must describe
    # a check they actually performed, not a machine-generated verification claim.

    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / f).exists() for f in ("report.pdf", "report.md", "report_manifest.json")):
        raise ValueError("Report files already exist; use a fresh output directory")
    # Embed an actual font, rather than relying on optional viewer CJK language packs.
    font_path = Path(font) if font else Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
    embedded = TTFont("ReportFont", str(font_path))
    pdfmetrics.registerFont(embedded)
    all_text = ("".join(title + text for blocks in sections for title, text in blocks)
                + "".join(cell for row in table for cell in row)
                + "".join(cell for row in (error_table or []) for cell in row))
    missing_glyphs = sorted({c for c in all_text if not c.isspace() and ord(c) not in embedded.face.charToGlyph})
    if missing_glyphs:
        raise ValueError(f"Report font lacks glyphs {missing_glyphs[:12]}; pass --font /path/to/a/covering.ttf before rendering")
    bold_name = "ReportFont"
    if not font:
        pdfmetrics.registerFont(TTFont("ReportFontBold", str(font_path.with_name("VeraBd.ttf"))))
        bold_name = "ReportFontBold"
    ink = colors.HexColor("#213446")
    blue = colors.HexColor("#173d5b")
    three_model_layout = len(p["models"]) == 3
    style = ParagraphStyle("body", fontName="ReportFont",
                           fontSize=9.35 if three_model_layout else 9.6,
                           leading=13.0 if three_model_layout else 12.3,
                           spaceAfter=7 if three_model_layout else 4.5, textColor=ink)
    page_one_body = ParagraphStyle("page-one-body", parent=style,
                                   fontSize=9.0 if three_model_layout else 9.2,
                                   leading=12.2 if three_model_layout else 12.0,
                                   spaceAfter=5 if three_model_layout else 4)
    heading = ParagraphStyle("heading", parent=style, fontName=bold_name, fontSize=11.5, leading=15,
                             spaceBefore=7 if three_model_layout else 6.5, spaceAfter=3, textColor=blue)
    page_one_heading = ParagraphStyle("page-one-heading", parent=heading,
                                      spaceBefore=7)
    title_style = ParagraphStyle("title", parent=heading, fontSize=18, leading=22,
                                 spaceBefore=0, spaceAfter=2, textColor=blue)
    cover_title = ParagraphStyle("cover-title", parent=title_style, fontSize=27, leading=33,
                                 alignment=1, spaceBefore=0, spaceAfter=13, textColor=colors.black)
    cover_badge = ParagraphStyle("cover-badge", parent=style, fontName=bold_name, fontSize=12.5,
                                 leading=16, alignment=1, textColor=colors.black, spaceAfter=0)
    cover_detail = ParagraphStyle("cover-detail", parent=style, fontSize=14, leading=19,
                                  alignment=1, textColor=colors.black, spaceAfter=10)
    eyebrow = ParagraphStyle("eyebrow", parent=style, fontSize=9.1, leading=12,
                             textColor=colors.HexColor("#526574"), spaceAfter=3)
    model_heading = ParagraphStyle("model-heading", parent=heading, fontSize=10.5, leading=14,
                                   spaceBefore=6, spaceAfter=2)
    page_one_model_heading = ParagraphStyle("page-one-model-heading", parent=model_heading,
                                            spaceBefore=5)
    reflection_body = ParagraphStyle("reflection-body", parent=style, fontSize=9.6, leading=13.2,
                                     spaceAfter=5)
    reflection_heading = ParagraphStyle("reflection-heading", parent=heading, fontSize=11.2, leading=14,
                                        spaceBefore=8, spaceAfter=3)
    compact = ParagraphStyle("compact", parent=style,
                             fontSize=8.8 if three_model_layout else 9.4,
                             leading=11.7 if three_model_layout else 12.5, spaceAfter=0)
    table_header = ParagraphStyle("table-header", parent=compact, fontName=bold_name,
                                  textColor=blue)
    table_primary = ParagraphStyle("table-primary", parent=compact, fontName=bold_name,
                                   textColor=blue)
    def para(text, sty=style):
        return Paragraph(escape(text).replace("\n", "<br/>"), sty)
    def para_markup(markup, sty=style):
        return Paragraph(markup, sty)
    def grid(rows, widths, highlighted=()):
        cells = [[para(cell, table_header if row == 0 else table_primary if row in highlighted else compact)
                  for cell in values] for row, values in enumerate(rows)]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        cell_padding = 5 if three_model_layout else 4
        rules = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8f0")),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")]),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("TOPPADDING", (0, 0), (-1, -1), cell_padding),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), cell_padding),
                 ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#9eb4c4"))]
        rules.extend(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#e8f2f8"))
                     for row in highlighted)
        t.setStyle(TableStyle(rules))
        return t

    def markdown_table(rows):
        return (["| " + " | ".join(rows[0]) + " |", "| " + " | ".join("---" for _ in rows[0]) + " |"]
                + ["| " + " | ".join(row) + " |" for row in rows[1:]] + [""])

    story, markdown = [], []
    for page, blocks in enumerate(sections):
        if page:
            story.append(PageBreak())
        for i, (title, text) in enumerate(blocks):
            if identification_cover and page == 0 and i == 0:
                story.append(Spacer(1, 92))
                story.append(para(title, cover_title))
                story.extend([para(text, cover_badge), Spacer(1, 92)])
            elif i == 0 and page == report_first_page:
                story.append(para(title, title_style))
                if text:
                    story.append(para(text, eyebrow))
            elif title == "At a glance":
                story.append(para(title, page_one_heading))
                callout_markup = escape(text).replace(
                    escape(summary_emphasis),
                    f'<font name="{bold_name}">{escape(summary_emphasis)}</font>', 1)
                callout = Table([[para_markup(callout_markup, page_one_body)]], colWidths=[500], hAlign="LEFT")
                callout.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8f2f8")),
                                            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#24709c")),
                                            ("LEFTPADDING", (0, 0), (-1, -1), 11),
                                            ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                                            ("TOPPADDING", (0, 0), (-1, -1), 8),
                                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                story.append(callout)
            elif title == "Submission details":
                story.append(para(title, page_one_heading))
                repo_url = submission_rows[2][1]
                repo_label = repo_url
                repo_value = (para_markup(f'<link href="{escape(repo_url)}" color="#176a9a">'
                                          f'<u>{escape(repo_label)}</u></link>', compact)
                              if repo_url != "[not supplied]" else para(repo_url, compact))
                meta_cells = [
                    [para(row[0], table_header), para(row[1], compact),
                     para(row[2], table_header), para(row[3], compact)]
                    for row in submission_rows[:2]
                ]
                meta_cells.append([para("Skill repository", table_header), repo_value, "", ""])
                metadata_table = Table(meta_cells, colWidths=[115, 115, 135, 135], hAlign="LEFT")
                metadata_table.setStyle(TableStyle([
                    ("SPAN", (1, 2), (3, 2)),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f2f8")),
                    ("BACKGROUND", (2, 0), (2, 1), colors.HexColor("#e8f2f8")),
                    ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("LINEBELOW", (0, 0), (-1, -1), .35, colors.HexColor("#c6d4de")),
                ]))
                story.append(metadata_table)
            elif identification_cover and page == 0 and not title:
                for key, value in identification_rows:
                    story.append(para_markup(
                        f'<font name="{bold_name}">{escape(key)}:</font> {escape(value)}',
                        cover_detail,
                    ))
            elif title in {"Identification details", "Skill information"}:
                rows = identification_rows if title == "Identification details" else skill_rows
                heading_style = page_one_heading if page == report_first_page else heading
                story.append(para(title, heading_style))
                meta_cells = []
                for key, value in rows:
                    if key == "Skill repository" and value != "[not supplied]":
                        rendered_value = para_markup(
                            f'<link href="{escape(value)}" color="#176a9a"><u>{escape(value)}</u></link>',
                            compact,
                        )
                    else:
                        rendered_value = para(value, compact)
                    meta_cells.append([para(key, table_header), rendered_value])
                metadata_table = Table(meta_cells, colWidths=[150, 350], hAlign="LEFT")
                metadata_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f2f8")),
                    ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW", (0, 0), (-1, -1), .35, colors.HexColor("#c6d4de")),
                ]))
                story.append(metadata_table)
            elif title == "Findings and discussion":
                story.append(para(title, heading))
                callout = Table([[para(text, style)]], colWidths=[500], hAlign="LEFT")
                callout.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f5f8")),
                                            ("LINEBEFORE", (0, 0), (0, -1), 2, colors.HexColor("#24709c")),
                                            ("LEFTPADDING", (0, 0), (-1, -1), 9),
                                            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                                            ("TOPPADDING", (0, 0), (-1, -1), 7),
                                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
                story.append(callout)
            else:
                if page == reflection_group and i > 0:
                    story.extend([para(title, reflection_heading), para(text, reflection_body)])
                else:
                    title_style_for_block = (title_style if i == 0 else
                                             page_one_model_heading if page == report_first_page and title.startswith("Model:") else
                                             page_one_heading if page == report_first_page else heading)
                    body_style_for_block = page_one_body if page == report_first_page else style
                    story.append(para(title, title_style_for_block))
                    if text:
                        story.append(para(text, body_style_for_block))
            if title == "Submission details":
                markdown.extend([f"## {title}", "", "| Property | Value |", "| --- | --- |"])
                for row in submission_rows:
                    value = (f"[{row[1]}]({row[1]})" if row[0] == "Skill repository"
                             and row[1] != "[not supplied]" else row[1])
                    markdown.append(f"| {row[0]} | {value} |")
                    if row[2]:
                        markdown.append(f"| {row[2]} | {row[3]} |")
                markdown.append("")
            elif identification_cover and page == 0 and not title:
                markdown.extend(["| Property | Value |", "| --- | --- |"])
                markdown.extend(f"| {key} | {value} |" for key, value in identification_rows)
                markdown.append("")
            elif title in {"Identification details", "Skill information"}:
                rows = identification_rows if title == "Identification details" else skill_rows
                markdown.extend([f"## {title}", "", "| Property | Value |", "| --- | --- |"])
                for key, value in rows:
                    rendered_value = (f"[{value}]({value})" if key == "Skill repository"
                                      and value != "[not supplied]" else value)
                    markdown.append(f"| {key} | {rendered_value} |")
                markdown.append("")
            else:
                markdown.extend([f"## {title}", "", text, ""])
            if page == report_second_page and title == "Evaluation and comparison":
                story.extend([grid(table, [128, 135, 138, 99], primary_rows), Spacer(1, 5)])
                markdown.extend(markdown_table(table))
            if title == "Prediction errors (confusion matrix)" and error_table:
                story.extend([grid(error_table, [160, 85, 85, 85, 85]), Spacer(1, 4)])
                markdown.extend(markdown_table(error_table))
    pdf = output_dir / "report.pdf"
    def footer(canvas, doc):
        canvas.setFont("ReportFont", 8)
        canvas.drawRightString(A4[0] - 45, 24, str(doc.page))
    SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=45, rightMargin=45,
                      topMargin=32, bottomMargin=36).build(story, onFirstPage=footer, onLaterPages=footer)
    pages = PdfReader(pdf).pages
    reflection_page = next((i for i, page in enumerate(pages) if reflection_title in (page.extract_text() or "")), None)
    if reflection_page != reflection_group:
        raise ValueError("Main report exceeded two pages; shorten narrative/metric table and regenerate in a fresh folder. PDF is not submission-ready.")
    (output_dir / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    renderer_hash = sha(Path(__file__))
    manifest = {"cover_pages": 1 if identification_cover else 0,
                "main_pages": 2, "total_pages": len(pages), "draft": draft,
                "missing_metadata": missing, "human_reflection_confirmed": n["human_reflection_confirmed"],
                "training_results_sha256": sha(training), "narrative_sha256": sha(narrative),
                "test_results_sha256": sha(test_results) if test_results else None,
                "model_lock_sha256": sha(lock) if lock else None,
                "renderer_sha256": renderer_hash,
                "renderer_changed_since_lock": renderer_changed,
                "changed_scripts_since_lock": sorted(changed_scripts) if test_results else [],
                "presentation_only_rerender": bool(presentation_only_rerender),
                "report_pdf_sha256": sha(pdf), "report_md_sha256": sha(output_dir / "report.md"),
                "verification": verification, "visual_inspection_required": True}
    write_json(output_dir / "report_manifest.json", manifest, exclusive=True)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--diagnosis", type=Path, required=True)
    p.add_argument("--narrative", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variant", default="baseline")
    p.add_argument("--test-results", type=Path)
    p.add_argument("--lock", type=Path)
    p.add_argument("--font", type=Path, help="Optional embeddable TrueType font covering all report characters")
    p.add_argument("--presentation-only-rerender", action="store_true",
                   help="Explicitly rerender verified saved results if only this report renderer changed since Model Lock")
    a = p.parse_args()
    print(generate(a.training, a.diagnosis, a.narrative, a.output_dir, a.variant, a.test_results, a.lock,
                   a.font, a.presentation_only_rerender))
