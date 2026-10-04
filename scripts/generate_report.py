#!/usr/bin/env python3
"""Generate traceable Markdown and a two-page main PDF plus separate Reflection."""
import argparse
from pathlib import Path
from xml.sax.saxutils import escape
from common import code_hashes, read_json, sha, write_json
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


def model_label(spec):
    return spec["type"].replace("_", " ").capitalize()


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
    vr = tr["variants"][variant]
    p = vr["plan"]
    text_keys = ("exploration", "preprocessing", "features", "model_rationale", "findings", "limitations")
    if any(not isinstance(n.get(k), str) or not n[k].strip() for k in text_keys):
        raise ValueError(f"Narrative must include nonempty text for {text_keys}")
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
    draft = bool(missing) or not n["human_reflection_confirmed"]
    metric = p["metrics"]["primary"]
    model_labels = {spec["name"]: model_label(spec) for spec in p["models"]}
    if missing and not n["human_reflection_confirmed"]:
        draft_note = " | DRAFT - metadata and Reflection pending"
    elif missing:
        draft_note = " | DRAFT - metadata pending"
    elif not n["human_reflection_confirmed"]:
        draft_note = " | DRAFT - Reflection pending"
    else:
        draft_note = ""
    sections = [[], [], []]
    def add(page, title, text):
        sections[page].append((title, str(text)))

    add(0, "Tabular classification report", "IN6227-Assignment-1 | Variant-2" + draft_note)
    if test:
        preferred = lk["review"]["preferred_model"]
        if preferred not in vr["models"]:
            raise ValueError("Preferred Model Lock model is absent from selected variant")
        cv_score = vr["models"][preferred]["outer_summary"][metric]["mean"]
        held_score = test["models"][preferred]["metrics"][metric]
        cv_display = "N/A" if cv_score is None else f"{cv_score:.4f}"
        held_display = "N/A" if held_score is None else f"{held_score:.4f}"
        summary = (f"Frozen pre-test choice: {model_labels[preferred]}. "
                   f"Nested-CV {metric.replace('_', ' ')}: {cv_display}; "
                   f"held-out: {held_display}. The model choice was not revised after held-out evaluation."
                   + (" This is a presentation-only derivative from verified saved evidence; no model was refitted."
                      if presentation_only_rerender else ""))
    else:
        summary = (f"Development-only comparison. Primary measure: {metric.replace('_', ' ')}. "
                   "No independent held-out result is claimed.")
    add(0, "At a glance", summary)
    add(0, "Submission details", f"{meta.get('full_name') or '[name not supplied]'} | "
        f"Matriculation number: {meta.get('matric_number') or '[not supplied]'}\n"
        f"LLM model/version: {meta.get('llm_model_version') or '[not supplied]'} | "
        f"Interface: {meta.get('llm_interface') or '[not supplied]'}\n"
        f"Skill repository: {meta.get('repository_url') or '[not supplied]'}")
    class_counts = "; ".join(f"{label}: {count:,}" for label, count in tr["class_counts"].items())
    add(0, "Data exploration and cleaning", f"Training: {tr['input_rows']:,} rows; {tr['eligible_rows']:,} labelled rows; "
        f"{tr['missing_targets']} missing targets excluded. Target: {p['target']}; class counts: {class_counts}. "
        f"Predictors retained: {len(p['features'])}. Exact duplicate rows: {dg['duplicates']['exact_duplicate_rows']}.\n" + n["exploration"])
    add(0, "Preprocessing and feature decisions", n["preprocessing"] + " " + n["features"])
    for spec in p["models"]:
        model = vr["models"][spec["name"]]
        boundary = model.get("tuning_boundary", {})
        add(0, f"Model: {model_labels[spec['name']]}", "Preparation: " + preprocessing_text(spec["preprocessing"])
            + ". Selected settings: " + settings_text(model["best_params"])
            + ". Fixed settings: " + settings_text(spec["params"]) + "."
            + (" Numeric search edge: " + ", ".join(f"{key.removeprefix('model__').replace('_', ' ')}="
                                                    f"{value['selected']} ({value['edge']})"
                 for key, value in boundary.items()) + "; values beyond this edge were not evaluated."
               if boundary else ""))
    add(0, "Model choice and stopping rules", n["model_rationale"]
        + f" The declared search used {tr['planned_fits']} planned fits across baseline and sensitivity analyses; "
        + "outer folds did not trigger grid expansion or a change of primary metric.")
    add(1, "Evaluation and comparison", f"Primary metric: {metric.replace('_', ' ')} "
        f"({'lower' if metric == 'log_loss' else 'higher'} is better). "
        f"Nested CV: {p['cv']['strategy']}, {p['cv']['outer_splits']} outer / {p['cv']['inner_splits']} inner folds; seed {p['seed']}. "
        "All learned preprocessing is fitted within folds. All classifiers share validation splits. "
        + (f"Binary positive class: {p['positive_class']}; threshold: {p['threshold']}. " if p["task"] == "binary" else "Multiclass prediction uses argmax. ")
        + (f"Held-out evaluation: {test['test_rows']:,} rows, {test['labelled_rows']:,} labelled, {test['missing_labels']} unlabelled. "
           if test else "No held-out evaluation performed; these are development estimates. ")
        + "Fold SD measures variability, not a confidence interval. Repeatedly comparing development variants and "
        + "retaining the best-performing one can introduce selection optimism, even when the held-out set remains untouched.")
    # The printed metric set is fixed by the pre-test plan order, never selected from
    # held-out performance. Full scores for every declared metric remain in evidence JSON.
    displayed_metrics = [metric, *p["metrics"]["secondary"][:3]]
    omitted_metrics = p["metrics"]["secondary"][3:]
    if omitted_metrics:
        add(1, "Additional prespecified metrics", "Full fold and held-out results for "
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
        add(1, "Undefined development scores", reasons
            + ". A secondary-metric summary is N/A if any fold is undefined; no partial-fold average is substituted.")
    error_table = None
    if test:
        if len(test["class_order"]) == 2 and p.get("positive_class") in test["class_order"]:
            pos = test["class_order"].index(p["positive_class"])
            neg = 1 - pos
            add(1, "Prediction errors (confusion matrix)",
                f"Positive class: {p['positive_class']}. "
                "TN = true negatives, FP = false positives, FN = false negatives, TP = true positives.")
            error_table = [["Model", "TN", "FP", "FN", "TP"]]
            for name, model in test["models"].items():
                matrix = model["confusion_matrix"]
                error_table.append([model_labels[name], f"{matrix[neg][neg]:,}", f"{matrix[neg][pos]:,}",
                                    f"{matrix[pos][neg]:,}", f"{matrix[pos][pos]:,}"])
                if model["undefined_metrics"]:
                    add(1, f"Undefined held-out metrics: {model_labels[name]}", settings_text(model["undefined_metrics"]))
        else:
            for name, model in test["models"].items():
                add(1, f"Prediction errors (confusion matrix): {model_labels[name]}",
                    confusion_text(model["confusion_matrix"], test["class_order"], p.get("positive_class"))
                    + (" Undefined metrics: " + settings_text(model["undefined_metrics"]) + "."
                       if model["undefined_metrics"] else ""))
    add(1, "Findings and discussion", n["findings"])
    add(1, "Limitations", n["limitations"] + f" Selected development variant: {variant}.")
    add(2, "Reflection - outside the two-page report limit", "Human review draft" if not n["human_reflection_confirmed"] else "Human-confirmed Reflection")
    reflection_titles = {
        "human_oversight": "Human in the Loop",
        "challenged_decision": "Critical Evaluation",
        "manual_verification": "Trustworthiness",
        "future_changes": "Future Changes",
    }
    for k in reflection_keys:
        add(2, reflection_titles[k], n["reflection"][k])
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
    style = ParagraphStyle("body", fontName="ReportFont", fontSize=10.1, leading=14.5,
                           spaceAfter=7, textColor=ink)
    heading = ParagraphStyle("heading", parent=style, fontName=bold_name, fontSize=11.5, leading=15,
                             spaceBefore=11, spaceAfter=4, textColor=blue)
    title_style = ParagraphStyle("title", parent=heading, fontSize=18, leading=22,
                                 spaceBefore=0, spaceAfter=2, textColor=blue)
    eyebrow = ParagraphStyle("eyebrow", parent=style, fontSize=9.1, leading=12,
                             textColor=colors.HexColor("#526574"), spaceAfter=3)
    model_heading = ParagraphStyle("model-heading", parent=heading, fontSize=10.5, leading=14,
                                   spaceBefore=6, spaceAfter=2)
    reflection_body = ParagraphStyle("reflection-body", parent=style, fontSize=9.6, leading=13.2,
                                     spaceAfter=5)
    reflection_heading = ParagraphStyle("reflection-heading", parent=heading, fontSize=11.2, leading=14,
                                        spaceBefore=8, spaceAfter=3)
    compact = ParagraphStyle("compact", parent=style, fontSize=9.4, leading=12.5, spaceAfter=0)
    table_header = ParagraphStyle("table-header", parent=compact, fontName=bold_name,
                                  textColor=blue)
    table_primary = ParagraphStyle("table-primary", parent=compact, fontName=bold_name,
                                   textColor=blue)
    def para(text, sty=style):
        return Paragraph(escape(text).replace("\n", "<br/>"), sty)
    def grid(rows, widths, highlighted=()):
        cells = [[para(cell, table_header if row == 0 else table_primary if row in highlighted else compact)
                  for cell in values] for row, values in enumerate(rows)]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        rules = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8f0")),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")]),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("TOPPADDING", (0, 0), (-1, -1), 5),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
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
            if page == 0 and i == 0:
                story.extend([para(title, title_style), para(text, eyebrow)])
            elif title == "At a glance":
                story.append(para(title, heading))
                callout = Table([[para(text, style)]], colWidths=[500], hAlign="LEFT")
                callout.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8f2f8")),
                                            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#24709c")),
                                            ("LEFTPADDING", (0, 0), (-1, -1), 11),
                                            ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                                            ("TOPPADDING", (0, 0), (-1, -1), 8),
                                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                story.append(callout)
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
                if page == 2 and i > 0:
                    story.extend([para(title, reflection_heading), para(text, reflection_body)])
                else:
                    title_style_for_block = title_style if i == 0 else model_heading if title.startswith("Model:") else heading
                    story.extend([para(title, title_style_for_block), para(text)])
            markdown.extend([f"## {title}", "", text, ""])
            if page == 1 and i == 0:
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
    reflection_page = next((i for i, page in enumerate(pages) if "Reflection - outside" in (page.extract_text() or "")), None)
    if reflection_page != 2:
        raise ValueError("Main report exceeded two pages; shorten narrative/metric table and regenerate in a fresh folder. PDF is not submission-ready.")
    (output_dir / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    renderer_hash = sha(Path(__file__))
    manifest = {"main_pages": 2, "total_pages": len(pages), "draft": draft,
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
