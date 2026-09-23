#!/usr/bin/env python3
"""Generate traceable Markdown and a two-page main PDF plus separate Reflection."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
from common import read_json, sha, write_json
from verify_results import verify, verify_training


def generate(training, diagnosis, narrative, output_dir, variant="baseline", test_results=None, lock=None, font=None):
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
        verification = verify(test_results, lock)
        lk = read_json(lock)
        if lk["training_results_sha256"] != sha(training):
            raise ValueError("Lock refers to different training results")
        variant = lk["review"]["selected_variant"]
        test = read_json(test_results)
    else:
        verification, test = None, None
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
    sections = [[], [], []]
    def add(page, title, text):
        sections[page].append((title, str(text)))

    add(0, "IN6227-Assignment-1 | Variant-2", "DRAFT - metadata or human Reflection pending" if draft else "Classification report")
    add(0, "Submission details", " | ".join(f"{k}: {meta.get(k) or '[not supplied]'}" for k in metadata_keys))
    add(0, "Data exploration and cleaning", f"Training: {tr['input_rows']} rows; {tr['eligible_rows']} labelled rows; "
        f"{tr['missing_targets']} missing targets excluded. Target: {p['target']}; classes: {json.dumps(tr['class_counts'], ensure_ascii=False)}. "
        f"Predictors retained: {len(p['features'])}. Exact duplicate rows: {dg['duplicates']['exact_duplicate_rows']}. " + n["exploration"])
    add(0, "Preprocessing and feature decisions", n["preprocessing"] + " " + n["features"])
    for spec in p["models"]:
        model = vr["models"][spec["name"]]
        add(0, f"Model: {spec['name']} ({spec['type']})", "Preprocessing: " + json.dumps(spec["preprocessing"], ensure_ascii=False)
            + ". Selected parameters: " + json.dumps(model["best_params"], ensure_ascii=False)
            + ". Fixed parameters: " + json.dumps(spec["params"], ensure_ascii=False) + ".")
    add(0, "Model choice and stopping rules", n["model_rationale"]
        + f" Limited grid search; {tr['planned_fits']} planned fits across baseline and declared sensitivities. "
        + "No early stopping is inferred; estimator stopping settings and resolved defaults are saved in training_results.json.")
    metric = p["metrics"]["primary"]
    add(1, "Evaluation and comparison", f"Primary metric: {metric} ({'lower' if metric == 'log_loss' else 'higher'} is better). "
        f"Nested CV: {p['cv']['strategy']}, {p['cv']['outer_splits']} outer / {p['cv']['inner_splits']} inner folds; seed {p['seed']}. "
        "All learned preprocessing is fitted within folds. All classifiers share validation splits. "
        + (f"Binary positive class: {p['positive_class']}; threshold: {p['threshold']}. " if p["task"] == "binary" else "Multiclass prediction uses argmax. ")
        + (f"Held-out evaluation: {test['test_rows']} rows, {test['labelled_rows']} labelled, {test['missing_labels']} unlabelled. "
           if test else "No held-out evaluation performed; these are development estimates. ")
        + "Fold SD measures variability, not a confidence interval. Sensitivity selection can introduce development selection optimism.")
    # The printed metric set is fixed by the pre-test plan order, never selected from
    # held-out performance. Full scores for every declared metric remain in evidence JSON.
    displayed_metrics = [metric, *p["metrics"]["secondary"][:2]]
    omitted_metrics = p["metrics"]["secondary"][2:]
    if omitted_metrics:
        add(1, "Additional prespecified metrics", "Full fold and held-out results for "
            + ", ".join(omitted_metrics) + " are retained in the verified result files; "
            "the compact PDF table shows the primary and first two secondary metrics in plan order.")
    table = [["Model / metric", "Outer mean (SD)", "Held-out"]]
    for name, model in vr["models"].items():
        for m in displayed_metrics:
            summary = model["outer_summary"][m]
            value = test["models"][name]["metrics"][m] if test else None
            development_score = "N/A" if summary["mean"] is None else f"{summary['mean']:.3f} ({summary['std']:.3f})"
            table.append([f"{name} / {m}", development_score, "N/A" if value is None else f"{value:.3f}"])
    undefined_development = {name: sorted({reason for fold in model["fold_results"] for reason in fold["undefined_metrics"].values()})
                             for name, model in vr["models"].items() if any(f["undefined_metrics"] for f in model["fold_results"])}
    if undefined_development:
        add(1, "Undefined development scores", json.dumps(undefined_development)
            + ". A secondary-metric summary is N/A if any fold is undefined; no partial-fold average is substituted.")
    if test:
        for name, model in test["models"].items():
            add(1, f"Confusion matrix: {name}", f"Rows=true, columns=predicted; order={test['class_order']}; "
                + json.dumps(model["confusion_matrix"]) + (". Undefined metrics: " + json.dumps(model["undefined_metrics"]) if model["undefined_metrics"] else ""))
    add(1, "Findings and discussion", n["findings"])
    add(1, "Limitations and reproducibility", n["limitations"] + f" Selected development variant: {variant}. "
        + f"Python {tr['environment']['python']}; scikit-learn {tr['environment']['packages']['scikit-learn']}. "
        + "Plans, folds, model files, predictions, exact package versions and SHA-256 evidence accompany this report.")
    add(2, "Reflection - outside the two-page report limit", "Human review draft" if not n["human_reflection_confirmed"] else "Human-confirmed Reflection")
    for k in reflection_keys:
        add(2, k.replace("_", " ").title(), n["reflection"][k])
    if verification:
        add(2, "Automated verification aid", json.dumps(verification["manual_check_candidates"], ensure_ascii=False)
            + ". This calculation is provided for the student to check; it is not a claim that the student has checked it.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / f).exists() for f in ("report.pdf", "report.md", "report_manifest.json")):
        raise ValueError("Report files already exist; use a fresh output directory")
    # Embed an actual font, rather than relying on optional viewer CJK language packs.
    font_path = Path(font) if font else Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
    embedded = TTFont("ReportFont", str(font_path))
    pdfmetrics.registerFont(embedded)
    all_text = "".join(title + text for blocks in sections for title, text in blocks) + "".join(cell for row in table for cell in row)
    missing_glyphs = sorted({c for c in all_text if not c.isspace() and ord(c) not in embedded.face.charToGlyph})
    if missing_glyphs:
        raise ValueError(f"Report font lacks glyphs {missing_glyphs[:12]}; pass --font /path/to/a/covering.ttf before rendering")
    style = ParagraphStyle("body", fontName="ReportFont", fontSize=9.5, leading=12.5, spaceAfter=5)
    heading = ParagraphStyle("heading", parent=style, fontSize=11, leading=14, spaceBefore=6, spaceAfter=4)
    def para(text, sty=style):
        return Paragraph(escape(text).replace("\n", "<br/>"), sty)
    story, markdown = [], []
    for page, blocks in enumerate(sections):
        if page:
            story.append(PageBreak())
        for i, (title, text) in enumerate(blocks):
            story.extend([para(title, heading), para(text)])
            markdown.extend([f"## {title}", "", text, ""])
            if page == 1 and i == 0:
                cells = [[para(cell) for cell in row] for row in table]
                t = Table(cells, colWidths=[235, 145, 120], repeatRows=1)
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9edf2")),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                                       ("LINEBELOW", (0, 0), (-1, 0), .5, colors.grey)]))
                story.extend([t, Spacer(1, 5)])
                markdown.extend(["| " + " | ".join(table[0]) + " |", "| --- | --- | --- |"])
                markdown.extend("| " + " | ".join(row) + " |" for row in table[1:])
                markdown.append("")
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
    manifest = {"main_pages": 2, "total_pages": len(pages), "draft": draft,
                "missing_metadata": missing, "human_reflection_confirmed": n["human_reflection_confirmed"],
                "training_results_sha256": sha(training), "narrative_sha256": sha(narrative),
                "test_results_sha256": sha(test_results) if test_results else None,
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
    a = p.parse_args()
    print(generate(a.training, a.diagnosis, a.narrative, a.output_dir, a.variant, a.test_results, a.lock, a.font))
