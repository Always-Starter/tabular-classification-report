---
name: tabular-classification-report
description: Select, train and compare two classifiers for a supplied tabular dataset, with train-only diagnosis, data-specific preprocessing and metrics, leakage-safe validation, optional frozen held-out evaluation, and an automatically generated report and Reflection draft. Use for binary or multiclass classification reports, including IN6227 Variant 2.
---

# Tabular Classification Report

Take a training dataset path as the starting input. Ask for the target when it is not supplied or unambiguous; never assume the last column is the target. Optional inputs are a held-out path, Excel worksheet (default `Data`), output folder, error costs and report metadata. CSV/TSV/XLSX are supported; legacy XLS requires `xlrd`. A separate test file is optional: nested CV can support a development-only report.

Use the installed skill directory for scripts and references. Save each experiment in a new output folder outside the skill and source data. Check Python dependencies from `requirements.txt` before running. Package fixtures are synthetic software tests, not course data or a model-selection prescription.

For every material choice, record **Observation → Decision → Rationale → Human-review point**. Ground decisions in the supplied training data. Do not infer customer churn, business meanings or label meanings from an example, filename or prior demonstration.

## Invariants

- Keep all held-out file contents sealed through diagnosis, planning and development; do not profile test predictors or labels. No test-based model revisions.
- Fit learned imputation, encoding, scaling, selection and transformations inside training folds. Choose grouped or temporal validation when observations are dependent.
- Exclude missing targets from supervised fitting/scoring; report counts. Predict unlabelled rows without claiming accuracy.
- Review identifiers, duplicate entities, target proxies and feature availability. Association alone does not prove leakage or justify deletion; unusual/outlying values are not automatically errors.
- Freeze both complete fitted pipelines before held-out access. Record actual human approval of the displayed lock digest before calling the evaluator. Never create an approval on the user's behalf without their explicit approval of that lock.
- The receipt and hashes enforce normal workflow checks, not security against someone deliberately editing/deleting the files. Preserve them. Never rerun or remove a receipt to improve test performance.

## Choose the review mode

At the start of a run, honour an explicit mode choice already made for that run. Otherwise, **before reading the dataset or running a checkpoint**, show a separate, user-visible choice prompt in the user's language. Ask the user to pick one of these two modes; merely announcing the default does not count as offering a choice:

- **Staged review (default; recommended for first use):** show the diagnosis after Checkpoint 1, the proposed plan after Checkpoint 2, and the training results and Model Lock after Checkpoint 3. Stop at each boundary and wait for the user's approval or requested changes before proceeding.
- **Continuous execution:** make and explain the intermediate decisions and complete Checkpoints 1–3 without routine approval pauses. Still show the exact Model Lock and wait for explicit approval before accessing a held-out test file. This mode does not authorise unattended test evaluation.

For example, ask: “How would you like to run this? A. Staged review (default): I pause after the diagnosis, plan and training results. B. Continuous execution: I make the intermediate decisions and pause at the Model Lock before using the test set. Which do you choose?” Use a choice/input control when available so the options are visible. If the question can remain open asynchronously and no answer has arrived, proceed under staged review only as far as Checkpoint 1; the question must already have been shown. If the interface cannot collect an asynchronous answer, wait for the user's choice before reading the dataset.

A broad request such as “run the whole analysis” or providing both train and test paths does not select continuous execution. Keep the chosen mode for the run instead of asking at every checkpoint. Honour later requests to stop or switch modes, and narrower limits such as “Checkpoint 1 only”, in either mode. Stop for unresolved target/provenance issues that could invalidate the experiment in either mode.

At every review pause, present the actual results and artifact links, explain the material decisions and uncertainties, identify what the user needs to review, and state exactly what approval would start next. Ask only for information the user may know; allow “unknown” and disclose resulting assumptions or limitations. Do not treat silence as approval or claim a manual check the user has not performed.

## Workflow and routing

Read [checkpoint-workflow.md](references/checkpoint-workflow.md) for commands and apply the selected review mode at each boundary.

1. **Diagnose training data.** Run `scripts/diagnose_training.py` with only the training path. Inspect missingness, class balance, duplicates, numeric ranges, cardinality and descriptive associations. Explain domain uncertainty and candidate leakage. This CLI has no test-path option.
2. **Propose the plan.** Read [modelling-decisions.md](references/modelling-decisions.md) and [model-plan-schema.md](references/model-plan-schema.md). Select two informative classifiers, per-model preprocessing, feature exclusions, primary/secondary metrics, splits, bounded grids and justified sensitivities. Write a schema-v2 JSON plan and validate with `scripts/validate_plan.py`. No fitting at this stage.
3. **Develop and lock.** Run `scripts/run_nested_cv.py`, which uses shared outer/inner splits, compares prespecified sensitivities and fits final candidates on eligible training rows. Review fold variability, warnings and sensitivity evidence. Select a complete variant before testing; write the review JSON. Read [model-lock.md](references/model-lock.md), then run `scripts/freeze_model_lock.py`. Show the selected plan, both pipelines, metrics, class order, threshold, uncertainties and SHA-256 digest for approval.
4. **Evaluate once and report.** After actual approval, use `scripts/approve_model_lock.py` to record it, then `scripts/evaluate_holdout.py` once. The evaluator checks approval, environment, code, schema and model hashes before accessing the test file. Use `scripts/verify_results.py` on saved predictions; it never reopens source test data. Do not refit or change a model after test access.

For reporting (including when there is no test file), read [report-template.md](references/report-template.md). Write evidence-based narrative JSON, then run `scripts/generate_report.py`. It generates Markdown, a PDF with two main pages plus Reflection, and an evidence manifest. Visually inspect the PDF. Numerical tables come from verified saved results; check that narrative claims agree with them. Keep unknown metadata and unperformed human actions explicit and mark the report as a draft.

## Capability boundaries

The packaged runner supports binary/multiclass targets, six established classifiers, per-model imputation/encoding/scaling, stratified/grouped/forward-time nested CV, and declared feature/preprocessing/model-parameter/threshold sensitivity variants. See the plan reference for exact options and budget limits. It does not automatically solve every possible tabular problem: multilabel targets, custom feature engineering, exotic estimators or large sparse encodings need an explicit extension.

When the dataset needs an unsupported method, explain the limitation and extend the reusable plan/pipeline/scoring implementation with a focused synthetic test **before development**, if authorized by the analysis task. Do not silently choose a supported but inappropriate method, improvise an evaluator that bypasses the lock, or claim universal support. Changing code after locking invalidates that lock; after test access, disclose the limitation without retuning.

## IN6227 Variant 2

The assignment requires data-specific AI model/preprocessing selection, two-model comparison, reusable execution and automated reporting. Review modes, nested CV and Model Lock are this skill's design choices, not quoted grading requirements. Include full name, matric number, `IN6227-Assignment-1`, `Variant-2`, actual verifiable LLM model/version, interface and GitHub URL in the report. The main PDF is at most two pages; Reflection follows in the same PDF and is outside that limit. Reflection must describe real human oversight, a challenged decision and a specific manually checked output. Automated verification can supply a calculation for the student to check, but cannot claim the student did so.
