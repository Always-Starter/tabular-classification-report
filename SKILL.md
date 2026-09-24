---
name: tabular-classification-report
description: Select, train and compare data-appropriate classifiers for a supplied tabular dataset, with train-only diagnosis, data-specific preprocessing and metrics, leakage-safe validation, optional frozen held-out evaluation, and an automatically generated report and Reflection draft. Use for binary or multiclass classification reports, including IN6227 Variant 2.
---

# Tabular Classification Report

Take a training dataset path as the starting input. Resolve the target from the user's explicit instruction or authoritative task/dataset metadata before supervised diagnosis. If none identifies it unambiguously, inspect the training schema and ask. Target identification is task semantics, not statistical inference: never guess from column order, name, apparent class count, filename, association or prior examples. Optional inputs are a held-out path, Excel worksheet (default `Data`), output folder, error costs and report metadata. CSV/TSV/XLSX are supported; legacy XLS requires `xlrd`. A separate test file is optional: nested CV can support a development-only report.

Use the installed skill directory for scripts and references. Save each experiment in a new output folder outside the skill and source data. Check Python dependencies from `requirements.txt` before running. Package fixtures are synthetic software tests, not course data or a model-selection prescription.

For every material choice, record **Observation → Decision → Rationale → Human-review point**. Ground decisions in the supplied training data. Do not infer customer churn, business meanings or label meanings from an example, filename or prior demonstration.

Use **English by default** for the mode-choice prompt, checkpoint results, review questions, and generated report narrative. If the user's current request clearly uses another language, respond in that language; an explicit language preference takes precedence. When a request contains only paths, commands, or otherwise gives no clear language signal, use English. Follow a later language switch for subsequent messages. Do not carry an earlier conversation language into a new English-language request, and do not translate dataset column names or observed label values.

## Invariants

- Keep all held-out file contents sealed through diagnosis, planning and development; do not profile test predictors or labels. No test-based model revisions.
- Fit learned imputation, encoding, scaling, selection and transformations inside training folds. Choose grouped or temporal validation when observations are dependent.
- Exclude missing targets from supervised fitting/scoring; report counts. Predict unlabelled rows without claiming accuracy.
- Review identifiers, duplicate entities, target proxies and feature availability. Association alone does not prove leakage or justify deletion; unusual/outlying values are not automatically errors.
- If an independent held-out file is supplied, freeze every selected complete fitted pipeline before held-out access. Record actual human approval of the displayed lock digest before calling the evaluator. Never create an approval on the user's behalf without their explicit approval of that lock. Do not require a holdout or Model Lock for a training-only report.
- The receipt and hashes enforce normal workflow checks, not security against someone deliberately editing/deleting the files. Preserve them. Never rerun or remove a receipt to improve test performance.

## Choose the review mode

At the start of a run, honour an explicit mode choice already made for that run. Otherwise, **before reading the dataset or running a checkpoint**, show a separate, user-visible choice prompt using the language rule above. Ask the user to pick one of these two modes; merely announcing the default does not count as offering a choice:

- **Staged review (default; recommended for first use):** show the diagnosis after Checkpoint 1, the proposed plan after Checkpoint 2, and the training results and Model Lock after Checkpoint 3. Stop at each boundary and wait for the user's approval or requested changes before proceeding.
- **Continuous execution:** make and explain the intermediate decisions and complete Checkpoints 1–3 without routine approval pauses. Still show the exact Model Lock and wait for explicit approval before accessing a held-out test file. This mode does not authorise unattended test evaluation.

For example, ask: “How would you like to run this? A. Staged review (default): I pause after the diagnosis, plan and training results. B. Continuous execution: I make the intermediate decisions and pause at the Model Lock before using the test set. Which do you choose?” Use a choice/input control when available so the options are visible. If the question can remain open asynchronously and no answer has arrived, proceed under staged review only as far as Checkpoint 1; the question must already have been shown. If the interface cannot collect an asynchronous answer, wait for the user's choice before reading the dataset.

A broad request such as “run the whole analysis” or providing both train and test paths does not select continuous execution. Keep the chosen mode for the run instead of asking at every checkpoint. Honour later requests to stop or switch modes, and narrower limits such as “Checkpoint 1 only”, in either mode. Stop for unresolved target/provenance issues that could invalidate the experiment in either mode.

At every review pause, present actual results and artifact links. For each material decision, show what the AI decided, why, what the human should review, and a reasonable alternative worth challenging. State exactly what approval would start next. Ask only for information the user may know; allow “unknown” and disclose resulting assumptions or limitations. Do not treat silence as approval or claim a manual check the user has not performed.

## Workflow and routing

Read [checkpoint-workflow.md](references/checkpoint-workflow.md) for commands and apply the selected review mode at each boundary.

**Preflight (before Checkpoint 1): resolve target and validate input.** Run `scripts/inspect_training_schema.py` on training only, with `--target` only if the user or authoritative metadata supplied it. If unresolved, ask and stop. If the named column is absent, show the actual columns and stop rather than substituting another.

1. **Diagnose training data.** Once resolved, run `scripts/diagnose_training.py` with both `--train` and explicit `--target`. Inspect missingness, class balance, duplicates, numeric ranges, cardinality and descriptive associations. Explain domain uncertainty and candidate leakage. Neither CLI has a test-path option.
2. **Propose the plan.** Read [modelling-decisions.md](references/modelling-decisions.md) and [model-plan-schema.md](references/model-plan-schema.md). Select appropriate candidate classifier families for empirical comparison, not a presumed winner: two by default, a third only for a distinct training-evidence-based question. For each, explain expected strength and known limitation. Decide per-model preprocessing, optional feature handling, a predeclared primary metric and interpretive secondary metrics, dependency-safe splits, bounded grids with an explicit stopping rule, and sensitivities only for material uncertainty. Write a schema-v2 JSON plan and validate with `scripts/validate_plan.py`. No fitting at this stage.
3. **Develop; lock if holdout exists.** Run `scripts/run_nested_cv.py`: inner CV tunes, outer CV evaluates the tuned procedure. Outer results must not drive grid expansion, family/preprocessing redesign or a new primary metric within the same evaluation claim; a later redesign is a new development iteration. Stop tuning after the prespecified bounded inner search. Review fold variability, near-ties, warnings, sensitivity evidence and numeric-grid boundary flags without automatically expanding the search. If an independent holdout exists, select a complete variant, write the review JSON, read [model-lock.md](references/model-lock.md), then run `scripts/freeze_model_lock.py`. Show all frozen pipelines and the exact digest for approval.
4. **Optional one-time holdout, verification and report.** Only after actual lock approval, run `scripts/approve_model_lock.py` and `scripts/evaluate_holdout.py` once. Verify saved predictions with `scripts/verify_results.py` without reopening test data. With no holdout, use nested-CV development evidence after the required review pause. Never revise model decisions using held-out results.

For reporting (including when there is no test file), read [report-template.md](references/report-template.md). Write evidence-based narrative JSON, then run `scripts/generate_report.py`. It generates Markdown, a PDF with two main pages plus Reflection, and an evidence manifest. Visually inspect the PDF. Numerical tables come from verified saved results; check that narrative claims agree with them. Keep unknown metadata and unperformed human actions explicit and mark the report as a draft.

## Capability boundaries

The packaged runner supports binary/multiclass targets, six established classifiers, per-model imputation/encoding/scaling, stratified/grouped/forward-time nested CV, and declared feature/preprocessing/model-parameter/threshold sensitivity variants. See the plan reference for exact options and budget limits. It does not automatically solve every possible tabular problem: multilabel targets, custom feature engineering, exotic estimators or large sparse encodings need an explicit extension.

When the dataset needs an unsupported method, explain the limitation and extend the reusable plan/pipeline/scoring implementation with a focused synthetic test **before development**, if authorized by the analysis task. Do not silently choose a supported but inappropriate method, improvise an evaluator that bypasses the lock, or claim universal support. Changing code after locking invalidates that lock; after test access, disclose the limitation without retuning.

## IN6227 Variant 2

The assignment explicitly requires a reusable skill for tabular classification, dataset-path input, AI-selected models and preprocessing, automated PDF reporting against Variant 1's criteria, model/version and interface metadata, GitHub submission, and Reflection. Variant 1 asks for two models; Variant 2 does not explicitly impose an exact model count. This runner's two-or-three-model bound (two by default), target-resolution gate, review modes, nested CV, sensitivity framework, bounded search, optional Model Lock, sealed holdout, hashes, receipts, verification and evidence manifest are skill enhancements, not quoted grading requirements. Preserve them while optimising for justified, reproducible, data-specific and reviewable decisions rather than maximum accuracy. Include full name, matric number, `IN6227-Assignment-1`, `Variant-2`, verifiable LLM model/version, interface and GitHub URL in the report. The main PDF is at most two pages; Reflection follows outside that limit. Draft Reflection must mark unperformed human review, challenge, approval and manual verification as pending; never invent them.
