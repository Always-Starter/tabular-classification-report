# Checkpoints and commands

Run commands from the skill root with a Python environment containing `requirements.txt`. Paths below are examples to replace. Preserve each run directory; do not write into the skill or overwrite source data.

Follow [execution and review](../SKILL.md#execution-and-review). End-to-end execution is the default; staged review, a checkpoint limit or a pre-test approval pause applies only when explicitly requested. At each requested pause, show concrete results, review points and what approval would start next. Obtain host/tool permission before reading a file when required; that permission is separate from a Model Lock review pause.

## Preflight: Target resolution and input validation

This prerequisite occurs before Checkpoint 1; it is not an additional checkpoint. Inspect training schema before supervised diagnosis:

`python scripts/inspect_training_schema.py --train /data/train.csv --output /runs/run1/schema.json`

If the user explicitly supplies `label`, add `--target label`; this overrides authoritative metadata. Otherwise use authoritative assignment/task/dataset metadata when available. An absent specified column stops the workflow; do not silently substitute another. Without either source, inspect the training-only `target_candidates`, `target_inference` and `target_resolution`. For `ai_inferred`, explain the selected column's score components, show the ranked alternatives and tie-breaker, record its source, and proceed unless visible task context contradicts it. Mark semantics confirmation pending in the report. If `unresolved`, no classification-compatible candidate exists, so ask for the target before supervised diagnosis. The heuristic combines naming, classification-compatible cardinality, storage and completeness; it predicts a column but does not prove its meaning. Never choose solely by column order, filename, association or prior examples. Do not inspect any held-out file to resolve the target.

## 1. Training diagnosis

`python scripts/diagnose_training.py --train /data/train.csv --target label --output /runs/run1/diagnosis.json`

Excel adds `--sheet Data` to schema inspection and diagnosis. The diagnosis CLI requires the resolved target and does not infer it. Diagnose training only. Ask about dependent observations or unresolved high-risk feature provenance when these could invalidate the experiment. In explicitly requested staged review, show observations, uncertainties, candidate leakage/provenance issues, and a reasonable challenge point; stop before drafting a plan. Approval starts Checkpoint 2. Otherwise, document the findings and proceed if no blocking uncertainty remains.

For each numeric column, inspect the saved `distribution_evidence`: sample skewness and its declared material threshold, excess kurtosis as a descriptive tail-weight signal, and Tukey-IQR potential-outlier counts. Keep the concepts separate. Do not claim heavy tails from skewness or extreme min/max values, do not call IQR flags data errors, and do not use "tails differ" as a generic justification. If a preprocessing choice depends on these properties, cite the relevant measured field and retain its limitation.

## 2. Modelling plan

Use `tests/fixtures/example-plan.json` as a schema example, not as a default experiment. Write the actual plan from training evidence. Then:

`python scripts/validate_plan.py /runs/run1/plan.json`

Explain each candidate model's observed basis, expected strength and limitation; data-specific preprocessing; class-weight decision; primary/secondary metric roles; optional feature handling; dependency-aware splits; threshold policy; compute budget; bounded tuning and stopping; and any material sensitivity rationale. For every fixed estimator parameter, record both why it was not tuned and the source/rationale for its exact value; schema v6 specifically requires explicit `max_iter`, ensemble `n_estimators`, and SVM `calibration_cv` unless tuned. Record runner execution controls separately. Predeclare the executable final-family policy, including practical-tie tolerance provenance and complete preference order. Classify decision bases using the schema-v6 categories. The plan proposes candidates, not a proven winner. Validation does not fit models. In explicitly requested staged review, show the validated plan, specific review questions and challengeable alternatives, then stop; approval starts Checkpoint 3 fitting. Otherwise, document the plan and proceed within its declared limits.

## 3. Development and optional Model Lock

`python scripts/run_nested_cv.py --train /data/train.csv --plan /runs/run1/plan.json --output-dir /runs/run1/development`

The development directory must be empty. All candidate and sensitivity preprocessing is fitted inside nested CV; final candidates are refitted on all eligible training rows. Before fitting, SVM plans are checked against every outer-inner, outer-refit, final-inner and full-refit training subset for calibration class-count feasibility. Inner CV selects hyperparameters and, only for a declared tuned policy, the threshold; outer CV evaluates that complete selection procedure. The runner then applies the predeclared final-family policy and saves `model_selection`. Review that evidence, per-fold/final threshold evidence, fold variability, numeric-grid boundary metadata, warnings and any sensitivities. Sensitivity variants are labelled `interpretive_sensitivity` and cannot be selected or locked; if one motivates a change, create a new independent plan/run with that configuration as the baseline. A two-value endpoint indicates coarse coverage only because either choice must be an endpoint. An edge selected after interior candidates were evaluated supports a future independent untested-direction question, not automatic widening of the current run. Do not treat a tiny score difference as proof of superiority. If an independent holdout exists, write `review.json` with `selected_variant` fixed to `baseline` and `preferred_model` copied from `model_selection.selected_model`:

```json
{
  "selected_variant": "baseline",
  "preferred_model": "SELECTED_BY_EXECUTABLE_POLICY",
  "rationale": "Replace with evidence from the actual train-only comparison.",
  "selection_rule": "Apply the predeclared primary-metric comparison on shared outer folds, then the declared non-score considerations.",
  "tie_breaker": "State the actually applied interpretability/stability/compute tie-breaker, or that no tie-breaker was needed.",
  "sensitivity_review": "Interpret each declared sensitivity without selecting it; state whether it motivates a future independent run.",
  "warnings_review": "Explain any recorded warning or state that none occurred."
}
```

`python scripts/freeze_model_lock.py --results /runs/run1/development/training_results.json --model-dir /runs/run1/development --review /runs/run1/review.json --output /runs/run1/model-lock.json`

The command writes the lock and a digest seal. In the default end-to-end run, save training results, all frozen configurations, remaining uncertainties, alternatives and the digest, then proceed to the supplied held-out file without a routine approval pause. If the user requested staged review or pre-test approval, add `--require-human-approval` when freezing, show the exact digest and stop; approval starts Checkpoint 4. With training data only, skip the lock; requested staged review still pauses after development before reporting.

## 4. One-time evaluation and report (approval only when requested)

Only when the user explicitly requested a pre-test approval pause, after they approve the displayed lock:

`python scripts/approve_model_lock.py --lock /runs/run1/model-lock.json --expected-sha256 DIGEST_FROM_FREEZE --approver 'actual reviewer' --statement 'actual approval statement'`

In a default end-to-end run, skip the approval command. In either case, evaluate the frozen pipelines once:

`python scripts/evaluate_holdout.py --test /data/test.csv --lock /runs/run1/model-lock.json --model-dir /runs/run1/development --output-dir /runs/run1/holdout`

`python scripts/verify_results.py /runs/run1/holdout/test_results.json --lock /runs/run1/model-lock.json --output /runs/run1/verification.json`

Proceed to the report reference. Do not re-read test data to verify metrics: saved predictions contain the necessary evidence. If the test file has no target column, predictions are still saved but supervised metrics are undefined. If evaluation fails after access is reserved, preserve the receipt and disclose the failure; no automatic retry, relocking or retuning.

Without a separate test file, omit held-out lock approval/evaluation. In requested staged review, pause after Checkpoint 3 to review the training results; approval starts development-only reporting. Otherwise, proceed directly to that report. Never describe nested-CV values as held-out test performance.
