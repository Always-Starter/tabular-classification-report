# Checkpoints and commands

Run commands from the skill root with a Python environment containing `requirements.txt`. Paths below are examples to replace. Preserve each run directory; do not write into the skill or overwrite source data.

Use the [review mode selected at startup](../SKILL.md#choose-the-review-mode). Staged review is the default; continuous execution requires an explicit choice. At each staged pause, show concrete results, review points and what approval would start next. A user-requested checkpoint limit applies in either mode.

## Preflight: Target resolution and input validation

This prerequisite occurs before Checkpoint 1; it is not an additional checkpoint. Inspect training schema before supervised diagnosis:

`python scripts/inspect_training_schema.py --train /data/train.csv --output /runs/run1/schema.json`

If the user or authoritative assignment/task/dataset metadata explicitly supplies `label`, add `--target label` to validate it. Otherwise show the columns, ask for the target and stop. An absent specified column also stops the workflow; do not silently substitute another. Record the source of the target choice. Column order, label-like names, apparent class count, filenames, associations and prior examples are not task semantics. Do not inspect any held-out file to resolve the target.

## 1. Training diagnosis

`python scripts/diagnose_training.py --train /data/train.csv --target label --output /runs/run1/diagnosis.json`

Excel adds `--sheet Data` to schema inspection and diagnosis. The diagnosis CLI requires the resolved target and does not infer it. Diagnose training only. Ask about dependent observations or unresolved high-risk feature provenance. In staged review, show observations, uncertainties, candidate leakage/provenance issues, and a reasonable challenge point; stop before drafting a plan. Approval starts Checkpoint 2. In continuous execution, explain the findings and proceed if no blocking uncertainty remains.

## 2. Modelling plan

Use `tests/fixtures/example-plan.json` as a schema example, not as a default experiment. Write the actual plan from training evidence. Then:

`python scripts/validate_plan.py /runs/run1/plan.json`

Explain each candidate model's observed basis, expected strength and limitation; data-specific preprocessing; primary/secondary metric roles; optional feature handling; splits; bounded tuning and stopping; and any material sensitivity rationale. The plan proposes candidates, not a proven winner. Validation does not fit models. In staged review, show the validated plan, specific review questions and challengeable alternatives, then stop; approval starts Checkpoint 3 fitting. In continuous execution, explain the plan and proceed within its declared limits.

## 3. Development and optional Model Lock

`python scripts/run_nested_cv.py --train /data/train.csv --plan /runs/run1/plan.json --output-dir /runs/run1/development`

The development directory must be empty. All candidate and sensitivity preprocessing is fitted inside nested CV; final candidates are refitted on all eligible training rows. Inner CV selects hyperparameters; outer CV evaluates the tuned procedure. Review `training_results.json`, fold variability, near-ties, numeric-grid boundary flags, warnings and any sensitivities. Do not automatically widen an edge-hit grid or treat a tiny score difference as proof of superiority. If an independent holdout exists, write `review.json`:

```json
{
  "selected_variant": "baseline",
  "preferred_model": "linear",
  "rationale": "Replace with evidence from the actual train-only comparison.",
  "sensitivity_review": "Explain each declared sensitivity and the selected configuration; state why none was needed if applicable.",
  "warnings_review": "Explain any recorded warning or state that none occurred."
}
```

`python scripts/freeze_model_lock.py --results /runs/run1/development/training_results.json --model-dir /runs/run1/development --review /runs/run1/review.json --output /runs/run1/model-lock.json`

Show training results, all frozen configurations, remaining uncertainties, challengeable alternatives and the printed digest. In both modes, stop before accessing the test file until this exact lock is approved; approval starts Checkpoint 4. Selecting continuous execution is not approval of a lock that has not yet been displayed. With training data only, skip the lock and held-out approval; staged review still pauses after development before reporting.

## 4. Approval, one-time evaluation and report

After the user explicitly approves the displayed lock:

`python scripts/approve_model_lock.py --lock /runs/run1/model-lock.json --expected-sha256 DIGEST_FROM_FREEZE --approver 'actual reviewer' --statement 'actual approval statement'`

`python scripts/evaluate_holdout.py --test /data/test.csv --lock /runs/run1/model-lock.json --model-dir /runs/run1/development --output-dir /runs/run1/holdout`

`python scripts/verify_results.py /runs/run1/holdout/test_results.json --lock /runs/run1/model-lock.json --output /runs/run1/verification.json`

Proceed to the report reference. Do not re-read test data to verify metrics: saved predictions contain the necessary evidence. If the test file has no target column, predictions are still saved but supervised metrics are undefined. If evaluation fails after access is reserved, preserve the receipt and disclose the failure; no automatic retry, relocking or retuning.

Without a separate test file, omit held-out lock approval/evaluation. In staged review, still pause after Checkpoint 3 to review the training results; approval starts development-only reporting. In continuous execution, proceed directly to that report. Never describe nested-CV values as held-out test performance.
