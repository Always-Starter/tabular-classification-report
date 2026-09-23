# Checkpoints and commands

Run commands from the skill root with a Python environment containing `requirements.txt`. Paths below are examples to replace. Preserve each run directory; do not write into the skill or overwrite source data.

## 1. Training diagnosis

`python scripts/diagnose_training.py --train /data/train.csv --target label --output /runs/run1/diagnosis.json`

Excel adds `--sheet Data`. Diagnose training only. In checkpoint mode, report findings and pause. Ask about ambiguous target, dependent observations or unresolved high-risk feature provenance.

## 2. Modelling plan

Use `tests/fixtures/example-plan.json` as a schema example, not as a default experiment. Write the actual plan from training evidence. Then:

`python scripts/validate_plan.py /runs/run1/plan.json`

Explain models, preprocessing, metrics, features, splits, tuning limits, stopping criteria and sensitivity rationales. Validation does not fit models. In checkpoint mode, pause for review.

## 3. Development and Model Lock

`python scripts/run_nested_cv.py --train /data/train.csv --plan /runs/run1/plan.json --output-dir /runs/run1/development`

The development directory must be empty. All candidate and sensitivity preprocessing is fitted inside nested CV; final candidates are refitted on all eligible training rows. Review `training_results.json`, fold scores and warnings. Write `review.json`:

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

Show both frozen configurations and the printed digest. Stop before accessing the test file until this exact lock is approved.

## 4. Approval, one-time evaluation and report

After the user explicitly approves the displayed lock:

`python scripts/approve_model_lock.py --lock /runs/run1/model-lock.json --expected-sha256 DIGEST_FROM_FREEZE --approver 'actual reviewer' --statement 'actual approval statement'`

`python scripts/evaluate_holdout.py --test /data/test.csv --lock /runs/run1/model-lock.json --model-dir /runs/run1/development --output-dir /runs/run1/holdout`

`python scripts/verify_results.py /runs/run1/holdout/test_results.json --lock /runs/run1/model-lock.json --output /runs/run1/verification.json`

Proceed to the report reference. Do not re-read test data to verify metrics: saved predictions contain the necessary evidence. If the test file has no target column, predictions are still saved but supervised metrics are undefined. If evaluation fails after access is reserved, preserve the receipt and disclose the failure; no automatic retry, relocking or retuning.

Without a separate test file, omit lock approval/evaluation and generate a development-only report. Never describe nested-CV values as held-out test performance.
