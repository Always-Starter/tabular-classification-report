# Checkpoints and commands

Run commands from the skill root with a Python environment containing `requirements.txt`. Paths below are examples to replace. Preserve each run directory; do not write into the skill or overwrite source data.

Follow [execution and review](../SKILL.md#execution-and-review). End-to-end execution is the default; staged review, a checkpoint limit or a pre-test approval pause applies only when explicitly requested. At each requested pause, show concrete results, review points and what approval would start next. Obtain host/tool permission before reading a file when required; that permission is separate from a Model Lock review pause.

## Preflight: Target resolution and input validation

This prerequisite occurs before Checkpoint 1; it is not an additional checkpoint. Inspect training schema before supervised diagnosis:

`python scripts/inspect_training_schema.py --train /data/train.csv --output /runs/run1/schema.json`

If the user or authoritative assignment/task/dataset metadata explicitly supplies `label`, add `--target label` to validate it. An absent specified column stops the workflow; do not silently substitute another. Without a supplied target, inspect the training-only `target_candidates` and `target_resolution`. A unique conventional target name with 2–20 nonmissing classes and adequate examples may yield `provisionally_inferred`: explain why it is merely a possible label, record its source and alternatives, and proceed using that name only when task context does not contradict it. Mark target confirmation pending in the report. If `unresolved`, show candidates and ask for the target before supervised diagnosis. Neither name nor class count alone proves task semantics; never choose solely by column order, filename, association or prior examples. Do not inspect any held-out file to resolve the target.

## 1. Training diagnosis

`python scripts/diagnose_training.py --train /data/train.csv --target label --output /runs/run1/diagnosis.json`

Excel adds `--sheet Data` to schema inspection and diagnosis. The diagnosis CLI requires the resolved target and does not infer it. Diagnose training only. Ask about dependent observations or unresolved high-risk feature provenance when these could invalidate the experiment. In explicitly requested staged review, show observations, uncertainties, candidate leakage/provenance issues, and a reasonable challenge point; stop before drafting a plan. Approval starts Checkpoint 2. Otherwise, document the findings and proceed if no blocking uncertainty remains.

## 2. Modelling plan

Use `tests/fixtures/example-plan.json` as a schema example, not as a default experiment. Write the actual plan from training evidence. Then:

`python scripts/validate_plan.py /runs/run1/plan.json`

Explain each candidate model's observed basis, expected strength and limitation; data-specific preprocessing; primary/secondary metric roles; optional feature handling; splits; bounded tuning and stopping; and any material sensitivity rationale. The plan proposes candidates, not a proven winner. Validation does not fit models. In explicitly requested staged review, show the validated plan, specific review questions and challengeable alternatives, then stop; approval starts Checkpoint 3 fitting. Otherwise, document the plan and proceed within its declared limits.

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

The command writes the lock and a digest seal. In the default end-to-end run, save training results, all frozen configurations, remaining uncertainties, alternatives and the digest, then proceed to the supplied held-out file without a routine approval pause. If the user requested staged review or pre-test approval, add `--require-human-approval` when freezing, show the exact digest and stop; approval starts Checkpoint 4. With training data only, skip the lock; requested staged review still pauses after development before reporting.

## 4. One-time evaluation and report (approval only when requested)

Only when the user explicitly requested a pre-test approval pause, after they approve the displayed lock:

`python scripts/approve_model_lock.py --lock /runs/run1/model-lock.json --expected-sha256 DIGEST_FROM_FREEZE --approver 'actual reviewer' --statement 'actual approval statement'`

In a default end-to-end run, skip the approval command. In either case, evaluate the frozen pipelines once:

`python scripts/evaluate_holdout.py --test /data/test.csv --lock /runs/run1/model-lock.json --model-dir /runs/run1/development --output-dir /runs/run1/holdout`

`python scripts/verify_results.py /runs/run1/holdout/test_results.json --lock /runs/run1/model-lock.json --output /runs/run1/verification.json`

Proceed to the report reference. Do not re-read test data to verify metrics: saved predictions contain the necessary evidence. If the test file has no target column, predictions are still saved but supervised metrics are undefined. If evaluation fails after access is reserved, preserve the receipt and disclose the failure; no automatic retry, relocking or retuning.

Without a separate test file, omit held-out lock approval/evaluation. In requested staged review, pause after Checkpoint 3 to review the training results; approval starts development-only reporting. Otherwise, proceed directly to that report. Never describe nested-CV values as held-out test performance.
