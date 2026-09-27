# Model Lock and test-access boundary

The lock contains the selected complete plan, every trained artifact and hash, training-derived class order, full estimator parameters, preprocessing, features/exclusions, metrics, threshold, seed/CV, sensitivity review, training-data/result hashes, exact Python/package versions and script hashes.

`freeze_model_lock.py` verifies OOF evidence and artifact hashes, writes a `frozen` lock and a separate `model-lock.json.seal.json` binding its SHA-256. In the default end-to-end run, the agent proceeds to the supplied held-out file without a routine human approval pause. When the user explicitly requested staged review or pre-test approval, freeze with `--require-human-approval`; `model-lock.json.approval.json` then binds the actual reviewer's statement and time to the exact lock digest. Do not edit the lock to approve it. `approve_model_lock.py` records this optional approval; it cannot authenticate that a human really gave it. Never call it without explicit approval of the displayed lock.

Before test access, `evaluate_holdout.py` checks the lock seal, any required approval, model hashes, fitted feature/class schema and unchanged code/environment. No model is fitted in this stage; full-training refits happened during development. It atomically creates `model-lock.json.holdout.json` next to the lock, then reads test bytes once and evaluates every frozen model. Changing the output folder cannot avoid the receipt. Repeated invocation is refused. A file path in a prompt does not bypass host/tool read permissions.

On failure after the receipt is created, the receipt remains consumed. Preserve it and report what happened. Do not delete the receipt, copy the lock elsewhere or refreeze to evade the one-evaluation rule. A genuinely necessary recovery requires an explicit human decision and disclosure; it is not implemented as an automatic retry. These are local workflow safeguards, not tamper-proof access control.

Test rows with missing labels still receive predictions. If no labels exist, no supervised metric is claimed. Unknown target classes or missing predictors fail visibly, without modifying the models. Verification and report generation use saved predictions/results, never reopen the held-out dataset.

Only load joblib files created by this trusted local workflow; pickle can execute code. Integrity hashes detect accidental changes, not malicious artifact origin.
