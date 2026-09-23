# Model Lock and test-access boundary

The lock contains the selected complete plan, every trained artifact and hash, training-derived class order, full estimator parameters, preprocessing, features/exclusions, metrics, threshold, seed/CV, sensitivity review, training-data/result hashes, exact Python/package versions and script hashes.

`freeze_model_lock.py` verifies OOF evidence and artifact hashes. The lock itself remains `pending_human_approval`; a separate `model-lock.json.approval.json` binds the actual reviewer's statement and time to its exact SHA-256. Do not edit the lock to approve it. `approve_model_lock.py` records approval; it cannot authenticate that a human really gave it. The agent must only call it after explicit approval of the displayed lock.

Before test access, `evaluate_holdout.py` checks the approval digest, model hashes, fitted feature/class schema and unchanged code/environment. No model is fitted in this stage; full-training refits happened during development. It atomically creates `model-lock.json.holdout.json` next to the lock, then reads test bytes once and evaluates every frozen model. Changing the output folder cannot avoid the receipt. Repeated invocation is refused.

On failure after the receipt is created, the receipt remains consumed. Preserve it and report what happened. Do not delete the receipt, copy the lock elsewhere or refreeze to evade the one-evaluation rule. A genuinely necessary recovery requires an explicit human decision and disclosure; it is not implemented as an automatic retry. These are local workflow safeguards, not tamper-proof access control.

Test rows with missing labels still receive predictions. If no labels exist, no supervised metric is claimed. Unknown target classes or missing predictors fail visibly, without modifying the models. Verification and report generation use saved predictions/results, never reopen the held-out dataset.

Only load joblib files created by this trusted local workflow; pickle can execute code. Integrity hashes detect accidental changes, not malicious artifact origin.
