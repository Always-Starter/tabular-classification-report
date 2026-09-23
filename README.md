# Tabular Classification Report

A reusable Codex skill for selecting and comparing two classifiers from training-data evidence, with configurable preprocessing/metrics, nested CV, optional one-time held-out evaluation and automatic Markdown/PDF reporting. No course dataset, model or assessment result is bundled.

## Install from a local clone

Requires Python 3.10+ (tested with 3.12). After cloning/downloading this repository, copy the whole skill directory, including scripts and references, to `~/.agents/skills/tabular-classification-report`. Do not copy only SKILL.md. If the repository root is the skill directory, the local steps are:

```bash
mkdir -p ~/.agents/skills
cp -R /absolute/path/to/cloned-skill ~/.agents/skills/tabular-classification-report
cd ~/.agents/skills/tabular-classification-report
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

Use a fresh destination; do not nest a copy inside an existing skill. A symlink to the local clone is an alternative for development. In Codex, invoke `$tabular-classification-report`; if discovery has not refreshed, start a new task. A portable fallback is: “Read and follow /absolute/path/to/tabular-classification-report/SKILL.md”. Tell the agent to use this skill's `.venv/bin/python`.

## Try it yourself

```text
Use $tabular-classification-report. Run Checkpoint 1 only on /absolute/path/train.csv,
target label. Keep /absolute/path/test.csv sealed. Save outputs under /absolute/path/run1.
```

You can also request the entire analysis from a dataset path. The agent still stops after Checkpoints 1, 2 and 3 for your explicit review; a request to run both train and test data does not waive these stops. The agent asks about an ambiguous target and requires approval of the exact Model Lock before held-out access. With no test file, it can generate a development-only report after the required reviews. See [checkpoint commands](references/checkpoint-workflow.md).

The agent chooses the plan; Python executes it. Six model families, binary/multiclass targets, configurable metrics and per-model preprocessing are supported. Grouped and forward-time nested CV are available. Unsupported models or data structures are rejected explicitly and require a tested extension. This is not an unlimited AutoML package. See [plan contract](references/model-plan-schema.md).

## Test and reproduce

`python -m unittest discover -s tests -v` uses only synthetic fixtures in temporary folders. It tests successful workflows and refusal paths, including approval, tampered models, repeat evaluation, label ordering, multiclass, CV dependence and generated reports. It never accesses course files. Fixtures are not recommended modelling defaults.

For a visible self-test, run `python tests/smoke_demo.py --output-dir /absolute/path/to/new-demo-folder`. It generates its own data, runs train-only development and writes a sample PDF/Markdown report. It cannot accept or read your course dataset. The sample is marked as a draft and is not a course submission.

`requirements.txt` declares supported ranges. `requirements-tested.txt` records the exact direct package versions used for this revision's verification (not a cross-platform transitive lock). Every run also saves exact Python/package versions and code hashes, and evaluation requires the same environment as training. Preserve the virtual environment or record `pip freeze` for a fully reproducible deployment.

Approval and one-time receipts are auditable local workflow controls, not tamper-proof security. A failed attempt after test access remains recorded; do not remove it and rerun. Load only trusted local joblib artifacts.

## GitHub handoff

Publish this entire directory, not runtime output folders. `.gitignore` excludes environments, trained models and generated results. Replace the repository metadata in the actual report after publishing; no GitHub URL is invented here. The report is automatically generated as two main pages plus a Reflection draft. The student must provide their details, verify an output themselves, and confirm the Reflection before submission.

The skill remains generic: do not describe a supplied dataset as customer churn or assign business meanings without source evidence. Four checkpoints, nested CV and a Model Lock are implementation choices; IN6227 grades justified reasoning, reusable implementation, report and Reflection.
