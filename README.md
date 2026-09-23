# Tabular Classification Report

A reusable Codex skill for selecting and comparing two or three classifiers from training-data evidence, with configurable preprocessing/metrics, nested CV, optional one-time held-out evaluation and automatic Markdown/PDF reporting. Two is the usual choice; a third needs a distinct, documented comparison question. This range is an implementation choice, not an explicit Variant 2 model-count rule. No course dataset, model or assessment result is bundled.

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

User-facing prompts and results default to English. If your current request is clearly in another language, the agent uses that language; you can also state a language preference explicitly. A new English-language request is answered in English even if earlier messages used another language. Dataset field names and label values stay as written in the data.

You do not need to know checkpoint names. Unless you have already chosen a mode for this run, the agent must **show the following choice before it reads the dataset**. Simply telling you which mode it assumed does not fulfill this step:

| Mode | When the agent pauses | Suitable for |
| --- | --- | --- |
| Staged review (default) | After the data diagnosis, after the modelling plan, and after training results and Model Lock | First use, learning and reviewing decisions |
| Continuous execution | Before held-out evaluation, after presenting the Model Lock; also when a blocking issue needs your input | Users who want the agent to make intermediate decisions |

The question is: “Choose staged review (default) or continuous execution?” If the interface lets you answer while the agent works and you have not answered yet, it proceeds at most through the diagnosis under staged review, stopping sooner if the target is unresolved. Otherwise, it waits for your choice before reading the data. At each pause it shows results, explains what needs your review and states the next action. You can request a pause or switch modes during the run. A general request to “run everything” does not select continuous execution.

Before supervised diagnosis, the agent inspects the training schema and validates a target explicitly supplied by you or authoritative task/dataset metadata. If none is available, it asks; it never guesses from the last column, a label-like name or statistical patterns. `python scripts/inspect_training_schema.py --train /absolute/path/train.csv` performs this schema-only check (add `--target name` to validate a supplied column). The diagnosis command separately requires `--target`.

For staged review:

```text
Use $tabular-classification-report in staged review mode on /absolute/path/train.csv,
target label. The held-out file is /absolute/path/test.csv.
Save outputs under /absolute/path/run1.
```

For continuous execution:

```text
Use $tabular-classification-report in continuous execution mode on
/absolute/path/train.csv, target label. The held-out file is /absolute/path/test.csv.
Save outputs under /absolute/path/run2. Show the Model Lock for my approval before testing.
```

In either mode, the held-out data stays sealed until you approve the displayed Model Lock. Continuous execution therefore does not mean unattended test evaluation. You can also request a specific limit, such as “Checkpoint 1 only”; the agent honours it in either mode. With no test file, staged review still pauses after training results before reporting, while continuous execution proceeds to a development-only report. See [checkpoint commands](references/checkpoint-workflow.md).

The agent proposes data-appropriate candidate families and preprocessing; Python executes the reviewed plan. The primary metric, bounded grid and stopping rule are determined before fitting. Development results flag selected numeric-grid edges as uncertainty without automatically widening the search. Six model families, binary/multiclass targets, configurable metrics and per-model preprocessing are supported. Grouped and forward-time nested CV are available. Unsupported models or data structures are rejected explicitly and require a tested extension. This is not an unlimited AutoML package. See [plan contract](references/model-plan-schema.md).

## Test and reproduce

`python -m unittest discover -s tests -v` uses only synthetic fixtures in temporary folders. It tests successful workflows and refusal paths, including approval, tampered models, repeat evaluation, label ordering, multiclass, CV dependence and generated reports. It never accesses course files. Fixtures are not recommended modelling defaults.

For a visible self-test, run `python tests/smoke_demo.py --output-dir /absolute/path/to/new-demo-folder`. It generates its own data, runs train-only development and writes a sample PDF/Markdown report. It cannot accept or read your course dataset. The sample is marked as a draft and is not a course submission.

`requirements.txt` declares supported ranges. `requirements-tested.txt` records the exact direct package versions used for this revision's verification (not a cross-platform transitive lock). Every run also saves exact Python/package versions and code hashes, and evaluation requires the same environment as training. Preserve the virtual environment or record `pip freeze` for a fully reproducible deployment.

Approval and one-time receipts are auditable local workflow controls, not tamper-proof security. A failed attempt after test access remains recorded; do not remove it and rerun. Load only trusted local joblib artifacts.

## GitHub handoff

Publish this entire directory, not runtime output folders. `.gitignore` excludes environments, trained models and generated results. Replace the repository metadata in the actual report after publishing; no GitHub URL is invented here. The report is automatically generated as two main pages plus a Reflection draft. The student must provide their details, verify an output themselves, and confirm the Reflection before submission.

The skill remains generic: do not describe a supplied dataset as customer churn or assign business meanings without source evidence. Four checkpoints, nested CV and a Model Lock are implementation choices; IN6227 grades justified reasoning, reusable implementation, report and Reflection.
