# Tabular Classification Report

A reusable Agent Skill for selecting and comparing two or three classifiers from training-data evidence, with configurable preprocessing/metrics, nested CV, a pre-fit executable final-family policy, optional one-time held-out evaluation and automatic Markdown/PDF reporting. It was developed and tested with Codex. Claude Code also supports the `SKILL.md` format, but this package has not been tested there. Two models are the usual choice; a third needs a distinct, documented comparison question. This range is an implementation choice, not an explicit Variant 2 model-count rule. No course dataset, model or assessment result is bundled.

## Installation

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

Claude Code users can place the whole folder under `~/.claude/skills/tabular-classification-report` and invoke `/tabular-classification-report`, following the [Claude Code skill documentation](https://code.claude.com/docs/en/skills). This is a format-based installation route, not a claim that this package has been tested in Claude Code.

## Usage

User-facing prompts and results default to English. If your current request is clearly in another language, the agent uses that language; you can also state a language preference explicitly. A new English-language request is answered in English even if earlier messages used another language. Dataset field names and label values stay as written in the data.

You do not need to know checkpoint names or choose a mode. By default, a request with the necessary training details runs end to end: diagnosis, modelling plan, nested-CV training, optional one-time held-out evaluation and draft report. The agent documents its choices without routine approval pauses. If you explicitly request staged review, it stops after the diagnosis, plan and training results; with a held-out file, it also waits for your approval of the exact Model Lock before testing.

The agent still stops when no classification-compatible target can be inferred, visible task context contradicts the guess, a file is inaccessible or a material data-validity problem requires your input. Providing a file path does not override system permissions; if the host asks for read access, that permission must be granted before the agent continues. Missing report identity metadata can remain marked pending in a draft rather than blocking the analysis.

Before supervised diagnosis, the agent validates a target explicitly supplied by you (which takes precedence) or authoritative task/dataset metadata. If neither is available, `python scripts/inspect_training_schema.py --train /absolute/path/train.csv` ranks classification-compatible columns using recorded naming, cardinality, storage and completeness evidence (add `--target name` to validate a supplied column). It predicts the highest-ranked candidate, discloses every score component and tie-breaker, and marks task/label semantics unconfirmed. Neither column position nor low cardinality alone determines the label. The diagnosis command still requires an explicit `--target`, including when the preflight inferred one.

For the default end-to-end run:

```text
Use $tabular-classification-report on /absolute/path/train.csv,
target label. The held-out file is /absolute/path/test.csv.
Save outputs under /absolute/path/run1.
```

To review decisions one checkpoint at a time:

```text
Use $tabular-classification-report in staged review mode on
/absolute/path/train.csv, target label. The held-out file is /absolute/path/test.csv.
Save outputs under /absolute/path/run2. Wait for my review after each checkpoint.
```

In either mode, the held-out data stays sealed through training development, and no model decision changes after held-out results. You can request a specific limit, such as “Checkpoint 1 only”. With no test file, the skill produces a development-only report; staged review still pauses after training results before reporting. See [checkpoint commands](references/checkpoint-workflow.md).

The agent proposes data-appropriate candidate families and preprocessing; Python executes the validated plan. The primary metric, threshold policy, bounded grid, compute budget and stopping rule are determined before fitting. A tuned binary threshold is selected only inside inner CV; fixed and model-default policies retain their declared provenance. Development results flag selected numeric-grid edges as uncertainty without automatically widening the search. Seven model families, binary/multiclass targets, configurable metrics and per-model preprocessing are supported. Grouped and forward-time nested CV are available. Unsupported models, resampling strategies or data structures are rejected explicitly and require a tested extension. This is not an unlimited AutoML package. See [plan contract](references/model-plan-schema.md).

## Tests and reproducibility

`python -m unittest discover -s tests -v` uses only synthetic fixtures in temporary folders. It tests both end-to-end and requested staged workflows, including lock integrity, approval when required, tampered models, repeat evaluation, label ordering, multiclass, CV dependence and generated reports. These fixtures are for software checks only: they are not used when the skill is run on an instructor's own training and test files, and they are not recommended modelling defaults.

For a visible self-test, run `python tests/smoke_demo.py --output-dir /absolute/path/to/new-demo-folder`. It generates its own data, runs train-only development and writes a sample PDF/Markdown report. It cannot accept or read your course dataset. The sample is marked as a draft and is not a course submission.

`requirements.txt` declares supported dependency ranges. This package has been tested with Python 3.12; it does not require a particular set of pinned package versions. Every run saves the actual Python/package versions and code hashes, and held-out evaluation requires the same environment as training. Preserve the virtual environment or record `pip freeze` when exact environment reproduction is needed.

Lock seals, optional approval records and one-time receipts are auditable local workflow controls, not tamper-proof security. A failed attempt after test access remains recorded; do not remove it and rerun. Load only trusted local joblib artifacts.

## Submission notes

The full skill is published at [Always-Starter/tabular-classification-report](https://github.com/Always-Starter/tabular-classification-report) for instructors to clone and inspect. Do not commit runtime output folders; `.gitignore` excludes environments, trained models and generated results. Supply the repository URL and your own details in the actual report metadata. The report is automatically generated as two main pages plus a Reflection draft. The student must verify an output themselves and confirm the Reflection before submission.

The skill remains generic: do not describe a supplied dataset as customer churn or assign business meanings without source evidence. Four checkpoints, nested CV and a Model Lock are implementation choices; IN6227 grades justified reasoning, reusable implementation, report and Reflection.
