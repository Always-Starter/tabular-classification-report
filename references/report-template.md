# Evidence-based report and Reflection

Use [narrative-template.json](narrative-template.json) for the input structure. Replace explanatory text with actual evidence and reasoning. Do not supply invented identity, model-version, GitHub URL or human actions. Record the verifiable model/version and interface at execution time; if unavailable, leave it blank and keep draft status.

Required narrative fields: exploration, preprocessing, features, model_rationale, findings, limitations. Schema-v5 reports also require an `evidence_summary` with `fact`, `interpretation`, `limitation_unknown`, `decision` and `future_work`; the renderer labels these explicitly. Prioritise data diagnosis, preprocessing/feature rationale, candidate-model rationale, bounded tuning and stopping, evaluation/metrics/comparison, findings and required metadata in the two-page main report. Keep hashes, receipts and evidence-manifest mechanics in artifacts rather than filling the report with them. Required Reflection fields: human_oversight, challenged_decision, manual_verification, future_changes. Set `human_reflection_confirmed` true only after the student confirms the account accurately describes their actions. A concise draft should normally use 1–3 sentences per field. If the report overflows, shorten prose/select a compact metric set before the experiment; do not remove locked metrics after seeing performance to make the models look better.

Distribution language must be auditable against `diagnosis.json`. Report material skewness only when a column meets the saved absolute-skew threshold. Describe high excess kurtosis as a descriptive signal consistent with heavier-than-normal tails, not proof of a heavy-tailed distribution. Describe Tukey-IQR results as potential/statistical outlier candidates, not confirmed errors, unless independent domain review supplies that conclusion. Do not infer heavy tails from skewness, min/max or separated quantiles, and do not write vague justifications such as "distribution tails differ". The renderer rejects these common unsupported formulations so a narrative must cite the measured evidence it relies on.

For each model, report fixed settings as predeclared rather than CV-selected. Summarise separately why each parameter was not tuned and the source/rationale for its exact value; if a legacy or explicit-unknown plan lacks the latter, say that its exact-value basis is unknown. For numeric endpoint selections, distinguish a two-value grid (inevitable endpoint and coarse coverage only) from an edge selected after interior candidates were evaluated (an untested-direction question for a future independent run). Neither case authorizes retuning a locked run, and a two-value endpoint must not be presented as evidence that the optimum lies outside the evaluated range.

Report a fixed threshold with value, source, exact-value rationale, non-tuning reason and timing. Report tuned thresholds as inner-CV selections for each final model, never as held-out optimization. Report model-default behavior as such. Do not use “optimal” or “best” where only the best tested candidate within a bounded search was established.

With a held-out result:

```bash
python scripts/generate_report.py \
  --training /runs/run1/development/training_results.json \
  --diagnosis /runs/run1/diagnosis.json \
  --narrative /runs/run1/narrative.json \
  --test-results /runs/run1/holdout/test_results.json \
  --lock /runs/run1/model-lock.json \
  --output-dir /runs/run1/report
```

Without a separate test dataset, omit `--test-results` and `--lock`. The report uses the selection-eligible baseline; sensitivity variants remain interpretive evidence and cannot be selected with `--variant`. To adopt one, create a new independent plan/run where it is the baseline. This generates a development-only report, clearly labelled as such.

Outputs: report.md, report.pdf, report_manifest.json. The PDF main report is two pages, followed by Reflection in the same document. Page-count validation refuses a main report longer than two pages; it does not shrink text to illegibility. The opening summary highlights the frozen pre-test choice and its development and held-out primary scores without implying that held-out performance changed the choice. The comparison table uses full model-family names and visually emphasizes the prespecified primary metric; it displays that metric and the first three secondary metrics in plan order. Every declared metric remains in the verified result files. Binary confusion counts are shown in a labelled comparison table; multiclass counts remain explicitly labelled prose. Model settings are presented as readable prose rather than raw JSON. Tables are generated from verified saved metrics, never chosen after viewing holdout scores. Arithmetic aids stay in the manifest rather than appearing as an automated claim in the Reflection. A manifest binds report/narrative/result hashes and records if the renderer changed after the model lock. If only the report renderer changed after locking, a presentation-only rerender requires the explicit `--presentation-only-rerender` flag and a fresh output directory. This uses saved results, preserves the original locked run and never reopens the held-out dataset or revises model choices; changes to any other locked script are refused. Missing personal/LLM/repository metadata or unconfirmed Reflection keeps the document marked DRAFT.

Before delivery, render the PDF and inspect all pages for clipping, oversized tables, readable fonts and correct Reflection placement. Numerical verification cannot validate every possible free-text interpretation: compare narrative claims with the saved evidence even after the automated distribution-language guard passes. Do not claim this template is an already-completed course report.

The renderer embeds ReportLab's bundled TrueType font by default. For Chinese or other characters outside that font's coverage, pass `--font /absolute/path/to/a/covering.ttf`. Missing glyphs fail explicitly before PDF generation instead of producing blank text. Choose an embeddable font installed in the execution environment and visually verify the output.

For manual verification, use an explicitly labelled confusion matrix: for any number of classes, accuracy = sum(diagonal) / sum(all cells). For binary precision, locate the locked positive class rather than assuming it is the second row/column. The verifier supplies arithmetic candidates, but the student must actually check one and explain it in their own Reflection.
