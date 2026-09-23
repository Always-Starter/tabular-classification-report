# Evidence-based report and Reflection

Use [narrative-template.json](narrative-template.json) for the input structure. Replace explanatory text with actual evidence and reasoning. Do not supply invented identity, model-version, GitHub URL or human actions. Record the verifiable model/version and interface at execution time; if unavailable, leave it blank and keep draft status.

Required narrative fields: exploration, preprocessing, features, model_rationale, findings, limitations. Prioritise data diagnosis, preprocessing/feature rationale, candidate-model rationale, bounded tuning and stopping, evaluation/metrics/comparison, findings and required metadata in the two-page main report. Keep hashes, receipts and evidence-manifest mechanics in artifacts rather than filling the report with them. Required Reflection fields: human_oversight, challenged_decision, manual_verification, future_changes. Set `human_reflection_confirmed` true only after the student confirms the account accurately describes their actions. A concise draft should normally use 1–3 sentences per field. If the report overflows, shorten prose/select a compact metric set before the experiment; do not remove locked metrics after seeing performance to make the models look better.

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

Without a separate test dataset, omit `--test-results` and `--lock`; optionally select a train-only `--variant`. This generates a development-only report, clearly labelled as such.

Outputs: report.md, report.pdf, report_manifest.json. The PDF main report is two pages, followed by Reflection in the same document. Page-count validation refuses a main report longer than two pages; it does not shrink text to illegibility. To keep the PDF comparable across two or three models, its table displays the prespecified primary and first two secondary metrics in plan order; every declared metric remains in the verified result files. Tables are generated from verified saved metrics, never chosen after viewing holdout scores. A manifest binds report/narrative/result hashes. Missing personal/LLM/repository metadata or unconfirmed Reflection keeps the document marked DRAFT.

Before delivery, render the PDF and inspect all pages for clipping, oversized tables, readable fonts and correct Reflection placement. Numerical verification cannot validate free-text interpretations: compare narrative claims with the saved evidence. Do not claim this template is an already-completed course report.

The renderer embeds ReportLab's bundled TrueType font by default. For Chinese or other characters outside that font's coverage, pass `--font /absolute/path/to/a/covering.ttf`. Missing glyphs fail explicitly before PDF generation instead of producing blank text. Choose an embeddable font installed in the execution environment and visually verify the output.

For manual verification, use an explicitly labelled confusion matrix: for any number of classes, accuracy = sum(diagonal) / sum(all cells). For binary precision, locate the locked positive class rather than assuming it is the second row/column. The verifier supplies arithmetic candidates, but the student must actually check one and explain it in their own Reflection.
