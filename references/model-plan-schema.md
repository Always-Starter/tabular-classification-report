# Executable plan contract (schema v5)

`scripts/validate_plan.py` is the executable validator. [../tests/fixtures/example-plan.json](../tests/fixtures/example-plan.json) is a complete binary example, not a recommended real-data plan. Schemas v2-v4 remain readable for preserved runs; create new plans as v5. Legacy scalar thresholds are read with unknown provenance rather than receiving reconstructed explanations.

## Decision classes and top-level fields

Schema v5 separates `invariant_methodological_rule`, `dataset_specific_evidence`, `user_domain_constraint`, `conventional_default`, and `unknown`. Every decision-trace record has a topic, one of those bases, and Observation / Decision / Rationale / Human-review point. Required topics cover target semantics, feature typing, preprocessing, imbalance, model shortlist, metric, CV, decision rule and compute budget.

| Field | Contract |
| --- | --- |
| schema_version | 5 |
| target, task | Column name; `binary` or `multiclass` |
| features and feature groups | Ordered predictors partitioned exactly into numeric/categorical lists |
| excluded_features | Every unused predictor mapped to a reason |
| semantics | Target source/meaning, positive-class meaning, FP/FN costs and row dependence; null/unknown stays unknown |
| feature_provenance | Material derivation, outcome-information and prediction-time-availability records |
| metrics | Predeclared primary, interpretive secondary metrics, rationale and confirmed/provisional status |
| cv | Strategy/splits plus rationale, status and decision basis; grouped/time columns are excluded predictors |
| compute_budget | Positive `max_explicit_fits`, source and rationale; internal SVM calibration is additional work |
| candidate_selection | Exact registry snapshot, shortlist basis/rationale and considered alternatives |
| models | Two or three bounded candidate specifications in this implementation |
| sensitivities | Zero to three predeclared interpretive alternatives; never selection-eligible or lockable |

## Binary decision-threshold policy

Binary schema-v5 plans use `threshold_policy`; they do not use a scalar `threshold`. Multiclass plans use argmax and omit both threshold fields.

- `fixed`: record a value in [0,1], its source, the exact-value rationale (or source `unknown` with null rationale), why it was not tuned, and `predeclared_before_development` timing. A conventional 0.5 value is allowed only as an explicitly labelled baseline, not an operational optimum.
- `tuned`: value is null in the plan; declare 2-21 distinct search values, a threshold-dependent primary objective, `joint_inner_cv_grid`, `inner_cv` provenance and `inner_cv_only` scope. The runner jointly selects model parameters and threshold independently inside every outer fold, then again in the final full-training inner search. The selected value is frozen per final model.
- `model_default`: record `estimator_default_class_decision`, model-default provenance, the rationale and why threshold tuning was not performed. The auditable implementation uses maximum frozen-class probability.

Held-out data never select or revise a threshold. A threshold sensitivity is a predeclared alternative configuration, not threshold optimization.

## Metrics, CV and feasibility

Supported metrics are `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`, `log_loss`; binary also supports `f1`, `precision`, `recall`, `roc_auc`, `average_precision`; multiclass supports `roc_auc_ovr_macro`. Only log loss is minimized. A tuned threshold currently requires a threshold-dependent primary metric so one objective governs joint inner selection. Ranking/probability objectives can use a fixed or model-default class decision.

CV strategies are shuffled stratified, stratified group and forward time. Split counts are 2-10 but are dataset decisions: training checks class support, groups and timestamp ordering before fitting, with no silent fallback. Group/time columns must be excluded predictors. Unknown row dependence must be reported as a limitation, not converted to independence.

Binary denominator-zero precision/recall/F1 values are null with reasons. Macro/weighted metrics use frozen classes. Every training scoring fold must contain every class. Secondary summaries become null if any fold is undefined; the primary must be defined.

## Per-model specification

Each model records name, registry type, fixed `params`, `fixed_param_rationale`, bounded estimator `grid`, preprocessing, `preprocessing_rationale`, `imbalance_handling`, family rationale, grid rationale and stopping rule. Registry types are logistic regression, random forest, extra trees, decision tree, KNN, Gaussian Naive Bayes and calibrated SVM.

For every fixed parameter, separately record why it was not tuned and where its exact value came from. Sources are training diagnosis, compute budget, convergence requirement, user/authoritative requirement, domain constraint, prior independent evidence, literature, heuristic, implementation constraint, predeclared rule, library default or unknown. Unknown requires a null exact-value rationale.

`imbalance_handling` explicitly declares `none`, `fixed_class_weight` or `tuned_class_weight`, its basis and rationale, and must agree with `class_weight` in params/grid. Resampling is not bundled; a dataset that needs it requires a leakage-safe tested extension. Metric selection must not be inferred from prevalence alone.

Preprocessing supports per-model numeric imputation (median/mean/most-frequent/constant), optional missing indicators/log1p, none/standard/robust scaling, categorical imputation and one-hot/ordinal encoding. Learned preprocessing is always fitted inside folds. Numeric storage does not prove continuous semantics; the plan must preserve the feature-typing basis in its decision trace.

Only estimator parameters are in `grid`; preprocessing/feature/threshold alternatives use declared interpretive sensitivities. Up to 16 model candidates and, for tuned thresholds, 64 joint scored combinations are allowed per model. The plan declares the total explicit-fit ceiling. Dense encoding above the conservative memory estimate is refused. Calibrated SVM requires scaling and stratified CV; before fitting, every nested training subset is checked for sufficient per-class examples for the largest declared calibration fold count. Nonlinear candidates also have a pairwise-memory guard.

Sensitivity variants are labelled `interpretive_sensitivity` in results. They may describe robustness but may not select or replace the baseline, set a threshold, or enter Model Lock. A sensitivity worth adopting becomes a prespecified baseline in a new independent plan/run.

## Search evidence and boundaries

Every outer and final inner search saves all candidate scores, selected model parameters, selected threshold where applicable, split scores and final-refit evidence. Numeric boundaries record distinct value count, whether interior candidates existed, selected edge and whether an outside-range question is supported. A two-value endpoint means coarse coverage only. An edge after interior candidates supports a future independent question, not automatic expansion. Fold instability and inconsistent selected edges must be reported as uncertainty.

Outer or held-out results do not authorize family, preprocessing, metric, threshold or grid changes within the locked claim. Such changes are a new independent development run.

## Sensitivities and capability boundary

Sensitivity overrides may replace whole feature lists, exclusions, models, or the binary `threshold_policy`; they retain target, metrics, class semantics and CV. Every variant uses identical outer splits and is independently validated/refitted. Post-hoc held-out selection is forbidden.

The implementation is single-target classification, not multilabel/regression. It does not bundle automated feature engineering, resampling, sparse large-scale encoding, cost-curve optimization or scalable kernel approximation. Unsupported needs must be disclosed and extended with focused synthetic tests before development.
