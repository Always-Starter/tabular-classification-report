# Executable plan contract (schema v4)

`scripts/validate_plan.py` is the executable validator. [../tests/fixtures/example-plan.json](../tests/fixtures/example-plan.json) is a complete binary example. Schemas v2 and v3 remain readable for preserved runs; create new plans as v4.

Top-level fields:

| Field | Contract |
| --- | --- |
| schema_version | 4 |
| target, task | Column name; `binary` or `multiclass` |
| features | Nonempty ordered predictor names; excludes target |
| numeric_features, categorical_features | Disjoint lists exactly partitioning features |
| excluded_features | Object mapping excluded column names to reasons; every training predictor must be selected or explicitly excluded, in every variant |
| semantics | Structured target source/meaning, positive-class meaning, FP/FN costs and row-dependence status; use null/unknown rather than inference |
| feature_provenance | Material feature provenance records: derivation, outcome-information use, prediction-time availability and confirmed/unknown status |
| positive_class, threshold | Binary only: string class and fixed threshold in [0,1]; omit both for multiclass |
| seed | Nonnegative integer |
| cv | strategy, outer_splits and inner_splits (2–10); see below |
| metrics | primary string; secondary list, rationale and confirmed/provisional status |
| candidate_selection | Exact registry snapshot, selection basis, shortlist rationale and considered alternatives |
| models | Two or three named model specifications in this implementation; two is the default, a third needs a documented training-data reason (not an explicit Variant 2 limit) |
| model_count_rationale | Required nonempty string for every v3 plan. Explain why the selected count is sufficient, not a hoped-for score gain. |
| decision_trace | Nonempty list of observation/decision/rationale/human_review_point strings |
| sensitivities | Zero to three predeclared sensitivity specifications |

## Metrics and class semantics

All tasks: `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`, `log_loss`.
Binary adds `f1`, `precision`, `recall`, `roc_auc`, `average_precision`.
Multiclass adds `roc_auc_ovr_macro`. Only log loss is minimized; all other supported metrics are maximized. Prediction uses the declared binary threshold or multiclass argmax. Training determines sorted class order, which is frozen before testing. Binary ranking metrics explicitly binarize against `positive_class`, independent of label sorting. Metrics requiring unavailable classes are null with a reason at holdout, not silently zero/NaN. Every training validation fold must contain all classes.

For binary precision/recall/F1, use null with a reason when the actual denominator is zero: precision needs predicted positives, recall needs actual positives, F1 needs at least one actual or predicted positive. A mathematically defined zero remains zero (e.g. all predicted positives are false positives). Macro/weighted metrics average over frozen classes with per-class zero-division convention 0. If any outer fold has an undefined secondary metric, its mean/SD are null and fold-level reasons remain saved; the primary metric must be defined in every training scoring fold.

## CV

- `stratified`: shuffled stratified folds with the saved seed.
- `stratified_group`: also specify `group_column`; one entity stays in one fold.
- `time`: also specify `time_column`; forward splits over sorted distinct timestamps; optional nonnegative `gap`. Equal timestamps cannot straddle a split.

Group/time columns must appear in `excluded_features`. Training checks fold feasibility before fitting; no silent fallback to shuffled CV.

## Per-model specification

Each object has `name` (lowercase safe filename), `type`, `params` (fixed estimator parameters), `fixed_param_rationale`, `grid` (list-valued `model__` parameters), `preprocessing`, `rationale`, `grid_rationale`, and `stopping_rule`. Types: `logistic_regression`, `random_forest`, `extra_trees`, `decision_tree`, `knn`, `gaussian_nb`, `support_vector_classifier`.

`fixed_param_rationale` must cover every `params` key exactly. Each record contains:

- `not_tuned_reason`: why this parameter was excluded from the tuning grid;
- `value_source`: one of `training_diagnosis`, `compute_budget`, `convergence_requirement`, `user_supplied`, `authoritative_requirement`, `prior_independent_evidence`, `implementation_constraint`, `predeclared_rule`, `library_default`, or `unknown`;
- `value_rationale`: why this exact value was chosen rather than another value. It must be a nonempty string unless `value_source` is `unknown`; an unknown source requires JSON `null` so a rationale cannot be reconstructed after the fact.

These are two separate questions. A limited tuning budget can explain why `max_depth` was not tuned but does not explain why its fixed value is 12. Empty `params` therefore requires an empty rationale object; unspecified library defaults remain recorded in resolved training results rather than receiving invented plan rationales.

Preprocessing requires:

- `numeric_imputer`: median / mean / most_frequent / constant; optional numeric_fill_value (default 0).
- `missing_indicator`: boolean; `numeric_transform`: none / log1p.
- `scaler`: none / standard / robust.
- `categorical_imputer`: most_frequent / constant (`<MISSING>`).
- `categorical_encoder`: onehot / ordinal. Onehot optionally accepts min_frequency and max_categories; unseen categories are ignored. Ordinal uses -1 for unseen categories.

Only estimator parameters are tuned in `grid`. Use a declared sensitivity for preprocessing/feature/threshold alternatives. Empty grid `{}` means one fixed candidate. Up to 16 candidates per model; default total budget 1200 fits including refits and sensitivities. Estimator seeds are controlled by the top-level seed; internal jobs are 1. Unspecified estimator defaults are resolved and saved in training results; package versions are recorded. This runner produces dense encoded features and rejects a conservative estimated matrix above 512 MiB; high-cardinality/sparse datasets may need an extension.

SVM requires standard or robust numeric scaling and currently supports only stratified CV because its three-fold probability calibration is not yet group/time-aware. The runner applies a conservative 2 GiB pairwise-kernel guard to nonlinear candidates. Calibration adds internal work beyond the explicit grid-fit count, so keep SVM grids especially small and document the compute trade-off.

Before fitting, justify the primary metric, candidate-set scope, each family, each grid, every fixed parameter and per-model preprocessing from the training diagnosis or another recorded source. The bounded search stops after its prespecified inner-CV candidates; outer folds assess the tuned procedure, not an adaptive-refinement signal. Development results preserve candidate-level inner scores for each outer search and the final full-training search, plus final-refit evidence.

For numeric grids with at least two distinct values, results save `tuning_boundary` for outer-fold and final-inner endpoint choices. The metadata records the number of distinct values, whether interior candidates existed, a boundary type and whether the result supports an outside-range question. In a two-value grid, every selection is an endpoint: this is `two_value_grid_endpoint`, evidence of coarse coverage only, and never directional evidence that the optimum lies outside the range. `edge_with_interior_candidates` means at least one interior value was evaluated before an edge won; it supports an untested-direction question for a future independent run, not adaptive expansion or retuning of the current run.

## Sensitivities

Each object contains `name`, `rationale`, `overrides`. Overrides may replace only `features`, `numeric_features`, `categorical_features`, `excluded_features`, `models`, or binary `threshold`. Overrides replace whole fields, not recursive fragments. Preserve model names/order, target, metrics and CV. Every variant is independently validated, evaluated on identical splits, fitted on full training and saved. Select one variant in the lock review; never silently promote a model based on test performance.

The bundled implementation is single-target classification, not multilabel/regression. Custom methods require extending this contract and its tests before development; unknown fields and unsupported estimators fail rather than being ignored.
