# Heart Disease Repository Rebuild Design

**Status:** Architecture and modeling direction approved; interactive application deferred until the modeling milestone is complete.

**Audience:** Data Science and Machine Learning Engineering recruiters, reviewers, and interviewers.

## 1. Purpose

Rebuild the existing SprintUp course repository as an independently implemented, AI-assisted, reproducible machine-learning case study. The project will answer a more useful question than “which classifier gets the highest accuracy?”:

> How well does a heart-disease classifier developed on one hospital cohort generalize to patients from other hospitals with different prevalence and missing-data patterns?

The repository will demonstrate both data-science judgment and ML-engineering execution. It will disclose AI assistance, preserve evidence of the original audit, and make every published result reproducible.

The project is educational research. It is not a diagnostic device, medical advice, or evidence of clinical utility.

## 2. Current Problems Being Corrected

The rebuild must remove, rather than conceal, the original methodological defects:

- The README states a binary task while the notebooks train a five-class target.
- Scaling and feature selection are fitted before the train/test split.
- Converting the complete encoded frame to integers truncates decimal `oldpeak` measurements.
- Feature selection sees the complete target before evaluation.
- A single, non-stratified 60-record test split is reused to compare many models.
- The exported model does not contain raw-input preprocessing and depends on a notebook-local custom class.
- The chosen export is inconsistent with the reported tuning result.
- PCA and clustering are disconnected from the predictive question, and the clustering notebook contains stale or invalid state.
- The repository has no locked environment, tests, CI, data manifest, model card, or reproducible command.

A preliminary leakage-safe audit on the official Cleveland data produced approximately 0.84 accuracy, 0.91 ROC-AUC, and 0.82 F1 for simple binary logistic regression under repeated stratified cross-validation. These are feasibility observations, not final published results or acceptance thresholds.

## 3. Scope

### Included in the first milestone

- Reproducible acquisition and validation of four official UCI processed cohorts.
- Correct binary target construction: `0` is absence and `1` through `4` are presence.
- Leakage-safe preprocessing and model evaluation.
- Internal Cleveland model development and external hospital validation.
- Cohort-shift, missingness, calibration, error, and limited subgroup analysis.
- A reusable Python package, command-line workflow, tests, CI, documentation, figures, and a model card.
- A versioned trained artifact with matching metadata after final model selection.
- A transparent AI-assistance report and before/after audit.
- One polished analysis notebook that calls package code instead of duplicating it.

### Explicitly deferred

- Streamlit or another interactive application.
- Cloud deployment, online monitoring, a feature store, experiment-tracking servers, or orchestration infrastructure.
- Clinical recommendations or diagnostic claims.
- Causal claims about features.
- Deep learning, which is unjustified for this small tabular dataset.
- PCA, clustering, or feature selection included solely to demonstrate an algorithm.

The package will expose a stable raw-record prediction interface so an application can be added without changing training or evaluation code.

## 4. Chosen Approach and Alternatives

### Chosen: scientific case study with production-quality implementation

This approach combines careful multi-cohort validation with a small, testable ML package. It provides the strongest hybrid Data Science/ML Engineering signal without inventing infrastructure that the dataset does not require.

### Rejected: notebook-only refresh

This would be quicker but would remain difficult to test, reuse, and deploy. It would not provide enough engineering evidence.

### Rejected for now: full MLOps platform

An API gateway, registry service, monitoring stack, and cloud orchestration would be disproportionate to a dataset of fewer than one thousand records. These additions can be considered only after the scientific core is credible.

## 5. Data Design

### Sources

Use the processed Cleveland, Hungary, Switzerland, and VA Long Beach cohorts from the UCI Heart Disease dataset. Each source is recorded with its official URL, DOI, retrieval date, SHA-256 checksum, row count, expected columns, and license information.

The source files are small and licensed for redistribution with attribution. Exact raw snapshots will be stored unchanged in `data/raw/` so a clean clone remains reproducible. `data/manifest.json` will prove provenance and detect accidental modification. A fetch command will be able to re-download and verify the same files.

### Schema and target

The 13 input features retain their documented clinical types. Decimal values such as `oldpeak` must remain decimal. The raw `num` field is preserved in source data; modeling constructs `disease_present = (num > 0)`.

Missing feature values are retained and handled by preprocessing inside model-fitting folds. Records are not silently dropped. A missing or invalid target is a hard error. Schema validation checks column names, numeric coercion, allowed categorical values, impossible ranges where documentation is unambiguous, duplicate records, and target values.

### Cohort roles

- **Cleveland:** the only development cohort used for model comparison, preprocessing decisions, calibration decisions, and threshold selection.
- **Hungary, Switzerland, and VA Long Beach:** untouched external cohorts. They are inspected for schema and descriptive shift, but their labels cannot influence model or threshold selection.

## 6. Repository Architecture

```text
.
├── data/
│   ├── raw/
│   ├── manifest.json
│   └── README.md
├── artifacts/
│   ├── model.skops
│   └── metadata.json
├── docs/
│   ├── ai-assisted-development.md
│   ├── audit-before-after.md
│   └── model-card.md
├── notebooks/
│   └── analysis.ipynb
├── reports/
│   ├── figures/
│   ├── metrics.json
│   ├── experiment-manifest.json
│   └── external-validation.csv
├── src/heart_disease/
│   ├── data.py
│   ├── validation.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── artifacts.py
│   └── cli.py
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
├── uv.lock
├── README.md
└── LICENSE
```

Responsibilities are separated as follows:

- `data.py` retrieves immutable raw data and applies only documented target construction.
- `validation.py` validates source and inference schemas and fails with actionable messages.
- `preprocessing.py` creates unfitted scikit-learn transformers and pipelines.
- `modeling.py` defines candidate estimators, bounded search spaces, and deterministic seeds.
- `evaluation.py` owns cross-validation, metrics, confidence intervals, curves, and external evaluation.
- `artifacts.py` saves and loads only project-produced, versioned model artifacts and validates their metadata.
- `cli.py` exposes data verification, training, evaluation, and full reproduction commands.

The notebook imports these modules. It contains narrative and visual interpretation, not an alternative implementation.

## 7. Modeling Methodology

### Preprocessing

A scikit-learn `ColumnTransformer` is part of every candidate pipeline:

- Continuous/count features: median imputation and standard scaling where the estimator benefits from it.
- Categorical features: most-frequent imputation and one-hot encoding with unknown-category handling.
- No preprocessing object is fitted outside its training fold.
- No target-driven feature selection is performed. With 13 documented features, regularization and stability analysis are preferable.

### Candidate models

Use a deliberately small comparison set:

1. Prior/majority dummy classifier.
2. Regularized logistic regression as the interpretable baseline and default preference.
3. Random forest as a nonlinear bagging comparison.
4. Gradient boosting as a nonlinear boosting comparison.

Search spaces remain small and documented. Every stochastic component receives a fixed seed. No model is added only to enlarge a leaderboard.

Logistic regression uses its native probabilities. If a nonlinear model is selected, sigmoid calibration is fitted only within the corresponding outer training data before outer-fold prediction. External cohorts are never used to choose or fit calibration.

### Internal validation

Use repeated nested stratified cross-validation on Cleveland:

- Outer loop: five stratified folds repeated five times with deterministic seeds.
- Inner loop: five stratified folds for the bounded hyperparameter search.
- Identical outer partitions are used for all candidates.
- The primary selection metric is ROC-AUC.
- If a simpler model is within one standard error of the best mean ROC-AUC, prefer the simpler model.
- Calibration and balanced accuracy act as documented tie-breakers, not post-hoc reasons to select a favorite model.

Outer-fold predictions are retained. Repeated predictions for the same patient are averaged before producing patient-level curves and bootstrap intervals. Published tables distinguish fold variability from patient-level 95% bootstrap confidence intervals.

### Metrics

Report, with denominators and uncertainty where applicable:

- Accuracy and balanced accuracy.
- Sensitivity/recall and specificity.
- Precision and F1.
- ROC-AUC and average precision/PR-AUC.
- Log loss and Brier score.
- Confusion matrix and calibration curve.

The default 0.5 threshold is the primary classification operating point. A secondary educational screening operating point may be selected from Cleveland out-of-fold predictions by maximizing specificity subject to recall of at least 0.85. That threshold is locked before any external labels are evaluated.

### Final fit and external validation

After the model family, hyperparameters, calibration strategy, and thresholds are frozen, refit on all Cleveland records. Evaluate once on each external cohort. Report external metrics even when materially worse than internal results.

Patient-level stratified bootstrap intervals are used for external metrics when both classes permit them. Extremely imbalanced cohorts must show class counts and wide uncertainty rather than a misleading point estimate.

### Shift and subgroup analysis

For each cohort, report feature missingness, target prevalence, continuous-feature distribution differences, and categorical frequency differences. Connect performance degradation to observed shift only as an association.

Predeclared subgroup slices are sex and broad age bands. Metrics are accompanied by group size and uncertainty. Groups below the minimum reliable sample count are labeled underpowered rather than interpreted as fairness conclusions.

### Interpretation

For the selected model, report global permutation importance on held-out predictions. If logistic regression is selected, also report standardized coefficient direction and fold-to-fold stability. Feature associations are not described as causal effects.

## 8. Artifact and Result Integrity

Each reproduction run writes an experiment manifest containing:

- Data hashes and cohort row counts.
- Git commit when available.
- Python and library versions.
- Random seeds and cross-validation configuration.
- Candidate parameters and selected parameters.
- Threshold-selection rule.
- Model artifact hash.

Machine-readable metrics and plotted reports are generated from the same evaluation objects. README numbers are updated from generated result files, not typed independently.

The final artifact accepts the documented 13 raw features. It includes preprocessing and prediction in one object, is stored as `artifacts/model.skops` with `skops.io`, and is accompanied by dependency, schema, data, model, and threshold metadata. Loading requires an explicit allowlist of the artifact's trusted estimator types. The artifact is round-trip tested, and the project never loads arbitrary third-party pickle files.

## 9. AI-Assisted Development Evidence

The repository will openly state that AI performed substantial auditing, implementation, experiment design, testing, and documentation work. It will not claim that one project proves the replacement of a senior engineer.

`docs/ai-assisted-development.md` will show:

- The initial human objective and constraints.
- Defects AI discovered in the original repository.
- Major suggestions accepted, rejected, or revised.
- The automated checks used to verify AI-produced work.
- Before/after methodology and reproducibility comparisons.
- Known limitations and areas requiring domain-expert review.

The intended conclusion is evidence-based: effective AI direction plus rigorous verification can produce a substantially stronger ML artifact than unreviewed manual or copied notebook work.

## 10. Error Handling and Safety

- Network failures, checksum mismatches, malformed rows, schema mismatches, unknown targets, and artifact incompatibility fail loudly with remediation guidance.
- Feature values are never silently truncated or coerced after failed validation.
- External cohorts cannot enter tuning code paths.
- Training and inference use the same pipeline and schema.
- Logs contain aggregate experiment information, not patient rows.
- Documentation includes dataset age, cohort limitations, missingness, lack of prospective validation, and a non-medical-use warning.

## 11. Testing and Continuous Integration

Unit tests cover target mapping, preservation of decimal measurements, schema validation, cohort-role enforcement, deterministic splits, preprocessing, unknown categories, missing values, metric calculations, and threshold rules.

Integration tests cover data checksum verification, a reduced-cost training run, external evaluation, artifact round-trip prediction, CLI commands, and notebook execution. A regression test ensures the README/report metrics source remains machine-generated.

GitHub Actions will run formatting/lint checks, unit and integration tests, and a lightweight reproducibility smoke test on supported Python versions. The full nested experiment remains locally reproducible through one documented command and may run in scheduled or manually triggered CI to control runtime.

## 12. Legacy and Git Strategy

The current head will be preserved with a `legacy-v1` tag. Main will then receive a series of meaningful commits for data provenance, tests, pipelines, evaluation, documentation, and final results. Existing copied notebook code will not be moved into the new package.

If the original source repository is identified, it will be credited. The rebuild is independently implemented from the UCI documentation and library documentation.

## 13. Acceptance Criteria

The modeling milestone is complete only when:

1. A clean environment can reproduce data validation, training, internal evaluation, and external evaluation through documented commands.
2. All preprocessing is demonstrably inside evaluation folds.
3. The target definition and cohort roles match the documented study.
4. Tests catch the original target mismatch, decimal truncation, and train/inference mismatch classes of defect.
5. Published numbers and figures are generated from versioned result artifacts.
6. External performance, uncertainty, missingness, and limitations are reported without selective omission.
7. The trained artifact accepts raw documented inputs and passes round-trip tests.
8. CI passes and the analysis notebook executes from a clean kernel.
9. The README, model card, audit, and AI-assistance report are understandable without opening every source file.
10. No medical or senior-engineer-replacement claim exceeds the evidence.

After these criteria pass, the interactive application will be designed as a separate milestone using the stable inference interface.
