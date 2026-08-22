# Heart Disease Across Hospitals

[![CI](https://github.com/Rida-khsiouine/SprintUp-Heart-Disease-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Rida-khsiouine/SprintUp-Heart-Disease-Prediction/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

An independently rebuilt, AI-assisted machine-learning study of how a heart-disease classifier generalizes across hospitals. The engineering question is more interesting than a vanity accuracy score: **does a model developed only on Cleveland retain useful discrimination in Hungary, Switzerland, and VA Long Beach?**

The answer is mixed—and reported without hiding the weak cohort. Logistic regression is competitive with more complex candidates internally and degrades substantially on VA Long Beach. That is the project’s central result, not an inconvenience to remove.

> Educational portfolio project only. It is not medical advice, is not clinically validated, and must not be used for diagnosis or care decisions.

## Start here

| Recruiter question | Fastest path |
|---|---|
| What did the project find? | Read the [supervised results](#results), then the [exploratory unsupervised evidence](#unsupervised-patient-profile-evidence). |
| How rigorous is the ML work? | Follow the eight-stage [project guide](docs/project-guide.md) and inspect the [model card](docs/model-card.md). |
| Where are the generated numbers and figures? | Use the annotated [report index](reports/README.md); the notebook is only a report consumer. |
| Can the trained model make a prediction? | Inspect the raw-record [`predict_record` contract](src/heart_disease/artifacts.py) and run the CLI example below. |

- **Supervised track:** heart-disease prediction, nested tuning, threshold selection, and external validation live in [`evaluation.py`](src/heart_disease/evaluation.py).
- **Unsupervised track:** label-isolated PCA, K-Means stability, Ward comparison, and frozen cohort transfer live in [`unsupervised.py`](src/heart_disease/unsupervised.py).
- **Engineering track:** provenance, reproducibility, safe serialization, inference, and CI are mapped in the [project guide](docs/project-guide.md).

## What this demonstrates

- Leakage-safe preprocessing, tuning, and calibration inside training folds.
- Repeated nested stratified cross-validation with identical outer folds per candidate.
- One-standard-error model selection that favors a simpler model when performance is statistically indistinguishable.
- Frozen-model external validation on three untouched hospital cohorts with bootstrap uncertainty.
- Dataset provenance, immutable checksums, schema validation, safe `skops` serialization, and strict raw-record inference.
- Machine-readable evidence, an executable thin notebook, CI, and transparent AI-assisted development notes.

```mermaid
flowchart LR
    C[Cleveland\ndevelopment only] --> O[Repeated outer CV]
    O --> I[Inner tuning +\nfold-local preprocessing]
    I --> S[One-SE selection +\nfrozen thresholds]
    S --> F[Final Cleveland fit]
    F --> H[Hungary\nexternal]
    F --> W[Switzerland\nexternal]
    F --> V[VA Long Beach\nexternal]
```

## Results

ROC-AUC is the primary selection metric. Brackets are patient-bootstrap 95% intervals; candidate `±` values are outer-fold standard deviations. The secondary screening threshold is selected on averaged Cleveland out-of-fold predictions and never adjusted using external labels.

<!-- GENERATED_RESULTS_START -->
_Generated from `reports/metrics.json` (full profile). Intervals are 95% stratified patient-bootstrap intervals._

| Evidence | ROC-AUC | Balanced accuracy | Brier score |
|---|---:|---:|---:|
| Cleveland nested CV — `dummy` | 0.500 ± 0.000 | 0.500 | 0.248 |
| Cleveland nested CV — `logistic` | 0.917 ± 0.032 | 0.852 | 0.116 |
| Cleveland nested CV — `gradient_boosting` | 0.903 ± 0.041 | 0.829 | 0.124 |
| Cleveland nested CV — `random_forest` | 0.914 ± 0.034 | 0.830 | 0.119 |
| Cleveland selected OOF — `logistic` | 0.916 [0.882, 0.946] | 0.852 [0.810, 0.894] | 0.114 [0.093, 0.136] |
| External — hungary | 0.896 [0.854, 0.933] | 0.784 [0.732, 0.833] | 0.124 [0.105, 0.144] |
| External — switzerland | 0.752 [0.574, 0.898] | 0.659 [0.521, 0.752] | 0.333 [0.288, 0.377] |
| External — va | 0.707 [0.623, 0.781] | 0.634 [0.561, 0.702] | 0.271 [0.238, 0.305] |
<!-- GENERATED_RESULTS_END -->

See [the model card](docs/model-card.md) for interpretation limits and [the generated figures](reports/figures) for ROC, precision–recall, calibration, confusion matrices, missingness, cohort shift, and feature stability.

## Unsupervised patient-profile evidence

This scientifically separate track asks whether natural patient groups appear in Cleveland's feature space and whether those frozen groups transfer to the other hospitals. Disease labels are excluded from preprocessing, PCA, cluster-count selection, and fitting; they are joined only for post-hoc description. These exploratory clusters are not diagnoses or clinical subtypes.

<!-- GENERATED_UNSUPERVISED_START -->
_Generated from `reports/unsupervised/summary.json` (full profile). Labels were not used for fitting._

| Exploratory evidence | Value |
|---|---:|
| Selected clusters | 2 |
| PCA components retained | 12 |
| Cumulative variance retained | 0.908 |
| Silhouette score | 0.180 |
| Subsample stability ARI | 0.891 |
| K-Means vs Ward ARI | 0.355 |

| Frozen external transfer | Cluster-proportion distance |
|---|---:|
| hungary | 0.170 |
| switzerland | 0.028 |
| va | 0.136 |
<!-- GENERATED_UNSUPERVISED_END -->

The complete selection, stability, hierarchy, profile, and transfer evidence is indexed in [`reports/README.md`](reports/README.md).

## Reproduce it

Requirements: Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --frozen --all-extras
uv run heart-disease validate-data
uv run heart-disease analyze-unsupervised --profile smoke
uv run heart-disease reproduce --profile smoke
uv run heart-disease predict --input tests/fixtures/patient.json
```

The smoke profile exercises the complete path with smaller resampling counts. The committed evidence is produced with:

```powershell
uv run heart-disease reproduce --profile full
```

Raw snapshots are committed because the source is small and redistribution is permitted with attribution. `heart-disease fetch-data` verifies downloads before any atomic replacement. See [data provenance and citation](data/README.md).

## Evaluation contract

- Target: `disease_present = (num > 0)`; raw severity `num` remains available for description.
- Development: Cleveland only (303 records).
- External validation: Hungary (294), Switzerland (123), and VA Long Beach (200).
- Candidates: prevalence dummy, logistic regression, random forest, and gradient boosting.
- Primary threshold: 0.5. Secondary threshold: maximum specificity subject to Cleveland out-of-fold recall ≥ 0.85.
- No imputer, encoder, scaler, calibrator, estimator, threshold, or model choice sees an outer test fold or external label during fitting.
- Small predeclared sex and broad age-band slices are labeled `underpowered` below 25 patients.

## Repository map

```text
src/heart_disease/   installable data, evaluation, reporting, artifact, and CLI code
tests/               leakage, provenance, determinism, inference, and notebook checks
data/raw/            checksum-verified UCI snapshots
artifacts/           safe model plus integrity-bound metadata
reports/             generated metrics, tables, manifests, and figures
notebooks/           one thin report consumer; no second training implementation
docs/                design, model card, before/after audit, and AI-development audit
```

The original notebook submission remains inspectable at the annotated Git tag `legacy-v1`; it is not part of this implementation.

## AI assistance and ownership

AI helped audit the legacy work, propose experiments, implement code, and draft documentation. Every accepted behavior is backed by tests, immutable data hashes, executable reproduction, or artifact round-trip checks. [The AI-assisted development report](docs/ai-assisted-development.md) records accepted and rejected suggestions and is deliberately narrower than a claim that AI “replaced a senior engineer.” The repository demonstrates effective human direction plus verifiable automation—not clinical expertise or infallibility.

New code is MIT licensed. The UCI Heart Disease data remains under CC BY 4.0; attribution details are in [data/README.md](data/README.md).
