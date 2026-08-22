# Project Guide

This is the shortest route through the project for a technical reviewer. Each numbered stage points to the canonical implementation, the strongest focused test suite, and the evidence produced by that code. The notebook does not contain a second implementation.

## 1. Data acquisition and validation

Official UCI snapshots are downloaded atomically, verified by checksum and row count, and recorded with their source and license. Start with the [acquisition code](../src/heart_disease/data.py), verify its behavior in the [data provenance tests](../tests/test_data.py), and inspect the immutable [data manifest](../data/manifest.json).

## 2. Cleaning and leakage-safe preprocessing

Schema and target validation preserve missing feature values and `oldpeak` decimals; imputers, scaling, and one-hot encoding remain inside fitted pipelines. Read the [preprocessing implementation](../src/heart_disease/models.py), its [pipeline tests](../tests/test_models.py), and the generated [missingness evidence](../reports/missingness.csv).

## 3. Supervised candidates and nested tuning

Dummy, logistic-regression, random-forest, and gradient-boosting candidates use identical repeated outer folds while all tuning occurs in inner folds. Inspect the [nested evaluation code](../src/heart_disease/evaluation.py), the [leakage and selection tests](../tests/test_evaluation.py), and the patient-level [out-of-fold predictions](../reports/oof-predictions.csv).

## 4. Model selection and thresholds

ROC-AUC is primary, the one-standard-error rule favors simpler competitive models, and the screening threshold is derived only from Cleveland out-of-fold predictions. Follow the [selection implementation](../src/heart_disease/evaluation.py), its [constructed-example tests](../tests/test_evaluation.py), and the canonical [metrics report](../reports/metrics.json).

## 5. External validation and dataset shift

The model, calibration, and thresholds are frozen before Hungary, Switzerland, and VA Long Beach labels are evaluated; uncertainty and cohort shift are reported even when performance is weak. Read the [external evaluation code](../src/heart_disease/external.py), its [bootstrap and shift tests](../tests/test_external.py), and the [external-validation table](../reports/external-validation.csv).

## 6. Unsupervised PCA and patient-profile discovery

Cleveland features alone fit preprocessing, PCA, K-Means selection, stability analysis, and Ward linkage. Labels enter only after all choices are frozen, and external cohorts only receive frozen transformations. Inspect the [unsupervised implementation](../src/heart_disease/unsupervised.py), the [label-isolation tests](../tests/test_unsupervised.py), and the annotated [report index](../reports/README.md).

## 7. Final training, safe serialization, and inference

The selected end-to-end pipeline is refit on Cleveland, serialized with `skops`, integrity-bound to metadata, and exposed through a strict 13-feature inference contract. Review the [artifact and inference code](../src/heart_disease/artifacts.py), the [round-trip and rejection tests](../tests/test_artifacts.py), and the committed [artifact metadata](../artifacts/metadata.json).

## 8. Generated evidence, notebook, tests, and CI

Package code generates synchronized reports, README blocks, and the thin notebook; CI checks the locked environment on supported Python versions. Start with the [reporting code](../src/heart_disease/reporting.py), run the [report and provenance tests](../tests/test_reporting.py), and inspect the [experiment manifest](../reports/experiment-manifest.json).

For interpretation boundaries—including dataset age, mixed-data geometry, subgroup power, and non-clinical use—finish with the model card named `docs/model-card.md` in the repository.
