# Development Process

## Project ownership

I rebuilt this project around a clear question: whether a model developed on the Cleveland cohort can retain useful discrimination across three independent hospital cohorts. I defined the scope, approved the evaluation design, reviewed the resulting evidence, and remain responsible for explaining the work and its limitations.

AI-assisted tools supported repository auditing, implementation, test generation, and documentation. I treated their output as proposals to inspect rather than conclusions to accept automatically. The published methodology and results are backed by executable code, tests, immutable data hashes, and reproducible reports.

## Engineering decisions

I replaced the notebook-driven workflow with an installable Python package and made the following decisions explicit:

- define the binary target as `disease_present = (num > 0)` while retaining the original severity label for description;
- use Cleveland only for development and preserve Hungary, Switzerland, and VA Long Beach for external validation;
- fit preprocessing, tuning, calibration, and model selection only within the appropriate training folds;
- compare logistic regression, random forest, and gradient boosting against a dummy baseline;
- favor the simpler model when candidates are statistically competitive under the one-standard-error rule;
- preserve weak external results and uncertainty rather than optimize the presentation around a single favorable score;
- keep PCA and clustering label-isolated and exploratory, without presenting clusters as diagnoses or clinical subtypes;
- serialize the complete raw-feature pipeline with schema, version, and integrity checks.

## Verification

The project uses repeatable evidence instead of relying on implementation claims alone:

- checksums, byte sizes, row counts, and offline verification protect data provenance;
- schema tests preserve missing values and decimal `oldpeak` measurements while rejecting invalid targets and categories;
- fold-membership tests guard against preprocessing, tuning, calibration, and model-selection leakage;
- deterministic tests cover seeds, bootstrap intervals, model artifacts, and predictions;
- external-validation tests cover single-class metrics, underpowered subgroups, and cohort-shift reporting;
- artifact tests cover round-trip predictions, trusted-type restrictions, and metadata integrity;
- unsupervised tests enforce target isolation, Cleveland-only fitting, stable selection, and frozen external assignments;
- the notebook runs from a clean kernel and consumes package-generated evidence rather than implementing a second training pipeline;
- Ruff, pytest, and GitHub Actions verify the locked project on supported Python versions.

## Deliberate exclusions

I did not add techniques merely to make the project look more complex. Deep learning is not justified for 303 development records. Resampling was not introduced without evidence that it improved the declared evaluation. Hospitals were not pooled before splitting because that would weaken the external-generalization study. Clinical recommendation language was excluded because this historical dataset does not support clinical use.

An interactive application remains a possible next layer. It can call the existing `predict_record` interface without changing the training or evaluation contract.

## Remaining limitations

Passing tests cannot establish that these historical cohorts represent modern patients, that retrospective discrimination implies clinical utility, or that every source of bias has been modeled. The model card therefore documents dataset age, missingness, distribution shift, subgroup power, and non-clinical status alongside the performance results.
