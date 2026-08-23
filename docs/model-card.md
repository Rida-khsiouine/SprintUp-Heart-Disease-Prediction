# Model Card: Heart Disease Across Hospitals v2.0.0

## Summary

This educational binary classifier estimates `P(num > 0)` from the 13 raw attributes in the UCI Heart Disease dataset. Cleveland is the only development cohort. Hungary, Switzerland, and VA Long Beach are external validation cohorts and cannot affect model choice, calibration, or thresholds.

The artifact is intended to demonstrate reproducible data-science and ML-engineering practice. It is **not medical advice**, is **not clinically validated**, and must not be used for diagnosis, triage, treatment, or any patient-care decision.

## Model and decision rules

The candidate set contains a prevalence dummy, logistic regression, random forest, and gradient boosting. Candidate hyperparameters are tuned inside repeated nested stratified cross-validation. ROC-AUC is primary; the one-standard-error rule favors the simpler candidate in this order: logistic regression, gradient boosting, random forest.

The primary decision threshold is 0.5. A secondary educational “screening” threshold maximizes specificity subject to Cleveland out-of-fold sensitivity of at least 0.85. The label does not make the threshold clinically appropriate.

## Data

The four processed cohorts were donated to UCI in 1988 and reflect historical clinical populations and measurement practices. The data is therefore unsuitable as evidence of present-day clinical performance. Exact source URLs, row counts, retrieval date, SHA-256 hashes, DOI, and CC BY 4.0 attribution are recorded in `data/manifest.json` and `data/README.md`.

Feature missingness differs sharply by cohort, especially for `ca`, `thal`, and `slope`. The pipeline preserves every patient row and fits imputers only on training data. Duplicate counts and missingness are recorded rather than silently deleted.

## Evaluation

Internal evidence comes from averaged patient-level out-of-fold probabilities; it is not training-set performance. External results use a frozen final pipeline and frozen thresholds. Stratified patient bootstraps quantify uncertainty. AUC is explicitly unavailable when a cohort or resample has only one class.

The generated diagnostics compare outer-training and held-out nested ROC-AUC, show how the selected model changes as training rows increase, and expose hyperparameter-selection frequency. These views cannot prove that overfitting is absent; they help distinguish internal fit behavior from the separate problem of cross-hospital transportability. The synchronized evidence and graphs are in `reports/model-diagnostics.json`, `reports/learning-curve.csv`, and `reports/model-diagnostics/`.

The external results show meaningful **distribution shift** and a material decline in discrimination on VA Long Beach. No cohort is omitted because its result is weak. Refer to `reports/metrics.json`, `reports/external-validation.csv`, and `reports/cohort-shift.csv` for the generated evidence.

## Subgroups and fairness

Sex and broad age bands were predeclared. A subgroup below 25 patients is marked `underpowered`; only counts are shown and no performance claim is made. These limited slices are not a fairness certification. The dataset’s binary sex coding, historical collection, missingness, and small sample sizes prevent a comprehensive fairness assessment.

## Exploratory unsupervised analysis

The separate PCA and clustering track explores whether patient groups appear in Cleveland's 13-feature representation. Disease labels are not used to fit preprocessing, PCA, K-Means, the selected cluster count, centroids, or Ward linkage. Target prevalence is joined only after those choices are frozen, for descriptive comparison rather than supervised validation.

The representation combines standardized numerical variables with one-hot categorical indicators. PCA, K-Means, and Ward clustering therefore impose **Euclidean** geometry on mixed data; results can change with encoding, scaling, distance choice, cohort composition, and missingness. PCA loadings are representation summaries, not causal effects or automatically meaningful latent constructs.

Cluster numbers are arbitrary identifiers, not an ordering of health or severity. The clusters are exploratory and are **not clinical** phenotypes, diagnoses, subtypes, or validated patient segments. External assignment only measures how frozen Cleveland centroids partition other historical cohorts; it does not validate the groups clinically.

## Limitations

- Historical, small observational cohorts with incomplete documentation.
- Binary target collapses severity levels 1–4 into disease present.
- Missingness may be informative and differs by site.
- External discrimination does not establish calibration, utility, or safety in a new population.
- No prospective evaluation, clinician review, causal analysis, or clinical validation.
- Unsupervised groups depend on mixed-data Euclidean geometry and have no demonstrated clinical meaning.
- The demonstration artifact is version-locked and rejects incompatible metadata; that is an engineering control, not a medical safeguard.

## Intended and prohibited use

Appropriate: education, portfolio review, reproducibility inspection, and software testing with non-sensitive example records.

Prohibited: real patient assessment, diagnosis, treatment, emergency decisions, insurance or employment decisions, or claims of regulatory/clinical approval.
