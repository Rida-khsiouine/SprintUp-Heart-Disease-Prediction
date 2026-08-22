# AI-Assisted Development Audit

## Authorship disclosure

This rebuild is explicitly **AI-assisted**. The repository owner supplied the goal, approved the scientific design and constraints, authorized the rebuild, and remains accountable for publishing and explaining the work. An AI coding agent audited the legacy repository, researched primary documentation, implemented the package and tests, ran experiments, and drafted the documentation.

That process does not prove that AI replaces a senior ML engineer. It demonstrates a narrower and defensible point: clear human direction, strong constraints, automated implementation, and independent verification can turn AI output into reviewable engineering evidence.

## Defects detected with AI assistance

- The documented binary task did not match the five-class notebook target.
- `oldpeak` decimals were destroyed by integer conversion.
- fit-dependent preprocessing and feature selection occurred before evaluation splits.
- one holdout was reused for comparison without a protected external cohort.
- the pickle depended on notebook-local code and did not accept raw records safely.
- clustering code referenced undefined or incorrect evaluation variables.
- metrics, README claims, and the serialized model did not form one reproducible contract.

## Suggestions accepted

- Treat Cleveland as development-only and the other hospitals as external validation.
- Use repeated nested stratified evaluation and fold-local pipelines.
- Select by ROC-AUC with a one-standard-error simplicity rule rather than a vanity-score target.
- Report a default and a clearly secondary educational threshold without using external labels.
- Commit checksum-verified raw snapshots with UCI attribution.
- Use `skops` with explicit unknown-type handling, canonical deterministic archives, and metadata integrity checks.
- Generate the README results, tables, figures, and notebook from shared report objects.
- State weak external results, subgroup power limits, dataset age, and non-clinical status prominently.

## Suggestions rejected or deferred

- **“Prove AI replaces a senior engineer.”** Rejected as an unsupported conclusion; the project documents capability and verification instead.
- **Optimize until accuracy looks impressive.** Rejected because it encourages test reuse and selective reporting.
- **Combine all hospitals before splitting.** Rejected because it erases the external-generalization question.
- **SMOTE before cross-validation.** Rejected for this milestone; it was unnecessary and easy to leak across folds.
- **Deep learning.** Rejected as unjustified for 303 development records and tabular features.
- **Interactive app, cloud deployment, and Docker.** Deferred until the scientific and artifact contracts pass. A future app can call `predict_record` without changing training.
- **Clinical recommendation language.** Rejected because the data and evaluation do not support it.

## Verification performed

Verification is encoded in tests and reproducible commands rather than trust in generated prose:

- immutable source hashes, byte sizes, row counts, and offline verification;
- adversarial schema and categorical-domain tests;
- a fold-membership test proving outer test patients do not enter inner search;
- deterministic seed, bootstrap, artifact-byte, and prediction checks;
- one-standard-error and threshold tests against constructed examples;
- external single-class and underpowered-subgroup failure behavior;
- safe artifact round-trip and malicious-type rejection;
- README/report synchronization and clean-kernel notebook execution;
- Ruff, coverage, Windows/Linux CI, and a manual full reproduction job.

## Remaining limitations

Tests can show that implemented invariants hold; they cannot establish that the historical data represents modern patients, that the selected metrics imply clinical utility, or that no unmodeled bias exists. Human reviewers should inspect the design, generated evidence, code diff, and model card. AI-assisted code also inherits risks from library behavior and specification errors, so locked dependencies and skepticism remain necessary.
