# Unsupervised Patient Profiles and Recruiter Navigation Design

**Status:** Approved in conversation on 2026-08-22; implementation pending

**Repository:** `SprintUp-Heart-Disease-Prediction`

**Branch:** `feat/recruiter-ready-rebuild`

## Goal

Extend the recruiter-ready heart-disease study with a scientifically separate unsupervised analysis that asks whether natural patient profiles exist in the Cleveland feature space and whether those profiles transfer coherently to the three external hospital cohorts. Make the supervised, unsupervised, and production-engineering work easy for a recruiter to locate without restoring disconnected legacy notebooks.

This design narrowly supersedes the original rebuild specification's exclusion of PCA and clustering. They are no longer being added solely to demonstrate algorithms: they answer a declared exploratory question, use explicit stability evidence, and remain isolated from supervised model selection.

## Non-goals

- Do not use clusters, PCA components, disease labels, or external-cohort evidence to change the existing production classifier.
- Do not describe clusters as clinical phenotypes, disease subtypes, diagnoses, or causal groups.
- Do not claim that a project is complete merely because it includes many algorithms.
- Do not restore the copied legacy notebooks or create a second implementation of preprocessing and training.
- Do not add the interactive application or deployment in this milestone.

## Analytical questions

The unsupervised track will answer four questions:

1. How much of the variance in the leakage-safe Cleveland feature representation is retained by successive principal components?
2. Does the Cleveland cohort contain stable, well-separated patient groups under K-Means?
3. Do K-Means and hierarchical clustering recover broadly similar group structure?
4. When the Cleveland-fitted representation and K-Means centroids are frozen, how do cluster membership, centroid distance, and post-hoc disease prevalence change in Hungary, Switzerland, and VA Long Beach?

The disease target is never supplied to preprocessing, PCA, cluster fitting, cluster-count selection, or stability estimation. It is joined only after all unsupervised choices are frozen, for descriptive comparison.

## Architecture

### Production package

Add one focused module:

```text
src/heart_disease/
├── unsupervised.py              preprocessing, PCA, clustering, stability, profiles, transfer
└── unsupervised_reporting.py    machine-readable outputs and figures for this track
```

The module will reuse the documented feature schema and categorical/numerical column definitions from the existing package. Shared preprocessing must have one canonical implementation; the unsupervised track must not copy column lists or validation rules into notebooks.

Proposed public interfaces:

```python
@dataclass(frozen=True)
class UnsupervisedConfig:
    seed: int = 42
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6)
    retained_variance: float = 0.90
    stability_iterations: int = 200
    stability_sample_fraction: float = 0.80
    min_cluster_size: int = 25

@dataclass(frozen=True)
class FittedProfiles:
    preprocessor: ColumnTransformer
    pca: PCA
    kmeans: KMeans
    selected_k: int
    retained_components: int
    development_assignments: np.ndarray
    hierarchical_assignments: np.ndarray
    linkage_matrix: np.ndarray
    cluster_selection: pd.DataFrame
    pca_summary: pd.DataFrame
    pca_loadings: pd.DataFrame
    hierarchical_comparison: pd.DataFrame

@dataclass(frozen=True)
class UnsupervisedStudy:
    selected_k: int
    retained_components: int
    cluster_selection: pd.DataFrame
    pca_summary: pd.DataFrame
    pca_loadings: pd.DataFrame
    cluster_profiles: pd.DataFrame
    patient_assignments: pd.DataFrame
    external_transfer: pd.DataFrame
    interpretation: dict[str, object]

def fit_unsupervised_profiles(
    development_features: pd.DataFrame,
    config: UnsupervisedConfig,
) -> FittedProfiles: ...

def describe_unsupervised_profiles(
    fitted: FittedProfiles,
    development: CohortData,
    external: tuple[CohortData, ...],
    config: UnsupervisedConfig,
) -> UnsupervisedStudy: ...
```

The concrete result may use smaller internal dataclasses if that makes validation and testing clearer, but the public contract must expose typed configuration, deterministic results, and data frames suitable for report generation. The fitting interface accepts features rather than `CohortData`; this structural boundary prevents a target from reaching preprocessing, PCA, or clustering.

### Workflow and CLI

The existing `reproduce` workflow will generate both analytical tracks. A dedicated command will also permit focused execution:

```text
heart-disease analyze-unsupervised --profile smoke|full
heart-disease reproduce --profile smoke|full
```

The smoke profile will reduce stability iterations while exercising the complete data flow. The full profile will generate committed evidence. Both profiles use fixed seeds.

### Reports and notebook

Generated evidence will live under a visibly separate directory:

```text
reports/unsupervised/
├── summary.json
├── pca-summary.csv
├── pca-loadings.csv
├── cluster-selection.csv
├── hierarchical-comparison.csv
├── cluster-profiles.csv
├── patient-assignments.csv
├── external-transfer.csv
└── figures/
    ├── pca-explained-variance.png
    ├── pca-clusters.png
    ├── cluster-selection.png
    ├── hierarchical-dendrogram.png
    └── cluster-profiles.png
```

The existing `notebooks/analysis.ipynb` will gain a clearly labeled unsupervised section that imports package functions or reads generated reports. It will not fit PCA or clustering independently.

## Data boundaries and preprocessing

- Cleveland remains the sole development cohort.
- Raw feature validation continues through `load_cohort`.
- Numerical features use median imputation followed by standard scaling.
- Categorical features use most-frequent imputation and dense one-hot encoding with unknown-category support.
- The Cleveland-fitted transformer is frozen before transforming Hungary, Switzerland, or VA Long Beach.
- Disease labels are stored separately from the feature path and cannot be accepted by PCA or clustering interfaces.
- The fitting entry point validates the exact 13-feature schema and rejects a target column, extra columns, missing columns, or reordered columns.
- Patient IDs are cohort-local row identifiers used only to join frozen assignments to descriptive labels and to reproduce reports.

The encoded feature space mixes standardized continuous values with one-hot categorical indicators. The report must state that Euclidean distance and PCA are pragmatic choices with limitations for mixed clinical data; no latent clinical construct is claimed.

## PCA design

- Fit deterministic full PCA on the transformed Cleveland feature matrix.
- Report explained variance per component and cumulative explained variance.
- Retain the smallest number of components whose cumulative explained variance is at least 90% for clustering; use `max(2, components_at_threshold)` so two-dimensional visualization is always available.
- Produce a two-dimensional PC1/PC2 plot colored only after cluster assignments are frozen.
- Report the largest absolute loadings per component using transformed feature names.
- Treat loading signs as directions within this fitted representation, not causal effects or clinical importance.

PCA will support representation, visualization, and clustering. It will not be inserted into the production logistic-regression pipeline unless a future, separately approved supervised experiment demonstrates a validated benefit.

## K-Means selection and stability

- Evaluate `k` from 2 through 6 using identical PCA features for every candidate.
- Use deterministic seeds and an explicit `n_init` value.
- Primary selection metric: full-Cleveland silhouette score, which does not use disease labels.
- Supporting metrics: Davies-Bouldin index and Calinski-Harabasz score.
- If silhouette scores are equal after rounding to three decimals, select the smaller `k`.
- Estimate stability through repeated seeded 80% patient subsamples. For each subsample, fit K-Means on the sampled patients and compare its assignments for those patients with the full-Cleveland model's assignments using adjusted Rand index.
- Report the entire candidate table even when evidence is weak or conflicting.
- Canonicalize final K-Means labels by lexicographically ordering centroids on PC1, then subsequent components. This makes cluster IDs deterministic while preserving the fact that the numbers have no severity meaning.

No minimum silhouette score will be imposed to manufacture a positive conclusion. The generated interpretation will state the observed separation and stability metrics without converting them into undeclared qualitative grades.

## Hierarchical comparison

- Compute one Ward linkage matrix on the retained Cleveland PCA representation, derive flat assignments at the K-Means-selected cluster count, and produce the dendrogram from that same matrix.
- Compare hierarchical and K-Means assignments using adjusted Rand index and a contingency table.
- Do not select `k` or linkage using disease labels.
- Hierarchical clustering is a development-cohort comparison only; external patients will not be assigned by inventing a prediction interface that the estimator does not provide.

## Post-hoc profiles and external transfer

After the representation, `k`, K-Means model, and hierarchical assignments are frozen:

- Report cluster size and disease prevalence.
- Report numerical medians and interquartile ranges by cluster.
- Report categorical level proportions by cluster.
- Compare frozen cluster assignments with the binary target using adjusted mutual information and a contingency table. These are descriptive associations, not supervised performance metrics.
- Assign external patients with the frozen Cleveland preprocessor, PCA transformer, and K-Means centroids.
- Report external cluster proportions, total-variation distance from Cleveland cluster proportions, distance-to-nearest-centroid summaries, and post-hoc disease prevalence by cluster.
- Mark groups below `config.min_cluster_size` as `underpowered` and avoid comparative performance claims.

External evidence cannot change retained components, selected `k`, centroids, labels, profile definitions, or interpretation thresholds.

## Recruiter browsing

The root README will gain a short **Start here** table with direct paths for four audiences:

| Recruiter question | Destination |
|---|---|
| What is the result? | README result summary and figures |
| How was supervised learning validated? | evaluation code, model card, and experiment manifest |
| Where is unsupervised learning? | unsupervised module, report index, and notebook section |
| Can the model make a prediction? | inference interface, CLI example, and artifact metadata |

Add `docs/project-guide.md` as a concise, numbered map:

1. Data acquisition and validation
2. Cleaning and leakage-safe preprocessing
3. Supervised candidates and nested tuning
4. Model selection and thresholds
5. External validation and dataset shift
6. Unsupervised PCA and patient-profile discovery
7. Final training, safe serialization, and inference
8. Generated evidence, notebook, tests, and CI

Each stage will link to canonical source files, relevant tests, and generated outputs. Add `reports/README.md` to index machine-readable tables and figures. These documents explain where code lives; they do not contain a duplicate implementation.

The README will visually distinguish:

- **Supervised track:** predicting heart-disease presence.
- **Unsupervised track:** exploring natural patient group structure without labels.
- **Engineering track:** provenance, testing, reproducibility, serialization, and inference.

## Error handling

The unsupervised workflow will fail explicitly when:

- the development cohort has fewer patients than the largest requested `k`;
- requested `k` values are less than 2 or not unique;
- retained variance is outside `(0, 1]`;
- stability iterations are less than one, the sample fraction is outside `(0, 1)`, or a stability sample cannot contain more patients than the largest requested `k`;
- minimum cluster size is less than one;
- a transformed matrix contains non-finite values;
- a clustering result contains fewer than two populated clusters;
- feature schemas differ between development and external transformation;
- generated reports contain assignments that cannot be joined one-to-one with cohort-local patient IDs.

A metric that is mathematically unavailable will be serialized as unavailable with a reason. It will not be replaced by zero or silently omitted.

## Testing

Required tests include:

```python
def test_unsupervised_fit_does_not_accept_or_read_target(): ...
def test_preprocessor_and_pca_are_fit_on_cleveland_only(): ...
def test_component_count_reaches_declared_variance(): ...
def test_cluster_selection_uses_silhouette_not_disease_labels(): ...
def test_cluster_selection_prefers_smaller_k_on_declared_tie(): ...
def test_stability_is_reproducible_with_seed(): ...
def test_hierarchical_comparison_uses_selected_k(): ...
def test_external_data_cannot_change_components_or_centroids(): ...
def test_underpowered_external_cluster_is_labeled(): ...
def test_unsupervised_reports_match_summary_json(): ...
def test_project_guide_links_to_existing_paths(): ...
def test_notebook_executes_with_unsupervised_section(): ...
```

The implementation will follow test-driven development. CI will run the smoke unsupervised workflow on Python 3.11 and 3.12. Full local reproduction will regenerate and verify all supervised and unsupervised evidence before completion.

## Dependencies

Use the existing pandas, NumPy, scikit-learn, matplotlib, and seaborn stack. Add SciPy as a direct locked dependency only because the dendrogram is a declared output; do not rely on its presence merely as a transitive scikit-learn dependency.

## Documentation and claims

- README and model card must state that clusters are exploratory groupings in an old, small dataset.
- Cluster numbers are arbitrary identifiers and do not represent severity ordering.
- Disease prevalence differences are post-hoc descriptions and do not validate the clusters clinically.
- Weak stability, overlap, or poor external transfer is a valid and publishable finding.
- AI assistance remains disclosed, including the reasoning for accepting this extension after initially rejecting disconnected clustering.

## Acceptance criteria

- The supervised production model, thresholds, artifact probability, and existing external-validation results remain unchanged.
- Target values cannot enter unsupervised fitting or selection code.
- All unsupervised transformations and centroids are fitted on Cleveland only.
- PCA variance, cluster selection, stability, hierarchical agreement, profiles, and external transfer are emitted as machine-readable evidence and figures.
- Smoke and full profiles are deterministic under the locked environment.
- The notebook consumes canonical package/report outputs and contains no second implementation.
- README, project guide, report index, notebook, summary JSON, and generated figures agree.
- A recruiter can reach the supervised methodology, unsupervised methodology, production inference, and verification evidence from the README in one click.
- All tests and CI pass on Python 3.11 and 3.12.
