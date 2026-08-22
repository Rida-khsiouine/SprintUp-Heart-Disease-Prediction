# Unsupervised Patient Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, label-isolated PCA and clustering study with external-cohort transfer evidence, then make every supervised, unsupervised, and production stage reachable from the README in one click.

**Architecture:** `heart_disease.unsupervised` owns feature-only fitting, PCA, K-Means selection and stability, Ward hierarchical comparison, post-hoc profiles, and frozen external transfer. `heart_disease.unsupervised_reporting` owns CSV/JSON/figure generation; the existing workflow and notebook consume these canonical outputs without changing the supervised production artifact.

**Tech Stack:** Python 3.11–3.12, uv, pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn, nbclient, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-22-unsupervised-patient-profiles-design.md`

## Global Constraints

- Cleveland is the only cohort permitted to fit preprocessing, PCA, cluster count, centroids, linkage, or interpretation rules.
- The exact 13-feature schema is required in canonical order; target, missing, extra, and reordered columns are rejected by the feature-only fitting entry point.
- Disease labels are joined only after every unsupervised choice is frozen.
- Hungary, Switzerland, and VA Long Beach can be transformed and described but cannot influence any fitted object or selection.
- The existing supervised selection, thresholds, probability output, external metrics, and `artifacts/model.skops` bytes must remain unchanged.
- K-Means evaluates `k=(2,3,4,5,6)`, selects maximum full-Cleveland silhouette, and prefers smaller `k` when scores tie after rounding to three decimals.
- Full PCA retains `max(2, smallest component count reaching 90% cumulative variance)`.
- Full stability uses 200 seeded 80% subsamples; smoke uses 5 while exercising the identical path.
- Cluster labels are canonicalized by lexicographic centroid order and never imply severity.
- SciPy is a direct locked dependency because the Ward linkage matrix and dendrogram are declared outputs.
- Notebook code reads package/report outputs and contains no `.fit(` call or duplicate algorithm implementation.
- All cluster claims remain exploratory, non-clinical, and explicit about mixed-data Euclidean-distance limitations.
- Use test-driven development and meaningful incremental commits.

## File Structure

**Create**

- `src/heart_disease/unsupervised.py` — typed configuration, schema checks, feature-only representation fitting, K-Means selection/stability, hierarchical comparison, profiles, and external transfer.
- `src/heart_disease/unsupervised_reporting.py` — deterministic JSON/CSV serialization and five declared figures.
- `tests/test_unsupervised.py` — label isolation, schema, PCA, selection, stability, hierarchy, profiles, and transfer tests.
- `tests/test_unsupervised_reporting.py` — output contract and synchronization tests.
- `docs/project-guide.md` — numbered recruiter map from each ML stage to canonical code, tests, and evidence.
- `reports/README.md` — index and interpretation guide for all generated evidence.
- `reports/unsupervised/*` — full-profile machine-readable evidence and figures.

**Modify**

- `src/heart_disease/models.py` — expose the existing canonical preprocessor builder for both analytical tracks.
- `src/heart_disease/workflow.py` — configure, execute, serialize, and return unsupervised outputs.
- `src/heart_disease/cli.py` — add `analyze-unsupervised` and reuse the canonical profile configuration.
- `src/heart_disease/reporting.py` — synchronize the README's unsupervised summary and record unsupervised provenance in the experiment manifest.
- `pyproject.toml`, `uv.lock` — declare and lock SciPy directly.
- `tests/test_models.py`, `tests/test_cli.py`, `tests/test_reporting.py`, `tests/test_notebook.py` — cover shared preprocessing, CLI/workflow output, documentation synchronization, and thin-notebook execution.
- `notebooks/analysis.ipynb` — add report-consuming unsupervised evidence and limitations sections.
- `README.md`, `docs/model-card.md`, `docs/ai-assisted-development.md` — navigation, result interpretation, limitations, and AI decision record.

---

### Task 1: Shared preprocessor and validated unsupervised configuration

**Files:**
- Modify: `src/heart_disease/models.py`
- Create: `src/heart_disease/unsupervised.py`
- Modify: `tests/test_models.py`
- Create: `tests/test_unsupervised.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `build_preprocessor() -> ColumnTransformer`
- Produces: `UnsupervisedConfig`, `UnsupervisedValidationError`, `validate_unsupervised_inputs(features, config) -> None`
- Consumes: `FEATURE_COLUMNS`, `NUMERICAL_COLUMNS`, `CATEGORICAL_COLUMNS`

- [ ] **Step 1: Write failing tests for the shared preprocessor and configuration contract**

Create deterministic helpers at the top of `tests/test_unsupervised.py`:

```python
def _features(rows: int, *, age_offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame({
        "age": 40.0 + age_offset + index % 35,
        "sex": index % 2,
        "cp": index % 4 + 1,
        "trestbps": 110.0 + index % 30,
        "chol": 180.0 + index % 80,
        "fbs": index % 2,
        "restecg": index % 3,
        "thalach": 180.0 - index % 60,
        "exang": index % 2,
        "oldpeak": 0.25 + (index % 8) / 2,
        "slope": index % 3 + 1,
        "ca": index % 4,
        "thal": np.asarray((3, 6, 7))[index % 3],
    }).loc[:, FEATURE_COLUMNS]

def _cohort(
    cohort: Cohort, rows: int, *, age_offset: float = 0.0
) -> CohortData:
    target = pd.Series(np.arange(rows) % 2, name="disease_present", dtype="int8")
    return CohortData(
        cohort=cohort,
        features=_features(rows, age_offset=age_offset),
        target=target,
        raw_target=target.rename("num"),
    )
```

```python
def test_build_preprocessor_is_unfitted() -> None:
    with pytest.raises(NotFittedError):
        check_is_fitted(build_preprocessor())

def test_unsupervised_fit_rejects_target_or_reordered_schema() -> None:
    features = _features(30)
    with pytest.raises(UnsupervisedValidationError, match="exact 13-feature schema"):
        validate_unsupervised_inputs(
            features.assign(disease_present=0), UnsupervisedConfig()
        )
    with pytest.raises(UnsupervisedValidationError, match="canonical order"):
        validate_unsupervised_inputs(
            features.loc[:, list(reversed(FEATURE_COLUMNS))], UnsupervisedConfig()
        )

@pytest.mark.parametrize(
    ("config", "message"),
    [
        (UnsupervisedConfig(k_values=(1, 2)), "at least 2"),
        (UnsupervisedConfig(k_values=(2, 2)), "unique"),
        (UnsupervisedConfig(retained_variance=0), "retained_variance"),
        (UnsupervisedConfig(stability_iterations=0), "stability_iterations"),
        (UnsupervisedConfig(stability_sample_fraction=1), "sample_fraction"),
        (UnsupervisedConfig(min_cluster_size=0), "min_cluster_size"),
    ],
)
def test_invalid_unsupervised_config_is_rejected(config, message) -> None:
    with pytest.raises(UnsupervisedValidationError, match=message):
        validate_unsupervised_inputs(_features(30), config)
```

- [ ] **Step 2: Run the focused tests and verify the imports fail**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_models.py tests/test_unsupervised.py -v
```

Expected: collection fails because `build_preprocessor`, `UnsupervisedConfig`, and validation interfaces do not exist.

- [ ] **Step 3: Expose the existing preprocessor without duplicating it**

Rename `_preprocessor` to `build_preprocessor` and update `build_pipeline`:

```python
def build_preprocessor() -> ColumnTransformer:
    """Create an unfitted transformer for the canonical 13 raw features."""
    numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("numeric", numerical, list(NUMERICAL_COLUMNS)),
            ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
```

- [ ] **Step 4: Add typed configuration and explicit validation**

```python
class UnsupervisedValidationError(ValueError):
    """Raised when feature-only analysis inputs violate the declared contract."""

@dataclass(frozen=True)
class UnsupervisedConfig:
    seed: int = 42
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6)
    retained_variance: float = 0.90
    stability_iterations: int = 200
    stability_sample_fraction: float = 0.80
    min_cluster_size: int = 25

def validate_unsupervised_inputs(
    features: pd.DataFrame, config: UnsupervisedConfig
) -> None:
    expected = tuple(FEATURE_COLUMNS)
    actual = tuple(features.columns)
    if set(actual) != set(expected):
        raise UnsupervisedValidationError("Expected the exact 13-feature schema")
    if actual != expected:
        raise UnsupervisedValidationError("Features must use canonical order")
    if not config.k_values or min(config.k_values) < 2:
        raise UnsupervisedValidationError("Every k value must be at least 2")
    if len(set(config.k_values)) != len(config.k_values):
        raise UnsupervisedValidationError("k_values must be unique")
    if not 0 < config.retained_variance <= 1:
        raise UnsupervisedValidationError("retained_variance must be in (0, 1]")
    if config.stability_iterations < 1:
        raise UnsupervisedValidationError("stability_iterations must be positive")
    if not 0 < config.stability_sample_fraction < 1:
        raise UnsupervisedValidationError("stability_sample_fraction must be in (0, 1)")
    if config.min_cluster_size < 1:
        raise UnsupervisedValidationError("min_cluster_size must be positive")
    sample_size = int(np.floor(len(features) * config.stability_sample_fraction))
    if len(features) <= max(config.k_values) or sample_size <= max(config.k_values):
        raise UnsupervisedValidationError("Patient and subsample counts must exceed max k")
```

Validate every numeric range from the spec, require finite row count, and require `floor(len(features) * sample_fraction) > max(k_values)`.

- [ ] **Step 5: Declare SciPy directly and update the frozen lock**

Run:

```powershell
uv add "scipy>=1.14,<2"
uv lock --check
```

Expected: `pyproject.toml` lists SciPy and `uv.lock` remains internally consistent without changing unrelated dependency families.

- [ ] **Step 6: Run focused tests and lint**

```powershell
uv run pytest tests/test_models.py tests/test_unsupervised.py -v
uv run ruff check src/heart_disease/models.py src/heart_disease/unsupervised.py tests/test_models.py tests/test_unsupervised.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml uv.lock src/heart_disease/models.py src/heart_disease/unsupervised.py tests/test_models.py tests/test_unsupervised.py
git commit -m "feat: establish unsupervised analysis contract"
```

---

### Task 2: PCA representation and deterministic K-Means selection

**Files:**
- Modify: `src/heart_disease/unsupervised.py`
- Modify: `tests/test_unsupervised.py`

**Interfaces:**
- Consumes: `build_preprocessor()`, `UnsupervisedConfig`, validated feature-only frame
- Produces: `PCARepresentation`, `KMeansSelection`
- Produces: `fit_pca_representation(features, config) -> PCARepresentation`
- Produces: `select_cluster_count(cluster_selection) -> int`
- Produces: `fit_kmeans_candidates(representation, config) -> KMeansSelection`

- [ ] **Step 1: Write failing PCA and selection tests**

```python
def test_component_count_reaches_declared_variance() -> None:
    result = fit_pca_representation(
        _features(80), UnsupervisedConfig(retained_variance=0.90)
    )
    assert result.retained_components >= 2
    assert result.pca_summary.loc[
        result.retained_components - 1, "cumulative_explained_variance"
    ] >= 0.90

def test_cluster_selection_prefers_smaller_k_on_declared_tie() -> None:
    table = pd.DataFrame(
        {"k": [2, 3, 4], "silhouette": [0.40149, 0.40148, 0.39]}
    )
    assert select_cluster_count(table) == 2

def test_same_seed_produces_same_pca_and_cluster_assignments() -> None:
    first = fit_kmeans_candidates(
        fit_pca_representation(_features(90), UnsupervisedConfig(seed=17)),
        UnsupervisedConfig(seed=17),
    )
    second = fit_kmeans_candidates(
        fit_pca_representation(_features(90), UnsupervisedConfig(seed=17)),
        UnsupervisedConfig(seed=17),
    )
    np.testing.assert_array_equal(first.assignments, second.assignments)
    pd.testing.assert_frame_equal(first.cluster_selection, second.cluster_selection)

def test_collapsed_candidate_is_rejected(monkeypatch) -> None:
    representation = fit_pca_representation(_features(60), UnsupervisedConfig())
    monkeypatch.setattr(KMeans, "fit_predict", lambda self, values: np.zeros(len(values), dtype=int))
    with pytest.raises(UnsupervisedValidationError, match="populated clusters"):
        fit_kmeans_candidates(representation, UnsupervisedConfig())
```

- [ ] **Step 2: Run the tests and confirm missing interfaces**

```powershell
uv run pytest tests/test_unsupervised.py -k "component_count or cluster_selection or same_seed" -v
```

Expected: FAIL because PCA and K-Means interfaces are absent.

- [ ] **Step 3: Implement the typed representation**

```python
@dataclass(frozen=True)
class PCARepresentation:
    preprocessor: ColumnTransformer
    pca: PCA
    transformed: np.ndarray
    retained: np.ndarray
    retained_components: int
    feature_names: tuple[str, ...]
    pca_summary: pd.DataFrame
    pca_loadings: pd.DataFrame

def fit_pca_representation(
    features: pd.DataFrame, config: UnsupervisedConfig
) -> PCARepresentation:
    validate_unsupervised_inputs(features, config)
    preprocessor = build_preprocessor().fit(features)
    transformed = np.asarray(preprocessor.transform(features), dtype=float)
    if not np.isfinite(transformed).all():
        raise UnsupervisedValidationError("Transformed features contain non-finite values")
    pca = PCA(svd_solver="full").fit(transformed)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    retained_components = max(2, int(np.searchsorted(cumulative, config.retained_variance) + 1))
    retained = pca.transform(transformed)[:, :retained_components]
    feature_names = tuple(str(name) for name in preprocessor.get_feature_names_out())
    pca_summary = pd.DataFrame({
        "component": np.arange(1, len(cumulative) + 1),
        "explained_variance": pca.explained_variance_ratio_,
        "cumulative_explained_variance": cumulative,
        "retained": np.arange(1, len(cumulative) + 1) <= retained_components,
    })
    pca_loadings = pd.DataFrame(
        pca.components_,
        columns=feature_names,
        index=np.arange(1, len(pca.components_) + 1),
    ).rename_axis("component").reset_index().melt(
        id_vars="component", var_name="feature", value_name="loading"
    )
    return PCARepresentation(
        preprocessor=preprocessor,
        pca=pca,
        transformed=transformed,
        retained=retained,
        retained_components=retained_components,
        feature_names=feature_names,
        pca_summary=pca_summary,
        pca_loadings=pca_loadings,
    )
```

Create one summary row per component and one loading row per component/transformed feature.

- [ ] **Step 4: Implement candidate metrics, selection, and canonical labels**

```python
@dataclass(frozen=True)
class KMeansSelection:
    kmeans: KMeans
    selected_k: int
    assignments: np.ndarray
    cluster_selection: pd.DataFrame

def select_cluster_count(cluster_selection: pd.DataFrame) -> int:
    ranked = cluster_selection.assign(
        rounded_silhouette=cluster_selection["silhouette"].round(3)
    ).sort_values(["rounded_silhouette", "k"], ascending=[False, True])
    return int(ranked.iloc[0]["k"])
```

For every `k`, fit `KMeans(n_clusters=k, random_state=config.seed, n_init=50)` and report silhouette, Davies-Bouldin, and Calinski-Harabasz. Reject a candidate when `np.unique(assignments).size != k`. Refit the selected candidate and remap both `labels_` and `cluster_centers_` by lexicographically sorted centroids so subsequent `.predict()` calls emit canonical labels.

- [ ] **Step 5: Run focused and regression tests**

```powershell
uv run pytest tests/test_unsupervised.py tests/test_models.py -v
uv run ruff check src/heart_disease/unsupervised.py tests/test_unsupervised.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/heart_disease/unsupervised.py tests/test_unsupervised.py
git commit -m "feat: add PCA and deterministic cluster selection"
```

---

### Task 3: Stability evidence and Ward hierarchical comparison

**Files:**
- Modify: `src/heart_disease/unsupervised.py`
- Modify: `tests/test_unsupervised.py`

**Interfaces:**
- Consumes: `PCARepresentation`, `KMeansSelection`, `UnsupervisedConfig`
- Produces: `estimate_cluster_stability(representation, selection, config) -> pd.DataFrame`
- Produces: `fit_hierarchical_comparison(representation, selection) -> HierarchicalComparison`

- [ ] **Step 1: Write failing reproducibility and hierarchy tests**

```python
def test_stability_is_reproducible_with_seed() -> None:
    config = UnsupervisedConfig(seed=23, stability_iterations=8)
    representation = fit_pca_representation(_features(100), config)
    selection = fit_kmeans_candidates(representation, config)
    first = estimate_cluster_stability(representation, selection, config)
    second = estimate_cluster_stability(representation, selection, config)
    pd.testing.assert_frame_equal(first, second)

def test_hierarchical_comparison_uses_selected_k() -> None:
    config = UnsupervisedConfig(k_values=(2, 3), stability_iterations=2)
    representation = fit_pca_representation(_features(80), config)
    selection = fit_kmeans_candidates(representation, config)
    comparison = fit_hierarchical_comparison(representation, selection)
    assert np.unique(comparison.assignments).size == selection.selected_k
    assert comparison.linkage_matrix.shape == (len(representation.retained) - 1, 4)
    assert comparison.metrics.loc[0, "adjusted_rand_index"] <= 1
```

- [ ] **Step 2: Run focused tests and verify missing functions**

```powershell
uv run pytest tests/test_unsupervised.py -k "stability or hierarchical" -v
```

Expected: FAIL because stability and hierarchy functions are absent.

- [ ] **Step 3: Implement subsample stability without labels**

Use `np.random.default_rng(config.seed)`, sample without replacement, fit the same selected `k`, predict the sampled rows, and compare against the full-fit assignments with `adjusted_rand_score`. Emit `iteration`, `sample_size`, and `adjusted_rand_index`, then add mean and standard deviation columns to the candidate selection row for the selected `k`.

```python
indices = generator.choice(
    len(representation.retained), size=sample_size, replace=False
)
subsample_model = KMeans(
    n_clusters=selection.selected_k,
    random_state=config.seed + iteration,
    n_init=50,
).fit(representation.retained[indices])
ari = adjusted_rand_score(
    selection.assignments[indices], subsample_model.predict(representation.retained[indices])
)
```

- [ ] **Step 4: Implement one Ward linkage source for assignments and dendrogram**

```python
@dataclass(frozen=True)
class HierarchicalComparison:
    linkage_matrix: np.ndarray
    assignments: np.ndarray
    metrics: pd.DataFrame

matrix = linkage(representation.retained, method="ward", optimal_ordering=True)
assignments = fcluster(matrix, t=selection.selected_k, criterion="maxclust") - 1
metrics = pd.DataFrame(
    [{
        "selected_k": selection.selected_k,
        "adjusted_rand_index": adjusted_rand_score(selection.assignments, assignments),
    }]
)
```

Add the K-Means-by-hierarchical contingency counts as deterministic columns or long-form rows in `metrics`.

- [ ] **Step 5: Run focused tests and lint**

```powershell
uv run pytest tests/test_unsupervised.py -v
uv run ruff check src/heart_disease/unsupervised.py tests/test_unsupervised.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/heart_disease/unsupervised.py tests/test_unsupervised.py
git commit -m "feat: measure cluster stability and hierarchy agreement"
```

---

### Task 4: Frozen post-hoc profiles and external transfer

**Files:**
- Modify: `src/heart_disease/unsupervised.py`
- Modify: `tests/test_unsupervised.py`

**Interfaces:**
- Produces: final `FittedProfiles`, `UnsupervisedStudy`
- Produces: `fit_unsupervised_profiles(features, config) -> FittedProfiles`
- Produces: `describe_unsupervised_profiles(fitted, development, external, config) -> UnsupervisedStudy`

- [ ] **Step 1: Write failing target-isolation, frozen-state, and underpowered tests**

```python
def test_fit_is_independent_of_target_values() -> None:
    features = _features(100)
    first = fit_unsupervised_profiles(features, UnsupervisedConfig(stability_iterations=3))
    second = fit_unsupervised_profiles(features.copy(), UnsupervisedConfig(stability_iterations=3))
    np.testing.assert_array_equal(first.development_assignments, second.development_assignments)
    np.testing.assert_array_equal(first.kmeans.cluster_centers_, second.kmeans.cluster_centers_)

def test_external_data_cannot_change_components_or_centroids() -> None:
    development = _cohort(Cohort.CLEVELAND, rows=100)
    external = _cohort(Cohort.HUNGARY, rows=40, age_offset=10_000)
    fitted = fit_unsupervised_profiles(
        development.features, UnsupervisedConfig(stability_iterations=3)
    )
    components = fitted.pca.components_.copy()
    centroids = fitted.kmeans.cluster_centers_.copy()
    describe_unsupervised_profiles(
        fitted, development, (external,), UnsupervisedConfig(stability_iterations=3)
    )
    np.testing.assert_array_equal(fitted.pca.components_, components)
    np.testing.assert_array_equal(fitted.kmeans.cluster_centers_, centroids)

def test_small_external_cluster_is_marked_underpowered() -> None:
    config = UnsupervisedConfig(
        k_values=(2,), stability_iterations=2, min_cluster_size=25
    )
    development = _cohort(Cohort.CLEVELAND, rows=80)
    external = _cohort(Cohort.HUNGARY, rows=12)
    fitted = fit_unsupervised_profiles(development.features, config)
    study = describe_unsupervised_profiles(
        fitted, development, (external,), config
    )
    assert "underpowered" in set(study.external_transfer["status"])

def test_assignment_join_must_be_one_to_one() -> None:
    config = UnsupervisedConfig(k_values=(2,), stability_iterations=2)
    development = _cohort(Cohort.CLEVELAND, rows=80)
    malformed = dataclasses.replace(
        development, target=development.target.iloc[:-1]
    )
    fitted = fit_unsupervised_profiles(development.features, config)
    with pytest.raises(UnsupervisedValidationError, match="one-to-one"):
        describe_unsupervised_profiles(fitted, malformed, (), config)
```

- [ ] **Step 2: Run the focused tests and verify missing orchestration**

```powershell
uv run pytest tests/test_unsupervised.py -k "independent_of_target or external_data or underpowered" -v
```

Expected: FAIL because final orchestration and profile tables are absent.

- [ ] **Step 3: Assemble immutable fitted evidence**

`fit_unsupervised_profiles` calls validation, PCA, candidate selection, stability, and hierarchy in that order. Return all fitted objects and evidence in `FittedProfiles`; do not accept a target argument.

```python
@dataclass(frozen=True)
class FittedProfiles:
    preprocessor: ColumnTransformer
    pca: PCA
    kmeans: KMeans
    selected_k: int
    retained_components: int
    development_embedding: np.ndarray
    development_assignments: np.ndarray
    hierarchical_assignments: np.ndarray
    linkage_matrix: np.ndarray
    cluster_selection: pd.DataFrame
    stability: pd.DataFrame
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
    hierarchical_comparison: pd.DataFrame
    cluster_profiles: pd.DataFrame
    patient_assignments: pd.DataFrame
    external_transfer: pd.DataFrame
    interpretation: dict[str, object]
```

- [ ] **Step 4: Build deterministic post-hoc tables**

`patient_assignments` columns:

```text
cohort, patient_id, kmeans_cluster, hierarchical_cluster, pc1, pc2,
centroid_distance, disease_present
```

`cluster_profiles` is long-form with:

```text
cohort, cluster, n, status, feature, statistic, level, value
```

For numeric features emit `median`, `q1`, and `q3`; for categorical features emit a `proportion` row for every observed level, including `missing`.

`external_transfer` columns:

```text
cohort, cluster, n, proportion, disease_prevalence, median_centroid_distance,
cluster_proportion_total_variation, status
```

Transform external features using only `fitted.preprocessor.transform`, `fitted.pca.transform`, and `fitted.kmeans.predict`. Compute adjusted mutual information only after assignments are complete and store it in `interpretation` alongside numeric silhouette, stability, and hierarchy agreement.

Before joining descriptive targets, assert that every `(cohort, patient_id)` key is unique and that assignment and cohort lengths match exactly; otherwise raise `UnsupervisedValidationError("Assignments must join one-to-one with patients")`.

- [ ] **Step 5: Run all unsupervised tests and lint**

```powershell
uv run pytest tests/test_unsupervised.py -v
uv run ruff check src/heart_disease/unsupervised.py tests/test_unsupervised.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/heart_disease/unsupervised.py tests/test_unsupervised.py
git commit -m "feat: profile and transfer frozen patient clusters"
```

---

### Task 5: Machine-readable unsupervised reports and figures

**Files:**
- Create: `src/heart_disease/unsupervised_reporting.py`
- Create: `tests/test_unsupervised_reporting.py`
- Modify: `src/heart_disease/reporting.py`
- Modify: `tests/test_reporting.py`

**Interfaces:**
- Consumes: `FittedProfiles`, `UnsupervisedStudy`, output directory
- Produces: `write_unsupervised_reports(*, fitted, study, profile, config, reports_dir) -> dict[str, Path]`
- Produces: `render_unsupervised_markdown(summary) -> str`, `sync_readme_unsupervised(readme_path, summary) -> None`

- [ ] **Step 1: Write failing output-contract and synchronization tests**

```python
PROJECT_ROOT = Path(__file__).parents[1]

def _study_fixture() -> tuple[FittedProfiles, UnsupervisedStudy, UnsupervisedConfig]:
    data_dir = PROJECT_ROOT / "data"
    development = load_cohort(Cohort.CLEVELAND, data_dir)
    external = tuple(
        load_cohort(cohort, data_dir)
        for cohort in (Cohort.HUNGARY, Cohort.SWITZERLAND, Cohort.VA)
    )
    config = UnsupervisedConfig(k_values=(2, 3), stability_iterations=2)
    fitted = fit_unsupervised_profiles(development.features, config)
    study = describe_unsupervised_profiles(fitted, development, external, config)
    return fitted, study, config

def test_unsupervised_reports_write_declared_outputs(tmp_path: Path) -> None:
    fitted, study, config = _study_fixture()
    paths = write_unsupervised_reports(
        fitted=fitted,
        study=study,
        profile="smoke",
        config=config,
        reports_dir=tmp_path,
    )
    expected = {
        "summary", "pca_summary", "pca_loadings", "cluster_selection",
        "hierarchical_comparison", "cluster_profiles", "patient_assignments",
        "external_transfer", "figures",
    }
    assert expected == set(paths)
    assert all(path.exists() for path in paths.values())

def test_unsupervised_readme_sync_uses_summary_only(tmp_path: Path) -> None:
    summary = {
        "selected_k": 2,
        "retained_components": 7,
        "retained_variance": 0.91,
        "silhouette": 0.24,
        "stability_ari_mean": 0.72,
        "hierarchical_ari": 0.55,
        "external_transfer": [],
    }
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        f"before\n{UNSUPERVISED_START}\nstale\n{UNSUPERVISED_END}\nafter\n",
        encoding="utf-8",
    )
    sync_readme_unsupervised(readme_path, summary)
    readme = readme_path.read_text(encoding="utf-8")
    generated = readme.split(UNSUPERVISED_START, 1)[1].split(UNSUPERVISED_END, 1)[0].strip()
    assert generated == render_unsupervised_markdown(summary).strip()
```

- [ ] **Step 2: Run focused tests and verify they fail**

```powershell
uv run pytest tests/test_unsupervised_reporting.py tests/test_reporting.py -v
```

Expected: FAIL because report functions, markers, and committed outputs are absent.

- [ ] **Step 3: Serialize deterministic JSON and CSV evidence**

Write `summary.json` with `profile`, `config=asdict(config)`, fit-boundary booleans, selected `k`, retained-component evidence, cluster metrics, and external-transfer summaries. Use sorted keys, two-space indentation, `allow_nan=False`, and a trailing newline. Write every DataFrame with stable row ordering, explicit columns, `index=False`, and `lineterminator="\n"`. Represent unavailable values with `status` and `reason` fields before JSON serialization rather than NaN.

- [ ] **Step 4: Generate the five declared figures**

Use the existing headless Matplotlib configuration and `_save_figure` conventions:

1. cumulative and per-component explained variance;
2. PC1/PC2 scatter colored by canonical K-Means cluster;
3. silhouette, Davies-Bouldin, and stability evidence by `k` without mixed-axis deception;
4. Ward dendrogram from `fitted.linkage_matrix`;
5. heatmap of cluster-wise standardized numeric medians.

Every title and caption must say `exploratory`; no figure may label clusters as disease classes.

- [ ] **Step 5: Add fixed README markers and renderer**

```python
UNSUPERVISED_START = "<!-- GENERATED_UNSUPERVISED_START -->"
UNSUPERVISED_END = "<!-- GENERATED_UNSUPERVISED_END -->"
```

Render selected `k`, retained component count and variance, silhouette, mean stability ARI, hierarchical agreement ARI, and external transfer distances from `summary.json` only.

- [ ] **Step 6: Run report tests and lint**

```powershell
uv run pytest tests/test_unsupervised_reporting.py tests/test_reporting.py -v
uv run ruff check src/heart_disease/unsupervised_reporting.py src/heart_disease/reporting.py tests/test_unsupervised_reporting.py tests/test_reporting.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/heart_disease/unsupervised_reporting.py src/heart_disease/reporting.py tests/test_unsupervised_reporting.py tests/test_reporting.py
git commit -m "feat: report unsupervised evidence"
```

---

### Task 6: Workflow, provenance, and CLI integration

**Files:**
- Modify: `src/heart_disease/workflow.py`
- Modify: `src/heart_disease/cli.py`
- Modify: `src/heart_disease/reporting.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_reporting.py`

**Interfaces:**
- Produces: `unsupervised_config_for_profile(profile) -> UnsupervisedConfig`
- Produces: `analyze_unsupervised(data_dir, output_dir, profile) -> dict[str, Path]`
- Extends: `reproduce_study` return mapping and experiment manifest

- [ ] **Step 1: Write failing CLI and provenance tests**

```python
def test_cli_smoke_unsupervised_creates_expected_outputs(tmp_path: Path) -> None:
    exit_code = main([
        "analyze-unsupervised", "--profile", "smoke",
        "--data-dir", str(PROJECT_ROOT / "data"),
        "--output-dir", str(tmp_path),
    ])
    assert exit_code == 0
    assert (tmp_path / "reports/unsupervised/summary.json").is_file()
    assert (tmp_path / "reports/unsupervised/figures/pca-clusters.png").is_file()

def test_smoke_reproduce_includes_unsupervised_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation, "CANDIDATE_MODELS", ("dummy", "logistic"))
    exit_code = main([
        "reproduce", "--profile", "smoke",
        "--data-dir", str(PROJECT_ROOT / "data"),
        "--output-dir", str(tmp_path),
    ])
    assert exit_code == 0
    assert (tmp_path / "reports/unsupervised/external-transfer.csv").is_file()

def test_manifest_builder_records_unsupervised_boundaries() -> None:
    manifest = build_experiment_manifest(
        profile="smoke",
        config=ExperimentConfig(outer_splits=2, outer_repeats=1, inner_splits=2),
        data_manifest=json.loads((PROJECT_ROOT / "data/manifest.json").read_text()),
        artifact_metadata=json.loads((PROJECT_ROOT / "artifacts/metadata.json").read_text()),
        selection=Selection("logistic", "test", 0.5, 0.4),
        parameters={},
        unsupervised_summary={
            "config": {"seed": 42},
            "selected_k": 2,
            "retained_components": 7,
        },
    )
    assert manifest["partitions"]["unsupervised_fit"] == ["cleveland"]
    assert manifest["unsupervised"]["target_used_for_fit"] is False
```

- [ ] **Step 2: Run focused tests and verify missing command/output**

```powershell
uv run pytest tests/test_cli.py tests/test_reporting.py -v
```

Expected: FAIL because `analyze-unsupervised` and manifest fields do not exist.

- [ ] **Step 3: Add smoke and full unsupervised configuration**

```python
def unsupervised_config_for_profile(profile: Profile) -> UnsupervisedConfig:
    if profile == "smoke":
        return UnsupervisedConfig(stability_iterations=5)
    if profile == "full":
        return UnsupervisedConfig()
    raise ValueError(f"Unknown reproduction profile: {profile}")
```

- [ ] **Step 4: Add focused orchestration and compose it into reproduction**

```python
def analyze_unsupervised(*, profile: Profile, data_dir: Path, output_dir: Path) -> dict[str, Path]:
    config = unsupervised_config_for_profile(profile)
    development = load_cohort(Cohort.CLEVELAND, data_dir)
    external = tuple(load_cohort(cohort, data_dir) for cohort in EXTERNAL_COHORTS)
    fitted = fit_unsupervised_profiles(development.features, config)
    study = describe_unsupervised_profiles(fitted, development, external, config)
    paths = write_unsupervised_reports(
        fitted=fitted,
        study=study,
        profile=profile,
        config=config,
        reports_dir=output_dir / "reports",
    )
    readme_path = output_dir / "README.md"
    if readme_path.is_file():
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        sync_readme_unsupervised(readme_path, summary)
    return paths
```

Within `reproduce_study`, call this after supervised model selection, final fit, external evaluation, and artifact save but before `build_experiment_manifest`. Load `paths["summary"]`, pass it as the new `unsupervised_summary` argument to the manifest builder, then generate the existing supervised evidence. Prefix returned keys with `unsupervised_` to avoid collisions. This ordering lets the manifest describe both tracks without letting unsupervised results affect any supervised operation.

- [ ] **Step 5: Add the CLI subcommand**

Add `analyze-unsupervised` with required `--profile` and the same `--data-dir`/`--output-dir` defaults as `reproduce`. Print the returned path mapping as JSON.

- [ ] **Step 6: Extend provenance without changing supervised evidence**

Add:

```python
manifest["partitions"]["unsupervised_fit"] = ["cleveland"]
manifest["unsupervised"] = {
    "target_used_for_fit": False,
    "external_used_for_selection": False,
    "config": unsupervised_summary["config"],
    "selected_k": int(unsupervised_summary["selected_k"]),
    "retained_components": int(unsupervised_summary["retained_components"]),
}
```

Keep every existing manifest key intact.

- [ ] **Step 7: Run CLI smoke tests and supervised regressions**

```powershell
uv run pytest tests/test_cli.py tests/test_reporting.py tests/test_artifacts.py tests/test_evaluation.py -v
uv run ruff check src/heart_disease/workflow.py src/heart_disease/cli.py src/heart_disease/reporting.py tests/test_cli.py tests/test_reporting.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add src/heart_disease/workflow.py src/heart_disease/cli.py src/heart_disease/reporting.py tests/test_cli.py tests/test_reporting.py
git commit -m "feat: integrate unsupervised reproduction workflow"
```

---

### Task 7: Recruiter navigation and thin notebook

**Files:**
- Create: `docs/project-guide.md`
- Create: `reports/README.md`
- Modify: `README.md`
- Modify: `docs/model-card.md`
- Modify: `docs/ai-assisted-development.md`
- Modify: `notebooks/analysis.ipynb`
- Modify: `tests/test_reporting.py`
- Modify: `tests/test_notebook.py`

**Interfaces:**
- Consumes: canonical source paths and generated unsupervised reports
- Produces: one-click recruiter navigation and executable report-consumer notebook

- [ ] **Step 1: Write failing navigation, limitation, and notebook tests**

```python
def test_project_guide_links_to_existing_paths() -> None:
    guide = (PROJECT_ROOT / "docs/project-guide.md").read_text(encoding="utf-8")
    linked_paths = re.findall(r"\]\((?!https?://)([^)#]+)", guide)
    assert linked_paths
    assert all((PROJECT_ROOT / "docs" / path).resolve().exists() for path in linked_paths)

def test_readme_exposes_three_recruiter_tracks() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "supervised track" in readme
    assert "unsupervised track" in readme
    assert "engineering track" in readme
    assert "docs/project-guide.md" in readme

def test_model_card_limits_cluster_claims() -> None:
    card = (PROJECT_ROOT / "docs/model-card.md").read_text(encoding="utf-8").lower()
    assert "exploratory" in card
    assert "cluster numbers" in card
    assert "not clinical" in card
```

Extend the notebook test:

```python
assert "reports/unsupervised/summary.json" in code
assert "reports/unsupervised/cluster-selection.csv" in code
assert ".fit(" not in code
```

- [ ] **Step 2: Run focused tests and verify missing navigation**

```powershell
uv run pytest tests/test_reporting.py tests/test_notebook.py -v
```

Expected: FAIL because the guide, index, markers, limitations, and notebook section are absent.

- [ ] **Step 3: Add the README Start-here table and three-track map**

Place the recruiter table before detailed results. Link directly to supervised methodology, `src/heart_disease/unsupervised.py`, `reports/README.md`, inference code, model card, and project guide. Add the fixed unsupervised generated markers consumed by Task 5.

- [ ] **Step 4: Write the numbered project guide and report index**

For each of the eight stages declared by the spec, include exactly three links: canonical source, strongest test, and generated evidence. Use paths relative to `docs/project-guide.md`. In `reports/README.md`, explain every CSV/JSON/figure, whether it is supervised, unsupervised, or engineering evidence, and what it cannot prove.

- [ ] **Step 5: Extend the model card and AI decision record**

State that PCA/K-Means use Euclidean geometry over scaled numeric and one-hot categorical data, cluster IDs are arbitrary, disease association is post-hoc, and no clinical subtype was discovered. Record why the earlier disconnected clustering suggestion was rejected and why the question-driven replacement was accepted with tests.

- [ ] **Step 6: Extend the notebook as a report consumer**

Using an `apply_patch` edit to the notebook JSON, add:

1. an **Unsupervised patient-profile discovery** heading;
2. code loading `summary.json`, `cluster-selection.csv`, `cluster-profiles.csv`, and `external-transfer.csv`;
3. displays for the selection table and PCA/cluster figures;
4. a limitations cell stating labels were post-hoc and clusters are not diagnoses.

Do not add sklearn imports, `.fit(`, cluster selection, or preprocessing code to the notebook.

- [ ] **Step 7: Run documentation and clean-kernel tests**

```powershell
uv run pytest tests/test_reporting.py tests/test_notebook.py -v
uv run ruff check .
```

Expected: PASS except synchronization assertions requiring Task 8's generated evidence.

- [ ] **Step 8: Commit**

```powershell
git add README.md docs/project-guide.md reports/README.md docs/model-card.md docs/ai-assisted-development.md notebooks/analysis.ipynb tests/test_reporting.py tests/test_notebook.py
git commit -m "docs: add recruiter paths through every ML stage"
```

---

### Task 8: Generate full unsupervised evidence and synchronize outputs

**Files:**
- Create: `reports/unsupervised/summary.json`
- Create: `reports/unsupervised/pca-summary.csv`
- Create: `reports/unsupervised/pca-loadings.csv`
- Create: `reports/unsupervised/cluster-selection.csv`
- Create: `reports/unsupervised/hierarchical-comparison.csv`
- Create: `reports/unsupervised/cluster-profiles.csv`
- Create: `reports/unsupervised/patient-assignments.csv`
- Create: `reports/unsupervised/external-transfer.csv`
- Create: `reports/unsupervised/figures/*.png`
- Modify: `README.md`
- Modify: `reports/experiment-manifest.json`
- Modify: `tests/test_unsupervised_reporting.py`

**Interfaces:**
- Consumes: full unsupervised workflow and committed raw cohort hashes
- Produces: synchronized, deterministic portfolio evidence

- [ ] **Step 1: Capture supervised invariants before generation**

Run and record:

```powershell
Get-FileHash artifacts/model.skops -Algorithm SHA256
Get-Content -Raw reports/metrics.json
Get-Content -Raw reports/external-validation.csv
```

Expected model hash before generation:

```text
443cf6971fa5daaecab29ca4fed7241aab608c89d0e3bda60639dc96329fa2ba
```

- [ ] **Step 2: Generate the full unsupervised report**

```powershell
uv run heart-disease analyze-unsupervised --profile full --data-dir data --output-dir .
```

Expected: all eight machine-readable outputs and five figures are created; summary declares Cleveland-only fitting and label-free selection.

- [ ] **Step 3: Synchronize README and overall manifest through full reproduction**

```powershell
uv run heart-disease reproduce --profile full --data-dir data --output-dir .
```

Expected: supervised metrics are numerically unchanged, the artifact hash remains the captured value, unsupervised markers are populated, and the experiment manifest records both tracks.

- [ ] **Step 4: Add committed-output assertions using observed evidence**

Update `tests/test_unsupervised_reporting.py` to assert:

- summary profile equals `full`;
- selected `k` belongs to `[2, 3, 4, 5, 6]`;
- retained cumulative variance is at least `0.90`;
- all four cohort names appear in assignments/transfer evidence as declared;
- summary values equal the corresponding selected rows in CSV files;
- every declared figure exists and has non-zero size.

Add the repository-level synchronization check only after the full files exist:

```python
def test_committed_readme_unsupervised_results_match_summary() -> None:
    summary = json.loads(
        (PROJECT_ROOT / "reports/unsupervised/summary.json").read_text(encoding="utf-8")
    )
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    generated = readme.split(UNSUPERVISED_START, 1)[1].split(
        UNSUPERVISED_END, 1
    )[0].strip()
    assert generated == render_unsupervised_markdown(summary).strip()
```

- [ ] **Step 5: Verify deterministic unsupervised bytes**

Generate two smoke outputs in separate temporary directories and compare SHA-256 for every JSON and CSV under `reports/unsupervised`. Compare numeric source tables rather than PNG hashes across operating systems.

```powershell
uv run heart-disease analyze-unsupervised --profile smoke --data-dir data --output-dir work\unsup-smoke-a
uv run heart-disease analyze-unsupervised --profile smoke --data-dir data --output-dir work\unsup-smoke-b
```

Expected: all machine-readable hashes match pairwise.

- [ ] **Step 6: Run focused tests**

```powershell
uv run pytest tests/test_unsupervised.py tests/test_unsupervised_reporting.py tests/test_reporting.py tests/test_cli.py tests/test_notebook.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit generated evidence**

```powershell
git add reports/unsupervised README.md reports/experiment-manifest.json tests/test_unsupervised_reporting.py
git commit -m "docs: publish unsupervised patient-profile evidence"
```

---

### Task 9: Full verification, audit, and PR update

**Files:**
- Inspect: all changed files and generated artifacts
- Modify only if verification exposes a defect; use a fresh failing regression test before each correction

**Interfaces:**
- Produces: verified feature branch and updated PR evidence

- [ ] **Step 1: Verify the locked environment**

```powershell
uv sync --frozen --all-extras
uv lock --check
```

Expected: environment sync and lock validation succeed on Python 3.12 locally; CI covers Python 3.11.

- [ ] **Step 2: Run lint and the complete non-full suite with coverage**

```powershell
uv run ruff check .
uv run pytest -m "not full" --cov=heart_disease --cov-report=term-missing
```

Expected: all tests pass and package coverage does not fall below the existing 90% baseline.

- [ ] **Step 3: Re-run critical user-facing commands**

```powershell
uv run heart-disease validate-data
uv run heart-disease analyze-unsupervised --profile smoke
uv run heart-disease predict --input tests/fixtures/patient.json
uv run pytest tests/test_notebook.py -v
```

Expected: 920 raw records validate; unsupervised smoke outputs are generated; prediction probability remains approximately `0.440740`; the notebook executes from a clean kernel.

- [ ] **Step 4: Prove supervised invariants**

Compare against the pre-change evidence:

- artifact SHA-256 is `443cf6971fa5daaecab29ca4fed7241aab608c89d0e3bda60639dc96329fa2ba`;
- selected model remains logistic;
- default threshold remains `0.5`;
- screening threshold remains `0.3938657755782479`;
- Cleveland selected OOF ROC-AUC remains approximately `0.9163`;
- Hungary, Switzerland, and VA ROC-AUC values remain approximately `0.8958`, `0.7522`, and `0.7067`.

- [ ] **Step 5: Inspect scientific and recruiter-facing boundaries**

Run:

```powershell
rg -n "phenotype|subtype|diagnos|causal|clinically validated|medical advice" README.md docs reports/README.md src/heart_disease/unsupervised.py
rg -n "\.fit\(|KMeans|PCA|linkage" notebooks/analysis.ipynb
git diff --check
git status --short
```

Expected: cautious limitation language is present; notebook contains no fitting implementation; diff check is clean; only intended generated or source changes remain.

- [ ] **Step 6: Inspect the complete branch diff**

```powershell
git diff 1975c4c...HEAD --stat
git diff 1975c4c...HEAD
```

Check for target leakage, external-selection leakage, stale generated values, copied legacy code, unsafe medical claims, nondeterministic order, secret material, and broken relative links.

- [ ] **Step 7: Commit any provenance-only refresh**

If the only post-verification change is the experiment manifest's generating commit, commit exactly that file:

```powershell
git add reports/experiment-manifest.json
git commit -m "docs: refresh unsupervised experiment provenance"
```

- [ ] **Step 8: Push and confirm CI**

```powershell
git push origin feat/recruiter-ready-rebuild
gh pr checks 1 --watch
```

Expected: Python 3.11 and 3.12 checks pass; the manual full nested-reproduction job remains intentionally skipped in ordinary PR CI.

- [ ] **Step 9: Update the PR description**

Add an unsupervised methodology section with selected `k`, retained variance, stability, hierarchy agreement, external transfer findings, reproduction command, and limitations. Add direct links to `docs/project-guide.md` and `reports/README.md`.
