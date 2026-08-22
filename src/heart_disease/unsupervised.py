"""Label-isolated PCA and clustering analysis across UCI cohorts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score

from heart_disease.models import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    build_preprocessor,
)
from heart_disease.validation import FEATURE_COLUMNS, CohortData


class UnsupervisedValidationError(ValueError):
    """Raised when unsupervised inputs violate the declared study contract."""


@dataclass(frozen=True)
class UnsupervisedConfig:
    """Deterministic configuration for patient-profile discovery."""

    seed: int = 42
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6)
    retained_variance: float = 0.90
    stability_iterations: int = 200
    stability_sample_fraction: float = 0.80
    min_cluster_size: int = 25


@dataclass(frozen=True)
class PCARepresentation:
    """Cleveland-fitted preprocessing and principal-component evidence."""

    preprocessor: ColumnTransformer
    pca: PCA
    transformed: np.ndarray
    retained: np.ndarray
    retained_components: int
    feature_names: tuple[str, ...]
    pca_summary: pd.DataFrame
    pca_loadings: pd.DataFrame


@dataclass(frozen=True)
class KMeansSelection:
    """Frozen K-Means selection and label-free candidate evidence."""

    kmeans: KMeans
    selected_k: int
    assignments: np.ndarray
    cluster_selection: pd.DataFrame


@dataclass(frozen=True)
class HierarchicalComparison:
    """Ward linkage, flat assignments, and agreement with K-Means."""

    linkage_matrix: np.ndarray
    assignments: np.ndarray
    metrics: pd.DataFrame


@dataclass(frozen=True)
class FittedProfiles:
    """All Cleveland-fitted objects and label-free selection evidence."""

    config: UnsupervisedConfig
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
    """Post-hoc profiles and frozen external-transfer evidence."""

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


def validate_unsupervised_inputs(
    features: pd.DataFrame,
    config: UnsupervisedConfig,
) -> None:
    """Validate the feature-only schema and bounded experiment settings."""

    expected = tuple(FEATURE_COLUMNS)
    actual = tuple(features.columns)
    if set(actual) != set(expected):
        raise UnsupervisedValidationError(
            "Expected the exact 13-feature schema without targets or extras"
        )
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
        raise UnsupervisedValidationError(
            "stability_sample_fraction must be in (0, 1)"
        )
    if config.min_cluster_size < 1:
        raise UnsupervisedValidationError("min_cluster_size must be positive")

    sample_size = int(np.floor(len(features) * config.stability_sample_fraction))
    largest_k = max(config.k_values)
    if len(features) <= largest_k or sample_size <= largest_k:
        raise UnsupervisedValidationError(
            "Patient and stability subsample counts must exceed the largest k"
        )


def fit_pca_representation(
    features: pd.DataFrame,
    config: UnsupervisedConfig,
) -> PCARepresentation:
    """Fit preprocessing and full PCA using feature-only development data."""

    validate_unsupervised_inputs(features, config)
    preprocessor = build_preprocessor().fit(features)
    transformed = np.asarray(preprocessor.transform(features), dtype=float)
    if not np.isfinite(transformed).all():
        raise UnsupervisedValidationError(
            "Transformed features contain non-finite values"
        )

    pca = PCA(svd_solver="full").fit(transformed)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    threshold_position = int(
        np.searchsorted(cumulative, config.retained_variance, side="left") + 1
    )
    retained_components = min(
        len(cumulative),
        max(2, threshold_position),
    )
    retained = pca.transform(transformed)[:, :retained_components]
    feature_names = tuple(
        str(name) for name in preprocessor.get_feature_names_out()
    )
    component_numbers = np.arange(1, len(cumulative) + 1)
    pca_summary = pd.DataFrame(
        {
            "component": component_numbers,
            "explained_variance": pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
            "retained": component_numbers <= retained_components,
        }
    )
    pca_loadings = (
        pd.DataFrame(
            pca.components_,
            columns=feature_names,
            index=component_numbers,
        )
        .rename_axis("component")
        .reset_index()
        .melt(
            id_vars="component",
            var_name="feature",
            value_name="loading",
        )
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


def select_cluster_count(cluster_selection: pd.DataFrame) -> int:
    """Select maximum silhouette, preferring smaller k at 0.001 ties."""

    required = {"k", "silhouette"}
    if not required.issubset(cluster_selection.columns):
        raise UnsupervisedValidationError(
            "Cluster selection requires k and silhouette columns"
        )
    if cluster_selection.empty:
        raise UnsupervisedValidationError("Cluster selection cannot be empty")
    ranked = (
        cluster_selection.assign(
            rounded_silhouette=cluster_selection["silhouette"].round(3)
        )
        .sort_values(
            ["rounded_silhouette", "k"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return int(ranked.loc[0, "k"])


def _canonicalize_kmeans(model: KMeans) -> np.ndarray:
    order = np.asarray(
        sorted(
            range(model.n_clusters),
            key=lambda cluster: tuple(model.cluster_centers_[cluster]),
        ),
        dtype=int,
    )
    mapping = np.empty(model.n_clusters, dtype=int)
    mapping[order] = np.arange(model.n_clusters)
    assignments = mapping[np.asarray(model.labels_, dtype=int)]
    model.cluster_centers_ = model.cluster_centers_[order]
    model.labels_ = assignments
    return assignments


def fit_kmeans_candidates(
    representation: PCARepresentation,
    config: UnsupervisedConfig,
) -> KMeansSelection:
    """Evaluate declared k values and freeze the silhouette-selected model."""

    rows: list[dict[str, float | int]] = []
    for k in config.k_values:
        candidate = KMeans(
            n_clusters=k,
            random_state=config.seed,
            n_init=50,
        )
        assignments = candidate.fit_predict(representation.retained)
        if np.unique(assignments).size != k:
            raise UnsupervisedValidationError(
                f"K-Means k={k} produced fewer than {k} populated clusters"
            )
        rows.append(
            {
                "k": k,
                "silhouette": float(
                    silhouette_score(representation.retained, assignments)
                ),
                "davies_bouldin": float(
                    davies_bouldin_score(representation.retained, assignments)
                ),
                "calinski_harabasz": float(
                    calinski_harabasz_score(
                        representation.retained,
                        assignments,
                    )
                ),
            }
        )

    cluster_selection = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    selected_k = select_cluster_count(cluster_selection)
    kmeans = KMeans(
        n_clusters=selected_k,
        random_state=config.seed,
        n_init=50,
    ).fit(representation.retained)
    if np.unique(kmeans.labels_).size != selected_k:
        raise UnsupervisedValidationError(
            "Selected K-Means model produced too few populated clusters"
        )
    assignments = _canonicalize_kmeans(kmeans)
    return KMeansSelection(
        kmeans=kmeans,
        selected_k=selected_k,
        assignments=assignments,
        cluster_selection=cluster_selection,
    )


def estimate_cluster_stability(
    representation: PCARepresentation,
    selection: KMeansSelection,
    config: UnsupervisedConfig,
) -> pd.DataFrame:
    """Measure selected-cluster agreement over deterministic subsamples."""

    generator = np.random.default_rng(config.seed)
    sample_size = int(
        np.floor(
            len(representation.retained) * config.stability_sample_fraction
        )
    )
    rows: list[dict[str, float | int]] = []
    for iteration in range(config.stability_iterations):
        indices = np.sort(
            generator.choice(
                len(representation.retained),
                size=sample_size,
                replace=False,
            )
        )
        subsample_model = KMeans(
            n_clusters=selection.selected_k,
            random_state=config.seed + iteration,
            n_init=50,
        )
        subsample_assignments = subsample_model.fit_predict(
            representation.retained[indices]
        )
        if np.unique(subsample_assignments).size != selection.selected_k:
            raise UnsupervisedValidationError(
                "Stability fit produced too few populated clusters"
            )
        rows.append(
            {
                "iteration": iteration,
                "sample_size": sample_size,
                "adjusted_rand_index": float(
                    adjusted_rand_score(
                        selection.assignments[indices],
                        subsample_assignments,
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _canonicalize_assignments(
    values: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    labels = np.unique(assignments)
    centers = {
        int(label): tuple(values[assignments == label].mean(axis=0))
        for label in labels
    }
    order = sorted((int(label) for label in labels), key=centers.__getitem__)
    mapping = {old: new for new, old in enumerate(order)}
    return np.asarray([mapping[int(label)] for label in assignments], dtype=int)


def fit_hierarchical_comparison(
    representation: PCARepresentation,
    selection: KMeansSelection,
) -> HierarchicalComparison:
    """Compare one Ward hierarchy with the frozen K-Means partition."""

    linkage_matrix = linkage(
        representation.retained,
        method="ward",
        optimal_ordering=True,
    )
    raw_assignments = cut_tree(
        linkage_matrix,
        n_clusters=[selection.selected_k],
    ).reshape(-1)
    assignments = _canonicalize_assignments(
        representation.retained,
        raw_assignments,
    )
    if np.unique(assignments).size != selection.selected_k:
        raise UnsupervisedValidationError(
            "Hierarchical fit produced too few populated clusters"
        )

    agreement = float(
        adjusted_rand_score(selection.assignments, assignments)
    )
    contingency = pd.crosstab(
        pd.Series(selection.assignments, name="kmeans_cluster"),
        pd.Series(assignments, name="hierarchical_cluster"),
        dropna=False,
    )
    rows = [
        {
            "selected_k": selection.selected_k,
            "adjusted_rand_index": agreement,
            "kmeans_cluster": int(kmeans_cluster),
            "hierarchical_cluster": int(hierarchical_cluster),
            "count": int(contingency.loc[kmeans_cluster, hierarchical_cluster]),
        }
        for kmeans_cluster in contingency.index
        for hierarchical_cluster in contingency.columns
    ]
    return HierarchicalComparison(
        linkage_matrix=np.asarray(linkage_matrix, dtype=float),
        assignments=assignments,
        metrics=pd.DataFrame(rows),
    )


def fit_unsupervised_profiles(
    development_features: pd.DataFrame,
    config: UnsupervisedConfig,
) -> FittedProfiles:
    """Fit the complete unsupervised track without accepting target values."""

    representation = fit_pca_representation(development_features, config)
    selection = fit_kmeans_candidates(representation, config)
    stability = estimate_cluster_stability(representation, selection, config)
    hierarchical = fit_hierarchical_comparison(representation, selection)

    cluster_selection = selection.cluster_selection.copy()
    cluster_selection["selected"] = cluster_selection["k"].eq(
        selection.selected_k
    )
    cluster_selection["stability_ari_mean"] = np.nan
    cluster_selection["stability_ari_std"] = np.nan
    selected_mask = cluster_selection["selected"]
    cluster_selection.loc[selected_mask, "stability_ari_mean"] = float(
        stability["adjusted_rand_index"].mean()
    )
    cluster_selection.loc[selected_mask, "stability_ari_std"] = float(
        stability["adjusted_rand_index"].std(ddof=0)
    )

    return FittedProfiles(
        config=config,
        preprocessor=representation.preprocessor,
        pca=representation.pca,
        kmeans=selection.kmeans,
        selected_k=selection.selected_k,
        retained_components=representation.retained_components,
        development_embedding=representation.retained,
        development_assignments=selection.assignments,
        hierarchical_assignments=hierarchical.assignments,
        linkage_matrix=hierarchical.linkage_matrix,
        cluster_selection=cluster_selection,
        stability=stability,
        pca_summary=representation.pca_summary,
        pca_loadings=representation.pca_loadings,
        hierarchical_comparison=hierarchical.metrics,
    )


def _validate_feature_schema(features: pd.DataFrame) -> None:
    if tuple(features.columns) != tuple(FEATURE_COLUMNS):
        raise UnsupervisedValidationError(
            "Every cohort must use the canonical 13-feature schema and order"
        )


def _validate_cohort_join(cohort: CohortData) -> None:
    lengths = {
        len(cohort.features),
        len(cohort.target),
        len(cohort.raw_target),
    }
    if len(lengths) != 1:
        raise UnsupervisedValidationError(
            "Assignments must join one-to-one with patients"
        )
    indexes = (
        cohort.features.index,
        cohort.target.index,
        cohort.raw_target.index,
    )
    if not all(index.is_unique for index in indexes) or not all(
        index.equals(indexes[0]) for index in indexes[1:]
    ):
        raise UnsupervisedValidationError(
            "Assignments must join one-to-one with patients"
        )
    _validate_feature_schema(cohort.features)


def _transform_external(
    fitted: FittedProfiles,
    features: pd.DataFrame,
) -> np.ndarray:
    _validate_feature_schema(features)
    transformed = np.asarray(fitted.preprocessor.transform(features), dtype=float)
    if not np.isfinite(transformed).all():
        raise UnsupervisedValidationError(
            "Transformed features contain non-finite values"
        )
    return fitted.pca.transform(transformed)[:, : fitted.retained_components]


def _assignment_frame(
    fitted: FittedProfiles,
    cohort: CohortData,
    *,
    development: bool,
) -> pd.DataFrame:
    _validate_cohort_join(cohort)
    if development:
        embedding = fitted.development_embedding
        assignments = fitted.development_assignments
        hierarchical = pd.array(
            fitted.hierarchical_assignments,
            dtype="Int64",
        )
    else:
        embedding = _transform_external(fitted, cohort.features)
        assignments = np.asarray(fitted.kmeans.predict(embedding), dtype=int)
        hierarchical = pd.array([pd.NA] * len(cohort.features), dtype="Int64")
    if len(assignments) != len(cohort.features):
        raise UnsupervisedValidationError(
            "Assignments must join one-to-one with patients"
        )
    centroid_distance = np.linalg.norm(
        embedding - fitted.kmeans.cluster_centers_[assignments],
        axis=1,
    )
    return pd.DataFrame(
        {
            "cohort": cohort.cohort.value,
            "patient_id": np.arange(len(cohort.features), dtype=int),
            "kmeans_cluster": assignments,
            "hierarchical_cluster": hierarchical,
            "pc1": embedding[:, 0],
            "pc2": embedding[:, 1],
            "centroid_distance": centroid_distance,
            "disease_present": cohort.target.to_numpy(dtype=int),
        }
    )


def _cluster_profiles(
    fitted: FittedProfiles,
    development: CohortData,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in range(fitted.selected_k):
        mask = fitted.development_assignments == cluster
        size = int(mask.sum())
        status = (
            "available"
            if size >= fitted.config.min_cluster_size
            else "underpowered"
        )
        cluster_features = development.features.loc[mask]
        for feature in NUMERICAL_COLUMNS:
            values = pd.to_numeric(cluster_features[feature]).dropna()
            statistics = {
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
            }
            for statistic, value in statistics.items():
                rows.append(
                    {
                        "cohort": development.cohort.value,
                        "cluster": cluster,
                        "n": size,
                        "status": status,
                        "feature": feature,
                        "statistic": statistic,
                        "level": "",
                        "value": float(value),
                    }
                )
        for feature in CATEGORICAL_COLUMNS:
            levels = (
                pd.to_numeric(cluster_features[feature])
                .astype("Float64")
                .astype("string")
                .fillna("missing")
            )
            proportions = levels.value_counts(normalize=True).sort_index()
            for level, value in proportions.items():
                rows.append(
                    {
                        "cohort": development.cohort.value,
                        "cluster": cluster,
                        "n": size,
                        "status": status,
                        "feature": feature,
                        "statistic": "proportion",
                        "level": str(level),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _external_transfer(
    fitted: FittedProfiles,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    development = assignments.loc[assignments["cohort"].eq("cleveland")]
    baseline = np.asarray(
        [
            development["kmeans_cluster"].eq(cluster).mean()
            for cluster in range(fitted.selected_k)
        ],
        dtype=float,
    )
    rows: list[dict[str, object]] = []
    for cohort_name in assignments["cohort"].drop_duplicates():
        cohort_rows = assignments.loc[assignments["cohort"].eq(cohort_name)]
        proportions = np.asarray(
            [
                cohort_rows["kmeans_cluster"].eq(cluster).mean()
                for cluster in range(fitted.selected_k)
            ],
            dtype=float,
        )
        total_variation = float(0.5 * np.abs(proportions - baseline).sum())
        for cluster in range(fitted.selected_k):
            cluster_rows = cohort_rows.loc[
                cohort_rows["kmeans_cluster"].eq(cluster)
            ]
            size = len(cluster_rows)
            status = (
                "available"
                if size >= fitted.config.min_cluster_size
                else "underpowered"
            )
            rows.append(
                {
                    "cohort": str(cohort_name),
                    "cluster": cluster,
                    "n": size,
                    "proportion": float(proportions[cluster]),
                    "disease_prevalence": (
                        float(cluster_rows["disease_present"].mean())
                        if size
                        else float("nan")
                    ),
                    "median_centroid_distance": (
                        float(cluster_rows["centroid_distance"].median())
                        if size
                        else float("nan")
                    ),
                    "cluster_proportion_total_variation": total_variation,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def describe_unsupervised_profiles(
    fitted: FittedProfiles,
    development: CohortData,
    external: tuple[CohortData, ...],
    config: UnsupervisedConfig,
) -> UnsupervisedStudy:
    """Join frozen assignments to labels for descriptive evidence only."""

    if config != fitted.config:
        raise UnsupervisedValidationError(
            "Description config must match the fitted unsupervised config"
        )
    _validate_cohort_join(development)
    assignment_frames = [
        _assignment_frame(fitted, development, development=True)
    ]
    assignment_frames.extend(
        _assignment_frame(fitted, cohort, development=False)
        for cohort in external
    )
    patient_assignments = pd.concat(assignment_frames, ignore_index=True)
    if patient_assignments.duplicated(["cohort", "patient_id"]).any():
        raise UnsupervisedValidationError(
            "Assignments must join one-to-one with patients"
        )

    selected = fitted.cluster_selection.loc[
        fitted.cluster_selection["selected"]
    ].iloc[0]
    hierarchical_ari = float(
        fitted.hierarchical_comparison["adjusted_rand_index"].iloc[0]
    )
    development_ami = float(
        adjusted_mutual_info_score(
            development.target.to_numpy(dtype=int),
            fitted.development_assignments,
        )
    )
    retained_variance = float(
        fitted.pca_summary.loc[
            fitted.retained_components - 1,
            "cumulative_explained_variance",
        ]
    )
    interpretation: dict[str, object] = {
        "selected_k": fitted.selected_k,
        "retained_components": fitted.retained_components,
        "retained_variance": retained_variance,
        "silhouette": float(selected["silhouette"]),
        "davies_bouldin": float(selected["davies_bouldin"]),
        "calinski_harabasz": float(selected["calinski_harabasz"]),
        "stability_ari_mean": float(selected["stability_ari_mean"]),
        "stability_ari_std": float(selected["stability_ari_std"]),
        "hierarchical_ari": hierarchical_ari,
        "development_target_ami": development_ami,
        "target_used_for_fit": False,
        "external_used_for_selection": False,
        "distance_limitation": (
            "Euclidean distance over scaled numeric and one-hot categorical "
            "features is an exploratory mixed-data approximation."
        ),
    }
    return UnsupervisedStudy(
        selected_k=fitted.selected_k,
        retained_components=fitted.retained_components,
        cluster_selection=fitted.cluster_selection.copy(),
        pca_summary=fitted.pca_summary.copy(),
        pca_loadings=fitted.pca_loadings.copy(),
        hierarchical_comparison=fitted.hierarchical_comparison.copy(),
        cluster_profiles=_cluster_profiles(fitted, development),
        patient_assignments=patient_assignments,
        external_transfer=_external_transfer(fitted, patient_assignments),
        interpretation=interpretation,
    )
