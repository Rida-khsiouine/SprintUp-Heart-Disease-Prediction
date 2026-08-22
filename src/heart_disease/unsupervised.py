"""Label-isolated PCA and clustering analysis across UCI cohorts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from heart_disease.models import build_preprocessor
from heart_disease.validation import FEATURE_COLUMNS


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
