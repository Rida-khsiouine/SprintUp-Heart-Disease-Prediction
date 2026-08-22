from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from heart_disease.data import Cohort
from heart_disease.unsupervised import (
    UnsupervisedConfig,
    UnsupervisedValidationError,
    describe_unsupervised_profiles,
    estimate_cluster_stability,
    fit_hierarchical_comparison,
    fit_kmeans_candidates,
    fit_pca_representation,
    fit_unsupervised_profiles,
    select_cluster_count,
    validate_unsupervised_inputs,
)
from heart_disease.validation import FEATURE_COLUMNS, CohortData


def _features(rows: int, *, age_offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
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
        }
    ).loc[:, FEATURE_COLUMNS]


def _cohort(
    cohort: Cohort, rows: int, *, age_offset: float = 0.0
) -> CohortData:
    target = pd.Series(
        np.arange(rows) % 2,
        name="disease_present",
        dtype="int8",
    )
    return CohortData(
        cohort=cohort,
        features=_features(rows, age_offset=age_offset),
        target=target,
        raw_target=target.rename("num"),
    )


def test_unsupervised_input_rejects_target_column() -> None:
    features = _features(30).assign(disease_present=0)

    with pytest.raises(
        UnsupervisedValidationError,
        match="exact 13-feature schema",
    ):
        validate_unsupervised_inputs(features, UnsupervisedConfig())


def test_unsupervised_input_rejects_reordered_schema() -> None:
    reordered = _features(30).loc[:, list(reversed(FEATURE_COLUMNS))]

    with pytest.raises(UnsupervisedValidationError, match="canonical order"):
        validate_unsupervised_inputs(reordered, UnsupervisedConfig())


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
def test_invalid_unsupervised_config_is_rejected(
    config: UnsupervisedConfig,
    message: str,
) -> None:
    with pytest.raises(UnsupervisedValidationError, match=message):
        validate_unsupervised_inputs(_features(30), config)


def test_unsupervised_input_requires_enough_patients_for_subsamples() -> None:
    config = UnsupervisedConfig(
        k_values=(2, 3, 4, 5, 6),
        stability_sample_fraction=0.2,
    )

    with pytest.raises(UnsupervisedValidationError, match="subsample"):
        validate_unsupervised_inputs(_features(30), config)


def test_component_count_reaches_declared_variance() -> None:
    result = fit_pca_representation(
        _features(80),
        UnsupervisedConfig(retained_variance=0.90),
    )

    assert result.retained_components >= 2
    assert result.pca_summary.loc[
        result.retained_components - 1,
        "cumulative_explained_variance",
    ] >= 0.90


def test_cluster_selection_prefers_smaller_k_on_declared_tie() -> None:
    table = pd.DataFrame(
        {
            "k": [2, 3, 4],
            "silhouette": [0.40149, 0.40148, 0.39],
        }
    )

    assert select_cluster_count(table) == 2


def test_same_seed_produces_same_pca_and_cluster_assignments() -> None:
    config = UnsupervisedConfig(
        seed=17,
        k_values=(2, 3),
        stability_iterations=2,
    )
    first_representation = fit_pca_representation(_features(90), config)
    second_representation = fit_pca_representation(_features(90), config)
    first = fit_kmeans_candidates(first_representation, config)
    second = fit_kmeans_candidates(second_representation, config)

    np.testing.assert_array_equal(first.assignments, second.assignments)
    pd.testing.assert_frame_equal(
        first.cluster_selection,
        second.cluster_selection,
    )


def test_collapsed_candidate_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    config = UnsupervisedConfig(k_values=(2,), stability_iterations=2)
    representation = fit_pca_representation(_features(60), config)

    monkeypatch.setattr(
        KMeans,
        "fit_predict",
        lambda self, values: np.zeros(len(values), dtype=int),
    )

    with pytest.raises(UnsupervisedValidationError, match="populated clusters"):
        fit_kmeans_candidates(representation, config)


def test_stability_is_reproducible_with_seed() -> None:
    config = UnsupervisedConfig(
        seed=23,
        k_values=(2, 3),
        stability_iterations=8,
    )
    representation = fit_pca_representation(_features(100), config)
    selection = fit_kmeans_candidates(representation, config)

    first = estimate_cluster_stability(representation, selection, config)
    second = estimate_cluster_stability(representation, selection, config)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [
        "iteration",
        "sample_size",
        "adjusted_rand_index",
    ]
    assert len(first) == 8


def test_hierarchical_comparison_uses_selected_k() -> None:
    config = UnsupervisedConfig(
        k_values=(2, 3),
        stability_iterations=2,
    )
    representation = fit_pca_representation(_features(80), config)
    selection = fit_kmeans_candidates(representation, config)

    comparison = fit_hierarchical_comparison(representation, selection)

    assert np.unique(comparison.assignments).size == selection.selected_k
    assert comparison.linkage_matrix.shape == (
        len(representation.retained) - 1,
        4,
    )
    agreement = float(comparison.metrics.loc[0, "adjusted_rand_index"])
    assert -1 <= agreement <= 1
    assert int(comparison.metrics["count"].sum()) == len(
        representation.retained
    )


def test_fit_is_independent_of_target_values() -> None:
    features = _features(100)
    config = UnsupervisedConfig(
        k_values=(2, 3),
        stability_iterations=3,
    )

    first = fit_unsupervised_profiles(features, config)
    second = fit_unsupervised_profiles(features.copy(), config)

    np.testing.assert_array_equal(
        first.development_assignments,
        second.development_assignments,
    )
    np.testing.assert_array_equal(
        first.kmeans.cluster_centers_,
        second.kmeans.cluster_centers_,
    )


def test_external_data_cannot_change_components_or_centroids() -> None:
    config = UnsupervisedConfig(
        k_values=(2, 3),
        stability_iterations=3,
    )
    development = _cohort(Cohort.CLEVELAND, rows=100)
    external = _cohort(Cohort.HUNGARY, rows=40, age_offset=10_000)
    fitted = fit_unsupervised_profiles(development.features, config)
    components = fitted.pca.components_.copy()
    centroids = fitted.kmeans.cluster_centers_.copy()

    describe_unsupervised_profiles(
        fitted,
        development,
        (external,),
        config,
    )

    np.testing.assert_array_equal(fitted.pca.components_, components)
    np.testing.assert_array_equal(fitted.kmeans.cluster_centers_, centroids)


def test_small_external_cluster_is_marked_underpowered() -> None:
    config = UnsupervisedConfig(
        k_values=(2,),
        stability_iterations=2,
        min_cluster_size=25,
    )
    development = _cohort(Cohort.CLEVELAND, rows=80)
    external = _cohort(Cohort.HUNGARY, rows=12)
    fitted = fit_unsupervised_profiles(development.features, config)

    study = describe_unsupervised_profiles(
        fitted,
        development,
        (external,),
        config,
    )

    hungary = study.external_transfer.loc[
        study.external_transfer["cohort"].eq("hungary")
    ]
    assert not hungary.empty
    assert set(hungary["status"]) == {"underpowered"}


def test_assignment_join_must_be_one_to_one() -> None:
    config = UnsupervisedConfig(k_values=(2,), stability_iterations=2)
    development = _cohort(Cohort.CLEVELAND, rows=80)
    malformed = dataclasses.replace(
        development,
        target=development.target.iloc[:-1],
    )
    fitted = fit_unsupervised_profiles(development.features, config)

    with pytest.raises(UnsupervisedValidationError, match="one-to-one"):
        describe_unsupervised_profiles(fitted, malformed, (), config)


def test_assignment_join_rejects_misaligned_patient_indices() -> None:
    config = UnsupervisedConfig(k_values=(2,), stability_iterations=2)
    development = _cohort(Cohort.CLEVELAND, rows=80)
    malformed = dataclasses.replace(
        development,
        target=development.target.set_axis(np.arange(1, 81)),
    )
    fitted = fit_unsupervised_profiles(development.features, config)

    with pytest.raises(UnsupervisedValidationError, match="one-to-one"):
        describe_unsupervised_profiles(fitted, malformed, (), config)


def test_profile_outputs_cover_declared_cohorts_and_statistics() -> None:
    config = UnsupervisedConfig(k_values=(2,), stability_iterations=2)
    development = _cohort(Cohort.CLEVELAND, rows=80)
    external = (
        _cohort(Cohort.HUNGARY, rows=30),
        _cohort(Cohort.SWITZERLAND, rows=30),
        _cohort(Cohort.VA, rows=30),
    )
    fitted = fit_unsupervised_profiles(development.features, config)

    study = describe_unsupervised_profiles(
        fitted,
        development,
        external,
        config,
    )

    assert set(study.patient_assignments["cohort"]) == {
        "cleveland",
        "hungary",
        "switzerland",
        "va",
    }
    assert set(study.external_transfer["cohort"]) == {
        "cleveland",
        "hungary",
        "switzerland",
        "va",
    }
    assert {"median", "q1", "q3", "standardized_median", "proportion"} <= set(
        study.cluster_profiles["statistic"]
    )
    assert study.interpretation["target_used_for_fit"] is False
    assert study.interpretation["external_used_for_selection"] is False
