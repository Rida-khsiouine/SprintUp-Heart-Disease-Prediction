from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import heart_disease.evaluation as evaluation
from heart_disease.data import Cohort
from heart_disease.external import (
    Thresholds,
    compare_cohorts,
    evaluate_external,
    evaluate_subgroups,
)
from heart_disease.models import ExperimentConfig, build_pipeline
from heart_disease.validation import CohortData


def _features(rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": np.arange(rows, dtype=float) + 35,
            "sex": np.arange(rows) % 2,
            "cp": np.arange(rows) % 4 + 1,
            "trestbps": np.arange(rows, dtype=float) + 105,
            "chol": np.arange(rows, dtype=float) + 175,
            "fbs": np.arange(rows) % 2,
            "restecg": np.arange(rows) % 3,
            "thalach": 185 - np.arange(rows, dtype=float),
            "exang": np.arange(rows) % 2,
            "oldpeak": np.arange(rows, dtype=float) / 10,
            "slope": np.arange(rows) % 3 + 1,
            "ca": np.arange(rows) % 4,
            "thal": np.resize([3, 6, 7], rows),
        }
    )


def _cohort(
    cohort: Cohort = Cohort.HUNGARY,
    *,
    single_class: bool = False,
) -> CohortData:
    features = _features()
    target = pd.Series(
        np.ones(len(features), dtype=int)
        if single_class
        else np.arange(len(features)) % 2,
        name="disease_present",
    )
    return CohortData(cohort, features, target, target.rename("num"))


def _fitted_pipeline() -> object:
    features = _features()
    target = np.arange(len(features)) % 2
    return build_pipeline("logistic", seed=42).fit(features, target)


def test_external_labels_cannot_reach_selection_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_: object, **__: object) -> None:
        raise AssertionError("external labels reached model selection")

    monkeypatch.setattr(evaluation, "select_candidate", fail_if_called)

    result = evaluate_external(
        _fitted_pipeline(),
        (_cohort(),),
        Thresholds(default=0.5, screening=0.4),
        ExperimentConfig(bootstrap_iterations=20),
    )

    assert set(result["cohort"]) == {"hungary"}


def test_bootstrap_is_reproducible_with_seed() -> None:
    arguments = (
        _fitted_pipeline(),
        (_cohort(),),
        Thresholds(default=0.5, screening=0.4),
        ExperimentConfig(seed=17, bootstrap_iterations=40),
    )

    first = evaluate_external(*arguments)
    second = evaluate_external(*arguments)

    assert_frame_equal(first, second)


def test_single_class_auc_is_reported_as_unavailable() -> None:
    result = evaluate_external(
        _fitted_pipeline(),
        (_cohort(single_class=True),),
        Thresholds(default=0.5, screening=0.4),
        ExperimentConfig(bootstrap_iterations=20),
    )

    assert result["roc_auc"].isna().all()
    assert result["roc_auc_status"].str.contains("unavailable.*one class").all()


def test_small_subgroup_is_marked_underpowered() -> None:
    features = _features(30)
    features.loc[:4, "sex"] = 0
    features.loc[5:, "sex"] = 1
    truth = pd.Series(np.arange(30) % 2)
    probability = np.linspace(0.1, 0.9, 30)

    subgroups = evaluate_subgroups(
        truth,
        probability,
        features,
        Thresholds(default=0.5, screening=0.4),
        min_size=25,
    )

    female = subgroups[
        (subgroups["attribute"] == "sex") & (subgroups["group"] == "female")
    ]
    assert set(female["status"]) == {"underpowered"}
    assert set(female["n"]) == {5}
    assert female["roc_auc"].isna().all()


def test_shift_report_includes_missingness_and_prevalence() -> None:
    development = _cohort(Cohort.CLEVELAND)
    external = _cohort(Cohort.SWITZERLAND)
    shifted = external.features.copy()
    shifted.loc[:9, "chol"] = np.nan
    external = CohortData(
        external.cohort,
        shifted,
        external.target,
        external.raw_target,
    )

    report = compare_cohorts(development, external)

    assert {"missingness", "target_prevalence"} <= set(report["measure"])
    cholesterol = report[
        (report["measure"] == "missingness") & (report["feature"] == "chol")
    ].iloc[0]
    assert cholesterol["external_value"] == pytest.approx(10 / 30)
    assert cholesterol["shift_value"] == pytest.approx(10 / 30)


def test_shift_treats_integer_and_float_category_encodings_as_equal() -> None:
    development = _cohort(Cohort.CLEVELAND)
    external = _cohort(Cohort.HUNGARY)
    float_features = development.features.copy()
    float_features["sex"] = float_features["sex"].astype(float)
    development = CohortData(
        development.cohort,
        float_features,
        development.target,
        development.raw_target,
    )

    report = compare_cohorts(development, external)

    sex_shift = report[
        (report["measure"] == "total_variation_distance")
        & (report["feature"] == "sex")
    ].iloc[0]
    assert sex_shift["shift_value"] == pytest.approx(0.0)
