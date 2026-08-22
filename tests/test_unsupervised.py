from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heart_disease.data import Cohort
from heart_disease.unsupervised import (
    UnsupervisedConfig,
    UnsupervisedValidationError,
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
