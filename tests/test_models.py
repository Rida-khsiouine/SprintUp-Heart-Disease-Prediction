from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from heart_disease.models import (
    NUMERICAL_COLUMNS,
    build_pipeline,
    parameter_grid,
)


def _features(rows: int = 4) -> pd.DataFrame:
    records = []
    for index in range(rows):
        records.append(
            {
                "age": 40.0 + index,
                "sex": index % 2,
                "cp": index % 4 + 1,
                "trestbps": 120.0 + index,
                "chol": 200.0 + index,
                "fbs": index % 2,
                "restecg": index % 3,
                "thalach": 160.0 - index,
                "exang": index % 2,
                "oldpeak": 0.25 + index / 2,
                "slope": index % 3 + 1,
                "ca": index % 4,
                "thal": (3, 6, 7)[index % 3],
            }
        )
    return pd.DataFrame(records)


def test_pipeline_is_unfitted_when_created() -> None:
    pipeline = build_pipeline("logistic", seed=42)

    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline)


def test_scaler_statistics_come_only_from_training_rows() -> None:
    training = _features(4)
    training["age"] = [10.0, 20.0, 30.0, 40.0]
    held_out = _features(1)
    held_out["age"] = 10_000.0
    pipeline = build_pipeline("logistic", seed=42)

    pipeline.fit(training, [0, 0, 1, 1])
    pipeline.predict_proba(held_out)

    scaler = pipeline.named_steps["preprocess"].named_transformers_[
        "numeric"
    ].named_steps["scaler"]
    age_position = NUMERICAL_COLUMNS.index("age")
    assert scaler.mean_[age_position] == pytest.approx(25.0)


def test_pipeline_accepts_missing_and_unseen_categories() -> None:
    training = _features(8)
    pipeline = build_pipeline("logistic", seed=42).fit(
        training, [0, 0, 0, 0, 1, 1, 1, 1]
    )
    patient = _features(1)
    patient.loc[0, "chol"] = np.nan
    patient.loc[0, "cp"] = 99
    patient.loc[0, "thal"] = np.nan

    probability = pipeline.predict_proba(patient)

    assert probability.shape == (1, 2)
    assert np.isfinite(probability).all()


def test_oldpeak_reaches_transformer_as_float() -> None:
    training = _features(4)
    training["oldpeak"] = [0.1, 0.9, 1.7, 2.3]
    pipeline = build_pipeline("logistic", seed=42).fit(
        training, [0, 0, 1, 1]
    )

    scaler = pipeline.named_steps["preprocess"].named_transformers_[
        "numeric"
    ].named_steps["scaler"]
    oldpeak_position = NUMERICAL_COLUMNS.index("oldpeak")

    assert scaler.mean_[oldpeak_position] == pytest.approx(1.25)


def test_same_seed_produces_same_predictions() -> None:
    features = _features(24)
    target = np.array([0, 1] * 12)
    first = build_pipeline("random_forest", seed=17).fit(features, target)
    second = build_pipeline("random_forest", seed=17).fit(features, target)

    np.testing.assert_array_equal(
        first.predict_proba(features), second.predict_proba(features)
    )


def test_parameter_grids_match_declared_search_space() -> None:
    assert parameter_grid("dummy") == {}
    assert parameter_grid("logistic") == {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
    }
    assert parameter_grid("random_forest")["model__max_depth"] == [None, 5]
    assert parameter_grid("gradient_boosting")["model__learning_rate"] == [0.03, 0.1]
