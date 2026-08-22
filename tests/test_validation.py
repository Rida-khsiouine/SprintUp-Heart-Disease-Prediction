from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import heart_disease.validation as validation
from heart_disease.data import Cohort
from heart_disease.validation import DataValidationError, load_cohort


def _raw_frame(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
        "num": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row], columns=validation.RAW_COLUMNS)


def _load(frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> validation.CohortData:
    monkeypatch.setattr(validation, "read_raw_cohort", lambda *_: frame.copy())
    return load_cohort(Cohort.CLEVELAND, Path("unused"))


def test_target_maps_zero_to_absent_and_one_through_four_to_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.concat([_raw_frame(num=value) for value in range(5)], ignore_index=True)

    cohort = _load(frame, monkeypatch)

    assert cohort.raw_target.tolist() == [0, 1, 2, 3, 4]
    assert cohort.target.tolist() == [0, 1, 1, 1, 1]
    assert cohort.target.name == "disease_present"


def test_oldpeak_decimal_values_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = _load(_raw_frame(oldpeak=1.7), monkeypatch)

    assert cohort.features.loc[0, "oldpeak"] == pytest.approx(1.7)
    assert pd.api.types.is_float_dtype(cohort.features["oldpeak"])


def test_missing_features_do_not_drop_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.concat(
        [_raw_frame(chol=np.nan), _raw_frame(age=54, num=2)], ignore_index=True
    )

    cohort = _load(frame, monkeypatch)

    assert len(cohort.features) == len(frame)
    assert pd.isna(cohort.features.loc[0, "chol"])
    assert cohort.missing_counts["chol"] == 1


def test_missing_target_raises_data_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(DataValidationError, match="num.*missing"):
        _load(_raw_frame(num=np.nan), monkeypatch)


def test_invalid_categorical_value_reports_column_and_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(DataValidationError, match=r"cp.*9"):
        _load(_raw_frame(cp=9), monkeypatch)


def test_validation_records_duplicates_without_deleting_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.concat([_raw_frame(), _raw_frame()], ignore_index=True)

    cohort = _load(frame, monkeypatch)

    assert len(cohort.features) == 2
    assert cohort.duplicate_count == 1
