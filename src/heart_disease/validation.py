"""Schema validation and target construction for UCI heart-disease cohorts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from heart_disease.data import Cohort, read_raw_cohort

RAW_COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
)
FEATURE_COLUMNS = RAW_COLUMNS[:-1]

CATEGORICAL_DOMAINS: Mapping[str, frozenset[int]] = {
    "sex": frozenset({0, 1}),
    "cp": frozenset({1, 2, 3, 4}),
    "fbs": frozenset({0, 1}),
    "restecg": frozenset({0, 1, 2}),
    "exang": frozenset({0, 1}),
    "slope": frozenset({1, 2, 3}),
    "ca": frozenset({0, 1, 2, 3}),
    "thal": frozenset({3, 6, 7}),
}


class DataValidationError(ValueError):
    """Raised when a cohort violates the documented UCI schema."""


@dataclass(frozen=True)
class CohortData:
    """Validated features and targets plus non-destructive quality evidence."""

    cohort: Cohort
    features: pd.DataFrame
    target: pd.Series
    raw_target: pd.Series
    duplicate_count: int = 0
    missing_counts: Mapping[str, int] = field(default_factory=dict)


def _as_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    original = frame[column]
    numeric = pd.to_numeric(original, errors="coerce")
    invalid = original.notna() & numeric.isna()
    if invalid.any():
        values = sorted({str(value) for value in original[invalid]})
        raise DataValidationError(
            f"Column '{column}' contains non-numeric values: {values}"
        )

    present = numeric.dropna().astype(float)
    if not np.isfinite(present).all():
        values = sorted({str(value) for value in present[~np.isfinite(present)]})
        raise DataValidationError(
            f"Column '{column}' contains non-finite values: {values}"
        )
    return numeric


def validate_raw_frame(frame: pd.DataFrame, cohort: Cohort) -> None:
    """Validate shape, numerical values, target, and categorical domains."""

    missing_columns = [column for column in RAW_COLUMNS if column not in frame]
    unexpected_columns = [column for column in frame if column not in RAW_COLUMNS]
    if missing_columns or unexpected_columns:
        raise DataValidationError(
            f"{cohort.value} schema mismatch; missing={missing_columns}, "
            f"unexpected={unexpected_columns}"
        )
    if frame.empty:
        raise DataValidationError(f"{cohort.value} cohort contains no patient rows")

    numeric_columns = {
        column: _as_numeric(frame, column) for column in RAW_COLUMNS
    }
    raw_target = numeric_columns["num"]
    if raw_target.isna().any():
        rows = raw_target.index[raw_target.isna()].tolist()
        raise DataValidationError(f"Column 'num' has missing targets at rows {rows}")

    invalid_targets = sorted(set(raw_target) - {0, 1, 2, 3, 4})
    if invalid_targets:
        raise DataValidationError(
            f"Column 'num' contains invalid target values: {invalid_targets}"
        )

    for column, domain in CATEGORICAL_DOMAINS.items():
        values = numeric_columns[column].dropna()
        invalid_values = sorted(set(values) - domain)
        if invalid_values:
            raise DataValidationError(
                f"Column '{column}' contains invalid categorical values: "
                f"{invalid_values}"
            )


def load_cohort(cohort: Cohort, data_dir: Path) -> CohortData:
    """Load and validate one cohort without deleting incomplete or duplicate rows."""

    raw = read_raw_cohort(cohort, data_dir)
    validate_raw_frame(raw, cohort)

    numeric = raw.loc[:, RAW_COLUMNS].apply(pd.to_numeric, errors="raise")
    features = numeric.loc[:, FEATURE_COLUMNS].copy()
    features["oldpeak"] = features["oldpeak"].astype(float)
    raw_target = numeric["num"].astype("int8").rename("num")
    target = raw_target.gt(0).astype("int8").rename("disease_present")
    missing_counts = {
        column: int(count)
        for column, count in features.isna().sum().items()
    }

    return CohortData(
        cohort=cohort,
        features=features,
        target=target,
        raw_target=raw_target,
        duplicate_count=int(raw.duplicated().sum()),
        missing_counts=missing_counts,
    )
