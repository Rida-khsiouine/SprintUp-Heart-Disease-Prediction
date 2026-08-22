"""Label-isolated PCA and clustering analysis across UCI cohorts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
