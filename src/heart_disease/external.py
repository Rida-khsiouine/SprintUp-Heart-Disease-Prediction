"""External validation, cohort-shift, and predeclared subgroup analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from heart_disease.evaluation import classification_metrics
from heart_disease.models import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    ExperimentConfig,
)
from heart_disease.validation import FEATURE_COLUMNS, CohortData

INTERVAL_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
)


@dataclass(frozen=True)
class Thresholds:
    default: float
    screening: float


def _stratified_bootstrap_indices(
    truth: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(seed)
    class_positions = [np.flatnonzero(truth == value) for value in np.unique(truth)]
    samples = []
    for _ in range(iterations):
        sampled_classes = [
            generator.choice(positions, size=len(positions), replace=True)
            for positions in class_positions
        ]
        combined = np.concatenate(sampled_classes)
        samples.append(generator.permutation(combined))
    return tuple(samples)


def _interval(values: list[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def evaluate_external(
    fitted_pipeline: Pipeline,
    cohorts: tuple[CohortData, ...],
    thresholds: Thresholds,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Evaluate a frozen model and thresholds on untouched external cohorts."""

    check_is_fitted(fitted_pipeline)
    # Compute every probability before target access. External labels are used only
    # below for frozen evaluation, never for tuning, calibration, or thresholds.
    frozen_predictions = tuple(
        (
            cohort,
            np.asarray(fitted_pipeline.predict_proba(cohort.features)[:, 1]),
        )
        for cohort in cohorts
    )
    rows: list[dict[str, object]] = []
    for cohort_index, (cohort, probability) in enumerate(frozen_predictions):
        truth = cohort.target.to_numpy(dtype=int)
        bootstrap_indices = _stratified_bootstrap_indices(
            truth,
            iterations=config.bootstrap_iterations,
            seed=config.seed + cohort_index,
        )
        both_classes = np.unique(truth).size == 2
        for threshold_name, threshold in (
            ("default", thresholds.default),
            ("screening", thresholds.screening),
        ):
            metrics = classification_metrics(truth, probability, threshold=threshold)
            row: dict[str, object] = {
                "cohort": cohort.cohort.value,
                "n": len(truth),
                "positive_count": int(truth.sum()),
                "prevalence": float(truth.mean()),
                "threshold_name": threshold_name,
                "threshold": threshold,
                **metrics,
                "roc_auc_status": "available"
                if both_classes
                else "unavailable: cohort has only one class",
            }
            bootstrap_metrics = [
                classification_metrics(
                    truth[indices], probability[indices], threshold=threshold
                )
                for indices in bootstrap_indices
            ]
            for metric in INTERVAL_METRICS:
                low, high = _interval(
                    [float(sample[metric]) for sample in bootstrap_metrics]
                )
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            if both_classes and not np.isfinite(row["roc_auc_ci_low"]):
                row["roc_auc_status"] = (
                    "unavailable: no valid two-class bootstrap resamples"
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _shift_row(
    development: CohortData,
    external: CohortData,
    *,
    measure: str,
    feature: str,
    development_value: float,
    external_value: float,
    shift_value: float,
    status: str = "available",
) -> dict[str, object]:
    return {
        "development_cohort": development.cohort.value,
        "external_cohort": external.cohort.value,
        "measure": measure,
        "feature": feature,
        "development_value": development_value,
        "external_value": external_value,
        "shift_value": shift_value,
        "status": status,
    }


def compare_cohorts(
    development: CohortData,
    external: CohortData,
) -> pd.DataFrame:
    """Quantify prevalence, missingness, and univariate distribution shifts."""

    rows = [
        _shift_row(
            development,
            external,
            measure="target_prevalence",
            feature="disease_present",
            development_value=float(development.target.mean()),
            external_value=float(external.target.mean()),
            shift_value=float(external.target.mean() - development.target.mean()),
        )
    ]
    for feature in FEATURE_COLUMNS:
        development_missing = float(development.features[feature].isna().mean())
        external_missing = float(external.features[feature].isna().mean())
        rows.append(
            _shift_row(
                development,
                external,
                measure="missingness",
                feature=feature,
                development_value=development_missing,
                external_value=external_missing,
                shift_value=external_missing - development_missing,
            )
        )

    for feature in NUMERICAL_COLUMNS:
        development_values = development.features[feature].dropna().astype(float)
        external_values = external.features[feature].dropna().astype(float)
        development_mean = float(development_values.mean())
        external_mean = float(external_values.mean())
        pooled_variance = (
            float(development_values.var(ddof=1))
            + float(external_values.var(ddof=1))
        ) / 2
        if np.isfinite(pooled_variance) and pooled_variance > 0:
            shift_value = (external_mean - development_mean) / np.sqrt(
                pooled_variance
            )
            status = "available"
        else:
            shift_value = 0.0 if external_mean == development_mean else float("nan")
            status = "available" if shift_value == 0 else "unavailable: zero variance"
        rows.append(
            _shift_row(
                development,
                external,
                measure="standardized_mean_difference",
                feature=feature,
                development_value=development_mean,
                external_value=external_mean,
                shift_value=float(shift_value),
                status=status,
            )
        )

    for feature in CATEGORICAL_COLUMNS:
        development_distribution = (
            development.features[feature]
            .astype("string")
            .fillna("__MISSING__")
            .value_counts(normalize=True)
        )
        external_distribution = (
            external.features[feature]
            .astype("string")
            .fillna("__MISSING__")
            .value_counts(normalize=True)
        )
        categories = development_distribution.index.union(
            external_distribution.index
        )
        total_variation = 0.5 * sum(
            abs(
                float(development_distribution.get(category, 0.0))
                - float(external_distribution.get(category, 0.0))
            )
            for category in categories
        )
        rows.append(
            _shift_row(
                development,
                external,
                measure="total_variation_distance",
                feature=feature,
                development_value=float("nan"),
                external_value=float("nan"),
                shift_value=float(total_variation),
            )
        )
    return pd.DataFrame(rows)


def _empty_metrics() -> dict[str, float]:
    metric_names = (
        *INTERVAL_METRICS,
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    )
    return {metric: float("nan") for metric in metric_names}


def evaluate_subgroups(
    truth: pd.Series,
    probability: np.ndarray,
    features: pd.DataFrame,
    thresholds: Thresholds,
    min_size: int,
) -> pd.DataFrame:
    """Report predeclared sex and broad age-band slices conservatively."""

    truth_array = np.asarray(truth, dtype=int)
    probability_array = np.asarray(probability, dtype=float)
    if len(features) != len(truth_array) or len(truth_array) != len(probability_array):
        raise ValueError("Features, truths, and probabilities must have equal lengths")

    age_band = pd.cut(
        features["age"],
        bins=[-np.inf, 45, 65, np.inf],
        labels=["under_45", "45_to_64", "65_plus"],
        right=False,
    )
    groups: list[tuple[str, str, np.ndarray]] = [
        ("sex", "female", features["sex"].eq(0).to_numpy()),
        ("sex", "male", features["sex"].eq(1).to_numpy()),
        ("age_band", "under_45", age_band.eq("under_45").to_numpy()),
        ("age_band", "45_to_64", age_band.eq("45_to_64").to_numpy()),
        ("age_band", "65_plus", age_band.eq("65_plus").to_numpy()),
    ]
    if features["sex"].isna().any():
        groups.append(("sex", "missing", features["sex"].isna().to_numpy()))
    if features["age"].isna().any():
        groups.append(("age_band", "missing", features["age"].isna().to_numpy()))

    rows: list[dict[str, object]] = []
    for attribute, group, mask in groups:
        group_truth = truth_array[mask]
        group_probability = probability_array[mask]
        status = "available" if int(mask.sum()) >= min_size else "underpowered"
        for threshold_name, threshold in (
            ("default", thresholds.default),
            ("screening", thresholds.screening),
        ):
            metrics: dict[str, float | int]
            if status == "underpowered":
                metrics = _empty_metrics()
            else:
                metrics = classification_metrics(
                    group_truth, group_probability, threshold=threshold
                )
            both_classes = np.unique(group_truth).size == 2
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "n": int(mask.sum()),
                    "positive_count": int(group_truth.sum()),
                    "prevalence": float(group_truth.mean())
                    if group_truth.size
                    else float("nan"),
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    "status": status,
                    **metrics,
                    "roc_auc_status": "available"
                    if status == "available" and both_classes
                    else (
                        "unavailable: subgroup has only one class"
                        if status == "available"
                        else "unavailable: subgroup underpowered"
                    ),
                }
            )
    return pd.DataFrame(rows)
