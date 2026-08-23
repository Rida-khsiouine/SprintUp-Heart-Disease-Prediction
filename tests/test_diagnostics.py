from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

import heart_disease.diagnostics as diagnostics
from heart_disease.data import Cohort
from heart_disease.evaluation import CandidateResult, Selection
from heart_disease.models import ExperimentConfig, build_pipeline
from heart_disease.validation import CohortData


def test_diagnostics_module_is_available() -> None:
    assert importlib.util.find_spec("heart_disease.diagnostics") is not None


def _candidate(
    model_name: str,
    test_auc: list[float],
    train_auc: list[float],
    parameters: tuple[dict[str, object], ...],
) -> CandidateResult:
    return CandidateResult(
        model_name=model_name,
        fold_metrics=pd.DataFrame(
            {
                "repeat": [0, 0, 1, 1],
                "fold": [0, 1, 0, 1],
                "roc_auc": test_auc,
                "train_roc_auc": train_auc,
            }
        ),
        predictions=(),
        best_parameters=parameters,
    )


def _metrics() -> dict[str, object]:
    return {
        "profile": "full",
        "development": {
            "default": {
                "roc_auc": 0.83,
                "confidence_intervals": {
                    "roc_auc": {"low": 0.78, "high": 0.88}
                },
            }
        },
        "external": [
            {
                "cohort": "hungary",
                "roc_auc": 0.75,
                "confidence_intervals": {
                    "roc_auc": {"low": 0.70, "high": 0.80}
                },
            }
        ],
    }


def _learning_curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "training_rows": [20, 40],
            "train_roc_auc_mean": [0.95, 0.92],
            "train_roc_auc_std": [0.01, 0.02],
            "validation_roc_auc_mean": [0.78, 0.84],
            "validation_roc_auc_std": [0.04, 0.03],
        }
    )


def test_build_model_diagnostics_summarizes_generalization_and_stability() -> None:
    logistic_parameters = (
        {"model__C": 0.1, "model__class_weight": None},
        {"model__C": 0.1, "model__class_weight": None},
        {"model__C": 0.1, "model__class_weight": None},
        {"model__C": 1.0, "model__class_weight": "balanced"},
    )
    results = {
        "logistic": _candidate(
            "logistic",
            [0.80, 0.82, 0.84, 0.86],
            [0.90, 0.91, 0.92, 0.93],
            logistic_parameters,
        ),
        "random_forest": _candidate(
            "random_forest",
            [0.76, 0.78, 0.80, 0.82],
            [0.99, 0.99, 1.00, 1.00],
            ({}, {}, {}, {}),
        ),
    }

    summary = diagnostics.build_model_diagnostics(
        results=results,
        selection=Selection("logistic", "test", 0.5, 0.4),
        learning_curve=_learning_curve(),
        metrics=_metrics(),
    )

    assert summary["profile"] == "full"
    assert summary["selected_model"] == "logistic"
    assert summary["generalization"]["train_roc_auc_mean"] == 0.915
    assert summary["generalization"]["validation_roc_auc_mean"] == 0.83
    assert summary["generalization"]["roc_auc_gap"] == 0.085
    assert summary["repeat_stability"] == [
        {"repeat": 0, "mean_roc_auc": 0.81},
        {"repeat": 1, "mean_roc_auc": 0.85},
    ]
    assert summary["parameter_stability"][0]["count"] == 3
    assert summary["parameter_stability"][0]["parameters"] == {
        "model__C": 0.1,
        "model__class_weight": None,
    }
    assert summary["cohort_performance"] == [
        {
            "cohort": "cleveland_oof",
            "roc_auc": 0.83,
            "ci_low": 0.78,
            "ci_high": 0.88,
        },
        {
            "cohort": "hungary",
            "roc_auc": 0.75,
            "ci_low": 0.70,
            "ci_high": 0.80,
        },
    ]


def _development_data(rows: int = 80) -> CohortData:
    index = np.arange(rows)
    target = pd.Series((index % 4 >= 2).astype(int), name="disease_present")
    features = pd.DataFrame(
        {
            "age": 40.0 + index + target * 5,
            "sex": index % 2,
            "cp": index % 4 + 1,
            "trestbps": 110.0 + index % 30,
            "chol": 180.0 + index,
            "fbs": index % 2,
            "restecg": index % 3,
            "thalach": 180.0 - index / 2,
            "exang": index % 2,
            "oldpeak": index.astype(float) / 20,
            "slope": index % 3 + 1,
            "ca": index % 4,
            "thal": np.resize([3, 6, 7], rows),
        }
    )
    return CohortData(Cohort.CLEVELAND, features, target, target.copy())


def test_compute_learning_curve_is_deterministic_and_leakage_safe() -> None:
    data = _development_data()
    pipeline = build_pipeline("logistic", seed=42).set_params(model__C=0.1)
    config = ExperimentConfig(outer_splits=2, outer_repeats=1, inner_splits=2)

    first = diagnostics.compute_learning_curve(
        data,
        pipeline,
        config,
        train_fractions=(0.5, 1.0),
    )
    second = diagnostics.compute_learning_curve(
        data,
        pipeline,
        config,
        train_fractions=(0.5, 1.0),
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["training_rows"].tolist() == [20, 40]
    score_columns = [column for column in first if "roc_auc" in column]
    assert np.isfinite(first[score_columns].to_numpy()).all()
    assert ((first[score_columns] >= 0) & (first[score_columns] <= 1)).all().all()


def test_write_model_diagnostics_creates_machine_readable_report_and_graphs(
    tmp_path: Path,
) -> None:
    summary = diagnostics.build_model_diagnostics(
        results={
            "logistic": _candidate(
                "logistic",
                [0.80, 0.82, 0.84, 0.86],
                [0.90, 0.91, 0.92, 0.93],
                (
                    {"model__C": 0.1},
                    {"model__C": 0.1},
                    {"model__C": 1.0},
                    {"model__C": 1.0},
                ),
            )
        },
        selection=Selection("logistic", "test", 0.5, 0.4),
        learning_curve=_learning_curve(),
        metrics=_metrics(),
    )

    paths = diagnostics.write_model_diagnostics(summary, tmp_path)

    persisted = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert persisted == summary
    expected_figures = {
        "cohort-performance.png",
        "hyperparameter-stability.png",
        "learning-curve.png",
        "model-comparison.png",
        "train-validation-gap.png",
    }
    assert expected_figures == {path.name for path in paths["figures"].iterdir()}
    assert all(
        (paths["figures"] / filename).stat().st_size > 0
        for filename in expected_figures
    )
