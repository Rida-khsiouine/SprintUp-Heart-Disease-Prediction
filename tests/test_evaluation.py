from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold

import heart_disease.evaluation as evaluation
from heart_disease.data import Cohort
from heart_disease.evaluation import (
    CandidateResult,
    FoldPrediction,
    average_patient_predictions,
    choose_screening_threshold,
    classification_metrics,
    run_nested_cv,
    select_candidate,
)
from heart_disease.models import ExperimentConfig
from heart_disease.validation import CohortData


def _features(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": np.arange(rows, dtype=float) + 40,
            "sex": np.arange(rows) % 2,
            "cp": np.arange(rows) % 4 + 1,
            "trestbps": np.arange(rows, dtype=float) + 110,
            "chol": np.arange(rows, dtype=float) + 180,
            "fbs": np.arange(rows) % 2,
            "restecg": np.arange(rows) % 3,
            "thalach": 180 - np.arange(rows, dtype=float),
            "exang": np.arange(rows) % 2,
            "oldpeak": np.arange(rows, dtype=float) / 10,
            "slope": np.arange(rows) % 3 + 1,
            "ca": np.arange(rows) % 4,
            "thal": np.resize([3, 6, 7], rows),
        },
        index=np.arange(rows),
    )


def _candidate(
    model_name: evaluation.ModelName,
    auc_values: list[float],
) -> CandidateResult:
    metrics = pd.DataFrame(
        {
            "roc_auc": auc_values,
            "brier_score": [0.15] * len(auc_values),
            "balanced_accuracy": [0.75] * len(auc_values),
        }
    )
    predictions = tuple(
        FoldPrediction(
            patient_id=index,
            repeat=0,
            fold=0,
            truth=truth,
            probability=probability,
        )
        for index, (truth, probability) in enumerate(
            zip([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], strict=True)
        )
    )
    return CandidateResult(model_name, metrics, predictions, ({},))


def test_outer_test_patients_never_enter_inner_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = pd.Series([0, 1] * 10, name="disease_present")
    data = CohortData(Cohort.CLEVELAND, _features(), target, target.copy())
    seen_search_rows: list[set[int]] = []

    class RecordingSearch:
        def __init__(self, estimator: object, **_: object) -> None:
            self.best_estimator_ = estimator
            self.best_params_: dict[str, object] = {}

        def fit(self, features: pd.DataFrame, truth: pd.Series) -> RecordingSearch:
            seen_search_rows.append(set(features.index))
            self.best_estimator_.fit(features, truth)
            return self

    monkeypatch.setattr(evaluation, "CANDIDATE_MODELS", ("logistic",))
    monkeypatch.setattr(evaluation, "GridSearchCV", RecordingSearch)
    config = ExperimentConfig(outer_splits=2, outer_repeats=1, inner_splits=2)

    run_nested_cv(data, config)

    splitter = RepeatedStratifiedKFold(
        n_splits=2, n_repeats=1, random_state=config.seed
    )
    expected_training: list[set[int]] = []
    expected_tests: list[set[int]] = []
    for train_positions, test_positions in splitter.split(data.features, target):
        expected_training.append(set(data.features.index[train_positions]))
        expected_tests.append(set(data.features.index[test_positions]))
    assert seen_search_rows == expected_training
    assert all(
        searched.isdisjoint(held_out)
        for searched, held_out in zip(seen_search_rows, expected_tests, strict=True)
    )


def test_one_standard_error_rule_prefers_logistic() -> None:
    results = {
        "logistic": _candidate("logistic", [0.800, 0.801, 0.799, 0.800]),
        "random_forest": _candidate(
            "random_forest", [0.790, 0.820, 0.790, 0.820]
        ),
    }

    selection = select_candidate(results)

    assert selection.model_name == "logistic"
    assert "one-standard-error" in selection.reason


def test_screening_threshold_meets_recall_floor() -> None:
    truth = np.array([1, 1, 0, 0])
    probability = np.array([0.9, 0.6, 0.7, 0.2])

    threshold = choose_screening_threshold(truth, probability, recall_floor=1.0)
    metrics = classification_metrics(truth, probability, threshold=threshold)

    assert threshold == pytest.approx(0.6)
    assert metrics["sensitivity"] >= 1.0


def test_metric_bundle_matches_known_confusion_matrix() -> None:
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.7, 0.4]),
        threshold=0.5,
    )

    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.5)


def test_patient_predictions_are_averaged_across_repeats() -> None:
    predictions = (
        FoldPrediction(0, 0, 0, 0, 0.2),
        FoldPrediction(1, 0, 0, 1, 0.8),
        FoldPrediction(0, 1, 0, 0, 0.4),
        FoldPrediction(1, 1, 0, 1, 0.6),
    )

    averaged = average_patient_predictions(predictions)

    assert averaged["patient_id"].tolist() == [0, 1]
    assert averaged["truth"].tolist() == [0, 1]
    assert averaged["probability"].tolist() == pytest.approx([0.3, 0.7])


def test_selection_uses_calibration_as_a_tie_breaker() -> None:
    logistic = _candidate("logistic", [0.8, 0.8])
    forest = _candidate("random_forest", [0.8, 0.8])
    forest_metrics = forest.fold_metrics.assign(brier_score=0.10)
    forest = replace(forest, fold_metrics=forest_metrics)

    selection = select_candidate({"logistic": logistic, "random_forest": forest})

    # The one-standard-error simplicity rule has priority over metric tie-breakers.
    assert selection.model_name == "logistic"


def test_selection_uses_supplied_screening_recall_floor() -> None:
    logistic = _candidate("logistic", [0.8, 0.8])

    selection = select_candidate({"logistic": logistic}, recall_floor=0.5)

    assert selection.screening_threshold == pytest.approx(0.9)
