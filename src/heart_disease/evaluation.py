"""Nested evaluation, model selection, and decision-threshold utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from heart_disease.models import (
    ExperimentConfig,
    ModelName,
    build_pipeline,
    parameter_grid,
)
from heart_disease.validation import CohortData

CANDIDATE_MODELS: tuple[ModelName, ...] = (
    "dummy",
    "logistic",
    "random_forest",
    "gradient_boosting",
)
NONLINEAR_MODELS: frozenset[ModelName] = frozenset(
    {"random_forest", "gradient_boosting"}
)
MODEL_COMPLEXITY: dict[ModelName, int] = {
    "logistic": 0,
    "gradient_boosting": 1,
    "random_forest": 2,
    "dummy": 3,
}


@dataclass(frozen=True)
class FoldPrediction:
    patient_id: int
    repeat: int
    fold: int
    truth: int
    probability: float


@dataclass(frozen=True)
class CandidateResult:
    model_name: ModelName
    fold_metrics: pd.DataFrame
    predictions: tuple[FoldPrediction, ...]
    best_parameters: tuple[dict[str, object], ...]
    feature_importances: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class Selection:
    model_name: ModelName
    reason: str
    default_threshold: float
    screening_threshold: float


def classification_metrics(
    truth: np.ndarray | pd.Series,
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute the predeclared binary metric bundle at one threshold."""

    truth_array = np.asarray(truth, dtype=int)
    probability_array = np.asarray(probability, dtype=float)
    predicted = (probability_array >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        truth_array, predicted, labels=[0, 1]
    ).ravel()

    both_classes = np.unique(truth_array).size == 2
    sensitivity = float(
        recall_score(truth_array, predicted, pos_label=1, zero_division=0)
    )
    specificity = float(
        recall_score(truth_array, predicted, pos_label=0, zero_division=0)
    )
    if both_classes:
        balanced_accuracy = (sensitivity + specificity) / 2
    elif np.any(truth_array == 1):
        balanced_accuracy = sensitivity
    else:
        balanced_accuracy = specificity
    return {
        "accuracy": float(accuracy_score(truth_array, predicted)),
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(precision_score(truth_array, predicted, zero_division=0)),
        "f1": float(f1_score(truth_array, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth_array, probability_array))
        if both_classes
        else float("nan"),
        "average_precision": float(
            average_precision_score(truth_array, probability_array)
        )
        if np.any(truth_array == 1)
        else float("nan"),
        "log_loss": float(
            log_loss(truth_array, probability_array, labels=[0, 1])
        ),
        "brier_score": float(brier_score_loss(truth_array, probability_array)),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def average_patient_predictions(
    predictions: tuple[FoldPrediction, ...],
) -> pd.DataFrame:
    """Average repeated out-of-fold probabilities once per development patient."""

    if not predictions:
        return pd.DataFrame(columns=["patient_id", "truth", "probability"])
    frame = pd.DataFrame(
        {
            "patient_id": [prediction.patient_id for prediction in predictions],
            "truth": [prediction.truth for prediction in predictions],
            "probability": [prediction.probability for prediction in predictions],
        }
    )
    truth_counts = frame.groupby("patient_id")["truth"].nunique()
    if (truth_counts != 1).any():
        patients = truth_counts.index[truth_counts != 1].tolist()
        raise ValueError(f"Conflicting truths for patient IDs: {patients}")
    return (
        frame.groupby("patient_id", as_index=False, sort=True)
        .agg(truth=("truth", "first"), probability=("probability", "mean"))
        .loc[:, ["patient_id", "truth", "probability"]]
    )


def choose_screening_threshold(
    truth: np.ndarray,
    probability: np.ndarray,
    recall_floor: float,
) -> float:
    """Maximize specificity subject to the predeclared sensitivity floor."""

    truth_array = np.asarray(truth, dtype=int)
    probability_array = np.asarray(probability, dtype=float)
    if truth_array.shape != probability_array.shape:
        raise ValueError("Truth and probability arrays must have identical shapes")
    if np.unique(truth_array).size != 2:
        raise ValueError("Screening threshold requires both target classes")
    if not 0 <= recall_floor <= 1:
        raise ValueError("Recall floor must be between zero and one")
    if not np.isfinite(probability_array).all() or not (
        (probability_array >= 0) & (probability_array <= 1)
    ).all():
        raise ValueError("Probabilities must be finite and between zero and one")

    candidates: list[tuple[float, float]] = []
    for threshold in np.unique(probability_array):
        predicted = probability_array >= threshold
        positives = truth_array == 1
        negatives = ~positives
        sensitivity = float(predicted[positives].mean())
        specificity = float((~predicted[negatives]).mean())
        if sensitivity >= recall_floor:
            candidates.append((specificity, float(threshold)))
    if not candidates:
        raise ValueError(f"No threshold reaches recall floor {recall_floor}")
    _, threshold = max(candidates, key=lambda item: (item[0], item[1]))
    return threshold


def _fit_outer_candidate(
    model_name: ModelName,
    features: pd.DataFrame,
    truth: pd.Series,
    *,
    config: ExperimentConfig,
    split_seed: int,
) -> tuple[object, dict[str, object]]:
    pipeline = build_pipeline(model_name, config.seed)
    grid = parameter_grid(model_name)
    if not grid:
        return pipeline.fit(features, truth), {}

    inner_cv = StratifiedKFold(
        n_splits=config.inner_splits,
        shuffle=True,
        random_state=split_seed,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="roc_auc",
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
        error_score="raise",
    ).fit(features, truth)
    best_parameters = dict(search.best_params_)
    if model_name not in NONLINEAR_MODELS:
        return search.best_estimator_, best_parameters

    calibration_cv = StratifiedKFold(
        n_splits=config.inner_splits,
        shuffle=True,
        random_state=split_seed,
    )
    calibrated = CalibratedClassifierCV(
        estimator=clone(search.best_estimator_),
        method="sigmoid",
        cv=calibration_cv,
        n_jobs=-1,
    ).fit(features, truth)
    return calibrated, best_parameters


def _pipeline_importances(pipeline: Pipeline) -> pd.DataFrame:
    if "preprocess" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        return pd.DataFrame(columns=["feature", "importance"])
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_)[0]
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": feature_names, "importance": values})


def _feature_importances(fitted: object) -> pd.DataFrame:
    if not isinstance(fitted, Pipeline):
        if isinstance(fitted, CalibratedClassifierCV):
            frames = [
                _pipeline_importances(calibrated.estimator)
                for calibrated in fitted.calibrated_classifiers_
            ]
        else:
            frames = []
    elif "calibrated_model" in fitted.named_steps:
        calibrated_model = fitted.named_steps["calibrated_model"]
        frames = [
            _pipeline_importances(calibrated.estimator)
            for calibrated in calibrated_model.calibrated_classifiers_
        ]
    else:
        frames = [_pipeline_importances(fitted)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.concat(frames, ignore_index=True)
        .groupby("feature", as_index=False, sort=True)["importance"]
        .mean()
    )


def run_nested_cv(
    data: CohortData,
    config: ExperimentConfig,
) -> dict[ModelName, CandidateResult]:
    """Evaluate candidates on identical repeated outer stratified folds."""

    outer_cv = RepeatedStratifiedKFold(
        n_splits=config.outer_splits,
        n_repeats=config.outer_repeats,
        random_state=config.seed,
    )
    outer_splits = list(outer_cv.split(data.features, data.target))
    metric_rows: dict[ModelName, list[dict[str, float | int]]] = {
        model_name: [] for model_name in CANDIDATE_MODELS
    }
    predictions: dict[ModelName, list[FoldPrediction]] = {
        model_name: [] for model_name in CANDIDATE_MODELS
    }
    parameters: dict[ModelName, list[dict[str, object]]] = {
        model_name: [] for model_name in CANDIDATE_MODELS
    }
    feature_importances: dict[ModelName, list[pd.DataFrame]] = {
        model_name: [] for model_name in CANDIDATE_MODELS
    }

    for split_index, (train_positions, test_positions) in enumerate(outer_splits):
        repeat = split_index // config.outer_splits
        fold = split_index % config.outer_splits
        training_features = data.features.iloc[train_positions]
        training_truth = data.target.iloc[train_positions]
        test_features = data.features.iloc[test_positions]
        test_truth = data.target.iloc[test_positions]

        for model_name in CANDIDATE_MODELS:
            fitted, best_parameters = _fit_outer_candidate(
                model_name,
                training_features,
                training_truth,
                config=config,
                split_seed=config.seed + split_index,
            )
            probability = fitted.predict_proba(test_features)[:, 1]
            metrics = classification_metrics(test_truth, probability)
            metric_rows[model_name].append(
                {"repeat": repeat, "fold": fold, **metrics}
            )
            parameters[model_name].append(best_parameters)
            fold_importances = _feature_importances(fitted)
            if not fold_importances.empty:
                feature_importances[model_name].append(
                    fold_importances.assign(repeat=repeat, fold=fold)
                )
            predictions[model_name].extend(
                FoldPrediction(
                    patient_id=int(patient_position),
                    repeat=repeat,
                    fold=fold,
                    truth=int(truth),
                    probability=float(patient_probability),
                )
                for patient_position, truth, patient_probability in zip(
                    test_positions, test_truth, probability, strict=True
                )
            )

    return {
        model_name: CandidateResult(
            model_name=model_name,
            fold_metrics=pd.DataFrame(metric_rows[model_name]),
            predictions=tuple(predictions[model_name]),
            best_parameters=tuple(parameters[model_name]),
            feature_importances=pd.concat(
                feature_importances[model_name], ignore_index=True
            )
            if feature_importances[model_name]
            else pd.DataFrame(columns=["feature", "importance", "repeat", "fold"]),
        )
        for model_name in CANDIDATE_MODELS
    }


def _selection_summary(result: CandidateResult) -> dict[str, float]:
    auc = result.fold_metrics["roc_auc"].dropna()
    standard_error = float(auc.std(ddof=1) / np.sqrt(len(auc))) if len(auc) > 1 else 0.0
    return {
        "roc_auc": float(auc.mean()),
        "roc_auc_se": standard_error,
        "brier_score": float(result.fold_metrics["brier_score"].mean()),
        "balanced_accuracy": float(
            result.fold_metrics["balanced_accuracy"].mean()
        ),
    }


def select_candidate(results: dict[ModelName, CandidateResult]) -> Selection:
    """Apply ROC-AUC selection and the one-standard-error simplicity rule."""

    selectable = {
        name: result for name, result in results.items() if name != "dummy"
    }
    if not selectable:
        raise ValueError("At least one non-dummy candidate result is required")
    summaries = {
        name: _selection_summary(result) for name, result in selectable.items()
    }
    best_name = min(
        selectable,
        key=lambda name: (
            -summaries[name]["roc_auc"],
            summaries[name]["brier_score"],
            -summaries[name]["balanced_accuracy"],
            MODEL_COMPLEXITY[name],
            name,
        ),
    )
    cutoff = summaries[best_name]["roc_auc"] - summaries[best_name]["roc_auc_se"]
    eligible = [
        name
        for name in selectable
        if summaries[name]["roc_auc"] >= cutoff - np.finfo(float).eps
    ]
    selected_name = min(
        eligible,
        key=lambda name: (
            MODEL_COMPLEXITY[name],
            summaries[name]["brier_score"],
            -summaries[name]["balanced_accuracy"],
            name,
        ),
    )

    averaged = average_patient_predictions(selectable[selected_name].predictions)
    screening_threshold = choose_screening_threshold(
        averaged["truth"].to_numpy(),
        averaged["probability"].to_numpy(),
        recall_floor=0.85,
    )
    reason = (
        f"{selected_name} is the simplest candidate within the one-standard-error "
        f"ROC-AUC cutoff ({cutoff:.4f}); calibration error and balanced accuracy "
        "were deterministic tie-breakers."
    )
    return Selection(
        model_name=selected_name,
        reason=reason,
        default_threshold=0.5,
        screening_threshold=screening_threshold,
    )
