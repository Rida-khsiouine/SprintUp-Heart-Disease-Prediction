"""Final model refit after candidate and threshold decisions are frozen."""

from __future__ import annotations

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from heart_disease.evaluation import NONLINEAR_MODELS, Selection
from heart_disease.models import ExperimentConfig, build_pipeline, parameter_grid
from heart_disease.validation import CohortData


def train_final(
    development: CohortData,
    selection: Selection,
    config: ExperimentConfig,
) -> Pipeline:
    """Tune and refit the frozen candidate using Cleveland data only."""

    model_name = selection.model_name
    candidate = build_pipeline(model_name, config.seed)
    grid = parameter_grid(model_name)
    if grid:
        inner_cv = StratifiedKFold(
            n_splits=config.inner_splits,
            shuffle=True,
            random_state=config.seed,
        )
        search = GridSearchCV(
            estimator=candidate,
            param_grid=grid,
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,
            n_jobs=-1,
            error_score="raise",
        ).fit(development.features, development.target)
        fitted = search.best_estimator_
        parameters = dict(search.best_params_)
    else:
        fitted = candidate.fit(development.features, development.target)
        parameters = {}

    if model_name in NONLINEAR_MODELS:
        calibration_cv = StratifiedKFold(
            n_splits=config.inner_splits,
            shuffle=True,
            random_state=config.seed,
        )
        calibrated = CalibratedClassifierCV(
            estimator=clone(fitted),
            method="sigmoid",
            cv=calibration_cv,
            n_jobs=-1,
        ).fit(development.features, development.target)
        fitted = Pipeline([("calibrated_model", calibrated)])

    fitted.heart_disease_model_name_ = model_name
    fitted.heart_disease_parameters_ = parameters
    return fitted
