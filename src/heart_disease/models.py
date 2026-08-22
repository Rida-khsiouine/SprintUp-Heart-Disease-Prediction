"""Leakage-safe preprocessing and deterministic candidate estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ModelName = Literal["dummy", "logistic", "random_forest", "gradient_boosting"]

NUMERICAL_COLUMNS = ("age", "trestbps", "chol", "thalach", "oldpeak")
CATEGORICAL_COLUMNS = (
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Shared deterministic experiment settings."""

    seed: int = 42
    outer_splits: int = 5
    outer_repeats: int = 5
    inner_splits: int = 5
    bootstrap_iterations: int = 2000
    screening_recall: float = 0.85
    min_subgroup_size: int = 25


def _preprocessor() -> ColumnTransformer:
    numerical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numerical, list(NUMERICAL_COLUMNS)),
            ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def build_pipeline(model_name: ModelName, seed: int) -> Pipeline:
    """Create an unfitted end-to-end pipeline for a declared candidate."""

    if model_name == "dummy":
        estimator = DummyClassifier(strategy="prior", random_state=seed)
    elif model_name == "logistic":
        estimator = LogisticRegression(
            max_iter=2_000,
            random_state=seed,
            solver="liblinear",
        )
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=1,
        )
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("preprocess", _preprocessor()), ("model", estimator)])


def parameter_grid(model_name: ModelName) -> dict[str, list[object]]:
    """Return the bounded, predeclared inner-CV search space."""

    if model_name == "dummy":
        return {}
    if model_name == "logistic":
        return {
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [None, "balanced"],
        }
    if model_name == "random_forest":
        return {
            "model__max_depth": [None, 5],
            "model__min_samples_leaf": [1, 5],
            "model__class_weight": [None, "balanced"],
        }
    if model_name == "gradient_boosting":
        return {
            "model__learning_rate": [0.03, 0.1],
            "model__n_estimators": [100, 200],
            "model__max_depth": [1, 2],
        }
    raise ValueError(f"Unsupported model: {model_name}")
