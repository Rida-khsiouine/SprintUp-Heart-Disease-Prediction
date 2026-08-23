"""Deterministic model-diagnostic summaries and visual evidence."""

# ruff: noqa: E402

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import learning_curve as sklearn_learning_curve
from sklearn.pipeline import Pipeline

from heart_disease.evaluation import CandidateResult, Selection
from heart_disease.models import ExperimentConfig, ModelName
from heart_disease.validation import CohortData

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _rounded(value: float) -> float:
    return round(float(value), 12)


def compute_learning_curve(
    development: CohortData,
    pipeline: Pipeline,
    config: ExperimentConfig,
    *,
    train_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    """Measure selected-model learning behavior with leakage-safe resampling."""

    splitter = RepeatedStratifiedKFold(
        n_splits=config.outer_splits,
        n_repeats=config.outer_repeats,
        random_state=config.seed,
    )
    sizes, train_scores, validation_scores = sklearn_learning_curve(
        pipeline,
        development.features,
        development.target,
        train_sizes=np.asarray(train_fractions, dtype=float),
        cv=splitter,
        scoring="roc_auc",
        shuffle=True,
        random_state=config.seed,
        n_jobs=-1,
        error_score="raise",
    )
    return pd.DataFrame(
        {
            "training_rows": sizes.astype(int),
            "train_roc_auc_mean": train_scores.mean(axis=1),
            "train_roc_auc_std": train_scores.std(axis=1, ddof=1),
            "validation_roc_auc_mean": validation_scores.mean(axis=1),
            "validation_roc_auc_std": validation_scores.std(axis=1, ddof=1),
        }
    )


def _candidate_summary(
    model_name: ModelName,
    result: CandidateResult,
) -> dict[str, object]:
    folds = result.fold_metrics
    validation = folds["roc_auc"].astype(float)
    training = folds["train_roc_auc"].astype(float)
    return {
        "model_name": model_name,
        "folds": int(len(folds)),
        "train_roc_auc_mean": _rounded(training.mean()),
        "validation_roc_auc_mean": _rounded(validation.mean()),
        "validation_roc_auc_std": _rounded(validation.std(ddof=1)),
        "validation_roc_auc_se": _rounded(
            validation.std(ddof=1) / np.sqrt(len(validation))
        ),
        "roc_auc_gap": _rounded(training.mean() - validation.mean()),
    }


def _parameter_stability(result: CandidateResult) -> list[dict[str, object]]:
    serialized = [json.dumps(parameters, sort_keys=True) for parameters in result.best_parameters]
    counts = Counter(serialized)
    return [
        {
            "parameters": json.loads(parameters),
            "count": int(count),
            "fraction": _rounded(count / len(result.best_parameters)),
        }
        for parameters, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def _cohort_performance(metrics: Mapping[str, object]) -> list[dict[str, object]]:
    development = metrics["development"]
    if not isinstance(development, Mapping):
        raise TypeError("Development metrics must be a mapping")
    default = development["default"]
    if not isinstance(default, Mapping):
        raise TypeError("Default development metrics must be a mapping")

    rows: list[dict[str, object]] = []
    for cohort, metric in (("cleveland_oof", default),):
        intervals = metric["confidence_intervals"]
        interval = intervals["roc_auc"]
        rows.append(
            {
                "cohort": cohort,
                "roc_auc": _rounded(metric["roc_auc"]),
                "ci_low": _rounded(interval["low"]),
                "ci_high": _rounded(interval["high"]),
            }
        )
    external = metrics.get("external", [])
    if not isinstance(external, list):
        raise TypeError("External metrics must be a list")
    for metric in external:
        if not isinstance(metric, Mapping):
            raise TypeError("Each external metric must be a mapping")
        intervals = metric["confidence_intervals"]
        interval = intervals["roc_auc"]
        rows.append(
            {
                "cohort": str(metric["cohort"]),
                "roc_auc": _rounded(metric["roc_auc"]),
                "ci_low": _rounded(interval["low"]),
                "ci_high": _rounded(interval["high"]),
            }
        )
    return rows


def build_model_diagnostics(
    *,
    results: dict[ModelName, CandidateResult],
    selection: Selection,
    learning_curve: pd.DataFrame,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    """Build the JSON-safe diagnostic evidence consumed by every graph."""

    selected = results[selection.model_name]
    selected_summary = _candidate_summary(selection.model_name, selected)
    repeat_stability = (
        selected.fold_metrics.groupby("repeat", sort=True)["roc_auc"]
        .mean()
        .reset_index()
    )
    return {
        "profile": str(metrics["profile"]),
        "selected_model": selection.model_name,
        "generalization": {
            "train_roc_auc_mean": selected_summary["train_roc_auc_mean"],
            "validation_roc_auc_mean": selected_summary[
                "validation_roc_auc_mean"
            ],
            "roc_auc_gap": selected_summary["roc_auc_gap"],
        },
        "model_comparison": [
            _candidate_summary(model_name, result)
            for model_name, result in results.items()
        ],
        "learning_curve": [
            {
                key: int(value) if key == "training_rows" else _rounded(value)
                for key, value in row.items()
            }
            for row in learning_curve.to_dict(orient="records")
        ],
        "parameter_stability": _parameter_stability(selected),
        "repeat_stability": [
            {
                "repeat": int(row.repeat),
                "mean_roc_auc": _rounded(row.roc_auc),
            }
            for row in repeat_stability.itertuples(index=False)
        ],
        "cohort_performance": _cohort_performance(metrics),
    }


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_model_comparison(summary: Mapping[str, object], path: Path) -> None:
    table = pd.DataFrame(summary["model_comparison"]).sort_values(
        "validation_roc_auc_mean"
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.barh(
        table["model_name"],
        table["validation_roc_auc_mean"],
        xerr=table["validation_roc_auc_se"],
        alpha=0.85,
    )
    axis.axvline(0.5, linestyle="--", color="0.5", label="Chance")
    axis.set(
        xlabel="Mean nested ROC-AUC ± standard error",
        ylabel="Candidate",
        xlim=(0.45, 1.0),
        title="Candidate model comparison",
    )
    axis.legend()
    _save_figure(figure, path)


def _plot_learning_curve(summary: Mapping[str, object], path: Path) -> None:
    table = pd.DataFrame(summary["learning_curve"])
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for label, prefix in (("Training", "train"), ("Validation", "validation")):
        mean = table[f"{prefix}_roc_auc_mean"].to_numpy(float)
        std = table[f"{prefix}_roc_auc_std"].to_numpy(float)
        rows = table["training_rows"].to_numpy(int)
        axis.plot(rows, mean, marker="o", linewidth=2, label=label)
        axis.fill_between(rows, mean - std, mean + std, alpha=0.16)
    axis.set(
        xlabel="Training rows per fold",
        ylabel="ROC-AUC",
        ylim=(0.5, 1.01),
        title="Learning curve",
    )
    axis.legend()
    _save_figure(figure, path)


def _plot_train_validation_gap(summary: Mapping[str, object], path: Path) -> None:
    table = pd.DataFrame(summary["model_comparison"])
    melted = table.melt(
        id_vars="model_name",
        value_vars=["train_roc_auc_mean", "validation_roc_auc_mean"],
        var_name="evidence",
        value_name="roc_auc",
    )
    melted["evidence"] = melted["evidence"].map(
        {
            "train_roc_auc_mean": "Training",
            "validation_roc_auc_mean": "Nested validation",
        }
    )
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    sns.barplot(data=melted, x="model_name", y="roc_auc", hue="evidence", ax=axis)
    axis.set(
        xlabel="Candidate",
        ylabel="ROC-AUC",
        ylim=(0.45, 1.01),
        title="Training versus held-out performance",
    )
    axis.tick_params(axis="x", rotation=15)
    _save_figure(figure, path)


def _plot_parameter_stability(summary: Mapping[str, object], path: Path) -> None:
    table = pd.DataFrame(summary["parameter_stability"])
    table["label"] = table["parameters"].map(
        lambda value: ", ".join(
            f"{key.removeprefix('model__')}={setting}"
            for key, setting in value.items()
        )
        or "No tuned parameters"
    )
    table = table.sort_values(["count", "label"])
    figure, axis = plt.subplots(figsize=(8.0, max(3.5, len(table) * 0.55)))
    axis.barh(table["label"], table["count"], alpha=0.85)
    axis.set(
        xlabel="Outer folds selecting combination",
        ylabel="Inner-CV winner",
        title=f"{summary['selected_model']} hyperparameter stability",
    )
    _save_figure(figure, path)


def _plot_cohort_performance(summary: Mapping[str, object], path: Path) -> None:
    table = pd.DataFrame(summary["cohort_performance"])
    values = table["roc_auc"].to_numpy(float)
    errors = np.vstack(
        [
            values - table["ci_low"].to_numpy(float),
            table["ci_high"].to_numpy(float) - values,
        ]
    )
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    axis.errorbar(
        table["cohort"],
        values,
        yerr=errors,
        fmt="o",
        markersize=7,
        capsize=5,
        linewidth=2,
    )
    axis.axhline(0.5, linestyle="--", color="0.5", label="Chance")
    axis.set(
        xlabel="Frozen evaluation cohort",
        ylabel="ROC-AUC with 95% interval",
        ylim=(0.45, 1.01),
        title="Internal and external discrimination",
    )
    axis.tick_params(axis="x", rotation=15)
    axis.legend()
    _save_figure(figure, path)


def write_model_diagnostics(
    summary: dict[str, object],
    reports_dir: Path,
) -> dict[str, Path]:
    """Persist one canonical diagnostic summary and its five graph views."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "model-diagnostics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "model-diagnostics.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    learning_curve_path = reports_dir / "learning-curve.csv"
    pd.DataFrame(summary["learning_curve"]).to_csv(
        learning_curve_path, index=False, lineterminator="\n"
    )

    sns.set_theme(style="whitegrid", context="notebook")
    _plot_model_comparison(summary, figures_dir / "model-comparison.png")
    _plot_learning_curve(summary, figures_dir / "learning-curve.png")
    _plot_train_validation_gap(summary, figures_dir / "train-validation-gap.png")
    _plot_parameter_stability(
        summary, figures_dir / "hyperparameter-stability.png"
    )
    _plot_cohort_performance(summary, figures_dir / "cohort-performance.png")
    return {
        "summary": summary_path,
        "learning_curve": learning_curve_path,
        "figures": figures_dir,
    }
