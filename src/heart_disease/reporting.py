"""Generated evidence tables, figures, README synchronization, and provenance."""

# ruff: noqa: E402

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline

from heart_disease.evaluation import (
    CandidateResult,
    Selection,
    average_patient_predictions,
)
from heart_disease.external import (
    Thresholds,
    compare_cohorts,
    evaluate_subgroups,
)
from heart_disease.models import ExperimentConfig, ModelName
from heart_disease.validation import FEATURE_COLUMNS, CohortData

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

RESULTS_START = "<!-- GENERATED_RESULTS_START -->"
RESULTS_END = "<!-- GENERATED_RESULTS_END -->"
UNSUPERVISED_START = "<!-- GENERATED_UNSUPERVISED_START -->"
UNSUPERVISED_END = "<!-- GENERATED_UNSUPERVISED_END -->"
DIAGNOSTICS_START = "<!-- GENERATED_DIAGNOSTICS_START -->"
DIAGNOSTICS_END = "<!-- GENERATED_DIAGNOSTICS_END -->"


def _metric_interval(metric: dict[str, object], name: str) -> str:
    value = float(metric[name])
    intervals = metric.get("confidence_intervals", {})
    if isinstance(intervals, dict) and name in intervals:
        interval = intervals[name]
        if isinstance(interval, dict):
            return (
                f"{value:.3f} [{float(interval['low']):.3f}, "
                f"{float(interval['high']):.3f}]"
            )
    return f"{value:.3f}"


def render_results_markdown(metrics: dict[str, object]) -> str:
    """Render the README evidence table solely from metrics.json content."""

    lines = [
        f"_Generated from `reports/metrics.json` ({metrics['profile']} profile). "
        "Intervals are 95% stratified patient-bootstrap intervals._",
        "",
        "| Evidence | ROC-AUC | Balanced accuracy | Brier score |",
        "|---|---:|---:|---:|",
    ]
    candidates = metrics["candidates"]
    for name in ("dummy", "logistic", "gradient_boosting", "random_forest"):
        if name not in candidates:
            continue
        candidate = candidates[name]
        lines.append(
            f"| Cleveland nested CV — `{name}` | "
            f"{float(candidate['mean_roc_auc']):.3f} ± "
            f"{float(candidate['std_roc_auc']):.3f} | "
            f"{float(candidate['mean_balanced_accuracy']):.3f} | "
            f"{float(candidate['mean_brier_score']):.3f} |"
        )
    development = metrics["development"]["default"]
    selected = metrics["selection"]["model_name"]
    lines.append(
        f"| Cleveland selected OOF — `{selected}` | "
        f"{_metric_interval(development, 'roc_auc')} | "
        f"{_metric_interval(development, 'balanced_accuracy')} | "
        f"{_metric_interval(development, 'brier_score')} |"
    )
    for external in metrics.get("external", []):
        lines.append(
            f"| External — {external['cohort']} | "
            f"{_metric_interval(external, 'roc_auc')} | "
            f"{_metric_interval(external, 'balanced_accuracy')} | "
            f"{_metric_interval(external, 'brier_score')} |"
        )
    return "\n".join(lines)


def sync_readme_results(readme_path: Path, metrics: dict[str, object]) -> None:
    """Replace only the fixed generated-result block in the recruiter README."""

    content = readme_path.read_text(encoding="utf-8")
    if content.count(RESULTS_START) != 1 or content.count(RESULTS_END) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")
    before, remainder = content.split(RESULTS_START, 1)
    _, after = remainder.split(RESULTS_END, 1)
    block = render_results_markdown(metrics)
    readme_path.write_text(
        f"{before}{RESULTS_START}\n{block}\n{RESULTS_END}{after}",
        encoding="utf-8",
    )


def render_model_diagnostics_markdown(diagnostics: dict[str, object]) -> str:
    """Render an accessible graph index from model-diagnostics.json."""

    generalization = diagnostics["generalization"]
    if not isinstance(generalization, dict):
        raise TypeError("Diagnostic generalization evidence must be a mapping")
    selected = diagnostics["selected_model"]
    return "\n".join(
        [
            f"_Generated from `reports/model-diagnostics.json` "
            f"({diagnostics['profile']} profile)._",
            "",
            f"Selected `{selected}` mean training ROC-AUC: "
            f"**{float(generalization['train_roc_auc_mean']):.3f}**; "
            f"nested validation ROC-AUC: "
            f"**{float(generalization['validation_roc_auc_mean']):.3f}**; "
            f"observed gap: **{float(generalization['roc_auc_gap']):.3f}**.",
            "",
            "| Candidate comparison | Learning curve |",
            "|---|---|",
            "| ![Nested ROC-AUC by candidate]("
            "reports/model-diagnostics/model-comparison.png) "
            "| ![Training and validation learning curve]("
            "reports/model-diagnostics/learning-curve.png) |",
            "| Training versus validation | Hyperparameter stability |",
            "| ![Training versus held-out ROC-AUC]("
            "reports/model-diagnostics/train-validation-gap.png) "
            "| ![Outer-fold hyperparameter selection frequency]("
            "reports/model-diagnostics/hyperparameter-stability.png) |",
            "",
            "![Internal and external ROC-AUC with uncertainty]("
            "reports/model-diagnostics/cohort-performance.png)",
            "",
            "Inspect the [machine-readable diagnostic](reports/model-diagnostics.json) "
            "or the [learning-curve data](reports/learning-curve.csv).",
        ]
    )


def sync_readme_diagnostics(
    readme_path: Path,
    diagnostics: dict[str, object],
) -> None:
    """Replace only the fixed model-diagnostics README block."""

    content = readme_path.read_text(encoding="utf-8")
    if (
        content.count(DIAGNOSTICS_START) != 1
        or content.count(DIAGNOSTICS_END) != 1
    ):
        raise ValueError(
            "README must contain exactly one generated-diagnostics marker pair"
        )
    before, remainder = content.split(DIAGNOSTICS_START, 1)
    _, after = remainder.split(DIAGNOSTICS_END, 1)
    block = render_model_diagnostics_markdown(diagnostics)
    readme_path.write_text(
        f"{before}{DIAGNOSTICS_START}\n{block}\n{DIAGNOSTICS_END}{after}",
        encoding="utf-8",
    )


def render_unsupervised_markdown(summary: dict[str, object]) -> str:
    """Render README evidence solely from unsupervised summary content."""

    lines = [
        f"_Generated from `reports/unsupervised/summary.json` "
        f"({summary['profile']} profile). Labels were not used for fitting._",
        "",
        "| Exploratory evidence | Value |",
        "|---|---:|",
        f"| Selected clusters | {int(summary['selected_k'])} |",
        f"| PCA components retained | {int(summary['retained_components'])} |",
        f"| Cumulative variance retained | {float(summary['retained_variance']):.3f} |",
        f"| Silhouette score | {float(summary['silhouette']):.3f} |",
        f"| Subsample stability ARI | {float(summary['stability_ari_mean']):.3f} |",
        f"| K-Means vs Ward ARI | {float(summary['hierarchical_ari']):.3f} |",
    ]
    external = summary.get("external_transfer", [])
    if isinstance(external, list) and external:
        lines.extend(
            [
                "",
                "| Frozen external transfer | Cluster-proportion distance |",
                "|---|---:|",
            ]
        )
        for row in external:
            if isinstance(row, dict):
                lines.append(
                    f"| {row['cohort']} | "
                    f"{float(row['cluster_proportion_total_variation']):.3f} |"
                )
    return "\n".join(lines)


def sync_readme_unsupervised(
    readme_path: Path,
    summary: dict[str, object],
) -> None:
    """Replace only the fixed unsupervised README evidence block."""

    content = readme_path.read_text(encoding="utf-8")
    if (
        content.count(UNSUPERVISED_START) != 1
        or content.count(UNSUPERVISED_END) != 1
    ):
        raise ValueError(
            "README must contain exactly one generated-unsupervised marker pair"
        )
    before, remainder = content.split(UNSUPERVISED_START, 1)
    _, after = remainder.split(UNSUPERVISED_END, 1)
    block = render_unsupervised_markdown(summary)
    readme_path.write_text(
        f"{before}{UNSUPERVISED_START}\n{block}\n{UNSUPERVISED_END}{after}",
        encoding="utf-8",
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def generate_evidence_outputs(
    *,
    reports_dir: Path,
    development: CohortData,
    external_cohorts: tuple[CohortData, ...],
    fitted: Pipeline,
    results: dict[ModelName, CandidateResult],
    selection: Selection,
    thresholds: Thresholds,
) -> dict[str, Path]:
    """Write evidence tables and the seven predeclared diagnostic figures."""

    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    selected_result = results[selection.model_name]
    oof = average_patient_predictions(selected_result.predictions)
    truth = oof["truth"].to_numpy(dtype=int)
    probability = oof["probability"].to_numpy(dtype=float)
    oof_path = reports_dir / "oof-predictions.csv"
    oof.to_csv(oof_path, index=False, lineterminator="\n")

    false_positive_rate, true_positive_rate, _ = roc_curve(truth, probability)
    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    axis.plot(false_positive_rate, true_positive_rate, linewidth=2, label="Selected OOF")
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Chance")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Cleveland ROC")
    axis.legend()
    _save_figure(figure, figures_dir / "roc-curve.png")

    precision, recall, _ = precision_recall_curve(truth, probability)
    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    axis.plot(recall, precision, linewidth=2)
    axis.axhline(truth.mean(), linestyle="--", color="0.5", label="Prevalence")
    axis.set(xlabel="Recall", ylabel="Precision", title="Cleveland precision–recall")
    axis.legend()
    _save_figure(figure, figures_dir / "precision-recall-curve.png")

    observed, predicted = calibration_curve(
        truth, probability, n_bins=10, strategy="quantile"
    )
    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    axis.plot(predicted, observed, marker="o", linewidth=2, label="Selected OOF")
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Ideal")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="Calibration")
    axis.legend()
    _save_figure(figure, figures_dir / "calibration-curve.png")

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for axis, (name, threshold) in zip(
        axes,
        (("Default", thresholds.default), ("Screening", thresholds.screening)),
        strict=True,
    ):
        matrix = confusion_matrix(truth, probability >= threshold, labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cbar=False, ax=axis, cmap="Blues")
        axis.set(xlabel="Predicted", ylabel="Observed", title=f"{name} ({threshold:.3f})")
    figure.suptitle("Cleveland out-of-fold confusion matrices")
    _save_figure(figure, figures_dir / "confusion-matrices.png")

    cohorts = (development, *external_cohorts)
    missingness = pd.DataFrame(
        {
            cohort.cohort.value: cohort.features.isna().mean()
            for cohort in cohorts
        }
    ).T.loc[:, FEATURE_COLUMNS]
    missingness_path = reports_dir / "missingness.csv"
    missingness.to_csv(missingness_path, lineterminator="\n")
    figure, axis = plt.subplots(figsize=(11.0, 3.8))
    sns.heatmap(missingness, cmap="mako_r", vmin=0, vmax=1, ax=axis)
    axis.set(xlabel="Feature", ylabel="Cohort", title="Feature missingness by cohort")
    _save_figure(figure, figures_dir / "missingness.png")

    shifts = pd.concat(
        [compare_cohorts(development, cohort) for cohort in external_cohorts],
        ignore_index=True,
    )
    shift_path = reports_dir / "cohort-shift.csv"
    shifts.to_csv(shift_path, index=False, lineterminator="\n")
    plotted_shifts = shifts[
        shifts["measure"].isin(
            ["standardized_mean_difference", "total_variation_distance"]
        )
    ].copy()
    plotted_shifts["absolute_shift"] = plotted_shifts["shift_value"].abs()
    pivot = plotted_shifts.pivot(
        index="external_cohort", columns="feature", values="absolute_shift"
    )
    figure, axis = plt.subplots(figsize=(11.0, 3.6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="rocket_r", ax=axis)
    axis.set(xlabel="Feature", ylabel="External cohort", title="Absolute univariate cohort shift")
    _save_figure(figure, figures_dir / "cohort-shift.png")

    stability = selected_result.feature_importances.copy()
    stability_path = reports_dir / "feature-stability.csv"
    stability.to_csv(stability_path, index=False, lineterminator="\n")
    figure, axis = plt.subplots(figsize=(8.0, 5.8))
    if stability.empty:
        axis.text(0.5, 0.5, "Feature stability unavailable", ha="center", va="center")
        axis.set_axis_off()
    else:
        stability["absolute_importance"] = stability["importance"].abs()
        summary = (
            stability.groupby("feature")["absolute_importance"]
            .agg(["mean", "std"])
            .fillna(0)
            .nlargest(15, "mean")
            .sort_values("mean")
        )
        axis.barh(summary.index, summary["mean"], xerr=summary["std"], alpha=0.85)
        axis.set(
            xlabel="Mean absolute importance ± fold SD",
            title="Feature stability across outer folds",
        )
    _save_figure(figure, figures_dir / "feature-stability.png")

    subgroup_frames = []
    development_subgroups = evaluate_subgroups(
        oof["truth"], probability, development.features, thresholds, min_size=25
    ).assign(cohort=development.cohort.value, evidence="out_of_fold")
    subgroup_frames.append(development_subgroups)
    for cohort in external_cohorts:
        cohort_probability = fitted.predict_proba(cohort.features)[:, 1]
        subgroup_frames.append(
            evaluate_subgroups(
                cohort.target,
                cohort_probability,
                cohort.features,
                thresholds,
                min_size=25,
            ).assign(cohort=cohort.cohort.value, evidence="external")
        )
    subgroup_path = reports_dir / "subgroup-analysis.csv"
    pd.concat(subgroup_frames, ignore_index=True).to_csv(
        subgroup_path, index=False, lineterminator="\n"
    )
    return {
        "oof_predictions": oof_path,
        "missingness": missingness_path,
        "cohort_shift": shift_path,
        "feature_stability": stability_path,
        "subgroups": subgroup_path,
        "figures": figures_dir,
    }


def build_experiment_manifest(
    *,
    profile: str,
    config: ExperimentConfig,
    data_manifest: dict[str, object],
    artifact_metadata: dict[str, object],
    selection: Selection,
    parameters: dict[str, object],
    unsupervised_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build immutable provenance without wall-clock timestamps."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cohorts = data_manifest["cohorts"]
    manifest: dict[str, object] = {
        "profile": profile,
        "git_commit": commit or "unavailable",
        "data_sha256": {
            name: cohort["sha256"] for name, cohort in cohorts.items()
        },
        "artifact_sha256": artifact_metadata["model_sha256"],
        "package_versions": artifact_metadata["library_versions"],
        "seeds": asdict(config),
        "partitions": {
            "development": ["cleveland"],
            "external_validation": ["hungary", "switzerland", "va"],
            "outer_evaluation": (
                f"{config.outer_repeats}x{config.outer_splits} repeated stratified folds"
            ),
            "inner_tuning": f"{config.inner_splits} stratified folds",
        },
        "parameters": parameters,
        "selected_model": selection.model_name,
        "threshold_rule": {
            "default": 0.5,
            "screening": (
                "highest threshold maximizing specificity subject to Cleveland "
                f"out-of-fold recall >= {config.screening_recall}"
            ),
            "screening_value": selection.screening_threshold,
        },
    }
    if unsupervised_summary is not None:
        partitions = manifest["partitions"]
        if not isinstance(partitions, dict):
            raise TypeError("Manifest partitions must be a mapping")
        partitions["unsupervised_fit"] = ["cleveland"]
        manifest["unsupervised"] = {
            "target_used_for_fit": bool(
                unsupervised_summary["target_used_for_fit"]
            ),
            "external_used_for_selection": bool(
                unsupervised_summary["external_used_for_selection"]
            ),
            "config": unsupervised_summary["config"],
            "selected_k": int(unsupervised_summary["selected_k"]),
            "retained_components": int(
                unsupervised_summary["retained_components"]
            ),
        }
    return manifest
