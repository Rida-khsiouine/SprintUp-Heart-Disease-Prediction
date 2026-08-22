"""Deterministic reports and figures for unsupervised patient profiles."""

# ruff: noqa: E402

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from heart_disease.unsupervised import (
    FittedProfiles,
    UnsupervisedConfig,
    UnsupervisedStudy,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    sort_by: list[str],
) -> None:
    frame.sort_values(sort_by, kind="stable").to_csv(
        path,
        index=False,
        lineterminator="\n",
    )


def _summary(
    study: UnsupervisedStudy,
    *,
    profile: str,
    config: UnsupervisedConfig,
) -> dict[str, object]:
    external_rows = []
    external = study.external_transfer.loc[
        ~study.external_transfer["cohort"].eq("cleveland")
    ]
    for cohort_name, rows in external.groupby("cohort", sort=True):
        external_rows.append(
            {
                "cohort": str(cohort_name),
                "cluster_proportion_total_variation": float(
                    rows["cluster_proportion_total_variation"].iloc[0]
                ),
                "underpowered_clusters": int(
                    rows["status"].eq("underpowered").sum()
                ),
            }
        )
    return {
        "profile": profile,
        "config": asdict(config),
        "development_cohort": "cleveland",
        "external_cohorts": [row["cohort"] for row in external_rows],
        **study.interpretation,
        "external_transfer": external_rows,
        "cluster_labels_arbitrary": True,
        "clinical_interpretation": "unavailable: exploratory groupings only",
    }


def _plot_pca_variance(
    study: UnsupervisedStudy,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.bar(
        study.pca_summary["component"],
        study.pca_summary["explained_variance"],
        alpha=0.7,
        label="Per component",
    )
    axis.plot(
        study.pca_summary["component"],
        study.pca_summary["cumulative_explained_variance"],
        marker="o",
        linewidth=2,
        label="Cumulative",
    )
    axis.axhline(0.9, linestyle="--", color="0.4", label="90% target")
    axis.set(
        xlabel="Principal component",
        ylabel="Explained variance ratio",
        title="Exploratory PCA variance retention",
        ylim=(0, 1.02),
    )
    axis.legend()
    _save_figure(figure, path)


def _plot_pca_clusters(
    study: UnsupervisedStudy,
    path: Path,
) -> None:
    development = study.patient_assignments.loc[
        study.patient_assignments["cohort"].eq("cleveland")
    ]
    figure, axis = plt.subplots(figsize=(6.8, 5.2))
    sns.scatterplot(
        data=development,
        x="pc1",
        y="pc2",
        hue="kmeans_cluster",
        palette="colorblind",
        alpha=0.8,
        ax=axis,
    )
    axis.set(title="Exploratory Cleveland clusters in PCA space")
    _save_figure(figure, path)


def _plot_cluster_selection(
    study: UnsupervisedStudy,
    path: Path,
) -> None:
    table = study.cluster_selection.sort_values("k")
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for axis, column, title in (
        (axes[0], "silhouette", "Silhouette (higher is better)"),
        (axes[1], "davies_bouldin", "Davies–Bouldin (lower is better)"),
        (axes[2], "stability_ari_mean", "Selected-k stability ARI"),
    ):
        axis.plot(table["k"], table[column], marker="o")
        axis.axvline(study.selected_k, linestyle="--", color="0.4")
        axis.set(xlabel="k", title=title)
    figure.suptitle("Exploratory label-free cluster selection")
    _save_figure(figure, path)


def _plot_dendrogram(fitted: FittedProfiles, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.0, 5.2))
    dendrogram(
        fitted.linkage_matrix,
        no_labels=True,
        color_threshold=None,
        ax=axis,
    )
    axis.set(
        xlabel="Cleveland patients",
        ylabel="Ward distance",
        title="Exploratory Ward hierarchical dendrogram",
    )
    _save_figure(figure, path)


def _plot_cluster_profiles(
    study: UnsupervisedStudy,
    path: Path,
) -> None:
    medians = study.cluster_profiles.loc[
        study.cluster_profiles["statistic"].eq("standardized_median")
    ].pivot(index="cluster", columns="feature", values="value")
    medians = medians.replace([np.inf, -np.inf], np.nan).fillna(0)
    color_limit = max(float(medians.abs().to_numpy().max()), 0.5)
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    sns.heatmap(
        medians,
        annot=True,
        fmt=".2f",
        center=0,
        vmin=-color_limit,
        vmax=color_limit,
        cmap="vlag",
        ax=axis,
    )
    axis.set(
        xlabel="Numeric feature",
        ylabel="Canonical cluster ID",
        title="Exploratory Cleveland-standardized cluster medians",
    )
    _save_figure(figure, path)


def write_unsupervised_reports(
    *,
    fitted: FittedProfiles,
    study: UnsupervisedStudy,
    profile: str,
    config: UnsupervisedConfig,
    reports_dir: Path,
) -> dict[str, Path]:
    """Write synchronized unsupervised evidence beneath reports/unsupervised."""

    output_dir = reports_dir / "unsupervised"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    paths = {
        "summary": output_dir / "summary.json",
        "pca_summary": output_dir / "pca-summary.csv",
        "pca_loadings": output_dir / "pca-loadings.csv",
        "cluster_selection": output_dir / "cluster-selection.csv",
        "hierarchical_comparison": output_dir / "hierarchical-comparison.csv",
        "cluster_profiles": output_dir / "cluster-profiles.csv",
        "patient_assignments": output_dir / "patient-assignments.csv",
        "external_transfer": output_dir / "external-transfer.csv",
        "figures": figures_dir,
    }
    summary = _summary(study, profile=profile, config=config)
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_table(study.pca_summary, paths["pca_summary"], sort_by=["component"])
    _write_table(
        study.pca_loadings,
        paths["pca_loadings"],
        sort_by=["component", "feature"],
    )
    _write_table(
        study.cluster_selection,
        paths["cluster_selection"],
        sort_by=["k"],
    )
    _write_table(
        study.hierarchical_comparison,
        paths["hierarchical_comparison"],
        sort_by=["kmeans_cluster", "hierarchical_cluster"],
    )
    _write_table(
        study.cluster_profiles,
        paths["cluster_profiles"],
        sort_by=["cluster", "feature", "statistic", "level"],
    )
    _write_table(
        study.patient_assignments,
        paths["patient_assignments"],
        sort_by=["cohort", "patient_id"],
    )
    _write_table(
        study.external_transfer,
        paths["external_transfer"],
        sort_by=["cohort", "cluster"],
    )

    _plot_pca_variance(study, figures_dir / "pca-explained-variance.png")
    _plot_pca_clusters(study, figures_dir / "pca-clusters.png")
    _plot_cluster_selection(study, figures_dir / "cluster-selection.png")
    _plot_dendrogram(fitted, figures_dir / "hierarchical-dendrogram.png")
    _plot_cluster_profiles(study, figures_dir / "cluster-profiles.png")
    return paths
