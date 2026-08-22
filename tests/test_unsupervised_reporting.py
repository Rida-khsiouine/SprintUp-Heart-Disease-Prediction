from __future__ import annotations

import json
from pathlib import Path

import pytest

from heart_disease.data import Cohort
from heart_disease.reporting import (
    UNSUPERVISED_END,
    UNSUPERVISED_START,
    render_unsupervised_markdown,
    sync_readme_unsupervised,
)
from heart_disease.unsupervised import (
    FittedProfiles,
    UnsupervisedConfig,
    UnsupervisedStudy,
    describe_unsupervised_profiles,
    fit_unsupervised_profiles,
)
from heart_disease.unsupervised_reporting import write_unsupervised_reports
from heart_disease.validation import load_cohort

PROJECT_ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def study_fixture() -> tuple[
    FittedProfiles,
    UnsupervisedStudy,
    UnsupervisedConfig,
]:
    data_dir = PROJECT_ROOT / "data"
    development = load_cohort(Cohort.CLEVELAND, data_dir)
    external = tuple(
        load_cohort(cohort, data_dir)
        for cohort in (Cohort.HUNGARY, Cohort.SWITZERLAND, Cohort.VA)
    )
    config = UnsupervisedConfig(
        k_values=(2, 3),
        stability_iterations=2,
    )
    fitted = fit_unsupervised_profiles(development.features, config)
    study = describe_unsupervised_profiles(
        fitted,
        development,
        external,
        config,
    )
    return fitted, study, config


def test_unsupervised_reports_write_declared_outputs(
    tmp_path: Path,
    study_fixture: tuple[
        FittedProfiles,
        UnsupervisedStudy,
        UnsupervisedConfig,
    ],
) -> None:
    fitted, study, config = study_fixture

    paths = write_unsupervised_reports(
        fitted=fitted,
        study=study,
        profile="smoke",
        config=config,
        reports_dir=tmp_path,
    )

    assert set(paths) == {
        "summary",
        "pca_summary",
        "pca_loadings",
        "cluster_selection",
        "hierarchical_comparison",
        "cluster_profiles",
        "patient_assignments",
        "external_transfer",
        "figures",
    }
    assert all(path.exists() for path in paths.values())
    assert {
        "pca-explained-variance.png",
        "pca-clusters.png",
        "cluster-selection.png",
        "hierarchical-dendrogram.png",
        "cluster-profiles.png",
    } == {path.name for path in paths["figures"].iterdir()}

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["profile"] == "smoke"
    assert summary["selected_k"] == study.selected_k
    assert summary["retained_variance"] >= config.retained_variance
    assert summary["target_used_for_fit"] is False
    assert summary["external_used_for_selection"] is False


def test_unsupervised_readme_sync_uses_summary_only(tmp_path: Path) -> None:
    summary = {
        "profile": "smoke",
        "selected_k": 2,
        "retained_components": 7,
        "retained_variance": 0.91,
        "silhouette": 0.24,
        "stability_ari_mean": 0.72,
        "hierarchical_ari": 0.55,
        "external_transfer": [],
    }
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        f"before\n{UNSUPERVISED_START}\nstale\n"
        f"{UNSUPERVISED_END}\nafter\n",
        encoding="utf-8",
    )

    sync_readme_unsupervised(readme_path, summary)

    readme = readme_path.read_text(encoding="utf-8")
    generated = (
        readme.split(UNSUPERVISED_START, 1)[1]
        .split(UNSUPERVISED_END, 1)[0]
        .strip()
    )
    assert generated == render_unsupervised_markdown(summary).strip()
