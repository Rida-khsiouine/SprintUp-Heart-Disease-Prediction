from __future__ import annotations

import json
import re
from pathlib import Path

from heart_disease.evaluation import Selection
from heart_disease.models import ExperimentConfig
from heart_disease.reporting import (
    RESULTS_END,
    RESULTS_START,
    build_experiment_manifest,
    render_results_markdown,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_readme_results_match_metrics_json() -> None:
    metrics = json.loads(
        (PROJECT_ROOT / "reports" / "metrics.json").read_text(encoding="utf-8")
    )
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    generated = readme.split(RESULTS_START, 1)[1].split(RESULTS_END, 1)[0].strip()

    assert generated == render_results_markdown(metrics).strip()


def test_experiment_manifest_contains_required_provenance() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "reports" / "experiment-manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert {
        "artifact_sha256",
        "data_sha256",
        "git_commit",
        "package_versions",
        "parameters",
        "partitions",
        "profile",
        "seeds",
        "threshold_rule",
    } <= set(manifest)
    assert manifest["partitions"]["development"] == ["cleveland"]
    assert set(manifest["partitions"]["external_validation"]) == {
        "hungary",
        "switzerland",
        "va",
    }


def test_manifest_builder_records_unsupervised_boundaries() -> None:
    manifest = build_experiment_manifest(
        profile="smoke",
        config=ExperimentConfig(
            outer_splits=2,
            outer_repeats=1,
            inner_splits=2,
        ),
        data_manifest=json.loads(
            (PROJECT_ROOT / "data" / "manifest.json").read_text(
                encoding="utf-8"
            )
        ),
        artifact_metadata=json.loads(
            (PROJECT_ROOT / "artifacts" / "metadata.json").read_text(
                encoding="utf-8"
            )
        ),
        selection=Selection("logistic", "test", 0.5, 0.4),
        parameters={},
        unsupervised_summary={
            "config": {"seed": 42},
            "selected_k": 2,
            "retained_components": 7,
            "target_used_for_fit": False,
            "external_used_for_selection": False,
        },
    )

    assert manifest["partitions"]["unsupervised_fit"] == ["cleveland"]
    assert manifest["unsupervised"]["target_used_for_fit"] is False
    assert manifest["unsupervised"]["external_used_for_selection"] is False


def test_model_card_contains_non_medical_disclaimer() -> None:
    model_card = (PROJECT_ROOT / "docs" / "model-card.md").read_text(
        encoding="utf-8"
    )

    assert "not medical advice" in model_card.lower()
    assert "not clinically validated" in model_card.lower()
    assert "distribution shift" in model_card.lower()


def test_ai_report_discloses_ai_authorship_and_verification() -> None:
    report = (PROJECT_ROOT / "docs" / "ai-assisted-development.md").read_text(
        encoding="utf-8"
    )

    assert "ai-assisted" in report.lower()
    assert "human" in report.lower()
    assert "verification" in report.lower()
    assert "does not prove" in report.lower()


def test_project_guide_links_resolve_to_existing_paths() -> None:
    guide_path = PROJECT_ROOT / "docs" / "project-guide.md"
    guide = guide_path.read_text(encoding="utf-8")
    linked_paths = re.findall(r"\]\((?!https?://)([^)#]+)", guide)

    assert linked_paths
    assert all((guide_path.parent / path).resolve().exists() for path in linked_paths)


def test_readme_exposes_recruiter_paths_and_three_tracks() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Start here" in readme
    assert "Supervised track" in readme
    assert "Unsupervised track" in readme
    assert "Engineering track" in readme
    assert "docs/project-guide.md" in readme
    assert "reports/README.md" in readme


def test_model_card_limits_cluster_claims() -> None:
    model_card = (PROJECT_ROOT / "docs" / "model-card.md").read_text(
        encoding="utf-8"
    )
    normalized = model_card.lower()

    assert "exploratory" in normalized
    assert "cluster numbers" in normalized
    assert "not clinical" in normalized
    assert "euclidean" in normalized
