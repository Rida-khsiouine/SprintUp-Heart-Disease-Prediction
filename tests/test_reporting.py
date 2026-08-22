from __future__ import annotations

import json
from pathlib import Path

from heart_disease.reporting import (
    RESULTS_END,
    RESULTS_START,
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
