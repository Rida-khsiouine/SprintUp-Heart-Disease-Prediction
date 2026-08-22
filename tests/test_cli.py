from __future__ import annotations

import json
from pathlib import Path

import heart_disease.evaluation as evaluation
from heart_disease.cli import main


def test_cli_smoke_reproduce_creates_expected_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = Path(__file__).parents[1]
    monkeypatch.setattr(
        evaluation,
        "CANDIDATE_MODELS",
        ("dummy", "logistic"),
    )

    exit_code = main(
        [
            "reproduce",
            "--profile",
            "smoke",
            "--data-dir",
            str(project_root / "data"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "artifacts" / "model.skops").is_file()
    assert (tmp_path / "artifacts" / "metadata.json").is_file()
    assert (tmp_path / "reports" / "metrics.json").is_file()
    assert (tmp_path / "reports" / "external-validation.csv").is_file()
    metrics = json.loads(
        (tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["profile"] == "smoke"
    assert metrics["selection"]["model_name"] == "logistic"
