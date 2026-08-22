from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import heart_disease.artifacts as artifacts
import heart_disease.training as training
from heart_disease.artifacts import (
    ArtifactValidationError,
    load_artifact,
    predict_record,
    save_artifact,
)
from heart_disease.data import Cohort
from heart_disease.evaluation import Selection
from heart_disease.models import ExperimentConfig, build_pipeline
from heart_disease.training import train_final
from heart_disease.validation import CohortData


def _features(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": np.arange(rows, dtype=float) + 40,
            "sex": np.arange(rows) % 2,
            "cp": np.arange(rows) % 4 + 1,
            "trestbps": np.arange(rows, dtype=float) + 110,
            "chol": np.arange(rows, dtype=float) + 180,
            "fbs": np.arange(rows) % 2,
            "restecg": np.arange(rows) % 3,
            "thalach": 180 - np.arange(rows, dtype=float),
            "exang": np.arange(rows) % 2,
            "oldpeak": np.arange(rows, dtype=float) / 10,
            "slope": np.arange(rows) % 3 + 1,
            "ca": np.arange(rows) % 4,
            "thal": np.resize([3, 6, 7], rows),
        }
    )


def _record() -> dict[str, object]:
    return _features(1).iloc[0].to_dict()


def _metadata() -> dict[str, object]:
    return {
        "thresholds": {"default": 0.5, "screening": 0.4},
        "selected_model": "logistic",
    }


def _fitted_pipeline() -> object:
    return build_pipeline("logistic", seed=42).fit(
        _features(), np.arange(20) % 2
    )


def test_artifact_round_trip_preserves_probability(tmp_path: Path) -> None:
    pipeline = _fitted_pipeline()
    record = _record()
    record["oldpeak"] = 1.7
    before = pipeline.predict_proba(pd.DataFrame([record]))[0, 1]
    artifact_path = save_artifact(pipeline, _metadata(), tmp_path)
    metadata_path = tmp_path / "metadata.json"

    restored = load_artifact(artifact_path, metadata_path)
    after = restored.predict_proba(pd.DataFrame([record]))[0, 1]

    assert after == pytest.approx(before, abs=1e-15)


def test_artifact_bytes_are_deterministic(tmp_path: Path) -> None:
    pipeline = _fitted_pipeline()

    first = save_artifact(pipeline, _metadata(), tmp_path / "first")
    second = save_artifact(pipeline, _metadata(), tmp_path / "second")

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()


def test_load_rejects_unexpected_trusted_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = save_artifact(_fitted_pipeline(), _metadata(), tmp_path)
    monkeypatch.setattr(
        artifacts.skops_io,
        "get_untrusted_types",
        lambda **_: ["malicious.Payload"],
    )

    with pytest.raises(ArtifactValidationError, match="untrusted.*malicious.Payload"):
        load_artifact(artifact_path, tmp_path / "metadata.json")


def test_load_rejects_model_hash_mismatch(tmp_path: Path) -> None:
    artifact_path = save_artifact(_fitted_pipeline(), _metadata(), tmp_path)
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")

    with pytest.raises(ArtifactValidationError, match="model hash mismatch"):
        load_artifact(artifact_path, tmp_path / "metadata.json")


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("package_version", "999.0", "package-version mismatch"),
        ("schema", {"features": []}, "feature-schema mismatch"),
    ],
)
def test_load_rejects_version_or_schema_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    artifact_path = save_artifact(_fitted_pipeline(), _metadata(), tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata["metadata_sha256"] = artifacts._metadata_hash(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match=message):
        load_artifact(artifact_path, metadata_path)


def test_nonlinear_final_model_is_calibrated_and_serializable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _features(30)
    target = pd.Series(np.arange(30) % 2, name="disease_present")
    development = CohortData(
        Cohort.CLEVELAND, features, target, target.rename("num")
    )
    selection = Selection("gradient_boosting", "test", 0.5, 0.4)
    monkeypatch.setattr(
        training,
        "parameter_grid",
        lambda _: {
            "model__learning_rate": [0.1],
            "model__n_estimators": [5],
            "model__max_depth": [1],
        },
    )

    fitted = train_final(
        development,
        selection,
        ExperimentConfig(inner_splits=2),
    )
    before = fitted.predict_proba(features.iloc[[0]])[0, 1]
    artifact_path = save_artifact(fitted, _metadata(), tmp_path)
    restored = load_artifact(artifact_path, tmp_path / "metadata.json")

    assert "calibrated_model" in fitted.named_steps
    assert restored.predict_proba(features.iloc[[0]])[0, 1] == pytest.approx(before)


def test_predict_accepts_raw_thirteen_feature_record(tmp_path: Path) -> None:
    pipeline = _fitted_pipeline()
    save_artifact(pipeline, _metadata(), tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    prediction = predict_record(pipeline, metadata, _record())

    payload = asdict(prediction)
    assert 0 <= payload["probability"] <= 1
    assert payload["default_prediction"] in {0, 1}
    assert payload["screening_prediction"] in {0, 1}
    assert payload["default_threshold"] == 0.5
    assert payload["screening_threshold"] == 0.4
    assert "not medical advice" in payload["disclaimer"].lower()


@pytest.mark.parametrize(
    "record, message",
    [
        ({key: value for key, value in _record().items() if key != "oldpeak"}, "oldpeak"),
        ({**_record(), "cp": 9}, "cp.*9"),
    ],
)
def test_predict_rejects_missing_or_invalid_feature(
    record: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((ArtifactValidationError, ValueError), match=message):
        predict_record(_fitted_pipeline(), _metadata(), record)
