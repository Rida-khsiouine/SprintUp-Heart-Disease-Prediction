"""Safe model serialization, integrity validation, and raw-record inference."""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import sklearn
import skops
import skops.io as skops_io
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from heart_disease import __version__
from heart_disease.data import Cohort
from heart_disease.validation import (
    CATEGORICAL_DOMAINS,
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    validate_raw_frame,
)

DISCLAIMER = (
    "Educational demonstration only; this prediction is not medical advice, "
    "a diagnosis, or a clinically validated decision aid."
)

ALLOWED_ESTIMATOR_TYPES = {
    Pipeline,
    ColumnTransformer,
    SimpleImputer,
    StandardScaler,
    OneHotEncoder,
    DummyClassifier,
    LogisticRegression,
    RandomForestClassifier,
    GradientBoostingClassifier,
    CalibratedClassifierCV,
}
ALLOWED_SKOPS_TYPES = {
    "numpy.dtype",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.model_selection._split.StratifiedKFold",
}


class ArtifactValidationError(ValueError):
    """Raised when an artifact or its metadata fails integrity checks."""


@dataclass(frozen=True)
class Prediction:
    probability: float
    default_prediction: int
    screening_prediction: int
    default_threshold: float
    screening_threshold: float
    model_version: str
    disclaimer: str


def _expected_schema() -> dict[str, object]:
    return {
        "features": list(FEATURE_COLUMNS),
        "target": "disease_present",
        "categorical_domains": {
            name: sorted(domain) for name, domain in CATEGORICAL_DOMAINS.items()
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonicalize_skops_archive(artifact_path: Path) -> None:
    """Remove process IDs and ZIP timestamps from a skops archive."""

    with zipfile.ZipFile(artifact_path, "r") as source:
        entries = {name: source.read(name) for name in source.namelist()}
    schema = json.loads(entries.pop("schema.json").decode("utf-8"))
    id_mapping: dict[str, int] = {}

    def collect_ids(value: object) -> None:
        if isinstance(value, dict):
            if "__id__" in value:
                identifier = str(value["__id__"])
                id_mapping.setdefault(identifier, len(id_mapping) + 1)
            for key in sorted(value):
                if key != "__id__":
                    collect_ids(value[key])
        elif isinstance(value, list):
            for item in value:
                collect_ids(item)

    collect_ids(schema)

    def replace_ids(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: (
                    id_mapping[str(item)]
                    if key == "__id__"
                    else replace_ids(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [replace_ids(item) for item in value]
        if isinstance(value, str) and value.endswith(".npy"):
            identifier = value.removesuffix(".npy")
            if identifier in id_mapping:
                return f"{id_mapping[identifier]}.npy"
        return value

    canonical_schema = json.dumps(
        replace_ids(schema),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    canonical_entries: dict[str, bytes] = {}
    for name, payload in entries.items():
        identifier = name.removesuffix(".npy")
        if not name.endswith(".npy") or identifier not in id_mapping:
            raise ArtifactValidationError(
                f"Cannot canonicalize unexpected skops archive member: {name}"
            )
        canonical_entries[f"{id_mapping[identifier]}.npy"] = payload
    canonical_entries["schema.json"] = canonical_schema

    with tempfile.NamedTemporaryFile(
        dir=artifact_path.parent,
        prefix="model-",
        suffix=".skops.tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_STORED) as target:
            ordered_names = sorted(
                canonical_entries,
                key=lambda name: (
                    name == "schema.json",
                    int(name.removesuffix(".npy"))
                    if name.endswith(".npy")
                    else 0,
                ),
            )
            for name in ordered_names:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                target.writestr(info, canonical_entries[name])
        temporary_path.replace(artifact_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _metadata_hash(metadata: Mapping[str, object]) -> str:
    payload = {
        key: value for key, value in metadata.items() if key != "metadata_sha256"
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_artifact(
    pipeline: Pipeline,
    metadata: dict[str, object],
    artifact_dir: Path,
) -> Path:
    """Serialize a fitted pipeline and write integrity-bound metadata."""

    check_is_fitted(pipeline)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "model.skops"
    skops_io.dump(pipeline, artifact_path)
    _canonicalize_skops_archive(artifact_path)

    protected = {
        "artifact_format",
        "model_version",
        "package_version",
        "model_sha256",
        "metadata_sha256",
        "schema",
        "library_versions",
        "disclaimer",
    }
    conflicts = sorted(protected.intersection(metadata))
    if conflicts:
        raise ArtifactValidationError(
            f"Caller metadata cannot override protected fields: {conflicts}"
        )
    complete: dict[str, object] = {
        "artifact_format": "skops",
        "model_version": __version__,
        "package_version": __version__,
        "model_sha256": _sha256(artifact_path),
        "schema": _expected_schema(),
        "library_versions": {
            "python": platform.python_version(),
            "heart_disease": __version__,
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": sklearn.__version__,
            "skops": skops.__version__,
        },
        "disclaimer": DISCLAIMER,
        **metadata,
    }
    complete["metadata_sha256"] = _metadata_hash(complete)
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(complete, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _validate_estimator_types(pipeline: object) -> None:
    if not isinstance(pipeline, Pipeline):
        raise ArtifactValidationError(
            f"Artifact root must be sklearn Pipeline, got {type(pipeline).__name__}"
        )
    estimators = {
        value
        for value in pipeline.get_params(deep=True).values()
        if isinstance(value, BaseEstimator)
    }
    estimators.add(pipeline)
    unexpected = sorted(
        {
            f"{type(estimator).__module__}.{type(estimator).__qualname__}"
            for estimator in estimators
            if type(estimator) not in ALLOWED_ESTIMATOR_TYPES
        }
    )
    if unexpected:
        raise ArtifactValidationError(f"Unexpected estimator types: {unexpected}")


def load_artifact(artifact_path: Path, metadata_path: Path) -> Pipeline:
    """Verify hashes, schema, versions, and trusted types before loading."""

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read artifact metadata: {exc}") from exc
    if metadata.get("metadata_sha256") != _metadata_hash(metadata):
        raise ArtifactValidationError("Artifact metadata hash mismatch")
    if metadata.get("package_version") != __version__:
        raise ArtifactValidationError(
            "Artifact package-version mismatch: "
            f"expected {__version__}, got {metadata.get('package_version')}"
        )
    if metadata.get("schema") != _expected_schema():
        raise ArtifactValidationError("Artifact feature-schema mismatch")
    if not artifact_path.is_file() or metadata.get("model_sha256") != _sha256(
        artifact_path
    ):
        raise ArtifactValidationError("Artifact model hash mismatch")

    untrusted = skops_io.get_untrusted_types(file=artifact_path)
    unexpected_untrusted = sorted(set(untrusted) - ALLOWED_SKOPS_TYPES)
    if unexpected_untrusted:
        raise ArtifactValidationError(
            f"Artifact contains untrusted types: {unexpected_untrusted}"
        )
    pipeline = skops_io.load(
        artifact_path,
        trusted=sorted(set(untrusted).intersection(ALLOWED_SKOPS_TYPES)),
    )
    _validate_estimator_types(pipeline)
    check_is_fitted(pipeline)
    return pipeline


def predict_record(
    pipeline: Pipeline,
    metadata: Mapping[str, object],
    record: Mapping[str, object],
) -> Prediction:
    """Validate one raw 13-feature record and return thresholded predictions."""

    missing = [feature for feature in FEATURE_COLUMNS if feature not in record]
    unexpected = [feature for feature in record if feature not in FEATURE_COLUMNS]
    if missing or unexpected:
        raise ArtifactValidationError(
            f"Input schema mismatch; missing={missing}, unexpected={unexpected}"
        )
    raw = pd.DataFrame(
        [{**{feature: record[feature] for feature in FEATURE_COLUMNS}, "num": 0}],
        columns=RAW_COLUMNS,
    )
    validate_raw_frame(raw, Cohort.CLEVELAND)
    features = raw.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    features["oldpeak"] = features["oldpeak"].astype(float)

    thresholds = metadata.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ArtifactValidationError("Metadata is missing decision thresholds")
    try:
        default_threshold = float(thresholds["default"])
        screening_threshold = float(thresholds["screening"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError("Metadata decision thresholds are invalid") from exc
    if not 0 <= default_threshold <= 1 or not 0 <= screening_threshold <= 1:
        raise ArtifactValidationError("Metadata thresholds must be between zero and one")

    probability = float(pipeline.predict_proba(features)[0, 1])
    return Prediction(
        probability=probability,
        default_prediction=int(probability >= default_threshold),
        screening_prediction=int(probability >= screening_threshold),
        default_threshold=default_threshold,
        screening_threshold=screening_threshold,
        model_version=str(metadata.get("model_version", __version__)),
        disclaimer=str(metadata.get("disclaimer", DISCLAIMER)),
    )
