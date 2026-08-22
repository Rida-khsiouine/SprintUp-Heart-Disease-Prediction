"""Dataset acquisition and immutable-file verification."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd


class Cohort(StrEnum):
    """Supported UCI processed heart-disease cohorts."""

    CLEVELAND = "cleveland"
    HUNGARY = "hungary"
    SWITZERLAND = "switzerland"
    VA = "va"


class DataIntegrityError(RuntimeError):
    """Raised when source data is missing or does not match its manifest."""


def _load_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise DataIntegrityError(f"Dataset manifest is missing: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _verify_payload(payload: bytes, spec: dict[str, Any], *, source: str) -> None:
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != spec["sha256"]:
        raise DataIntegrityError(
            f"Dataset checksum mismatch for {source}: expected {spec['sha256']}, "
            f"received {actual_hash}"
        )
    if len(payload) != spec["bytes"]:
        raise DataIntegrityError(
            f"Dataset byte-size mismatch for {source}: expected {spec['bytes']}, "
            f"received {len(payload)}"
        )
    rows = sum(bool(line.strip()) for line in payload.splitlines())
    if rows != spec["rows"]:
        raise DataIntegrityError(
            f"Dataset row-count mismatch for {source}: expected {spec['rows']}, received {rows}"
        )


def _verify_file(path: Path, spec: dict[str, Any]) -> None:
    if not path.exists():
        raise DataIntegrityError(
            f"Dataset file is missing: {path}. Run `heart-disease fetch-data`."
        )
    _verify_payload(path.read_bytes(), spec, source=str(path))


def verify_all(data_dir: Path) -> dict[Cohort, Path]:
    """Verify every manifest cohort and return its immutable raw path."""

    manifest = _load_manifest(data_dir)
    paths: dict[Cohort, Path] = {}
    for cohort_name, spec in manifest["cohorts"].items():
        cohort = Cohort(cohort_name)
        path = data_dir / "raw" / spec["filename"]
        _verify_file(path, spec)
        paths[cohort] = path
    return paths


def fetch_all(data_dir: Path, force: bool = False) -> dict[Cohort, Path]:
    """Fetch all official cohorts, validating bytes before atomic replacement."""

    manifest = _load_manifest(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[Cohort, Path] = {}

    for cohort_name, spec in manifest["cohorts"].items():
        cohort = Cohort(cohort_name)
        target = raw_dir / spec["filename"]
        if target.exists() and not force:
            _verify_file(target, spec)
            paths[cohort] = target
            continue

        try:
            with urllib.request.urlopen(spec["url"], timeout=30) as response:
                payload = response.read()
        except OSError as exc:
            raise DataIntegrityError(
                f"Could not download {cohort.value} from {spec['url']}: {exc}"
            ) from exc

        _verify_payload(payload, spec, source=spec["url"])
        with tempfile.NamedTemporaryFile(dir=raw_dir, delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        temporary_path.replace(target)
        paths[cohort] = target

    return paths


def read_raw_cohort(cohort: Cohort, data_dir: Path) -> pd.DataFrame:
    """Read a verified cohort using the manifest's canonical column names."""

    manifest = _load_manifest(data_dir)
    spec = manifest["cohorts"][cohort.value]
    path = data_dir / "raw" / spec["filename"]
    _verify_file(path, spec)
    return pd.read_csv(path, names=manifest["columns"], na_values="?")
