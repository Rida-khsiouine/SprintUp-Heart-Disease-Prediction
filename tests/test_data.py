from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from heart_disease.data import Cohort, DataIntegrityError, fetch_all, verify_all


def _write_manifest(data_dir: Path, *, payload: bytes, url: str) -> None:
    manifest = {
        "dataset": {"name": "fixture", "doi": "fixture", "license": "fixture"},
        "columns": ["feature", "num"],
        "cohorts": {
            "cleveland": {
                "url": url,
                "filename": "processed.cleveland.data",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "rows": len(payload.splitlines()),
            }
        },
    }
    data_dir.mkdir(parents=True)
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_rejects_modified_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_manifest(data_dir, payload=b"expected\n", url="file:///unused")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir()
    (raw_dir / "processed.cleveland.data").write_bytes(b"tampered\n")

    with pytest.raises(DataIntegrityError, match="checksum"):
        verify_all(data_dir)


def test_fetch_does_not_overwrite_valid_file_without_force(tmp_path: Path) -> None:
    payload = b"kept,0\n"
    data_dir = tmp_path / "data"
    _write_manifest(data_dir, payload=payload, url="file:///source-does-not-exist")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir()
    target = raw_dir / "processed.cleveland.data"
    target.write_bytes(payload)

    paths = fetch_all(data_dir)

    assert paths == {Cohort.CLEVELAND: target}
    assert target.read_bytes() == payload


def test_manifest_covers_all_four_cohorts() -> None:
    data_dir = Path(__file__).parents[1] / "data"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))

    assert set(manifest["cohorts"]) == {cohort.value for cohort in Cohort}


def test_raw_files_match_committed_checksums() -> None:
    data_dir = Path(__file__).parents[1] / "data"

    paths = verify_all(data_dir)

    assert set(paths) == set(Cohort)


def test_raw_files_are_not_subject_to_line_ending_conversion() -> None:
    result = subprocess.run(
        ["git", "check-attr", "text", "--", "data/raw/processed.cleveland.data"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().endswith("text: unset")
