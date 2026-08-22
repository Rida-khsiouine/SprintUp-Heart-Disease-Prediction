"""Command-line interface for data, reproduction, and inference workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from heart_disease.artifacts import load_artifact, predict_record
from heart_disease.data import Cohort, fetch_all
from heart_disease.validation import load_cohort
from heart_disease.workflow import reproduce_study


def _fetch(args: argparse.Namespace) -> int:
    paths = fetch_all(args.data_dir, force=args.force)
    print(json.dumps({cohort.value: str(path) for cohort, path in paths.items()}))
    return 0


def _validate(args: argparse.Namespace) -> int:
    summaries = {}
    for cohort in Cohort:
        data = load_cohort(cohort, args.data_dir)
        summaries[cohort.value] = {
            "rows": len(data.features),
            "positive_count": int(data.target.sum()),
            "duplicate_count": data.duplicate_count,
            "missing_feature_values": int(sum(data.missing_counts.values())),
        }
    print(json.dumps(summaries, sort_keys=True))
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    paths = reproduce_study(
        profile=args.profile,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}))
    return 0


def _predict(args: argparse.Namespace) -> int:
    record = json.loads(args.input.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    pipeline = load_artifact(args.artifact, args.metadata)
    prediction = predict_record(pipeline, metadata, record)
    print(json.dumps(asdict(prediction), sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heart-disease")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch-data")
    fetch.add_argument("--data-dir", type=Path, default=Path("data"))
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(handler=_fetch)

    validate = subparsers.add_parser("validate-data")
    validate.add_argument("--data-dir", type=Path, default=Path("data"))
    validate.set_defaults(handler=_validate)

    reproduce = subparsers.add_parser("reproduce")
    reproduce.add_argument("--profile", choices=("smoke", "full"), required=True)
    reproduce.add_argument("--data-dir", type=Path, default=Path("data"))
    reproduce.add_argument("--output-dir", type=Path, default=Path("."))
    reproduce.set_defaults(handler=_reproduce)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument(
        "--artifact", type=Path, default=Path("artifacts/model.skops")
    )
    predict.add_argument(
        "--metadata", type=Path, default=Path("artifacts/metadata.json")
    )
    predict.set_defaults(handler=_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
