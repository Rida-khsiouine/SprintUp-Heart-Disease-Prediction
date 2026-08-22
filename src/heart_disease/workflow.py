"""End-to-end deterministic study orchestration used by the CLI."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from heart_disease import evaluation
from heart_disease.artifacts import save_artifact
from heart_disease.data import Cohort
from heart_disease.evaluation import average_patient_predictions, classification_metrics
from heart_disease.external import (
    Thresholds,
    bootstrap_metric_intervals,
    evaluate_external,
)
from heart_disease.models import ExperimentConfig
from heart_disease.reporting import (
    build_experiment_manifest,
    generate_evidence_outputs,
    sync_readme_results,
)
from heart_disease.training import train_final
from heart_disease.validation import load_cohort

Profile = Literal["smoke", "full"]


def config_for_profile(profile: Profile) -> ExperimentConfig:
    if profile == "smoke":
        return ExperimentConfig(
            outer_splits=2,
            outer_repeats=1,
            inner_splits=2,
            bootstrap_iterations=20,
        )
    if profile == "full":
        return ExperimentConfig()
    raise ValueError(f"Unknown reproduction profile: {profile}")


def _metric_with_intervals(
    truth,
    probability,
    *,
    threshold: float,
    config: ExperimentConfig,
) -> dict[str, object]:
    return {
        **classification_metrics(truth, probability, threshold=threshold),
        "confidence_intervals": bootstrap_metric_intervals(
            truth,
            probability,
            threshold=threshold,
            iterations=config.bootstrap_iterations,
            seed=config.seed,
        ),
    }


def _external_summary(external_metrics) -> list[dict[str, object]]:
    rows = []
    default_rows = external_metrics[external_metrics["threshold_name"] == "default"]
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier_score",
    )
    for _, source in default_rows.iterrows():
        confidence_intervals = {
            metric: {
                "low": float(source[f"{metric}_ci_low"]),
                "high": float(source[f"{metric}_ci_high"]),
            }
            for metric in metric_names
        }
        rows.append(
            {
                "cohort": str(source["cohort"]),
                "n": int(source["n"]),
                "prevalence": float(source["prevalence"]),
                **{metric: float(source[metric]) for metric in metric_names},
                "confidence_intervals": confidence_intervals,
            }
        )
    return rows


def reproduce_study(
    *,
    profile: Profile,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run the study and persist synchronized evidence and inference artifacts."""

    config = config_for_profile(profile)
    development = load_cohort(Cohort.CLEVELAND, data_dir)
    results = evaluation.run_nested_cv(development, config)
    selection = evaluation.select_candidate(results)
    fitted = train_final(development, selection, config)
    thresholds = Thresholds(
        default=selection.default_threshold,
        screening=selection.screening_threshold,
    )
    external_cohorts = tuple(
        load_cohort(cohort, data_dir)
        for cohort in (Cohort.HUNGARY, Cohort.SWITZERLAND, Cohort.VA)
    )
    external_metrics = evaluate_external(
        fitted, external_cohorts, thresholds, config
    )

    selected_oof = average_patient_predictions(
        results[selection.model_name].predictions
    )
    oof_truth = selected_oof["truth"].to_numpy(dtype=int)
    oof_probability = selected_oof["probability"].to_numpy(dtype=float)
    metrics: dict[str, object] = {
        "profile": profile,
        "selection": asdict(selection),
        "development": {
            "n": len(development.target),
            "prevalence": float(development.target.mean()),
            "default": _metric_with_intervals(
                oof_truth,
                oof_probability,
                threshold=thresholds.default,
                config=config,
            ),
            "screening": _metric_with_intervals(
                oof_truth,
                oof_probability,
                threshold=thresholds.screening,
                config=config,
            ),
        },
        "candidates": {
            name: {
                "mean_roc_auc": float(result.fold_metrics["roc_auc"].mean()),
                "std_roc_auc": float(result.fold_metrics["roc_auc"].std(ddof=1)),
                "mean_balanced_accuracy": float(
                    result.fold_metrics["balanced_accuracy"].mean()
                ),
                "mean_brier_score": float(
                    result.fold_metrics["brier_score"].mean()
                ),
                "folds": len(result.fold_metrics),
                "fold_metrics": json.loads(
                    result.fold_metrics.to_json(orient="records")
                ),
                "best_parameters": list(result.best_parameters),
            }
            for name, result in results.items()
        },
        "external": _external_summary(external_metrics),
    }

    reports_dir = output_dir / "reports"
    artifact_dir = output_dir / "artifacts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    external_path = reports_dir / "external-validation.csv"
    external_metrics.to_csv(external_path, index=False, lineterminator="\n")

    data_manifest = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    data_sha256 = {
        name: specification["sha256"]
        for name, specification in data_manifest["cohorts"].items()
    }
    parameters = fitted.heart_disease_parameters_
    artifact_path = save_artifact(
        fitted,
        {
            "profile": profile,
            "selected_model": selection.model_name,
            "selected_parameters": parameters,
            "selection_reason": selection.reason,
            "thresholds": asdict(thresholds),
            "seed": config.seed,
            "development_cohort": Cohort.CLEVELAND.value,
            "development_rows": len(development.target),
            "data_sha256": data_sha256,
        },
        artifact_dir,
    )
    artifact_metadata = json.loads(
        (artifact_dir / "metadata.json").read_text(encoding="utf-8")
    )
    manifest = build_experiment_manifest(
        profile=profile,
        config=config,
        data_manifest=data_manifest,
        artifact_metadata=artifact_metadata,
        selection=selection,
        parameters=parameters,
    )
    manifest_path = reports_dir / "experiment-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    evidence_paths = generate_evidence_outputs(
        reports_dir=reports_dir,
        development=development,
        external_cohorts=external_cohorts,
        fitted=fitted,
        results=results,
        selection=selection,
        thresholds=thresholds,
    )
    readme_path = output_dir / "README.md"
    if readme_path.is_file():
        sync_readme_results(readme_path, metrics)
    return {
        "artifact": artifact_path,
        "metadata": artifact_dir / "metadata.json",
        "metrics": metrics_path,
        "external_validation": external_path,
        "experiment_manifest": manifest_path,
        **evidence_paths,
    }
