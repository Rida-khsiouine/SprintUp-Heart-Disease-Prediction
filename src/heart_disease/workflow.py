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
from heart_disease.external import Thresholds, evaluate_external
from heart_disease.models import ExperimentConfig
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


def reproduce_study(
    *,
    profile: Profile,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run the modeling path and persist machine-readable core outputs."""

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
    default_metrics = classification_metrics(
        selected_oof["truth"].to_numpy(),
        selected_oof["probability"].to_numpy(),
        threshold=thresholds.default,
    )
    screening_metrics = classification_metrics(
        selected_oof["truth"].to_numpy(),
        selected_oof["probability"].to_numpy(),
        threshold=thresholds.screening,
    )
    metrics: dict[str, object] = {
        "profile": profile,
        "selection": asdict(selection),
        "development": {
            "n": len(development.target),
            "prevalence": float(development.target.mean()),
            "default": default_metrics,
            "screening": screening_metrics,
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
            }
            for name, result in results.items()
        },
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

    artifact_path = save_artifact(
        fitted,
        {
            "profile": profile,
            "selected_model": selection.model_name,
            "selected_parameters": fitted.heart_disease_parameters_,
            "selection_reason": selection.reason,
            "thresholds": asdict(thresholds),
            "seed": config.seed,
            "development_cohort": Cohort.CLEVELAND.value,
            "development_rows": len(development.target),
        },
        artifact_dir,
    )
    return {
        "artifact": artifact_path,
        "metadata": artifact_dir / "metadata.json",
        "metrics": metrics_path,
        "external_validation": external_path,
    }
