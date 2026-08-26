# Legacy Audit: Before and After

The original notebook collection is preserved at annotated tag `legacy-v1`. The rebuild is independent package code; no notebook implementation was copied forward.

| Area | Legacy evidence | Rebuild control |
|---|---|---|
| Target | README described binary prediction while notebooks modeled raw labels 0–4. | One tested definition: `disease_present = (num > 0)`; raw severity is retained separately. |
| Precision | A blanket integer cast truncated decimal `oldpeak` values; 173 of 303 Cleveland values changed in the audit. | Numeric validation preserves `oldpeak` as float through loading, preprocessing, serialization, and inference. |
| Leakage | Cleaning, encoding, scaling, and feature selection occurred before the split. | All fit-dependent transforms are pipeline steps fitted inside the relevant training fold. |
| Evaluation | One non-stratified holdout was repeatedly reused for model comparison. | Identical repeated outer stratified folds evaluate every candidate; tuning occurs only in inner folds. |
| Generalization | Only the cleaned Cleveland subset was evaluated. | Cleveland is development-only; all three other hospitals are frozen external validations with uncertainty and shift evidence. |
| Artifact | Pickle referenced a notebook-local `__main__.CustomFeatureSelector` and did not own raw preprocessing. | End-to-end `skops` artifact, explicit type allowlist, deterministic bytes, schema/version/hash checks, and strict 13-field inference. |
| Reproducibility | Multiple notebooks duplicated stateful logic and depended on execution order. | Installable package, locked dependencies, CLI profiles, generated reports, thin executable notebook, and CI. |
| Claims | Metrics were presented without a model card or clinical limitation. | Model card, uncertainty, underpowered subgroup status, and explicit non-clinical use. |

## Why the old accuracy looked mediocre

The audit found that the notebooks unintentionally attempted a much harder five-class severity task while describing a binary task. On 297 complete Cleveland rows, the severity classes were highly imbalanced (160/54/35/35/13). A repeated stratified audit of that five-class formulation produced roughly 0.59 accuracy and 0.32 balanced accuracy. Those numbers were not evidence that “machine learning failed”; they exposed a mismatch between the stated problem, target construction, and evaluation.

The rebuild does not chase a guaranteed score. It corrects the question, prevents leakage, quantifies uncertainty, compares against a dummy, and reports external degradation. Final generated numbers live only in `reports/metrics.json`; the README reads from that file to prevent stale hand-edited claims.
