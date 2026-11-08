# Outputs

This directory documents output locations; the primary checked-in outputs remain beside the paper and scripts so existing paths stay reproducible.

## Committed paper-facing outputs

- `VOILA_Percom/generated/` — CSV summaries, LaTeX tables, figures, bootstrap summaries, and the artifact manifest.
- `VOILA_Percom/figures/` and `VOILA_Percom/plots/` — paper figures and plot PDFs/PNGs.
- `VOILA_Percom/percom2027.pdf` and `VOILA_Percom/percom2027_submission.pdf` — latest committed working and submission PDFs.
- `scripts/results_final/` — compact analysis results used by the existing paper workflow.

## Generated locally and ignored

- `results/` — experiment outputs and caches.
- `model_artifacts/` and `scripts/model_artifacts/` — fitted models and calibrators.
- `raw_data/` — downloaded datasets and measured payloads.
- `tmp/` and LaTeX build intermediates — review renders and transient build state.

## Reproducibility

Use `VOILA_Percom/generated/artifact_manifest.json` to verify the checked-in generated files. For a full rerun, follow `PERCOM_2027_RUN_HANDOFF.md`; do not infer missing inputs from the paper PDF alone.
