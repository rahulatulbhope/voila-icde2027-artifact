# Outputs

This directory documents the output layout for the reproducibility package.

## Tracked outputs

- `scripts/results_final/` — compact JSON/CSV result records and analysis
  figures.
- `ICDE_2027/paper/icde2027_author_edited.pdf` — submission PDF only.

## Generated locally and ignored

- `results/` — experiment outputs and caches.
- `model_artifacts/` and `scripts/model_artifacts/` — fitted models and
  calibrators.
- `raw_data/` — downloaded datasets and measured payloads.
- LaTeX/build intermediates and temporary review files.

See `ICDE_2027/artifact/manifest.json` for the external bundle link and
`ICDE_2027/artifact/restore_artifacts.ps1` for restoration.
