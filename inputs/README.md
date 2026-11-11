# Inputs

Large and private inputs are intentionally kept outside Git. Restore them at
the paths below when running the offline experiments.

## Required inputs

- `files/` — cached dataset/model CSV outputs consumed by the runners.
- `raw_data/` — optional source imagery and payload inputs.
- `model_artifacts/` and `scripts/model_artifacts/` — saved router and
  calibration artifacts.
- `results/` — externally supplied result snapshots and runtime caches when a
  workflow requires them.

Optional model generation may require `HF_TOKEN` or
`AZURE_OPENAI_API_KEY`. Keep credentials in environment variables or another
secure store; never commit them.

## Restore policy

Use the Drive bundle and `ICDE_2027/artifact/restore_artifacts.ps1` to restore
ignored paths. Do not place datasets, checkpoints, or credentials in this
repository for convenience.
