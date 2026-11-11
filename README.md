# VOILA ICDE 2027 artifact

This repository contains the reproducibility code, tests, compact result
records, and instructions for the VOILA ICDE 2027 artifact. Submission-paper
source files and unrelated conference drafts are intentionally not included.

## Contents

- `scripts/` — offline runners, analysis utilities, and optional model-output
  generation tools.
- `scripts/results_final/` — compact JSON/CSV result records and figures.
- `tests/` — regression tests for routing, splits, calibration, and analysis.
- `ICDE_2027/artifact/` — the Drive-link manifest and non-destructive restore
  helper.
- `ICDE_2027/paper/icde2027_author_edited.pdf` — the submission PDF only; no
  editable LaTeX source is included.
- `inputs/` and `outputs/` — expected external inputs and output inventory.

## Quick start

Create an environment and install the dependencies:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
```

The full experiment suite requires the ignored data and model bundle described
in `inputs/README.md`. Do not commit datasets, checkpoints, credentials, or
runtime caches.

## External artifact bundle

The supplemental bundle is linked in
`ICDE_2027/artifact/manifest.json`:

<https://drive.google.com/file/d/1i6orLyJELF1ZFj-bpSZmMcZqAW60SOV7/view?usp=drive_link>

After downloading and extracting the bundle, restore its ignored directories
with:

```powershell
pwsh -File ICDE_2027/artifact/restore_artifacts.ps1 `
  -DriveBundleRoot 'D:\path\to\downloaded\VOILA_ICDE_2027_artifacts'
```

The restore helper does not delete existing files.
