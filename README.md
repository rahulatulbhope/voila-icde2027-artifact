# VOILA

VOILA (Value-of-Information Guided Fidelity Selection) predicts how much visual fidelity a multimodal question needs from the question text alone, then selects the lowest-cost representation that meets the configured utility target.

This repository contains the experiment code, reproducibility notes, paper sources, generated paper-facing artifacts, and the PerCom 2027 revision tooling.

## Repository map

- `scripts/run_all_experiments.py` — main offline experiment runner. It consumes cached CSV outputs under `files/` and does not invoke a VLM.
- `scripts/percom_experiments.py` — PerCom-specific payload, routing, mission-budget, CellReplay, and paper-artifact tooling.
- `scripts/generate_hf_vqa_outputs.py` — optional Hugging Face VQA output generation; model downloads and tokens are external.
- `tests/` — regression tests for routing, splits, calibration, and PerCom utilities.
- `VOILA_Percom/` — PerCom 2027 paper sources, figures, PDFs, generated tables, and build helper.
- `inputs/` and `outputs/` — the expected external inputs and the committed/generated output inventory.
- `EXTERNAL_DATA_AND_ARTIFACTS.md` — what is intentionally kept outside Git and how to restore it.
- `PERCOM_2027_RUN_HANDOFF.md` — current PerCom experiment state and continuation notes.

## Quick start

Create a Python environment and install the core dependencies listed in [`requirements.txt`](requirements.txt). Hugging Face generation has separate optional dependencies documented in `scripts/generate_hf_vqa_outputs.py`. Then run:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
python scripts/percom_experiments.py --help
```

The full experiment suite requires the ignored data/model bundle described in `inputs/README.md`. A smoke run, when the cached CSV inputs are present, is:

```powershell
python scripts/run_all_experiments.py --datasets vqav2 --models pixtral `
  --experiments main oracle cascade obs2_quant --max_rows 1000 --n_cv_folds 2
```

To build the PerCom paper, install a working LaTeX toolchain with `pdflatex` and `bibtex`, then run:

```powershell
pwsh -File VOILA_Percom/build_percom.ps1 -Mode submission
```

Use `-Mode working` for the extended working paper or `-Mode both` for both builds. The build writes LaTeX intermediates beside the source; those intermediates are ignored by Git.

## Data, secrets, and artifacts

Datasets, downloaded model checkpoints, runtime caches, API keys, and large raw imagery are deliberately not committed. Put them at the paths documented in `inputs/README.md`; use environment variables for credentials. Never commit `secrets/azure_openai_api_key.txt`, `.env` files, or downloaded datasets.

Small, paper-facing generated tables, figures, summaries, manifests, and PDFs under `VOILA_Percom/` are included so the checked-in paper can be inspected without rerunning expensive jobs. See `outputs/README.md` for the artifact inventory.

## Agent and contributor instructions

Read [`AGENTS.md`](AGENTS.md) before modifying the project. It records the source-of-truth files, test/build commands, artifact policy, and safety rules for automated contributors.
