# Inputs

The repository expects several large or private inputs that are intentionally kept outside Git. Restore them with the same relative paths under the repository root.

## Required for the main offline runner

`files/` contains cached dataset/model CSV outputs consumed by `scripts/run_all_experiments.py`. These files hold VLM answers, probabilities, and log-probability traces; the runner does not call a VLM during analysis.

## Required for PerCom reproduction

- `raw_data/` — the official FloodNet image archive and extracted images used for measured payloads.
- `results/percom2027/` and `results/percom2027_final/` — frozen query-level results, common intersections, and measured-size summaries referenced by the handoff.
- `model_artifacts/` and `scripts/model_artifacts/` — saved router/calibration artifacts for reuse.
- `results/cache/` — optional router/runtime caches.

The exact expected files, populations, hashes, and sizes are recorded in `PERCOM_2027_RUN_HANDOFF.md` and `VOILA_Percom/generated/artifact_manifest.json`.

## Optional model-generation inputs

`HF_TOKEN` may be set for gated Hugging Face models. `AZURE_OPENAI_API_KEY` may be set for the Azure OpenAI router baseline. Local file fallback is documented in `secrets/README.md`; the key file is ignored.

## Restore policy

Do not place datasets or credentials in this repository for convenience. Keep them in a secured external artifact store and restore them only into the ignored paths above when running experiments.
