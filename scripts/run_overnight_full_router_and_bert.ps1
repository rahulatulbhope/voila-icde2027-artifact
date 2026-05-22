$ErrorActionPreference = "Stop"

# Overnight run:
# 1. GPT router on ALL available sampled questions for every configured dataset.
#    Labels are collected once per dataset and reused for every model.
# 2. Supervised BERT fidelity router for every dataset-model pair.
# 3. Regenerate tables.
#
# Notes:
# - --n_llm_router_samples 0 means use all rows/questions in each dataset CSV.
# - Existing GPT router labels and model artifacts are checkpoints; reruns resume.
# - Remove --bert_router_allow_download if the encoder is already cached and you
#   want to forbid network downloads.

$OutputDir = "scripts/results_final"
$ArtifactDir = "model_artifacts"
$Jobs = 4

python scripts/run_all_experiments.py `
  --experiments llm_router bert_router tables `
  --output_dir $OutputDir `
  --model_artifact_dir $ArtifactDir `
  --n_jobs $Jobs `
  --n_llm_router_samples 0 `
  --llm_router_max_completion_tokens 256 `
  --llm_router_retries 5 `
  --bert_router_allow_download
