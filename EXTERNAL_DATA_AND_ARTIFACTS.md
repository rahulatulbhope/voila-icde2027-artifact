# External Data And Artifacts

This repository is intended to track code, experiment definitions, result
summaries, and small paper-facing result files. Large offline assets are
excluded from git and should be stored separately, for example in Google Drive.

## Excluded From Git

The `.gitignore` excludes:

- `files/`: dataset/model CSV outputs used by the offline experiments
- `model_artifacts/`: saved GBR, isotonic, BERT, and other fitted artifacts
- `scripts/model_artifacts/`: older/generated model artifacts
- `results/cache/`: GPT-4o router caches and other runtime caches
- old/throwaway result snapshots such as `results_old*/`, `results_cost100/`,
  and `results_llm_router_test/`
- local secrets in `secrets/*`, except `secrets/README.md`

## Suggested Google Drive Bundle

Upload these folders to Drive:

```text
files/
model_artifacts/
scripts/model_artifacts/
results/cache/
```

Optional old snapshots, if you want to preserve them:

```text
results_cost100/
results_llm_router_test/
results_old/
results_old_1/
results_old_2/
```

## Restoring A Fresh Clone

After cloning the private repo, copy the Drive folders back into the repo root
with the same relative paths:

```text
D:\VOILA_code\files
D:\VOILA_code\model_artifacts
D:\VOILA_code\scripts\model_artifacts
D:\VOILA_code\results\cache
```

Then rerun or reuse experiments normally. The runners expect those paths by
default.

## Secret Key

The Azure OpenAI key is not tracked. To run the GPT-4o router, create:

```text
secrets/azure_openai_api_key.txt
```

or set:

```text
AZURE_OPENAI_API_KEY
```
