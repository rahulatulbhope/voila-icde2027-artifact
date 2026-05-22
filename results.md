# VOILA Results Summary

This file explains the current experiments under `scripts/results_final`, what
each method is doing, where to find its outputs, and how to read the latest
numbers. Costs are normalized so full fidelity has cost `100`.

## Current Status

| Experiment | Status | Result path |
|---|---:|---|
| VOILA main | 26/26 dataset-model pairs | `scripts/results_final/main/{dataset}/{model}/voila_results.json` |
| VOILA-CDF | 26/26 | `scripts/results_final/voila_cdf/{dataset}/{model}/results.json` |
| Logprob cascade | 26/26 | `scripts/results_final/baselines/logprob_cascade/{dataset}/{model}/results.json` |
| GPT-4o LLM router | 26/26 | `scripts/results_final/baselines/llm_router/{dataset}/{model}/results.json` |
| Oracle | 26/26 | `scripts/results_final/oracle/{dataset}/{model}/results.json` |
| OOD transfer | 12/12 | `scripts/results_final/ood/{source}_to_{target}/{model}/results.json` |
| LoCoMo transfer | 2/2 | `scripts/results_final/locomo/{model}/results.json` |
| Calibration ablations | 12/12 | `scripts/results_final/ablations/calibration/{dataset}/{model}/results.json` |
| Predictor ablations | 2/2 | `scripts/results_final/ablations/predictor/{model}/{dataset}.json` |
| Feature ablations | 1/1 | `scripts/results_final/ablations/features/vqav2/pixtral/results.json` |
| BERT router | 26/26 | `scripts/results_final/baselines/bert_router/{dataset}/{model}/results.json` |
| Tables | Present | `scripts/results_final/tables/` |

All planned experiment outputs are now present under `scripts/results_final`.
The GPT-4o router labels were collected once per dataset and then reused across
models. Some LLM-router sample counts differ by model because the source CSVs
for those dataset-model pairs contain fewer rows than the dataset-level label
store.

## Main VOILA

**Technique.** VOILA trains question-only predictors for each fidelity. For a
query, it estimates the probability that each fidelity will answer correctly,
calibrates those probabilities with isotonic regression, then escalates only
when the expected accuracy gain is worth the additional normalized cost.

**Intuition.** Instead of running the VLM repeatedly, VOILA learns which kinds
of questions need more visual detail. It spends full-resolution cost mostly on
queries where low-fidelity representations are predicted to fail.

**Overall result.**

| Metric | Value |
|---|---:|
| VOILA accuracy | 66.67 |
| Full-fidelity accuracy | 71.98 |
| Gap to full | 5.31 |
| Average cost | 27.29 |
| Cost reduction | 72.71% |
| Gain over caption-only | 17.04 |

**By dataset.**

| Dataset | VOILA Acc | Full Acc | Gap to Full | Cost | Cost Reduction |
|---|---:|---:|---:|---:|---:|
| FloodNet | 63.48 | 60.13 | -3.34 | 19.31 | 80.69% |
| GQA | 59.56 | 70.19 | 10.63 | 27.92 | 72.08% |
| LoCoMo | 48.50 | 59.82 | 11.32 | 28.00 | 72.00% |
| TextVQA | 84.82 | 86.63 | 1.82 | 26.54 | 73.46% |
| VQAv2 | 64.88 | 75.02 | 10.14 | 35.14 | 64.86% |

**Interpretation.** The headline is strong cost reduction with moderate accuracy
loss overall. FloodNet is especially strong: VOILA beats full fidelity on
average while reducing cost by about 81%. TextVQA is also strong, with only a
small accuracy gap. GQA, VQAv2, and LoCoMo are the datasets where the current
tradeoff is more aggressive.

## VOILA-CDF

**Technique.** VOILA-CDF still uses the same calibrated per-fidelity predictors,
but it chooses pairwise escalation thresholds using CDF separation between
correct and incorrect examples at each cascade edge.

**Output.** `scripts/results_final/voila_cdf/{dataset}/{model}/results.json`

**Overall result.**

| Metric | Value |
|---|---:|
| Accuracy | 59.05 |
| Average cost | 26.36 |
| Cost reduction | 73.64% |

**Interpretation.** VOILA-CDF is not currently better than main VOILA. It has
roughly similar cost, but VOILA is +7.62 accuracy points better on average.
This should be presented as a variant or ablation, not the main method.

## Logprob Cascade Baseline

**Technique.** The cascade runs the VLM at low fidelity, checks the answer
logprob, and escalates if confidence is below a learned threshold. Thresholds
are learned per dataset-model pair and per fidelity transition using CDF
separation of correct vs incorrect logprobs.

**Cost accounting.** Cascade cost is cumulative. If it reaches multiple
fidelities, it pays for every fidelity visited, not just the final selected
fidelity. Each result JSON records:

```json
"cost_accounting": "cumulative_sum_of_all_fidelities_reached"
```

**Output.** `scripts/results_final/baselines/logprob_cascade/{dataset}/{model}/results.json`

**Overall result.**

| Metric | Value |
|---|---:|
| Accuracy | 56.21 |
| Average cost | 38.87 |
| Cost reduction | 61.13% |

**Interpretation.** This is a strong win for VOILA. Main VOILA is +10.46
accuracy points better and costs 11.58 less on average. The result supports the
argument that pre-retrieval question routing is better than post-retrieval
confidence cascading for this setting.

## GPT-4o LLM Router Baseline

**Technique.** GPT-4o reads the question and predicts whether it needs low,
medium, or high visual fidelity. Labels are collected once per dataset and then
reused across all VLM models for that dataset.

**Cache.**

The copied result JSONs record the dataset-level label store used for each run
in the `dataset_label_store` field. On this machine, the local cache folder also
contains the earlier VQAv2/GQA caches:

- `results/cache/llm_router/datasets/vqav2_ternary_n1000_seed42.json`
- `results/cache/llm_router/datasets/gqa_ternary_n1000_seed42.json`

**Outputs.** `scripts/results_final/baselines/llm_router/{dataset}/{model}/results.json`

**Table.** `scripts/results_final/tables/llm_router_table.csv`

**Scope.** Current run has all 26 dataset-model outputs. The number of evaluated
questions follows the available rows in each dataset-model CSV.

**Overall result.**

| Metric | Value |
|---|---:|
| Accuracy | 64.76 |
| Average cost | 51.10 |
| Cost reduction | 48.90% |

**By dataset.**

| Dataset | Accuracy | Cost | Cost Reduction |
|---|---:|---:|---:|
| FloodNet | 55.14 | 46.68 | 53.32% |
| GQA | 57.71 | 40.53 | 59.47% |
| LoCoMo | 48.50 | 43.28 | 56.72% |
| TextVQA | 84.45 | 70.63 | 29.37% |
| VQAv2 | 67.15 | 49.15 | 50.85% |

**Interpretation.** GPT-4o routing is competitive in accuracy but spends much
more than VOILA. Across all 26 pairs, it is 1.91 points below VOILA while
costing 23.81 more. TextVQA is the clearest example: GPT-4o routing nearly
matches VOILA accuracy, but at 70.63 cost versus VOILA's 26.54. This supports
the main paper framing that external LLM routing can be reasonable, but VOILA
gets the accuracy/cost tradeoff without an LLM call at test time.

## Oracle

**Technique.** The oracle chooses the cheapest fidelity that is actually correct
for each query, using hindsight. It is not deployable, but it gives an upper
bound on the possible fidelity-routing benefit.

**Output.** `scripts/results_final/oracle/{dataset}/{model}/results.json`

**Overall result.**

| Metric | Value |
|---|---:|
| Oracle accuracy | 86.52 |
| Oracle cost | 30.19 |

**Interpretation.** There is still a large gap between VOILA and the hindsight
oracle, so better routing policies could improve accuracy without necessarily
raising cost much. The oracle also shows that full fidelity is not always the
best or only correct representation.

## OOD Transfer

**Technique.** Train VOILA on one dataset and apply the router to another
dataset with the same VLM model. This tests whether the learned question-to-
fidelity policy transfers across dataset distributions.

**Output.** `scripts/results_final/ood/{source}_to_{target}/{model}/results.json`

**Overall result.**

| Metric | Value |
|---|---:|
| Accuracy | 54.18 |
| Average cost | 27.02 |
| Cost reduction | 72.98% |

**By transfer.**

| Transfer | Accuracy | Cost | Cost Reduction |
|---|---:|---:|---:|
| GQA to VQAv2 | 57.11 | 28.15 | 71.85% |
| VQAv2 to GQA | 51.26 | 25.89 | 74.11% |

**Interpretation.** OOD transfer keeps the cost low but loses accuracy. It is
useful as a robustness experiment, but not as the main deployment setting.

## LoCoMo Transfer

**Technique.** Train on VQAv2+GQA and evaluate on LoCoMo. This tests transfer to
a conversational/memory-oriented dataset.

**Output.** `scripts/results_final/locomo/{model}/results.json`

| Model | Accuracy | Cost | Cost Reduction |
|---|---:|---:|---:|
| Pixtral | 40.09 | 42.93 | 57.07% |
| Qwen7B | 50.13 | 30.35 | 69.65% |

**Interpretation.** LoCoMo is currently the weakest transfer setting. It should
be described as a harder OOD case rather than a main win.

## Supervised BERT Router

**Technique.** This baseline trains a supervised multiclass classifier
per dataset-model pair. The target label is the cheapest fidelity that answers
correctly. The default feature mode uses a local SentenceTransformer/BERT
encoder, with a TF-IDF+struct fallback for smoke tests.

**Output path.**
`scripts/results_final/baselines/bert_router/{dataset}/{model}/results.json`

**Table path.**
`scripts/results_final/tables/bert_router_table.csv`

**Current status.** Complete for all 26 dataset-model pairs. The reusable fold
artifacts are saved under `model_artifacts/bert_router`.

**Overall result.**

| Metric | Value |
|---|---:|
| Accuracy | 62.31 |
| Average cost | 38.94 |
| Cost reduction | 61.06% |

**By dataset.**

| Dataset | Accuracy | Cost | Cost Reduction |
|---|---:|---:|---:|
| FloodNet | 59.75 | 39.19 | 60.81% |
| GQA | 59.70 | 40.85 | 59.15% |
| LoCoMo | 49.76 | 43.89 | 56.11% |
| TextVQA | 71.92 | 33.12 | 66.88% |
| VQAv2 | 62.04 | 40.95 | 59.05% |

**Interpretation.** Directly classifying the cheapest sufficient fidelity is not
as strong as VOILA's calibrated per-fidelity value calculation. BERT router is
4.36 points below VOILA and costs 11.65 more on average. This strengthens the
argument that the VOI framing matters; simply predicting a fidelity label leaves
both accuracy and cost efficiency on the table.

## Current Paper Takeaways

1. VOILA is the strongest cost-aware method currently: 66.67 accuracy at 27.29
   cost, or 72.71% cost reduction.
2. VOILA beats logprob cascade clearly: +10.46 accuracy points and 11.58 lower
   cost.
3. VOILA beats GPT-4o routing overall: +1.91 accuracy points and 23.81 lower
   cost, while also avoiding an external LLM call at test time.
4. VOILA beats supervised BERT routing: +4.36 accuracy points and 11.65 lower
   cost.
5. VOILA-CDF is useful as an ablation but not as the primary method.
6. FloodNet and TextVQA are the strongest dataset stories. GQA, VQAv2, and
   LoCoMo need careful framing because the accuracy gap to full is larger.

## Useful Commands

Regenerate tables from existing JSONs:

```powershell
python scripts\run_all_experiments.py --experiments tables --output_dir scripts/results_final
```

Rerun the BERT-router experiment, reusing saved checkpoints when present:

```powershell
python scripts\run_all_experiments.py --experiments bert_router tables --output_dir scripts/results_final --model_artifact_dir model_artifacts --n_jobs 1 --bert_router_allow_download
```

Run the faster supervised-router fallback:

```powershell
python scripts\run_all_experiments.py --experiments bert_router tables --bert_router_features tfidf+struct --output_dir scripts/results_final --model_artifact_dir model_artifacts --n_jobs 4
```
