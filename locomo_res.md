# LoCoMo Results

This file documents the LoCoMo experiments in `scripts/results_final`, what each
result means, and where to find the exact JSON/CSV outputs.

## Evaluation Setup

LoCoMo is evaluated only for the two model pairs with available dataset files:

| Model key | CSV |
|---|---|
| `pixtral` | `files/locomo/locomo_compression_results_Pixtral-12B-2409_0_2275_1.csv` |
| `qwen7b` | `files/locomo/locomo_compression_results_Qwen2.5-VL-7B-Instruct_0_2275.csv` |

The experiment runner reads precomputed VQA answers and log probabilities from
these CSVs. No VLM calls are made during evaluation.

The LoCoMo fidelity tags differ slightly from the other datasets:

| Logical fidelity | LoCoMo answer/logprob tag |
|---|---|
| `caption` | `blip_caption` |
| `resize_32x32` | `resize_32x32` |
| `jpeg_q1` | `jpeg_extreme` |
| `jpeg_q10` | `jpeg_moderate` |
| `full` | `orig` |

The cost model normalizes full fidelity to `100`:

| Fidelity | Cost |
|---|---:|
| Caption | 7.58 |
| Resize 32x32 | 15.08 |
| JPEG Q1 / extreme | 37.83 |
| JPEG Q10 / moderate | 53.25 |
| Full / original | 100.00 |

Accuracy is computed with the fuzzy recall-style matching used across the
paper: normalized text comparison, numeric word normalization when relevant,
token-subset matching, and `rapidfuzz.partial_ratio >= 70`.

## Result Index

| Result group | What it evaluates | Path |
|---|---|---|
| In-domain VOILA | Train/test on LoCoMo folds | `scripts/results_final/main/locomo/{model}/voila_results.json` |
| In-domain VOILA-CDF | CDF thresholding variant on LoCoMo folds | `scripts/results_final/voila_cdf/locomo/{model}/results.json` |
| In-domain logprob cascade | LoCoMo fold-trained CDF logprob cascade | `scripts/results_final/baselines/logprob_cascade/locomo/{model}/results.json` |
| In-domain GPT-4o router | Question-only GPT-4o labels reused across models | `scripts/results_final/baselines/llm_router/locomo/{model}/results.json` |
| In-domain BERT router | LoCoMo fold-trained cheapest-sufficient classifier | `scripts/results_final/baselines/bert_router/locomo/{model}/results.json` |
| Oracle | Cheapest fidelity that is actually correct per example | `scripts/results_final/oracle/locomo/{model}/results.json` |
| VOILA transfer | Train on VQAv2+GQA, test on LoCoMo | `scripts/results_final/locomo/{model}/results.json` |
| OOD baselines | Baselines under VQAv2+GQA -> LoCoMo setting | `scripts/results_final/baselines/locomo_ood/{baseline}/{model}/results.json` |
| OOD baseline table | Flat summary for the new OOD baseline run | `scripts/results_final/tables/locomo_ood_baselines_table.csv` |

The focused OOD runner is:

```bash
python scripts/run_locomo_ood_baselines.py \
  --models pixtral qwen7b \
  --output_dir scripts/results_final \
  --model_artifact_dir model_artifacts \
  --save_models \
  --reuse_models \
  --n_llm_router_samples 0
```

## In-Domain LoCoMo

These are the standard LoCoMo fold-based results. Learned methods train and
tune on LoCoMo train/calibration folds and evaluate on held-out LoCoMo folds.

| Method | Pixtral Acc | Pixtral Cost | Pixtral Saving | Qwen7B Acc | Qwen7B Cost | Qwen7B Saving |
|---|---:|---:|---:|---:|---:|---:|
| VOILA | 41.15 | 25.38 | 74.62% | 55.86 | 30.62 | 69.38% |
| VOILA-CDF | 38.46 | 23.91 | 76.09% | 49.03 | 24.91 | 75.09% |
| Logprob cascade | 34.85 | 50.81 | 49.19% | 26.83 | 7.70 | 92.30% |
| GPT-4o router | 38.55 | 43.28 | 56.72% | 58.46 | 43.28 | 56.72% |
| BERT router | 41.98 | 43.26 | 56.74% | 57.53 | 44.51 | 55.49% |
| Full fidelity | 46.70 | 100.00 | 0.00% | 72.95 | 100.00 | 0.00% |
| Oracle | 72.64 | 46.84 | - | 85.37 | 37.05 | - |

Interpretation:

LoCoMo is a hard setting. For Pixtral, VOILA cuts cost sharply but gives up
about 5.55 accuracy points relative to full fidelity. For Qwen7B, full fidelity
is much stronger than the routers, but the oracle shows a large gap between
what is possible and what the learned routers currently recover. The oracle is
especially useful here: it reaches higher accuracy than full fidelity at far
lower cost, which means many examples are answerable at cheaper fidelities, but
the routing problem is difficult.

The GPT-4o and BERT routers are stronger than VOILA for Qwen7B in LoCoMo
in-domain evaluation, but they spend more cost. VOILA is more aggressive about
cost reduction.

## VQAv2+GQA -> LoCoMo VOILA Transfer

This experiment trains VOILA on combined VQAv2 and GQA data, tunes the
accuracy/cost tradeoff on source validation data, and evaluates directly on
LoCoMo. It is a harder OOD test than the regular in-domain LoCoMo fold setup.

Path:

`scripts/results_final/locomo/{model}/results.json`

| Model | Accuracy | Cost | Cost Reduction |
|---|---:|---:|---:|
| Pixtral | 40.09 | 42.93 | 57.07% |
| Qwen7B | 50.13 | 30.35 | 69.65% |

Interpretation:

Transfer is mixed. Pixtral transfer lands close to in-domain VOILA accuracy but
spends much more cost. Qwen7B transfer is lower than in-domain VOILA accuracy
but keeps similar cost. This should be framed as a domain-shift robustness test,
not as the main deployment result.

## VQAv2+GQA -> LoCoMo OOD Baselines

These results were added by `scripts/run_locomo_ood_baselines.py`. They put the
baselines into the same OOD setting as the LoCoMo transfer experiment:

- source training data: VQAv2 + GQA
- target test data: LoCoMo
- target models: `pixtral`, `qwen7b`
- output root: `scripts/results_final/baselines/locomo_ood`

Fixed fidelity and oracle do not train on VQAv2/GQA because they are target-only
evaluation references. Logprob cascade and BERT router are trained on
VQAv2+GQA. GPT-4o router is question-only zero-shot, so the runner reuses the
existing LoCoMo GPT-4o labels/results instead of making new API calls.

| Baseline | Pixtral Acc | Pixtral Cost | Pixtral Saving | Qwen7B Acc | Qwen7B Cost | Qwen7B Saving |
|---|---:|---:|---:|---:|---:|---:|
| Fixed caption | 32.51 | 7.58 | 92.42% | 26.78 | 7.58 | 92.42% |
| Fixed resize 32x32 | 19.07 | 15.08 | 84.92% | 40.09 | 15.08 | 84.92% |
| Fixed JPEG Q1 | 27.84 | 37.83 | 62.17% | 56.43 | 37.83 | 62.17% |
| Fixed JPEG Q10 | 43.70 | 53.25 | 46.75% | 70.18 | 53.25 | 46.75% |
| Fixed full | 46.70 | 100.00 | 0.00% | 72.95 | 100.00 | 0.00% |
| Oracle | 72.64 | 46.84 | - | 85.37 | 37.05 | - |
| Logprob cascade OOD | 36.04 | 66.09 | 33.91% | 26.87 | 7.62 | 92.38% |
| GPT-4o router | 38.55 | 43.28 | 56.72% | 58.46 | 43.28 | 56.72% |
| BERT router OOD | 40.22 | 58.03 | 41.97% | 54.19 | 45.57 | 54.43% |

### OOD Baseline Details

Fixed fidelity:

Always chooses a single LoCoMo fidelity. These rows are the easiest way to see
the raw fidelity-accuracy curve. For Qwen7B, JPEG Q10 is close to full fidelity
at roughly half the cost. For Pixtral, JPEG Q10 is also the best compressed
single-fidelity point.

Oracle:

Selects the cheapest fidelity that is correct for each example after seeing
ground truth. This is not deployable, but it is the upper bound for a router
that chooses among the available fidelity outputs. The oracle being much better
than full fidelity means LoCoMo contains many cases where lower fidelity answers
are correct even when full fidelity is not.

Logprob cascade OOD:

Learns pairwise CDF logprob thresholds on the source domain and applies them to
LoCoMo. Pixtral spends substantial cost but remains below fixed JPEG Q10.
Qwen7B collapses to very cheap routing and loses substantial accuracy. This is
evidence that raw logprob thresholds are not stable under this domain shift.

GPT-4o router:

Uses question-only GPT-4o labels with ternary routing. The labels are reused
across LoCoMo models. It does not train on VQAv2/GQA, so it is best described as
a zero-shot semantic routing baseline. It is cost-effective, but it does not
match full fidelity on LoCoMo.

BERT router OOD:

Trains a supervised cheapest-sufficient fidelity classifier on combined
VQAv2+GQA and applies it to LoCoMo. Artifacts are saved under
`model_artifacts/locomo_ood_bert_router/...` and are intentionally ignored by
Git. It beats GPT-4o for Pixtral but is weaker than GPT-4o for Qwen7B.

## Main Takeaways

LoCoMo is the weakest transfer dataset in the current result suite. This is
expected because its question distribution and answer style differ from regular
single-image VQA.

The most paper-useful LoCoMo story is not that every router beats full fidelity.
Instead, it is:

1. Full fidelity is expensive and not always the best answer source.
2. The oracle shows large routing headroom: 72.64% at cost 46.84 for Pixtral and
   85.37% at cost 37.05 for Qwen7B.
3. Current learned and prompted routers recover only part of that headroom under
   domain shift.
4. Source-trained routing transfers imperfectly, especially for logprob-based
   cascades, which makes LoCoMo a strong stress test for robust routing.

For the paper, use LoCoMo as the hard OOD evidence: VOILA and the baselines can
reduce cost, but cross-domain routing remains challenging and leaves clear room
for better domain-adaptive routers.
