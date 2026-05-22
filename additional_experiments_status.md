# Additional Experiments Status

This file maps `additional_exp.md` to the current repository state and the new
standalone runner:

```powershell
python scripts\run_additional_experiments.py --experiments all --output_dir scripts\results_final --model_artifact_dir model_artifacts\additional
```

The new runner does not modify `scripts/run_all_experiments.py`.

## Already Complete In `scripts/results_final`

| Item | Status | Location |
|---|---:|---|
| Main VOILA | Complete, 26/26 | `scripts/results_final/main/` |
| VOILA-CDF | Complete, 26/26 | `scripts/results_final/voila_cdf/` |
| Logprob cascade | Complete, 26/26 | `scripts/results_final/baselines/logprob_cascade/` |
| GPT-4o LLM router | Complete, 26/26 | `scripts/results_final/baselines/llm_router/` |
| BERT router | Complete, 26/26 | `scripts/results_final/baselines/bert_router/` |
| Oracle | Complete, 26/26 | `scripts/results_final/oracle/` |
| VQAv2/GQA OOD transfer | Complete, 12/12 | `scripts/results_final/ood/` |
| LoCoMo transfer | Complete, 2/2 | `scripts/results_final/locomo/` |

## Covered By `scripts/run_additional_experiments.py`

| `additional_exp.md` item | Runner experiment | Output |
|---|---|---|
| EXP-1 feature group ablation with OOD story | `feature_ood` | `scripts/results_final/additional/feature_ood/` and `additional/tables/feature_ood_table.csv` |
| EXP-1 TF-IDF vocab size ablation | `tfidf_vocab` | `scripts/results_final/additional/tfidf_vocab/` and `additional/tables/tfidf_vocab_table.csv` |
| EXP-2 ECE/Brier before and after isotonic calibration | `ece_calibration` | `scripts/results_final/additional/ece_calibration/` and `additional/tables/ece_calibration_table.csv` |
| FIG-3 reliability diagram | `reliability_diagram` or `ece_calibration` | `scripts/results_final/additional/figures/reliability_*.pdf` |
| EXP-3 extended OOD transfer | `extended_ood` | `scripts/results_final/additional/ood_extended/` and `additional/tables/ood_extended_table.csv` |
| EXP-4 oracle gap decomposition | `oracle_decomp` | `scripts/results_final/additional/oracle_decomp/` and `additional/tables/oracle_decomp_table.csv` |
| EXP-5 FloodNet question-type analysis | `floodnet_analysis` | `scripts/results_final/additional/tables/floodnet_question_type_accuracy.csv` |
| EXP-7 cascade degeneracy analysis | `cascade_degeneracy` | `scripts/results_final/additional/cascade_degeneracy/` and `additional/tables/cascade_degeneracy_table.csv` |
| FIG-1 fidelity saturation curves | `fidelity_saturation` | `scripts/results_final/additional/figures/fidelity_saturation_*.pdf` |
| FIG-2 ROC escalation decision | `roc_escalation` | `scripts/results_final/additional/figures/roc_escalation_decision.pdf` |
| FIG-4 routing words | `routing_words` | `scripts/results_final/additional/figures/routing_words_*.pdf` |
| FIG-5 iso-accuracy contour | `iso_accuracy_contour` | `scripts/results_final/additional/figures/iso_accuracy_contour_*.pdf` |

## Manual Or Partially Covered

| Item | Status | Why |
|---|---:|---|
| EXP-5 VLM internal resolution cap check | Manual | Requires model documentation/config inspection and possibly raw preprocessing code, not just the offline CSVs. |
| EXP-5 visual inspection of FloodNet examples | Manual | Requires raw image files and qualitative inspection. The runner computes the quantitative question-type piece. |
| FIG-6 redesigned overview figure | Manual/design | Needs actual selected example images and paper figure design choices. |
| Unified figure color scheme | Partially covered | New runner uses a shared `FIDELITY_COLORS` map for generated figures. Existing paper figures may still need visual cleanup. |
| CARES and Azizian citations | Manual paper edit | Bibliography/paper source files are not part of the experiment runner. |

## Recommended Paper Run

For the full paper-support suite:

```powershell
python scripts\run_additional_experiments.py --experiments all --output_dir scripts\results_final --model_artifact_dir model_artifacts\additional
```

For just the blocking quantitative experiments:

```powershell
python scripts\run_additional_experiments.py --experiments feature_ood tfidf_vocab ece_calibration extended_ood oracle_decomp tables --output_dir scripts\results_final --model_artifact_dir model_artifacts\additional
```

For reuse/checkpointing after an interrupted run:

```powershell
python scripts\run_additional_experiments.py --experiments all --reuse_artifacts --output_dir scripts\results_final --model_artifact_dir model_artifacts\additional
```
