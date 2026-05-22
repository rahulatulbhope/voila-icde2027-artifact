# Additional VOILA Results Guide

This file explains the additional experiments under
`scripts/results_final/additional`, what each result compares, where the files
are, and how to interpret the numbers for the paper. These experiments come
from `additional_exp.md` and are separate from the main experiment suite in
`results.md`.

All costs use the normalized fidelity cost model where `full = 100`.

## Output Map

| Experiment | What it answers | Main table |
|---|---|---|
| Feature OOD ablation | Which question features support in-domain and OOD routing? | `scripts/results_final/additional/tables/feature_ood_table.csv` |
| TF-IDF vocab size | Whether 500/1000/2000/5000 TF-IDF features matter | `scripts/results_final/additional/tables/tfidf_vocab_table.csv` |
| ECE/Brier calibration | Whether isotonic calibration improves probability reliability | `scripts/results_final/additional/tables/ece_calibration_table.csv` |
| Extended OOD | Whether VQA/GQA-trained VOILA transfers to TextVQA/FloodNet | `scripts/results_final/additional/tables/ood_extended_table.csv` |
| Oracle decomposition | What causes the VOILA-to-oracle gap | `scripts/results_final/additional/tables/oracle_decomp_table.csv` |
| FloodNet question types | Which FloodNet question types prefer which fidelity | `scripts/results_final/additional/tables/floodnet_question_type_accuracy.csv` |
| Cascade degeneracy | When logprob cascade never escalates or escalates too much | `scripts/results_final/additional/tables/cascade_degeneracy_table.csv` |
| Escalation ROC | Whether VOILA scores beat logprob for escalation decisions | `scripts/results_final/additional/tables/roc_escalation_table.csv` |
| Fidelity saturation | Where question types plateau across fidelities | `scripts/results_final/additional/tables/fidelity_saturation_table.csv` |
| Routing words | Which words influence low/full fidelity prediction | `scripts/results_final/additional/tables/routing_words_table.csv` |
| Iso-accuracy contour | Accuracy as a function of model scale and fidelity | `scripts/results_final/additional/tables/iso_accuracy_contour_data.csv` |

Figures are in:

`scripts/results_final/additional/figures/`

Model checkpoints for these additional experiments are under:

`model_artifacts/additional/`

## Completion Status

The additional run produced:

- `70` result JSON files under `scripts/results_final/additional`
- `15` table/index files under `scripts/results_final/additional/tables`
- PDF and PNG figures for reliability, ROC, fidelity saturation, routing words,
  and iso-accuracy contours

The only items from `additional_exp.md` that are not fully automated are the
manual/documentation pieces:

- VLM internal resolution-cap documentation for FloodNet
- Manual visual inspection of raw FloodNet examples
- Redesigning the paper overview figure
- Paper bibliography edits for the new related-work citations

## EXP-1: Feature Group OOD Ablation

**Files.**

- Table: `scripts/results_final/additional/tables/feature_ood_table.csv`
- Per-run JSONs:
  `scripts/results_final/additional/feature_ood/vqav2_to_all/{model}/{feature_group}/results.json`

**What it compares.** Models are trained on VQAv2 and evaluated on:

- VQAv2 held-out split, in-domain
- GQA, similar-domain OOD
- TextVQA, OCR-heavy OOD
- FloodNet, aerial/disaster OOD

The feature groups are:

- `tfidf_only`: TF-IDF unigrams only
- `qtype_only`: compact question-type flags
- `lexical_only`: lexical/statistical flags
- `qtype_lexical`: all structured flags, no TF-IDF
- `full`: TF-IDF plus structured flags

This was run for `pixtral` and `qwen7b`.

**Mean results across the two models.**

| Feature group | Target | Accuracy | Cost | Cost reduction |
|---|---|---:|---:|---:|
| `full` | VQAv2 | 65.00 | 40.70 | 59.30% |
| `full` | GQA | 53.58 | 36.50 | 63.50% |
| `full` | TextVQA | 72.90 | 42.32 | 57.68% |
| `full` | FloodNet | 35.27 | 26.32 | 73.68% |
| `qtype_only` | VQAv2 | 66.88 | 44.86 | 55.14% |
| `qtype_only` | GQA | 58.38 | 41.10 | 58.90% |
| `qtype_only` | TextVQA | 69.12 | 48.66 | 51.34% |
| `qtype_only` | FloodNet | 53.14 | 49.58 | 50.42% |
| `tfidf_only` | VQAv2 | 63.06 | 36.57 | 63.43% |
| `tfidf_only` | GQA | 52.39 | 30.87 | 69.13% |
| `tfidf_only` | TextVQA | 67.43 | 40.88 | 59.12% |
| `tfidf_only` | FloodNet | 42.70 | 34.60 | 65.40% |

**Best rows by target.**

| Target | Best model | Best feature group | Accuracy | Cost |
|---|---|---|---:|---:|
| VQAv2 | Pixtral | `qtype_only` | 68.77 | 51.94 |
| GQA | Pixtral | `qtype_only` | 65.28 | 48.16 |
| TextVQA | Qwen7B | `qtype_only` | 89.22 | 50.56 |
| FloodNet | Qwen7B | `tfidf_only` | 61.23 | 45.91 |

**Interpretation.** The cleanest story is that structured question-type cues
transfer surprisingly well. `qtype_only` is the best average performer on
VQAv2, GQA, and TextVQA, though it spends more. Pure TF-IDF is often cheaper
and remains useful, but it is less consistently OOD-robust. FloodNet is
different: the best single row is Qwen7B with `tfidf_only`, which suggests
dataset-specific lexical regularities in FloodNet questions are strong.

For the paper, this supports the feature story, but with nuance: question-type
features are not merely decorative; they can be competitive by themselves and
help explain cross-dataset transfer.

## EXP-1b: TF-IDF Vocabulary Size

**Files.**

- Table: `scripts/results_final/additional/tables/tfidf_vocab_table.csv`
- Per-run JSONs:
  `scripts/results_final/additional/tfidf_vocab/vqav2/{model}/vocab_{size}/results.json`

**What it compares.** VQAv2-only `tfidf_only` VOILA routing with vocabulary
sizes `500`, `1000`, `2000`, and `5000`, for Pixtral and Qwen7B.

| Model | Vocab size | Accuracy | Cost | Cost reduction |
|---|---:|---:|---:|---:|
| Pixtral | 500 | 65.17 | 43.82 | 56.18% |
| Pixtral | 1000 | 62.35 | 34.30 | 65.70% |
| Pixtral | 2000 | 65.50 | 42.07 | 57.93% |
| Pixtral | 5000 | 64.80 | 42.49 | 57.51% |
| Qwen7B | 500 | 60.39 | 29.58 | 70.42% |
| Qwen7B | 1000 | 64.85 | 38.50 | 61.50% |
| Qwen7B | 2000 | 60.81 | 30.44 | 69.56% |
| Qwen7B | 5000 | 61.32 | 30.65 | 69.35% |

**Interpretation.** There is no monotonic win from simply increasing vocabulary
size. Pixtral is best at `2000`, while Qwen7B is best at `1000`; `5000` is
reasonable but not uniquely optimal. If the paper currently says 5000 is
chosen because it is clearly better, soften that claim. A safer phrasing is:
`5000` is a conservative high-capacity setting, but results are not highly
sensitive to vocabulary size.

## EXP-2: Calibration ECE and Brier Score

**Files.**

- Table: `scripts/results_final/additional/tables/ece_calibration_table.csv`
- LaTeX: `scripts/results_final/additional/tables/ece_calibration_table.tex`
- JSONs:
  - `scripts/results_final/additional/ece_calibration/vqav2/pixtral/results.json`
  - `scripts/results_final/additional/ece_calibration/gqa/qwen7b/results.json`
- Figures:
  - `scripts/results_final/additional/figures/reliability_vqav2_pixtral.pdf`
  - `scripts/results_final/additional/figures/reliability_gqa_qwen7b.pdf`

**What it compares.** For each fidelity, it compares raw GBR scores against
isotonic-calibrated VOILA probabilities using:

- ECE with equal-frequency bins
- Brier score

**Overall average.**

| Metric | Uncalibrated | Isotonic | Improvement |
|---|---:|---:|---:|
| ECE | 0.0335 | 0.0162 | 0.0173 lower |
| Brier | 0.2049 | 0.2019 | 0.0030 lower |

**By pair.**

| Dataset/model | ECE raw | ECE iso | ECE improvement | Brier raw | Brier iso |
|---|---:|---:|---:|---:|---:|
| VQAv2/Pixtral | 0.0252 | 0.0120 | 0.0132 | 0.2049 | 0.2031 |
| GQA/Qwen7B | 0.0418 | 0.0203 | 0.0215 | 0.2048 | 0.2007 |

**Interpretation.** Isotonic calibration roughly halves ECE in both tested
settings. Brier improves too, though more modestly. This gives the calibration
section a direct quantitative claim: isotonic is not just changing routing
outcomes; it improves probability reliability.

## EXP-3: Extended OOD Transfer

**Files.**

- Table: `scripts/results_final/additional/tables/ood_extended_table.csv`
- LaTeX: `scripts/results_final/additional/tables/ood_extended_table.tex`
- Per-run JSONs:
  `scripts/results_final/additional/ood_extended/{source}_to_{target}/{model}/results.json`

**What it compares.** Train VOILA on VQAv2 or GQA, then evaluate without
retraining on TextVQA and FloodNet.

**Mean results across six VLMs.**

| Transfer | Accuracy | Cost | Cost reduction | OOD / in-domain accuracy ratio |
|---|---:|---:|---:|---:|
| GQA -> FloodNet | 53.53 | 23.35 | 76.65% | 0.842 |
| GQA -> TextVQA | 67.29 | 23.36 | 76.64% | 0.796 |
| VQAv2 -> FloodNet | 47.59 | 22.54 | 77.46% | 0.745 |
| VQAv2 -> TextVQA | 78.04 | 36.83 | 63.17% | 0.917 |

**Interpretation.** Cross-domain transfer is real but uneven. VQAv2 -> TextVQA
is the strongest harder-OOD result, retaining about `91.7%` of in-domain VOILA
accuracy on average. FloodNet is harder, especially VQAv2 -> FloodNet, which
retains only `74.5%`. GQA -> FloodNet is better than VQAv2 -> FloodNet on
average.

For the paper, this extends the OOD story beyond VQAv2/GQA. The right framing
is not “OOD is solved,” but “VOILA policies retain useful routing behavior
under substantial domain shift, with degradation increasing as the target
domain becomes visually and linguistically farther away.”

## EXP-4: Oracle Gap Decomposition

**Files.**

- Table: `scripts/results_final/additional/tables/oracle_decomp_table.csv`
- JSONs:
  - `scripts/results_final/additional/oracle_decomp/vqav2/pixtral/results.json`
  - `scripts/results_final/additional/oracle_decomp/gqa/pixtral/results.json`

**What it compares.** For each query, compare VOILA's selected fidelity against
the hindsight oracle's cheapest correct fidelity.

Categories:

- `correct_routing`: VOILA selected the oracle fidelity and was correct
- `over_escalation_error`: VOILA was correct but paid more than oracle
- `under_escalation_error`: VOILA selected too low a fidelity and got it wrong
- `irreducible_error`: no fidelity level answers correctly

| Dataset/model | Correct routing | Over-escalation | Under-escalation | Irreducible |
|---|---:|---:|---:|---:|
| VQAv2/Pixtral | 23.54% | 40.73% | 23.05% | 12.68% |
| GQA/Pixtral | 29.66% | 30.12% | 25.05% | 15.17% |

**Interpretation.** A large part of the oracle gap is not total failure:
VOILA often gets the answer right while using a more expensive fidelity than
the oracle. Under-escalation is still substantial, around `23-25%`, and is the
main reducible accuracy error. Irreducible errors are `12.7-15.2%`, meaning
there is a non-trivial ceiling imposed by the VLM itself, not by routing.

This is a strong discussion result: future gains should focus on reducing
under-escalation while preserving VOILA's cost savings.

## EXP-5: FloodNet Question-Type Analysis

**Files.**

- Table:
  `scripts/results_final/additional/tables/floodnet_question_type_accuracy.csv`
- Status JSON:
  `scripts/results_final/additional/floodnet_analysis/status.json`

**What it compares.** FloodNet accuracy by `question_type`, model, and fidelity.
This is the quantitative part of the FloodNet anomaly investigation.

**Best fidelity by model and FloodNet question type.**

| Model | Question type | Best fidelity | Accuracy |
|---|---|---|---:|
| LLaMA4 | Complex_Counting | full | 32.47 |
| LLaMA4 | Condition_Recognition | jpeg_q10 | 80.39 |
| LLaMA4 | Simple_Counting | full | 36.95 |
| LLaMA4 | Yes_No | full | 77.51 |
| LLaVA | Complex_Counting | full | 23.23 |
| LLaVA | Condition_Recognition | caption | 91.27 |
| LLaVA | Simple_Counting | jpeg_q10 | 16.04 |
| LLaVA | Yes_No | full | 88.70 |
| Pixtral | Complex_Counting | jpeg_q1 | 17.17 |
| Pixtral | Condition_Recognition | jpeg_q10 | 61.90 |
| Pixtral | Simple_Counting | jpeg_q1 | 19.97 |
| Pixtral | Yes_No | full | 55.94 |
| Qwen235B | Complex_Counting | full | 37.52 |
| Qwen235B | Condition_Recognition | jpeg_q1 | 97.97 |
| Qwen235B | Simple_Counting | full | 43.08 |
| Qwen235B | Yes_No | full | 93.77 |
| Qwen72B | Complex_Counting | full | 16.88 |
| Qwen72B | Condition_Recognition | resize_32x32 | 98.83 |
| Qwen72B | Simple_Counting | full | 17.36 |
| Qwen72B | Yes_No | resize_32x32 | 50.70 |
| Qwen7B | Complex_Counting | full | 33.48 |
| Qwen7B | Condition_Recognition | jpeg_q10 | 83.93 |
| Qwen7B | Simple_Counting | full | 35.53 |
| Qwen7B | Yes_No | full | 73.47 |

**Interpretation.** FloodNet is not uniformly “full resolution helps.” Counting
usually benefits from full fidelity, but condition recognition often peaks at a
compressed or even caption-level representation. This helps explain why VOILA
can sometimes beat full fidelity on FloodNet: the best fidelity depends strongly
on question type and model, and full resolution is not always the accuracy
winner.

The internal VLM resolution-cap check is still manual; this table is the
offline quantitative evidence.

## EXP-7: Cascade Degeneracy

**Files.**

- Table:
  `scripts/results_final/additional/tables/cascade_degeneracy_table.csv`
- Per-run JSONs:
  `scripts/results_final/additional/cascade_degeneracy/{dataset}/{model}/results.json`

**What it compares.** For logprob cascade, this records:

- Accuracy
- Average cumulative cascade cost
- Average number of fidelity levels executed
- Escalation rate

**Overall means across 24 dataset-model pairs.**

| Metric | Mean |
|---|---:|
| Accuracy | 58.32 |
| Cumulative cost | 39.67 |
| Levels executed | 1.97 |
| Escalation rate | 49.99% |

**By dataset.**

| Dataset | Accuracy | Cumulative cost | Levels executed | Escalation rate |
|---|---:|---:|---:|---:|
| FloodNet | 55.79 | 38.08 | 1.95 | 54.89% |
| GQA | 51.13 | 44.69 | 2.10 | 54.92% |
| TextVQA | 71.91 | 39.31 | 1.98 | 47.35% |
| VQAv2 | 54.46 | 36.58 | 1.86 | 42.82% |

**Degenerate low-escalation examples.**

| Dataset | Model | Accuracy | Cost | Levels | Escalation |
|---|---|---:|---:|---:|---:|
| TextVQA | Qwen72B | 73.32 | 7.58 | 1.00 | 0.00% |
| GQA | Qwen72B | 30.78 | 7.59 | 1.00 | 0.03% |
| VQAv2 | Qwen72B | 40.23 | 7.72 | 1.00 | 0.18% |
| GQA | LLaMA4 | 34.68 | 7.75 | 1.01 | 0.28% |
| VQAv2 | LLaMA4 | 45.65 | 7.90 | 1.01 | 0.42% |

**High-escalation examples.**

| Dataset | Model | Accuracy | Cost | Levels | Escalation |
|---|---|---:|---:|---:|---:|
| GQA | Qwen235B | 65.02 | 83.77 | 3.13 | 99.96% |
| GQA | Qwen7B | 52.04 | 74.83 | 3.03 | 99.78% |
| FloodNet | Qwen72B | 70.24 | 59.94 | 2.64 | 97.79% |
| VQAv2 | Qwen7B | 61.37 | 74.66 | 2.96 | 96.19% |

**Interpretation.** This is a very useful baseline failure analysis. Some
model/dataset pairs barely escalate at all, so they are cheap but wrong because
caption-level confidence is too high. Others escalate almost always, so they
recover accuracy but lose the cost advantage. This supports the paper's claim
that answer logprob is an unstable routing signal.

## FIG-1: Fidelity Saturation Curves

**Files.**

- Data: `scripts/results_final/additional/tables/fidelity_saturation_table.csv`
- Figures:
  - `scripts/results_final/additional/figures/fidelity_saturation_vqav2_pixtral.pdf`
  - `scripts/results_final/additional/figures/fidelity_saturation_gqa_pixtral.pdf`

**What it compares.** Accuracy across fidelities by question type for
VQAv2/Pixtral and GQA/Pixtral. It also marks the first fidelity where the
question type reaches at least `95%` of its full-fidelity accuracy.

**Plateau fidelities.**

| Dataset | Question type | N | Plateau fidelity |
|---|---|---:|---|
| VQAv2 | color | 2117 | jpeg_q10 |
| VQAv2 | counting | 2551 | full |
| VQAv2 | ocr_text | 208 | full |
| VQAv2 | other | 7095 | full |
| VQAv2 | spatial | 1090 | full |
| VQAv2 | yes_no | 8374 | jpeg_q10 |
| GQA | color | 803 | full |
| GQA | ocr_text | 25 | jpeg_q10 |
| GQA | other | 3387 | jpeg_q10 |
| GQA | spatial | 971 | jpeg_q10 |
| GQA | yes_no | 5044 | jpeg_q10 |

**Interpretation.** The saturation curves make the central VOILA intuition
visual: different question types reach their useful fidelity at different
points. Counting and OCR-like VQAv2 questions often still need full fidelity,
while yes/no and many GQA categories plateau earlier.

## FIG-2: Escalation ROC

**Files.**

- Table: `scripts/results_final/additional/tables/roc_escalation_table.csv`
- Figure:
  `scripts/results_final/additional/figures/roc_escalation_decision.pdf`

**What it compares.** Binary decision: should caption escalate? Positive means
caption is wrong but full fidelity is correct.

| Dataset/model | Score | AUC | N | Positive rate |
|---|---|---:|---:|---:|
| VQAv2/Pixtral | Log-prob | 0.522 | 4287 | 0.301 |
| VQAv2/Pixtral | VOILA | 0.581 | 4287 | 0.301 |
| VQAv2/Qwen72B | Log-prob | 0.338 | 4287 | 0.399 |
| VQAv2/Qwen72B | VOILA | 0.556 | 4287 | 0.399 |

**Interpretation.** VOILA's question-conditioned score beats logprob for the
caption escalation decision in both pairs. The Qwen72B logprob AUC is below
random, which is especially strong evidence for the logprob-cascade critique:
confidence can be actively misleading, not merely noisy.

## FIG-3: Reliability Diagrams

**Files.**

- `scripts/results_final/additional/figures/reliability_vqav2_pixtral.pdf`
- `scripts/results_final/additional/figures/reliability_gqa_qwen7b.pdf`

**What they show.** Each plot compares:

- VLM logprob-derived confidence
- Raw GBR score
- Isotonic VOILA probability
- Perfect calibration diagonal

The quantitative companion table is
`scripts/results_final/additional/tables/ece_calibration_table.csv`.

**Interpretation.** Use these figures with the ECE table. The main claim is:
isotonic calibration visibly pulls the VOILA probabilities closer to empirical
accuracy and cuts ECE by about half.

## FIG-4: Routing Word Importance

**Files.**

- Table: `scripts/results_final/additional/tables/routing_words_table.csv`
- Figures:
  - `scripts/results_final/additional/figures/routing_words_caption.pdf`
  - `scripts/results_final/additional/figures/routing_words_full.pdf`

**What it compares.** Top TF-IDF unigram importances from the VQAv2/Pixtral
GBR models for caption and full fidelity.

**Top words.**

| Fidelity | Top words |
|---|---|
| Caption | what, is, time, the, many, a, this?, how, this, zoo? |
| Full | what, time, color, is, this, many, why, the, are, sport |

**Interpretation.** The word lists are somewhat noisy as unigram importances,
but they are still useful as a qualitative sanity check: full fidelity weights
include words like `color`, `many`, and `sport`, which correspond to visual
detail or category distinctions. This is better as an appendix/diagnostic
figure than as a core claim.

## FIG-5: Iso-Accuracy Contours

**Files.**

- Data: `scripts/results_final/additional/tables/iso_accuracy_contour_data.csv`
- Figures:
  - `scripts/results_final/additional/figures/iso_accuracy_contour_vqav2.pdf`
  - `scripts/results_final/additional/figures/iso_accuracy_contour_gqa.pdf`

**What it compares.** Fixed-fidelity accuracy by model and fidelity. This
visualizes the tradeoff between model scale and image fidelity.

**VQAv2 fixed-fidelity accuracy.**

| Fidelity | LLaMA4 | LLaVA | Pixtral | Qwen235B | Qwen72B | Qwen7B |
|---|---:|---:|---:|---:|---:|---:|
| caption | 45.62 | 57.75 | 49.52 | 57.72 | 40.20 | 35.89 |
| resize_32x32 | 39.67 | 59.52 | 33.31 | 50.39 | 44.71 | 48.67 |
| jpeg_q1 | 51.24 | 58.30 | 57.14 | 63.69 | 60.63 | 56.98 |
| jpeg_q10 | 66.32 | 70.68 | 70.60 | 76.86 | 74.67 | 72.59 |
| full | 68.39 | 73.06 | 74.15 | 79.12 | 77.63 | 77.76 |

**GQA fixed-fidelity accuracy.**

| Fidelity | LLaMA4 | LLaVA | Pixtral | Qwen235B | Qwen72B | Qwen7B |
|---|---:|---:|---:|---:|---:|---:|
| caption | 34.63 | 63.21 | 42.13 | 29.00 | 30.77 | 39.74 |
| resize_32x32 | 43.18 | 66.41 | 37.66 | 53.30 | 47.95 | 45.92 |
| jpeg_q1 | 50.65 | 62.89 | 59.83 | 62.26 | 61.65 | 47.38 |
| jpeg_q10 | 60.76 | 72.19 | 68.21 | 70.27 | 71.19 | 59.03 |
| full | 62.91 | 74.14 | 71.10 | 73.22 | 73.54 | 66.23 |

**Interpretation.** Fidelity generally matters more than model scale once
caption-level information is insufficient. The heatmaps are good motivation for
adaptive fidelity: a smaller model at a higher fidelity can sometimes beat a
larger model at a lower fidelity, and full fidelity is not always needed to get
near-full performance.

## Recommended Paper Use

Use the additional experiments like this:

1. Feature/OOD ablation: explain why question-only routing works and why
   structured question-type signals are useful OOD.
2. Calibration table and reliability diagrams: support the isotonic calibration
   claim directly with ECE/Brier numbers.
3. Extended OOD: extend the robustness story beyond VQAv2/GQA.
4. Oracle decomposition: explain the oracle gap and identify under-escalation
   as the main reducible error mode.
5. Cascade degeneracy and ROC: strengthen the critique of logprob confidence
   routing.
6. FloodNet question-type table: explain why compressed fidelities can beat
   full resolution on FloodNet for some question types.
7. Saturation and contour figures: use as visual motivation for adaptive
   fidelity selection.

## Reproduction Commands

Run all additional experiments:

```powershell
python scripts\run_additional_experiments.py --experiments all --output_dir scripts\results_final --model_artifact_dir model_artifacts\additional
```

Reuse checkpoints after an interrupted run:

```powershell
python scripts\run_additional_experiments.py --experiments all --reuse_artifacts --output_dir scripts\results_final --model_artifact_dir model_artifacts\additional
```

Regenerate only the table index:

```powershell
python scripts\run_additional_experiments.py --experiments tables --output_dir scripts\results_final --model_artifact_dir model_artifacts\additional
```
