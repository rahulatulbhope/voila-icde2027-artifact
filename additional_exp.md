# VOILA — Additional Experiments and Visualizations for EMNLP 2026

This document instructs on every additional experiment that needs to be run and every new figure/table that needs to be produced before the paper is submission-ready. Items are organized by priority. Each entry explains **why** the experiment is needed (which reviewer objection it addresses), **exactly what to run**, **what numbers to report**, and **what figure or table to produce**.

---

## Experiment Infrastructure Notes

All experiments should use the existing evaluation pipeline in `D:\VOILA_code\scripts\`. Results should be saved to `D:\VOILA_code\scripts\results_final\` following the same structure as existing results:

```
results_final/
  voila/         {dataset}/{model}/results.json       → keys: accuracy_mean, avg_cost_mean
  oracle/        {dataset}/{model}/results.json
  baselines/
    logprob_cascade/ {dataset}/{model}/results.json   → keys: accuracy, avg_cost
    llm_router/  {dataset}/{model}/results.json
  ablations/
    feature_ablation/ {dataset}/{model}/{feature_group}/results.json  ← NEW
    ece_calibration/  {dataset}/{model}/results.json                  ← NEW
    ood_extended/     {dataset}/{model}/results.json                  ← NEW
    oracle_decomp/    {dataset}/{model}/results.json                  ← NEW
```

Models: `Pixtral-12B-2409`, `Llama-4-Maverick`, `Qwen2-VL-72B-Instruct`, `LLaVA-1.5-7B-hf`, `Qwen2.5-VL-7B-Instruct`, `Qwen3-VL-235B-A22B`

Datasets: `vqav2`, `gqa`, `textvqa`, `floodnet`

Fidelity levels: `caption`, `resize_32x32`, `jpeg_q1`, `jpeg_q10`, `full`

Cost values (normalized, full=100): caption=7.6, resize=15.1, jpeg_q1=37.8, jpeg_q10=53.2, full=100.0

---

## PRIORITY 1 — Blocking Experiments (must run before submission)

These experiments directly address reviewer objections that the paper currently cannot answer quantitatively.

---

### EXP-1: Feature Group Ablation with OOD Story

**Why this is needed:** Reviewers will ask "why TF-IDF + question-type flags + lexical? Could you just use one?" (W2). More importantly, the OOD transfer results in Table 4 (VQA→GQA transfer) currently look like a happy accident. This experiment provides the mechanistic explanation: question-type flags are language-universal and drive OOD generalization, while TF-IDF is corpus-specific and helps in-distribution.

**What to run:**

Train 4 separate VOILA models, each using a different subset of the 5,017-dim feature vector:

| Condition | Features used | Dimensions |
|-----------|--------------|-----------|
| `tfidf_only` | Top-5,000 TF-IDF unigrams only | 5,000 |
| `qtype_only` | 5 question-type binary flags only | 5 |
| `lexical_only` | 12 lexical statistics only (length, numeric tokens, superlatives, negations) | 12 |
| `qtype_lexical` | Question-type flags + lexical stats combined (no TF-IDF) | 17 |
| `full` | All features (TF-IDF + lexical + question-type) | 5,017 |

For each condition, train a GBR on VQA-v2 training set and evaluate on:
- VQA-v2 validation (in-distribution)
- GQA balanced test (cross-dataset OOD, using VQA-v2-trained model on GQA)
- TextVQA validation (cross-domain OOD, OCR-heavy questions)
- FloodNet test (cross-domain OOD, aerial imagery questions)

Run with isotonic calibration using the standard 80/20 train/calibration split.

**Additionally:** For the `tfidf_only` condition, ablate vocabulary size: train with 500, 1000, 2000, 5000 TF-IDF unigrams (in-distribution on VQA-v2 only) to justify the 5,000-token choice.

**Run for:** Pixtral-12B-2409 and Qwen2.5-VL-7B-Instruct (two representative models — a larger and smaller one).

**What to report:** A table with columns: Feature Group | VQA-v2 (Acc%/Cost) | GQA OOD (Acc%/Cost) | TextVQA OOD (Acc%/Cost) | FloodNet OOD (Acc%/Cost). Separate tables for each model, or combine with ±std across both models.

**Expected story:** `tfidf_only` should show the largest in-distribution performance but sharpest OOD degradation. `qtype_only` and `qtype_lexical` should show more stable OOD performance. Full features should be best or near-best everywhere. This tells a clean story: "TF-IDF captures dataset-specific vocabulary patterns; question-type flags capture language-universal routing signals. The OOD transfer results (Table 4) are explained by the generalization of the latter."

**Where it goes in the paper:** New ablation table in the results section or appendix ablations. Also add 1-2 sentences to the Section 3 feature description paragraph explaining this expected in-dist vs. OOD tradeoff.

**Vocab size sub-experiment:** Small table or inline sentence: "Performance improves from 500 to 5,000 TF-IDF features but plateaus beyond 2,000; we use 5,000 as a conservative default." Goes inline in Section 3 method description.

---

### EXP-2: ECE and Brier Score Before/After Calibration

**Why this is needed:** The calibration section (Section 3.2) claims isotonic regression improves probability reliability, but Table 5 only shows Acc%/Cost, not calibration quality directly. Reviewers will ask "how much does calibration actually help, quantitatively?" (Q4).

**What to run:**

For each of the 5 fidelity levels (`caption`, `resize_32x32`, `jpeg_q1`, `jpeg_q10`, `full`), on the held-out calibration split (20% of training data):

1. Compute raw GBR scores `r_f(q)` for all queries.
2. Compute isotonic-calibrated probabilities `hat_p_f(q)` for all queries.
3. For both, compute:
   - **ECE (Expected Calibration Error)** using 15 equal-frequency bins: `ECE = sum_b (|bin_b| / N) * |avg_confidence_b - avg_accuracy_b|`
   - **Brier Score**: `BS = (1/N) * sum_i (hat_p_f(q_i) - y_i)^2` where y_i is binary correctness label

Run for: VQA-v2 with Pixtral-12B-2409 (primary), and also GQA with Qwen2.5-VL-7B-Instruct (secondary, to show robustness).

**What to report:** A table like:

| Fidelity | ECE (uncalib) | ECE (iso) | Brier (uncalib) | Brier (iso) |
|---------|--------------|-----------|----------------|------------|
| Caption | 0.XX | 0.XX | 0.XX | 0.XX |
| Resize 32×32 | ... | | | |
| JPEG Q1 | ... | | | |
| JPEG Q10 | ... | | | |
| Full | ... | | | |

**Expected story:** Calibration should show the largest benefit at low fidelity (caption, resize) where the GBR's raw scores are most miscalibrated — the VLM's behavior is least predictable when visual information is almost completely absent.

**Where it goes:** Appendix calibration section or as additional columns in the existing ablation Table 5. Reference it in the results text: "Isotonic calibration reduces ECE from X to Y at caption fidelity, with the largest benefit at the fidelity levels where raw GBR scores are most unreliable."

**Figure to make:** A **reliability diagram** (calibration plot) — for the caption fidelity level, plot 3 curves on one axes: (x-axis = predicted probability, y-axis = empirical accuracy): (1) VLM output confidence (raw log-prob), (2) uncalibrated GBR score, (3) isotonic-calibrated GBR. Perfect calibration is the diagonal. This single figure makes the calibration contribution visually self-evident. Include this as a new subfigure in Figure 3 (the calibration scatter figure already in the paper).

---

### EXP-3: Extended OOD Transfer (VQA→TextVQA and VQA→FloodNet)

**Why this is needed:** The current OOD table (Table 4) only shows VQA→GQA and GQA→VQA transfer, which are similar-domain datasets. Reviewers will correctly ask: "does this generalize to OOD domains that are linguistically very different from VQA-v2?" (Q2). TextVQA (OCR-heavy) and FloodNet (aerial disaster, short binary answers) are the hardest tests.

**What to run:**

Take the VOILA GBR trained on VQA-v2 (no retraining) and evaluate directly on:
- TextVQA validation set
- FloodNet test set

Also take the GBR trained on GQA and evaluate on:
- TextVQA validation set
- FloodNet test set

Run isotonic calibration using VQA-v2 held-out calibration set (not the target dataset's calibration set, since this is a true zero-shot transfer).

Run for: All 6 VLMs (or at minimum Pixtral-12B, LLaVA-1.5-7B, Qwen2-VL-72B, Qwen3-VL-235B — one small, one large from each family).

**What to report:** Extend Table 4 (OOD table) with two new columns: `VQA→TextVQA` and `VQA→FloodNet`, plus optionally `GQA→TextVQA` and `GQA→FloodNet`. Report Acc%/Cost as in the existing OOD table.

**Critical comparison to include:** For context, show the in-distribution VOILA result alongside the cross-domain transfer result. The ratio `OOD_accuracy / in-dist_accuracy` tells you how much is lost to domain shift.

**Expected story:** If the `qtype_only` features from EXP-1 transfer well but `tfidf_only` degrades, the cross-domain OOD table directly validates the feature ablation story. The two experiments reinforce each other.

**Where it goes:** Extend existing Table 4. Add a paragraph after the current OOD narrative discussing these harder cross-domain results.

---

### EXP-4: Oracle Gap Decomposition

**Why this is needed:** The Oracle-VOILA accuracy gap is about 22 points on VQA-v2 (Oracle: 86.9%, VOILA: 64.9%). This is a large gap. Reviewers will ask what is causing it and whether it's reducible (W6). Currently the paper has no explanation.

**What to run:**

For each query in the evaluation set, given VOILA's routing decision `f_voila` and the oracle's optimal decision `f_oracle`:

Categorize each query into one of four buckets:

1. **Correct routing**: `f_voila == f_oracle` AND the answer is correct at `f_voila`. (VOILA perfectly identified the cheapest sufficient fidelity.)

2. **Equivalent routing**: VOILA chose a different fidelity than oracle but still got the correct answer. (e.g., VOILA went to JPEG Q10, oracle would have chosen JPEG Q1, but VOILA still got it right.)

3. **Under-escalation error**: VOILA routed to fidelity `f_voila < f_oracle` AND the answer is incorrect at `f_voila` but correct at `f_oracle`. (VOILA should have retrieved higher fidelity but didn't.)

4. **Over-escalation error**: VOILA routed to fidelity `f_voila > f_oracle` AND gets the correct answer but paid more than necessary. (VOILA retrieved more than needed — wasted cost but correct answer.)

5. **Irreducible error**: The answer is incorrect at ALL fidelity levels including full-resolution. (No fidelity can fix this — it's a model limitation, not a routing limitation.)

Compute the percentage of each category over the full evaluation set.

Run for: VQA-v2 with Pixtral-12B (primary), GQA with Pixtral-12B (secondary).

**What to report:** A breakdown table:

| Category | VQA-v2 (%) | GQA (%) |
|----------|-----------|---------|
| Correct routing | X% | X% |
| Equivalent routing (correct, suboptimal cost) | X% | X% |
| Under-escalation error (could have been fixed) | X% | X% |
| Over-escalation error (correct but costly) | X% | X% |
| Irreducible error (no fidelity works) | X% | X% |

**Expected story:** The under-escalation error rate tells you how much of the Oracle gap is reducible by a better predictor. If under-escalation is, say, 15% of all queries, then a perfect predictor could close 15 percentage points of the Oracle gap. If irreducible errors account for 10% of all queries, then the theoretical ceiling for any routing system is ~90% accuracy on this dataset. This is a publishable finding in itself.

**Where it goes:** New paragraph after the Oracle gap discussion in results. Could be a small inline table or reported as percentages in the text.

---

## PRIORITY 2 — Validation Experiments (strongly recommended)

These experiments validate specific claims and answer specific likely reviewer questions.

---

### EXP-5: FloodNet VLM Resolution Cap Investigation (W5/Q1)

**Why this is needed:** VOILA exceeds full-resolution accuracy on FloodNet for some models (Qwen2-VL-72B: 73.4% VOILA vs. 58.6% full-res). This is surprising and needs a mechanistic explanation (W5). The most likely explanation is that VLMs internally resize all inputs to a fixed token count, making "full resolution" vs. "JPEG Q10" images equivalent after internal processing.

**What to run:**

**Step 1: Check VLM internal resolution caps.**
For each VLM, inspect the model configuration or documentation to find the maximum image resolution the VLM processes internally (the "native resolution" or token count for images). Common caps:
- Qwen2-VL-72B: supports up to 1280px (dynamically padded)
- Pixtral-12B: 1024px tiles
- LLaVA-1.5: fixed 336×336 resize

Compare the actual image sizes in the FloodNet dataset:
- `full` resolution: average size in pixels (read from dataset)
- `jpeg_q10` resolution: same dimensions, lower quality
- Check if `full` resolution exceeds the VLM's internal cap

If `full` FloodNet images are, say, 3000×4000 but the VLM downsizes to 1280×1280, then the VLM is processing the same effective resolution for `full` and `jpeg_q10` inputs — making "full" retrieval purely wasteful with no accuracy benefit.

**Step 2: Per-question-type error analysis on FloodNet.**
FloodNet has specific question types: "Is the road flooded?", "What is the condition of the building?", "How many flooded buildings are visible?" Group questions by type and compute accuracy at each fidelity. Report which question types drive the full-resolution accuracy drop.

**Step 3: Visual inspection.**
Sample 10-20 queries where VOILA at JPEG Q10 succeeds but full-resolution fails. Look at the actual images. Is there a pattern? (e.g., full-res triggers hallucination on fine-grained texture?)

**What to report:** One paragraph in the results discussion explaining the FloodNet anomaly with either: (a) "Full-resolution FloodNet images are internally downsampled by Qwen2-VL-72B to [X]px, making them equivalent to JPEG Q10 inputs after preprocessing; the apparent accuracy gain is a preprocessing artifact." OR (b) "Error analysis shows full-resolution imagery triggers hallucination on fine-grained texture in X% of disaster-assessment queries for [model]; JPEG Q10 preserves semantic content while removing distracting texture."

**Where it goes:** New paragraph in the FloodNet results discussion (currently the section "VOILA Enables Cost-Aware Perception in IoT / CPS Systems").

---

### EXP-6: LLM Router on TextVQA and FloodNet (W4)

**Why this is needed:** The GPT-4o Router is currently only evaluated on VQA-v2 and GQA. Reviewers will ask why it's not on TextVQA (OCR-heavy) and FloodNet (aerial imagery), since those are the hardest settings for linguistic routing (W4).

**What to run:**

Sample 1,000 questions from TextVQA validation and 1,000 from FloodNet test (or all FloodNet if < 1,000).

For each question, prompt GPT-4o (or GPT-4o-mini for cost) with:
```
Given this visual question, classify the visual information level required:
- "low": can be answered from a text description or small thumbnail
- "medium": requires a compressed image (low JPEG quality)
- "high": requires full-resolution image

Question: {question_text}

Respond with only one word: low, medium, or high.
```

Map ternary labels to VOILA fidelity space:
- "low" → caption (cost 7.6)
- "medium" → JPEG Q10 (cost 53.2)
- "high" → full (cost 100.0)

For each label, evaluate VLM accuracy at the designated fidelity. Report accuracy/cost for all 6 VLMs (labels are shared; only accuracy changes per VLM).

**What to report:** Add TextVQA and FloodNet columns to the GPT-4o Router rows in Table 1 (master comparison table), replacing the "--" entries.

**Note on cost:** If GPT-4o API cost is a concern, use GPT-4o-mini for the classification calls. The comparison is about ternary routing quality, not about GPT-4o's underlying intelligence. If you use mini, note this explicitly.

---

### EXP-7: Cascade Degeneracy Analysis (W7)

**Why this is needed:** Some Log-prob Cascade rows in Table 1 have near-caption cost (LLaMA-4-Maverick: cost=7.7-7.9, Qwen2-VL-72B on VQA-v2: cost=7.7), indicating the cascade never escalates. This is actually strong evidence for Observation 2 (log-prob is unreliable), but is currently unexplained in the text.

**What to run:**

For each model-dataset combination, compute the fraction of queries where the cascade escalates at least once (escalation rate). Also compute the average number of fidelity levels executed before the cascade stops.

This data should already be computable from existing cascade results by logging escalation decisions during evaluation.

**What to report:** Small inline table or sentence in the results discussion:

| Model | VQA-v2 escalation rate | GQA escalation rate |
|-------|----------------------|---------------------|
| Pixtral-12B | X% | X% |
| LLaMA-4-Maverick | X% | X% |
| Qwen2-VL-72B | X% | X% |
| ... | | |

For models with near-zero escalation rate, add an explicit note: "The cascade never escalates for [model] because its output confidence at caption fidelity is uniformly high — the model produces confident but wrong answers (see Observation 2, Figure 2b), making the escalation threshold unreachable."

**Where it goes:** Paragraph after the cascade comparison in results. Reframes the degeneracy as supporting evidence for Observation 2 rather than a confusing anomaly.

---

## PRIORITY 3 — Visualization Improvements

These are improvements to existing figures and new figures to add. The paper's motivation section (Section 2) currently has three plots but they could all be significantly more compelling.

---

### FIG-1: Fidelity Saturation Curves (replaces or supplements Figure 2a)

**What it shows:** For each question type (counting, yes/no, color, spatial, OCR/text), plot accuracy vs. fidelity level as a line curve across the 5 fidelity levels. Mark a dot on each curve at the point where accuracy first plateaus (the "minimum required fidelity").

**Why it's better than the current bar chart:** The current Figure 2a shows accuracy as grouped bars per question type per fidelity. The saturation curve makes the "minimum required fidelity threshold" concept physically visible — you can literally see each question type's curve flatten at a different fidelity level. This is the core concept of the paper made visual.

**How to make it:**

```python
import matplotlib.pyplot as plt
import numpy as np

# fidelity levels in order
fidelities = ['caption', 'resize_32x32', 'jpeg_q1', 'jpeg_q10', 'full']
costs = [7.6, 15.1, 37.8, 53.2, 100.0]

# For each question type, load accuracy at each fidelity from existing results
# Group questions by their first word / question-type flag
question_types = ['counting', 'yes/no', 'color', 'spatial', 'ocr/text']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

fig, ax = plt.subplots(figsize=(7, 4))
for qtype, color in zip(question_types, colors):
    accs = [...]  # load from results grouped by question type
    ax.plot(fidelities, accs, marker='o', label=qtype, color=color, linewidth=2)
    # Mark saturation point: first fidelity where acc[i] >= 0.95 * acc[-1]
    for i, acc in enumerate(accs):
        if acc >= 0.95 * accs[-1]:
            ax.plot(fidelities[i], acc, 'D', color=color, markersize=10, zorder=5)
            break
ax.set_xlabel('Fidelity Level')
ax.set_ylabel('Accuracy (%)')
ax.legend()
plt.savefig('plots/fidelity_saturation_curves.pdf', bbox_inches='tight')
```

**Use the same 5 fidelity colors consistently across ALL paper figures:** define a global color map at the top of every plotting script:
```python
FIDELITY_COLORS = {
    'caption': '#1f77b4',       # blue
    'resize_32x32': '#2ca02c',  # green
    'jpeg_q1': '#ff7f0e',       # orange
    'jpeg_q10': '#d62728',      # red
    'full': '#9467bd'           # purple
}
```
Apply this color scheme to Figure 2a, 2b, 2c, and the Pareto frontier plots. Consistency across figures dramatically improves readability.

---

### FIG-2: ROC Curve for Escalation Decision (new figure, replaces or supplements Figure 2b)

**What it shows:** For the binary decision "should we escalate from caption fidelity to the next level?", plot ROC curves for:
1. Log-probability threshold (using VLM's output confidence at caption fidelity)
2. VOILA's calibrated probability `hat_p_caption(q)` (inverted: low `hat_p` means escalate)
3. Random baseline (diagonal)

The x-axis is false positive rate (unnecessarily escalated queries), y-axis is true positive rate (correctly escalated queries). AUC values in the legend.

**Why this is powerful:** If log-prob AUC ≈ 0.50–0.55 (near-random) and VOILA AUC > 0.70, this is a single number that proves the entire paper's core claim — that confidence-based escalation is essentially random, while question-conditioned prediction is reliable. This directly quantifies Observation 2.

**How to compute:**

For each query, define the "ground truth" escalation label as: 1 if the query answers correctly at full fidelity but incorrectly at caption fidelity (i.e., escalation is needed), 0 otherwise.

The log-prob score is: `-log_prob_caption(q)` (higher = more uncertain = more likely to need escalation).
The VOILA score is: `1 - hat_p_caption(q)` (higher = lower predicted success = more likely to need escalation).

Use `sklearn.metrics.roc_curve` and `roc_auc_score`.

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# labels: 1 if escalation needed, 0 if caption is sufficient
labels = [...]
logprob_scores = [-lp for lp in caption_logprobs]  # negate: higher = more uncertain
voila_scores = [1 - p for p in voila_caption_probs]

fpr_lp, tpr_lp, _ = roc_curve(labels, logprob_scores)
fpr_v, tpr_v, _ = roc_curve(labels, voila_scores)
auc_lp = roc_auc_score(labels, logprob_scores)
auc_v = roc_auc_score(labels, voila_scores)

plt.plot(fpr_lp, tpr_lp, label=f'Log-prob (AUC={auc_lp:.2f})')
plt.plot(fpr_v, tpr_v, label=f'VOILA (AUC={auc_v:.2f})')
plt.plot([0,1],[0,1],'--k', label='Random (AUC=0.50)')
plt.xlabel('False Positive Rate (unnecessary escalations)')
plt.ylabel('True Positive Rate (necessary escalations caught)')
plt.legend()
plt.savefig('plots/roc_escalation_decision.pdf', bbox_inches='tight')
```

Run for: VQA-v2 with Pixtral-12B, and VQA-v2 with Qwen2-VL-72B. Show both on the same axes with dotted/solid line differentiation.

**Where it goes:** Replace or supplement Figure 2b. This is far more informative than the current overlapping log-prob distribution histogram.

---

### FIG-3: Reliability Diagram (calibration plot)

**What it shows:** For caption fidelity (the most important one), plot the calibration curves for:
1. VLM output confidence (raw output log-probability, rescaled to [0,1])
2. Uncalibrated GBR score (raw `r_f(q)` values, min-max normalized)
3. Isotonic-calibrated VOILA probability `hat_p_f(q)`

x-axis: predicted probability (10 equal-frequency bins), y-axis: empirical accuracy within each bin. Diagonal = perfect calibration.

**How to make it:**

```python
from sklearn.calibration import calibration_curve

prob_true_lp, prob_pred_lp = calibration_curve(y_true, logprob_scores_scaled, n_bins=10, strategy='quantile')
prob_true_gbr, prob_pred_gbr = calibration_curve(y_true, gbr_raw_scores_scaled, n_bins=10, strategy='quantile')
prob_true_iso, prob_pred_iso = calibration_curve(y_true, voila_iso_probs, n_bins=10, strategy='quantile')

plt.plot([0,1],[0,1],'--k', label='Perfect calibration')
plt.plot(prob_pred_lp, prob_true_lp, 'o-', label='VLM log-prob')
plt.plot(prob_pred_gbr, prob_true_gbr, 's-', label='GBR (uncalibrated)')
plt.plot(prob_pred_iso, prob_true_iso, '^-', label='VOILA (isotonic)')
```

Do this for caption fidelity AND full fidelity (two separate panels). The caption panel will show the most dramatic improvement from calibration; the full-resolution panel will show modest improvement.

**Where it goes:** Add as a new subfigure in the existing calibration figure (Figure 3 / `fig:cali:scatter`), or as a standalone figure in the ablations section.

---

### FIG-4: Routing Word Cloud (new figure)

**What it shows:** For each fidelity level, show a word cloud of the question words most predictive of routing to that fidelity. Word size = TF-IDF feature weight magnitude in the trained GBR for that fidelity class.

**How to make it:**

After training the GBR for caption fidelity (GBR_caption) and full fidelity (GBR_full), extract feature importances. The TF-IDF features correspond to specific unigrams — look up which unigrams are in positions 0–4999 of the feature vector, and get the corresponding GBR feature importances.

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# For each fidelity f:
# feature_importance[f] = gbr_f.feature_importances_  # shape: (5017,)
# tfidf_vocabulary = vectorizer.get_feature_names_out()  # shape: (5000,)

# For the "low fidelity" word cloud (words that predict caption is sufficient):
low_fidelity_importances = {
    tfidf_vocabulary[i]: feature_importance['caption'][i]
    for i in range(5000)
    if feature_importance['caption'][i] > 0
}

wc = WordCloud(width=400, height=200, background_color='white')
wc.generate_from_frequencies(low_fidelity_importances)

# Repeat for high fidelity (full-res)
high_fidelity_importances = {
    tfidf_vocabulary[i]: feature_importance['full'][i]
    for i in range(5000)
    if feature_importance['full'][i] > 0
}
```

Show two word clouds side-by-side: "Low-fidelity questions" and "High-fidelity questions". Expected: "is", "there", "what type", "does" in low; "how many", "read", "spell", "color", "text", "number" in high. This is a visually compelling demonstration that the routing is doing something linguistically meaningful.

**Where it goes:** New figure in Section 2 (observations) or Section 3 (method, showing what the TF-IDF features learned). Excellent for the paper's visual appeal.

---

### FIG-5: Iso-Accuracy Contour Plot (new figure for Observation 3)

**What it shows:** A 2D heatmap with model size on the x-axis (7B, 12B, 17B×12E, 72B, 235B) and fidelity level on the y-axis (caption, resize, JPEG Q1, JPEG Q10, full), with accuracy as the cell color. Contour lines connect cells with equal accuracy.

**Why this is better than the current line chart:** The contour map makes it visually immediate that you can trade fidelity for model size along a contour — a 7B model at full resolution can achieve the same accuracy as a 235B model at JPEG Q1 for certain question types. And crucially, at caption fidelity, no model size helps (the contour lines are nearly vertical), while at full resolution, accuracy improves monotonically with scale (horizontal contours).

**How to make it:**

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

models = ['LLaVA-1.5-7B', 'Qwen2.5-VL-7B', 'Pixtral-12B', 'Qwen2-VL-72B', 'Qwen3-VL-235B']
model_sizes = [7, 7, 12, 72, 235]  # params in billions
fidelity_labels = ['Caption', 'Resize', 'JPEG Q1', 'JPEG Q10', 'Full']
fidelity_costs = [7.6, 15.1, 37.8, 53.2, 100.0]

# Build accuracy matrix from Table 1 data
accuracy_matrix = np.array([
    [57.7, 35.9, 33.3, 48.7, 50.4],  # caption: LLaVA, Qwen2.5, Pixtral, Qwen2-72B, Qwen3
    # ... fill from Table 1
])

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(accuracy_matrix, aspect='auto', cmap='RdYlGn', vmin=20, vmax=85)
cs = ax.contour(accuracy_matrix, levels=10, colors='black', linewidths=0.5, alpha=0.5)
ax.clabel(cs, inline=True, fontsize=7)
ax.set_xticks(range(len(models)))
ax.set_xticklabels([f'{m}\n({s}B)' for m, s in zip(models, model_sizes)], fontsize=8)
ax.set_yticks(range(len(fidelity_labels)))
ax.set_yticklabels(fidelity_labels)
plt.colorbar(im, ax=ax, label='Accuracy (%)')
ax.set_title('VQA-v2: Accuracy across Model Size × Fidelity')
plt.savefig('plots/iso_accuracy_contour.pdf', bbox_inches='tight')
```

Run for VQA-v2 and GQA (two datasets). Show as two panels side by side.

**Where it goes:** Replaces or supplements Figure 2c. Much more informative than the current line chart.

---

### FIG-6: Bilateral Figure 1 (redesign of voila_overview.pdf)

**What it shows:** The current Figure 1 shows only one example (high-fidelity needed for airline logo). Redesign it to show TWO side-by-side examples:

- **Left panel:** "Is there a dog in the image?" → VOILA routes to caption (7.6 cost). Show the caption text and a small thumbnail. Label: "Coarse semantic question → caption sufficient."

- **Right panel:** "How many cars are in the parking lot?" → VOILA routes to JPEG Q10 (53.2 cost). Show the compressed image. Label: "Counting question → high fidelity required."

Below each panel, show a small bar chart or cost label: "Cost: 7.6" vs "Cost: 53.2" (vs. "Full: 100" baseline), with the savings highlighted.

This makes the routing decision bilateral — showing VOILA's ability to BOTH upgrade fidelity for hard questions AND downgrade for easy ones. It immediately communicates the adaptive nature of the system.

**Alternatively (better):** A three-panel figure showing VOILA's actual decision mechanism:
1. Question text input (displayed as text)
2. VOILA's predicted probability bars `hat_p_f(q)` for each fidelity (a simple bar chart showing predicted accuracy at caption/resize/JPEG Q1/JPEG Q10/full), with the VOI decision indicated by a vertical dashed line
3. The selected fidelity image + the correct answer

This makes the mechanism transparent rather than just showing input-output behavior.

---

## Reporting Checklist

Before submission, verify that the paper contains:

- [ ] Feature ablation table (EXP-1): in-distribution AND OOD, per feature group
- [ ] TF-IDF vocab size ablation (EXP-1 sub): inline sentence or small table
- [ ] ECE/Brier table before/after calibration (EXP-2): per fidelity level
- [ ] Reliability diagram figure (FIG-3): 3 curves on one axes
- [ ] Extended OOD table (EXP-3): VQA→TextVQA, VQA→FloodNet columns added to Table 4
- [ ] Oracle gap decomposition (EXP-4): breakdown into 5 error categories
- [ ] FloodNet resolution cap explanation (EXP-5): VLM internal resize cap checked
- [ ] LLM Router on TextVQA + FloodNet (EXP-6): "--" entries in Table 1 replaced
- [ ] Cascade degeneracy analysis (EXP-7): escalation rates per model
- [ ] Fidelity saturation curves (FIG-1): replaces or supplements Figure 2a
- [ ] ROC curve for escalation decision (FIG-2): VOILA vs. log-prob vs. random
- [ ] Routing word cloud (FIG-4): high-fidelity words vs. low-fidelity words
- [ ] Iso-accuracy contour plot (FIG-5): model size × fidelity heatmap
- [ ] Unified fidelity color scheme across all figures
- [ ] CARES citation added to bib (arxiv 2510.19496) and related works paragraph
- [ ] azizian2026promptguided citation added to bib (arxiv 2604.00314)

---

## Notes on What NOT to Spend Time On

- **BERT embedding baseline**: probably not worth the compute. The story is that TF-IDF + question-type is sufficient AND fast — adding BERT would improve accuracy slightly but kill the "no GPU needed at decision time" claim. Skip unless a reviewer specifically asks.
- **Cross-model transfer**: training on Pixtral labels and testing on Qwen2-VL-72B — not worth it; the per-VLM training requirement is already acknowledged in Limitations.
- **More datasets**: the 5 current datasets are sufficient. Adding more would dilute the narrative.
- **Video/audio fidelity**: explicitly scoped out in Limitations. Don't open this can of worms.
