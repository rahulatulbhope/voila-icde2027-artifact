# VOILA — Complete Technical Specification

> This document is the authoritative technical reference for the VOILA system.
> It is written to be sufficient for implementing automated experiment scripts
> from scratch, without needing to read the paper or prior conversation history.

---

## 1. Problem Definition

Given:
- A natural language question `q`
- An ordered set of visual fidelities `F = {f_1, f_2, ..., f_K}` sorted by ascending acquisition cost
- Each fidelity `f` has cost `c(f) > 0` (retrieval + bandwidth + storage tier penalty)
- A vision-language model (VLM) that, given question `q` and image at fidelity `f`, produces answer `â_f`

Goal: Select `f* = argmax_{f ∈ F} [ Pr(â_f is correct | q) - λ * c(f) ]`

**Key constraint:** The probability of correctness must be estimated **from question text alone** — no image retrieval, no VLM execution — at decision time. This is the pre-retrieval constraint.

---

## 2. Fidelity Set

```
F = {caption, resize_32x32, jpeg_q1, jpeg_q10, full}
```

| Fidelity | Description | Typical size (VQA-v2) | Normalized cost c(f) |
|---|---|---|---|
| `caption` | Text-only BLIP-2 caption, no image pixels | ~2 KB (text) | 10 |
| `resize_32x32` | Bilinear downscale to 32×32 pixels, PNG | ~3–5 KB | ~15 |
| `jpeg_q1` | JPEG quality=1, extreme compression | ~3–8 KB | ~20 |
| `jpeg_q10` | JPEG quality=10, heavy compression | ~15–50 KB | ~30–40 |
| `full` | Original full-resolution image (JPEG/PNG) | ~300–700 KB | 120 |

**Cost formula** (from cost_model appendix):
```
c̃(f) = b_tier(f) + w_bw * r(f)
c(f)  = 120 * c̃(f) / c̃(full)
```

Where:
- `b_tier(f)` = base tier penalty: edge=0, warm=0.3, cold=1.0 (relative)
- `r(f)` = empirical bandwidth ratio = size(f) / size(full)
- `w_bw` = bandwidth weight (default: 1.0)
- Tier assignments: caption/resize → edge (on-device), jpeg_q1/jpeg_q10 → warm (hot cloud), full → cold (archival)

**Implementation note:** Materialize all fidelities on disk before any experiment. Measure `size(f)` per dataset, compute `r(f)` empirically, then fix costs. `c(full)` is always 120 by definition.

---

## 3. Question Feature Extraction

Feature vector `φ(q)` is a concatenation of two groups:

### 3a. TF-IDF Features
- Standard sklearn `TfidfVectorizer` on question text
- Preprocessing: lowercase, whitespace tokenize (no stemming, no stopword removal)
- Vocabulary fitted **only on training folds** within each CV split (never on test)
- Default max features: use all (or cap at 5000 for memory efficiency)
- Output: sparse vector, shape `(n_tfidf_features,)`

### 3b. Structured / Lexical Features
Binary and numeric indicators derived from question string:

```python
def extract_structured_features(q: str) -> np.ndarray:
    q_lower = q.lower()
    tokens = q_lower.split()
    return np.array([
        # Numeric
        len(tokens),                                          # question length (tokens)
        int(any(t.isdigit() for t in tokens)),               # contains numeric token
        int(any(t in {'many', 'much', 'number', 'count',    # counting indicator
                      'few', 'total'} for t in tokens)),
        # Question type indicators
        int(q_lower.startswith(('is ', 'are ', 'was ', 'were ',
                                 'does ', 'do ', 'did ', 'has ',
                                 'have ', 'can ', 'could '))),  # yes/no question
        int(any(t in {'color', 'colour', 'colored', 'coloured'} for t in tokens)),  # color
        int(any(t in {'where', 'location', 'position', 'left',
                      'right', 'behind', 'front', 'above',
                      'below', 'beside', 'next'} for t in tokens)),  # spatial
        int(any(t in {'read', 'text', 'written', 'say', 'says',
                      'write', 'letter', 'word', 'number',
                      'sign', 'label'} for t in tokens)),    # OCR/text-reading
        int(any(t in {'what', 'which', 'who', 'whom'} for t in tokens)),  # WH question
        int(q_lower.startswith('how many')),                  # explicit counting
        int(q_lower.startswith('how ')),                      # other how-questions
        int(any(t in {'type', 'kind', 'brand', 'model',
                      'species'} for t in tokens)),           # category/type
        int(any(t in {'action', 'doing', 'activity', 'happen',
                      'happening'} for t in tokens)),         # activity/action
    ], dtype=np.float32)
```

**Final feature vector:**
```python
phi_q = np.concatenate([tfidf_vectorizer.transform([q]).toarray()[0],
                         extract_structured_features(q)])
```

---

## 4. Training Pipeline (per VLM, per dataset)

### 4a. Data Collection (prerequisite — already done)

For each (dataset, VLM) pair, you need a CSV/JSON with:
```
question_id, question_text, image_id, fidelity, vlm_answer, ground_truth, is_correct (0/1)
```

This requires running each VLM at each of the 5 fidelities on all questions. This is the expensive step — assumed to be done already.

### 4b. 5-Fold Cross-Validation Setup

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Split at question level (not image level)
question_ids = df['question_id'].unique()
splits = list(kf.split(question_ids))
```

### 4c. Per-Fold Training

For each fold and each fidelity `f`:

```python
# Step 1: Fit TF-IDF on training fold only
tfidf = TfidfVectorizer(lowercase=True)
tfidf.fit(train_df['question_text'])

# Step 2: Extract features
X_train = build_features(train_df, tfidf)   # shape: (n_train, n_features)
y_train = train_df[f'correct_{fidelity}'].values  # binary 0/1

# Step 3: Train GBR
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    random_state=42
)
gbr.fit(X_train, y_train)

# Step 4: Isotonic calibration on training fold
# (Use out-of-bag or a held-out portion of train; simple approach: use train fold itself)
raw_scores_train = gbr.predict(X_train)
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(raw_scores_train, y_train)

# Step 5: Predict on test fold
X_test = build_features(test_df, tfidf)
raw_scores_test = gbr.predict(X_test)
p_hat_test = iso.transform(raw_scores_test)   # calibrated P(correct | q, f)
```

**GBR hyperparameter search grid:**
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [2, 3, 5],
}
```
Search within training fold only. Select config maximizing validation accuracy-cost objective.

### 4d. VOI Threshold Tuning

`λ` (cost penalty) and `τ` (decision threshold) are tuned on the training fold:

```python
lambda_grid = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
tau_grid    = [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15]

best_score = -inf
for lam in lambda_grid:
    for tau in tau_grid:
        # Run VOI inference on validation portion of train fold
        acc, cost = evaluate_voila(val_df, p_hats_val, lam, tau, costs)
        # Objective: maximize acc - 0.5 * (cost / 120)   [tune as needed]
        score = acc - 0.5 * (cost / 120)
        if score > best_score:
            best_score = score
            best_lam, best_tau = lam, tau
```

---

## 5. Inference Algorithm (VOILA)

```python
def voila_select_fidelity(q, calibrated_predictors, costs, lam, tau):
    """
    q                    : question string
    calibrated_predictors: dict {fidelity: (tfidf, gbr, iso)} per fidelity
    costs                : dict {fidelity: float}
    lam                  : cost penalty (λ)
    tau                  : VOI threshold (τ)

    Returns: selected fidelity string
    """
    fidelities_ordered = ['caption', 'resize_32x32', 'jpeg_q1', 'jpeg_q10', 'full']

    # Compute calibrated P(correct | q, f) for all fidelities
    p_hat = {}
    for f in fidelities_ordered:
        tfidf, gbr, iso = calibrated_predictors[f]
        phi = build_features_single(q, tfidf)
        raw = gbr.predict([phi])[0]
        p_hat[f] = float(iso.transform([raw])[0])

    # Greedy VOI escalation
    current_f = fidelities_ordered[0]
    for next_f in fidelities_ordered[1:]:
        voi = (p_hat[next_f] - p_hat[current_f]) - lam * costs[next_f]
        if voi > tau:
            current_f = next_f
        else:
            break   # no further escalation

    return current_f
```

**After fidelity selection:**
```python
selected_fidelity = voila_select_fidelity(q, ...)
image = retrieve_image(image_id, fidelity=selected_fidelity)  # first and only retrieval
answer = run_vlm(q, image, model=vlm)                          # first and only VLM call
```

---

## 6. Datasets

### 6a. VQA-v2
- **Source:** `https://visualqa.org/download.html`
- **Split used:** validation split (standard annotations)
- **Question types:** binary yes/no, counting, other (open-ended)
- **Evaluation metric:** VQA accuracy (soft matching against 10 human answers)
- **Size:** ~214K questions on ~40K images (full val); you likely use a subset — document exact size

### 6b. GQA
- **Source:** `https://cs.stanford.edu/people/dorarad/gqa/download.html`
- **Split used:** balanced validation split
- **Question types:** compositional, relational, spatial
- **Evaluation metric:** exact string match accuracy
- **Size:** ~132K questions (balanced val)

### 6c. TextVQA
- **Source:** `https://textvqa.org/dataset/`
- **Split used:** validation split
- **Question types:** OCR-dependent, reading, recognition
- **Evaluation metric:** VQA accuracy (soft matching)
- **Size:** ~5K questions

### 6d. FloodNet
- **Source:** `https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021`
- **Split used:** [CHECK: original train/val split or custom subset?]
- **Question types:** damage assessment (yes/no + counting variants)
- **Evaluation metric:** exact match accuracy
- **NOTE:** FloodNet is not natively open-ended VQA. Document exactly how questions/answers were set up.

### 6e. LoCoMo-VQA (custom benchmark)
- **Source:** Generated from LoCoMo conversational memory dataset using GPT-4o
- **Size:** 3,500 QA pairs
- **Generation prompt:** See `datasets.tex` appendix for exact prompt
- **Question types:** counting, yes/no, spatial, attribute identification, scene understanding (20% each by construction)
- **Important:** Ground truth answers are GPT-4o outputs, not human annotations
- **Split:** OOD evaluation only — VOILA trained on VQA-v2/GQA, evaluated on LoCoMo without retraining

---

## 7. Models

All models run in zero-shot instruction-following mode.

| Model ID | Short name | Parameters | Architecture |
|---|---|---|---|
| `mistralai/Pixtral-12B-2409` | Pixtral-12B | 12B | Dense |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | LLaMA-4-Maverick | 17B×12E (MoE) | Mixture-of-Experts |
| `llava-hf/llava-1.5-7b-hf` | LLaVA-1.5-7B | 7B | Dense |
| `Qwen/Qwen2-VL-72B-Instruct` | Qwen2-VL-72B | 72B | Dense |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Qwen2.5-VL-7B | 7B | Dense |
| `Qwen/QVQ-72B-Preview` or `Qwen/Qwen2.5-VL-72B-Instruct` | Qwen3-VL-235B | 235B-A22B | Mixture-of-Experts |

**Hardware used:** 4× V100-SXM2-32GB for VLM inference, RTX 5000 Ada (32GB) for VOILA training

**VQA inference prompt template (zero-shot):**
```
Question: {question}
Answer with a single word or short phrase only.
```

---

## 8. Evaluation Metrics

### Primary metrics
- **Accuracy (%):** Per dataset, using standard metric (VQA soft-match for VQA-v2/TextVQA, exact match for GQA/FloodNet)
- **Average normalized cost:** Mean of `c(selected_fidelity)` across all test queries. Full resolution = 120 by definition.

### Derived metrics
- **Cost reduction (%):** `(120 - avg_cost) / 120 * 100`
- **Accuracy gap (pp):** `acc_full_resolution - acc_voila`

### Ablation-specific
- **Brier Score:** `mean((p_hat_f - y_f)^2)` — measures calibration quality per fidelity
- **ECE (Expected Calibration Error):** Histogram-based, 10 bins, measures reliability of calibrated probabilities

---

## 9. All Experiments — Automated Script Specification

### Experiment 0: Prerequisites (data collection — run once)

**What:** For each (VLM, dataset, fidelity) triple, run all questions through the VLM and record correctness.

**Output file:** `results/raw/{dataset}/{vlm_name}/{fidelity}.csv`
```
question_id, question_text, image_id, vlm_answer, ground_truth, is_correct
```

**Script:** `scripts/00_collect_vlm_outputs.py`
```python
# Args: --dataset {vqa2,gqa,textvqa,floodnet,locomo}
#       --model {pixtral,llava,llama4,qwen2_7b,qwen2_72b,qwen3_235b}
#       --fidelity {caption,resize_32x32,jpeg_q1,jpeg_q10,full}
#       --output_dir results/raw/
```

---

### Experiment 1: Main VOILA Results

**What:** Train VOILA per (VLM, dataset), run 5-fold CV, report accuracy and average cost.

**Datasets:** VQA-v2, GQA, TextVQA, FloodNet (train/test on same dataset)
**Models:** All 6

**Output:** `results/main/{dataset}/{vlm_name}/voila_results.json`
```json
{
  "accuracy_mean": 68.67,
  "accuracy_std": 0.42,
  "avg_cost_mean": 45.41,
  "avg_cost_std": 1.2,
  "fidelity_distribution": {"caption": 0.32, "jpeg_q1": 0.18, ...},
  "lambda": 0.01,
  "tau": 0.05
}
```

**Fixed-fidelity baselines** (free — just compute from raw data):
For each fidelity `f`, report accuracy when always using `f` (no routing). Cost = `c(f)` for all queries.

**Script:** `scripts/01_run_voila_main.py`
```
Args: --dataset, --model, --raw_dir results/raw/, --output_dir results/main/
```

---

### Experiment 2: Log-prob Cascade Baseline (FrugalGPT-style)

**What:** Sequential post-retrieval cascade. Run VLM at lowest fidelity; if max token log-probability < θ, escalate to next fidelity; repeat.

**Algorithm:**
```python
def logprob_cascade(q, image_id, vlm, fidelities_ordered, costs, theta):
    for f in fidelities_ordered:
        image = retrieve(image_id, f)
        answer, logprob = vlm.generate_with_logprob(q, image)
        if logprob >= theta:
            return answer, f, logprob   # confident enough — stop
    return answer, f, logprob           # exhausted all fidelities

# Tune theta on validation: try theta in [-5, -4, -3, -2, -1, 0]
# (log-prob scale; typically 0 = p=1, -2 ≈ p=0.13, etc.)
```

**Cost counting:** Cascade incurs cost `c(f)` for EVERY fidelity reached, not just the final one.
```python
# Total cost for a query = sum of c(f) for all f accessed before stopping
cascade_cost = sum(c(f) for f in fidelities_accessed)
```

**Output:** `results/baselines/logprob_cascade/{dataset}/{vlm_name}/results.json`
- Same format as Experiment 1 but with cascade cost accounting

**Note on logprob access:** Different VLMs expose log-probs differently:
- HuggingFace models: `outputs.scores` or `outputs.logits` → convert to log-prob of generated tokens
- For answer log-prob: compute mean log-prob of answer tokens only (not the full sequence)
- If log-prob is unavailable for a model's API, use the negative perplexity of the answer as a proxy

**Script:** `scripts/02_run_logprob_cascade.py`
```
Args: --dataset, --model, --theta (default: tune on val), --output_dir
```

---

### Experiment 3: Zero-Shot LLM Router Baseline

**What:** Use GPT-3.5-turbo (or GPT-4o-mini) to predict whether a question needs a high-resolution image.

**Prompt:**
```
You are helping a vision system decide how much visual detail to retrieve.

Question: "{question}"

Does answering this question correctly require a high-resolution image?
Consider: counting fine details, reading text, identifying colors precisely,
or resolving spatial relationships at pixel level.

Answer with exactly one word: yes or no.
```

**Routing rule:**
- `yes` → route to `full` (cost = 120)
- `no` → route to `caption` (cost = 10)

**Variant (3b):** Three-way routing with a medium option:
```
Answer with exactly one word: low, medium, or high.
- low: caption or thumbnail is sufficient
- medium: compressed image needed
- high: full resolution required
```
Map: low → caption, medium → jpeg_q10, high → full.

**Dataset:** Run on 1,000 random questions from VQA-v2 validation and 1,000 from GQA (to control API cost). Use the same random seed for reproducibility.

**Output:** `results/baselines/llm_router/{dataset}/results.json`

**Script:** `scripts/03_run_llm_router.py`
```
Args: --dataset, --n_samples 1000, --seed 42,
      --model gpt-3.5-turbo, --output_dir
      --routing {binary, ternary}
```

---

### Experiment 4: Oracle Upper Bound

**What:** For each query, select the cheapest fidelity that produces a correct answer (computed in hindsight from raw data).

**Algorithm:**
```python
def oracle_fidelity(question_id, raw_data, fidelities_ordered):
    for f in fidelities_ordered:   # ascending cost order
        if raw_data[question_id][f]['is_correct']:
            return f
    return fidelities_ordered[-1]  # full resolution (last resort)
```

**Cost:** `c(oracle_fidelity(q))` — minimum possible cost for a correct answer per query.

**NOTE:** Oracle accuracy ≠ 100% because for some queries no fidelity produces a correct answer (the VLM fails at all fidelities). Oracle accuracy = fraction of queries where at least one fidelity is correct.

**Output:** `results/oracle/{dataset}/{vlm_name}/results.json`
```json
{
  "oracle_accuracy": 79.3,
  "oracle_avg_cost": 28.4,
  "fraction_unsolvable": 0.12,
  "fidelity_distribution": {"caption": 0.38, ...}
}
```

**Script:** `scripts/04_compute_oracle.py` (fast — no VLM needed, uses raw data only)

---

### Experiment 5: OOD Cross-Dataset Generalization

**What:** Train VOILA on one dataset's questions, evaluate on another's questions (same VLM).

**Matrix:** VQA-v2 → GQA, GQA → VQA-v2

**Protocol:**
- Train GBR + isotonic on full training split of source dataset
- Evaluate on full test split of target dataset
- **Important:** TF-IDF vocabulary is trained on source dataset only — this tests whether lexical features transfer across domains

**Output:** `results/ood/{source}→{target}/{vlm_name}/results.json`

**Script:** `scripts/05_run_ood.py`
```
Args: --source {vqa2,gqa}, --target {vqa2,gqa}, --model, --output_dir
```

---

### Experiment 6: LoCoMo OOD Evaluation

**What:** Train VOILA on VQA-v2 + GQA (combined), evaluate on LoCoMo without retraining.

**Protocol:**
- Train GBR + isotonic on VQA-v2 + GQA combined training data
- Apply to LoCoMo questions with NO fidelity-specific retraining
- LoCoMo only has `caption` and `full` natively; intermediate fidelities are synthesized

**Output:** `results/locomo/{vlm_name}/results.json`

---

### Experiment 7: Calibration Ablation

**What:** Compare four calibration strategies, all using GBR predictor, on VQA-v2 and GQA.

| Name | Description |
|---|---|
| `none` | Raw GBR output used directly as probability (uncalibrated) |
| `iso` | GBR + isotonic regression (**VOILA default**) |
| `temp` | GBR + temperature scaling (fit single scalar T on val: p_cal = sigmoid(logit(p)/T)) |
| `acc` | Always use full resolution (accuracy-only upper bound) |
| `thr` | Fixed threshold policy: escalate if raw GBR score < 0.5 |

**Output:** `results/ablations/calibration/{dataset}/{vlm_name}/results.json`
- Report Accuracy, Average Cost, Brier Score, and ECE for each variant

**Script:** `scripts/07_ablation_calibration.py`

---

### Experiment 8: Predictor Architecture Ablation

**What:** Fix calibration to isotonic, vary the regression model.

| Predictor | Library | Notes |
|---|---|---|
| `GBR` | `sklearn.ensemble.GradientBoostingRegressor` | Default |
| `LogReg` | `sklearn.linear_model.LogisticRegression` | Baseline |
| `Ridge` | `sklearn.linear_model.Ridge` | Linear baseline |
| `MLP` | `sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(64,32))` | Nonlinear |

**Evaluated on:** VQA-v2 and GQA, Pixtral-12B only (matches paper Table 4)
**Output:** `results/ablations/predictor/{vlm_name}/results.json`

---

### Experiment 9: Feature Ablation (NEW — not yet in paper)

**What:** Ablate which parts of `φ(q)` drive performance.

| Feature set | Description |
|---|---|
| `tfidf_only` | TF-IDF bag-of-words only |
| `struct_only` | Structured features only (no TF-IDF) |
| `qtype_only` | Question-type indicators only (subset of structured) |
| `tfidf+struct` | Full φ(q) — **VOILA default** |
| `bert` | Sentence-BERT embeddings (`all-MiniLM-L6-v2`) instead of TF-IDF |

**Evaluated on:** VQA-v2, Pixtral-12B
**Metric:** Accuracy, Average Cost, Brier Score

**Where to insert:** New subsection in `new_methodology.tex`

**Script:** `scripts/09_ablation_features.py`

---

### Experiment 10: Observation 2 Quantification (NEW — not yet in paper)

**What:** Measure how unreliable VLM confidence is at low fidelity (validates Observation 2 numerically).

**Metric:**
```python
# At caption fidelity, separate questions into correct and incorrect
logprobs_correct   = [logprob(q, f='caption') for q where is_correct(q, 'caption')]
logprobs_incorrect = [logprob(q, f='caption') for q where NOT is_correct(q, 'caption')]

# Compute overlap
median_correct = np.median(logprobs_correct)
overlap_frac = mean(lp > median_correct for lp in logprobs_incorrect)
# Report: "X% of incorrect caption-fidelity answers have log-prob above
#          the median confidence of correct caption-fidelity answers"
```

**Output:** Add one quantitative sentence to `insights.tex` Observation 2.

**Script:** `scripts/10_obs2_quantification.py`

---

### Experiment 11: Linguistic Feature Importance (NEW — analysis section)

**What:** Extract and report the most predictive question features per fidelity from the GBR.

**Approach:**
```python
# For each fidelity, get GBR feature importances
importances = gbr.feature_importances_   # shape: (n_features,)
feature_names = tfidf.get_feature_names_out().tolist() + STRUCT_FEATURE_NAMES

# Top-20 most important features per fidelity
top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:20]
```

**Expected output:** Table showing top-10 predictive tokens per fidelity:
- For `full` fidelity: expected top tokens like "many", "color", "read", "text", "number", "how many"
- For `caption` fidelity: expected top tokens like "is", "there", "does", "type", "what kind"

**Where to insert:** New `\section{Analysis}` after Results in `results.tex` (or separate `analysis.tex`)

---

## 10. Consolidated Experiment Runner

All experiments should be orchestrated by a single script:

**Script:** `scripts/run_all_experiments.py`

```bash
# Full pipeline
python scripts/run_all_experiments.py \
    --datasets vqa2 gqa textvqa floodnet locomo \
    --models pixtral llava llama4 qwen2_7b qwen2_72b qwen3_235b \
    --experiments main cascade oracle llm_router ood locomo_ood \
                  ablation_calibration ablation_predictor \
                  ablation_features obs2_quant feature_importance \
    --raw_dir results/raw/ \
    --output_dir results/ \
    --n_cv_folds 5 \
    --seed 42
```

**Dependency graph (run in this order):**
```
00_collect_vlm_outputs     (expensive, parallelizable per VLM × dataset × fidelity)
    ↓
04_compute_oracle          (free, uses raw data)
10_obs2_quantification     (uses raw logprobs from VLM outputs)
    ↓
01_run_voila_main          (requires raw correctness labels)
02_run_logprob_cascade     (requires raw logprobs)
05_run_ood                 (requires raw data for source+target)
06_run_locomo              (requires raw data)
07_ablation_calibration    (requires raw data)
08_ablation_predictor      (requires raw data)
09_ablation_features       (requires raw data)
11_feature_importance      (requires trained GBR from exp 1)
    ↓
03_run_llm_router          (independent, only needs question text)
```

---

## 11. Results Aggregation and Table Generation

After all experiments complete, generate the paper tables:

**Script:** `scripts/generate_tables.py`

| Table | Source experiments | Output file |
|---|---|---|
| Main Pareto Figure | exp 1 (VOILA) + fixed-fidelity from raw | `tables/pareto_figure_data.csv` |
| Table 1 (OOD) | exp 5 | `tables/ood_table.tex` |
| Table 2 (LoCoMo) | exp 6 | `tables/locomo_table.tex` |
| Table 3 (Ablation) | exp 7 + new cascade row | `tables/ablation_table.tex` |
| Table 4 (Predictor) | exp 8 | `tables/predictor_table.tex` |
| Oracle comparison | exp 4 | `tables/oracle_comparison.tex` |
| Feature importance | exp 11 | `tables/feature_importance_table.tex` |
| New: LLM Router | exp 3 | added to ablation table |
| New: Log-prob Cascade | exp 2 | added to ablation table |

---

## 12. Key Implementation Notes

1. **VLM correctness labels per fidelity are model-specific.** The GBR for Pixtral is trained on Pixtral's correctness labels. You cannot reuse labels across VLMs.

2. **TF-IDF vocabulary must be fit on training fold only.** Never fit on test data, even the vocabulary. Use `sklearn.pipeline.Pipeline` to ensure this is respected inside CV.

3. **Isotonic regression must be fit on held-out data, not the same data used to train GBR.** Use the 5-fold CV structure: train GBR on folds 1-4, fit isotonic on fold 5 predictions, evaluate on fold 5 held-out.

4. **VOI cost is normalized cost c(f), not raw file size.** The λ parameter is calibrated against the normalized scale (full=120). If you change the cost normalization, retune λ.

5. **Log-prob cascade total cost ≠ VOILA cost.** For the cascade, add up costs at every fidelity accessed. For VOILA, cost = c(selected fidelity) only — no intermediate costs.

6. **LoCoMo has only caption and full fidelities natively.** Intermediate fidelities (resize_32x32, jpeg_q1, jpeg_q10) must be synthesized from the original image. Document this in the LoCoMo section.

7. **FloodNet question setup must be documented.** Check whether the VQA format was constructed from the original damage assessment labels or separately annotated. This affects how correctness is computed.

8. **Random seed = 42 everywhere** for GBR, CV splits, LLM router sampling.

9. **The "VOILA vs VOI" naming:** Use `VOILA` everywhere in code, logs, and output files. The paper had a bug where results tables used `VOI` — this is fixed in the LaTeX but keep code consistent with `VOILA`.
