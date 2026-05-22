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
| `caption` | Text-only BLIP-2 caption, no image pixels | ~2 KB (text) | 7.58 |
| `resize_32x32` | Bilinear downscale to 32x32 pixels, PNG | ~3-5 KB | 15.08 |
| `jpeg_q1` | JPEG quality=1, extreme compression | ~3-8 KB | 37.83 |
| `jpeg_q10` | JPEG quality=10, heavy compression | ~15-50 KB | 53.25 |
| `full` | Original full-resolution image (JPEG/PNG) | ~300-700 KB | 100 |

**Cost formula** (from cost_model appendix):
```
c̃(f) = b_tier(f) + w_bw * r(f)
c(f)  = 100 * c̃(f) / c̃(full)
```

Where:
- `b_tier(f)` = base tier penalty: edge=0, warm=0.3, cold=1.0 (relative)
- `r(f)` = empirical bandwidth ratio = size(f) / size(full)
- `w_bw` = bandwidth weight (default: 1.0)
- Tier assignments: caption/resize → edge (on-device), jpeg_q1/jpeg_q10 → warm (hot cloud), full → cold (archival)

**Implementation note:** Materialize all fidelities on disk before any experiment. Measure `size(f)` per dataset, compute `r(f)` empirically, then fix costs. `c(full)` is always 100 by definition. If using the fixed cost table above, it is the old experimental table `{9.1, 18.1, 45.4, 63.9, 120}` rescaled by `100/120`.

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
Search within training fold only. Select predictor hyperparameters using the
same validation protocol as the routing tradeoff below.

### 4d. VOI Tradeoff Tuning

`λ` (cost penalty) and `τ` (decision threshold) are tuned separately for every
dataset-model pair inside each training fold. Do not reuse a global
accuracy-cost weight across all datasets/models; VLM calibration, answer
quality, and log-probability scale differ substantially by model and benchmark.

Default selection uses a validation Pareto-knee rule:

```python
lambda_grid = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
tau_grid    = [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15]

candidates = []
for lam in lambda_grid:
    for tau in tau_grid:
        # Run VOI inference on validation portion of train fold
        acc, cost = evaluate_voila(val_df, p_hats_val, lam, tau, costs)
        candidates.append((lam, tau, acc, cost))

# Keep non-dominated validation points:
# A point is dominated if another candidate has >= accuracy and <= cost,
# with at least one strict improvement.
frontier = pareto_frontier(candidates)

# Normalize accuracy and cost on this dataset-model validation frontier.
acc_norm    = normalize([p.acc for p in frontier])          # higher is better
saving_norm = normalize([max_cost - p.cost for p in frontier])  # higher is better

# Select the local knee closest to the ideal corner:
# high validation accuracy and low validation cost.
best = argmin_p sqrt((1 - p.acc_norm)^2 + (1 - p.saving_norm)^2)
best_lam, best_tau = best.lambda, best.tau
```

For sensitivity experiments, a fixed scalarized objective can still be reported:
`score = accuracy - α * (cost / 100)`, but `α` must be treated as an
experiment setting, not as the default production choice.

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
- **Evaluation metric in VOILA scripts:** fuzzy recall matching against the stored answer field
- **Size:** ~214K questions on ~40K images (full val); you likely use a subset — document exact size

### 6b. GQA
- **Source:** `https://cs.stanford.edu/people/dorarad/gqa/download.html`
- **Split used:** balanced validation split
- **Question types:** compositional, relational, spatial
- **Evaluation metric in VOILA scripts:** fuzzy recall matching against the stored answer field
- **Size:** ~132K questions (balanced val)

### 6c. TextVQA
- **Source:** `https://textvqa.org/dataset/`
- **Split used:** validation split
- **Question types:** OCR-dependent, reading, recognition
- **Evaluation metric in VOILA scripts:** fuzzy recall matching against each semicolon-separated stored answer
- **Size:** ~5K questions

### 6d. FloodNet
- **Source:** `https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021`
- **Split used:** [CHECK: original train/val split or custom subset?]
- **Question types:** damage assessment (yes/no + counting variants)
- **Evaluation metric in VOILA scripts:** fuzzy recall matching against the stored answer field
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
- **Accuracy (%):** Per dataset-model pair, recompute correctness from the stored
  VLM answer and ground truth using fuzzy recall matching with numeric
  normalization. Default threshold is `rapidfuzz.partial_ratio >= 70`.
- **Average normalized cost:** Mean of `c(selected_fidelity)` across all test queries. Full resolution = 100 by definition.

Correctness matcher:
```python
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def contains_number_like(text):
    if pd.isna(text):
        return False
    text = str(text).lower()
    return bool(re.search(r"\d", text)) or any(word in text for word in [
        "zero","one","two","three","four","five","six","seven","eight","nine","ten",
        "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
        "eighteen","nineteen","twenty","thirty","forty","fifty","sixty",
        "seventy","eighty","ninety","hundred","thousand"
    ])

def normalize_if_numeric(text, gt_text):
    if pd.isna(text):
        return ""
    text = clean_text(text)
    gt_text = clean_text(gt_text)
    if not contains_number_like(gt_text):
        return text
    try:
        return str(w2n.word_to_num(text))
    except Exception:
        words = text.split()
        norm_words = []
        for word in words:
            try:
                norm_words.append(str(w2n.word_to_num(word)))
            except Exception:
                norm_words.append(word)
        return " ".join(norm_words)

def recall_match(gt, pred, threshold=70):
    if pd.isna(gt) or pd.isna(pred):
        return False
    norm_gt = normalize_if_numeric(gt, gt)
    norm_pred = normalize_if_numeric(pred, gt)
    if norm_gt in norm_pred or norm_pred in norm_gt:
        return True
    gt_tokens = set(clean_text(norm_gt).split())
    pred_tokens = set(clean_text(norm_pred).split())
    if gt_tokens and gt_tokens.issubset(pred_tokens):
        return True
    return fuzz.partial_ratio(norm_gt, norm_pred) >= threshold
```

### Derived metrics
- **Cost reduction (%):** `(100 - avg_cost) / 100 * 100`
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

### Experiment 1b: VOILA-CDF Pairwise Router

**What:** Train the same question-only GBR + isotonic predictors as VOILA, but
replace the explicit VOI escalation rule with learned pairwise CDF thresholds
over calibrated probability gains.

For each adjacent fidelity edge `f_i -> f_{i+1}`, define:
```python
gain_i(q) = p_hat(q, f_{i+1}) - p_hat(q, f_i)
score_i(q) = gain_i(q) - lambda_cost * (c(f_{i+1}) - c(f_i))
```

Learn an edge-specific threshold `theta_i` on the validation fold using only
examples that would reach `f_i` under previously learned edge thresholds.

Positive class for edge `f_i -> f_{i+1}`:
```python
helpful_i(q) = (
    answer at f_i is incorrect
    and at least one higher fidelity f_j, j > i, is correct
)
```

CDF threshold:
```python
F_helpful_i(theta) = Pr(score_i <= theta | helpful_i)
F_other_i(theta)   = Pr(score_i <= theta | not helpful_i)

theta_i = argmax_theta F_other_i(theta) - F_helpful_i(theta)
```

Inference:
```python
current = caption
for edge f_i -> f_{i+1}:
    if score_i(q) >= theta_i:
        current = f_{i+1}   # predicted gain is high enough to pay for escalation
    else:
        break
return current
```

**Cost-aware tradeoff:** `lambda_cost` is tuned per dataset-model pair on the
validation fold. For each candidate `lambda_cost`, learn all pairwise CDF
thresholds, evaluate validation accuracy/cost, keep the non-dominated frontier,
and select the Pareto knee exactly as in Section 4d.

**Output:** `results/voila_cdf/{dataset}/{vlm_name}/results.json`

**Why include this:** It tests whether a non-parametric pairwise escalation
boundary over VOILA's calibrated predictions can outperform the analytic VOI
formula while preserving the pre-retrieval constraint.

---

### Experiment 2: Log-prob Cascade Baseline (FrugalGPT-style)

**What:** Sequential post-retrieval cascade. Run VLM at the lowest fidelity; at
each fidelity edge `f_i -> f_{i+1}`, stop if the current answer log-probability
is above that edge's learned threshold `θ_i`; otherwise escalate to the next
fidelity and pay the additional retrieval/inference cost.

**Algorithm:**
```python
def logprob_cascade(q, image_id, vlm, fidelities_ordered, costs, thresholds):
    for i, f in enumerate(fidelities_ordered):
        image = retrieve(image_id, f)
        answer, logprob = vlm.generate_with_logprob(q, image)
        if i == len(fidelities_ordered) - 1:
            return answer, f, logprob
        edge = (f, fidelities_ordered[i + 1])
        if logprob >= thresholds[edge]:
            return answer, f, logprob   # confident enough — stop
    return answer, f, logprob           # exhausted all fidelities

# Tune each edge threshold on validation using correct-vs-incorrect log-prob CDF
# separation at the current fidelity.
#
# For edge f_i -> f_{i+1}, use only validation examples that would reach f_i
# under the thresholds already learned for earlier edges.
#
# Let F_correct_i(theta)   = Pr(logprob_i <= theta | answer at f_i correct)
# Let F_incorrect_i(theta) = Pr(logprob_i <= theta | answer at f_i incorrect)
# Pick:
#     theta_i = argmax_theta F_incorrect_i(theta) - F_correct_i(theta)
#
# This is a one-sided Kolmogorov-Smirnov separation. It learns where incorrect
# answers at the current fidelity accumulate below the threshold much more than
# correct answers. Thresholds are learned per dataset-model pair and per
# fidelity transition; do not share one theta globally across models or edges.
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
Args: --dataset, --model, --output_dir
```

---

### Experiment 3: Zero-Shot LLM Router Baseline

**What:** Use GPT-4o to predict whether a question needs a high-resolution image.

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
- `yes` → route to `full` (cost = 100)
- `no` → route to `caption` (cost = 7.58)

**Variant (3b):** Three-way routing with a medium option:
```
Answer with exactly one word: low, medium, or high.
- low: caption or thumbnail is sufficient
- medium: compressed image needed
- high: full resolution required
```
Map: low → caption, medium → jpeg_q10, high → full.

**Dataset:** Run on 1,000 random questions from VQA-v2 validation and 1,000 from GQA (to control API cost). Use the same random seed for reproducibility.

**Output:** `results/baselines/llm_router/{dataset}/{vlm_name}/results.json`

**Script:** `scripts/03_run_llm_router.py`
```
Args: --dataset, --n_samples 1000, --seed 42,
      --model gpt-4o, --output_dir
      --routing {binary, ternary}
```

---

### Experiment 3c: Supervised BERT Fidelity Router

**What:** Train a supervised question-only classifier per dataset-model pair that
directly predicts the lowest fidelity expected to be sufficient.

**Training labels:** For each training query, assign the cheapest fidelity that
produces a correct answer:
```python
label(q) = argmin_f c(f) such that answer(q, f) is correct
```
If no fidelity answers correctly, assign `full` as the conservative fallback.

**Model:** Encode the question with a local BERT/SentenceTransformer encoder
(default: `sentence-transformers/all-MiniLM-L6-v2`) and train a multiclass
logistic-regression classifier over fidelity labels. The classifier output is a
single fidelity selection; no VLM call is made during this experiment because
answers/correctness are read from the existing CSVs.

**Output:** `results/baselines/bert_router/{dataset}/{vlm_name}/results.json`

**Checkpointing:** The fitted encoder configuration, label encoder, and
classifier are saved under `model_artifacts/bert_router/...`. Existing artifacts
are loaded as checkpoints.

**Fallback feature mode:** If the BERT encoder is not cached locally, the result
is marked skipped by default. For a non-BERT smoke test, run with:
```bash
python scripts/run_all_experiments.py \
    --experiments bert_router \
    --bert_router_features tfidf+struct
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
    --experiments main voila_cdf cascade oracle llm_router ood locomo_ood \
                  ablation_calibration ablation_predictor \
                  ablation_features obs2_quant feature_importance \
    --raw_dir results/raw/ \
    --output_dir results/ \
    --n_cv_folds 5 \
    --seed 42 \
    --n_jobs 4
```

`--n_jobs` parallelizes independent dataset-model jobs and OOD jobs. Start with
`--n_jobs 2` or `--n_jobs 4` for the full CSVs because each worker loads large
files into memory; `--n_jobs 0` uses `CPU count - 1`.

Trained question-only predictors are checkpointed by default. To run the full
suite and save reusable artifacts:
```bash
python scripts/run_all_experiments.py \
    --experiments main voila_cdf \
    --model_artifact_dir model_artifacts \
    --n_jobs 4
```

To reuse saved predictors and only retune/rerun routing under a new cost model:
```bash
python scripts/run_all_experiments.py \
    --experiments main voila_cdf \
    --reuse_models \
    --strict_reuse_models \
    --model_artifact_dir model_artifacts \
    --full_cost 100 \
    --n_jobs 4
```

For application-specific storage models, either rescale the whole table with
`--full_cost 100` or override individual tiers:
```bash
python scripts/run_all_experiments.py \
    --reuse_models \
    --strict_reuse_models \
    --cost caption=4 resize_32x32=9 jpeg_q1=18 jpeg_q10=35 full=100
```

Saved artifacts contain the fitted TF-IDF vectorizer, structured feature
configuration, per-fidelity GBR predictor, and isotonic calibrator for each
fold. They do not contain VLM calls. Existing artifacts are treated as
checkpoints and loaded instead of retrained. Use `--no_save_models` only if you
explicitly do not want artifacts. `--strict_reuse_models` is recommended for
cost-model sweeps because it prevents accidental retraining if an artifact is
missing.

Main VOILA and VOILA-CDF share the same saved GBR/isotonic fold artifacts. The
saved model predicts calibrated per-fidelity correctness probabilities; each
experiment then separately retunes its lambda/threshold policy under the active
cost model.

Do not use JSON-only rescaling for final VOILA tradeoff results when the
relative costs change. JSON rescaling is only a display/unit conversion. For
final numbers, reuse saved artifacts and rerun the selector so lambda, tau, CDF
thresholds, selected fidelities, accuracy, and cost are recomputed under the new
cost model.

For the real GPT-4o router baseline, keep the API key in the local ignored file
`secrets/azure_openai_api_key.txt` or set it in the environment. The runner reads
the file first and falls back to `AZURE_OPENAI_API_KEY`.
```bash
python scripts/run_all_experiments.py \
    --experiments llm_router \
    --llm_router_routing ternary \
    --n_llm_router_samples 1000
```

The LLM router caches full GPT-4o responses once per dataset under
`results/cache/llm_router/datasets/{dataset}_{routing}_n{N}_seed{seed}.json`.
Every VLM/model result for that dataset reuses this same label store, so the API
is not called separately per dataset-model pair. GPT-4o is the default backend.
If an API call fails, the run fails rather than silently substituting heuristic
labels; pass `--llm_router_allow_fallback` only for debugging/non-paper runs.

When GPT-4o routing is part of a larger parallel run, the runner schedules the
router label collection as one job per dataset after the offline experiments.
After that, per-model router scoring is just a cheap read from the saved dataset
label store. This avoids duplicate paid API calls across model pairs.

**Dependency graph (run in this order):**
```
00_collect_vlm_outputs     (expensive, parallelizable per VLM × dataset × fidelity)
    ↓
04_compute_oracle          (free, uses raw data)
10_obs2_quantification     (uses raw logprobs from VLM outputs)
    ↓
01_run_voila_main          (requires raw correctness labels)
01b_run_voila_cdf          (requires raw correctness labels + calibrated VOILA predictors)
02_run_logprob_cascade     (requires raw logprobs)
03c_run_bert_router        (requires raw correctness labels + question text)
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
| VOILA-CDF Comparison | exp 1b | `tables/voila_cdf_table.tex` |
| Table 1 (OOD) | exp 5 | `tables/ood_table.tex` |
| Table 2 (LoCoMo) | exp 6 | `tables/locomo_table.tex` |
| Table 3 (Ablation) | exp 7 + new cascade row | `tables/ablation_table.tex` |
| Table 4 (Predictor) | exp 8 | `tables/predictor_table.tex` |
| Oracle comparison | exp 4 | `tables/oracle_comparison.tex` |
| Feature importance | exp 11 | `tables/feature_importance_table.tex` |
| New: LLM Router | exp 3 | added to ablation table |
| New: BERT Router | exp 3c | `tables/bert_router_table.tex` |
| New: Log-prob Cascade | exp 2 | added to ablation table |

---

## 12. Key Implementation Notes

1. **VLM correctness labels per fidelity are model-specific.** The GBR for Pixtral is trained on Pixtral's correctness labels. You cannot reuse labels across VLMs.

2. **TF-IDF vocabulary must be fit on training fold only.** Never fit on test data, even the vocabulary. Use `sklearn.pipeline.Pipeline` to ensure this is respected inside CV.

3. **Isotonic regression must be fit on held-out data, not the same data used to train GBR.** Use the 5-fold CV structure: train GBR on folds 1-4, fit isotonic on fold 5 predictions, evaluate on fold 5 held-out.

4. **VOI cost is normalized cost c(f), not raw file size.** The λ parameter is calibrated against the normalized scale (full=100). If you change the cost normalization, retune λ.

5. **Log-prob cascade total cost ≠ VOILA cost.** For the cascade, add up costs at every fidelity accessed. For VOILA, cost = c(selected fidelity) only — no intermediate costs.

6. **LoCoMo has only caption and full fidelities natively.** Intermediate fidelities (resize_32x32, jpeg_q1, jpeg_q10) must be synthesized from the original image. Document this in the LoCoMo section.

7. **FloodNet question setup must be documented.** Check whether the VQA format was constructed from the original damage assessment labels or separately annotated. This affects how correctness is computed.

8. **Random seed = 42 everywhere** for GBR, CV splits, LLM router sampling.

9. **The "VOILA vs VOI" naming:** Use `VOILA` everywhere in code, logs, and output files. The paper had a bug where results tables used `VOI` — this is fixed in the LaTeX but keep code consistent with `VOILA`.
