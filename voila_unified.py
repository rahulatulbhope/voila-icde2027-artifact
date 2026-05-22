#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VOILA Unified Experiment Runner
================================
Runs VOILA (GBR + Isotonic calibration + VOI routing) across ALL datasets and models.

Datasets  : vqav2, gqa, textvqa, floodnet, locomo
Models    : pixtral, llava, llama4, qwen7b, qwen72b, qwen235b

Usage:
    # Run everything
    python voila_unified.py

    # Run specific dataset(s)
    python voila_unified.py --datasets vqav2 gqa

    # Run specific model(s)
    python voila_unified.py --models pixtral llava

    # Run one combo
    python voila_unified.py --datasets textvqa --models qwen7b

    # Control lambda / threshold search
    python voila_unified.py --lambda_grid 0.0001 0.001 0.01 --tau_grid 0.1 0.3 0.5

Results are saved to:
    results/<dataset>/<model>/voila_results.json   – per-run metrics
    results/summary_all.csv                        – aggregated table
"""

import os, ast, re, string, json, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

from rapidfuzz import fuzz
try:
    from word2number import w2n
    HAS_W2N = True
except ImportError:
    HAS_W2N = False
    print("[WARN] word2number not installed – numeric normalisation disabled")

from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import brier_score_loss

# ============================================================
# 1. DATASET × MODEL → CSV FILE MAP
# ============================================================

DATA_DIR = Path("files")   # relative to this script

MODEL_CSV_MAP = {
    "vqav2": {
        "pixtral":  "vqav2/16k_log_prob_vqa_compression_results_pixtraltop_5000_1.csv",
        "llava":    "vqav2/16k_log_prob_vqa_compression_results_llava_all_1.csv",
        "llama4":   "vqav2/16k_log_prob_vqa_compression_results_llama4top_5000_1.csv",
        "qwen7b":   "vqav2/16k_log_prob_vqa_compression_results_qwen7b_top_5000_1.csv",
        "qwen72b":  "vqav2/16k_log_prob_vqa_compression_results_qwentop_5000_1.csv",
        "qwen235b": "vqav2/16k_log_prob_vqa_compression_results_qwen_235b_5000_1.csv",
    },
    "gqa": {
        "pixtral":  "gqa/log_prob_gqa_compression_results_pixtral_all_1.csv",
        "llava":    "gqa/log_prob_gqa_compression_results_llava_all.csv",
        "llama4":   "gqa/log_prob_gqa_compression_results_llama4_all_2.csv",
        "qwen7b":   "gqa/log_prob_gqa_compression_results_qwen2vl_all.csv",
        "qwen72b":  "gqa/log_prob_gqa_compression_results_qwen_all_1.csv",
        "qwen235b": "gqa/log_prob_gqa_compression_results_qwen_235b_all_1.csv",
    },
    "textvqa": {
        "pixtral":  "textvqa/log_prob_textvqa_compression_results_pixtral_top_5000_0_34602_batch0.csv",
        "llava":    "textvqa/log_prob_textvqa_compression_results_llava_top_5000_0_34602_batch0.csv",
        "llama4":   "textvqa/log_prob_textvqa_compression_results_meta_llama_llama_4_maverick_17b_128e_instruct_fp8_top_5000_0_10000_batch0.csv",
        "qwen7b":   "textvqa/log_prob_textvqa_compression_results_qwen_top_5000_0_34602_batch0_full.csv",
        "qwen72b":  "textvqa/log_prob_textvqa_compression_results_qwen_qwen2.5_vl_72b_instruct_top_5000_0_10000_batch0.csv",
        "qwen235b": "textvqa/log_prob_textvqa_compression_results_qwen_qwen3_vl_235b_a22b_instruct_top_5000.csv",
    },
    "floodnet": {
        "pixtral":  "floodnet/flood_net_compression_results_pixtral_all_4511.csv",
        "llava":    "floodnet/flood_net_compression_results_llava_all.csv",
        "llama4":   "floodnet/flood_net_compression_results_llama_4_all.csv",
        "qwen7b":   "floodnet/flood_net_compression_results_qwen2vl_all_7b.csv",
        "qwen72b":  "floodnet/flood_net_compression_results_qwen2vl_all.csv",
        "qwen235b": "floodnet/flood_net_compression_results_qwen2vl_all_235B.csv",
    },
    "locomo": {
        "pixtral":  "locomo/locomo_compression_results_Pixtral-12B-2409_0_2275_1.csv",
        "qwen7b":   "locomo/locomo_compression_results_Qwen2.5-VL-7B-Instruct_0_2275.csv",
    },
}

# Dataset-level configuration
DATASET_CONFIGS = {
    "vqav2": {
        "question_id_col": "question_id",
        "question_col":    "question",
        "gt_col":          "ground_truth",
        "gt_type":         "single",
        # Preferred caption column in descending priority; first one found is used
        "caption_candidates": ["caption_set2", "blip_caption"],
        # Canonical fidelity → actual CSV column suffix for standard datasets
        "col_remap":  {},   # empty = use canonical names directly
        "lpd_prefix": "log_prob_dist_",
        "costs": {"caption": 10.0, "resize_32x32": 20.0, "jpeg_q1": 50.0, "jpeg_q10": 70.0, "original": 100.0},
    },
    "gqa": {
        "question_id_col": "question_id",
        "question_col":    "question",
        "gt_col":          "ground_truth",
        "gt_type":         "single",
        "caption_candidates": ["caption_set2", "caption1", "caption2"],
        "col_remap":  {},
        "lpd_prefix": "log_prob_dist_",
        "costs": {"caption": 10.0, "resize_32x32": 20.0, "jpeg_q1": 50.0, "jpeg_q10": 70.0, "original": 100.0},
    },
    "textvqa": {
        "question_id_col": "question_id",
        "question_col":    "question",
        "gt_col":          "ground_truth_answers",
        "gt_type":         "semicolon_list",   # "ans1; ans2; ans3; ..."
        "caption_candidates": ["caption_detailed", "caption_confident"],
        "col_remap":  {},
        "lpd_prefix": "log_prob_dist_",
        "costs": {"caption": 11.29, "resize_32x32": 20.37, "jpeg_q1": 65.33, "jpeg_q10": 75.37, "original": 100.0},
    },
    "floodnet": {
        "question_id_col": None,   # no question_id column; row index used
        "question_col":    "question",
        "gt_col":          "ground_truth",
        "gt_type":         "single",
        "caption_candidates": ["caption1", "caption2", "caption3"],
        "col_remap":  {},
        "lpd_prefix": "log_prob_dist_",
        "costs": {"caption": 5.0, "resize_32x32": 10.0, "jpeg_q1": 20.0, "jpeg_q10": 50.0, "original": 100.0},
    },
    "locomo": {
        "question_id_col": "question_id",
        "question_col":    "question",
        "gt_col":          "gt_answer",
        "gt_type":         "single",
        "caption_candidates": ["blip_caption"],
        # Canonical fidelity → actual CSV tag (used to build ans_/prob_/dist_ columns)
        "col_remap":  {
            "original":   "orig",
            "jpeg_q1":    "jpeg_extreme",
            "jpeg_q10":   "jpeg_moderate",
        },
        "lpd_prefix": "dist_",   # LoCoMo uses "dist_" not "log_prob_dist_"
        "costs": {"caption": 10.0, "resize_32x32": 20.0, "jpeg_q1": 50.0, "jpeg_q10": 70.0, "original": 100.0},
    },
}

# Canonical fidelity ordering (cheapest → most expensive)
CANONICAL_FIDELITIES = ["caption", "resize_32x32", "jpeg_q1", "jpeg_q10", "original"]

# ============================================================
# 2. TEXT & MATCHING UTILITIES
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def _contains_number_like(text):
    if pd.isna(text):
        return False
    text = str(text).lower()
    number_words = {
        "zero","one","two","three","four","five","six","seven","eight","nine","ten",
        "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
        "eighteen","nineteen","twenty","thirty","forty","fifty","sixty",
        "seventy","eighty","ninety","hundred","thousand",
    }
    return bool(re.search(r"\d", text)) or any(w in text for w in number_words)

def _normalize_numeric(text, gt_text):
    if pd.isna(text):
        return ""
    text = clean_text(text)
    gt_text = clean_text(gt_text)
    if not _contains_number_like(gt_text):
        return text
    if HAS_W2N:
        try:
            return str(w2n.word_to_num(text))
        except Exception:
            pass
        words = text.split()
        norm = []
        for w in words:
            try:
                norm.append(str(w2n.word_to_num(w)))
            except Exception:
                norm.append(w)
        return " ".join(norm)
    return text

def recall_match(gt, pred, threshold=70):
    """Fuzzy recall-based correctness check (used for all datasets)."""
    if pd.isna(gt) or pd.isna(pred):
        return False
    norm_gt   = _normalize_numeric(gt, gt)
    norm_pred = _normalize_numeric(pred, gt)
    if norm_gt in norm_pred or norm_pred in norm_gt:
        return True
    gt_tokens   = set(clean_text(norm_gt).split())
    pred_tokens = set(clean_text(norm_pred).split())
    if gt_tokens and gt_tokens.issubset(pred_tokens):
        return True
    return fuzz.partial_ratio(norm_gt, norm_pred) >= threshold

def is_correct(gt, pred, gt_type="single"):
    """
    Unified correctness check.
    gt_type='single'         → single ground-truth string
    gt_type='semicolon_list' → '; '-separated list (TextVQA) – correct if any matches
    """
    if gt_type == "semicolon_list":
        answers = [a.strip() for a in str(gt).split(";") if a.strip()]
        return any(recall_match(a, pred) for a in answers)
    return recall_match(gt, pred)

# ============================================================
# 3. COLUMN RESOLUTION
# ============================================================

def resolve_fidelity_specs(df_cols, dataset_cfg):
    """
    Return a list of fidelity dicts, one per canonical fidelity, with resolved
    column names from the actual CSV.

    Each dict has:
        canonical  : canonical fidelity name (e.g. "jpeg_q1")
        actual_tag : actual CSV suffix       (e.g. "jpeg_extreme" for LoCoMo)
        ans_col    : column name for answer  (e.g. "ans_jpeg_extreme")
        prob_col   : column name for prob
        lpd_col    : column name for log-prob distribution
    """
    col_remap   = dataset_cfg.get("col_remap", {})
    lpd_prefix  = dataset_cfg.get("lpd_prefix", "log_prob_dist_")
    cap_cands   = dataset_cfg.get("caption_candidates", [])

    # Resolve caption column
    cap_tag = None
    for cand in cap_cands:
        if f"ans_{cand}" in df_cols:
            cap_tag = cand
            break
    if cap_tag is None:
        raise ValueError(f"No caption column found. Tried: {cap_cands}")

    specs = []
    for canon in CANONICAL_FIDELITIES:
        if canon == "caption":
            actual = cap_tag
        else:
            actual = col_remap.get(canon, canon)

        ans_col  = f"ans_{actual}"
        prob_col = f"prob_{actual}"
        lpd_col  = f"{lpd_prefix}{actual}"

        if ans_col not in df_cols:
            print(f"  [SKIP] fidelity '{canon}' -> '{actual}': '{ans_col}' not found")
            continue

        specs.append(dict(
            canonical=canon,
            actual_tag=actual,
            ans_col=ans_col,
            prob_col=prob_col,
            lpd_col=lpd_col,
        ))
    return specs

# ============================================================
# 4. FEATURE EXTRACTION
# ============================================================

def extract_structured_features(df, question_col="question"):
    """
    Structured lexical features as defined in voila_exp.md §3b.
    Returns a DataFrame with one row per question.
    """
    feat = pd.DataFrame(index=df.index)
    q = df[question_col].fillna("").astype(str).str.lower()

    feat["len_words"]     = q.str.split().apply(len)
    feat["len_chars"]     = q.str.len()
    feat["has_digit"]     = q.str.contains(r"\d").astype(int)
    feat["has_count"]     = q.str.contains(r"\bhow many\b|number of|count|total|few\b").astype(int)
    feat["is_yesno"]      = q.str.match(
        r"^(is |are |was |were |does |do |did |has |have |can |could )").astype(int)
    feat["has_color"]     = q.str.contains(r"\bcolor|colour|colored\b").astype(int)
    feat["has_spatial"]   = q.str.contains(
        r"\bwhere|location|position|left|right|behind|front|above|below|beside|next\b").astype(int)
    feat["has_ocr"]       = q.str.contains(
        r"\bread|text|written|say|says|write|letter|word|sign|label\b").astype(int)
    feat["has_wh"]        = q.str.contains(r"\bwhat|which|who|whom\b").astype(int)
    feat["starts_how_many"] = q.str.startswith("how many").astype(int)
    feat["starts_how"]    = q.str.startswith("how ").astype(int)
    feat["has_type"]      = q.str.contains(r"\btype|kind|brand|model|species\b").astype(int)
    feat["has_action"]    = q.str.contains(r"\baction|doing|activity|happen\b").astype(int)
    feat["num_qmarks"]    = q.str.count(r"\?")
    return feat

def extract_tfidf_features(df_train, df_test, question_col="question", max_features=300):
    """
    Fit TF-IDF on train only, transform both train and test.
    Returns (train_df, test_df, vectorizer).
    """
    tfidf = TfidfVectorizer(lowercase=True, max_features=max_features, ngram_range=(1, 2))
    train_q = df_train[question_col].fillna("").astype(str)
    test_q  = df_test[question_col].fillna("").astype(str)

    Xtr = tfidf.fit_transform(train_q).toarray()
    Xte = tfidf.transform(test_q).toarray()

    cols = [f"tfidf_{i}" for i in range(Xtr.shape[1])]
    return (
        pd.DataFrame(Xtr, columns=cols, index=df_train.index),
        pd.DataFrame(Xte, columns=cols, index=df_test.index),
        tfidf,
    )

def build_features(df_struct_feat, df_tfidf_feat):
    """Concatenate structured + TF-IDF features into a single aligned DataFrame."""
    return pd.concat([df_struct_feat, df_tfidf_feat], axis=1)

# ============================================================
# 5. LOG-PROB UTILITIES
# ============================================================

def parse_logprob_dist(x):
    """
    Parse log-prob distributions.  Handles two formats:
      - Standard: list of {"token": ..., "logprob": <float>} dicts (VQAv2/GQA/TextVQA/FloodNet)
      - LoCoMo:   {"avg_logprob": <float>, "token_logprobs": [<float>, ...]} dict
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.array([], dtype=float)
    if isinstance(x, str):
        try:
            obj = ast.literal_eval(x)
        except Exception:
            return np.array([], dtype=float)
    else:
        obj = x
    # --- LoCoMo dict format ---
    if isinstance(obj, dict):
        # Prefer avg_logprob directly
        if "avg_logprob" in obj:
            v = obj["avg_logprob"]
            if v is not None and np.isfinite(float(v)):
                return np.array([float(v)], dtype=float)
        # Fallback to token_logprobs list
        if "token_logprobs" in obj:
            tl = obj["token_logprobs"]
            if isinstance(tl, (list, tuple)) and tl:
                vals = [float(t) for t in tl if t is not None and np.isfinite(float(t))]
                if vals:
                    return np.array(vals, dtype=float)
        return np.array([], dtype=float)
    # --- Standard list format ---
    if not isinstance(obj, (list, tuple)) or len(obj) == 0:
        return np.array([], dtype=float)
    lps = []
    for item in obj:
        if isinstance(item, dict) and "logprob" in item:
            lp = item["logprob"]
            if lp is not None and np.isfinite(lp):
                lps.append(float(lp))
        elif isinstance(item, (int, float)) and np.isfinite(item):
            lps.append(float(item))   # plain float list
    return np.array(lps, dtype=float)

def mean_logprob_score(row, prob_col, lpd_col):
    """Return mean log-prob token score; fall back to prob_col; NaN if unavailable."""
    arr = parse_logprob_dist(row.get(lpd_col))
    if arr.size > 0:
        return float(arr.mean())
    p = row.get(prob_col)
    if p is not None:
        try:
            p = float(p)
            if np.isfinite(p):
                return p
        except Exception:
            pass
    return np.nan

def extract_logprob_column(df, prob_col, lpd_col):
    """
    Vectorised logprob extraction for an entire DataFrame column.
    Falls back to prob_col when lpd_col is unavailable/unparseable.
    Returns a np.ndarray of float32, NaN where unavailable.
    """
    n = len(df)
    out = np.full(n, np.nan, dtype=np.float64)

    # --- Try lpd_col first (vectorised for common formats) ---
    if lpd_col in df.columns:
        raw = df[lpd_col].values
        for i, x in enumerate(raw):
            arr = parse_logprob_dist(x)
            if arr.size > 0:
                out[i] = arr.mean()

    # --- Fall back to prob_col for rows still NaN ---
    nan_mask = np.isnan(out)
    if nan_mask.any() and prob_col in df.columns:
        prob_vals = pd.to_numeric(df[prob_col], errors="coerce").values
        fill_mask = nan_mask & np.isfinite(prob_vals)
        out[fill_mask] = prob_vals[fill_mask]

    return out.astype(np.float32)

# ============================================================
# 6. DATA LOADING
# ============================================================

def load_csv(dataset_name, model_name, data_dir=DATA_DIR):
    rel = MODEL_CSV_MAP.get(dataset_name, {}).get(model_name)
    if rel is None:
        return None, None
    path = data_dir / rel
    if not path.exists():
        print(f"  [MISSING] {path}")
        return None, None
    df = pd.read_csv(path, header=0, low_memory=False)
    return df, str(path)

def preprocess_dataframe(df, dataset_cfg):
    """
    Standardise the DataFrame:
    - Add synthetic 'question_id' for FloodNet (none exists).
    - Return a clean copy.
    """
    df = df.copy()
    qid_col = dataset_cfg["question_id_col"]
    if qid_col is None:
        df["question_id"] = np.arange(len(df))
        dataset_cfg = dict(dataset_cfg, question_id_col="question_id")
    elif qid_col not in df.columns:
        df["question_id"] = np.arange(len(df))
        dataset_cfg = dict(dataset_cfg, question_id_col="question_id")
    return df, dataset_cfg

# ============================================================
# 7. CORRECTNESS LABELS
# ============================================================

def compute_correctness_labels(df, fidelity_specs, gt_col, gt_type):
    """
    Add boolean `correct_<canonical>` column for every fidelity.
    Uses vectorised apply for speed.
    Returns modified df.
    """
    df = df.copy()
    gt_vals = df[gt_col].values
    for spec in fidelity_specs:
        canon   = spec["canonical"]
        ans_col = spec["ans_col"]
        if ans_col not in df.columns:
            df[f"correct_{canon}"] = False
            continue
        ans_vals = df[ans_col].values
        df[f"correct_{canon}"] = [
            is_correct(g, a, gt_type)
            for g, a in zip(gt_vals, ans_vals)
        ]
    return df

# ============================================================
# 8. VOILA TRAINING  (GBR + Isotonic per fidelity)
# ============================================================

def fit_voila(df_train, df_test, fidelity_specs, question_col,
              tfidf_train, tfidf_test, seed=42):
    """
    For each fidelity:
    1. Build feature matrix (structured + TF-IDF + log-prob score).
    2. Train GBR on training fold.
    3. Fit IsotonicRegression on GBR's training predictions.
    4. Predict calibrated p_hat on test fold.

    Returns df_test with p_hat_<canonical> columns added.
    """
    df_train = df_train.copy().reset_index(drop=True)
    df_test  = df_test.copy().reset_index(drop=True)

    # Structured features (computed per split)
    sf_train = extract_structured_features(df_train, question_col)
    sf_test  = extract_structured_features(df_test,  question_col)

    feat_train = build_features(sf_train.reset_index(drop=True),
                                 tfidf_train.reset_index(drop=True))
    feat_test  = build_features(sf_test.reset_index(drop=True),
                                 tfidf_test.reset_index(drop=True))

    calibration_metrics = {}

    for spec in fidelity_specs:
        canon   = spec["canonical"]
        ans_col = spec["ans_col"]
        prob_col = spec["prob_col"]
        lpd_col  = spec["lpd_col"]
        corr_col = f"correct_{canon}"
        phat_col = f"p_hat_{canon}"

        if corr_col not in df_train.columns:
            df_test[phat_col] = np.nan
            continue

        # ---------- Logprob feature (vectorised) ----------
        lp_train = extract_logprob_column(df_train, prob_col, lpd_col)
        lp_test  = extract_logprob_column(df_test,  prob_col, lpd_col)
        lp_train_mean = float(np.nanmean(lp_train)) if not np.all(np.isnan(lp_train)) else 0.0

        # Impute NaN logprob values with training mean (keeps all rows)
        lp_train_filled = np.where(np.isnan(lp_train), lp_train_mean, lp_train)
        lp_te_filled    = np.where(np.isnan(lp_test),  lp_train_mean, lp_test)

        # ---------- Build augmented feature matrices ----------
        X_tr = feat_train.values.copy().astype(np.float32)
        X_tr = np.hstack([X_tr, lp_train_filled.reshape(-1, 1)])

        X_te = feat_test.values.copy().astype(np.float32)
        X_te = np.hstack([X_te, lp_te_filled.reshape(-1, 1)])

        y_tr = df_train[corr_col].values.astype(int)

        # Filter captions with unanswerable/moreinfo
        if "caption" in canon:
            if ans_col in df_train.columns:
                mask = ~df_train[ans_col].astype(str).str.lower().isin(
                    ["unanswerable", "moreinfo"])
                X_tr_f = X_tr[mask]
                y_tr_f = y_tr[mask]
            else:
                X_tr_f, y_tr_f = X_tr, y_tr
        else:
            X_tr_f, y_tr_f = X_tr, y_tr

        # Remove NaN/inf rows
        valid = np.all(np.isfinite(X_tr_f), axis=1)
        X_tr_f = X_tr_f[valid]
        y_tr_f = y_tr_f[valid]

        if len(X_tr_f) < 10 or len(np.unique(y_tr_f)) < 2:
            print(f"    [SKIP] {canon}: not enough data (n={len(X_tr_f)}, "
                  f"classes={np.unique(y_tr_f)})")
            df_test[phat_col] = np.nan
            continue

        # ---------- GBR ----------
        gbr = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            min_samples_split=2, random_state=seed
        )
        gbr.fit(X_tr_f, y_tr_f)

        # ---------- Isotonic on training predictions ----------
        raw_tr = gbr.predict(X_tr_f)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_tr, y_tr_f)

        # ---------- Predict on test ----------
        X_te_clean = X_te.copy()
        X_te_clean = np.where(np.isfinite(X_te_clean), X_te_clean, 0.0)
        raw_te = gbr.predict(X_te_clean)
        df_test[phat_col] = iso.transform(raw_te)

        # Brier score on test
        corr_te = df_test[corr_col].values.astype(int) if corr_col in df_test.columns else None
        if corr_te is not None:
            brier = brier_score_loss(corr_te, df_test[phat_col].clip(0, 1))
        else:
            brier = np.nan

        calibration_metrics[canon] = {
            "brier": round(float(brier), 5),
            "base_rate": round(float(y_tr_f.mean()), 4),
            "n_train": int(len(X_tr_f)),
            "n_test": int(len(df_test)),
        }
        print(f"    [{canon}] Brier={brier:.4f}  base={y_tr_f.mean():.3f}  n_train={len(X_tr_f)}")

    return df_test, calibration_metrics

# ============================================================
# 9. VOI ROUTING POLICY
# ============================================================

def run_voi_policy(df_test, fidelity_specs, costs, lambda_cost, tau):
    """
    Incremental VOI fidelity selection (vectorised).
    For each row starts at the cheapest fidelity with a valid p_hat and
    escalates as long as incremental VOI > 0.
    """
    df = df_test.copy()

    ordered = sorted(
        [s for s in fidelity_specs if f"p_hat_{s['canonical']}" in df.columns],
        key=lambda s: costs.get(s["canonical"], 999)
    )
    if not ordered:
        df["selected_fidelity"] = "unknown"
        df["selected_cost"]     = 100.0
        return df

    n = len(df)
    selected = np.array([ordered[0]["canonical"]] * n, dtype=object)
    cur_p    = np.where(
        df[f"p_hat_{ordered[0]['canonical']}"].fillna(0).values >= tau,
        df[f"p_hat_{ordered[0]['canonical']}"].fillna(0).values,
        0.0
    )
    cur_cost = np.full(n, costs[ordered[0]["canonical"]], dtype=np.float64)

    for nxt in ordered[1:]:
        nxt_canon = nxt["canonical"]
        nxt_cost  = costs[nxt_canon]
        nxt_p_raw = df[f"p_hat_{nxt_canon}"].fillna(0).values
        nxt_p_eff = np.where(nxt_p_raw >= tau, nxt_p_raw, 0.0)

        inc_voi = (nxt_p_eff - cur_p) - lambda_cost * (nxt_cost - cur_cost)
        escalate = inc_voi > 0

        selected[escalate] = nxt_canon
        cur_p[escalate]    = nxt_p_eff[escalate]
        cur_cost[escalate] = nxt_cost

    df["selected_fidelity"] = selected
    df["selected_cost"]     = df["selected_fidelity"].map(costs)
    return df

def tune_lambda_tau(df_train, df_val, fidelity_specs, costs, lambda_grid, tau_grid, gt_col, gt_type):
    """
    Grid-search λ and τ on validation data.
    Objective: accuracy − 0.5 * (avg_cost / max_cost)
    """
    max_cost = max(costs.values())
    best_score = -np.inf
    best_lambda, best_tau = lambda_grid[0], tau_grid[0]

    for lam in lambda_grid:
        for tau in tau_grid:
            df_eval = run_voi_policy(df_val, fidelity_specs, costs, lam, tau)
            acc, avg_cost = _compute_voila_metrics(df_eval, fidelity_specs, gt_col, gt_type, costs)
            score = acc - 0.5 * (avg_cost / max_cost)
            if score > best_score:
                best_score = score
                best_lambda, best_tau = lam, tau

    return best_lambda, best_tau

# ============================================================
# 10. EVALUATION
# ============================================================

def _compute_voila_metrics(df_eval, fidelity_specs, gt_col, gt_type, costs):
    """Returns (accuracy, avg_cost) for VOILA routing."""
    df_eval = df_eval.copy()
    gt_vals  = df_eval[gt_col].values
    sel_vals = df_eval["selected_fidelity"].values

    # Build a quick lookup: canonical -> ans_col
    spec_map = {s["canonical"]: s["ans_col"] for s in fidelity_specs}

    correct = np.zeros(len(df_eval), dtype=bool)
    for i, (gt, sel) in enumerate(zip(gt_vals, sel_vals)):
        ans_col = spec_map.get(sel)
        if ans_col and ans_col in df_eval.columns:
            pred = df_eval.iloc[i][ans_col]
            correct[i] = is_correct(gt, pred, gt_type)

    acc      = float(correct.mean())
    avg_cost = float(df_eval["selected_cost"].mean())
    return acc, avg_cost

def compute_all_metrics(df_test, fidelity_specs, costs, gt_col, gt_type):
    """
    Compute VOILA + all fixed-fidelity baselines + Oracle.
    Returns (results_dict, df_test_with_oracle_cols).
    """
    max_cost = max(costs.values())
    results  = {}
    df_test  = df_test.copy()
    gt_vals  = df_test[gt_col].values

    # ---- VOILA ----
    voila_acc, voila_cost = _compute_voila_metrics(df_test, fidelity_specs, gt_col, gt_type, costs)
    results["voila_accuracy"]          = round(voila_acc * 100, 3)
    results["voila_avg_cost"]          = round(voila_cost, 3)
    results["voila_cost_reduction_pct"] = round((max_cost - voila_cost) / max_cost * 100, 2)
    fdist = df_test["selected_fidelity"].value_counts(normalize=True).to_dict()
    results["voila_fidelity_dist"] = {k: round(v, 4) for k, v in fdist.items()}

    # ---- Fixed fidelity baselines (already computed as correct_<canon> cols) ----
    for spec in fidelity_specs:
        canon    = spec["canonical"]
        corr_col = f"correct_{canon}"
        if corr_col not in df_test.columns:
            continue
        results[f"always_{canon}_accuracy"] = round(float(df_test[corr_col].mean()) * 100, 3)
        results[f"always_{canon}_cost"]     = costs[canon]

    # ---- Oracle ----
    corr_cols = [f"correct_{s['canonical']}" for s in fidelity_specs
                 if f"correct_{s['canonical']}" in df_test.columns]
    ordered_specs = sorted(fidelity_specs, key=lambda s: costs[s["canonical"]])

    oracle_correct = np.zeros(len(df_test), dtype=bool)
    oracle_cost    = np.full(len(df_test), max_cost, dtype=np.float64)
    for spec in ordered_specs:
        cc = f"correct_{spec['canonical']}"
        if cc not in df_test.columns:
            continue
        mask = df_test[cc].values.astype(bool)
        # Only update rows where oracle hasn't succeeded yet and this fidelity succeeds
        update = mask & ~oracle_correct
        oracle_correct |= update
        oracle_cost[update] = costs[spec["canonical"]]

    df_test["oracle_correct"] = oracle_correct
    df_test["oracle_cost"]    = oracle_cost
    results["oracle_accuracy"] = round(float(oracle_correct.mean()) * 100, 3)
    results["oracle_avg_cost"] = round(float(oracle_cost.mean()), 3)

    return results, df_test

# ============================================================
# 11. MAIN EXPERIMENT LOOP
# ============================================================

DEFAULT_LAMBDA_GRID = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
DEFAULT_TAU_GRID    = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

def run_single_experiment(dataset_name, model_name,
                           lambda_grid=DEFAULT_LAMBDA_GRID,
                           tau_grid=DEFAULT_TAU_GRID,
                           test_size=0.2, seed=42,
                           output_dir=Path("results")):
    """
    Full VOILA pipeline for one (dataset, model) pair.
    Returns a results dict.
    """
    print(f"\n{'='*65}")
    print(f"  Dataset={dataset_name}  Model={model_name}")
    print(f"{'='*65}")

    # ---- Load data ----
    df, csv_path = load_csv(dataset_name, model_name)
    if df is None:
        print(f"  [SKIP] No CSV available.")
        return None

    dataset_cfg   = dict(DATASET_CONFIGS[dataset_name])  # shallow copy
    df, dataset_cfg = preprocess_dataframe(df, dataset_cfg)
    qid_col    = dataset_cfg["question_id_col"]
    question_col = dataset_cfg["question_col"]
    gt_col     = dataset_cfg["gt_col"]
    gt_type    = dataset_cfg["gt_type"]
    costs      = dict(dataset_cfg["costs"])

    print(f"  Loaded {len(df)} rows from {csv_path}")

    # ---- Resolve fidelity columns ----
    try:
        fidelity_specs = resolve_fidelity_specs(list(df.columns), dataset_cfg)
    except ValueError as e:
        print(f"  [ERROR] {e}")
        return None

    print(f"  Fidelities resolved: {[s['canonical'] for s in fidelity_specs]}")

    # ---- Compute correctness labels ----
    df = compute_correctness_labels(df, fidelity_specs, gt_col, gt_type)

    # ---- Train / test split at question level ----
    qids = df[qid_col].values
    train_idx, test_val_idx = train_test_split(
        np.arange(len(df)), test_size=test_size + 0.1, random_state=seed
    )
    # Further split test_val into val (for λ/τ tuning) and test
    val_idx, test_idx = train_test_split(
        test_val_idx, test_size=2/3, random_state=seed
    )

    df_train = df.iloc[train_idx].copy()
    df_val   = df.iloc[val_idx].copy()
    df_test  = df.iloc[test_idx].copy()
    print(f"  Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # ---- TF-IDF: fit on train only ----
    tfidf_tr, tfidf_val, tfidf_vec = extract_tfidf_features(df_train, df_val, question_col)
    # Re-use tfidf_vec to transform test
    test_q  = df_test[question_col].fillna("").astype(str)
    Xte     = tfidf_vec.transform(test_q).toarray()
    tfidf_te = pd.DataFrame(
        Xte,
        columns=[f"tfidf_{i}" for i in range(Xte.shape[1])],
        index=df_test.index
    )

    # ---- Fit VOILA calibrators on train; predict on val ----
    print("  Fitting VOILA calibrators (train -> val)...")
    df_val_pred, calib_metrics_val = fit_voila(
        df_train, df_val, fidelity_specs, question_col,
        tfidf_tr, tfidf_val, seed=seed
    )

    # ---- Tune λ and τ on val ----
    print("  Tuning lambda / tau on val set...")
    best_lam, best_tau = tune_lambda_tau(
        df_train, df_val_pred, fidelity_specs, costs,
        lambda_grid, tau_grid, gt_col, gt_type
    )
    print(f"  Best lambda={best_lam}, tau={best_tau}")

    # ---- Re-fit on train+val combined; predict on test ----
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    # Recompute TF-IDF on train+val
    tfidf_tv, tfidf_te2, tfidf_vec2 = extract_tfidf_features(
        df_trainval, df_test, question_col
    )

    print("  Fitting VOILA calibrators (train+val -> test)...")
    df_test_pred, calib_metrics = fit_voila(
        df_trainval, df_test, fidelity_specs, question_col,
        tfidf_tv, tfidf_te2, seed=seed
    )

    # ---- Apply VOI routing ----
    df_test_pred = run_voi_policy(
        df_test_pred, fidelity_specs, costs, best_lam, best_tau
    )

    # ---- Evaluate ----
    metrics, df_test_pred = compute_all_metrics(
        df_test_pred, fidelity_specs, costs, gt_col, gt_type
    )

    # ---- Package results ----
    out = {
        "dataset":        dataset_name,
        "model":          model_name,
        "csv":            csv_path,
        "n_total":        len(df),
        "n_test":         len(df_test),
        "best_lambda":    best_lam,
        "best_tau":       best_tau,
        "calibration":    calib_metrics,
        **metrics,
    }

    # ---- Save results ----
    out_dir = output_dir / dataset_name / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "voila_results.json"
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved -> {result_path}")

    # ---- Print summary ----
    print(f"\n  === RESULTS: {dataset_name} / {model_name} ===")
    print(f"  VOILA     accuracy={out['voila_accuracy']:.1f}%  "
          f"avg_cost={out['voila_avg_cost']:.1f}  "
          f"cost_reduction={out['voila_cost_reduction_pct']:.1f}%")
    for spec in fidelity_specs:
        c = spec["canonical"]
        k = f"always_{c}_accuracy"
        if k in out:
            print(f"  Always {c:12s}  accuracy={out[k]:.1f}%  cost={out.get(f'always_{c}_cost', '?')}")
    print(f"  Oracle    accuracy={out.get('oracle_accuracy', 0):.1f}%  "
          f"avg_cost={out.get('oracle_avg_cost', 0):.1f}")

    return out

# ============================================================
# 12. SUMMARY TABLE
# ============================================================

def build_summary_table(all_results, output_dir=Path("results")):
    """Aggregate all per-run results into a single CSV summary table."""
    rows = []
    for r in all_results:
        if r is None:
            continue
        row = {
            "dataset":           r["dataset"],
            "model":             r["model"],
            "n_test":            r["n_test"],
            "voila_accuracy":    r.get("voila_accuracy"),
            "voila_avg_cost":    r.get("voila_avg_cost"),
            "voila_cost_reduc":  r.get("voila_cost_reduction_pct"),
            "oracle_accuracy":   r.get("oracle_accuracy"),
            "oracle_avg_cost":   r.get("oracle_avg_cost"),
            "best_lambda":       r.get("best_lambda"),
            "best_tau":          r.get("best_tau"),
        }
        # Add fixed-fidelity baselines
        for fname in CANONICAL_FIDELITIES:
            key = f"always_{fname}_accuracy"
            row[key] = r.get(key)
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    out_path = output_dir / "summary_all.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(out_path, index=False)
    print(f"\nSummary table saved -> {out_path}")
    print(df_summary.to_string(index=False))
    return df_summary

# ============================================================
# 13. ENTRYPOINT
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="VOILA Unified Experiment Runner")
    p.add_argument("--datasets", nargs="+",
                   default=list(DATASET_CONFIGS.keys()),
                   choices=list(DATASET_CONFIGS.keys()),
                   help="Datasets to run")
    p.add_argument("--models", nargs="+",
                   default=["pixtral", "llava", "llama4", "qwen7b", "qwen72b", "qwen235b"],
                   help="Models to run")
    p.add_argument("--lambda_grid", nargs="+", type=float,
                   default=DEFAULT_LAMBDA_GRID)
    p.add_argument("--tau_grid", nargs="+", type=float,
                   default=DEFAULT_TAU_GRID)
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Fraction of data used for final test (default 0.2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="results",
                   help="Output directory for results")
    p.add_argument("--data_dir", type=str, default="files",
                   help="Root folder containing dataset sub-folders")
    return p.parse_args()


def main():
    args = parse_args()
    global DATA_DIR
    DATA_DIR = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    all_results = []

    for dataset in args.datasets:
        for model in args.models:
            # Skip combos where no CSV file is configured
            if model not in MODEL_CSV_MAP.get(dataset, {}):
                print(f"  [SKIP] {dataset}/{model}: not in MODEL_CSV_MAP")
                continue

            result = run_single_experiment(
                dataset_name=dataset,
                model_name=model,
                lambda_grid=args.lambda_grid,
                tau_grid=args.tau_grid,
                test_size=args.test_size,
                seed=args.seed,
                output_dir=output_dir,
            )
            all_results.append(result)

    if all_results:
        build_summary_table(all_results, output_dir)
    else:
        print("[WARN] No results produced.")


if __name__ == "__main__":
    main()
