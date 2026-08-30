#!/usr/bin/env python3
"""
VOILA paper experiment runner.

This script runs the experiments described in voila_exp.md over the CSV files
already present under files/. It does not call a VLM. The VLM answers,
probabilities, and log-probability traces are read from the dataset/model CSVs.

Examples
--------
Run the full offline suite:
    python scripts/run_all_experiments.py

Run a quick smoke test:
    python scripts/run_all_experiments.py --datasets vqav2 --models pixtral \
        --experiments main oracle cascade obs2_quant --max_rows 1000 --n_cv_folds 2

Run only the main VOILA CV table:
    python scripts/run_all_experiments.py --experiments main
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import string
import sys
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Keep native math libraries from oversubscribing when --n_jobs launches multiple
# Python workers. Users can override these in the shell before running.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from rapidfuzz import fuzz
except ImportError as exc:  # pragma: no cover
    raise SystemExit("rapidfuzz is required. Install it with: pip install rapidfuzz") from exc

try:
    from word2number import w2n
except ImportError:  # pragma: no cover
    w2n = None

from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.neural_network import MLPRegressor

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "files"
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_AZURE_OPENAI_API_KEY_FILE = ROOT / "secrets" / "azure_openai_api_key.txt"

FIDELITIES = ["caption", "resize_32x32", "jpeg_q1", "jpeg_q10", "full"]
DEFAULT_COSTS = {
    "caption": 7.583333333333333,
    "resize_32x32": 15.083333333333332,
    "jpeg_q1": 37.833333333333336,
    "jpeg_q10": 53.25,
    "full": 100.0,
}


def resolve_costs_from_args(args):
    costs = dict(DEFAULT_COSTS)
    if getattr(args, "full_cost", None) is not None:
        scale = float(args.full_cost) / costs["full"]
        costs = {k: v * scale for k, v in costs.items()}
    for item in getattr(args, "cost", []) or []:
        if "=" not in item:
            raise ValueError(f"Invalid --cost entry '{item}'. Use fidelity=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in costs:
            raise ValueError(f"Unknown fidelity in --cost: {key}")
        costs[key] = float(value)
    return costs


def get_costs(args):
    return dict(getattr(args, "resolved_costs", DEFAULT_COSTS))


def read_secret_file(path):
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def progress_iter(iterable, desc=None, total=None, disable=False):
    if tqdm is None or disable:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)

STRUCT_FEATURE_NAMES = [
    "len_tokens",
    "has_numeric_token",
    "has_counting_word",
    "is_yes_no",
    "has_color_word",
    "has_spatial_word",
    "has_ocr_word",
    "has_wh_word",
    "starts_how_many",
    "starts_how",
    "has_category_word",
    "has_activity_word",
]

QTYPE_FEATURE_NAMES = [
    "has_counting_word",
    "is_yes_no",
    "has_color_word",
    "has_spatial_word",
    "has_ocr_word",
    "has_wh_word",
    "starts_how_many",
    "starts_how",
    "has_category_word",
    "has_activity_word",
]

MODEL_CSV_MAP = {
    "vqav2": {
        "pixtral": "vqav2/16k_log_prob_vqa_compression_results_pixtraltop_5000_1.csv",
        "llava": "vqav2/16k_log_prob_vqa_compression_results_llava_all_1.csv",
        "llama4": "vqav2/16k_log_prob_vqa_compression_results_llama4top_5000_1.csv",
        "qwen7b": "vqav2/16k_log_prob_vqa_compression_results_qwen7b_top_5000_1.csv",
        "qwen72b": "vqav2/16k_log_prob_vqa_compression_results_qwentop_5000_1.csv",
        "qwen235b": "vqav2/16k_log_prob_vqa_compression_results_qwen_235b_5000_1.csv",
    },
    "gqa": {
        "pixtral": "gqa/log_prob_gqa_compression_results_pixtral_all_1.csv",
        "llava": "gqa/log_prob_gqa_compression_results_llava_all.csv",
        "llama4": "gqa/log_prob_gqa_compression_results_llama4_all_2.csv",
        "qwen7b": "gqa/log_prob_gqa_compression_results_qwen2vl_all.csv",
        "qwen72b": "gqa/log_prob_gqa_compression_results_qwen_all_1.csv",
        "qwen235b": "gqa/log_prob_gqa_compression_results_qwen_235b_all_1.csv",
    },
    "textvqa": {
        "pixtral": "textvqa/log_prob_textvqa_compression_results_pixtral_top_5000_0_34602_batch0.csv",
        "llava": "textvqa/log_prob_textvqa_compression_results_llava_top_5000_0_34602_batch0.csv",
        "llama4": "textvqa/log_prob_textvqa_compression_results_meta_llama_llama_4_maverick_17b_128e_instruct_fp8_top_5000_0_10000_batch0.csv",
        "qwen7b": "textvqa/log_prob_textvqa_compression_results_qwen_top_5000_0_34602_batch0_full.csv",
        "qwen72b": "textvqa/log_prob_textvqa_compression_results_qwen_qwen2.5_vl_72b_instruct_top_5000_0_10000_batch0.csv",
        "qwen235b": "textvqa/log_prob_textvqa_compression_results_qwen_qwen3_vl_235b_a22b_instruct_top_5000.csv",
    },
    "floodnet": {
        "pixtral": "floodnet/flood_net_compression_results_pixtral_all_4511.csv",
        "llava": "floodnet/flood_net_compression_results_llava_all.csv",
        "llama4": "floodnet/flood_net_compression_results_llama_4_all.csv",
        "qwen7b": "floodnet/flood_net_compression_results_qwen2vl_all_7b.csv",
        "qwen72b": "floodnet/flood_net_compression_results_qwen2vl_all.csv",
        "qwen235b": "floodnet/flood_net_compression_results_qwen2vl_all_235B.csv",
    },
    "locomo": {
        "pixtral": "locomo/locomo_compression_results_Pixtral-12B-2409_0_2275_1.csv",
        "qwen7b": "locomo/locomo_compression_results_Qwen2.5-VL-7B-Instruct_0_2275.csv",
    },
}

DATASET_CONFIGS = {
    "vqav2": {
        "qid": "question_id",
        "question": "question",
        "gt": "ground_truth",
        "gt_type": "single",
        "caption_candidates": ["caption_set2", "blip_caption", "caption_set1", "caption_set3"],
        "tag_remap": {"full": "original"},
        "lpd_prefix": "log_prob_dist_",
    },
    "gqa": {
        "qid": "question_id",
        "question": "question",
        "gt": "ground_truth",
        "gt_type": "single",
        "caption_candidates": ["caption_set2", "caption2", "caption1", "caption3"],
        "tag_remap": {"full": "original"},
        "lpd_prefix": "log_prob_dist_",
    },
    "textvqa": {
        "qid": "question_id",
        "question": "question",
        "gt": "ground_truth_answers",
        "gt_type": "semicolon_list",
        "caption_candidates": ["caption_detailed", "caption_confident"],
        "tag_remap": {"full": "original"},
        "lpd_prefix": "log_prob_dist_",
    },
    "floodnet": {
        "qid": None,
        "question": "question",
        "gt": "ground_truth",
        "gt_type": "single",
        "caption_candidates": ["caption1", "caption2", "caption3"],
        "tag_remap": {"full": "original"},
        "lpd_prefix": "log_prob_dist_",
    },
    "locomo": {
        "qid": "question_id",
        "question": "question",
        "gt": "gt_answer",
        "gt_type": "single",
        "caption_candidates": ["blip_caption"],
        "tag_remap": {"full": "orig", "jpeg_q1": "jpeg_extreme", "jpeg_q10": "jpeg_moderate"},
        "lpd_prefix": "dist_",
    },
}

EXPERIMENTS = [
    "main",
    "floodnet_stress",
    "vocab_audit",
    "voila_cdf",
    "cascade",
    "llm_router",
    "bert_router",
    "oracle",
    "ood",
    "locomo_ood",
    "ablation_calibration",
    "ablation_predictor",
    "ablation_features",
    "obs2_quant",
    "feature_importance",
    "tables",
]

PERCOM_OFFLINE_EXPERIMENTS = [
    "main",
    "floodnet_stress",
    "vocab_audit",
    "oracle",
    "ablation_calibration",
    "obs2_quant",
    "tables",
]

PAIRWISE_EXPERIMENTS = {
    "oracle",
    "obs2_quant",
    "main",
    "floodnet_stress",
    "vocab_audit",
    "voila_cdf",
    "cascade",
    "llm_router",
    "bert_router",
    "ablation_calibration",
    "ablation_predictor",
    "ablation_features",
    "feature_importance",
}


@dataclass(frozen=True)
class FidelitySpec:
    fidelity: str
    tag: str
    ans_col: str
    prob_col: str
    lpd_col: str


class ConstantModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x):
        return np.full(len(x), self.value, dtype=np.float64)

    def predict_proba(self, x):
        p = np.clip(self.predict(x), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


class NoCalibrator:
    def fit(self, scores, labels):
        return self

    def transform(self, scores):
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


class TemperatureCalibrator:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, scores, labels):
        scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-5, 1.0 - 1e-5)
        labels = np.asarray(labels, dtype=np.float64)
        z = logit(scores)

        def objective(t):
            p = expit(z / t)
            p = np.clip(p, 1e-8, 1.0 - 1e-8)
            return -np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))

        res = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x) if res.success else 1.0
        return self

    def transform(self, scores):
        scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-5, 1.0 - 1e-5)
        return expit(logit(scores) / self.temperature)


class SharedSlopePlattModel:
    """One logistic slope shared by fidelities, with one intercept per fidelity."""

    def __init__(self, fidelities: Sequence[str]):
        self.fidelities = list(fidelities)
        self.model = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")

    def _design(self, scores, fidelity):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        offsets = np.zeros((len(scores), len(self.fidelities)), dtype=np.float64)
        offsets[:, self.fidelities.index(fidelity)] = 1.0
        return np.hstack([scores, offsets])

    def fit(self, scores_by_fidelity, labels_by_fidelity):
        x = np.vstack([self._design(scores_by_fidelity[f], f) for f in self.fidelities])
        y = np.concatenate([np.asarray(labels_by_fidelity[f], dtype=int) for f in self.fidelities])
        if len(np.unique(y)) < 2:
            self.model = ConstantModel(float(np.mean(y)))
            self._constant = True
        else:
            self.model.fit(x, y)
            self._constant = False
        return self

    def transform(self, scores, fidelity):
        x = self._design(scores, fidelity)
        if getattr(self, "_constant", False):
            return self.model.predict(x)
        return self.model.predict_proba(x)[:, 1]


class SharedSlopePlattView:
    """Fidelity-specific serializable view over a shared Platt model."""

    def __init__(self, shared_model, fidelity):
        self.shared_model = shared_model
        self.fidelity = fidelity

    def fit(self, scores, labels):
        return self

    def transform(self, scores):
        return self.shared_model.transform(scores, self.fidelity)


@dataclass
class TrainedFidelityModel:
    model: Any
    calibrator: Any
    y_cal: np.ndarray
    p_cal: np.ndarray
    p_test: Optional[np.ndarray] = None


# Keep serialized artifacts importable when this file is first run as a script
# and later loaded by a separate benchmark or application process.
sys.modules.setdefault("run_all_experiments", sys.modules[__name__])
for _artifact_class in (
    ConstantModel,
    NoCalibrator,
    TemperatureCalibrator,
    SharedSlopePlattModel,
    SharedSlopePlattView,
    TrainedFidelityModel,
):
    _artifact_class.__module__ = "run_all_experiments"


def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower().translate(str.maketrans("", "", string.punctuation)).strip()


def contains_number_like(text):
    if pd.isna(text):
        return False
    t = str(text).lower()
    words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
        "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
        "thousand",
    }
    return bool(re.search(r"\d", t)) or any(w in t for w in words)


def normalize_if_numeric(text, gt_text):
    text = clean_text(text)
    gt_text = clean_text(gt_text)
    if not contains_number_like(gt_text):
        return text
    if w2n is None:
        return text
    try:
        return str(w2n.word_to_num(text))
    except Exception:
        out = []
        for word in text.split():
            try:
                out.append(str(w2n.word_to_num(word)))
            except Exception:
                out.append(word)
        return " ".join(out)


RECALL_MATCH_THRESHOLD = 70


def recall_match(gt, pred, threshold=None):
    if threshold is None:
        threshold = RECALL_MATCH_THRESHOLD
    if pd.isna(gt) or pd.isna(pred):
        return False
    norm_gt = normalize_if_numeric(gt, gt)
    norm_pred = normalize_if_numeric(pred, gt)
    if not norm_gt and not norm_pred:
        return True
    if norm_gt and (norm_gt in norm_pred or norm_pred in norm_gt):
        return True
    gt_tokens = set(norm_gt.split())
    pred_tokens = set(norm_pred.split())
    if gt_tokens and gt_tokens.issubset(pred_tokens):
        return True
    return fuzz.partial_ratio(norm_gt, norm_pred) >= threshold


def is_correct(gt, pred, gt_type):
    if gt_type == "semicolon_list":
        answers = [a.strip() for a in str(gt).split(";") if a.strip()]
        return any(recall_match(a, pred) for a in answers)
    return recall_match(gt, pred)


def parse_logprob_dist(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=np.float64)
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return np.array([], dtype=np.float64)
    if isinstance(value, dict):
        if value.get("avg_logprob") is not None:
            try:
                return np.array([float(value["avg_logprob"])], dtype=np.float64)
            except Exception:
                pass
        vals = value.get("token_logprobs")
        if isinstance(vals, (list, tuple)):
            return np.array([float(v) for v in vals if _finite(v)], dtype=np.float64)
        return np.array([], dtype=np.float64)
    if isinstance(value, (list, tuple)):
        vals = []
        for item in value:
            if isinstance(item, dict) and item.get("logprob") is not None:
                vals.append(item["logprob"])
            elif isinstance(item, (int, float)):
                vals.append(item)
        return np.array([float(v) for v in vals if _finite(v)], dtype=np.float64)
    return np.array([], dtype=np.float64)


def _finite(value):
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def logprob_scores(df, spec: FidelitySpec):
    out = np.full(len(df), np.nan, dtype=np.float64)
    if spec.lpd_col in df.columns:
        for i, value in enumerate(df[spec.lpd_col].values):
            arr = parse_logprob_dist(value)
            if len(arr):
                out[i] = float(np.mean(arr))
    missing = np.isnan(out)
    if missing.any() and spec.prob_col in df.columns:
        probs = pd.to_numeric(df[spec.prob_col], errors="coerce").values
        out[missing & np.isfinite(probs)] = probs[missing & np.isfinite(probs)]
    return out


def resolve_fidelity_specs(columns: Sequence[str], cfg: Dict[str, Any]):
    columns = set(columns)
    tag_remap = cfg.get("tag_remap", {})
    lpd_prefix = cfg.get("lpd_prefix", "log_prob_dist_")

    caption_tag = None
    for candidate in cfg["caption_candidates"]:
        if f"ans_{candidate}" in columns:
            caption_tag = candidate
            break
    if caption_tag is None:
        raise ValueError(f"No caption answer column found among {cfg['caption_candidates']}")

    specs = []
    for fidelity in FIDELITIES:
        tag = caption_tag if fidelity == "caption" else tag_remap.get(fidelity, fidelity)
        ans_col = f"ans_{tag}"
        if ans_col not in columns:
            continue
        specs.append(
            FidelitySpec(
                fidelity=fidelity,
                tag=tag,
                ans_col=ans_col,
                prob_col=f"prob_{tag}",
                lpd_col=f"{lpd_prefix}{tag}",
            )
        )
    return specs


def load_dataset(dataset, model, data_dir, max_rows=None, seed=42, common_keys_csv=None):
    rel = MODEL_CSV_MAP.get(dataset, {}).get(model)
    if rel is None:
        raise FileNotFoundError(f"No CSV configured for dataset={dataset}, model={model}")
    path = Path(data_dir) / rel
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path, low_memory=False)
    source_duplicate_rows = 0
    if common_keys_csv:
        common = pd.read_csv(common_keys_csv, low_memory=False)
        keys = [key for key in ("image_id", "question") if key in df and key in common]
        if len(keys) != 2:
            raise ValueError("Common-population filtering requires image_id and question columns")
        for key in keys:
            df[key] = df[key].astype(str)
            common[key] = common[key].astype(str)
        if common.duplicated(keys).any():
            raise ValueError("Common-population file contains duplicate image/question keys")
        df = df.merge(common[keys], on=keys, how="inner", validate="many_to_one")
        source_duplicate_rows = int(df.duplicated(keys, keep=False).sum())
        if source_duplicate_rows:
            duplicates = df[df.duplicated(keys, keep=False)]
            compare_columns = [column for column in df.columns if column not in keys + ["question_id"]]
            conflicting = duplicates.groupby(keys, dropna=False)[compare_columns].nunique(dropna=False).gt(1).any(axis=1)
            if conflicting.any():
                raise ValueError(
                    f"Found {int(conflicting.sum())} conflicting duplicate image/question pairs in {path}"
                )
            df = df.drop_duplicates(keys, keep="first")
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
    cfg = dict(DATASET_CONFIGS[dataset])
    if cfg["qid"] is None or cfg["qid"] not in df.columns:
        df["question_id"] = np.arange(len(df), dtype=np.int64)
        cfg["qid"] = "question_id"
    specs = resolve_fidelity_specs(df.columns, cfg)
    if not specs:
        raise ValueError(f"No usable fidelity columns in {path}")
    df = compute_correctness(df, specs, cfg)
    df.attrs["source_duplicate_image_question_rows"] = source_duplicate_rows
    return df, cfg, specs, path


def compute_correctness(df, specs, cfg):
    df = df.copy()
    gt_values = df[cfg["gt"]].values
    for spec in specs:
        df[f"correct_{spec.fidelity}"] = [
            is_correct(gt, pred, cfg["gt_type"])
            for gt, pred in zip(gt_values, df[spec.ans_col].values)
        ]
    return df


def structured_features(questions: Sequence[str], names: Optional[Sequence[str]] = None):
    rows = []
    for question in questions:
        q = "" if pd.isna(question) else str(question).lower()
        tokens = q.split()
        row = {
            "len_tokens": float(len(tokens)),
            "has_numeric_token": float(any(t.isdigit() for t in tokens)),
            "has_counting_word": float(any(t in {"many", "much", "number", "count", "few", "total"} for t in tokens)),
            "is_yes_no": float(q.startswith(("is ", "are ", "was ", "were ", "does ", "do ", "did ", "has ", "have ", "can ", "could "))),
            "has_color_word": float(any(t in {"color", "colour", "colored", "coloured"} for t in tokens)),
            "has_spatial_word": float(any(t in {"where", "location", "position", "left", "right", "behind", "front", "above", "below", "beside", "next"} for t in tokens)),
            "has_ocr_word": float(any(t in {"read", "text", "written", "say", "says", "write", "letter", "word", "number", "sign", "label"} for t in tokens)),
            "has_wh_word": float(any(t in {"what", "which", "who", "whom"} for t in tokens)),
            "starts_how_many": float(q.startswith("how many")),
            "starts_how": float(q.startswith("how ")),
            "has_category_word": float(any(t in {"type", "kind", "brand", "model", "species"} for t in tokens)),
            "has_activity_word": float(any(t in {"action", "doing", "activity", "happen", "happening"} for t in tokens)),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if names is not None:
        frame = frame[list(names)]
    return frame.values.astype(np.float32)


def fit_feature_state(train_df, question_col, feature_set, max_tfidf_features):
    train_q = train_df[question_col].fillna("").astype(str).values
    names = []
    state = {
        "question_col": question_col,
        "feature_set": feature_set,
        "max_tfidf_features": max_tfidf_features,
        "tfidf": None,
        "feature_names": None,
    }

    if feature_set in {"tfidf_only", "tfidf+struct"}:
        tfidf = TfidfVectorizer(
            lowercase=True,
            tokenizer=str.split,
            token_pattern=None,
            max_features=max_tfidf_features,
        )
        tfidf.fit(train_q)
        state["tfidf"] = tfidf
        names.extend(tfidf.get_feature_names_out().tolist())

    if feature_set in {"struct_only", "tfidf+struct"}:
        names.extend(STRUCT_FEATURE_NAMES)

    if feature_set == "qtype_only":
        names.extend(QTYPE_FEATURE_NAMES)

    if feature_set == "bert":
        raise RuntimeError(
            "Feature set 'bert' requires sentence-transformers and a local/all-MiniLM-L6-v2 model. "
            "Install/cache it first, then rerun this ablation."
        )

    if not names:
        raise ValueError(f"Unknown feature_set={feature_set}")

    state["feature_names"] = names
    return state


def transform_features_with_state(df, feature_state):
    question_col = feature_state["question_col"]
    feature_set = feature_state["feature_set"]
    questions = df[question_col].fillna("").astype(str).values
    blocks = []

    if feature_set in {"tfidf_only", "tfidf+struct"}:
        tfidf = feature_state["tfidf"]
        blocks.append(tfidf.transform(questions).astype(np.float32).toarray())

    if feature_set in {"struct_only", "tfidf+struct"}:
        blocks.append(structured_features(questions, STRUCT_FEATURE_NAMES))

    if feature_set == "qtype_only":
        blocks.append(structured_features(questions, QTYPE_FEATURE_NAMES))

    if not blocks:
        raise ValueError(f"Unknown feature_set={feature_set}")

    return np.hstack(blocks).astype(np.float32)


def build_feature_matrices(train_df, cal_df, test_df, question_col, feature_set, max_tfidf_features):
    feature_state = fit_feature_state(train_df, question_col, feature_set, max_tfidf_features)
    return (
        transform_features_with_state(train_df, feature_state),
        transform_features_with_state(cal_df, feature_state),
        transform_features_with_state(test_df, feature_state),
        feature_state["feature_names"],
    )


def build_predictor(name, seed):
    if name == "GBR":
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            random_state=seed,
        )
    if name == "LogReg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if name == "Ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if name == "MLP":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            early_stopping=True,
            random_state=seed,
        )
    raise ValueError(f"Unknown predictor={name}")


def raw_predict(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return np.clip(model.predict(x), 0.0, 1.0)


def build_calibrator(name):
    if name == "none":
        return NoCalibrator()
    if name == "iso":
        return IsotonicRegression(out_of_bounds="clip")
    if name == "temp":
        return TemperatureCalibrator()
    raise ValueError(f"Unknown calibrator={name}")


def train_fidelity_models(
    train_df,
    cal_df,
    test_df,
    specs,
    cfg,
    predictor_name="GBR",
    calibration="iso",
    feature_set="tfidf+struct",
    max_tfidf_features=500,
    seed=42,
    return_artifact=False,
):
    feature_state = fit_feature_state(train_df, cfg["question"], feature_set, max_tfidf_features)
    x_train = transform_features_with_state(train_df, feature_state)
    x_cal = transform_features_with_state(cal_df, feature_state)
    x_test = transform_features_with_state(test_df, feature_state)
    feature_names = feature_state["feature_names"]
    trained: Dict[str, TrainedFidelityModel] = {}
    cal_pred = pd.DataFrame(index=cal_df.index)
    test_pred = pd.DataFrame(index=test_df.index)

    raw_by_fidelity = {}
    labels_by_fidelity = {}
    models_by_fidelity = {}
    for spec in specs:
        corr_col = f"correct_{spec.fidelity}"
        if corr_col not in train_df.columns or corr_col not in cal_df.columns:
            continue
        y_train = train_df[corr_col].astype(int).values
        y_cal = cal_df[corr_col].astype(int).values
        if len(np.unique(y_train)) < 2:
            model = ConstantModel(float(np.mean(y_train)))
        else:
            model = clone(build_predictor(predictor_name, seed))
            model.fit(x_train, y_train)

        models_by_fidelity[spec.fidelity] = model
        labels_by_fidelity[spec.fidelity] = y_cal
        raw_by_fidelity[spec.fidelity] = {
            "cal": raw_predict(model, x_cal),
            "test": raw_predict(model, x_test),
        }

    shared_calibrator = None
    if calibration == "shared_platt":
        shared_calibrator = SharedSlopePlattModel(list(raw_by_fidelity))
        shared_calibrator.fit(
            {f: values["cal"] for f, values in raw_by_fidelity.items()},
            labels_by_fidelity,
        )

    for spec in specs:
        if spec.fidelity not in raw_by_fidelity:
            continue
        model = models_by_fidelity[spec.fidelity]
        y_cal = labels_by_fidelity[spec.fidelity]
        raw_cal = raw_by_fidelity[spec.fidelity]["cal"]
        raw_test = raw_by_fidelity[spec.fidelity]["test"]
        if shared_calibrator is not None:
            calibrator = SharedSlopePlattView(shared_calibrator, spec.fidelity)
        else:
            calibrator = build_calibrator(calibration)
            if calibration == "iso" and len(np.unique(y_cal)) < 2:
                calibrator = NoCalibrator()
            calibrator.fit(raw_cal, y_cal)
        p_cal = np.clip(calibrator.transform(raw_cal), 0.0, 1.0)
        p_test = np.clip(calibrator.transform(raw_test), 0.0, 1.0)
        cal_pred[f"p_{spec.fidelity}"] = p_cal
        test_pred[f"p_{spec.fidelity}"] = p_test
        trained[spec.fidelity] = TrainedFidelityModel(model, calibrator, y_cal, p_cal, p_test)

    if return_artifact:
        artifact = {
            "feature_state": feature_state,
            "feature_names": feature_names,
            "trained": trained,
            "predictor_name": predictor_name,
            "calibration": calibration,
            "feature_set": feature_set,
            "max_tfidf_features": max_tfidf_features,
            "seed": seed,
        }
        return trained, cal_pred, test_pred, feature_names, artifact
    return trained, cal_pred, test_pred, feature_names


def predict_with_artifact(artifact, cal_df, test_df):
    feature_state = artifact["feature_state"]
    x_cal = transform_features_with_state(cal_df, feature_state)
    x_test = transform_features_with_state(test_df, feature_state)
    cal_pred = pd.DataFrame(index=cal_df.index)
    test_pred = pd.DataFrame(index=test_df.index)
    for fidelity, fitted in artifact["trained"].items():
        raw_cal = raw_predict(fitted.model, x_cal)
        raw_test = raw_predict(fitted.model, x_test)
        cal_pred[f"p_{fidelity}"] = np.clip(fitted.calibrator.transform(raw_cal), 0.0, 1.0)
        test_pred[f"p_{fidelity}"] = np.clip(fitted.calibrator.transform(raw_test), 0.0, 1.0)
    return cal_pred, test_pred


def fold_artifact_path(base_dir, dataset, model, experiment, fold, predictor, calibration, feature_set, max_tfidf_features, seed):
    safe_feature = feature_set.replace("+", "_plus_")
    name = (
        f"fold{fold:02d}_{predictor}_{calibration}_{safe_feature}_"
        f"tfidf{max_tfidf_features}_seed{seed}.joblib"
    )
    return Path(base_dir) / experiment / dataset / model / name


def save_model_artifact(path, artifact, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"metadata": metadata, "artifact": artifact}, path, compress=3)


def load_model_artifact(path):
    payload = joblib.load(path)
    return payload["metadata"], payload["artifact"]


def missing_artifact_error(path):
    return FileNotFoundError(
        f"Missing model artifact for --reuse_models: {path}. "
        "Run once with --save_models using the same dataset/model/fold/predictor/"
        "calibration/features/seed settings, or remove --strict_reuse_models to allow retraining."
    )


def select_legacy_sequential(pred_df, specs, costs, lam, tau=0.0):
    """Historical sequential rule retained only for reproducibility ablations."""
    ordered = [s.fidelity for s in specs if f"p_{s.fidelity}" in pred_df.columns]
    ordered = sorted(ordered, key=lambda f: costs[f])
    selected = np.array([ordered[0]] * len(pred_df), dtype=object)
    current_p = pred_df[f"p_{ordered[0]}"].fillna(0.0).values.astype(np.float64)
    for next_f in ordered[1:]:
        next_p = pred_df[f"p_{next_f}"].fillna(0.0).values.astype(np.float64)
        voi = (next_p - current_p) - lam * costs[next_f]
        escalate = voi > tau
        selected[escalate] = next_f
        current_p[escalate] = next_p[escalate]
    return selected


def select_voi(pred_df, specs, costs, lam, tau=None):
    """Exact global argmax of calibrated correctness minus acquisition cost.

    The optional ``tau`` parameter is ignored and retained solely so older
    callers fail safe while result files migrate away from the old interface.
    Columns with missing predictions are unavailable.  Stable cost ordering
    makes ties select the cheaper fidelity.
    """
    ordered = [s.fidelity for s in specs if f"p_{s.fidelity}" in pred_df.columns]
    ordered = sorted(ordered, key=lambda f: (costs[f], f))
    if not ordered:
        raise ValueError("No fidelity probability columns are available")
    utility = np.column_stack(
        [
            pred_df[f"p_{f}"].astype(float).fillna(-np.inf).to_numpy()
            - float(lam) * float(costs[f])
            for f in ordered
        ]
    )
    if np.any(np.all(~np.isfinite(utility), axis=1)):
        raise ValueError("At least one query has no finite fidelity prediction")
    row_max = np.max(utility, axis=1, keepdims=True)
    tied_for_best = np.isclose(utility, row_max, atol=1e-12, rtol=0.0)
    return np.asarray(ordered, dtype=object)[np.argmax(tied_for_best, axis=1)]


def select_voila_cdf(pred_df, specs, costs, thresholds, lambda_cost):
    ordered = [s.fidelity for s in specs if f"p_{s.fidelity}" in pred_df.columns]
    ordered = sorted(ordered, key=lambda f: costs[f])
    selected = np.array([ordered[0]] * len(pred_df), dtype=object)
    for idx, current_f in enumerate(ordered[:-1]):
        next_f = ordered[idx + 1]
        edge = f"{current_f}->{next_f}"
        current_p = pred_df[f"p_{current_f}"].fillna(0.0).values.astype(np.float64)
        next_p = pred_df[f"p_{next_f}"].fillna(0.0).values.astype(np.float64)
        incremental_cost = costs[next_f] - costs[current_f]
        score = (next_p - current_p) - lambda_cost * incremental_cost
        theta = thresholds.get(edge, np.inf)
        at_current = selected == current_f
        escalate = at_current & np.isfinite(score) & (score >= theta)
        selected[escalate] = next_f
    return selected


def select_threshold(pred_df, specs, threshold=0.5):
    ordered = [s.fidelity for s in specs if f"p_{s.fidelity}" in pred_df.columns]
    selected = np.array([ordered[0]] * len(pred_df), dtype=object)
    for i in range(len(pred_df)):
        for f in ordered:
            selected[i] = f
            if pred_df.iloc[i][f"p_{f}"] >= threshold:
                break
    return selected


def evaluate_selection(df, selected, specs, costs):
    correct = np.zeros(len(df), dtype=bool)
    spec_map = {s.fidelity: s for s in specs}
    for i, fidelity in enumerate(selected):
        corr_col = f"correct_{fidelity}"
        if corr_col in df.columns:
            correct[i] = bool(df.iloc[i][corr_col])
    avg_cost = float(np.mean([costs.get(f, DEFAULT_COSTS["full"]) for f in selected]))
    dist = {k: round(v / len(selected), 6) for k, v in Counter(selected).items()}
    return {
        "accuracy": float(correct.mean()),
        "avg_cost": avg_cost,
        "cost_reduction_pct": float((costs["full"] - avg_cost) / costs["full"] * 100.0),
        "fidelity_distribution": dist,
    }


def calibration_quality(df, pred_df, specs):
    rows = {}
    for spec in specs:
        col = f"p_{spec.fidelity}"
        corr = f"correct_{spec.fidelity}"
        if col not in pred_df.columns or corr not in df.columns:
            continue
        y = df[corr].astype(int).values
        p = np.clip(pred_df[col].values, 0.0, 1.0)
        rows[spec.fidelity] = {
            "brier": float(brier_score_loss(y, p)),
            "ece": float(expected_calibration_error(y, p)),
        }
    pairwise_errors = []
    available = [s for s in specs if f"p_{s.fidelity}" in pred_df and f"correct_{s.fidelity}" in df]
    for left_idx, left in enumerate(available):
        for right in available[left_idx + 1:]:
            predicted_gain = (
                pred_df[f"p_{right.fidelity}"].to_numpy(dtype=float)
                - pred_df[f"p_{left.fidelity}"].to_numpy(dtype=float)
            )
            observed_gain = (
                df[f"correct_{right.fidelity}"].to_numpy(dtype=float)
                - df[f"correct_{left.fidelity}"].to_numpy(dtype=float)
            )
            pairwise_errors.extend(np.abs(predicted_gain - observed_gain).tolist())
    rows["_decision"] = {
        "pairwise_gain_mae": float(np.mean(pairwise_errors)) if pairwise_errors else None,
    }
    return rows


def selection_regret(df, selected, specs, costs, lam):
    """Mean realized utility regret against the cached per-query oracle."""
    realized = np.column_stack(
        [
            df[f"correct_{s.fidelity}"].to_numpy(dtype=float) - float(lam) * float(costs[s.fidelity])
            for s in specs
            if f"correct_{s.fidelity}" in df
        ]
    )
    selected_utility = np.asarray(
        [float(df.iloc[i][f"correct_{f}"]) - float(lam) * float(costs[f]) for i, f in enumerate(selected)]
    )
    return float(np.mean(np.max(realized, axis=1) - selected_utility))


def query_level_records(df, pred_df, selected, specs, costs, lam, fold, group_col, cfg):
    records = []
    for pos, (_, row) in enumerate(df.iterrows()):
        chosen = str(selected[pos])
        record = {
            "fold": int(fold),
            "cluster_id": str(row[group_col]),
            "question_id": str(row[cfg["qid"]]),
            "image_id": str(row["image_id"]) if "image_id" in row else None,
            "question": str(row[cfg["question"]]),
            "lambda": float(lam) if np.isfinite(lam) else None,
            "selected_fidelity": chosen,
            "selected_cost": float(costs[chosen]),
            "selected_correct": bool(row[f"correct_{chosen}"]),
        }
        for spec in specs:
            f = spec.fidelity
            p_col = f"p_{f}"
            if p_col not in pred_df:
                continue
            probability = float(pred_df.iloc[pos][p_col])
            record[f"p_{f}"] = probability
            record[f"utility_{f}"] = probability - float(lam) * float(costs[f])
            record[f"cost_{f}"] = float(costs[f])
            record[f"correct_{f}"] = bool(row[f"correct_{f}"])
        records.append(record)
    return records


def clustered_bootstrap_ci(records, n_boot=1000, seed=42):
    """Cluster bootstrap CIs for accuracy and mean selected cost."""
    if not records:
        return {}
    frame = pd.DataFrame(records)
    grouped = frame.groupby("cluster_id", sort=False).agg(
        n=("selected_correct", "size"),
        correct=("selected_correct", "sum"),
        cost=("selected_cost", "sum"),
    )
    rng = np.random.default_rng(seed)
    accuracy = np.empty(n_boot, dtype=float)
    mean_cost = np.empty(n_boot, dtype=float)
    values = grouped[["n", "correct", "cost"]].to_numpy(dtype=float)
    for idx in range(n_boot):
        sample = values[rng.integers(0, len(values), size=len(values))]
        total_n = sample[:, 0].sum()
        accuracy[idx] = sample[:, 1].sum() / total_n
        mean_cost[idx] = sample[:, 2].sum() / total_n
    return {
        "n_clusters": int(len(grouped)),
        "n_boot": int(n_boot),
        "accuracy_95ci": [float(x) for x in np.quantile(accuracy, [0.025, 0.975])],
        "avg_cost_95ci": [float(x) for x in np.quantile(mean_cost, [0.025, 0.975])],
    }


def expected_calibration_error(y, p, n_bins=10):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (p >= left) & (p < right if right < 1.0 else p <= right)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return ece


def is_dominated(candidate, others):
    for other in others:
        if other is candidate:
            continue
        at_least_as_good = (
            other["accuracy"] >= candidate["accuracy"]
            and other["avg_cost"] <= candidate["avg_cost"]
        )
        strictly_better = (
            other["accuracy"] > candidate["accuracy"]
            or other["avg_cost"] < candidate["avg_cost"]
        )
        if at_least_as_good and strictly_better:
            return True
    return False


def select_tradeoff_candidate(candidates, mode="pareto_knee", cost_weight=0.5):
    if not candidates:
        raise ValueError("No VOILA candidates were produced for tradeoff tuning")
    if mode == "accuracy_cost":
        for cand in candidates:
            cand["selection_score"] = cand["accuracy"] - cost_weight * (cand["avg_cost"] / cand["full_cost"])
        return max(candidates, key=lambda c: (c["selection_score"], c["accuracy"], -c["avg_cost"]))

    frontier = [c for c in candidates if not is_dominated(c, candidates)]
    acc_values = np.array([c["accuracy"] for c in frontier], dtype=np.float64)
    cost_values = np.array([c["avg_cost"] for c in frontier], dtype=np.float64)
    acc_span = float(acc_values.max() - acc_values.min())
    cost_span = float(cost_values.max() - cost_values.min())

    for cand in frontier:
        acc_norm = 1.0 if acc_span == 0 else (cand["accuracy"] - acc_values.min()) / acc_span
        saving_norm = 1.0 if cost_span == 0 else (cost_values.max() - cand["avg_cost"]) / cost_span
        # The Pareto knee is the frontier point closest to the local ideal
        # corner: best validation accuracy and lowest validation cost.
        cand["accuracy_norm"] = float(acc_norm)
        cand["saving_norm"] = float(saving_norm)
        cand["selection_score"] = -float(math.hypot(1.0 - acc_norm, 1.0 - saving_norm))

    return max(frontier, key=lambda c: (c["selection_score"], c["accuracy"], -c["avg_cost"]))


def tune_lam_tau(
    cal_df,
    cal_pred,
    specs,
    costs,
    lambda_grid,
    tau_grid,
    tradeoff_mode="pareto_knee",
    cost_weight=0.5,
    accuracy_retention=0.95,
):
    """Tune the exact utility policy on calibration data only.

    The primary operating point is the cheapest lambda whose accuracy reaches
    ``accuracy_retention`` times the best fixed-fidelity accuracy.  ``tau_grid``
    and the older tradeoff parameters remain in the signature for CLI/result
    compatibility but no longer affect primary selection.
    """
    candidates = []
    for lam in lambda_grid:
        selected = select_voi(cal_pred, specs, costs, lam)
        metrics = evaluate_selection(cal_df, selected, specs, costs)
        regret = selection_regret(cal_df, selected, specs, costs, lam)
        candidates.append(
            {
                "lambda": float(lam),
                "tau": None,
                "accuracy": metrics["accuracy"],
                "avg_cost": metrics["avg_cost"],
                "cost_reduction_pct": metrics["cost_reduction_pct"],
                "full_cost": costs["full"],
                "selection_regret": regret,
            }
        )
    fixed_accuracy = {
        spec.fidelity: float(cal_df[f"correct_{spec.fidelity}"].mean())
        for spec in specs
        if f"correct_{spec.fidelity}" in cal_df.columns
    }
    best_fixed_accuracy = max(fixed_accuracy.values())
    accuracy_floor = float(accuracy_retention) * best_fixed_accuracy
    feasible = [c for c in candidates if c["accuracy"] + 1e-12 >= accuracy_floor]
    if feasible:
        best = min(feasible, key=lambda c: (c["avg_cost"], -c["accuracy"], c["lambda"]))
        selection_status = "accuracy_floor_met"
    else:
        best = max(candidates, key=lambda c: (c["accuracy"], -c["avg_cost"], -c["lambda"]))
        selection_status = "accuracy_floor_unmet_best_accuracy_used"
    frontier = [c for c in candidates if not is_dominated(c, candidates)]
    frontier = sorted(frontier, key=lambda c: (-c["accuracy"], c["avg_cost"], c["lambda"]))
    diagnostics = {
        "mode": "cheapest_at_accuracy_retention",
        "accuracy_retention": float(accuracy_retention),
        "best_fixed_validation_accuracy": best_fixed_accuracy,
        "fixed_validation_accuracy": fixed_accuracy,
        "accuracy_floor": accuracy_floor,
        "selection_status": selection_status,
        "selected": {
            "lambda": best["lambda"],
            "accuracy": best["accuracy"],
            "avg_cost": best["avg_cost"],
            "cost_reduction_pct": best["cost_reduction_pct"],
            "selection_regret": best["selection_regret"],
        },
        "validation_accuracy": best["accuracy"],
        "validation_avg_cost": best["avg_cost"],
        "validation_cost_reduction_pct": best["cost_reduction_pct"],
        "selection_score": best.get("selection_score"),
        "accuracy_norm": best.get("accuracy_norm"),
        "saving_norm": best.get("saving_norm"),
        "n_candidates": len(candidates),
        "n_pareto": len(frontier),
        "pareto_frontier": [
            {
                "lambda": c["lambda"],
                "accuracy": c["accuracy"],
                "avg_cost": c["avg_cost"],
                "cost_reduction_pct": c["cost_reduction_pct"],
                "selection_regret": c["selection_regret"],
                "selection_score": c.get("selection_score"),
                "accuracy_norm": c.get("accuracy_norm"),
                "saving_norm": c.get("saving_norm"),
            }
            for c in frontier
        ],
    }
    return best["lambda"], None, diagnostics


def tune_legacy_lam_tau(cal_df, cal_pred, specs, costs, lambda_grid, tau_grid):
    """Validation-only tuning for the historical sequential-rule ablation."""
    candidates = []
    for lam in lambda_grid:
        for tau in tau_grid:
            selected = select_legacy_sequential(cal_pred, specs, costs, lam, tau)
            metrics = evaluate_selection(cal_df, selected, specs, costs)
            candidates.append(
                {
                    "lambda": float(lam),
                    "tau": float(tau),
                    "accuracy": metrics["accuracy"],
                    "avg_cost": metrics["avg_cost"],
                    "cost_reduction_pct": metrics["cost_reduction_pct"],
                    "full_cost": costs["full"],
                }
            )
    best = select_tradeoff_candidate(candidates, "pareto_knee")
    return best["lambda"], best["tau"], best


def learn_voila_cdf_thresholds(cal_df, cal_pred, specs, costs, lambda_cost):
    ordered = [s.fidelity for s in specs if f"p_{s.fidelity}" in cal_pred.columns]
    ordered = sorted(ordered, key=lambda f: costs[f])
    reached = np.ones(len(cal_df), dtype=bool)
    thresholds = {}
    details = {}

    for idx, current_f in enumerate(ordered[:-1]):
        next_f = ordered[idx + 1]
        edge = f"{current_f}->{next_f}"
        current_p = cal_pred[f"p_{current_f}"].fillna(0.0).values.astype(np.float64)
        next_p = cal_pred[f"p_{next_f}"].fillna(0.0).values.astype(np.float64)
        incremental_cost = costs[next_f] - costs[current_f]
        score = (next_p - current_p) - lambda_cost * incremental_cost

        current_correct = cal_df[f"correct_{current_f}"].astype(bool).values
        future_correct = np.zeros(len(cal_df), dtype=bool)
        for future_f in ordered[idx + 1:]:
            future_correct |= cal_df[f"correct_{future_f}"].astype(bool).values
        helpful = (~current_correct & future_correct).astype(int)
        mask = reached & np.isfinite(score)
        info = cdf_separation_threshold_from_scores(score[mask], helpful[mask])
        info.update(
            {
                "edge": edge,
                "current_fidelity": current_f,
                "next_fidelity": next_f,
                "score": "p_next - p_current - lambda_cost * incremental_cost",
                "lambda_cost": lambda_cost,
                "incremental_cost": incremental_cost,
                "n_reached_edge": int(np.sum(reached)),
                "n_finite_score": int(np.sum(mask)),
                "positive_label": "current_wrong_any_higher_fidelity_correct",
            }
        )
        thresholds[edge] = info["theta"]
        details[edge] = info
        reached = reached & np.isfinite(score) & (score >= info["theta"])

    return thresholds, details


def tune_voila_cdf_thresholds(
    cal_df,
    cal_pred,
    specs,
    costs,
    lambda_grid,
    tradeoff_mode="pareto_knee",
    cost_weight=0.5,
):
    candidates = []
    for lam in lambda_grid:
        thresholds, threshold_details = learn_voila_cdf_thresholds(cal_df, cal_pred, specs, costs, lam)
        selected = select_voila_cdf(cal_pred, specs, costs, thresholds, lam)
        metrics = evaluate_selection(cal_df, selected, specs, costs)
        candidates.append(
            {
                "lambda": lam,
                "tau": np.nan,
                "accuracy": metrics["accuracy"],
                "avg_cost": metrics["avg_cost"],
                "cost_reduction_pct": metrics["cost_reduction_pct"],
                "full_cost": costs["full"],
                "thresholds": thresholds,
                "threshold_details": threshold_details,
            }
        )

    best = select_tradeoff_candidate(candidates, tradeoff_mode, cost_weight)
    frontier = [c for c in candidates if not is_dominated(c, candidates)]
    frontier = sorted(frontier, key=lambda c: (-c["accuracy"], c["avg_cost"], c["lambda"]))
    diagnostics = {
        "mode": tradeoff_mode,
        "cost_weight": cost_weight if tradeoff_mode == "accuracy_cost" else None,
        "selected": {
            "lambda": best["lambda"],
            "thresholds": best["thresholds"],
            "accuracy": best["accuracy"],
            "avg_cost": best["avg_cost"],
            "cost_reduction_pct": best["cost_reduction_pct"],
        },
        "validation_accuracy": best["accuracy"],
        "validation_avg_cost": best["avg_cost"],
        "validation_cost_reduction_pct": best["cost_reduction_pct"],
        "selection_score": best.get("selection_score"),
        "accuracy_norm": best.get("accuracy_norm"),
        "saving_norm": best.get("saving_norm"),
        "n_candidates": len(candidates),
        "n_pareto": len(frontier),
        "pareto_frontier": [
            {
                "lambda": c["lambda"],
                "accuracy": c["accuracy"],
                "avg_cost": c["avg_cost"],
                "cost_reduction_pct": c["cost_reduction_pct"],
                "thresholds": c["thresholds"],
                "selection_score": c.get("selection_score"),
                "accuracy_norm": c.get("accuracy_norm"),
                "saving_norm": c.get("saving_norm"),
            }
            for c in frontier
        ],
        "threshold_details": best["threshold_details"],
    }
    return best["lambda"], best["thresholds"], diagnostics


def normalize_question_template(question):
    return " ".join(clean_text(question).split())


def prepare_split_groups(df, cfg, dataset_name=None, split_strategy="auto"):
    """Return a dataframe and group column used for leakage-safe CV."""
    out = df.copy()
    strategy = split_strategy
    if strategy == "auto":
        strategy = "image_grouped" if dataset_name == "floodnet" else "question_id"
    if strategy == "image_grouped":
        if "image_id" not in out.columns:
            raise ValueError("image_grouped splitting requires an image_id column")
        group_col = "image_id"
    elif strategy == "question_grouped":
        group_col = "__question_template_group"
        out[group_col] = out[cfg["question"]].map(normalize_question_template)
    elif strategy == "question_id":
        group_col = cfg["qid"]
    else:
        raise ValueError(f"Unknown split_strategy={split_strategy}")
    if out[group_col].isna().any():
        raise ValueError(f"Split group column {group_col} contains missing values")
    return out, group_col, strategy


def split_by_question_ids(df, qid_col, test_qids):
    return df[df[qid_col].isin(set(test_qids))].copy()


def make_outer_splits(df, group_col, n_folds, seed):
    groups = pd.Series(df[group_col].unique())
    n_splits = min(n_folds, len(groups))
    if n_splits < 2:
        raise ValueError("Need at least two unique groups for CV")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(groups):
        yield groups.iloc[train_idx].values, groups.iloc[test_idx].values


def summarize_metric(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)), float(np.std(values, ddof=1) if len(values) > 1 else 0.0)


def run_cv_voila(
    df,
    cfg,
    specs,
    costs,
    n_folds,
    seed,
    lambda_grid,
    tau_grid,
    predictor="GBR",
    calibration="iso",
    feature_set="tfidf+struct",
    policy="voi",
    max_tfidf_features=500,
    tradeoff_mode="pareto_knee",
    cost_weight=0.5,
    dataset_name=None,
    model_name=None,
    experiment_name="main",
    model_artifact_dir=None,
    save_models=False,
    reuse_models=False,
    strict_reuse_models=False,
    split_strategy="auto",
    accuracy_retention=0.95,
    bootstrap_samples=1000,
    return_query_records=False,
):
    fold_rows = []
    all_query_records = []
    df, group_col, resolved_split_strategy = prepare_split_groups(
        df, cfg, dataset_name=dataset_name, split_strategy=split_strategy
    )
    split_iter = list(make_outer_splits(df, group_col, n_folds, seed))
    desc = f"{dataset_name}/{model_name} {experiment_name} folds" if dataset_name and model_name else "VOILA folds"
    for fold, (train_groups, test_groups) in enumerate(progress_iter(split_iter, desc=desc, total=len(split_iter)), start=1):
        train_all = split_by_question_ids(df, group_col, train_groups)
        test_df = split_by_question_ids(df, group_col, test_groups)
        train_ids, tuning_ids = train_test_split(
            pd.Series(train_all[group_col].unique()),
            test_size=0.3,
            random_state=seed + fold,
        )
        cal_ids, validation_ids = train_test_split(
            pd.Series(tuning_ids),
            test_size=0.5,
            random_state=seed + fold + 1000,
        )
        train_df = split_by_question_ids(train_all, group_col, train_ids)
        cal_df = split_by_question_ids(train_all, group_col, cal_ids)
        validation_df = split_by_question_ids(train_all, group_col, validation_ids)
        internal_sets = [set(train_df[group_col]), set(cal_df[group_col]), set(validation_df[group_col])]
        if any(internal_sets[left] & internal_sets[right] for left in range(3) for right in range(left + 1, 3)):
            raise AssertionError("Training/calibration/validation split group leakage")
        if set(train_all[group_col]).intersection(set(test_df[group_col])):
            raise AssertionError("Outer CV split group leakage")
        artifact_path = None
        if model_artifact_dir and dataset_name and model_name:
            artifact_path = fold_artifact_path(
                model_artifact_dir,
                dataset_name,
                model_name,
                experiment_name,
                fold,
                predictor,
                calibration,
                feature_set,
                max_tfidf_features,
                seed + fold,
            )
        if (reuse_models or save_models) and artifact_path and artifact_path.exists():
            metadata, artifact = load_model_artifact(artifact_path)
            cal_pred, test_pred = predict_with_artifact(artifact, cal_df, test_df)
            _, validation_pred = predict_with_artifact(artifact, cal_df, validation_df)
            print(f"    fold {fold}: loaded model artifact {artifact_path}")
        elif reuse_models and strict_reuse_models:
            raise missing_artifact_error(artifact_path)
        else:
            trained_result = train_fidelity_models(
                train_df,
                cal_df,
                test_df,
                specs,
                cfg,
                predictor_name=predictor,
                calibration=calibration,
                feature_set=feature_set,
                max_tfidf_features=max_tfidf_features,
                seed=seed + fold,
                return_artifact=True,
            )
            _, cal_pred, test_pred, _, artifact = trained_result
            _, validation_pred = predict_with_artifact(artifact, cal_df, validation_df)
            if save_models and artifact_path:
                metadata = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "experiment": experiment_name,
                    "fold": fold,
                    "predictor": predictor,
                    "calibration": calibration,
                    "feature_set": feature_set,
                    "max_tfidf_features": max_tfidf_features,
                    "seed": seed + fold,
                    "split_strategy": resolved_split_strategy,
                    "split_group_column": group_col,
                    "train_groups": list(map(str, train_ids)),
                    "cal_groups": list(map(str, cal_ids)),
                    "validation_groups": list(map(str, validation_ids)),
                    "test_groups": list(map(str, test_groups)),
                    "fidelities": [s.__dict__ for s in specs],
                    "question_col": cfg["question"],
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_model_artifact(artifact_path, artifact, metadata)
                print(f"    fold {fold}: saved model artifact {artifact_path}")
        if policy == "acc":
            selected = np.array(["full"] * len(test_df), dtype=object)
            lam, tau, tradeoff = np.nan, np.nan, {"mode": "always_full"}
        elif policy == "thr":
            selected = select_threshold(test_pred, specs, threshold=0.5)
            lam, tau, tradeoff = np.nan, 0.5, {"mode": "fixed_probability_threshold"}
        elif policy == "legacy":
            lam, tau, tradeoff = tune_legacy_lam_tau(
                validation_df, validation_pred, specs, costs, lambda_grid, tau_grid
            )
            selected = select_legacy_sequential(test_pred, specs, costs, lam, tau)
        else:
            lam, tau, tradeoff = tune_lam_tau(
                validation_df,
                validation_pred,
                specs,
                costs,
                lambda_grid,
                tau_grid,
                tradeoff_mode=tradeoff_mode,
                cost_weight=cost_weight,
                accuracy_retention=accuracy_retention,
            )
            selected = select_voi(test_pred, specs, costs, lam)

        metrics = evaluate_selection(test_df, selected, specs, costs)
        test_frontier = []
        if policy == "voi":
            for report_lambda in lambda_grid:
                report_selected = select_voi(test_pred, specs, costs, report_lambda)
                report_metrics = evaluate_selection(test_df, report_selected, specs, costs)
                test_frontier.append(
                    {
                        "lambda": float(report_lambda),
                        "accuracy": report_metrics["accuracy"],
                        "avg_cost": report_metrics["avg_cost"],
                        "cost_reduction_pct": report_metrics["cost_reduction_pct"],
                    }
                )
        calib = calibration_quality(test_df, test_pred, specs)
        validation_calib = calibration_quality(validation_df, validation_pred, specs)
        regret = selection_regret(test_df, selected, specs, costs, 0.0 if not np.isfinite(lam) else lam)
        fold_query_records = query_level_records(
            test_df,
            test_pred,
            selected,
            specs,
            costs,
            0.0 if not np.isfinite(lam) else lam,
            fold,
            group_col,
            cfg,
        )
        all_query_records.extend(fold_query_records)
        fold_rows.append(
            {
                "fold": fold,
                "n_test": int(len(test_df)),
                "lambda": lam,
                "tau": tau,
                "accuracy": metrics["accuracy"],
                "avg_cost": metrics["avg_cost"],
                "cost_reduction_pct": metrics["cost_reduction_pct"],
                "fidelity_distribution": metrics["fidelity_distribution"],
                "calibration": calib,
                "validation_calibration": validation_calib,
                "selection_regret": regret,
                "tradeoff": tradeoff,
                "test_accuracy_cost_frontier": test_frontier,
                "n_train_groups": int(len(train_ids)),
                "n_cal_groups": int(len(cal_ids)),
                "n_validation_groups": int(len(validation_ids)),
                "n_test_groups": int(len(test_groups)),
            }
        )
        print(
            f"    fold {fold}: acc={metrics['accuracy']*100:.2f}% "
            f"cost={metrics['avg_cost']:.2f} lambda={lam} tau={tau}"
        )
    out = aggregate_folds(fold_rows)
    out["routing_rule"] = "exact_argmax_probability_minus_lambda_cost"
    out["split_strategy"] = resolved_split_strategy
    out["split_group_column"] = group_col
    out["accuracy_retention"] = float(accuracy_retention)
    out["bootstrap"] = clustered_bootstrap_ci(all_query_records, bootstrap_samples, seed)
    out["selection_regret_mean"] = float(np.mean([r["selection_regret"] for r in fold_rows]))
    if return_query_records:
        out["_query_records"] = all_query_records
    return out


def run_cv_voila_cdf(
    df,
    cfg,
    specs,
    costs,
    n_folds,
    seed,
    lambda_grid,
    predictor="GBR",
    calibration="iso",
    feature_set="tfidf+struct",
    max_tfidf_features=500,
    tradeoff_mode="pareto_knee",
    cost_weight=0.5,
    dataset_name=None,
    model_name=None,
    experiment_name="voila_cdf",
    model_artifact_dir=None,
    save_models=False,
    reuse_models=False,
    strict_reuse_models=False,
):
    fold_rows = []
    qid_col = cfg["qid"]
    split_iter = list(make_outer_splits(df, qid_col, n_folds, seed))
    desc = f"{dataset_name}/{model_name} {experiment_name} folds" if dataset_name and model_name else "VOILA-CDF folds"
    for fold, (train_qids, test_qids) in enumerate(progress_iter(split_iter, desc=desc, total=len(split_iter)), start=1):
        train_all = split_by_question_ids(df, qid_col, train_qids)
        test_df = split_by_question_ids(df, qid_col, test_qids)
        train_ids, cal_ids = train_test_split(
            pd.Series(train_all[qid_col].unique()),
            test_size=0.2,
            random_state=seed + fold,
        )
        train_df = split_by_question_ids(train_all, qid_col, train_ids)
        cal_df = split_by_question_ids(train_all, qid_col, cal_ids)
        artifact_path = None
        if model_artifact_dir and dataset_name and model_name:
            artifact_path = fold_artifact_path(
                model_artifact_dir,
                dataset_name,
                model_name,
                experiment_name,
                fold,
                predictor,
                calibration,
                feature_set,
                max_tfidf_features,
                seed + fold,
            )
        if (reuse_models or save_models) and artifact_path and artifact_path.exists():
            metadata, artifact = load_model_artifact(artifact_path)
            cal_pred, test_pred = predict_with_artifact(artifact, cal_df, test_df)
            print(f"    fold {fold}: loaded model artifact {artifact_path}")
        elif reuse_models and strict_reuse_models:
            raise missing_artifact_error(artifact_path)
        else:
            trained_result = train_fidelity_models(
                train_df,
                cal_df,
                test_df,
                specs,
                cfg,
                predictor_name=predictor,
                calibration=calibration,
                feature_set=feature_set,
                max_tfidf_features=max_tfidf_features,
                seed=seed + fold,
                return_artifact=bool(save_models and artifact_path),
            )
            if save_models and artifact_path:
                _, cal_pred, test_pred, _, artifact = trained_result
                metadata = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "experiment": experiment_name,
                    "fold": fold,
                    "predictor": predictor,
                    "calibration": calibration,
                    "feature_set": feature_set,
                    "max_tfidf_features": max_tfidf_features,
                    "seed": seed + fold,
                    "train_qids": list(map(str, train_ids)),
                    "cal_qids": list(map(str, cal_ids)),
                    "test_qids": list(map(str, test_qids)),
                    "fidelities": [s.__dict__ for s in specs],
                    "question_col": cfg["question"],
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_model_artifact(artifact_path, artifact, metadata)
                print(f"    fold {fold}: saved model artifact {artifact_path}")
            else:
                _, cal_pred, test_pred, _ = trained_result
        lam, thresholds, tradeoff = tune_voila_cdf_thresholds(
            cal_df,
            cal_pred,
            specs,
            costs,
            lambda_grid,
            tradeoff_mode=tradeoff_mode,
            cost_weight=cost_weight,
        )
        selected = select_voila_cdf(test_pred, specs, costs, thresholds, lam)
        metrics = evaluate_selection(test_df, selected, specs, costs)
        calib = calibration_quality(test_df, test_pred, specs)
        fold_rows.append(
            {
                "fold": fold,
                "n_test": int(len(test_df)),
                "lambda": lam,
                "thresholds": thresholds,
                "accuracy": metrics["accuracy"],
                "avg_cost": metrics["avg_cost"],
                "cost_reduction_pct": metrics["cost_reduction_pct"],
                "fidelity_distribution": metrics["fidelity_distribution"],
                "calibration": calib,
                "tradeoff": tradeoff,
            }
        )
        compact = ", ".join(f"{edge}:{theta:.3f}" for edge, theta in thresholds.items())
        print(
            f"    fold {fold}: voila_cdf acc={metrics['accuracy']*100:.2f}% "
            f"cost={metrics['avg_cost']:.2f} lambda={lam} thresholds=[{compact}]"
        )
    out = aggregate_folds(
        [
            {
                "fold": r["fold"],
                "n_test": r["n_test"],
                "lambda": r["lambda"],
                "tau": np.nan,
                "accuracy": r["accuracy"],
                "avg_cost": r["avg_cost"],
                "cost_reduction_pct": r["cost_reduction_pct"],
                "fidelity_distribution": r["fidelity_distribution"],
                "calibration": r["calibration"],
                "tradeoff": r["tradeoff"],
            }
            for r in fold_rows
        ]
    )
    out["threshold_values"] = [r["thresholds"] for r in fold_rows]
    out["folds"] = fold_rows
    out["routing_rule"] = "pairwise_cdf_on_calibrated_probability_gain"
    return out


def aggregate_folds(fold_rows):
    acc_mean, acc_std = summarize_metric([r["accuracy"] * 100.0 for r in fold_rows])
    cost_mean, cost_std = summarize_metric([r["avg_cost"] for r in fold_rows])
    red_mean, red_std = summarize_metric([r["cost_reduction_pct"] for r in fold_rows])
    dist = defaultdict(float)
    for row in fold_rows:
        for k, v in row["fidelity_distribution"].items():
            dist[k] += v / len(fold_rows)
    return {
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "avg_cost_mean": cost_mean,
        "avg_cost_std": cost_std,
        "cost_reduction_pct_mean": red_mean,
        "cost_reduction_pct_std": red_std,
        "fidelity_distribution": {k: round(v, 6) for k, v in sorted(dist.items())},
        "lambda_values": [r["lambda"] for r in fold_rows],
        "tau_values": [r["tau"] for r in fold_rows],
        "folds": fold_rows,
    }


def fixed_fidelity_baselines(df, specs, costs):
    out = {}
    for spec in specs:
        col = f"correct_{spec.fidelity}"
        if col in df.columns:
            acc = float(df[col].mean() * 100.0)
            out[spec.fidelity] = {
                "accuracy": acc,
                "avg_cost": costs[spec.fidelity],
                "cost_reduction_pct": (costs["full"] - costs[spec.fidelity]) / costs["full"] * 100.0,
            }
    return out


def oracle_metrics(df, specs, costs):
    ordered = sorted(specs, key=lambda s: costs[s.fidelity])
    selected = []
    solved = []
    for _, row in df.iterrows():
        picked = "full"
        ok = False
        for spec in ordered:
            if bool(row.get(f"correct_{spec.fidelity}", False)):
                picked = spec.fidelity
                ok = True
                break
        selected.append(picked)
        solved.append(ok)
    dist = {k: round(v / len(selected), 6) for k, v in Counter(selected).items()}
    avg_cost = float(np.mean([costs[f] for f in selected]))
    return {
        "oracle_accuracy": float(np.mean(solved) * 100.0),
        "oracle_avg_cost": avg_cost,
        "fraction_unsolvable": float(1.0 - np.mean(solved)),
        "fidelity_distribution": dist,
    }


def cascade_metrics(df, specs, costs, thresholds):
    ordered = sorted(specs, key=lambda s: costs[s.fidelity])
    lp = {s.fidelity: logprob_scores(df, s) for s in ordered}
    selected = []
    total_cost = []
    correct = []
    for i in range(len(df)):
        spent = 0.0
        picked = ordered[-1].fidelity
        for idx, spec in enumerate(ordered):
            spent += costs[spec.fidelity]
            picked = spec.fidelity
            if idx == len(ordered) - 1:
                break
            score = lp[spec.fidelity][i]
            edge = f"{spec.fidelity}->{ordered[idx + 1].fidelity}"
            theta = thresholds.get(edge, np.inf)
            if np.isfinite(score) and score >= theta:
                break
        selected.append(picked)
        total_cost.append(spent)
        correct.append(bool(df.iloc[i].get(f"correct_{picked}", False)))
    return {
        "accuracy": float(np.mean(correct)),
        "avg_cost": float(np.mean(total_cost)),
        "cost_reduction_pct": float((costs["full"] - np.mean(total_cost)) / costs["full"] * 100.0),
        "fidelity_distribution": {k: round(v / len(selected), 6) for k, v in Counter(selected).items()},
    }


def cdf_separation_threshold_from_scores(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if len(scores) == 0:
        return {
            "theta": np.inf,
            "separation": 0.0,
            "n_correct": 0,
            "n_incorrect": 0,
            "status": "fallback_no_finite_logprobs_escalate",
        }
    unique = np.unique(labels)
    if len(unique) < 2:
        all_correct = int(unique[0]) == 1
        return {
            "theta": -np.inf if all_correct else np.inf,
            "separation": 0.0,
            "n_correct": int(np.sum(labels == 1)),
            "n_incorrect": int(np.sum(labels == 0)),
            "status": "fallback_all_correct_stop" if all_correct else "fallback_all_incorrect_escalate",
        }

    thresholds = np.unique(scores)
    correct_scores = scores[labels == 1]
    incorrect_scores = scores[labels == 0]
    best = {
        "theta": float(thresholds[0]),
        "separation": -np.inf,
        "cdf_correct": 0.0,
        "cdf_incorrect": 0.0,
    }
    for theta in thresholds:
        cdf_correct = float(np.mean(correct_scores <= theta))
        cdf_incorrect = float(np.mean(incorrect_scores <= theta))
        separation = cdf_incorrect - cdf_correct
        if separation > best["separation"]:
            best = {
                "theta": float(theta),
                "separation": float(separation),
                "cdf_correct": cdf_correct,
                "cdf_incorrect": cdf_incorrect,
            }
    best.update(
        {
            "n_correct": int(len(correct_scores)),
            "n_incorrect": int(len(incorrect_scores)),
            "status": "ok",
        }
    )
    return best


def learn_pairwise_cascade_thresholds(df, specs, costs):
    ordered = sorted(specs, key=lambda s: costs[s.fidelity])
    reached = np.ones(len(df), dtype=bool)
    thresholds = {}
    details = {}
    for idx, spec in enumerate(ordered[:-1]):
        next_spec = ordered[idx + 1]
        edge = f"{spec.fidelity}->{next_spec.fidelity}"
        lp = logprob_scores(df, spec)
        labels = df[f"correct_{spec.fidelity}"].astype(int).values
        mask = reached & np.isfinite(lp)
        info = cdf_separation_threshold_from_scores(lp[mask], labels[mask])
        info.update(
            {
                "edge": edge,
                "current_fidelity": spec.fidelity,
                "next_fidelity": next_spec.fidelity,
                "n_reached_edge": int(np.sum(reached)),
                "n_finite_logprob": int(np.sum(mask)),
            }
        )
        thresholds[edge] = info["theta"]
        details[edge] = info

        # Simulate training-time flow so later edge thresholds are learned only
        # on examples that would have escalated to that edge.
        reached = reached & (~np.isfinite(lp) | (lp < info["theta"]))
    return thresholds, details


def run_cv_cascade(df, cfg, specs, costs, n_folds, seed):
    rows = []
    split_iter = list(make_outer_splits(df, cfg["qid"], n_folds, seed))
    for fold, (train_qids, test_qids) in enumerate(progress_iter(split_iter, desc="cascade folds", total=len(split_iter)), start=1):
        train_df = split_by_question_ids(df, cfg["qid"], train_qids)
        test_df = split_by_question_ids(df, cfg["qid"], test_qids)
        thresholds, threshold_info = learn_pairwise_cascade_thresholds(train_df, specs, costs)
        m = cascade_metrics(test_df, specs, costs, thresholds)
        rows.append({"fold": fold, "thresholds": thresholds, "threshold_info": threshold_info, **m})
        compact = ", ".join(f"{edge}:{theta:.3f}" for edge, theta in thresholds.items())
        print(
            f"    fold {fold}: cascade acc={m['accuracy']*100:.2f}% "
            f"cost={m['avg_cost']:.2f} thresholds=[{compact}]"
        )
    out = aggregate_folds(
        [
            {
                "fold": r["fold"],
                "n_test": 0,
                "lambda": np.nan,
                "tau": np.nan,
                "accuracy": r["accuracy"],
                "avg_cost": r["avg_cost"],
                "cost_reduction_pct": r["cost_reduction_pct"],
                "fidelity_distribution": r["fidelity_distribution"],
                "calibration": {},
            }
            for r in rows
        ]
    )
    out["threshold_values"] = [r["thresholds"] for r in rows]
    out["threshold_method"] = "pairwise_max_cdf_separation"
    out["threshold_details"] = [r["threshold_info"] for r in rows]
    return out


def obs2_quantification(df, specs):
    caption = next((s for s in specs if s.fidelity == "caption"), None)
    if caption is None:
        return {"status": "skipped", "reason": "no caption fidelity"}
    scores = logprob_scores(df, caption)
    correct = df[f"correct_{caption.fidelity}"].astype(bool).values
    good = scores[correct & np.isfinite(scores)]
    bad = scores[(~correct) & np.isfinite(scores)]
    if len(good) == 0 or len(bad) == 0:
        return {"status": "skipped", "reason": "insufficient caption logprob/correctness data"}
    median_correct = float(np.median(good))
    overlap = float(np.mean(bad > median_correct))
    return {
        "status": "ok",
        "median_logprob_correct": median_correct,
        "median_logprob_incorrect": float(np.median(bad)),
        "overlap_fraction": overlap,
        "sentence": (
            f"{overlap*100:.1f}% of incorrect caption-fidelity answers have log-probability "
            f"above the median confidence of correct caption-fidelity answers."
        ),
        "n_correct_caption": int(len(good)),
        "n_incorrect_caption": int(len(bad)),
    }


def heuristic_router_label(question, routing):
    q = "" if pd.isna(question) else str(question).lower()
    tokens = set(q.split())
    high_terms = {
        "read", "text", "written", "say", "says", "letter", "word", "sign",
        "label", "serial", "license", "small", "tiny", "number",
    }
    medium_terms = {
        "many", "count", "few", "total", "color", "colour", "left", "right",
        "behind", "front", "above", "below", "beside", "where", "brand",
        "type", "kind", "model",
    }
    high = bool(tokens & high_terms) or q.startswith("how many")
    medium = bool(tokens & medium_terms)
    if routing == "binary":
        return "yes" if (high or medium) else "no"
    if high:
        return "high"
    if medium:
        return "medium"
    return "low"


def run_llm_router_surrogate(df, cfg, specs, costs, routing, n_samples, seed):
    """Offline surrogate for Experiment 3.

    The real paper baseline uses an external LLM prompt. This local version does
    not call any model; it applies the prompt's criteria as deterministic lexical
    rules so the experiment slot can be run from the existing CSVs.
    """
    if n_samples and len(df) > n_samples:
        eval_df = df.sample(n=n_samples, random_state=seed).copy()
    else:
        eval_df = df.copy()
    available = {s.fidelity for s in specs}
    selected = []
    labels = []
    for q in eval_df[cfg["question"]].values:
        label = heuristic_router_label(q, routing)
        labels.append(label)
        if routing == "binary":
            selected.append("full" if label == "yes" else "caption")
        else:
            if label == "high":
                selected.append("full")
            elif label == "medium" and "jpeg_q10" in available:
                selected.append("jpeg_q10")
            else:
                selected.append("caption")
    metrics = evaluate_selection(eval_df, np.array(selected, dtype=object), specs, costs)
    return {
        "status": "ok",
        "router_kind": "offline_heuristic_surrogate",
        "routing": routing,
        "n_samples": int(len(eval_df)),
        "label_distribution": {k: round(v / len(labels), 6) for k, v in Counter(labels).items()},
        "accuracy": metrics["accuracy"] * 100.0,
        "avg_cost": metrics["avg_cost"],
        "cost_reduction_pct": metrics["cost_reduction_pct"],
        "fidelity_distribution": metrics["fidelity_distribution"],
        "caveat": "This is not the GPT-3.5/GPT-4o-mini API baseline; it is an offline no-model-call surrogate.",
    }


def llm_router_prompt(question, routing):
    if routing == "binary":
        return (
            "You are helping a vision system decide how much visual detail to retrieve.\n\n"
            f'Question: "{question}"\n\n'
            "Does answering this question correctly require a high-resolution image?\n"
            "Consider: counting fine details, reading text, identifying colors precisely,\n"
            "or resolving spatial relationships at pixel level.\n\n"
            "Answer with exactly one word: yes or no."
        )
    return (
        "You are helping a vision system decide how much visual detail to retrieve.\n\n"
        f'Question: "{question}"\n\n'
        "Choose the lowest visual fidelity likely sufficient to answer correctly.\n"
        "Answer with exactly one word: low, medium, or high.\n"
        "- low: caption or thumbnail is sufficient\n"
        "- medium: compressed image needed\n"
        "- high: full resolution required"
    )


def normalize_llm_router_label(text, routing):
    text = clean_text(text)
    tokens = text.split()
    first = tokens[0] if tokens else ""
    if routing == "binary":
        return "yes" if first.startswith("y") else "no"
    if first in {"low", "medium", "high"}:
        return first
    if "high" in tokens:
        return "high"
    if "medium" in tokens:
        return "medium"
    return "low"


def azure_gpt4o_router_label(question, routing, args, cache_key):
    cache_dir = Path(args.llm_router_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if str(cached.get("raw_content", "")).strip():
            return cached["label"], cached

    api_key = read_secret_file(args.azure_openai_api_key_file) or os.environ.get(args.azure_openai_api_key_env)
    if not api_key:
        raise RuntimeError(
            "Missing Azure OpenAI API key. Put it in "
            f"{args.azure_openai_api_key_file} or set environment variable {args.azure_openai_api_key_env}."
        )

    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "api-key": api_key,
    }
    last_empty_response = None
    last_token_limit_error = None
    for attempt in range(1, args.llm_router_retries + 1):
        max_tokens = int(args.llm_router_max_completion_tokens * (2 ** (attempt - 1)))
        body = {
            "messages": [
                {"role": "system", "content": "You are a concise visual-fidelity router. Reply with exactly one visible word."},
                {"role": "user", "content": llm_router_prompt(question, routing)},
            ],
            "temperature": 0,
            "max_completion_tokens": max_tokens,
        }
        req = urllib.request.Request(
            args.azure_openai_url,
            headers=headers,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=args.llm_router_timeout) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 400 and (
                "max_tokens" in error_body
                or "max_completion_tokens" in error_body
                or "output limit was reached" in error_body
            ):
                last_token_limit_error = error_body
                if args.llm_router_sleep > 0:
                    time.sleep(args.llm_router_sleep)
                continue
            raise RuntimeError(f"Azure OpenAI HTTP {exc.code}: {error_body}") from exc

        content = data["choices"][0]["message"].get("content", "")
        if str(content).strip():
            break
        last_empty_response = data
        if args.llm_router_sleep > 0:
            time.sleep(args.llm_router_sleep)
    else:
        if last_token_limit_error:
            raise RuntimeError(
                "Azure OpenAI hit the router output token limit after "
                f"{args.llm_router_retries} attempt(s). Last error: {last_token_limit_error}"
            )
        raise RuntimeError(
            "Azure OpenAI returned empty router responses after "
            f"{args.llm_router_retries} attempt(s): {json.dumps(last_empty_response)[:1000]}"
        )
    label = normalize_llm_router_label(content, routing)
    record = {
        "label": label,
        "raw_content": content,
        "routing": routing,
        "question_hash": cache_key,
        "attempts": attempt,
        "max_completion_tokens": max_tokens,
        "model": data.get("model"),
        "usage": data.get("usage"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    if args.llm_router_sleep > 0:
        time.sleep(args.llm_router_sleep)
    return label, record


def llm_router_dataset_label_path(args, dataset, routing, n_samples, seed):
    sample_tag = "all" if not n_samples else str(int(n_samples))
    return Path(args.llm_router_cache_dir) / "datasets" / f"{dataset}_{routing}_n{sample_tag}_seed{seed}.json"


def router_question_rows(df, cfg, n_samples, seed):
    if n_samples and len(df) > n_samples:
        eval_df = df.sample(n=n_samples, random_state=seed).copy()
    else:
        eval_df = df.copy()
    rows = []
    qid_col = cfg["qid"]
    for row_idx, row in eval_df.iterrows():
        qid = str(row.get(qid_col, row_idx)) if qid_col else str(row_idx)
        rows.append({"qid": qid, "question": str(row[cfg["question"]])})
    # Preserve sample order while de-duplicating question ids.
    seen = set()
    unique_rows = []
    for item in rows:
        if item["qid"] in seen:
            continue
        seen.add(item["qid"])
        unique_rows.append(item)
    return unique_rows


def load_router_label_store(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_router_label_store(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def ensure_llm_router_dataset_labels(df, cfg, dataset, routing, n_samples, seed, args):
    path = llm_router_dataset_label_path(args, dataset, routing, n_samples, seed)
    questions = router_question_rows(df, cfg, n_samples, seed)
    store = load_router_label_store(path)
    if store is None:
        store = {
            "dataset": dataset,
            "routing": routing,
            "backend": "azure_gpt4o",
            "n_samples_requested": int(n_samples) if n_samples else None,
            "seed": seed,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "responses": [],
            "errors": [],
        }

    responses_by_qid = {str(item["qid"]): item for item in store.get("responses", [])}
    errors = store.setdefault("errors", [])
    pending = []
    for item in questions:
        qid = item["qid"]
        if (
            qid in responses_by_qid
            and responses_by_qid[qid].get("label")
            and (
                args.llm_router_allow_fallback
                or str(responses_by_qid[qid].get("raw_content", "")).strip()
            )
        ):
            continue
        pending.append(item)
    for item in progress_iter(pending, desc=f"GPT-4o router {dataset}", total=len(pending)):
        qid = item["qid"]
        question = item["question"]
        cache_raw = f"{dataset}|{qid}|{routing}|{question}"
        cache_key = hashlib.sha256(cache_raw.encode("utf-8")).hexdigest()
        try:
            label, record = azure_gpt4o_router_label(question, routing, args, cache_key)
            response = {
                "qid": qid,
                "question": question,
                "label": label,
                "raw_content": record.get("raw_content"),
                "routing": routing,
                "cache_key": cache_key,
                "attempts": record.get("attempts"),
                "max_completion_tokens": record.get("max_completion_tokens"),
                "model": record.get("model"),
                "usage": record.get("usage"),
                "status": "ok",
            }
        except Exception as exc:
            if not args.llm_router_allow_fallback:
                save_router_label_store(path, store)
                raise
            label = heuristic_router_label(question, routing)
            response = {
                "qid": qid,
                "question": question,
                "label": label,
                "raw_content": None,
                "routing": routing,
                "cache_key": cache_key,
                "status": "fallback",
                "error": str(exc),
            }
            errors.append({"qid": qid, "question": question, "error": str(exc), "fallback_label": label})
        responses_by_qid[qid] = response
        store["responses"] = list(responses_by_qid.values())
        save_router_label_store(path, store)
        if args.llm_router_sleep > 0:
            time.sleep(args.llm_router_sleep)

    store["responses"] = [responses_by_qid[item["qid"]] for item in questions if item["qid"] in responses_by_qid]
    store["n_responses"] = len(store["responses"])
    store["label_distribution"] = {
        k: round(v / max(1, len(store["responses"])), 6)
        for k, v in Counter(item["label"] for item in store["responses"]).items()
    }
    save_router_label_store(path, store)
    return store, path


def run_llm_router_api(df, cfg, specs, costs, routing, n_samples, seed, dataset, model, args):
    label_path = llm_router_dataset_label_path(args, dataset, routing, n_samples, seed)
    store = load_router_label_store(label_path)
    if store is None or int(store.get("n_responses", len(store.get("responses", [])))) == 0:
        raise RuntimeError(
            f"Missing dataset-level LLM router labels for {dataset}: {label_path}. "
            "The main scheduler should create this once per dataset before per-model scoring."
        )
    labels_by_qid = {str(item["qid"]): item["label"] for item in store["responses"]}
    qid_col = cfg["qid"]
    if qid_col:
        eval_df = df[df[qid_col].astype(str).isin(labels_by_qid)].copy()
        eval_df["_router_qid"] = eval_df[qid_col].astype(str)
    else:
        eval_df = df.loc[[int(qid) for qid in labels_by_qid if str(qid).isdigit()]].copy()
        eval_df["_router_qid"] = eval_df.index.astype(str)
    available = {s.fidelity for s in specs}
    selected = []
    labels = []
    for _, row in eval_df.iterrows():
        qid = str(row["_router_qid"])
        label = labels_by_qid[qid]
        labels.append(label)
        if routing == "binary":
            selected.append("full" if label == "yes" else "caption")
        else:
            if label == "high":
                selected.append("full")
            elif label == "medium" and "jpeg_q10" in available:
                selected.append("jpeg_q10")
            else:
                selected.append("caption")

    metrics = evaluate_selection(eval_df, np.array(selected, dtype=object), specs, costs)
    errors = store.get("errors", [])
    return {
        "status": "ok" if not any(item.get("status") == "fallback" for item in store["responses"]) else "ok_with_fallbacks",
        "router_kind": "azure_gpt4o",
        "routing": routing,
        "n_samples": int(len(eval_df)),
        "label_distribution": {k: round(v / len(labels), 6) for k, v in Counter(labels).items()},
        "accuracy": metrics["accuracy"] * 100.0,
        "avg_cost": metrics["avg_cost"],
        "cost_reduction_pct": metrics["cost_reduction_pct"],
        "fidelity_distribution": metrics["fidelity_distribution"],
        "n_errors": len(errors),
        "errors_sample": errors[:10],
        "cache_dir": str(Path(args.llm_router_cache_dir).resolve()),
        "dataset_label_store": str(Path(label_path).resolve()),
        "n_dataset_labels": int(store.get("n_responses", len(store["responses"]))),
        "n_missing_labels_for_model": int(len(labels_by_qid) - len(eval_df)),
    }


def cheapest_sufficient_labels(df, specs, costs):
    ordered = sorted(specs, key=lambda s: costs[s.fidelity])
    labels = []
    for _, row in df.iterrows():
        picked = ordered[-1].fidelity
        for spec in ordered:
            if bool(row.get(f"correct_{spec.fidelity}", False)):
                picked = spec.fidelity
                break
        labels.append(picked)
    return np.array(labels, dtype=object)


def encode_bert_questions(questions, model_name, allow_download=False, desc="BERT encode"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "bert_router requires sentence-transformers. Install/cache it or run with "
            "--bert_router_features tfidf+struct."
        ) from exc
    kwargs = {"local_files_only": not allow_download}
    try:
        model = SentenceTransformer(model_name, **kwargs)
    except TypeError:
        if not allow_download:
            raise RuntimeError(
                "Installed sentence-transformers does not support local_files_only. "
                "Use --bert_router_allow_download only if downloads are acceptable."
            )
        model = SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load BERT router model '{model_name}'. Cache it locally or use "
            "--bert_router_features tfidf+struct."
        ) from exc
    return model.encode(
        list(questions),
        show_progress_bar=(tqdm is not None),
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def fit_router_features(train_df, cfg, feature_set, max_tfidf_features, bert_model, allow_download):
    if feature_set == "bert":
        questions = train_df[cfg["question"]].fillna("").astype(str).values
        x_train = encode_bert_questions(questions, bert_model, allow_download, desc="BERT train encode")
        return x_train, {"feature_set": "bert", "bert_model": bert_model, "allow_download": allow_download}
    feature_state = fit_feature_state(train_df, cfg["question"], feature_set, max_tfidf_features)
    x_train = transform_features_with_state(train_df, feature_state)
    return x_train, {"feature_set": feature_set, "feature_state": feature_state}


def transform_router_features(df, cfg, artifact):
    feature_set = artifact["feature_set"]
    if feature_set == "bert":
        questions = df[cfg["question"]].fillna("").astype(str).values
        return encode_bert_questions(
            questions,
            artifact["bert_model"],
            artifact.get("allow_download", False),
            desc="BERT encode",
        )
    return transform_features_with_state(df, artifact["feature_state"])


def run_cv_bert_router(
    df,
    cfg,
    specs,
    costs,
    n_folds,
    seed,
    feature_set,
    max_tfidf_features,
    bert_model,
    bert_allow_download,
    dataset_name,
    model_name,
    model_artifact_dir,
    save_models,
    reuse_models,
    strict_reuse_models,
):
    rows = []
    qid_col = cfg["qid"]
    split_iter = list(make_outer_splits(df, qid_col, n_folds, seed))
    for fold, (train_qids, test_qids) in enumerate(
        progress_iter(split_iter, desc=f"{dataset_name}/{model_name} bert_router folds", total=len(split_iter)),
        start=1,
    ):
        train_df = split_by_question_ids(df, qid_col, train_qids)
        test_df = split_by_question_ids(df, qid_col, test_qids)
        artifact_path = fold_artifact_path(
            model_artifact_dir,
            dataset_name,
            model_name,
            "bert_router",
            fold,
            "LogReg",
            "none",
            feature_set,
            max_tfidf_features,
            seed + fold,
        )
        if (reuse_models or save_models) and artifact_path.exists():
            metadata, artifact = load_model_artifact(artifact_path)
            print(f"    fold {fold}: loaded router artifact {artifact_path}")
        elif reuse_models and strict_reuse_models:
            raise missing_artifact_error(artifact_path)
        else:
            y_train = cheapest_sufficient_labels(train_df, specs, costs)
            x_train, feature_artifact = fit_router_features(
                train_df,
                cfg,
                feature_set,
                max_tfidf_features,
                bert_model,
                bert_allow_download,
            )
            label_encoder = LabelEncoder()
            y_enc = label_encoder.fit_transform(y_train)
            if len(label_encoder.classes_) < 2:
                classifier = ConstantModel(float(y_enc[0]))
            else:
                classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed + fold)
                classifier.fit(x_train, y_enc)
            artifact = {
                **feature_artifact,
                "classifier": classifier,
                "label_encoder": label_encoder,
                "classes": label_encoder.classes_.tolist(),
            }
            if save_models:
                metadata = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "experiment": "bert_router",
                    "fold": fold,
                    "feature_set": feature_set,
                    "bert_model": bert_model,
                    "seed": seed + fold,
                    "train_qids": list(map(str, train_qids)),
                    "test_qids": list(map(str, test_qids)),
                    "fidelities": [s.__dict__ for s in specs],
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_model_artifact(artifact_path, artifact, metadata)
                print(f"    fold {fold}: saved router artifact {artifact_path}")

        x_test = transform_router_features(test_df, cfg, artifact)
        pred_enc = artifact["classifier"].predict(x_test)
        selected = artifact["label_encoder"].inverse_transform(np.asarray(pred_enc, dtype=int))
        metrics = evaluate_selection(test_df, selected, specs, costs)
        rows.append(
            {
                "fold": fold,
                "n_test": int(len(test_df)),
                "accuracy": metrics["accuracy"],
                "avg_cost": metrics["avg_cost"],
                "cost_reduction_pct": metrics["cost_reduction_pct"],
                "fidelity_distribution": metrics["fidelity_distribution"],
            }
        )
        print(f"    fold {fold}: bert_router acc={metrics['accuracy']*100:.2f}% cost={metrics['avg_cost']:.2f}")

    out = aggregate_folds(
        [
            {
                "fold": r["fold"],
                "n_test": r["n_test"],
                "lambda": np.nan,
                "tau": np.nan,
                "accuracy": r["accuracy"],
                "avg_cost": r["avg_cost"],
                "cost_reduction_pct": r["cost_reduction_pct"],
                "fidelity_distribution": r["fidelity_distribution"],
                "calibration": {},
            }
            for r in rows
        ]
    )
    out["router_kind"] = "supervised_fidelity_classifier"
    out["feature_set"] = feature_set
    out["target_label"] = "cheapest_sufficient_fidelity_else_full"
    out["folds"] = rows
    return out


def train_full_for_importance(df, cfg, specs, seed, max_tfidf_features):
    train_ids, cal_ids = train_test_split(
        pd.Series(df[cfg["qid"]].unique()),
        test_size=0.2,
        random_state=seed,
    )
    train_df = split_by_question_ids(df, cfg["qid"], train_ids)
    cal_df = split_by_question_ids(df, cfg["qid"], cal_ids)
    _, _, _, feature_names = build_feature_matrices(
        train_df, cal_df, cal_df, cfg["question"], "tfidf+struct", max_tfidf_features
    )
    x_train, _, _, _ = build_feature_matrices(
        train_df, cal_df, cal_df, cfg["question"], "tfidf+struct", max_tfidf_features
    )
    rows = {}
    for spec in specs:
        y = train_df[f"correct_{spec.fidelity}"].astype(int).values
        if len(np.unique(y)) < 2:
            rows[spec.fidelity] = []
            continue
        model = build_predictor("GBR", seed)
        model.fit(x_train, y)
        pairs = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])[:20]
        rows[spec.fidelity] = [{"feature": name, "importance": float(val)} for name, val in pairs]
    return rows


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def write_rows_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_main(args, dataset, model, df, cfg, specs, result_namespace="main", split_strategy=None):
    active_split_strategy = split_strategy or args.split_strategy
    print(f"  {result_namespace}: {dataset}/{model} split={active_split_strategy}")
    costs = get_costs(args)
    out = run_cv_voila(
        df,
        cfg,
        specs,
        costs,
        args.n_cv_folds,
        args.seed,
        args.lambda_grid,
        args.tau_grid,
        calibration=args.calibration,
        max_tfidf_features=args.max_tfidf_features,
        tradeoff_mode=args.tradeoff_mode,
        cost_weight=args.cost_weight,
        dataset_name=dataset,
        model_name=model,
        experiment_name=f"{result_namespace}_{active_split_strategy}_nested_validation",
        model_artifact_dir=args.model_artifact_dir,
        save_models=args.save_models,
        reuse_models=args.reuse_models,
        strict_reuse_models=args.strict_reuse_models,
        split_strategy=active_split_strategy,
        accuracy_retention=args.accuracy_retention,
        bootstrap_samples=args.bootstrap_samples,
        return_query_records=True,
    )
    query_records = out.pop("_query_records", [])
    for record in query_records:
        record.update(
            {
                "dataset": dataset,
                "model": model,
                "calibration_method": args.calibration,
                "split_strategy": active_split_strategy,
                "seed": int(args.seed),
                "cost_unit": args.cost_unit,
            }
        )
    source_path = Path(args.data_dir) / MODEL_CSV_MAP[dataset][model]
    out.update(
        {
            "dataset": dataset,
            "model": model,
            "costs": costs,
            "fixed_fidelity_baselines": fixed_fidelity_baselines(df, specs, costs),
            "provenance": {
                "source_csv": str(source_path),
                "source_sha256": sha256_file(source_path),
                "seed": int(args.seed),
                "n_rows": int(len(df)),
                "n_unique_questions": int(df[cfg["question"]].nunique()),
                "n_unique_images": int(df["image_id"].nunique()) if "image_id" in df else None,
                "fidelities": [s.fidelity for s in specs],
                "calibration": args.calibration,
                "max_tfidf_features": int(args.max_tfidf_features),
                "cost_unit": args.cost_unit,
                "duplicate_image_question_pairs": int(
                    df.duplicated(subset=["image_id", cfg["question"]], keep=False).sum()
                ) if "image_id" in df else None,
                "source_duplicate_image_question_rows": int(
                    df.attrs.get("source_duplicate_image_question_rows", 0)
                ),
                "common_keys_csv": args.common_keys_csv,
                "common_keys_sha256": (
                    sha256_file(args.common_keys_csv) if args.common_keys_csv else None
                ),
            },
        }
    )
    result_dir = Path(args.output_dir) / result_namespace / dataset / model
    write_json(result_dir / "voila_results.json", out)
    write_rows_csv(result_dir / "query_records.csv", query_records)
    return out


def run_voila_cdf(args, dataset, model, df, cfg, specs):
    print(f"  voila_cdf: {dataset}/{model}")
    costs = get_costs(args)
    out = run_cv_voila_cdf(
        df,
        cfg,
        specs,
        costs,
        args.n_cv_folds,
        args.seed,
        args.lambda_grid,
        max_tfidf_features=args.max_tfidf_features,
        tradeoff_mode=args.tradeoff_mode,
        cost_weight=args.cost_weight,
        dataset_name=dataset,
        model_name=model,
        experiment_name="main",
        model_artifact_dir=args.model_artifact_dir,
        save_models=args.save_models,
        reuse_models=args.reuse_models,
        strict_reuse_models=args.strict_reuse_models,
    )
    out.update(
        {
            "dataset": dataset,
            "model": model,
            "costs": costs,
            "fixed_fidelity_baselines": fixed_fidelity_baselines(df, specs, costs),
        }
    )
    write_json(Path(args.output_dir) / "voila_cdf" / dataset / model / "results.json", out)
    return out


def run_oracle(args, dataset, model, df, specs):
    print(f"  oracle: {dataset}/{model}")
    out = {"dataset": dataset, "model": model, **oracle_metrics(df, specs, get_costs(args))}
    write_json(Path(args.output_dir) / "oracle" / dataset / model / "results.json", out)
    return out


def run_cascade(args, dataset, model, df, cfg, specs):
    print(f"  cascade: {dataset}/{model}")
    costs = get_costs(args)
    out = run_cv_cascade(df, cfg, specs, costs, args.n_cv_folds, args.seed)
    out.update(
        {
            "dataset": dataset,
            "model": model,
            "costs": costs,
            "cost_accounting": "cumulative_sum_of_all_fidelities_reached",
            "max_possible_cascade_cost": float(sum(costs[s.fidelity] for s in specs)),
        }
    )
    write_json(Path(args.output_dir) / "baselines" / "logprob_cascade" / dataset / model / "results.json", out)
    return out


def run_llm_router(args, dataset, model, df, cfg, specs):
    print(f"  llm_router ({args.llm_router_backend}): {dataset}/{model}")
    costs = get_costs(args)
    if args.llm_router_backend == "azure_gpt4o":
        payload = run_llm_router_api(
            df,
            cfg,
            specs,
            costs,
            args.llm_router_routing,
            args.n_llm_router_samples,
            args.seed,
            dataset,
            model,
            args,
        )
    else:
        payload = run_llm_router_surrogate(
            df,
            cfg,
            specs,
            costs,
            args.llm_router_routing,
            args.n_llm_router_samples,
            args.seed,
        )
    out = {
        "dataset": dataset,
        "model": model,
        "costs": costs,
        **payload,
    }
    write_json(Path(args.output_dir) / "baselines" / "llm_router" / dataset / model / "results.json", out)
    return out


def run_llm_router_label_collection_job(args_dict, dataset):
    args = argparse.Namespace(**args_dict)
    global RECALL_MATCH_THRESHOLD
    RECALL_MATCH_THRESHOLD = args.recall_threshold
    model = next((m for m in args.models if m in MODEL_CSV_MAP.get(dataset, {})), None)
    if model is None:
        return {"kind": "llm_router_labels", "dataset": dataset, "status": "skipped", "error": "no configured model"}
    df, cfg, specs, csv_path = load_dataset(dataset, model, args.data_dir, args.max_rows, args.seed)
    print(f"\n=== llm_router labels: {dataset} ===")
    print(f"  using questions from {model}: {csv_path}")
    store, label_path = ensure_llm_router_dataset_labels(
        df,
        cfg,
        dataset,
        args.llm_router_routing,
        args.n_llm_router_samples,
        args.seed,
        args,
    )
    print(f"  label store: {label_path} ({store.get('n_responses', len(store.get('responses', [])))} labels)")
    return {"kind": "llm_router_labels", "dataset": dataset, "status": "ok", "path": str(label_path)}


def run_bert_router(args, dataset, model, df, cfg, specs):
    print(f"  bert_router ({args.bert_router_features}): {dataset}/{model}")
    try:
        payload = run_cv_bert_router(
            df,
            cfg,
            specs,
            get_costs(args),
            args.n_cv_folds,
            args.seed,
            args.bert_router_features,
            args.max_tfidf_features,
            args.bert_router_model,
            args.bert_router_allow_download,
            dataset,
            model,
            args.model_artifact_dir,
            args.save_models,
            args.reuse_models,
            args.strict_reuse_models,
        )
        out = {"dataset": dataset, "model": model, "costs": get_costs(args), **payload}
    except Exception as exc:
        if args.bert_router_fail_on_error:
            raise
        out = {
            "dataset": dataset,
            "model": model,
            "status": "skipped",
            "reason": str(exc),
            "feature_set": args.bert_router_features,
            "router_kind": "supervised_fidelity_classifier",
        }
        print(f"    skipped bert_router: {exc}")
    write_json(Path(args.output_dir) / "baselines" / "bert_router" / dataset / model / "results.json", out)
    return out


def run_obs2(args, dataset, model, df, specs):
    print(f"  obs2_quant: {dataset}/{model}")
    out = {"dataset": dataset, "model": model, **obs2_quantification(df, specs)}
    write_json(Path(args.output_dir) / "analysis" / "obs2_quantification" / dataset / model / "results.json", out)
    return out


def run_calibration_ablation(args, dataset, model, df, cfg, specs):
    print(f"  calibration ablation: {dataset}/{model}")
    variants = [
        ("none", "none", "voi"),
        ("iso", "iso", "voi"),
        ("shared_platt", "shared_platt", "voi"),
        ("legacy_sequential", "iso", "legacy"),
        ("acc", "iso", "acc"),
        ("thr", "none", "thr"),
    ]
    out = {"dataset": dataset, "model": model, "variants": {}}
    legacy_records = []
    for name, cal, policy in variants:
        print(f"    variant={name}")
        variant_result = run_cv_voila(
            df,
            cfg,
            specs,
            get_costs(args),
            args.n_cv_folds,
            args.seed,
            args.lambda_grid,
            args.tau_grid,
            calibration=cal,
            policy=policy,
            max_tfidf_features=args.max_tfidf_features,
            tradeoff_mode=args.tradeoff_mode,
            cost_weight=args.cost_weight,
            dataset_name=dataset,
            model_name=model,
            experiment_name=f"ablation_calibration_{name}_nested_validation",
            model_artifact_dir=args.model_artifact_dir,
            save_models=args.save_models,
            reuse_models=args.reuse_models,
            strict_reuse_models=args.strict_reuse_models,
            split_strategy=args.split_strategy,
            accuracy_retention=args.accuracy_retention,
            bootstrap_samples=args.bootstrap_samples,
            return_query_records=name == "legacy_sequential",
        )
        if name == "legacy_sequential":
            legacy_records = variant_result.pop("_query_records", [])
        out["variants"][name] = variant_result
    primary_names = ["none", "iso", "shared_platt"]
    summaries = {}
    for name in primary_names:
        result = out["variants"][name]
        fidelity_ece = {}
        for spec in specs:
            values = [
                fold["validation_calibration"].get(spec.fidelity, {}).get("ece")
                for fold in result["folds"]
            ]
            values = [value for value in values if value is not None]
            if values:
                fidelity_ece[spec.fidelity] = float(np.mean(values))
        validation_regrets = [
            fold.get("tradeoff", {}).get("selected", {}).get("selection_regret")
            for fold in result["folds"]
        ]
        validation_regrets = [value for value in validation_regrets if value is not None]
        summaries[name] = {
            "mean_validation_selection_regret": (
                float(np.mean(validation_regrets)) if validation_regrets else None
            ),
            "mean_validation_ece_by_fidelity": fidelity_ece,
        }
    iso_ece = summaries["iso"]["mean_validation_ece_by_fidelity"]
    eligible = []
    for name in primary_names:
        summary = summaries[name]
        ece_ok = all(
            fidelity not in iso_ece
            or summary["mean_validation_ece_by_fidelity"].get(fidelity, float("inf"))
            <= iso_ece[fidelity] + 0.01
            for fidelity in iso_ece
        )
        summary["ece_constraint_satisfied"] = bool(ece_ok)
        if ece_ok and summary["mean_validation_selection_regret"] is not None:
            eligible.append(name)
    recommended = min(
        eligible,
        key=lambda name: (summaries[name]["mean_validation_selection_regret"], primary_names.index(name)),
    ) if eligible else "iso"
    out["calibrator_selection"] = {
        "rule": "lowest validation selection regret subject to each fidelity ECE <= independent isotonic ECE + 0.01",
        "recommended": recommended,
        "summaries": summaries,
    }
    output_dir = Path(args.output_dir) / "ablations" / "calibration" / dataset / model
    write_json(output_dir / "results.json", out)
    if legacy_records:
        write_rows_csv(output_dir / "legacy_query_records.csv", legacy_records)
    return out


def run_vocab_audit(args, dataset, model, df, cfg, specs):
    """Evaluate the planned 500/1k/2k/5k TF--IDF vocabulary grid."""
    print(f"  TF-IDF vocabulary audit: {dataset}/{model}")
    vocabularies = [500, 1000, 2000, 5000]
    variants = {}
    for vocabulary in vocabularies:
        result = run_cv_voila(
            df,
            cfg,
            specs,
            get_costs(args),
            args.n_cv_folds,
            args.seed,
            args.lambda_grid,
            args.tau_grid,
            calibration=args.calibration,
            max_tfidf_features=vocabulary,
            dataset_name=dataset,
            model_name=model,
            experiment_name=f"vocab_audit_{vocabulary}_nested_validation",
            model_artifact_dir=args.model_artifact_dir,
            save_models=args.save_models,
            reuse_models=args.reuse_models,
            strict_reuse_models=args.strict_reuse_models,
            split_strategy=args.split_strategy,
            accuracy_retention=args.accuracy_retention,
            bootstrap_samples=args.bootstrap_samples,
        )
        validation_accuracy = [
            fold["tradeoff"]["selected"]["accuracy"] for fold in result["folds"]
        ]
        validation_cost = [
            fold["tradeoff"]["selected"]["avg_cost"] for fold in result["folds"]
        ]
        variants[str(vocabulary)] = {
            "result": result,
            "validation_accuracy_mean": float(np.mean(validation_accuracy)),
            "validation_mean_cost": float(np.mean(validation_cost)),
        }
    best_validation = max(row["validation_accuracy_mean"] for row in variants.values())
    accuracy_eligible = [
        vocabulary
        for vocabulary in vocabularies
        if variants[str(vocabulary)]["validation_accuracy_mean"] >= best_validation - 0.005
    ]
    selected = min(accuracy_eligible)
    output = {
        "dataset": dataset,
        "model": model,
        "cost_unit": args.cost_unit,
        "selection_rule": "smallest vocabulary within 0.5 absolute accuracy points of best validation accuracy; validation mean cost is reported for matched-cost review",
        "selected_vocabulary": selected,
        "best_validation_accuracy": best_validation,
        "variants": variants,
    }
    write_json(Path(args.output_dir) / "ablations" / "tfidf_vocabulary" / dataset / model / "results.json", output)
    return output


def run_predictor_ablation(args, dataset, model, df, cfg, specs):
    print(f"  predictor ablation: {dataset}/{model}")
    out = {"dataset": dataset, "model": model, "predictors": {}}
    for predictor in ["GBR", "LogReg", "Ridge", "MLP"]:
        print(f"    predictor={predictor}")
        out["predictors"][predictor] = run_cv_voila(
            df,
            cfg,
            specs,
            get_costs(args),
            args.n_cv_folds,
            args.seed,
            args.lambda_grid,
            args.tau_grid,
            predictor=predictor,
            calibration="iso",
            max_tfidf_features=args.max_tfidf_features,
            tradeoff_mode=args.tradeoff_mode,
            cost_weight=args.cost_weight,
            dataset_name=dataset,
            model_name=model,
            experiment_name=f"ablation_predictor_{predictor}",
            model_artifact_dir=args.model_artifact_dir,
            save_models=args.save_models,
            reuse_models=args.reuse_models,
            strict_reuse_models=args.strict_reuse_models,
        )
    write_json(Path(args.output_dir) / "ablations" / "predictor" / model / f"{dataset}.json", out)
    return out


def run_feature_ablation(args, dataset, model, df, cfg, specs):
    print(f"  feature ablation: {dataset}/{model}")
    out = {"dataset": dataset, "model": model, "features": {}}
    for feature_set in ["tfidf_only", "struct_only", "qtype_only", "tfidf+struct", "bert"]:
        print(f"    feature_set={feature_set}")
        try:
            out["features"][feature_set] = run_cv_voila(
                df,
                cfg,
                specs,
                get_costs(args),
                args.n_cv_folds,
                args.seed,
                args.lambda_grid,
                args.tau_grid,
                feature_set=feature_set,
                max_tfidf_features=args.max_tfidf_features,
                tradeoff_mode=args.tradeoff_mode,
                cost_weight=args.cost_weight,
                dataset_name=dataset,
                model_name=model,
                experiment_name=f"ablation_features_{feature_set.replace('+', '_plus_')}",
                model_artifact_dir=args.model_artifact_dir,
                save_models=args.save_models,
                reuse_models=args.reuse_models,
                strict_reuse_models=args.strict_reuse_models,
            )
        except Exception as exc:
            out["features"][feature_set] = {"status": "skipped", "reason": str(exc)}
            print(f"      skipped: {exc}")
    write_json(Path(args.output_dir) / "ablations" / "features" / dataset / model / "results.json", out)
    return out


def run_feature_importance(args, dataset, model, df, cfg, specs):
    print(f"  feature importance: {dataset}/{model}")
    top = train_full_for_importance(df, cfg, specs, args.seed, args.max_tfidf_features)
    out = {"dataset": dataset, "model": model, "top_features": top}
    write_json(Path(args.output_dir) / "analysis" / "feature_importance" / dataset / model / "results.json", out)
    rows = []
    for fidelity, features in top.items():
        for rank, item in enumerate(features, start=1):
            rows.append({"dataset": dataset, "model": model, "fidelity": fidelity, "rank": rank, **item})
    write_rows_csv(Path(args.output_dir) / "analysis" / "feature_importance" / dataset / model / "top_features.csv", rows)
    return out


def run_ood(args, source, target, model):
    print(f"  ood: {source}->{target}/{model}")
    costs = get_costs(args)
    source_df, source_cfg, source_specs, _ = load_dataset(source, model, args.data_dir, args.max_rows, args.seed)
    target_df, target_cfg, target_specs, _ = load_dataset(target, model, args.data_dir, args.max_rows, args.seed)
    shared = [s for s in source_specs if any(t.fidelity == s.fidelity for t in target_specs)]
    if not shared:
        raise ValueError(f"No shared fidelities for {source}->{target}/{model}")
    train_ids, cal_ids = train_test_split(
        pd.Series(source_df[source_cfg["qid"]].unique()),
        test_size=0.2,
        random_state=args.seed,
    )
    train_df = split_by_question_ids(source_df, source_cfg["qid"], train_ids)
    cal_df = split_by_question_ids(source_df, source_cfg["qid"], cal_ids)
    artifact_path = fold_artifact_path(
        args.model_artifact_dir,
        f"{source}_to_{target}",
        model,
        "ood",
        0,
        "GBR",
        "iso",
        "tfidf+struct",
        args.max_tfidf_features,
        args.seed,
    )
    if (args.reuse_models or args.save_models) and artifact_path.exists():
        metadata, artifact = load_model_artifact(artifact_path)
        cal_pred, target_pred = predict_with_artifact(artifact, cal_df, target_df)
        artifact_source = str(artifact_path)
        print(f"    loaded model artifact {artifact_path}")
    elif args.reuse_models and args.strict_reuse_models:
        raise missing_artifact_error(artifact_path)
    else:
        trained_result = train_fidelity_models(
            train_df,
            cal_df,
            target_df,
            shared,
            source_cfg,
            max_tfidf_features=args.max_tfidf_features,
            seed=args.seed,
            return_artifact=args.save_models,
        )
        if args.save_models:
            _, cal_pred, target_pred, _, artifact = trained_result
            metadata = {
                "source": source,
                "target": target,
                "model": model,
                "experiment": "ood",
                "predictor": "GBR",
                "calibration": "iso",
                "feature_set": "tfidf+struct",
                "max_tfidf_features": args.max_tfidf_features,
                "seed": args.seed,
                "train_qids": list(map(str, train_ids)),
                "cal_qids": list(map(str, cal_ids)),
                "fidelities": [s.__dict__ for s in shared],
                "question_col": source_cfg["question"],
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_model_artifact(artifact_path, artifact, metadata)
            artifact_source = str(artifact_path)
            print(f"    saved model artifact {artifact_path}")
        else:
            _, cal_pred, target_pred, _ = trained_result
            artifact_source = None
    lam, tau, tradeoff = tune_lam_tau(
        cal_df,
        cal_pred,
        shared,
        costs,
        args.lambda_grid,
        args.tau_grid,
        tradeoff_mode=args.tradeoff_mode,
        cost_weight=args.cost_weight,
    )
    selected = select_voi(target_pred, shared, costs, lam, tau)
    metrics = evaluate_selection(target_df, selected, target_specs, costs)
    out = {
        "source": source,
        "target": target,
        "model": model,
        "lambda": lam,
        "tau": tau,
        "tradeoff": tradeoff,
        "accuracy": metrics["accuracy"] * 100.0,
        "avg_cost": metrics["avg_cost"],
        "cost_reduction_pct": metrics["cost_reduction_pct"],
        "fidelity_distribution": metrics["fidelity_distribution"],
        "model_artifact": artifact_source,
    }
    dirname = f"{source}_to_{target}"
    write_json(Path(args.output_dir) / "ood" / dirname / model / "results.json", out)
    return out


def run_locomo_ood(args, model):
    print(f"  locomo_ood: vqav2+gqa->locomo/{model}")
    costs = get_costs(args)
    if model not in MODEL_CSV_MAP["locomo"]:
        print(f"    skipped: locomo CSV not configured for {model}")
        return None
    source_frames = []
    source_cfg = None
    source_specs = None
    for source in ["vqav2", "gqa"]:
        df, cfg, specs, _ = load_dataset(source, model, args.data_dir, args.max_rows, args.seed)
        keep = [cfg["qid"], cfg["question"]] + [f"correct_{s.fidelity}" for s in specs]
        part = df[keep].copy()
        part["source_dataset"] = source
        part[cfg["qid"]] = source + ":" + part[cfg["qid"]].astype(str)
        source_frames.append(part)
        source_cfg = {"qid": cfg["qid"], "question": cfg["question"]}
        source_specs = specs
    source_df = pd.concat(source_frames, ignore_index=True)
    target_df, target_cfg, target_specs, _ = load_dataset("locomo", model, args.data_dir, args.max_rows, args.seed)
    train_ids, cal_ids = train_test_split(
        pd.Series(source_df[source_cfg["qid"]].unique()),
        test_size=0.2,
        random_state=args.seed,
    )
    train_df = split_by_question_ids(source_df, source_cfg["qid"], train_ids)
    cal_df = split_by_question_ids(source_df, source_cfg["qid"], cal_ids)
    artifact_path = fold_artifact_path(
        args.model_artifact_dir,
        "vqav2_gqa_to_locomo",
        model,
        "locomo_ood",
        0,
        "GBR",
        "iso",
        "tfidf+struct",
        args.max_tfidf_features,
        args.seed,
    )
    if (args.reuse_models or args.save_models) and artifact_path.exists():
        metadata, artifact = load_model_artifact(artifact_path)
        cal_pred, target_pred = predict_with_artifact(artifact, cal_df, target_df)
        artifact_source = str(artifact_path)
        print(f"    loaded model artifact {artifact_path}")
    elif args.reuse_models and args.strict_reuse_models:
        raise missing_artifact_error(artifact_path)
    else:
        trained_result = train_fidelity_models(
            train_df,
            cal_df,
            target_df,
            source_specs,
            source_cfg,
            max_tfidf_features=args.max_tfidf_features,
            seed=args.seed,
            return_artifact=args.save_models,
        )
        if args.save_models:
            _, cal_pred, target_pred, _, artifact = trained_result
            metadata = {
                "source": "vqav2+gqa",
                "target": "locomo",
                "model": model,
                "experiment": "locomo_ood",
                "predictor": "GBR",
                "calibration": "iso",
                "feature_set": "tfidf+struct",
                "max_tfidf_features": args.max_tfidf_features,
                "seed": args.seed,
                "train_qids": list(map(str, train_ids)),
                "cal_qids": list(map(str, cal_ids)),
                "fidelities": [s.__dict__ for s in source_specs],
                "question_col": source_cfg["question"],
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_model_artifact(artifact_path, artifact, metadata)
            artifact_source = str(artifact_path)
            print(f"    saved model artifact {artifact_path}")
        else:
            _, cal_pred, target_pred, _ = trained_result
            artifact_source = None
    lam, tau, tradeoff = tune_lam_tau(
        cal_df,
        cal_pred,
        source_specs,
        costs,
        args.lambda_grid,
        args.tau_grid,
        tradeoff_mode=args.tradeoff_mode,
        cost_weight=args.cost_weight,
    )
    selected = select_voi(target_pred, source_specs, costs, lam, tau)
    metrics = evaluate_selection(target_df, selected, target_specs, costs)
    out = {
        "source": "vqav2+gqa",
        "target": "locomo",
        "model": model,
        "lambda": lam,
        "tau": tau,
        "tradeoff": tradeoff,
        "accuracy": metrics["accuracy"] * 100.0,
        "avg_cost": metrics["avg_cost"],
        "cost_reduction_pct": metrics["cost_reduction_pct"],
        "fidelity_distribution": metrics["fidelity_distribution"],
        "model_artifact": artifact_source,
    }
    write_json(Path(args.output_dir) / "locomo" / model / "results.json", out)
    return out


def generate_tables(output_dir):
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    main_rows = []
    for path in (output_dir / "main").glob("*/*/voila_results.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        main_rows.append(
            {
                "dataset": data["dataset"],
                "model": data["model"],
                "voila_accuracy_mean": data["accuracy_mean"],
                "voila_accuracy_std": data["accuracy_std"],
                "voila_avg_cost_mean": data["avg_cost_mean"],
                "voila_avg_cost_std": data["avg_cost_std"],
                "cost_reduction_pct_mean": data["cost_reduction_pct_mean"],
                "sample_count": sum(fold["n_test"] for fold in data["folds"]),
                "split_strategy": data.get("split_strategy"),
                "calibration": data.get("provenance", {}).get("calibration"),
                "common_keys_sha256": data.get("provenance", {}).get("common_keys_sha256"),
                "accuracy_ci_low": (
                    100.0 * data["bootstrap"]["accuracy_95ci"][0]
                    if data.get("bootstrap", {}).get("accuracy_95ci") else None
                ),
                "accuracy_ci_high": (
                    100.0 * data["bootstrap"]["accuracy_95ci"][1]
                    if data.get("bootstrap", {}).get("accuracy_95ci") else None
                ),
            }
        )
    if main_rows:
        write_rows_csv(table_dir / "pareto_figure_data.csv", main_rows)
        pd.DataFrame(main_rows).to_latex(table_dir / "main_results_table.tex", index=False, float_format="%.2f")

    stress_rows = []
    for path in (output_dir / "floodnet_stress").glob("*/*/voila_results.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        stress_rows.append(
            {
                "dataset": data["dataset"],
                "model": data["model"],
                "accuracy_mean": data["accuracy_mean"],
                "accuracy_std": data["accuracy_std"],
                "avg_cost_mean": data["avg_cost_mean"],
                "sample_count": sum(fold["n_test"] for fold in data["folds"]),
                "split_strategy": data.get("split_strategy"),
                "common_keys_sha256": data.get("provenance", {}).get("common_keys_sha256"),
            }
        )
    if stress_rows:
        write_rows_csv(table_dir / "floodnet_leave_template_out.csv", stress_rows)
        pd.DataFrame(stress_rows).to_latex(
            table_dir / "floodnet_leave_template_out.tex", index=False, float_format="%.2f"
        )

    voila_cdf_rows = []
    for path in (output_dir / "voila_cdf").glob("*/*/results.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        voila_cdf_rows.append(
            {
                "dataset": data["dataset"],
                "model": data["model"],
                "voila_cdf_accuracy_mean": data["accuracy_mean"],
                "voila_cdf_accuracy_std": data["accuracy_std"],
                "voila_cdf_avg_cost_mean": data["avg_cost_mean"],
                "voila_cdf_avg_cost_std": data["avg_cost_std"],
                "cost_reduction_pct_mean": data["cost_reduction_pct_mean"],
            }
        )
    if voila_cdf_rows:
        write_rows_csv(table_dir / "voila_cdf_table.csv", voila_cdf_rows)
        pd.DataFrame(voila_cdf_rows).to_latex(table_dir / "voila_cdf_table.tex", index=False, float_format="%.2f")

    bert_router_rows = []
    for path in (output_dir / "baselines" / "bert_router").glob("*/*/results.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        row = {
            "dataset": data.get("dataset"),
            "model": data.get("model"),
            "status": data.get("status", "ok"),
            "feature_set": data.get("feature_set"),
        }
        if data.get("status") != "skipped":
            row.update(
                {
                    "accuracy_mean": data.get("accuracy_mean"),
                    "accuracy_std": data.get("accuracy_std"),
                    "avg_cost_mean": data.get("avg_cost_mean"),
                    "avg_cost_std": data.get("avg_cost_std"),
                    "cost_reduction_pct_mean": data.get("cost_reduction_pct_mean"),
                }
            )
        else:
            row["reason"] = data.get("reason")
        bert_router_rows.append(row)
    if bert_router_rows:
        write_rows_csv(table_dir / "bert_router_table.csv", bert_router_rows)
        pd.DataFrame(bert_router_rows).to_latex(table_dir / "bert_router_table.tex", index=False, float_format="%.2f")

    oracle_rows = []
    for path in (output_dir / "oracle").glob("*/*/results.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        oracle_rows.append(data)
    if oracle_rows:
        write_rows_csv(table_dir / "oracle_comparison.csv", oracle_rows)
        pd.DataFrame(oracle_rows).to_latex(table_dir / "oracle_comparison.tex", index=False, float_format="%.2f")

    ood_rows = []
    for path in (output_dir / "ood").glob("*/*/results.json"):
        with open(path, encoding="utf-8") as f:
            ood_rows.append(json.load(f))
    if ood_rows:
        write_rows_csv(table_dir / "ood_table.csv", ood_rows)
        pd.DataFrame(ood_rows).to_latex(table_dir / "ood_table.tex", index=False, float_format="%.2f")

    locomo_rows = []
    for path in (output_dir / "locomo").glob("*/results.json"):
        with open(path, encoding="utf-8") as f:
            locomo_rows.append(json.load(f))
    if locomo_rows:
        write_rows_csv(table_dir / "locomo_table.csv", locomo_rows)
        pd.DataFrame(locomo_rows).to_latex(table_dir / "locomo_table.tex", index=False, float_format="%.2f")

    print(f"  tables written under {table_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Run VOILA paper experiments from existing CSV outputs.")
    p.add_argument("--datasets", nargs="+", default=["floodnet", "vqav2", "gqa", "textvqa"], choices=list(DATASET_CONFIGS))
    p.add_argument("--models", nargs="+", default=["pixtral", "llava", "llama4", "qwen7b", "qwen72b", "qwen235b"])
    p.add_argument(
        "--experiments",
        nargs="+",
        default=PERCOM_OFFLINE_EXPERIMENTS,
        choices=EXPERIMENTS,
        help="Defaults to the cache-only PerCom experiment suite; request llm_router explicitly if needed.",
    )
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument(
        "--common_keys_csv",
        default=None,
        help="Optional image_id/question manifest used to enforce a matched cross-model population.",
    )
    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--n_cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rows", type=int, default=None, help="Optional per CSV row cap for smoke tests.")
    p.add_argument("--max_tfidf_features", type=int, default=500)
    p.add_argument("--lambda_grid", nargs="+", type=float, default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    p.add_argument("--tau_grid", nargs="+", type=float, default=[-0.1, -0.05, 0.0, 0.05, 0.1, 0.15])
    p.add_argument("--full_cost", type=float, default=None, help="Uniformly rescale the default cost table so full has this value.")
    p.add_argument("--cost", nargs="*", default=[], help="Override individual costs, e.g. --cost caption=5 jpeg_q10=30 full=100.")
    p.add_argument("--cost_unit", choices=["normalized", "bytes"], default="normalized")
    p.add_argument("--calibration", choices=["none", "iso", "temp", "shared_platt"], default="none")
    p.add_argument(
        "--split_strategy",
        choices=["auto", "question_id", "image_grouped", "question_grouped"],
        default="auto",
        help="auto uses image-grouped folds for FloodNet and question-id folds elsewhere.",
    )
    p.add_argument("--accuracy_retention", type=float, default=0.95)
    p.add_argument("--bootstrap_samples", type=int, default=1000)
    p.add_argument("--tradeoff_mode", choices=["pareto_knee", "accuracy_cost"], default="pareto_knee")
    p.add_argument("--cost_weight", type=float, default=0.5, help="Only used when --tradeoff_mode accuracy_cost.")
    p.add_argument("--recall_threshold", type=float, default=70.0, help="rapidfuzz partial-ratio threshold for fuzzy recall correctness.")
    p.add_argument(
        "--save_models",
        action="store_true",
        default=True,
        help="Save fitted TF-IDF, GBR, and calibrator artifacts for each CV fold. Enabled by default.",
    )
    p.add_argument("--no_save_models", action="store_false", dest="save_models", help="Disable model artifact checkpointing.")
    p.add_argument("--reuse_models", action="store_true", help="Load saved fold artifacts and only rerun calibrated prediction/routing.")
    p.add_argument(
        "--strict_reuse_models",
        action="store_true",
        help="With --reuse_models, fail immediately if an expected artifact is missing instead of retraining.",
    )
    p.add_argument("--model_artifact_dir", default=str(ROOT / "model_artifacts"), help="Directory for reusable fold model artifacts.")
    p.add_argument("--n_llm_router_samples", type=int, default=1000)
    p.add_argument("--llm_router_routing", choices=["binary", "ternary"], default="ternary")
    p.add_argument("--llm_router_backend", choices=["heuristic", "azure_gpt4o"], default="heuristic")
    p.add_argument(
        "--azure_openai_url",
        default="https://azureapi.zotgpt.uci.edu/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01",
    )
    p.add_argument("--azure_openai_api_key_env", default="AZURE_OPENAI_API_KEY")
    p.add_argument(
        "--azure_openai_api_key_file",
        default=str(DEFAULT_AZURE_OPENAI_API_KEY_FILE),
        help="Local ignored file containing the Azure OpenAI API key. Env var is used if this file is absent.",
    )
    p.add_argument("--llm_router_cache_dir", default=str(ROOT / "results" / "cache" / "llm_router"))
    p.add_argument("--llm_router_timeout", type=float, default=60.0)
    p.add_argument("--llm_router_sleep", type=float, default=0.0)
    p.add_argument("--llm_router_retries", type=int, default=5)
    p.add_argument(
        "--llm_router_max_completion_tokens",
        type=int,
        default=128,
        help="Visible router answer budget. Retries double this budget when Azure returns empty content.",
    )
    p.add_argument(
        "--llm_router_allow_fallback",
        action="store_true",
        help="If Azure GPT-4o fails, use the offline heuristic for that question. Off by default for paper runs.",
    )
    p.add_argument("--bert_router_features", choices=["bert", "tfidf+struct"], default="bert")
    p.add_argument("--bert_router_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument(
        "--bert_router_allow_download",
        action="store_true",
        help="Allow sentence-transformers to download the BERT router encoder if it is not cached locally.",
    )
    p.add_argument(
        "--bert_router_fail_on_error",
        action="store_true",
        help="Fail the run if bert_router cannot load/train instead of writing a skipped result.",
    )
    p.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel dataset/model worker processes. Use 2-4 first for large CSVs; <=0 uses CPU count - 1.",
    )
    return p.parse_args()


def normalize_args(args):
    args.output_dir = str(Path(args.output_dir).resolve())
    args.data_dir = str(Path(args.data_dir).resolve())
    args.model_artifact_dir = str(Path(args.model_artifact_dir).resolve())
    args.llm_router_cache_dir = str(Path(args.llm_router_cache_dir).resolve())
    args.azure_openai_api_key_file = str(Path(args.azure_openai_api_key_file).resolve())
    if args.common_keys_csv:
        args.common_keys_csv = str(Path(args.common_keys_csv).resolve())
    args.resolved_costs = resolve_costs_from_args(args)
    if int(args.n_jobs) <= 0:
        args.n_jobs = max(1, (os.cpu_count() or 2) - 1)
    else:
        args.n_jobs = max(1, int(args.n_jobs))
    return args


def run_pairwise_job(args_dict, dataset, model, experiments):
    args = argparse.Namespace(**args_dict)
    global RECALL_MATCH_THRESHOLD
    RECALL_MATCH_THRESHOLD = args.recall_threshold
    try:
        df, cfg, specs, csv_path = load_dataset(
            dataset, model, args.data_dir, args.max_rows, args.seed, args.common_keys_csv
        )
    except Exception as exc:
        print(f"[skip] {dataset}/{model}: {exc}")
        return {"kind": "pairwise", "dataset": dataset, "model": model, "status": "skipped", "error": str(exc)}

    print(f"\n=== {dataset}/{model} ===")
    print(f"  loaded {len(df)} rows from {csv_path}")
    print(f"  fidelities: {[s.fidelity + ':' + s.tag for s in specs]}")

    if "oracle" in experiments:
        run_oracle(args, dataset, model, df, specs)
    if "obs2_quant" in experiments:
        run_obs2(args, dataset, model, df, specs)
    if "main" in experiments:
        run_main(args, dataset, model, df, cfg, specs)
    if "floodnet_stress" in experiments and dataset == "floodnet":
        run_main(
            args,
            dataset,
            model,
            df,
            cfg,
            specs,
            result_namespace="floodnet_stress",
            split_strategy="question_grouped",
        )
    if "voila_cdf" in experiments:
        run_voila_cdf(args, dataset, model, df, cfg, specs)
    if "cascade" in experiments:
        run_cascade(args, dataset, model, df, cfg, specs)
    if "llm_router" in experiments:
        run_llm_router(args, dataset, model, df, cfg, specs)
    if "bert_router" in experiments:
        run_bert_router(args, dataset, model, df, cfg, specs)
    if "vocab_audit" in experiments and dataset == "floodnet":
        run_vocab_audit(args, dataset, model, df, cfg, specs)
    if "ablation_calibration" in experiments and dataset in {"vqav2", "gqa", "textvqa", "floodnet"}:
        run_calibration_ablation(args, dataset, model, df, cfg, specs)
    if "ablation_predictor" in experiments and dataset in {"vqav2", "gqa"} and model == "pixtral":
        run_predictor_ablation(args, dataset, model, df, cfg, specs)
    if "ablation_features" in experiments and dataset == "vqav2" and model == "pixtral":
        run_feature_ablation(args, dataset, model, df, cfg, specs)
    if "feature_importance" in experiments and dataset in {"vqav2", "gqa"}:
        run_feature_importance(args, dataset, model, df, cfg, specs)

    return {"kind": "pairwise", "dataset": dataset, "model": model, "status": "ok"}


def run_ood_job(args_dict, source, target, model):
    args = argparse.Namespace(**args_dict)
    global RECALL_MATCH_THRESHOLD
    RECALL_MATCH_THRESHOLD = args.recall_threshold
    try:
        run_ood(args, source, target, model)
        return {"kind": "ood", "source": source, "target": target, "model": model, "status": "ok"}
    except Exception as exc:
        print(f"[skip] ood {source}->{target}/{model}: {exc}")
        return {"kind": "ood", "source": source, "target": target, "model": model, "status": "skipped", "error": str(exc)}


def run_locomo_ood_job(args_dict, model):
    args = argparse.Namespace(**args_dict)
    global RECALL_MATCH_THRESHOLD
    RECALL_MATCH_THRESHOLD = args.recall_threshold
    try:
        run_locomo_ood(args, model)
        return {"kind": "locomo_ood", "model": model, "status": "ok"}
    except Exception as exc:
        print(f"[skip] locomo_ood/{model}: {exc}")
        return {"kind": "locomo_ood", "model": model, "status": "skipped", "error": str(exc)}


def run_jobs(jobs, n_jobs, label):
    if not jobs:
        return []
    print(f"\nRunning {len(jobs)} {label} job(s) with n_jobs={n_jobs}", flush=True)
    if n_jobs <= 1:
        return [fn(*job_args) for fn, job_args in progress_iter(jobs, desc=label, total=len(jobs))]

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_map = {executor.submit(fn, *job_args): (fn, job_args) for fn, job_args in jobs}
        for future in progress_iter(as_completed(future_map), desc=label, total=len(future_map)):
            fn, job_args = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"[error] parallel job failed in {fn.__name__}: {exc}")
                results.append({"kind": label, "status": "failed", "error": str(exc)})
    return results


def main():
    args = normalize_args(parse_args())
    global RECALL_MATCH_THRESHOLD
    RECALL_MATCH_THRESHOLD = args.recall_threshold
    print(f"VOILA experiments | data={args.data_dir} | output={args.output_dir} | n_jobs={args.n_jobs}", flush=True)
    args_dict = vars(args).copy()
    manifest_sources = []
    for dataset in args.datasets:
        for model in args.models:
            relative = MODEL_CSV_MAP.get(dataset, {}).get(model)
            if not relative:
                continue
            source = Path(args.data_dir) / relative
            if source.exists():
                manifest_sources.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "path": str(source.resolve()),
                        "sha256": sha256_file(source),
                        "bytes": source.stat().st_size,
                    }
                )
    manifest_payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "arguments": args_dict,
        "runner_sha256": sha256_file(__file__),
        "sources": manifest_sources,
        "no_new_vlm_inference": "true unless llm_router is explicitly requested with azure_gpt4o",
    }
    manifest_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
    write_json(Path(args.output_dir) / "run_manifests" / manifest_name, manifest_payload)
    write_json(Path(args.output_dir) / "run_manifest.json", manifest_payload)

    pair_experiments = [e for e in args.experiments if e in PAIRWISE_EXPERIMENTS]
    sequential_router_experiments = []
    if "llm_router" in pair_experiments and args.llm_router_backend == "azure_gpt4o":
        pair_experiments = [e for e in pair_experiments if e != "llm_router"]
        sequential_router_experiments = ["llm_router"]

    pair_jobs = []
    if pair_experiments:
        for dataset in args.datasets:
            for model in args.models:
                if model in MODEL_CSV_MAP.get(dataset, {}):
                    pair_jobs.append((run_pairwise_job, (args_dict, dataset, model, pair_experiments)))
        run_jobs(pair_jobs, args.n_jobs, "dataset/model")

    if sequential_router_experiments:
        router_datasets = [
            dataset
            for dataset in args.datasets
            if any(model in MODEL_CSV_MAP.get(dataset, {}) for model in args.models)
        ]
        label_jobs = [(run_llm_router_label_collection_job, (args_dict, dataset)) for dataset in router_datasets]
        print("Collecting Azure GPT-4o router labels once per dataset.", flush=True)
        run_jobs(label_jobs, 1, "LLM router labels")

        router_eval_jobs = []
        for dataset in router_datasets:
            for model in args.models:
                if model in MODEL_CSV_MAP.get(dataset, {}):
                    router_eval_jobs.append((run_pairwise_job, (args_dict, dataset, model, sequential_router_experiments)))
        run_jobs(router_eval_jobs, args.n_jobs, "LLM router scoring")

    if "ood" in args.experiments:
        ood_jobs = []
        for model in args.models:
            if model in MODEL_CSV_MAP["vqav2"] and model in MODEL_CSV_MAP["gqa"]:
                ood_jobs.append((run_ood_job, (args_dict, "vqav2", "gqa", model)))
                ood_jobs.append((run_ood_job, (args_dict, "gqa", "vqav2", model)))
        run_jobs(ood_jobs, args.n_jobs, "OOD")

    if "locomo_ood" in args.experiments:
        locomo_jobs = [(run_locomo_ood_job, (args_dict, model)) for model in args.models]
        run_jobs(locomo_jobs, args.n_jobs, "LoCoMo OOD")

    if "tables" in args.experiments:
        generate_tables(args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
