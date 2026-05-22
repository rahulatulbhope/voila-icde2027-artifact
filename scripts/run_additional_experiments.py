#!/usr/bin/env python3
"""
Additional VOILA paper experiments from additional_exp.md.

This runner intentionally lives outside run_all_experiments.py. It reuses the
existing data loading, fuzzy correctness, VOILA routing, and baseline utilities,
but writes the newer appendix/reviewer-response experiments under:

    scripts/results_final/additional/

Examples
--------
Run the full additional suite:
    python scripts/run_additional_experiments.py --experiments all

Run only the blocking experiments:
    python scripts/run_additional_experiments.py --experiments feature_ood ece_calibration extended_ood oracle_decomp

Smoke test:
    python scripts/run_additional_experiments.py --experiments oracle_decomp cascade_degeneracy --datasets vqav2 --models pixtral --max_rows 500 --n_cv_folds 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_all_experiments as core


ADDITIONAL_EXPERIMENTS = [
    "feature_ood",
    "tfidf_vocab",
    "ece_calibration",
    "extended_ood",
    "oracle_decomp",
    "floodnet_analysis",
    "cascade_degeneracy",
    "fidelity_saturation",
    "roc_escalation",
    "reliability_diagram",
    "routing_words",
    "iso_accuracy_contour",
    "tables",
]

FIDELITY_COLORS = {
    "caption": "#1f77b4",
    "resize_32x32": "#2ca02c",
    "jpeg_q1": "#ff7f0e",
    "jpeg_q10": "#d62728",
    "full": "#9467bd",
}

FEATURE_GROUPS = {
    "tfidf_only": {
        "use_tfidf": True,
        "struct_names": [],
        "description": "TF-IDF unigrams only",
    },
    "qtype_only": {
        "use_tfidf": False,
        "struct_names": [
            "is_yes_no",
            "starts_how_many",
            "has_color_word",
            "has_spatial_word",
            "has_ocr_word",
        ],
        "description": "compact question-type flags",
    },
    "lexical_only": {
        "use_tfidf": False,
        "struct_names": [
            "len_tokens",
            "has_numeric_token",
            "has_counting_word",
            "has_wh_word",
            "starts_how",
            "has_category_word",
            "has_activity_word",
        ],
        "description": "lexical/statistical flags excluding compact qtype flags",
    },
    "qtype_lexical": {
        "use_tfidf": False,
        "struct_names": core.STRUCT_FEATURE_NAMES,
        "description": "all structured question flags",
    },
    "full": {
        "use_tfidf": True,
        "struct_names": core.STRUCT_FEATURE_NAMES,
        "description": "TF-IDF plus all structured question flags",
    },
}

MODEL_LABELS = {
    "pixtral": "Pixtral-12B-2409",
    "llava": "LLaVA-1.5-7B-hf",
    "llama4": "Llama-4-Maverick",
    "qwen7b": "Qwen2.5-VL-7B-Instruct",
    "qwen72b": "Qwen2-VL-72B-Instruct",
    "qwen235b": "Qwen3-VL-235B-A22B",
}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def ensure_plot_dir(args) -> Path:
    out = Path(args.output_dir) / "additional" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Run additional VOILA experiments from additional_exp.md.")
    p.add_argument("--experiments", nargs="+", default=["all"], choices=["all"] + ADDITIONAL_EXPERIMENTS)
    p.add_argument("--datasets", nargs="+", default=["vqav2", "gqa", "textvqa", "floodnet"])
    p.add_argument("--models", nargs="+", default=["pixtral", "llava", "llama4", "qwen7b", "qwen72b", "qwen235b"])
    p.add_argument("--feature_ood_models", nargs="+", default=["pixtral", "qwen7b"])
    p.add_argument("--calibration_pairs", nargs="+", default=["vqav2:pixtral", "gqa:qwen7b"])
    p.add_argument("--oracle_decomp_pairs", nargs="+", default=["vqav2:pixtral", "gqa:pixtral"])
    p.add_argument("--roc_pairs", nargs="+", default=["vqav2:pixtral", "vqav2:qwen72b"])
    p.add_argument("--data_dir", default=str(core.DEFAULT_DATA_DIR))
    p.add_argument("--output_dir", default=str(ROOT / "scripts" / "results_final"))
    p.add_argument("--model_artifact_dir", default=str(ROOT / "model_artifacts" / "additional"))
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--n_cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tfidf_features", type=int, default=5000)
    p.add_argument("--tfidf_vocab_sizes", nargs="+", type=int, default=[500, 1000, 2000, 5000])
    p.add_argument("--lambda_grid", nargs="+", type=float, default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    p.add_argument("--tau_grid", nargs="+", type=float, default=[-0.1, -0.05, 0.0, 0.05, 0.1, 0.15])
    p.add_argument("--tradeoff_mode", choices=["pareto_knee", "accuracy_cost"], default="pareto_knee")
    p.add_argument("--cost_weight", type=float, default=0.5)
    p.add_argument("--full_cost", type=float, default=None)
    p.add_argument("--cost", nargs="*", default=[])
    p.add_argument("--reuse_artifacts", action="store_true", help="Reuse additional experiment checkpoints when present.")
    p.add_argument("--no_save_artifacts", action="store_false", dest="save_artifacts", default=True)
    return p.parse_args()


def resolve_costs(args) -> Dict[str, float]:
    return core.resolve_costs_from_args(args)


def experiment_set(args) -> List[str]:
    if "all" in args.experiments:
        return list(ADDITIONAL_EXPERIMENTS)
    return list(dict.fromkeys(args.experiments))


def parse_pair(text: str) -> Tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"Expected dataset:model pair, got {text!r}")
    dataset, model = text.split(":", 1)
    return dataset, model


def load_dataset(args, dataset: str, model: str):
    return core.load_dataset(dataset, model, args.data_dir, args.max_rows, args.seed)


def split_source(df: pd.DataFrame, cfg: Dict[str, Any], seed: int, test_size: float = 0.2):
    qids = pd.Series(df[cfg["qid"]].unique())
    train_cal_ids, test_ids = train_test_split(qids, test_size=test_size, random_state=seed)
    train_ids, cal_ids = train_test_split(pd.Series(train_cal_ids), test_size=0.25, random_state=seed + 1)
    return (
        core.split_by_question_ids(df, cfg["qid"], train_ids),
        core.split_by_question_ids(df, cfg["qid"], cal_ids),
        core.split_by_question_ids(df, cfg["qid"], test_ids),
        list(map(str, train_ids)),
        list(map(str, cal_ids)),
        list(map(str, test_ids)),
    )


def feature_artifact_path(args, experiment: str, dataset: str, model: str, label: str, seed: int) -> Path:
    safe = label.replace("+", "_plus_").replace("/", "_").replace(":", "_")
    return Path(args.model_artifact_dir) / experiment / dataset / model / f"{safe}_seed{seed}.joblib"


def fit_custom_feature_state(train_df, question_col: str, feature_group: str, max_tfidf_features: int):
    spec = FEATURE_GROUPS[feature_group]
    questions = train_df[question_col].fillna("").astype(str).values
    state = {
        "question_col": question_col,
        "feature_group": feature_group,
        "max_tfidf_features": max_tfidf_features,
        "tfidf": None,
        "struct_names": list(spec["struct_names"]),
        "feature_names": [],
    }
    if spec["use_tfidf"]:
        tfidf = TfidfVectorizer(
            lowercase=True,
            tokenizer=str.split,
            token_pattern=None,
            max_features=max_tfidf_features,
        )
        tfidf.fit(questions)
        state["tfidf"] = tfidf
        state["feature_names"].extend(tfidf.get_feature_names_out().tolist())
    state["feature_names"].extend(state["struct_names"])
    return state


def transform_custom_features(df, state):
    question_col = state["question_col"]
    questions = df[question_col].fillna("").astype(str).values
    blocks = []
    if state["tfidf"] is not None:
        blocks.append(state["tfidf"].transform(questions).astype(np.float32).toarray())
    if state["struct_names"]:
        blocks.append(core.structured_features(questions, state["struct_names"]))
    if not blocks:
        raise ValueError("Feature state produced no features.")
    return np.hstack(blocks).astype(np.float32)


def fit_voila_artifact(
    train_df,
    cal_df,
    specs,
    cfg,
    feature_group: str,
    max_tfidf_features: int,
    seed: int,
):
    feature_state = fit_custom_feature_state(train_df, cfg["question"], feature_group, max_tfidf_features)
    x_train = transform_custom_features(train_df, feature_state)
    x_cal = transform_custom_features(cal_df, feature_state)
    trained = {}
    cal_pred = pd.DataFrame(index=cal_df.index)
    raw_cal_pred = pd.DataFrame(index=cal_df.index)
    for spec in specs:
        y_train = train_df[f"correct_{spec.fidelity}"].astype(int).values
        y_cal = cal_df[f"correct_{spec.fidelity}"].astype(int).values
        if len(np.unique(y_train)) < 2:
            model = core.ConstantModel(float(np.mean(y_train)))
        else:
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=2,
                random_state=seed,
            )
            model.fit(x_train, y_train)
        raw_cal = core.raw_predict(model, x_cal)
        calibrator = core.build_calibrator("iso")
        if len(np.unique(y_cal)) < 2:
            calibrator = core.NoCalibrator()
        calibrator.fit(raw_cal, y_cal)
        p_cal = np.clip(calibrator.transform(raw_cal), 0.0, 1.0)
        raw_cal_pred[f"p_{spec.fidelity}"] = np.clip(raw_cal, 0.0, 1.0)
        cal_pred[f"p_{spec.fidelity}"] = p_cal
        trained[spec.fidelity] = {
            "model": model,
            "calibrator": calibrator,
            "y_cal": y_cal,
            "raw_cal": np.clip(raw_cal, 0.0, 1.0),
            "p_cal": p_cal,
        }
    artifact = {
        "feature_state": feature_state,
        "trained": trained,
        "feature_group": feature_group,
        "max_tfidf_features": max_tfidf_features,
        "seed": seed,
    }
    return artifact, cal_pred, raw_cal_pred


def predict_voila_artifact(artifact, df):
    x = transform_custom_features(df, artifact["feature_state"])
    pred = pd.DataFrame(index=df.index)
    raw = pd.DataFrame(index=df.index)
    for fidelity, fitted in artifact["trained"].items():
        raw_scores = core.raw_predict(fitted["model"], x)
        raw[f"p_{fidelity}"] = np.clip(raw_scores, 0.0, 1.0)
        pred[f"p_{fidelity}"] = np.clip(fitted["calibrator"].transform(raw_scores), 0.0, 1.0)
    return pred, raw


def get_or_train_artifact(args, experiment, dataset, model, label, train_df, cal_df, specs, cfg, feature_group, max_tfidf_features):
    path = feature_artifact_path(args, experiment, dataset, model, label, args.seed)
    if args.reuse_artifacts and path.exists():
        payload = joblib.load(path)
        artifact = payload["artifact"]
        cal_pred, raw_cal = predict_voila_artifact(artifact, cal_df)
        print(f"    loaded artifact {path}")
        return artifact, cal_pred, raw_cal, str(path)
    artifact, cal_pred, raw_cal = fit_voila_artifact(
        train_df,
        cal_df,
        specs,
        cfg,
        feature_group,
        max_tfidf_features,
        args.seed,
    )
    if args.save_artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "metadata": {
                    "experiment": experiment,
                    "dataset": dataset,
                    "model": model,
                    "label": label,
                    "feature_group": feature_group,
                    "max_tfidf_features": max_tfidf_features,
                    "seed": args.seed,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "artifact": artifact,
            },
            path,
            compress=3,
        )
        print(f"    saved artifact {path}")
        return artifact, cal_pred, raw_cal, str(path)
    return artifact, cal_pred, raw_cal, None


def tune_and_eval(cal_df, cal_pred, eval_df, eval_pred, train_specs, eval_specs, costs, args):
    lam, tau, tradeoff = core.tune_lam_tau(
        cal_df,
        cal_pred,
        train_specs,
        costs,
        args.lambda_grid,
        args.tau_grid,
        tradeoff_mode=args.tradeoff_mode,
        cost_weight=args.cost_weight,
    )
    selected = core.select_voi(eval_pred, train_specs, costs, lam, tau)
    metrics = core.evaluate_selection(eval_df, selected, eval_specs, costs)
    return selected, {
        "lambda": lam,
        "tau": tau,
        "tradeoff": tradeoff,
        "accuracy": metrics["accuracy"] * 100.0,
        "avg_cost": metrics["avg_cost"],
        "cost_reduction_pct": metrics["cost_reduction_pct"],
        "fidelity_distribution": metrics["fidelity_distribution"],
    }


def run_feature_ood(args, costs):
    print("\n=== EXP-1 feature group OOD ablation ===")
    out_root = Path(args.output_dir) / "additional" / "feature_ood"
    rows = []
    targets = [d for d in ["vqav2", "gqa", "textvqa", "floodnet"] if d in args.datasets]
    for model in args.feature_ood_models:
        if model not in core.MODEL_CSV_MAP.get("vqav2", {}):
            continue
        print(f"  source vqav2/{model}")
        source_df, source_cfg, source_specs, _ = load_dataset(args, "vqav2", model)
        train_df, cal_df, source_test_df, train_ids, cal_ids, test_ids = split_source(source_df, source_cfg, args.seed)
        for feature_group in FEATURE_GROUPS:
            label = f"{feature_group}_tfidf{args.max_tfidf_features}"
            artifact, cal_pred, _, artifact_path = get_or_train_artifact(
                args,
                "feature_ood",
                "vqav2",
                model,
                label,
                train_df,
                cal_df,
                source_specs,
                source_cfg,
                feature_group,
                args.max_tfidf_features,
            )
            result = {
                "source": "vqav2",
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "feature_group": feature_group,
                "feature_description": FEATURE_GROUPS[feature_group]["description"],
                "max_tfidf_features": args.max_tfidf_features,
                "model_artifact": artifact_path,
                "targets": {},
                "split": {
                    "n_train": len(train_df),
                    "n_cal": len(cal_df),
                    "n_source_test": len(source_test_df),
                },
            }
            for target in targets:
                if model not in core.MODEL_CSV_MAP.get(target, {}):
                    continue
                if target == "vqav2":
                    eval_df, eval_cfg, eval_specs = source_test_df, source_cfg, source_specs
                    split = "source_test"
                else:
                    eval_df, eval_cfg, eval_specs, _ = load_dataset(args, target, model)
                    split = "target_all"
                shared = [s for s in source_specs if any(t.fidelity == s.fidelity for t in eval_specs)]
                eval_pred, _ = predict_voila_artifact(artifact, eval_df)
                _, metrics = tune_and_eval(cal_df, cal_pred, eval_df, eval_pred, shared, eval_specs, costs, args)
                metrics.update({"n_eval": len(eval_df), "split": split})
                result["targets"][target] = metrics
                rows.append(
                    {
                        "source": "vqav2",
                        "target": target,
                        "model": model,
                        "feature_group": feature_group,
                        "accuracy": metrics["accuracy"],
                        "avg_cost": metrics["avg_cost"],
                        "cost_reduction_pct": metrics["cost_reduction_pct"],
                        "n_eval": metrics["n_eval"],
                    }
                )
            write_json(out_root / "vqav2_to_all" / model / feature_group / "results.json", result)
    write_csv(Path(args.output_dir) / "additional" / "tables" / "feature_ood_table.csv", rows)
    if rows:
        pd.DataFrame(rows).to_latex(
            Path(args.output_dir) / "additional" / "tables" / "feature_ood_table.tex",
            index=False,
            float_format="%.2f",
        )


def run_tfidf_vocab(args, costs):
    print("\n=== EXP-1b TF-IDF vocabulary size ablation ===")
    out_root = Path(args.output_dir) / "additional" / "tfidf_vocab"
    rows = []
    for model in args.feature_ood_models:
        if model not in core.MODEL_CSV_MAP.get("vqav2", {}):
            continue
        source_df, cfg, specs, _ = load_dataset(args, "vqav2", model)
        train_df, cal_df, test_df, *_ = split_source(source_df, cfg, args.seed)
        for vocab_size in args.tfidf_vocab_sizes:
            label = f"tfidf_only_vocab{vocab_size}"
            artifact, cal_pred, _, artifact_path = get_or_train_artifact(
                args,
                "tfidf_vocab",
                "vqav2",
                model,
                label,
                train_df,
                cal_df,
                specs,
                cfg,
                "tfidf_only",
                vocab_size,
            )
            test_pred, _ = predict_voila_artifact(artifact, test_df)
            _, metrics = tune_and_eval(cal_df, cal_pred, test_df, test_pred, specs, specs, costs, args)
            payload = {
                "dataset": "vqav2",
                "model": model,
                "feature_group": "tfidf_only",
                "max_tfidf_features": vocab_size,
                "model_artifact": artifact_path,
                **metrics,
            }
            write_json(out_root / "vqav2" / model / f"vocab_{vocab_size}" / "results.json", payload)
            rows.append(payload)
    write_csv(Path(args.output_dir) / "additional" / "tables" / "tfidf_vocab_table.csv", rows)


def ece_score(y_true, confidence, n_bins=15):
    y_true = np.asarray(y_true, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    mask = np.isfinite(confidence)
    y_true = y_true[mask]
    confidence = np.clip(confidence[mask], 0.0, 1.0)
    if len(y_true) == 0:
        return float("nan")
    order = np.argsort(confidence)
    bins = np.array_split(order, min(n_bins, len(order)))
    total = len(y_true)
    ece = 0.0
    details = []
    for idx in bins:
        if len(idx) == 0:
            continue
        avg_conf = float(np.mean(confidence[idx]))
        avg_acc = float(np.mean(y_true[idx]))
        weight = len(idx) / total
        ece += weight * abs(avg_conf - avg_acc)
        details.append({"n": int(len(idx)), "avg_confidence": avg_conf, "avg_accuracy": avg_acc})
    return float(ece), details


def logprob_confidence(df, spec):
    lp = core.logprob_scores(df, spec)
    conf = np.exp(np.clip(lp, -50, 0))
    prob = pd.to_numeric(df.get(spec.prob_col, pd.Series(np.nan, index=df.index)), errors="coerce").values
    prob_conf = np.exp(np.clip(prob, -50, 0))
    mask = ~np.isfinite(conf)
    conf[mask & np.isfinite(prob_conf)] = prob_conf[mask & np.isfinite(prob_conf)]
    return np.clip(conf, 0.0, 1.0)


def run_ece_calibration(args, costs):
    print("\n=== EXP-2 ECE/Brier calibration metrics ===")
    out_root = Path(args.output_dir) / "additional" / "ece_calibration"
    rows = []
    for pair in args.calibration_pairs:
        dataset, model = parse_pair(pair)
        df, cfg, specs, _ = load_dataset(args, dataset, model)
        train_df, cal_df, _, *_ = split_source(df, cfg, args.seed)
        artifact, cal_pred, raw_cal = get_or_train_artifact(
            args,
            "ece_calibration",
            dataset,
            model,
            f"full_tfidf{args.max_tfidf_features}",
            train_df,
            cal_df,
            specs,
            cfg,
            "full",
            args.max_tfidf_features,
        )[:3]
        pair_rows = []
        for spec in specs:
            y = cal_df[f"correct_{spec.fidelity}"].astype(int).values
            raw = raw_cal[f"p_{spec.fidelity}"].values
            iso = cal_pred[f"p_{spec.fidelity}"].values
            ece_raw, bins_raw = ece_score(y, raw, 15)
            ece_iso, bins_iso = ece_score(y, iso, 15)
            row = {
                "dataset": dataset,
                "model": model,
                "fidelity": spec.fidelity,
                "ece_uncalibrated": ece_raw,
                "ece_isotonic": ece_iso,
                "brier_uncalibrated": float(np.mean((np.clip(raw, 0, 1) - y) ** 2)),
                "brier_isotonic": float(np.mean((np.clip(iso, 0, 1) - y) ** 2)),
                "n_cal": int(len(y)),
                "bins_uncalibrated": bins_raw,
                "bins_isotonic": bins_iso,
            }
            rows.append({k: v for k, v in row.items() if not k.startswith("bins_")})
            pair_rows.append(row)
        write_json(out_root / dataset / model / "results.json", {"dataset": dataset, "model": model, "rows": pair_rows})
        make_reliability_plot(args, dataset, model, cal_df, specs, raw_cal, cal_pred)
    write_csv(Path(args.output_dir) / "additional" / "tables" / "ece_calibration_table.csv", rows)
    if rows:
        pd.DataFrame(rows).to_latex(
            Path(args.output_dir) / "additional" / "tables" / "ece_calibration_table.tex",
            index=False,
            float_format="%.4f",
        )


def make_reliability_plot(args, dataset, model, cal_df, specs, raw_cal, cal_pred):
    if plt is None:
        return
    plot_dir = ensure_plot_dir(args)
    selected_specs = [s for s in specs if s.fidelity in {"caption", "full"}]
    if not selected_specs:
        return
    fig, axes = plt.subplots(1, len(selected_specs), figsize=(5 * len(selected_specs), 4), squeeze=False)
    for ax, spec in zip(axes[0], selected_specs):
        y = cal_df[f"correct_{spec.fidelity}"].astype(int).values
        series = {
            "VLM log-prob": logprob_confidence(cal_df, spec),
            "GBR raw": raw_cal[f"p_{spec.fidelity}"].values,
            "VOILA isotonic": cal_pred[f"p_{spec.fidelity}"].values,
        }
        ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Perfect")
        markers = ["o", "s", "^"]
        for marker, (name, values) in zip(markers, series.items()):
            mask = np.isfinite(values)
            if len(np.unique(y[mask])) < 2 or mask.sum() < 10:
                continue
            prob_true, prob_pred = calibration_curve(y[mask], np.clip(values[mask], 0, 1), n_bins=10, strategy="quantile")
            ax.plot(prob_pred, prob_true, marker=marker, linewidth=2, label=name)
        ax.set_title(f"{dataset}/{model}: {spec.fidelity}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"reliability_{dataset}_{model}.pdf", bbox_inches="tight")
    fig.savefig(plot_dir / f"reliability_{dataset}_{model}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_extended_ood(args, costs):
    print("\n=== EXP-3 extended OOD transfer ===")
    out_root = Path(args.output_dir) / "additional" / "ood_extended"
    rows = []
    sources = [d for d in ["vqav2", "gqa"] if d in args.datasets]
    targets = [d for d in ["textvqa", "floodnet"] if d in args.datasets]
    for source in sources:
        for model in args.models:
            if model not in core.MODEL_CSV_MAP.get(source, {}):
                continue
            source_df, source_cfg, source_specs, _ = load_dataset(args, source, model)
            train_df, cal_df, _, *_ = split_source(source_df, source_cfg, args.seed)
            artifact, cal_pred, _, artifact_path = get_or_train_artifact(
                args,
                "ood_extended",
                source,
                model,
                f"full_tfidf{args.max_tfidf_features}",
                train_df,
                cal_df,
                source_specs,
                source_cfg,
                "full",
                args.max_tfidf_features,
            )
            for target in targets:
                if model not in core.MODEL_CSV_MAP.get(target, {}):
                    continue
                target_df, target_cfg, target_specs, _ = load_dataset(args, target, model)
                shared = [s for s in source_specs if any(t.fidelity == s.fidelity for t in target_specs)]
                target_pred, _ = predict_voila_artifact(artifact, target_df)
                selected, metrics = tune_and_eval(cal_df, cal_pred, target_df, target_pred, shared, target_specs, costs, args)
                in_dist_path = Path(args.output_dir) / "main" / target / model / "voila_results.json"
                in_dist_accuracy = None
                if in_dist_path.exists():
                    with open(in_dist_path, encoding="utf-8") as f:
                        in_dist_accuracy = json.load(f).get("accuracy_mean")
                payload = {
                    "source": source,
                    "target": target,
                    "model": model,
                    "model_artifact": artifact_path,
                    "n_eval": int(len(target_df)),
                    "in_distribution_voila_accuracy": in_dist_accuracy,
                    "ood_to_indist_accuracy_ratio": (
                        metrics["accuracy"] / in_dist_accuracy if in_dist_accuracy not in (None, 0) else None
                    ),
                    **metrics,
                }
                write_json(out_root / f"{source}_to_{target}" / model / "results.json", payload)
                rows.append(payload)
    write_csv(Path(args.output_dir) / "additional" / "tables" / "ood_extended_table.csv", rows)
    if rows:
        pd.DataFrame(rows).to_latex(
            Path(args.output_dir) / "additional" / "tables" / "ood_extended_table.tex",
            index=False,
            float_format="%.2f",
        )


def cheapest_oracle(row, specs, costs) -> Optional[str]:
    ordered = sorted(specs, key=lambda s: costs[s.fidelity])
    for spec in ordered:
        if bool(row.get(f"correct_{spec.fidelity}", False)):
            return spec.fidelity
    return None


def decompose_rows(df, selected, specs, costs):
    counts = Counter()
    details = []
    for (_, row), chosen in zip(df.iterrows(), selected):
        oracle = cheapest_oracle(row, specs, costs)
        chosen_correct = bool(row.get(f"correct_{chosen}", False))
        if oracle is None:
            category = "irreducible_error"
        elif chosen == oracle and chosen_correct:
            category = "correct_routing"
        elif chosen_correct and costs[chosen] > costs[oracle]:
            category = "over_escalation_error"
        elif chosen_correct:
            category = "equivalent_routing"
        else:
            category = "under_escalation_error"
        counts[category] += 1
        details.append({"selected": chosen, "oracle": oracle, "category": category})
    n = len(selected)
    return {
        "counts": dict(counts),
        "percentages": {k: v / n * 100.0 for k, v in sorted(counts.items())},
        "n": n,
    }, details


def run_oracle_decomp(args, costs):
    print("\n=== EXP-4 oracle gap decomposition ===")
    out_root = Path(args.output_dir) / "additional" / "oracle_decomp"
    rows = []
    for pair in args.oracle_decomp_pairs:
        dataset, model = parse_pair(pair)
        df, cfg, specs, _ = load_dataset(args, dataset, model)
        split_iter = list(core.make_outer_splits(df, cfg["qid"], args.n_cv_folds, args.seed))
        aggregate = Counter()
        n_total = 0
        fold_payloads = []
        for fold, (train_qids, test_qids) in enumerate(core.progress_iter(split_iter, desc=f"{dataset}/{model} oracle decomp", total=len(split_iter)), start=1):
            train_all = core.split_by_question_ids(df, cfg["qid"], train_qids)
            test_df = core.split_by_question_ids(df, cfg["qid"], test_qids)
            train_ids, cal_ids = train_test_split(pd.Series(train_all[cfg["qid"]].unique()), test_size=0.2, random_state=args.seed + fold)
            train_df = core.split_by_question_ids(train_all, cfg["qid"], train_ids)
            cal_df = core.split_by_question_ids(train_all, cfg["qid"], cal_ids)
            artifact, cal_pred, _ = get_or_train_artifact(
                args,
                "oracle_decomp",
                dataset,
                model,
                f"fold{fold}_full_tfidf{args.max_tfidf_features}",
                train_df,
                cal_df,
                specs,
                cfg,
                "full",
                args.max_tfidf_features,
            )[:3]
            test_pred, _ = predict_voila_artifact(artifact, test_df)
            selected, metrics = tune_and_eval(cal_df, cal_pred, test_df, test_pred, specs, specs, costs, args)
            breakdown, _ = decompose_rows(test_df, selected, specs, costs)
            aggregate.update(breakdown["counts"])
            n_total += breakdown["n"]
            fold_payloads.append({"fold": fold, "metrics": metrics, "breakdown": breakdown})
        percentages = {k: v / n_total * 100.0 for k, v in sorted(aggregate.items())}
        payload = {"dataset": dataset, "model": model, "n": n_total, "counts": dict(aggregate), "percentages": percentages, "folds": fold_payloads}
        write_json(out_root / dataset / model / "results.json", payload)
        for category, pct in percentages.items():
            rows.append({"dataset": dataset, "model": model, "category": category, "percentage": pct, "count": aggregate[category], "n": n_total})
    write_csv(Path(args.output_dir) / "additional" / "tables" / "oracle_decomp_table.csv", rows)


def simulate_cascade_paths(df, specs, costs, thresholds):
    ordered = sorted(specs, key=lambda s: costs[s.fidelity])
    selected = []
    levels_executed = []
    total_cost = []
    escalated = []
    for _, row in df.iterrows():
        spent = 0.0
        chosen = ordered[-1].fidelity
        n_levels = 0
        did_escalate = False
        for idx, spec in enumerate(ordered):
            spent += costs[spec.fidelity]
            n_levels += 1
            chosen = spec.fidelity
            if idx == len(ordered) - 1:
                break
            lp = core.logprob_scores(pd.DataFrame([row]), spec)[0]
            edge = f"{spec.fidelity}->{ordered[idx + 1].fidelity}"
            theta = thresholds.get(edge, np.inf)
            if np.isfinite(lp) and lp >= theta:
                break
            did_escalate = True
        selected.append(chosen)
        levels_executed.append(n_levels)
        total_cost.append(spent)
        escalated.append(did_escalate)
    selected = np.array(selected, dtype=object)
    acc = np.mean([bool(row.get(f"correct_{f}", False)) for (_, row), f in zip(df.iterrows(), selected)])
    return {
        "accuracy": float(acc * 100.0),
        "avg_cumulative_cost": float(np.mean(total_cost)),
        "avg_levels_executed": float(np.mean(levels_executed)),
        "escalation_rate": float(np.mean(escalated) * 100.0),
        "fidelity_distribution": {k: v / len(selected) for k, v in Counter(selected).items()},
    }


def run_cascade_degeneracy(args, costs):
    print("\n=== EXP-7 cascade degeneracy analysis ===")
    out_root = Path(args.output_dir) / "additional" / "cascade_degeneracy"
    rows = []
    for dataset in args.datasets:
        for model in args.models:
            if model not in core.MODEL_CSV_MAP.get(dataset, {}):
                continue
            df, cfg, specs, _ = load_dataset(args, dataset, model)
            split_iter = list(core.make_outer_splits(df, cfg["qid"], args.n_cv_folds, args.seed))
            fold_rows = []
            for fold, (train_qids, test_qids) in enumerate(core.progress_iter(split_iter, desc=f"{dataset}/{model} cascade degeneracy", total=len(split_iter)), start=1):
                train_df = core.split_by_question_ids(df, cfg["qid"], train_qids)
                test_df = core.split_by_question_ids(df, cfg["qid"], test_qids)
                thresholds, threshold_info = core.learn_pairwise_cascade_thresholds(train_df, specs, costs)
                metrics = simulate_cascade_paths(test_df, specs, costs, thresholds)
                fold_rows.append({"fold": fold, "thresholds": thresholds, "threshold_info": threshold_info, **metrics})
            summary = {
                "dataset": dataset,
                "model": model,
                "accuracy_mean": float(np.mean([r["accuracy"] for r in fold_rows])),
                "avg_cumulative_cost_mean": float(np.mean([r["avg_cumulative_cost"] for r in fold_rows])),
                "avg_levels_executed_mean": float(np.mean([r["avg_levels_executed"] for r in fold_rows])),
                "escalation_rate_mean": float(np.mean([r["escalation_rate"] for r in fold_rows])),
                "folds": fold_rows,
            }
            write_json(out_root / dataset / model / "results.json", summary)
            rows.append({k: v for k, v in summary.items() if k != "folds"})
    write_csv(Path(args.output_dir) / "additional" / "tables" / "cascade_degeneracy_table.csv", rows)


def question_type(question: str, dataset: Optional[str] = None, row: Optional[pd.Series] = None):
    if dataset == "floodnet" and row is not None and "question_type" in row and pd.notna(row["question_type"]):
        return str(row["question_type"]).lower().replace(" ", "_")
    q = "" if pd.isna(question) else str(question).lower()
    tokens = set(q.split())
    if q.startswith("how many") or "number" in tokens or "count" in tokens:
        return "counting"
    if q.startswith(("is ", "are ", "was ", "were ", "does ", "do ", "did ", "has ", "have ", "can ")):
        return "yes_no"
    if "color" in tokens or "colour" in tokens:
        return "color"
    if tokens & {"where", "left", "right", "behind", "front", "above", "below", "beside", "next"}:
        return "spatial"
    if tokens & {"read", "text", "written", "say", "says", "letter", "word", "sign", "label"}:
        return "ocr_text"
    return "other"


def run_fidelity_saturation(args):
    print("\n=== FIG-1 fidelity saturation curves ===")
    rows = []
    plot_dir = ensure_plot_dir(args)
    pairs = [("vqav2", "pixtral"), ("gqa", "pixtral")]
    for dataset, model in pairs:
        if dataset not in args.datasets or model not in core.MODEL_CSV_MAP.get(dataset, {}):
            continue
        df, cfg, specs, _ = load_dataset(args, dataset, model)
        qtypes = [question_type(q, dataset) for q in df[cfg["question"]].values]
        df = df.copy()
        df["_qtype"] = qtypes
        for qt, group in df.groupby("_qtype"):
            if len(group) < 20:
                continue
            final_acc = None
            accs = []
            for spec in specs:
                acc = float(group[f"correct_{spec.fidelity}"].mean() * 100.0)
                accs.append((spec.fidelity, acc))
                rows.append({"dataset": dataset, "model": model, "question_type": qt, "fidelity": spec.fidelity, "accuracy": acc, "n": len(group)})
            final_acc = accs[-1][1]
            plateau = next((f for f, a in accs if final_acc <= 0 or a >= 0.95 * final_acc), accs[-1][0])
            for row in rows[-len(accs):]:
                row["plateau_fidelity"] = plateau
        if plt is not None:
            fig, ax = plt.subplots(figsize=(7, 4))
            for qt, group in df.groupby("_qtype"):
                if len(group) < 20:
                    continue
                fids = [s.fidelity for s in specs]
                acc = [float(group[f"correct_{s.fidelity}"].mean() * 100.0) for s in specs]
                ax.plot(fids, acc, marker="o", linewidth=2, label=f"{qt} (n={len(group)})")
                final = acc[-1]
                for i, a in enumerate(acc):
                    if final <= 0 or a >= 0.95 * final:
                        ax.plot(fids[i], a, "D", markersize=7, color=ax.lines[-1].get_color())
                        break
            ax.set_title(f"{dataset}/{model}: fidelity saturation by question type")
            ax.set_xlabel("Fidelity")
            ax.set_ylabel("Accuracy (%)")
            ax.tick_params(axis="x", rotation=20)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(plot_dir / f"fidelity_saturation_{dataset}_{model}.pdf", bbox_inches="tight")
            fig.savefig(plot_dir / f"fidelity_saturation_{dataset}_{model}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
    write_csv(Path(args.output_dir) / "additional" / "tables" / "fidelity_saturation_table.csv", rows)


def run_roc_escalation(args, costs):
    print("\n=== FIG-2 ROC for escalation decision ===")
    if plt is None:
        print("  matplotlib unavailable; skipping ROC plots")
        return
    rows = []
    plot_dir = ensure_plot_dir(args)
    fig, ax = plt.subplots(figsize=(6, 5))
    for pair in args.roc_pairs:
        dataset, model = parse_pair(pair)
        df, cfg, specs, _ = load_dataset(args, dataset, model)
        train_df, cal_df, test_df, *_ = split_source(df, cfg, args.seed)
        artifact, cal_pred, _ = get_or_train_artifact(
            args,
            "roc_escalation",
            dataset,
            model,
            f"full_tfidf{args.max_tfidf_features}",
            train_df,
            cal_df,
            specs,
            cfg,
            "full",
            args.max_tfidf_features,
        )[:3]
        test_pred, _ = predict_voila_artifact(artifact, test_df)
        cap_spec = next((s for s in specs if s.fidelity == "caption"), specs[0])
        full_spec = next((s for s in specs if s.fidelity == "full"), specs[-1])
        labels = ((~test_df[f"correct_{cap_spec.fidelity}"].astype(bool)) & test_df[f"correct_{full_spec.fidelity}"].astype(bool)).astype(int).values
        lp = core.logprob_scores(test_df, cap_spec)
        logprob_uncertainty = -lp
        voila_uncertainty = 1.0 - test_pred[f"p_{cap_spec.fidelity}"].values
        for name, score, style in [
            ("Log-prob", logprob_uncertainty, "--"),
            ("VOILA", voila_uncertainty, "-"),
        ]:
            mask = np.isfinite(score)
            if len(np.unique(labels[mask])) < 2:
                continue
            fpr, tpr, _ = roc_curve(labels[mask], score[mask])
            auc_value = roc_auc_score(labels[mask], score[mask])
            ax.plot(fpr, tpr, linestyle=style, linewidth=2, label=f"{dataset}/{model} {name} AUC={auc_value:.2f}")
            rows.append({"dataset": dataset, "model": model, "score": name, "auc": auc_value, "n": int(mask.sum()), "positive_rate": float(labels[mask].mean())})
    ax.plot([0, 1], [0, 1], ":", color="black", label="Random AUC=0.50")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Caption escalation decision")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "roc_escalation_decision.pdf", bbox_inches="tight")
    fig.savefig(plot_dir / "roc_escalation_decision.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    write_csv(Path(args.output_dir) / "additional" / "tables" / "roc_escalation_table.csv", rows)


def run_routing_words(args):
    print("\n=== FIG-4 routing word importances ===")
    if plt is None:
        return
    dataset, model = "vqav2", "pixtral"
    df, cfg, specs, _ = load_dataset(args, dataset, model)
    train_df, cal_df, _, *_ = split_source(df, cfg, args.seed)
    artifact, _, _ = get_or_train_artifact(
        args,
        "routing_words",
        dataset,
        model,
        f"full_tfidf{args.max_tfidf_features}",
        train_df,
        cal_df,
        specs,
        cfg,
        "full",
        args.max_tfidf_features,
    )[:3]
    feature_names = artifact["feature_state"]["feature_names"]
    rows = []
    for fidelity in ["caption", "full"]:
        fitted = artifact["trained"].get(fidelity)
        model_obj = fitted["model"] if fitted else None
        if not hasattr(model_obj, "feature_importances_"):
            continue
        importances = np.asarray(model_obj.feature_importances_)
        tfidf_names = [n for n in feature_names if n not in core.STRUCT_FEATURE_NAMES]
        n_tfidf = len(tfidf_names)
        idx = np.argsort(importances[:n_tfidf])[::-1][:25]
        for rank, i in enumerate(idx, start=1):
            rows.append({"dataset": dataset, "model": model, "fidelity": fidelity, "rank": rank, "word": tfidf_names[i], "importance": float(importances[i])})
        top = pd.DataFrame([r for r in rows if r["fidelity"] == fidelity]).sort_values("importance")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top["word"], top["importance"], color=FIDELITY_COLORS.get(fidelity, "#333333"))
        ax.set_title(f"Top routing words for {fidelity}")
        ax.set_xlabel("GBR feature importance")
        fig.tight_layout()
        plot_dir = ensure_plot_dir(args)
        fig.savefig(plot_dir / f"routing_words_{fidelity}.pdf", bbox_inches="tight")
        fig.savefig(plot_dir / f"routing_words_{fidelity}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    write_csv(Path(args.output_dir) / "additional" / "tables" / "routing_words_table.csv", rows)


def run_iso_accuracy_contour(args):
    print("\n=== FIG-5 iso-accuracy contour heatmaps ===")
    if plt is None:
        return
    plot_dir = ensure_plot_dir(args)
    rows = []
    model_order = ["llava", "qwen7b", "pixtral", "llama4", "qwen72b", "qwen235b"]
    model_sizes = {"llava": 7, "qwen7b": 7, "pixtral": 12, "llama4": 17, "qwen72b": 72, "qwen235b": 235}
    for dataset in [d for d in ["vqav2", "gqa"] if d in args.datasets]:
        matrix = []
        labels = []
        for fidelity in core.FIDELITIES:
            row = []
            for model in model_order:
                path = Path(args.output_dir) / "main" / dataset / model / "voila_results.json"
                value = np.nan
                if path.exists():
                    data = json.load(open(path, encoding="utf-8"))
                    baseline = data.get("fixed_fidelity_baselines", {}).get(fidelity)
                    if baseline:
                        value = baseline.get("accuracy", np.nan)
                row.append(value)
                rows.append({"dataset": dataset, "model": model, "model_size_b": model_sizes[model], "fidelity": fidelity, "accuracy": value})
            matrix.append(row)
            labels.append(fidelity)
        mat = np.asarray(matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        if np.isfinite(mat).all():
            cs = ax.contour(mat, levels=8, colors="black", linewidths=0.5, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=7)
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels([f"{m}\n({model_sizes[m]}B)" for m in model_order], fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(f"{dataset}: accuracy across model size x fidelity")
        fig.colorbar(im, ax=ax, label="Accuracy (%)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"iso_accuracy_contour_{dataset}.pdf", bbox_inches="tight")
        fig.savefig(plot_dir / f"iso_accuracy_contour_{dataset}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    write_csv(Path(args.output_dir) / "additional" / "tables" / "iso_accuracy_contour_data.csv", rows)


def run_floodnet_analysis(args):
    print("\n=== EXP-5 FloodNet question-type analysis ===")
    rows = []
    for model in args.models:
        if model not in core.MODEL_CSV_MAP.get("floodnet", {}):
            continue
        df, cfg, specs, _ = load_dataset(args, "floodnet", model)
        if "question_type" not in df.columns:
            continue
        for qt, group in df.groupby("question_type"):
            for spec in specs:
                rows.append(
                    {
                        "dataset": "floodnet",
                        "model": model,
                        "question_type": qt,
                        "fidelity": spec.fidelity,
                        "accuracy": float(group[f"correct_{spec.fidelity}"].mean() * 100.0),
                        "n": int(len(group)),
                    }
                )
    write_csv(Path(args.output_dir) / "additional" / "tables" / "floodnet_question_type_accuracy.csv", rows)
    write_json(
        Path(args.output_dir) / "additional" / "floodnet_analysis" / "status.json",
        {
            "status": "partial",
            "what_was_done": "Per-question-type FloodNet accuracy by fidelity was computed from available CSV columns.",
            "not_done": "Internal VLM resolution caps and visual image inspection require model documentation/raw image files, so this script does not claim that evidence.",
            "table": "scripts/results_final/additional/tables/floodnet_question_type_accuracy.csv",
        },
    )


def collect_additional_tables(args):
    print("\n=== Additional table index ===")
    table_dir = Path(args.output_dir) / "additional" / "tables"
    rows = []
    if table_dir.exists():
        for path in sorted(table_dir.glob("*")):
            if path.is_file():
                rows.append({"table": path.name, "path": str(path), "bytes": path.stat().st_size})
    write_csv(table_dir / "additional_table_index.csv", rows)


def main():
    args = parse_args()
    args.data_dir = str(Path(args.data_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    args.model_artifact_dir = str(Path(args.model_artifact_dir).resolve())
    args.resolved_costs = resolve_costs(args)
    core.RECALL_MATCH_THRESHOLD = 70.0
    experiments = experiment_set(args)
    costs = args.resolved_costs
    print(f"Additional VOILA experiments | output={args.output_dir}")
    print(f"Experiments: {', '.join(experiments)}")

    if "feature_ood" in experiments:
        run_feature_ood(args, costs)
    if "tfidf_vocab" in experiments:
        run_tfidf_vocab(args, costs)
    if "ece_calibration" in experiments:
        run_ece_calibration(args, costs)
    if "extended_ood" in experiments:
        run_extended_ood(args, costs)
    if "oracle_decomp" in experiments:
        run_oracle_decomp(args, costs)
    if "floodnet_analysis" in experiments:
        run_floodnet_analysis(args)
    if "cascade_degeneracy" in experiments:
        run_cascade_degeneracy(args, costs)
    if "fidelity_saturation" in experiments:
        run_fidelity_saturation(args)
    if "roc_escalation" in experiments:
        run_roc_escalation(args, costs)
    if "reliability_diagram" in experiments and "ece_calibration" not in experiments:
        run_ece_calibration(args, costs)
    if "routing_words" in experiments:
        run_routing_words(args)
    if "iso_accuracy_contour" in experiments:
        run_iso_accuracy_contour(args)
    if "tables" in experiments:
        collect_additional_tables(args)
    print("\nDone.")


if __name__ == "__main__":
    main()
