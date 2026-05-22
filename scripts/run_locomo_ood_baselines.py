#!/usr/bin/env python3
"""
Run LoCoMo OOD baseline experiments.

This is a focused companion to run_all_experiments.py. It trains any learned
baseline on VQAv2+GQA and evaluates on LoCoMo for the two LoCoMo model pairs
available in the paper data: pixtral and qwen7b.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_all_experiments as exp  # noqa: E402


SOURCE_DATASETS = ["vqav2", "gqa"]
TARGET_DATASET = "locomo"
LOCOMO_MODELS = ["pixtral", "qwen7b"]


def write_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def write_table(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_combined_source(args, model: str):
    frames = []
    source_specs = None
    source_cfg = None
    source_paths = {}
    for source in SOURCE_DATASETS:
        df, cfg, specs, csv_path = exp.load_dataset(source, model, args.data_dir, args.max_rows, args.seed)
        df = df.copy()
        qid_col = cfg["qid"]
        df[qid_col] = source + ":" + df[qid_col].astype(str)
        df["source_dataset"] = source
        frames.append(df)
        source_paths[source] = str(csv_path)
        if source == "vqav2":
            source_specs = specs
            source_cfg = dict(cfg)
    if source_specs is None or source_cfg is None:
        raise RuntimeError("Could not load VQAv2 source specs")
    return pd.concat(frames, ignore_index=True, sort=False), source_cfg, source_specs, source_paths


def load_target(args, model: str):
    return exp.load_dataset(TARGET_DATASET, model, args.data_dir, args.max_rows, args.seed)


def run_fixed_and_oracle(args, model, target_df, target_specs, costs):
    fixed = exp.fixed_fidelity_baselines(target_df, target_specs, costs)
    oracle = exp.oracle_metrics(target_df, target_specs, costs)
    out = {
        "source": "none",
        "target": TARGET_DATASET,
        "model": model,
        "setting": "target_only_no_training",
        "costs": costs,
        "fixed_fidelity_baselines": fixed,
        **oracle,
    }
    write_json(Path(args.output_dir) / "baselines" / "locomo_ood" / "fixed_oracle" / model / "results.json", out)
    return out


def run_logprob_cascade_ood(args, model, source_df, source_specs, target_df, target_specs, costs):
    thresholds, threshold_info = exp.learn_pairwise_cascade_thresholds(source_df, source_specs, costs)
    metrics = exp.cascade_metrics(target_df, target_specs, costs, thresholds)
    out = {
        "source": "vqav2+gqa",
        "target": TARGET_DATASET,
        "model": model,
        "experiment": "locomo_ood_logprob_cascade",
        "costs": costs,
        "cost_accounting": "cumulative_sum_of_all_fidelities_reached",
        "max_possible_cascade_cost": float(sum(costs[s.fidelity] for s in target_specs)),
        "threshold_method": "pairwise_max_cdf_separation",
        "thresholds": thresholds,
        "threshold_info": threshold_info,
        "accuracy": metrics["accuracy"] * 100.0,
        "avg_cost": metrics["avg_cost"],
        "cost_reduction_pct": metrics["cost_reduction_pct"],
        "fidelity_distribution": metrics["fidelity_distribution"],
    }
    write_json(Path(args.output_dir) / "baselines" / "locomo_ood" / "logprob_cascade" / model / "results.json", out)
    return out


def fit_or_load_bert_router(args, model, source_df, source_cfg, source_specs, costs):
    artifact_path = exp.fold_artifact_path(
        args.model_artifact_dir,
        "vqav2_gqa_to_locomo",
        model,
        "locomo_ood_bert_router",
        0,
        "LogReg",
        "none",
        args.bert_router_features,
        args.max_tfidf_features,
        args.seed,
    )
    if (args.reuse_models or args.save_models) and artifact_path.exists():
        metadata, artifact = exp.load_model_artifact(artifact_path)
        print(f"    loaded router artifact {artifact_path}")
        return artifact, str(artifact_path)
    if args.reuse_models and args.strict_reuse_models:
        raise exp.missing_artifact_error(artifact_path)

    y_train = exp.cheapest_sufficient_labels(source_df, source_specs, costs)
    x_train, feature_artifact = exp.fit_router_features(
        source_df,
        source_cfg,
        args.bert_router_features,
        args.max_tfidf_features,
        args.bert_router_model,
        args.bert_router_allow_download,
    )
    label_encoder = exp.LabelEncoder()
    y_enc = label_encoder.fit_transform(y_train)
    if len(label_encoder.classes_) < 2:
        classifier = exp.ConstantModel(float(y_enc[0]))
    else:
        classifier = exp.LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
        classifier.fit(x_train, y_enc)

    artifact = {
        **feature_artifact,
        "classifier": classifier,
        "label_encoder": label_encoder,
        "classes": label_encoder.classes_.tolist(),
    }
    artifact_source = None
    if args.save_models:
        metadata = {
            "source": "vqav2+gqa",
            "target": TARGET_DATASET,
            "model": model,
            "experiment": "locomo_ood_bert_router",
            "feature_set": args.bert_router_features,
            "bert_model": args.bert_router_model,
            "seed": args.seed,
            "n_train": int(len(source_df)),
            "fidelities": [s.__dict__ for s in source_specs],
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        exp.save_model_artifact(artifact_path, artifact, metadata)
        artifact_source = str(artifact_path)
        print(f"    saved router artifact {artifact_path}")
    return artifact, artifact_source


def run_bert_router_ood(args, model, source_df, source_cfg, source_specs, target_df, target_cfg, target_specs, costs):
    try:
        artifact, artifact_source = fit_or_load_bert_router(args, model, source_df, source_cfg, source_specs, costs)
        x_target = exp.transform_router_features(target_df, target_cfg, artifact)
        pred_enc = artifact["classifier"].predict(x_target)
        selected = artifact["label_encoder"].inverse_transform(np.asarray(pred_enc, dtype=int))
        metrics = exp.evaluate_selection(target_df, selected, target_specs, costs)
        out = {
            "source": "vqav2+gqa",
            "target": TARGET_DATASET,
            "model": model,
            "experiment": "locomo_ood_bert_router",
            "router_kind": "supervised_fidelity_classifier",
            "feature_set": args.bert_router_features,
            "costs": costs,
            "accuracy": metrics["accuracy"] * 100.0,
            "avg_cost": metrics["avg_cost"],
            "cost_reduction_pct": metrics["cost_reduction_pct"],
            "fidelity_distribution": metrics["fidelity_distribution"],
            "model_artifact": artifact_source,
        }
    except Exception as exc:
        if args.bert_router_fail_on_error:
            raise
        out = {
            "source": "vqav2+gqa",
            "target": TARGET_DATASET,
            "model": model,
            "experiment": "locomo_ood_bert_router",
            "status": "skipped",
            "reason": str(exc),
            "feature_set": args.bert_router_features,
        }
        print(f"    skipped bert router: {exc}")
    write_json(Path(args.output_dir) / "baselines" / "locomo_ood" / "bert_router" / model / "results.json", out)
    return out


def run_llm_router_ood(args, model, target_df, target_cfg, target_specs, costs):
    existing_path = Path(args.output_dir) / "baselines" / "llm_router" / TARGET_DATASET / model / "results.json"
    if args.reuse_existing_llm_router and existing_path.exists():
        with open(existing_path, encoding="utf-8") as f:
            payload = json.load(f)
        out = {
            **payload,
            "source": "zero_shot_prompt_only",
            "target": TARGET_DATASET,
            "model": model,
            "experiment": "locomo_ood_llm_router",
            "reused_from": str(existing_path),
        }
        write_json(Path(args.output_dir) / "baselines" / "locomo_ood" / "llm_router" / model / "results.json", out)
        return out

    if args.llm_router_backend == "azure_gpt4o":
        payload = exp.run_llm_router_api(
            target_df,
            target_cfg,
            target_specs,
            costs,
            args.llm_router_routing,
            args.n_llm_router_samples,
            args.seed,
            TARGET_DATASET,
            model,
            args,
        )
    else:
        payload = exp.run_llm_router_surrogate(
            target_df,
            target_cfg,
            target_specs,
            costs,
            args.llm_router_routing,
            args.n_llm_router_samples,
            args.seed,
        )
    out = {
        "source": "zero_shot_prompt_only",
        "target": TARGET_DATASET,
        "model": model,
        "experiment": "locomo_ood_llm_router",
        "costs": costs,
        **payload,
    }
    write_json(Path(args.output_dir) / "baselines" / "locomo_ood" / "llm_router" / model / "results.json", out)
    return out


def summarize(output_dir: Path, rows: List[Dict[str, Any]]):
    table_rows = []
    for row in rows:
        baseline = row["baseline"]
        model = row["model"]
        data = row["result"]
        if baseline == "fixed_oracle":
            for fidelity, metrics in data.get("fixed_fidelity_baselines", {}).items():
                table_rows.append(
                    {
                        "baseline": f"fixed_{fidelity}",
                        "model": model,
                        "accuracy": metrics.get("accuracy"),
                        "avg_cost": metrics.get("avg_cost"),
                        "cost_reduction_pct": metrics.get("cost_reduction_pct"),
                    }
                )
            table_rows.append(
                {
                    "baseline": "oracle",
                    "model": model,
                    "accuracy": data.get("oracle_accuracy"),
                    "avg_cost": data.get("oracle_avg_cost"),
                    "cost_reduction_pct": None,
                }
            )
        else:
            table_rows.append(
                {
                    "baseline": baseline,
                    "model": model,
                    "accuracy": data.get("accuracy"),
                    "avg_cost": data.get("avg_cost"),
                    "cost_reduction_pct": data.get("cost_reduction_pct"),
                    "status": data.get("status", "ok"),
                }
            )
    write_table(output_dir / "tables" / "locomo_ood_baselines_table.csv", table_rows)


def parse_args():
    p = argparse.ArgumentParser(description="Run VQAv2+GQA -> LoCoMo OOD baselines.")
    p.add_argument("--models", nargs="+", default=LOCOMO_MODELS, choices=LOCOMO_MODELS)
    p.add_argument("--baselines", nargs="+", default=["fixed_oracle", "logprob_cascade", "llm_router", "bert_router"])
    p.add_argument("--data_dir", default=str(exp.DEFAULT_DATA_DIR))
    p.add_argument("--output_dir", default=str(ROOT / "scripts" / "results_final"))
    p.add_argument("--model_artifact_dir", default=str(ROOT / "model_artifacts"))
    p.add_argument("--max_rows", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recall_threshold", type=int, default=70)
    p.add_argument("--max_tfidf_features", type=int, default=5000)
    p.add_argument("--save_models", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reuse_models", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict_reuse_models", action="store_true")

    p.add_argument("--full_cost", type=float, default=100.0)
    p.add_argument("--cost", action="append", default=[], help="Override fidelity cost, e.g. caption=8.")

    p.add_argument("--llm_router_routing", choices=["binary", "ternary"], default="ternary")
    p.add_argument("--llm_router_backend", choices=["heuristic", "azure_gpt4o"], default="azure_gpt4o")
    p.add_argument("--reuse_existing_llm_router", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n_llm_router_samples", type=int, default=0, help="0 means all LoCoMo rows.")
    p.add_argument("--llm_router_cache_dir", default=str(ROOT / "results" / "cache" / "llm_router"))
    p.add_argument("--llm_router_timeout", type=float, default=60.0)
    p.add_argument("--llm_router_sleep", type=float, default=0.0)
    p.add_argument("--llm_router_retries", type=int, default=5)
    p.add_argument("--llm_router_max_completion_tokens", type=int, default=48)
    p.add_argument("--llm_router_api_key")
    p.add_argument("--llm_router_api_key_file", default=str(exp.DEFAULT_AZURE_OPENAI_API_KEY_FILE))
    p.add_argument("--llm_router_allow_fallback", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--bert_router_features", choices=["bert", "tfidf+struct"], default="bert")
    p.add_argument("--bert_router_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--bert_router_allow_download", action="store_true")
    p.add_argument("--bert_router_fail_on_error", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    exp.RECALL_MATCH_THRESHOLD = args.recall_threshold
    args.resolved_costs = exp.resolve_costs_from_args(args)
    args.llm_router_cache_dir = str(Path(args.llm_router_cache_dir).resolve())
    costs = exp.get_costs(args)
    rows = []

    for model in exp.progress_iter(args.models, desc="LoCoMo OOD models", total=len(args.models)):
        print(f"\n=== LoCoMo OOD baselines: {model} ===")
        source_df, source_cfg, source_specs, source_paths = load_combined_source(args, model)
        target_df, target_cfg, target_specs, target_path = load_target(args, model)
        print(f"  sources={source_paths}")
        print(f"  target={target_path}")

        if "fixed_oracle" in args.baselines:
            result = run_fixed_and_oracle(args, model, target_df, target_specs, costs)
            rows.append({"baseline": "fixed_oracle", "model": model, "result": result})
        if "logprob_cascade" in args.baselines:
            result = run_logprob_cascade_ood(args, model, source_df, source_specs, target_df, target_specs, costs)
            rows.append({"baseline": "logprob_cascade", "model": model, "result": result})
        if "llm_router" in args.baselines:
            result = run_llm_router_ood(args, model, target_df, target_cfg, target_specs, costs)
            rows.append({"baseline": "llm_router", "model": model, "result": result})
        if "bert_router" in args.baselines:
            result = run_bert_router_ood(
                args,
                model,
                source_df,
                source_cfg,
                source_specs,
                target_df,
                target_cfg,
                target_specs,
                costs,
            )
            rows.append({"baseline": "bert_router", "model": model, "result": result})

    summarize(Path(args.output_dir), rows)
    print(f"\nWrote LoCoMo OOD baseline outputs under {Path(args.output_dir) / 'baselines' / 'locomo_ood'}")
    print(f"Summary table: {Path(args.output_dir) / 'tables' / 'locomo_ood_baselines_table.csv'}")


if __name__ == "__main__":
    main()
