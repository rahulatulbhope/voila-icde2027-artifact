#!/usr/bin/env python3
"""
Rescale existing VOILA result JSON files from one normalized full-resolution
cost scale to another.

This is intended for unit changes such as full=120 -> full=100. It does not
rerun models or change selected fidelities; it only rescales reported costs and
equivalent lambda values.

Example:
    python scripts/rescale_cost_results.py --src results --dst results_cost100 \
        --old_full_cost 120 --new_full_cost 100
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


COST_KEYS = {
    "avg_cost",
    "avg_cost_mean",
    "avg_cost_std",
    "validation_avg_cost",
    "voila_avg_cost",
    "voila_cost",
    "oracle_avg_cost",
    "always_caption_cost",
    "always_resize_32x32_cost",
    "always_jpeg_q1_cost",
    "always_jpeg_q10_cost",
    "always_full_cost",
    "incremental_cost",
    "full_cost",
}

LAMBDA_KEYS = {"lambda", "best_lambda", "lambda_cost"}


def transform(obj, cost_scale, lambda_scale, key=None):
    if isinstance(obj, dict):
        if key in {"costs", "cost_table"}:
            return {
                k: (v * cost_scale if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                for k, v in obj.items()
            }
        return {
            k: transform(v, cost_scale, lambda_scale, k)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [transform(v, cost_scale, lambda_scale, key) for v in obj]
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if key in COST_KEYS or (isinstance(key, str) and key.endswith("_cost")):
            return obj * cost_scale
        if key in LAMBDA_KEYS:
            return obj * lambda_scale
    return obj


def annotate(data, old_full_cost, new_full_cost):
    if isinstance(data, dict):
        data.setdefault("cost_rescale", {})
        data["cost_rescale"].update(
            {
                "old_full_cost": old_full_cost,
                "new_full_cost": new_full_cost,
                "cost_scale": new_full_cost / old_full_cost,
                "lambda_scale": old_full_cost / new_full_cost,
                "note": "Uniform unit conversion only; selected fidelities and accuracies are unchanged.",
            }
        )
    return data


def main():
    parser = argparse.ArgumentParser(description="Rescale VOILA result JSON cost units.")
    parser.add_argument("--src", default="results")
    parser.add_argument("--dst", default="results_cost100")
    parser.add_argument("--old_full_cost", type=float, default=120.0)
    parser.add_argument("--new_full_cost", type=float, default=100.0)
    parser.add_argument("--copy_non_json", action="store_true")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        raise SystemExit(f"Source folder does not exist: {src}")
    if src == dst:
        raise SystemExit("Refusing to write in-place. Choose a different --dst.")

    cost_scale = args.new_full_cost / args.old_full_cost
    lambda_scale = args.old_full_cost / args.new_full_cost

    for path in src.rglob("*"):
        rel = path.relative_to(src)
        out_path = dst / rel
        if path.is_dir():
            out_path.mkdir(parents=True, exist_ok=True)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data = transform(data, cost_scale, lambda_scale)
            data = annotate(data, args.old_full_cost, args.new_full_cost)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, allow_nan=True)
        elif args.copy_non_json:
            shutil.copy2(path, out_path)

    print(f"Rescaled JSON results written to {dst}")
    print(f"Cost scale={cost_scale:.6g}; lambda scale={lambda_scale:.6g}")


if __name__ == "__main__":
    main()
