#!/usr/bin/env python3
"""PerCom-specific offline systems experiments for VOILA.

This module never invokes a VLM.  It measures cached/derived representation
payloads, solves mission-level multiple-choice knapsack instances, benchmarks
saved router artifacts, and replays packet-opportunity traces for deadline
analysis.  All outputs are deterministic for a fixed seed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import statistics
import sys
import time
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIDELITIES = ["caption", "resize_32x32", "jpeg_q1", "jpeg_q10", "full"]


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def encode_image_payload(image, fidelity):
    """Return the exact encoded payload used by the PerCom transfer harness."""
    buffer = BytesIO()
    if fidelity == "resize_32x32":
        image.resize((32, 32)).save(buffer, format="PNG", optimize=False)
    elif fidelity in {"jpeg_q1", "jpeg_q10"}:
        quality = 1 if fidelity == "jpeg_q1" else 10
        image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=False)
    else:
        raise ValueError(f"Unsupported derived image fidelity: {fidelity}")
    return buffer.getvalue()


def index_images(image_dir):
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    index = {}
    for path in Path(image_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in supported:
            index.setdefault(path.stem, path)
            index.setdefault(path.name, path)
    return index


def resolve_image_path(image_index, image_id):
    candidates = [str(image_id), Path(str(image_id)).stem, Path(str(image_id)).name]
    for candidate in candidates:
        if candidate in image_index:
            return image_index[candidate]
    numeric = str(image_id).split(".")[0]
    for width in (4, 5, 6, 8):
        candidate = numeric.zfill(width)
        if candidate in image_index:
            return image_index[candidate]
    return None


def measure_representation_sizes(
    image_dir, cached_csv, output_csv, caption_col="caption1_question_based", payload_dir=None
):
    cached = pd.read_csv(cached_csv, low_memory=False)
    required = {"image_id", caption_col}
    missing = required.difference(cached.columns)
    if missing:
        raise ValueError(f"Cached CSV is missing columns: {sorted(missing)}")
    per_image = cached[["image_id", caption_col]].drop_duplicates("image_id")
    image_index = index_images(image_dir)
    rows = []
    missing_ids = []
    payload_root = Path(payload_dir) if payload_dir else None
    if payload_root:
        payload_root.mkdir(parents=True, exist_ok=True)
    for record in per_image.itertuples(index=False):
        image_id = str(record[0])
        caption = "" if pd.isna(record[1]) else str(record[1])
        source = resolve_image_path(image_index, image_id)
        if source is None:
            missing_ids.append(image_id)
            continue
        safe_id = "".join(character if character.isalnum() or character in "-_." else "_" for character in image_id)
        caption_payload = caption.encode("utf-8")
        caption_path = None
        if payload_root:
            caption_path = payload_root / "caption" / f"{safe_id}.txt"
            caption_path.parent.mkdir(parents=True, exist_ok=True)
            caption_path.write_bytes(caption_payload)
        rows.append(
            {
                "image_id": image_id,
                "fidelity": "caption",
                "bytes": len(caption_payload),
                "payload_path": None if caption_path is None else str(caption_path.resolve()),
                "encoding": "utf-8 cached at ingestion",
            }
        )
        rows.append(
            {
                "image_id": image_id,
                "fidelity": "full",
                "bytes": source.stat().st_size,
                "payload_path": str(source.resolve()),
                "encoding": source.suffix.lower().lstrip("."),
            }
        )
        with Image.open(source) as image:
            image.load()
            for fidelity in ("resize_32x32", "jpeg_q1", "jpeg_q10"):
                payload = encode_image_payload(image, fidelity)
                payload_path = None
                if payload_root:
                    suffix = ".png" if fidelity == "resize_32x32" else ".jpg"
                    payload_path = payload_root / fidelity / f"{safe_id}{suffix}"
                    payload_path.parent.mkdir(parents=True, exist_ok=True)
                    payload_path.write_bytes(payload)
                rows.append(
                    {
                        "image_id": image_id,
                        "fidelity": fidelity,
                        "bytes": len(payload),
                        "payload_path": None if payload_path is None else str(payload_path.resolve()),
                        "encoding": "png" if fidelity == "resize_32x32" else fidelity,
                    }
                )
    frame = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    summary = []
    if not frame.empty:
        for fidelity, group in frame.groupby("fidelity"):
            summary.append(
                {
                    "fidelity": fidelity,
                    "n": int(len(group)),
                    "median_bytes": float(group["bytes"].median()),
                    "p95_bytes": float(group["bytes"].quantile(0.95)),
                    "mean_bytes": float(group["bytes"].mean()),
                }
            )
    metadata = {
        "source_csv": str(Path(cached_csv).resolve()),
        "source_sha256": sha256_file(cached_csv),
        "image_dir": str(Path(image_dir).resolve()),
        "payload_dir": None if payload_root is None else str(payload_root.resolve()),
        "n_cached_images": int(len(per_image)),
        "n_measured_images": int(frame["image_id"].nunique()) if not frame.empty else 0,
        "missing_image_ids": missing_ids,
        "thumbnail_encoding": "32x32 PNG; VLM preprocessing previously used the same 32x32 pixels but not a network encoding",
        "caption_assumption": "generic caption generated once and cached during ingestion; generation cost excluded",
        "summary": summary,
    }
    write_json(output_csv.with_suffix(".summary.json"), metadata)
    return frame, metadata


def build_common_intersection(input_csvs, output_csv, key_cols=("image_id", "question")):
    """Materialize the common sample population for cross-model aggregation."""
    key_frames = []
    sources = []
    duplicate_key_frames = []
    for source in input_csvs:
        frame = pd.read_csv(source, usecols=list(key_cols), low_memory=False)
        for key in key_cols:
            frame[key] = frame[key].astype(str)
        duplicate_rows = int(frame.duplicated(list(key_cols), keep=False).sum())
        duplicated_keys = frame.loc[frame.duplicated(list(key_cols), keep=False), list(key_cols)].drop_duplicates()
        if not duplicated_keys.empty:
            duplicate_key_frames.append(duplicated_keys)
        unique = frame.drop_duplicates(list(key_cols))
        key_frames.append(unique)
        sources.append(
            {
                "path": str(Path(source).resolve()),
                "sha256": sha256_file(source),
                "rows": int(len(frame)),
                "unique_keys": int(len(unique)),
                "duplicate_rows": duplicate_rows,
            }
        )
    if not key_frames:
        raise ValueError("At least one input CSV is required")
    common = key_frames[0]
    for frame in key_frames[1:]:
        common = common.merge(frame, on=list(key_cols), how="inner", validate="one_to_one")
    if duplicate_key_frames:
        excluded = pd.concat(duplicate_key_frames, ignore_index=True).drop_duplicates(list(key_cols))
        common = common.merge(excluded.assign(_drop_duplicate_key=True), on=list(key_cols), how="left")
        common = common[common["_drop_duplicate_key"].isna()].drop(columns=["_drop_duplicate_key"])
    else:
        excluded = pd.DataFrame(columns=list(key_cols))
    common = common.sort_values(list(key_cols)).reset_index(drop=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    common.to_csv(output_csv, index=False)
    metadata = {
        "key_columns": list(key_cols),
        "common_sample_count": int(len(common)),
        "excluded_duplicate_key_count": int(len(excluded)),
        "excluded_duplicate_keys": excluded.sort_values(list(key_cols)).to_dict(orient="records"),
        "sources": sources,
    }
    write_json(output_csv.with_suffix(".summary.json"), metadata)
    return common, metadata


def enrich_query_records_with_sizes(query_records_csv, sizes_csv, output_csv):
    """Attach actual per-fidelity and selected payload bytes to every query."""
    records = pd.read_csv(query_records_csv, low_memory=False)
    sizes = pd.read_csv(sizes_csv)
    sizes["image_id"] = sizes["image_id"].astype(str)
    records["image_id"] = records["image_id"].astype(str)
    wide = sizes.pivot(index="image_id", columns="fidelity", values="bytes")
    wide = wide.rename(columns={fidelity: f"measured_bytes_{fidelity}" for fidelity in wide.columns})
    enriched = records.join(wide, on="image_id")
    measured = []
    for row in enriched.itertuples(index=False):
        fidelity = str(getattr(row, "selected_fidelity"))
        column = f"measured_bytes_{fidelity}"
        measured.append(getattr(row, column, np.nan))
    enriched["selected_measured_bytes"] = measured
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_csv, index=False)
    metadata = {
        "query_records": str(Path(query_records_csv).resolve()),
        "query_records_sha256": sha256_file(query_records_csv),
        "sizes": str(Path(sizes_csv).resolve()),
        "sizes_sha256": sha256_file(sizes_csv),
        "rows": int(len(enriched)),
        "rows_missing_selected_bytes": int(enriched["selected_measured_bytes"].isna().sum()),
    }
    write_json(output_csv.with_suffix(".summary.json"), metadata)
    return enriched, metadata


def build_transfer_manifest(query_records_csv, sizes_csv, output_csv, policy="exact_argmax_voila"):
    """Resolve each selected representation to the exact payload sent by TCP."""
    queries = pd.read_csv(query_records_csv, low_memory=False)
    sizes = pd.read_csv(sizes_csv, low_memory=False)
    for frame in (queries, sizes):
        frame["image_id"] = frame["image_id"].astype(str)
    selected = queries.rename(columns={"selected_fidelity": "fidelity"})
    columns = [
        column
        for column in ("question_id", "image_id", "question", "fidelity", "selected_correct")
        if column in selected
    ]
    manifest = selected[columns].merge(
        sizes[["image_id", "fidelity", "bytes", "payload_path"]],
        on=["image_id", "fidelity"],
        how="left",
        validate="many_to_one",
    )
    manifest["selected_fidelity"] = manifest["fidelity"]
    manifest["policy"] = policy
    if manifest["payload_path"].isna().any():
        missing = manifest.loc[manifest["payload_path"].isna(), ["image_id", "fidelity"]]
        raise ValueError(f"Missing exact payload files for {len(missing)} selected records")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)
    metadata = {
        "query_records_sha256": sha256_file(query_records_csv),
        "sizes_sha256": sha256_file(sizes_csv),
        "policy": policy,
        "rows": int(len(manifest)),
        "payload_bytes": int(manifest["bytes"].sum()),
    }
    write_json(output_csv.with_suffix(".summary.json"), metadata)
    return manifest, metadata


def summarize_deadline_transfers(
    transfers_jsonl, manifest_csv, output_json, deadlines=(0.5, 1.0, 2.0, 5.0, 10.0)
):
    transfer_rows = []
    with open(transfers_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                transfer_rows.append(json.loads(line))
    transfers = pd.DataFrame(transfer_rows)
    manifest = pd.read_csv(manifest_csv, low_memory=False)
    if "selected_fidelity" not in transfers and "fidelity" in transfers:
        transfers["selected_fidelity"] = transfers["fidelity"]
    keys = [key for key in ("question_id", "image_id", "selected_fidelity") if key in transfers and key in manifest]
    if not keys:
        raise ValueError("Transfer and manifest files have no common sample key")
    for key in keys:
        transfers[key] = transfers[key].astype(str)
        manifest[key] = manifest[key].astype(str)
    truth_columns = keys + [column for column in ("selected_correct", "policy") if column in manifest]
    merged = transfers.merge(manifest[truth_columns], on=keys, how="left", validate="many_to_one")
    if merged["selected_correct"].isna().any():
        raise ValueError("Some transfer records have no cached correctness match")
    if merged["selected_correct"].dtype != bool:
        merged["selected_correct"] = merged["selected_correct"].map(
            lambda value: str(value).strip().lower() in {"1", "true", "yes"}
        )
    group_columns = [column for column in ("trace", "loss_rate", "policy") if column in merged]
    rows = []
    for group_values, group in merged.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        identifiers = dict(zip(group_columns, group_values))
        for deadline in deadlines:
            completed = group["completion_s"] <= deadline
            completion_rate = float(completed.mean())
            conditional_accuracy = (
                float(group.loc[completed, "selected_correct"].mean()) if completed.any() else None
            )
            effective_accuracy = float((completed & group["selected_correct"]).mean())
            rows.append(
                {
                    **identifiers,
                    "deadline_s": float(deadline),
                    "n": int(len(group)),
                    "completion_rate": completion_rate,
                    "conditional_accuracy": conditional_accuracy,
                    "effective_accuracy_miss_is_failure": effective_accuracy,
                    "abstention_rate": 1.0 - completion_rate,
                    "mean_payload_bytes": float(group["payload_bytes"].mean()),
                    "mean_tcp_retransmissions": float(group["tcp_retransmissions"].mean()),
                }
            )
    payload = {
        "transfers_jsonl": str(Path(transfers_jsonl).resolve()),
        "transfers_sha256": sha256_file(transfers_jsonl),
        "manifest_csv": str(Path(manifest_csv).resolve()),
        "manifest_sha256": sha256_file(manifest_csv),
        "transport": "reliable TCP inside CellReplay",
        "rows": rows,
    }
    write_json(output_json, payload)
    return merged, payload


def mckp_dynamic_program(values, costs, budget, bucket_bytes=1024):
    """Exact sparse-Pareto MCKP over 1 KiB-style bucketed costs."""
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if values.shape != costs.shape or values.ndim != 2:
        raise ValueError("values and costs must be equally-shaped 2-D arrays")
    if bucket_bytes <= 0 or budget < 0 or np.any(costs < 0):
        raise ValueError("Costs, budget, and bucket size must be non-negative")
    weights = np.maximum(1, np.ceil(costs / bucket_bytes).astype(int))
    capacity = int(math.floor(budget / bucket_bytes))
    if int(weights.min(axis=1).sum()) > capacity:
        raise ValueError("Budget is infeasible: it cannot fund one fidelity per query")
    n_groups, n_items = values.shape
    states = {0: 0.0}
    history = []
    for group_idx in range(n_groups):
        candidates = {}
        for previous_weight, previous_value in states.items():
            for item_idx in range(n_items):
                destination = previous_weight + int(weights[group_idx, item_idx])
                if destination > capacity:
                    continue
                candidate_value = previous_value + float(values[group_idx, item_idx])
                incumbent = candidates.get(destination)
                if incumbent is None or candidate_value > incumbent[0] + 1e-12 or (
                    math.isclose(candidate_value, incumbent[0], abs_tol=1e-12)
                    and item_idx < incumbent[2]
                ):
                    candidates[destination] = (candidate_value, previous_weight, item_idx)
        if not candidates:
            raise ValueError("Budget is infeasible")
        retained = {}
        parents = {}
        best_value_so_far = -np.inf
        for destination in sorted(candidates):
            candidate_value, previous_weight, item_idx = candidates[destination]
            if candidate_value > best_value_so_far + 1e-12:
                retained[destination] = candidate_value
                parents[destination] = (previous_weight, item_idx)
                best_value_so_far = candidate_value
        states = retained
        history.append(parents)
    best_capacity = min(
        states,
        key=lambda state: (-states[state], state),
    )
    choices = np.empty(n_groups, dtype=int)
    cursor = best_capacity
    for group_idx in range(n_groups - 1, -1, -1):
        previous, item_idx = history[group_idx][cursor]
        choices[group_idx] = item_idx
        cursor = previous
    actual_cost = float(costs[np.arange(n_groups), choices].sum())
    if actual_cost > budget + 1e-9:
        raise AssertionError("Conservative bucket rounding violated the requested budget")
    return {
        "choices": choices,
        "value": float(values[np.arange(n_groups), choices].sum()),
        "actual_cost": actual_cost,
        "bucketed_cost": int(weights[np.arange(n_groups), choices].sum()) * bucket_bytes,
        "budget": float(budget),
    }


def mckp_bruteforce(values, costs, budget):
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)
    best = None
    for choices in itertools.product(range(values.shape[1]), repeat=values.shape[0]):
        idx = np.asarray(choices, dtype=int)
        cost = float(costs[np.arange(len(idx)), idx].sum())
        value = float(values[np.arange(len(idx)), idx].sum())
        if cost <= budget and (best is None or value > best[0] or (value == best[0] and cost < best[1])):
            best = (value, cost, idx)
    if best is None:
        raise ValueError("Budget is infeasible")
    return {"value": best[0], "actual_cost": best[1], "choices": best[2]}


def greedy_upgrades(values, costs, budget):
    """Start from each query's cheapest item, then take gain/byte upgrades."""
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)
    choices = np.argmin(costs, axis=1)
    spent = float(costs[np.arange(len(choices)), choices].sum())
    if spent > budget:
        raise ValueError("Budget is infeasible")
    upgrades = []
    for group_idx in range(len(choices)):
        base = choices[group_idx]
        for item_idx in range(values.shape[1]):
            delta_cost = costs[group_idx, item_idx] - costs[group_idx, base]
            delta_value = values[group_idx, item_idx] - values[group_idx, base]
            if delta_cost > 0 and delta_value > 0:
                upgrades.append((delta_value / delta_cost, delta_value, delta_cost, group_idx, item_idx))
    for _, _, delta_cost, group_idx, item_idx in sorted(upgrades, reverse=True):
        current = choices[group_idx]
        replacement_cost = costs[group_idx, item_idx] - costs[group_idx, current]
        replacement_gain = values[group_idx, item_idx] - values[group_idx, current]
        if replacement_gain > 0 and replacement_cost > 0 and spent + replacement_cost <= budget:
            choices[group_idx] = item_idx
            spent += replacement_cost
    return {
        "choices": choices,
        "value": float(values[np.arange(len(choices)), choices].sum()),
        "actual_cost": float(costs[np.arange(len(choices)), choices].sum()),
        "budget": float(budget),
    }


def load_mission_arrays(query_records_csv, sizes_csv, legacy_query_records_csv=None):
    records = pd.read_csv(query_records_csv, low_memory=False)
    sizes = pd.read_csv(sizes_csv)
    size_wide = sizes.pivot(index="image_id", columns="fidelity", values="bytes")
    records["image_id"] = records["image_id"].astype(str)
    size_wide.index = size_wide.index.astype(str)
    records = records.join(size_wide, on="image_id", rsuffix="_bytes")
    present = [f for f in FIDELITIES if f"p_{f}" in records and f"correct_{f}" in records and f in records]
    if not present:
        raise ValueError("No common probability/correctness/byte fidelity columns")
    required_value_columns = present + [f"p_{f}" for f in present] + [f"correct_{f}" for f in present]
    records = records.dropna(subset=required_value_columns).reset_index(drop=True)
    if legacy_query_records_csv:
        legacy = pd.read_csv(legacy_query_records_csv, low_memory=False)
        keys = [key for key in ("question_id", "image_id", "question") if key in records and key in legacy]
        if not keys:
            raise ValueError("Legacy query records have no stable join key")
        for key in keys:
            records[key] = records[key].astype(str)
            legacy[key] = legacy[key].astype(str)
        legacy = legacy[keys + ["selected_fidelity"]].rename(
            columns={"selected_fidelity": "legacy_selected_fidelity"}
        )
        if legacy.duplicated(keys).any():
            raise ValueError("Legacy query records contain duplicate join keys")
        records = records.merge(legacy, on=keys, how="left", validate="one_to_one")
    probabilities = records[[f"p_{f}" for f in present]].to_numpy(dtype=float)
    correctness = records[[f"correct_{f}" for f in present]].astype(int).to_numpy(dtype=float)
    costs = records[present].to_numpy(dtype=float)
    return records, present, probabilities, correctness, costs


def run_mission_budget_experiment(
    query_records_csv,
    sizes_csv,
    output_json,
    batch_sizes=(25, 50, 100),
    budget_fractions=(0.10, 0.25, 0.50, 0.75, 1.0),
    repeats=100,
    seed=42,
    bucket_bytes=1024,
    legacy_query_records_csv=None,
):
    records, fidelities, probabilities, correctness, byte_costs = load_mission_arrays(
        query_records_csv, sizes_csv, legacy_query_records_csv
    )
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    detail_csv = output_json.with_suffix(".csv")
    rows = []
    existing_keys = set()
    if detail_csv.exists() and detail_csv.stat().st_size > 0:
        existing = pd.read_csv(detail_csv)
        required = {"batch_size", "repeat", "budget_fraction"}
        if required.issubset(existing.columns):
            existing_keys = {
                (int(row.batch_size), int(row.repeat), float(row.budget_fraction))
                for row in existing.itertuples(index=False)
            }
    for batch_size in batch_sizes:
        if batch_size > len(records):
            continue
        for repeat in range(repeats):
            batch_seed = int(seed + batch_size * 1_000_003 + repeat)
            rng = np.random.default_rng(batch_seed)
            chosen_rows = rng.choice(len(records), size=batch_size, replace=False)
            p = probabilities[chosen_rows]
            y = correctness[chosen_rows]
            c = byte_costs[chosen_rows]
            all_full = float(c[:, fidelities.index("full")].sum()) if "full" in fidelities else float(c.max(axis=1).sum())
            for fraction in budget_fractions:
                setting_key = (int(batch_size), int(repeat), float(fraction))
                if setting_key in existing_keys:
                    continue
                requested_budget = all_full * fraction
                minimum_budget = float(c.min(axis=1).sum())
                if minimum_budget > requested_budget:
                    raise ValueError(
                        f"Requested {fraction:.0%} budget is infeasible for batch size {batch_size}"
                    )
                budget = requested_budget
                setting_rows = []
                solutions = {}
                for policy, values, solver in (
                    ("budgeted_voila", p, mckp_dynamic_program),
                    ("cached_oracle", y, mckp_dynamic_program),
                    ("value_per_byte_greedy", p, greedy_upgrades),
                ):
                    result = solver(values, c, budget, bucket_bytes) if solver is mckp_dynamic_program else solver(values, c, budget)
                    solutions[policy] = result
                oracle_choices = solutions["cached_oracle"]["choices"]
                oracle_accuracy = float(y[np.arange(batch_size), oracle_choices].mean())
                if "legacy_selected_fidelity" in records:
                    selected_names = records.iloc[chosen_rows]["legacy_selected_fidelity"].tolist()
                    if all(name in fidelities for name in selected_names):
                        legacy_choices = np.asarray([fidelities.index(name) for name in selected_names], dtype=int)
                        solutions["legacy_sequential"] = {
                            "choices": legacy_choices,
                            "actual_cost": float(c[np.arange(batch_size), legacy_choices].sum()),
                        }
                for policy, result in solutions.items():
                    choices = result["choices"]
                    effective_accuracy = float(y[np.arange(batch_size), choices].mean())
                    setting_rows.append(
                        {
                            "batch_size": batch_size,
                            "repeat": repeat,
                            "batch_seed": batch_seed,
                            "budget_fraction": fraction,
                            "requested_budget": requested_budget,
                            "effective_budget": budget,
                            "policy": policy,
                            "effective_accuracy": effective_accuracy,
                            "accuracy_regret_vs_cached_oracle": oracle_accuracy - effective_accuracy,
                            "predicted_value": float(p[np.arange(batch_size), choices].sum()),
                            "bytes": result["actual_cost"],
                            "budget_violation_bytes": max(0.0, result["actual_cost"] - budget),
                        }
                    )
                feasible_fixed = []
                for item_idx, fidelity in enumerate(fidelities):
                    total_cost = float(c[:, item_idx].sum())
                    if total_cost <= budget:
                        feasible_fixed.append((float(p[:, item_idx].mean()), total_cost, item_idx, fidelity))
                if feasible_fixed:
                    _, total_cost, item_idx, fidelity = max(
                        feasible_fixed, key=lambda row: (row[0], -row[1])
                    )
                    acc = float(y[:, item_idx].mean())
                    setting_rows.append(
                        {
                            "batch_size": batch_size,
                            "repeat": repeat,
                            "batch_seed": batch_seed,
                            "budget_fraction": fraction,
                            "requested_budget": requested_budget,
                            "effective_budget": budget,
                            "policy": "best_feasible_fixed",
                            "selected_fixed_fidelity": fidelity,
                            "effective_accuracy": acc,
                            "accuracy_regret_vs_cached_oracle": oracle_accuracy - acc,
                            "predicted_value": None,
                            "bytes": total_cost,
                            "budget_violation_bytes": 0.0,
                        }
                    )
                if setting_rows:
                    frame = pd.DataFrame(setting_rows)
                    frame.to_csv(detail_csv, mode="a", header=not detail_csv.exists(), index=False)
                    rows.extend(setting_rows)
                    existing_keys.add(setting_key)
    result_frame = pd.read_csv(detail_csv) if detail_csv.exists() else pd.DataFrame(rows)
    aggregate = (
        result_frame.groupby(["batch_size", "budget_fraction", "policy"], as_index=False)
        .agg(
            effective_accuracy_mean=("effective_accuracy", "mean"),
            effective_accuracy_std=("effective_accuracy", "std"),
            bytes_mean=("bytes", "mean"),
            budget_violation_max=("budget_violation_bytes", "max"),
            accuracy_regret_mean=("accuracy_regret_vs_cached_oracle", "mean"),
        )
    )
    payload = {
        "query_records": str(Path(query_records_csv).resolve()),
        "query_records_sha256": sha256_file(query_records_csv),
        "sizes": str(Path(sizes_csv).resolve()),
        "sizes_sha256": sha256_file(sizes_csv),
        "legacy_query_records": None if not legacy_query_records_csv else str(Path(legacy_query_records_csv).resolve()),
        "legacy_query_records_sha256": None if not legacy_query_records_csv else sha256_file(legacy_query_records_csv),
        "seed": seed,
        "repeats": repeats,
        "bucket_bytes": bucket_bytes,
        "fidelities": fidelities,
        "aggregate": aggregate.to_dict(orient="records"),
    }
    write_json(output_json, payload)
    return result_frame, payload


def benchmark_router_artifact(artifact_path, questions_csv, output_json, repeats=30, core=0, max_questions=None, sample_seed=42):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_all_experiments as core_exp

    # Backward compatibility for artifacts written before the runner assigned
    # stable import paths to its serializable helper classes.
    main_module = sys.modules["__main__"]
    for class_name in (
        "ConstantModel",
        "NoCalibrator",
        "TemperatureCalibrator",
        "SharedSlopePlattModel",
        "SharedSlopePlattView",
        "TrainedFidelityModel",
    ):
        setattr(main_module, class_name, getattr(core_exp, class_name))

    try:
        import psutil

        process = psutil.Process()
        available = process.cpu_affinity()
        process.cpu_affinity([available[min(core, len(available) - 1)]])
        rss_before = process.memory_info().rss
        pinned_cpu = process.cpu_affinity()
    except Exception:
        process = None
        rss_before = None
        pinned_cpu = None
    start = time.perf_counter_ns()
    metadata, artifact = core_exp.load_model_artifact(artifact_path)
    cold_load_ms = (time.perf_counter_ns() - start) / 1e6
    questions = pd.read_csv(questions_csv, low_memory=False)
    question_col = metadata.get("question_col", "question")
    if question_col not in questions:
        raise ValueError(f"Questions CSV lacks {question_col}")
    benchmark_df = questions[[question_col]].copy()
    benchmark_df.index = np.arange(len(benchmark_df))
    total_questions = int(len(benchmark_df))
    if max_questions is not None and max_questions > 0 and len(benchmark_df) > max_questions:
        benchmark_df = (
            benchmark_df.sample(n=max_questions, random_state=sample_seed)
            .sort_index()
            .reset_index(drop=True)
        )
    first = benchmark_df.iloc[:1]
    tick = time.perf_counter_ns()
    core_exp.predict_with_artifact(artifact, first, first)
    cold_first_query_ms = (time.perf_counter_ns() - tick) / 1e6
    peak_rss = process.memory_info().rss if process is not None else None
    timings = []
    for _ in range(repeats):
        for start_idx in range(0, len(benchmark_df), 1):
            one = benchmark_df.iloc[start_idx : start_idx + 1]
            tick = time.perf_counter_ns()
            core_exp.predict_with_artifact(artifact, one, one)
            timings.append((time.perf_counter_ns() - tick) / 1e6)
            if process is not None:
                peak_rss = max(peak_rss, process.memory_info().rss)
    rss_after = process.memory_info().rss if process is not None else None
    payload = {
        "artifact": str(Path(artifact_path).resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_bytes": Path(artifact_path).stat().st_size,
        "questions_csv": str(Path(questions_csv).resolve()),
        "n_questions_total": total_questions,
        "n_questions_benchmarked": int(len(benchmark_df)),
        "repeats": int(repeats),
        "max_questions": None if max_questions is None else int(max_questions),
        "sample_seed": int(sample_seed),
        "cold_load_ms": cold_load_ms,
        "cold_first_query_ms": cold_first_query_ms,
        "warm_p50_ms": float(np.quantile(timings, 0.50)),
        "warm_p95_ms": float(np.quantile(timings, 0.95)),
        "warm_p99_ms": float(np.quantile(timings, 0.99)),
        "rss_delta_bytes": None if rss_before is None else int(rss_after - rss_before),
        "peak_process_memory_bytes": None if peak_rss is None else int(peak_rss),
        "pinned_cpu_affinity": pinned_cpu,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "claim_scope": "commodity CPU reference; not an edge-energy or Jetson measurement",
    }
    write_json(output_json, payload)
    return payload


def parse_packet_opportunity_trace(trace_path):
    values = np.fromfile(trace_path, dtype=float, sep=" ")
    if values.size == 0:
        raise ValueError("Trace contains no packet opportunities")
    if np.nanmax(values) > 1000:
        values /= 1000.0
    return values - values.min()


def replay_deadline(payload_bytes, timestamps, loss_rate=0.0, mtu=1400, seed=42):
    """Serialize a payload over circular packet-delivery opportunities.

    The public CellReplay traces contain one 1,400-byte delivery opportunity per
    ``up-heavy-pdo`` entry.  ``loss_rate`` exists only for backwards-compatible
    synthetic tests; publication runs keep it at zero because the trace already
    records link variability.
    """
    needed = int(math.ceil(payload_bytes / mtu))
    rng = np.random.default_rng(seed)
    delivered = 0
    slot = 0
    retransmissions = 0
    cycle_duration = max(float(timestamps[-1]), 1e-6)
    while delivered < needed:
        if slot >= len(timestamps) * 10000:
            raise RuntimeError("Replay did not converge")
        if rng.random() >= loss_rate:
            delivered += 1
        else:
            retransmissions += 1
        slot += 1
    trace_idx = (slot - 1) % len(timestamps)
    cycles = (slot - 1) // len(timestamps)
    completion = cycles * cycle_duration + float(timestamps[trace_idx])
    return {"completion_s": completion, "retransmissions": retransmissions, "packets": needed}


TRACE_SCENARIOS = (
    ("tmobile", "driving"),
    ("tmobile", "stationary-good"),
    ("tmobile", "weak-signal"),
    ("verizon", "crowded"),
    ("verizon", "stationary-good"),
    ("verizon", "walking"),
)


def deterministic_trace_offsets(trace_duration_s, count=100, seed=2027, key=""):
    """Return reproducible offsets that uniformly cover a circular trace."""
    if count <= 0 or trace_duration_s <= 0:
        raise ValueError("count and trace duration must be positive")
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    phase = int.from_bytes(digest[:8], "big") / 2**64
    return ((np.arange(count, dtype=float) + phase) / count) * trace_duration_s


def parse_delay_trace(path):
    values = np.loadtxt(path, usecols=(0, 1), dtype=float)
    if values.size == 0:
        raise ValueError(f"Delay trace contains no rows: {path}")
    values = np.atleast_2d(values)
    values /= 1000.0
    values[:, 0] -= values[0, 0]
    return values[:, 0], values[:, 1]


def parse_latency_offsets(path):
    values = np.loadtxt(path, usecols=(0, 1), dtype=float)
    if values.size == 0:
        raise ValueError(f"Latency-offset trace contains no rows: {path}")
    values = np.atleast_2d(values)
    values[:, 1] /= 1000.0
    return values[:, 0], values[:, 1]


def circular_completion_latencies(
    payload_bytes,
    opportunity_timestamps,
    starts,
    delay_times=None,
    delay_values=None,
    offset_sizes=None,
    offset_values=None,
    mtu=1400,
):
    """Vectorized one-way completion times for circular opportunity traces."""
    timestamps = np.asarray(opportunity_timestamps, dtype=float)
    if timestamps.ndim != 1 or len(timestamps) == 0 or timestamps[0] > timestamps[-1]:
        raise ValueError("Opportunity timestamps must be a non-empty sorted vector")
    duration = float(timestamps[-1])
    if duration <= 0:
        raise ValueError("Trace duration must be positive")
    starts = np.mod(np.asarray(starts, dtype=float), duration)
    packets = max(1, int(math.ceil(float(payload_bytes) / mtu)))
    first = np.searchsorted(timestamps, starts, side="left")
    absolute_target = first + packets - 1
    cycles = absolute_target // len(timestamps)
    target = absolute_target % len(timestamps)
    completion = cycles * duration + timestamps[target] - starts
    if delay_times is not None and delay_values is not None:
        completion += np.interp(starts, delay_times, delay_values)
    if offset_sizes is not None and offset_values is not None:
        completion += float(np.interp(float(payload_bytes), offset_sizes, offset_values))
    return np.maximum(completion, 0.0)


def _infer_record_identity(path, frame):
    model = str(frame["model"].iloc[0]) if "model" in frame and frame["model"].notna().any() else Path(path).parent.name
    if "policy" in frame and frame["policy"].notna().any():
        policy = str(frame["policy"].iloc[0])
    else:
        token = str(path).lower()
        if "cascade" in token:
            policy = "logprob_cascade"
        elif "bert" in token:
            policy = "bert_router"
        else:
            policy = "voila"
    return model, policy


def _load_query_record_sets(paths, sizes_csv):
    sizes = pd.read_csv(sizes_csv, low_memory=False)
    sizes["image_id"] = sizes["image_id"].astype(str)
    size_lookup = sizes.pivot_table(index="image_id", columns="fidelity", values="bytes", aggfunc="first")
    records = []
    canonical = {}
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["image_id"] = frame["image_id"].astype(str)
        model, policy = _infer_record_identity(path, frame)
        frame["model"] = model
        frame["policy"] = policy
        if policy == "voila":
            canonical[model] = frame.copy()
        frame = frame.join(size_lookup, on="image_id", rsuffix="_payload")
        if frame["selected_fidelity"].isna().any():
            raise ValueError(f"Missing selected fidelity in {path}")
        exact = []
        for row in frame.itertuples(index=False):
            fidelity = str(row.selected_fidelity)
            if policy == "logprob_cascade":
                upto = FIDELITIES.index(fidelity) + 1
                exact.append(sum(float(getattr(row, item)) for item in FIDELITIES[:upto]))
            else:
                exact.append(float(getattr(row, fidelity)))
        frame["payload_bytes"] = exact
        records.append(frame)
    if not canonical:
        raise ValueError("At least one isotonic VOILA query-record file is required")
    return records, canonical, size_lookup


def run_trace_simulation(
    query_record_paths,
    sizes_csv,
    trace_root,
    output_dir,
    deadlines=(1.0, 5.0, 10.0),
    start_offsets=100,
    seed=2027,
):
    """Run the recorded-trace one-way uplink serialization simulation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_sets, canonical, size_lookup = _load_query_record_sets(query_record_paths, sizes_csv)
    simulated_sets = list(record_sets)
    for model, frame in canonical.items():
        base = frame.join(size_lookup, on="image_id", rsuffix="_payload")
        for policy, fidelity in (("full", "full"),):
            derived = base.copy()
            derived["model"] = model
            derived["policy"] = policy
            derived["selected_fidelity"] = fidelity
            derived["selected_correct"] = derived[f"correct_{fidelity}"].astype(bool)
            derived["payload_bytes"] = derived[fidelity].astype(float)
            simulated_sets.append(derived)

    trace_payload = {}
    input_hashes = {str(Path(sizes_csv).resolve()): sha256_file(sizes_csv)}
    for carrier, scenario in TRACE_SCENARIOS:
        root = Path(trace_root) / carrier / scenario
        heavy = root / "up-heavy-pdo"
        delay = root / "up-delay-light-pdo"
        offset = root / "latency-offset-up"
        for path in (heavy, delay, offset):
            if not path.exists():
                raise FileNotFoundError(path)
            input_hashes[str(path.resolve())] = sha256_file(path)
        timestamps = parse_packet_opportunity_trace(heavy)
        delay_times, delay_values = parse_delay_trace(delay)
        offset_sizes, offset_values = parse_latency_offsets(offset)
        trace_payload[(carrier, scenario)] = (
            timestamps,
            delay_times,
            delay_values,
            offset_sizes,
            offset_values,
        )

    raw_rows = []
    for frame in simulated_sets:
        model = str(frame["model"].iloc[0])
        policy = str(frame["policy"].iloc[0])
        for carrier, scenario in TRACE_SCENARIOS:
            timestamps, delay_times, delay_values, offset_sizes, offset_values = trace_payload[(carrier, scenario)]
            duration = float(timestamps[-1])
            cache = {}
            for row in frame.itertuples(index=False):
                key = (str(row.image_id), int(round(float(row.payload_bytes))))
                if key not in cache:
                    starts = deterministic_trace_offsets(duration, start_offsets, seed, f"{carrier}/{scenario}/{key[0]}")
                    cache[key] = circular_completion_latencies(
                        key[1], timestamps, starts, delay_times, delay_values, offset_sizes, offset_values
                    )
                latencies = cache[key]
                item = {
                    "carrier": carrier,
                    "scenario": scenario,
                    "model": model,
                    "policy": policy,
                    "fold": int(row.fold),
                    "image_id": str(row.image_id),
                    "question_id": str(row.question_id),
                    "selected_fidelity": str(row.selected_fidelity),
                    "payload_bytes": float(row.payload_bytes),
                    "correct": bool(row.selected_correct),
                    "latency_p50_s": float(np.quantile(latencies, 0.50)),
                    "latency_p95_s": float(np.quantile(latencies, 0.95)),
                }
                for deadline in deadlines:
                    suffix = str(deadline).rstrip("0").rstrip(".").replace(".", "p")
                    rate = float(np.mean(latencies <= deadline))
                    item[f"completion_{suffix}s"] = rate
                    item[f"dqa_{suffix}s"] = rate * float(bool(row.selected_correct))
                raw_rows.append(item)

    raw = pd.DataFrame(raw_rows)

    # Cross-validated best fixed fidelity: choose on the other four outer folds,
    # separately for each trace and deadline, and evaluate the held-out fold.
    fixed_rows = []
    for model, frame in canonical.items():
        base = frame.join(size_lookup, on="image_id", rsuffix="_payload")
        for carrier, scenario in TRACE_SCENARIOS:
            timestamps, delay_times, delay_values, offset_sizes, offset_values = trace_payload[(carrier, scenario)]
            duration = float(timestamps[-1])
            for deadline in deadlines:
                per_fidelity = {}
                for fidelity in FIDELITIES:
                    values = []
                    for row in base.itertuples(index=False):
                        starts = deterministic_trace_offsets(duration, start_offsets, seed, f"{carrier}/{scenario}/{row.image_id}")
                        latencies = circular_completion_latencies(
                            getattr(row, fidelity), timestamps, starts, delay_times, delay_values, offset_sizes, offset_values
                        )
                        values.append(float(np.mean(latencies <= deadline)) * float(bool(getattr(row, f"correct_{fidelity}"))))
                    per_fidelity[fidelity] = np.asarray(values)
                for fold in sorted(base["fold"].unique()):
                    train_mask = base["fold"].to_numpy() != fold
                    test_mask = ~train_mask
                    chosen = max(FIDELITIES, key=lambda f: (float(per_fidelity[f][train_mask].mean()), -FIDELITIES.index(f)))
                    subset = base.loc[test_mask]
                    for index, row in subset.iterrows():
                        starts = deterministic_trace_offsets(duration, start_offsets, seed, f"{carrier}/{scenario}/{row.image_id}")
                        latencies = circular_completion_latencies(
                            row[chosen], timestamps, starts, delay_times, delay_values, offset_sizes, offset_values
                        )
                        rate = float(np.mean(latencies <= deadline))
                        fixed_rows.append({
                            "carrier": carrier, "scenario": scenario, "model": model, "policy": "best_fixed",
                            "fold": int(fold), "image_id": str(row.image_id), "question_id": str(row.question_id),
                            "selected_fidelity": chosen, "payload_bytes": float(row[chosen]),
                            "correct": bool(row[f"correct_{chosen}"]), "latency_p50_s": float(np.quantile(latencies, .5)),
                            "latency_p95_s": float(np.quantile(latencies, .95)), "deadline_s": float(deadline),
                            "completion_rate": rate, "deadline_qualified_accuracy": rate * float(bool(row[f"correct_{chosen}"])),
                        })
    fixed = pd.DataFrame(fixed_rows)

    aggregate_rows = []
    group_cols = ["carrier", "scenario", "model", "policy"]
    for keys, group in raw.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        for deadline in deadlines:
            suffix = str(deadline).rstrip("0").rstrip(".").replace(".", "p")
            aggregate_rows.append({**base, "deadline_s": float(deadline), "n_queries": int(len(group)),
                "mean_payload_bytes": float(group["payload_bytes"].mean()),
                "transfer_latency_p50_s": float(group["latency_p50_s"].median()),
                "transfer_latency_p95_s": float(group["latency_p95_s"].quantile(.95)),
                "deadline_completion_rate": float(group[f"completion_{suffix}s"].mean()),
                "deadline_qualified_accuracy": float(group[f"dqa_{suffix}s"].mean())})
    if not fixed.empty:
        for keys, group in fixed.groupby(group_cols + ["deadline_s"], sort=True):
            base = dict(zip(group_cols + ["deadline_s"], keys))
            aggregate_rows.append({**base, "n_queries": int(len(group)), "mean_payload_bytes": float(group["payload_bytes"].mean()),
                "transfer_latency_p50_s": float(group["latency_p50_s"].median()),
                "transfer_latency_p95_s": float(group["latency_p95_s"].quantile(.95)),
                "deadline_completion_rate": float(group["completion_rate"].mean()),
                "deadline_qualified_accuracy": float(group["deadline_qualified_accuracy"].mean())})
    aggregate = pd.DataFrame(aggregate_rows).sort_values(group_cols + ["deadline_s"])
    raw.to_csv(output_dir / "query_trace_records.csv.gz", index=False, compression="gzip")
    fixed.to_csv(output_dir / "best_fixed_query_trace_records.csv.gz", index=False, compression="gzip")
    aggregate.to_csv(output_dir / "trace_aggregate.csv", index=False)

    import matplotlib.pyplot as plt
    plot = aggregate.groupby(["policy", "deadline_s"], as_index=False)["deadline_qualified_accuracy"].mean()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for policy, group in plot.groupby("policy"):
        ax.plot(group["deadline_s"], 100 * group["deadline_qualified_accuracy"], marker="o", label=policy.replace("_", " "))
    ax.set(xlabel="Evidence-arrival deadline (s)", ylabel="Deadline-qualified accuracy (%)", xticks=list(deadlines))
    ax.grid(alpha=.25); ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(output_dir / "deadline_qualified_accuracy.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "deadline_qualified_accuracy.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    metadata = {
        "claim_scope": "recorded-trace one-way uplink serialization simulation; not TCP, end-to-end CellReplay, VLM latency, or operator response time",
        "packet_opportunity_bytes": 1400, "synthetic_packet_loss": False, "deadlines_s": list(map(float, deadlines)),
        "start_offsets_per_query": int(start_offsets), "offset_sampling": "deterministic uniform phase over each circular trace",
        "seed": int(seed), "trace_scenarios": [f"{a}/{b}" for a, b in TRACE_SCENARIOS],
        "input_hashes": input_hashes | {str(Path(p).resolve()): sha256_file(p) for p in query_record_paths},
        "rows": aggregate.to_dict(orient="records"),
    }
    write_json(output_dir / "trace_aggregate.json", metadata)
    return raw, aggregate, metadata


MODEL_LABELS = {
    "pixtral": "Pixtral-12B", "llava": "LLaVA-7B", "llama4": "Llama-4-Scout",
    "qwen7b": "Qwen2-VL-7B", "qwen72b": "Qwen2-VL-72B", "qwen235b": "Qwen-235B",
}


def latex_escape(value):
    text = str(value)
    for source, target in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        text = text.replace(source, target)
    return text


def _exact_policy_frame(path, sizes, policy, cumulative=False):
    frame = pd.read_csv(path, low_memory=False)
    frame["image_id"] = frame["image_id"].astype(str)
    joined = frame.merge(sizes, on="image_id", how="left", validate="many_to_one")
    if joined[FIDELITIES].isna().any().any():
        raise ValueError(f"Size join failed for {path}")
    payload = []
    for row in joined.itertuples(index=False):
        fidelity = str(row.selected_fidelity)
        chosen = FIDELITIES[: FIDELITIES.index(fidelity) + 1] if cumulative else [fidelity]
        payload.append(sum(float(getattr(row, item)) for item in chosen))
    return pd.DataFrame({
        "model": joined.get("model", pd.Series([Path(path).parent.name] * len(joined))).astype(str),
        "policy": policy, "fold": joined["fold"].astype(int), "image_id": joined["image_id"].astype(str),
        "question_id": joined["question_id"].astype(str), "selected_fidelity": joined["selected_fidelity"].astype(str),
        "correct": joined["selected_correct"].astype(bool), "bytes": payload,
    })


def _bootstrap_policy_summary(frame, n_boot=2000, seed=2027):
    grouped = frame.groupby("image_id", sort=False).agg(n=("correct", "size"), correct=("correct", "sum"), bytes=("bytes", "sum"))
    values = grouped[["n", "correct", "bytes"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    stats = np.empty((n_boot, 2))
    for index in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        count = sample[:, 0].sum()
        stats[index] = sample[:, 1].sum() / count, sample[:, 2].sum() / count
    return {
        "n_queries": int(len(frame)), "n_image_clusters": int(len(grouped)),
        "accuracy": float(frame["correct"].mean()), "mean_bytes": float(frame["bytes"].mean()),
        "accuracy_ci_low": float(np.quantile(stats[:, 0], .025)), "accuracy_ci_high": float(np.quantile(stats[:, 0], .975)),
        "bytes_ci_low": float(np.quantile(stats[:, 1], .025)), "bytes_ci_high": float(np.quantile(stats[:, 1], .975)),
    }


def _paired_bootstrap(left, right, n_boot=2000, seed=2027):
    keys = ["image_id", "question_id"]
    paired = left[keys + ["correct", "bytes"]].merge(
        right[keys + ["correct", "bytes"]], on=keys, suffixes=("_left", "_right"), validate="one_to_one"
    )
    grouped = paired.groupby("image_id", sort=False).agg(
        n=("question_id", "size"), acc_left=("correct_left", "sum"), acc_right=("correct_right", "sum"),
        bytes_left=("bytes_left", "sum"), bytes_right=("bytes_right", "sum"),
    )
    values = grouped.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, 2))
    for index in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        count = sample[:, 0].sum()
        draws[index, 0] = (sample[:, 1].sum() - sample[:, 2].sum()) / count
        draws[index, 1] = (sample[:, 3].sum() - sample[:, 4].sum()) / count
    return {
        "accuracy_difference": float(left["correct"].mean() - right["correct"].mean()),
        "accuracy_difference_ci_low": float(np.quantile(draws[:, 0], .025)),
        "accuracy_difference_ci_high": float(np.quantile(draws[:, 0], .975)),
        "bytes_difference": float(left["bytes"].mean() - right["bytes"].mean()),
        "bytes_difference_ci_low": float(np.quantile(draws[:, 1], .025)),
        "bytes_difference_ci_high": float(np.quantile(draws[:, 1], .975)),
    }


def _matched_fixed_bootstrap(method, fixed_frames, n_boot=2000, seed=2027):
    keys = ["image_id", "question_id"]
    joined = method[keys + ["correct", "bytes"]].rename(columns={"correct": "method_correct", "bytes": "method_bytes"})
    for fidelity, frame in fixed_frames.items():
        joined = joined.merge(frame[keys + ["correct", "bytes"]].rename(
            columns={"correct": f"correct_{fidelity}", "bytes": f"bytes_{fidelity}"}), on=keys, validate="one_to_one")
    numeric = ["method_correct", "method_bytes"] + [column for fidelity in FIDELITIES for column in (f"correct_{fidelity}", f"bytes_{fidelity}")]
    grouped = joined.groupby("image_id", sort=False)[numeric].sum()
    counts = joined.groupby("image_id", sort=False).size().to_numpy(float)
    values = grouped.to_numpy(float)
    rng = np.random.default_rng(seed)
    savings, gains = [], []
    for _ in range(n_boot):
        selected = rng.integers(0, len(values), len(values))
        sample, count = values[selected], counts[selected].sum()
        method_accuracy, method_bytes = sample[:, 0].sum() / count, sample[:, 1].sum() / count
        fixed_stats = []
        offset = 2
        for _fidelity in FIDELITIES:
            fixed_stats.append((sample[:, offset].sum() / count, sample[:, offset + 1].sum() / count))
            offset += 2
        accuracy_eligible = [cost for accuracy, cost in fixed_stats if accuracy >= method_accuracy]
        cost_eligible = [accuracy for accuracy, cost in fixed_stats if cost <= method_bytes]
        if accuracy_eligible:
            savings.append(min(accuracy_eligible) - method_bytes)
        if cost_eligible:
            gains.append(method_accuracy - max(cost_eligible))
    def interval(values):
        return [float(x) for x in np.quantile(values, [.025, .975])] if values else [None, None]
    return {
        "bytes_saved_at_matched_accuracy_ci_low": interval(savings)[0],
        "bytes_saved_at_matched_accuracy_ci_high": interval(savings)[1],
        "accuracy_gain_at_matched_cost_ci_low": interval(gains)[0],
        "accuracy_gain_at_matched_cost_ci_high": interval(gains)[1],
        "matched_accuracy_valid_bootstraps": len(savings), "matched_cost_valid_bootstraps": len(gains),
    }


def generate_paper_artifacts(results_root, paper_output):
    """Generate every submission table, headline macro, and main figure from frozen results."""
    results_root, paper_output = Path(results_root), Path(paper_output)
    paper_output.mkdir(parents=True, exist_ok=True)
    size_source = ROOT / "results" / "percom2027_final" / "systems" / "floodnet_payload_sizes.csv"
    size_long = pd.read_csv(size_source, low_memory=False)
    size_long["image_id"] = size_long["image_id"].astype(str)
    sizes = size_long.pivot_table(index="image_id", columns="fidelity", values="bytes", aggfunc="first").reset_index()
    models = sorted(path.parent.name for path in (results_root / "main" / "floodnet").glob("*/query_records.csv"))
    all_records, summaries, paired_rows, fixed_points, failure_rows = [], [], [], [], []
    for model_index, model in enumerate(models):
        main_path = results_root / "main" / "floodnet" / model / "query_records.csv"
        voila = _exact_policy_frame(main_path, sizes, "voila")
        main_raw = pd.read_csv(main_path, low_memory=False)
        main_raw["image_id"] = main_raw["image_id"].astype(str)
        joined = main_raw.merge(sizes, on="image_id", how="left", validate="many_to_one")
        fixed_model = []
        for fidelity in FIDELITIES:
            point = pd.DataFrame({"model": model, "policy": f"fixed_{fidelity}", "fold": joined["fold"],
                "image_id": joined["image_id"], "question_id": joined["question_id"].astype(str),
                "selected_fidelity": fidelity, "correct": joined[f"correct_{fidelity}"].astype(bool), "bytes": joined[fidelity].astype(float)})
            fixed_model.append(point); fixed_points.append(point)
        full = fixed_model[-1].assign(policy="full")
        result = json.loads((results_root / "main" / "floodnet" / model / "voila_results.json").read_text(encoding="utf-8"))
        chosen_rows = []
        for fold_payload in result["folds"]:
            fixed_validation = fold_payload["tradeoff"]["fixed_validation_accuracy"]
            selected = max(FIDELITIES, key=lambda fidelity: (fixed_validation[fidelity], -FIDELITIES.index(fidelity)))
            chosen_rows.append(fixed_model[FIDELITIES.index(selected)].loc[joined["fold"] == fold_payload["fold"]])
        best_fixed = pd.concat(chosen_rows, ignore_index=True).assign(policy="best_fixed")
        cascade = _exact_policy_frame(results_root / "baselines" / "logprob_cascade" / "floodnet" / model / "query_records.csv", sizes, "logprob_cascade", True)
        bert = _exact_policy_frame(results_root / "baselines" / "bert_router" / "floodnet" / model / "query_records.csv", sizes, "bert_router")
        legacy = _exact_policy_frame(results_root / "ablations" / "calibration" / "floodnet" / model / "legacy_query_records.csv", sizes, "legacy_sequential")
        oracle = fixed_model[0].copy(); oracle["policy"] = "cached_oracle"
        oracle["selected_fidelity"] = "none"; oracle["correct"] = False; oracle["bytes"] = fixed_model[0]["bytes"]
        for row_index in range(len(oracle)):
            for fidelity_index, fidelity in enumerate(FIDELITIES):
                if bool(fixed_model[fidelity_index].iloc[row_index]["correct"]):
                    oracle.at[row_index, "selected_fidelity"] = fidelity
                    oracle.at[row_index, "correct"] = True
                    oracle.at[row_index, "bytes"] = fixed_model[fidelity_index].iloc[row_index]["bytes"]
                    break
        policies = {"full": full, "best_fixed": best_fixed, "logprob_cascade": cascade, "bert_router": bert,
                    "legacy_sequential": legacy, "voila": voila, "cached_oracle": oracle}
        aligned = voila.reset_index(drop=True).copy()
        aligned["oracle_correct"] = oracle.reset_index(drop=True)["correct"].astype(bool)
        aligned["oracle_bytes"] = oracle.reset_index(drop=True)["bytes"].astype(float)
        aligned["oracle_fidelity"] = oracle.reset_index(drop=True)["selected_fidelity"].astype(str)
        failure_rows.append({
            "model": model,
            "oracle_accuracy": float(aligned.oracle_correct.mean()),
            "oracle_exact_rate": float((aligned.oracle_correct & (aligned.selected_fidelity == aligned.oracle_fidelity)).mean()),
            "under_acquisition_rate": float((aligned.oracle_correct & ~aligned.correct).mean()),
            "over_acquisition_rate": float((aligned.oracle_correct & aligned.correct & (aligned.bytes > aligned.oracle_bytes)).mean()),
            "irreducible_rate": float((~aligned.oracle_correct).mean()),
        })
        for policy, frame in policies.items():
            all_records.append(frame)
            summaries.append({"model": model, "policy": policy, **_bootstrap_policy_summary(frame, seed=2027 + model_index)})
        for policy in ("full", "best_fixed", "logprob_cascade", "bert_router", "legacy_sequential"):
            paired_rows.append({"model": model, "left_policy": "voila", "right_policy": policy,
                                **_paired_bootstrap(voila, policies[policy], seed=2027 + model_index)})
    records = pd.concat(all_records, ignore_index=True)
    summary = pd.DataFrame(summaries)
    paired = pd.DataFrame(paired_rows)
    fixed_frame = pd.concat(fixed_points, ignore_index=True)
    records.to_csv(paper_output / "primary_policy_records.csv.gz", index=False, compression="gzip")
    summary.to_csv(paper_output / "primary_summary.csv", index=False)
    paired.to_csv(paper_output / "paired_bootstrap.csv", index=False)
    failure_frame = pd.DataFrame(failure_rows)
    failure_frame.to_csv(paper_output / "failure_analysis.csv", index=False)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Model & Oracle & Exact & Under & Over & None \\", r"\midrule"]
    for row in failure_frame.itertuples(index=False):
        lines.append(
            f"{MODEL_LABELS.get(row.model,row.model)} & {100*row.oracle_accuracy:.1f} & "
            f"{100*row.oracle_exact_rate:.1f} & {100*row.under_acquisition_rate:.1f} & "
            f"{100*row.over_acquisition_rate:.1f} & {100*row.irreducible_rate:.1f} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "failure_analysis_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Discrete matched-accuracy/cost comparisons against the five fixed points.
    matched_rows = []
    for model in models:
        fixed_summary = summary[(summary.model == model) & summary.policy.str.startswith("fixed_")]
        if fixed_summary.empty:
            fixed_summary = pd.DataFrame([{"policy": f"fixed_{f}", **_bootstrap_policy_summary(
                fixed_frame[(fixed_frame.model == model) & (fixed_frame.policy == f"fixed_{f}")])} for f in FIDELITIES])
        for policy in ("voila", "logprob_cascade", "bert_router", "legacy_sequential"):
            method = summary[(summary.model == model) & (summary.policy == policy)].iloc[0]
            method_frame = records[(records.model == model) & (records.policy == policy)]
            fixed_frames_by_fidelity = {fidelity: fixed_frame[(fixed_frame.model == model) & (fixed_frame.policy == f"fixed_{fidelity}")]
                                        for fidelity in FIDELITIES}
            accuracy_eligible = fixed_summary[fixed_summary.accuracy >= method.accuracy]
            matched_cost = float(accuracy_eligible.mean_bytes.min()) if not accuracy_eligible.empty else float("nan")
            cost_eligible = fixed_summary[fixed_summary.mean_bytes <= method.mean_bytes]
            matched_accuracy = float(cost_eligible.accuracy.max()) if not cost_eligible.empty else float("nan")
            matched_rows.append({"model": model, "policy": policy, "fixed_cost_at_matched_accuracy": matched_cost,
                "bytes_saved_at_matched_accuracy": matched_cost - method.mean_bytes if np.isfinite(matched_cost) else np.nan,
                "fixed_accuracy_at_matched_cost": matched_accuracy,
                "accuracy_gain_at_matched_cost": method.accuracy - matched_accuracy if np.isfinite(matched_accuracy) else np.nan,
                **_matched_fixed_bootstrap(method_frame, fixed_frames_by_fidelity, seed=2027 + models.index(model))})
    matched = pd.DataFrame(matched_rows); matched.to_csv(paper_output / "matched_frontier_comparisons.csv", index=False)

    # Six-panel FloodNet measured-byte frontier.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.15), sharex=True)
    colors = {"voila": "#0072B2", "logprob_cascade": "#D55E00", "bert_router": "#009E73", "best_fixed": "#6A3D9A", "full": "#444444"}
    for ax, model in zip(axes.flat, models):
        points = []
        for fidelity in FIDELITIES:
            frame = fixed_frame[(fixed_frame.model == model) & (fixed_frame.policy == f"fixed_{fidelity}")]
            points.append((frame.bytes.mean(), 100 * frame.correct.mean()))
        points.sort(); ax.plot([p[0] for p in points], [p[1] for p in points], "o--", color="#888888", lw=1, ms=3, label="fixed")
        for policy in colors:
            row = summary[(summary.model == model) & (summary.policy == policy)].iloc[0]
            ax.scatter(row.mean_bytes, 100 * row.accuracy, s=23, color=colors[policy], label=policy.replace("_", " "), zorder=3)
        ax.set_xscale("log"); ax.grid(alpha=.22); ax.set_title(MODEL_LABELS.get(model, model), fontsize=8)
    for ax in axes[-1]: ax.set_xlabel("Mean transmitted bytes/query")
    for ax in axes[:, 0]: ax.set_ylabel("Accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="lower center", fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(paper_output / "floodnet_frontier.pdf", bbox_inches="tight")
    fig.savefig(paper_output / "floodnet_frontier.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    policy_order = ["full", "best_fixed", "logprob_cascade", "bert_router", "voila"]
    policy_labels = {"full": "Full", "best_fixed": "Best fixed", "logprob_cascade": "Cascade", "bert_router": "BERT", "voila": "VOILA"}
    lines = [r"\begin{tabular}{l" + "rr" * len(policy_order) + "}", r"\toprule", "Model & " + " & ".join(f"{policy_labels[p]} acc. & MB" for p in policy_order) + r" \\", r"\midrule"]
    for model in models:
        cells = []
        for policy in policy_order:
            row = summary[(summary.model == model) & (summary.policy == policy)].iloc[0]
            cells += [f"{100*row.accuracy:.1f}", f"{row.mean_bytes/1e6:.3f}"]
        lines.append(MODEL_LABELS.get(model, model) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "main_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Fixed-fidelity accuracy explains why full resolution is not an accuracy oracle.
    fixed_accuracy_rows = []
    for model in models:
        row = {"model": model}
        for fidelity in FIDELITIES:
            frame = fixed_frame[(fixed_frame.model == model) & (fixed_frame.policy == f"fixed_{fidelity}")]
            row[fidelity] = float(frame.correct.mean())
        fixed_accuracy_rows.append(row)
    fixed_accuracy = pd.DataFrame(fixed_accuracy_rows); fixed_accuracy.to_csv(paper_output / "fixed_fidelity_accuracy.csv", index=False)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Model & Caption & 32px & Q1 & Q10 & Full \\", r"\midrule"]
    for row in fixed_accuracy.itertuples(index=False):
        lines.append(MODEL_LABELS.get(row.model, row.model) + " & " + " & ".join(f"{100*getattr(row,f):.1f}" for f in FIDELITIES) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "fixed_fidelity_accuracy_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Motivating observation: the linguistic form of a FloodNet question
    # separates fidelity requirements before any visual representation is read.
    observation_rows = []
    for model in models:
        frame = pd.read_csv(results_root / "main" / "floodnet" / model / "query_records.csv", low_memory=False)
        frame["question_group"] = np.where(
            frame["question"].str.lower().str.strip().str.startswith("how many"),
            "Counting questions", "Condition questions",
        )
        for question_group, group in frame.groupby("question_group"):
            for fidelity in FIDELITIES:
                observation_rows.append({
                    "model": model,
                    "question_group": question_group,
                    "fidelity": fidelity,
                    "accuracy": float(group[f"correct_{fidelity}"].mean()),
                })
    observation = pd.DataFrame(observation_rows)
    observation.to_csv(paper_output / "motivating_observation.csv", index=False)
    observation_mean = observation.groupby(["question_group", "fidelity"], as_index=False).accuracy.mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.2))
    fidelity_ticks = np.arange(len(FIDELITIES))
    fidelity_names = ["caption", "32px", "Q1", "Q10", "full"]
    for question_group, color, marker in (
        ("Condition questions", "#0072B2", "o"),
        ("Counting questions", "#D55E00", "s"),
    ):
        group = observation_mean[observation_mean.question_group == question_group].set_index("fidelity")
        axes[0].plot(
            fidelity_ticks,
            [100 * group.loc[fidelity, "accuracy"] for fidelity in FIDELITIES],
            marker=marker,
            color=color,
            lw=1.8,
            label=question_group,
        )
    axes[0].set_xticks(fidelity_ticks, fidelity_names)
    axes[0].set(ylabel="Mean accuracy across VLMs (%)", xlabel="Available visual fidelity")
    axes[0].grid(alpha=.22); axes[0].legend(fontsize=7, frameon=False)
    mean_bytes = size_long.groupby("fidelity").bytes.mean()
    for model in models:
        row = fixed_accuracy[fixed_accuracy.model == model].iloc[0]
        axes[1].plot(
            [mean_bytes.loc[fidelity] for fidelity in FIDELITIES],
            [100 * row[fidelity] for fidelity in FIDELITIES],
            marker="o", ms=3, lw=1, alpha=.78, label=MODEL_LABELS.get(model, model),
        )
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Mean measured bytes", ylabel="Accuracy (%)")
    axes[1].grid(alpha=.22); axes[1].legend(fontsize=5.8, frameon=False, ncol=2)
    axes[0].set_title("Question form predicts information need", fontsize=8.5)
    axes[1].set_title("No fixed fidelity dominates", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(paper_output / "motivating_observation.pdf", bbox_inches="tight")
    fig.savefig(paper_output / "motivating_observation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    selected = records[records.policy == "voila"].copy()
    distribution = selected.groupby(["model", "selected_fidelity"]).size().rename("count").reset_index()
    distribution["fraction"] = distribution["count"] / distribution.groupby("model")["count"].transform("sum")
    distribution.to_csv(paper_output / "selection_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    bottoms = np.zeros(len(models))
    palette = ["#56B4E9", "#009E73", "#F0E442", "#E69F00", "#D55E00"]
    for fidelity, color in zip(FIDELITIES, palette):
        values = []
        for model in models:
            match = distribution[(distribution.model == model) & (distribution.selected_fidelity == fidelity)]
            values.append(100 * (float(match.fraction.iloc[0]) if len(match) else 0.0))
        ax.bar(np.arange(len(models)), values, bottom=bottoms, color=color,
               label=fidelity_names[FIDELITIES.index(fidelity)], width=.72)
        bottoms += np.asarray(values)
    short_labels = [MODEL_LABELS.get(model, model).replace("Qwen2-VL-", "Q-") for model in models]
    ax.set_xticks(np.arange(len(models)), short_labels, rotation=25, ha="right", fontsize=6.5)
    ax.set(ylabel="Selected queries (%)", ylim=(0, 100))
    ax.legend(ncol=3, fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.18))
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(paper_output / "selection_distribution.pdf", bbox_inches="tight")
    fig.savefig(paper_output / "selection_distribution.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Paired accuracy and byte deltas for the primary competitor checks.
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Model & $\Delta$acc BF & $\Delta$MB BF & $\Delta$acc BERT & $\Delta$MB BERT \\", r"\midrule"]
    for model in models:
        cells = []
        for policy in ("best_fixed", "bert_router"):
            row = paired[(paired.model == model) & (paired.right_policy == policy)].iloc[0]
            cells += [f"{100*row.accuracy_difference:+.1f}", f"{row.bytes_difference/1e6:+.3f}"]
        lines.append(MODEL_LABELS.get(model, model) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "paired_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Compact confidence-interval table for the proposed operating point.
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Model & Accuracy & 95\% CI & MB & 95\% CI \\", r"\midrule"]
    for model in models:
        row = summary[(summary.model == model) & (summary.policy == "voila")].iloc[0]
        lines.append(f"{MODEL_LABELS.get(model,model)} & {100*row.accuracy:.1f} & [{100*row.accuracy_ci_low:.1f},{100*row.accuracy_ci_high:.1f}] & {row.mean_bytes/1e6:.3f} & [{row.bytes_ci_low/1e6:.3f},{row.bytes_ci_high/1e6:.3f}] " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "voila_ci_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    trace = pd.read_csv(results_root / "trace_sim" / "trace_aggregate.csv")
    trace_mean = trace.groupby(["policy", "deadline_s"], as_index=False).agg(
        bytes=("mean_payload_bytes", "mean"), p50=("transfer_latency_p50_s", "mean"), p95=("transfer_latency_p95_s", "mean"),
        completion=("deadline_completion_rate", "mean"), dqa=("deadline_qualified_accuracy", "mean"))
    trace_mean.to_csv(paper_output / "trace_summary.csv", index=False)
    shutil.copy2(results_root / "trace_sim" / "deadline_qualified_accuracy.pdf", paper_output / "deadline_qualified_accuracy.pdf")
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule", r"Policy & MB & p50 & p95 & C@1 & DQA@1 & DQA@5 & DQA@10 \\", r"\midrule"]
    for policy in policy_order:
        group = trace_mean[trace_mean.policy == policy]
        one = group[group.deadline_s == 1].iloc[0]; five = group[group.deadline_s == 5].iloc[0]; ten = group[group.deadline_s == 10].iloc[0]
        lines.append(f"{policy_labels[policy]} & {one.bytes/1e6:.3f} & {one.p50:.3f} & {one.p95:.3f} & {100*one.completion:.1f} & {100*one.dqa:.1f} & {100*five.dqa:.1f} & {100*ten.dqa:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "trace_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    scenario = trace[trace.deadline_s == 1].groupby(["carrier", "scenario", "policy"], as_index=False).agg(
        dqa=("deadline_qualified_accuracy", "mean"), completion=("deadline_completion_rate", "mean"), p95=("transfer_latency_p95_s", "mean"))
    scenario.to_csv(paper_output / "trace_scenario_summary.csv", index=False)
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Carrier & Scenario & Full DQA & BF DQA & VOILA DQA & VOILA p95 \\", r"\midrule"]
    for carrier, scenario_name in TRACE_SCENARIOS:
        group = scenario[(scenario.carrier == carrier) & (scenario.scenario == scenario_name)].set_index("policy")
        lines.append(f"{carrier.title()} & {latex_escape(scenario_name)} & {100*group.loc['full','dqa']:.1f} & {100*group.loc['best_fixed','dqa']:.1f} & {100*group.loc['voila','dqa']:.1f} & {group.loc['voila','p95']:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "trace_scenario_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stress_rows = []
    calibration_rows = []
    feature_rows = []
    for model in models:
        main = json.loads((results_root / "main" / "floodnet" / model / "voila_results.json").read_text())
        stress = json.loads((results_root / "floodnet_stress" / "floodnet" / model / "voila_results.json").read_text())
        cal = json.loads((results_root / "ablations" / "calibration" / "floodnet" / model / "results.json").read_text())
        feature = json.loads((results_root / "ablations" / "features" / "floodnet" / model / "results.json").read_text())
        stress_rows.append({"model": model, "main_accuracy": main["accuracy_mean"], "template_accuracy": stress["accuracy_mean"],
            "template_delta": stress["accuracy_mean"] - main["accuracy_mean"], "recommended_calibrator": cal["calibrator_selection"]["recommended"],
            "tfidf_only_accuracy": feature["features"]["tfidf_only"]["accuracy_mean"], "struct_only_accuracy": feature["features"]["struct_only"]["accuracy_mean"],
            "qtype_only_accuracy": feature["features"]["qtype_only"]["accuracy_mean"]})
        calibration_rows.append({"model": model, **{
            f"{variant}_{metric}": cal["variants"][variant][metric]
            for variant in ("none", "iso", "shared_platt")
            for metric in ("accuracy_mean", "avg_cost_mean")
        }})
        feature_rows.append({"model": model, **{
            f"{variant}_{metric}": feature["features"][variant][metric]
            for variant in ("tfidf+struct", "struct_only", "qtype_only")
            for metric in ("accuracy_mean", "avg_cost_mean")
        }})
    stress_frame = pd.DataFrame(stress_rows); stress_frame.to_csv(paper_output / "robustness_summary.csv", index=False)
    lines = [r"\begin{tabular}{lrrrrl}", r"\toprule", r"Model & Main & Template & $\Delta$ & Struct only & Cal. \\", r"\midrule"]
    for row in stress_frame.itertuples(index=False):
        lines.append(f"{MODEL_LABELS.get(row.model,row.model)} & {row.main_accuracy:.1f} & {row.template_accuracy:.1f} & {row.template_delta:+.1f} & {row.struct_only_accuracy:.1f} & {latex_escape(row.recommended_calibrator)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "robustness_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_frame.to_csv(paper_output / "calibration_ablation.csv", index=False)
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Model & None & Independent iso & Shared Platt \\", r"\midrule"]
    for row in calibration_frame.itertuples(index=False):
        cells = []
        for variant in ("none", "iso", "shared_platt"):
            cells.append(f"{getattr(row, variant + '_accuracy_mean'):.1f}/{getattr(row, variant + '_avg_cost_mean')/1e6:.3f}")
        lines.append(MODEL_LABELS.get(row.model, row.model) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "calibration_ablation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    feature_frame = pd.DataFrame(feature_rows)
    feature_frame.to_csv(paper_output / "feature_ablation.csv", index=False)
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Model & TF--IDF+struct & Struct only & Type only \\", r"\midrule"]
    for _, row in feature_frame.iterrows():
        cells = []
        for variant in ("tfidf+struct", "struct_only", "qtype_only"):
            cells.append(f"{row[variant + '_accuracy_mean']:.1f}/{row[variant + '_avg_cost_mean']/1e6:.3f}")
        lines.append(MODEL_LABELS.get(row["model"], row["model"]) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "feature_ablation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    mission = json.loads((results_root / "mission" / "qwen235b_iso.json").read_text())["aggregate"]
    mission = pd.DataFrame(mission); mission = mission[(mission.batch_size == 100) & (mission.budget_fraction.isin([.1, .25, .5]))]
    mission.to_csv(paper_output / "mission_summary.csv", index=False)
    chosen_policies = ["best_feasible_fixed", "budgeted_voila", "value_per_byte_greedy", "legacy_sequential"]
    mission_labels = {"best_feasible_fixed":"Best fixed", "budgeted_voila":"Budgeted VOILA", "value_per_byte_greedy":"Greedy", "legacy_sequential":"Legacy"}
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Policy & 10\% & 25\% & 50\% \\", r"\midrule"]
    for policy in chosen_policies:
        group = mission[mission.policy == policy].set_index("budget_fraction")
        lines.append(mission_labels[policy] + " & " + " & ".join(f"{100*group.loc[x,'effective_accuracy_mean']:.1f}" for x in (.1,.25,.5)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "mission_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    mission_all = pd.DataFrame(json.loads((results_root / "mission" / "qwen235b_iso.json").read_text())["aggregate"])
    mission_plot = mission_all[mission_all.batch_size == 100]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    plot_labels = {"best_feasible_fixed":"best fixed", "budgeted_voila":"budgeted VOILA", "value_per_byte_greedy":"greedy", "legacy_sequential":"legacy"}
    for policy, label in plot_labels.items():
        group = mission_plot[mission_plot.policy == policy].sort_values("budget_fraction")
        ax.plot(100*group.budget_fraction, 100*group.effective_accuracy_mean, marker="o", ms=3, label=label)
    ax.set(xlabel="Budget (% of full transfer)", ylabel="Mission accuracy (%)")
    ax.grid(alpha=.25); ax.legend(fontsize=6.5, frameon=False, ncol=2); fig.tight_layout()
    fig.savefig(paper_output / "mission_accuracy.pdf", bbox_inches="tight"); fig.savefig(paper_output / "mission_accuracy.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    external_rows = []
    external_root = ROOT / "results" / "percom2027_final" / "main"
    for dataset in ("vqav2", "gqa", "textvqa"):
        dataset_rows = []
        for path in (external_root / dataset).glob("*/voila_results.json"):
            payload = json.loads(path.read_text()); dataset_rows.append(payload)
        external_rows.append({"dataset": dataset, "models": len(dataset_rows),
            "accuracy_mean": float(np.mean([p["accuracy_mean"] for p in dataset_rows])),
            "cost_reduction_mean": float(np.mean([p["cost_reduction_pct_mean"] for p in dataset_rows]))})
    external = pd.DataFrame(external_rows); external.to_csv(paper_output / "external_validity.csv", index=False)
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Dataset & Models & Accuracy & Normalized cost reduction \\", r"\midrule"]
    for row in external.itertuples(index=False): lines.append(f"{latex_escape(row.dataset.upper())} & {row.models} & {row.accuracy_mean:.1f} & {row.cost_reduction_mean:.1f}\\% " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "external_validity_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    size_summary = size_long.groupby("fidelity", as_index=False).agg(
        median_bytes=("bytes", "median"), p95_bytes=("bytes", lambda values: values.quantile(.95)), mean_bytes=("bytes", "mean")
    )
    size_summary["fidelity"] = pd.Categorical(size_summary["fidelity"], FIDELITIES, ordered=True)
    size_summary = size_summary.sort_values("fidelity"); size_summary.to_csv(paper_output / "fidelity_sizes.csv", index=False)
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Representation & Median & p95 & Mean bytes \\", r"\midrule"]
    fidelity_labels = {"caption":"Cached caption", "resize_32x32":"32$\times$32 PNG", "jpeg_q1":"JPEG Q1", "jpeg_q10":"JPEG Q10", "full":"Full image"}
    for row in size_summary.itertuples(index=False):
        lines.append(f"{fidelity_labels[str(row.fidelity)]} & {row.median_bytes:,.0f} & {row.p95_bytes:,.0f} & {row.mean_bytes:,.0f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (paper_output / "fidelity_sizes_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    voila_mean = summary[summary.policy == "voila"].mean(numeric_only=True)
    full_mean = summary[summary.policy == "full"].mean(numeric_only=True)
    trace_voila = trace_mean[trace_mean.policy == "voila"].set_index("deadline_s")
    trace_full = trace_mean[trace_mean.policy == "full"].set_index("deadline_s")
    macros = {
        "PopulationSize": int(summary[summary.policy == "voila"].n_queries.iloc[0]),
        "MeanVoilaAccuracy": f"{100*voila_mean.accuracy:.1f}", "MeanVoilaMB": f"{voila_mean.mean_bytes/1e6:.3f}",
        "MeanFullAccuracy": f"{100*full_mean.accuracy:.1f}", "MeanFullMB": f"{full_mean.mean_bytes/1e6:.3f}",
        "MeanByteReduction": f"{100*(1-voila_mean.mean_bytes/full_mean.mean_bytes):.1f}",
        "OneSecondVoilaDQA": f"{100*trace_voila.loc[1.0,'dqa']:.1f}", "OneSecondFullDQA": f"{100*trace_full.loc[1.0,'dqa']:.1f}",
        "CpuPFiftyMs": f"{json.loads((results_root/'systems'/'cpu_qwen235b_iso_fold01.json').read_text())['warm_p50_ms']:.2f}",
    }
    (paper_output / "paper_numbers.tex").write_text("\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()) + "\n", encoding="utf-8")
    manifest = {"generator": str(Path(__file__).resolve()), "generator_sha256": sha256_file(__file__),
        "results_root": str(results_root.resolve()), "size_source": str(size_source.resolve()), "size_source_sha256": sha256_file(size_source),
        "outputs": {path.name: sha256_file(path) for path in paper_output.iterdir()
                    if path.is_file() and path.name != "artifact_manifest.json"}}
    write_json(paper_output / "artifact_manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="Measure actual FloodNet payload sizes")
    measure.add_argument("--image_dir", required=True)
    measure.add_argument("--cached_csv", required=True)
    measure.add_argument("--output_csv", required=True)
    measure.add_argument("--caption_col", default="caption1_question_based")
    measure.add_argument("--payload_dir", default=None)

    intersection = sub.add_parser("intersection", help="Create a common image/question population")
    intersection.add_argument("--input_csvs", nargs="+", required=True)
    intersection.add_argument("--output_csv", required=True)

    enrich = sub.add_parser("enrich", help="Attach measured bytes to query-level routing records")
    enrich.add_argument("--query_records", required=True)
    enrich.add_argument("--sizes", required=True)
    enrich.add_argument("--output_csv", required=True)

    manifest = sub.add_parser("manifest", help="Build an exact-payload CellReplay manifest")
    manifest.add_argument("--query_records", required=True)
    manifest.add_argument("--sizes", required=True)
    manifest.add_argument("--output_csv", required=True)
    manifest.add_argument("--policy", default="exact_argmax_voila")

    deadlines = sub.add_parser("deadlines", help="Join TCP completion with cached correctness")
    deadlines.add_argument("--transfers_jsonl", required=True)
    deadlines.add_argument("--manifest", required=True)
    deadlines.add_argument("--output_json", required=True)

    mission = sub.add_parser("mission", help="Run mission-budget MCKP experiments")
    mission.add_argument("--query_records", required=True)
    mission.add_argument("--sizes", required=True)
    mission.add_argument("--output_json", required=True)
    mission.add_argument("--repeats", type=int, default=100)
    mission.add_argument("--seed", type=int, default=42)
    mission.add_argument("--bucket_bytes", type=int, default=1024)
    mission.add_argument("--legacy_query_records", default=None)
    mission.add_argument("--batch_sizes", nargs="+", type=int, default=[25, 50, 100])
    mission.add_argument("--budget_fractions", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75, 1.0])

    cpu = sub.add_parser("cpu", help="Benchmark a saved router artifact")
    cpu.add_argument("--artifact", required=True)
    cpu.add_argument("--questions_csv", required=True)
    cpu.add_argument("--output_json", required=True)
    cpu.add_argument("--repeats", type=int, default=30)
    cpu.add_argument("--core", type=int, default=0)
    cpu.add_argument("--max_questions", type=int, default=None)
    cpu.add_argument("--sample_seed", type=int, default=42)

    trace = sub.add_parser("trace-sim", help="Run recorded-trace uplink serialization simulation")
    trace.add_argument("--query-records", nargs="+", required=True)
    trace.add_argument("--sizes", required=True)
    trace.add_argument("--trace-root", required=True)
    trace.add_argument("--output-dir", required=True)
    trace.add_argument("--deadlines", nargs="+", type=float, default=[1, 5, 10])
    trace.add_argument("--start-offsets", type=int, default=100)
    trace.add_argument("--seed", type=int, default=2027)

    artifacts = sub.add_parser("paper-artifacts", help="Generate LaTeX tables and figures from frozen results")
    artifacts.add_argument("--results-root", required=True)
    artifacts.add_argument("--paper-output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "measure":
        _, metadata = measure_representation_sizes(
            args.image_dir, args.cached_csv, args.output_csv, args.caption_col, args.payload_dir
        )
        print(json.dumps(metadata["summary"], indent=2))
    elif args.command == "intersection":
        _, metadata = build_common_intersection(args.input_csvs, args.output_csv)
        print(json.dumps(metadata, indent=2))
    elif args.command == "enrich":
        _, metadata = enrich_query_records_with_sizes(args.query_records, args.sizes, args.output_csv)
        print(json.dumps(metadata, indent=2))
    elif args.command == "manifest":
        _, metadata = build_transfer_manifest(
            args.query_records, args.sizes, args.output_csv, args.policy
        )
        print(json.dumps(metadata, indent=2))
    elif args.command == "deadlines":
        _, payload = summarize_deadline_transfers(
            args.transfers_jsonl, args.manifest, args.output_json
        )
        print(f"Wrote {len(payload['rows'])} deadline rows")
    elif args.command == "mission":
        _, payload = run_mission_budget_experiment(
            args.query_records,
            args.sizes,
            args.output_json,
            repeats=args.repeats,
            seed=args.seed,
            bucket_bytes=args.bucket_bytes,
            legacy_query_records_csv=args.legacy_query_records,
            batch_sizes=tuple(args.batch_sizes),
            budget_fractions=tuple(args.budget_fractions),
        )
        print(f"Wrote {len(payload['aggregate'])} aggregate rows")
    elif args.command == "cpu":
        payload = benchmark_router_artifact(
            args.artifact,
            args.questions_csv,
            args.output_json,
            args.repeats,
            args.core,
            args.max_questions,
            args.sample_seed,
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "trace-sim":
        _, aggregate, metadata = run_trace_simulation(
            args.query_records,
            args.sizes,
            args.trace_root,
            args.output_dir,
            tuple(args.deadlines),
            args.start_offsets,
            args.seed,
        )
        print(json.dumps({"aggregate_rows": len(aggregate), "claim_scope": metadata["claim_scope"]}, indent=2))
    elif args.command == "paper-artifacts":
        manifest = generate_paper_artifacts(args.results_root, args.paper_output)
        print(json.dumps({"outputs": len(manifest["outputs"]), "paper_output": args.paper_output}, indent=2))


if __name__ == "__main__":
    main()
