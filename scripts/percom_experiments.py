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
    timestamps = []
    with open(trace_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.replace(",", " ").split()
            try:
                timestamps.append(float(fields[0]))
            except (ValueError, IndexError):
                continue
    if not timestamps:
        raise ValueError("Trace contains no packet opportunities")
    values = np.asarray(timestamps, dtype=float)
    if np.nanmax(values) > 1000:
        values /= 1000.0
    return values - values.min()


def replay_deadline(payload_bytes, timestamps, loss_rate=0.0, mtu=1500, seed=42):
    """Deterministic preflight model of reliable delivery over packet slots.

    Lost packets are re-enqueued, approximating reliable transport.  Publication
    results must come from the CellReplay TCP harness; this function is for unit
    tests, trace selection, and expected-duration checks only.
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


if __name__ == "__main__":
    main()
