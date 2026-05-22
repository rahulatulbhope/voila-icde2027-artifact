#!/usr/bin/env python3
"""
Generate VOILA-compatible VQA output CSVs with Hugging Face models.

The offline VOILA experiments consume CSV files with one row per question and
answer/log-prob columns for each image fidelity. This script creates those CSVs
for dataset/model pairs by running HF-hosted vision-language models over:

  - caption text only
  - 32x32 resized image
  - JPEG quality 1 image
  - JPEG quality 10 image
  - original/full image

It checkpoints after every N rows and resumes from the same output file.
Large/gated models require a valid HF token and enough local GPU memory.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]

FIDELITIES = ["caption", "resize_32x32", "jpeg_q1", "jpeg_q10", "full"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    trust_remote_code: bool = True
    note: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    hf_id: Optional[str]
    split: str
    question_col: str
    gt_col: str
    image_col: str
    qid_col: Optional[str] = None
    image_id_col: Optional[str] = None
    gt_type: str = "single"
    caption_cols: Tuple[str, ...] = ()
    output_caption_col: str = "caption_set2"
    full_tag: str = "original"
    jpeg_q1_tag: str = "jpeg_q1"
    jpeg_q10_tag: str = "jpeg_q10"
    lpd_prefix: str = "log_prob_dist_"
    extra_cols: Tuple[str, ...] = ()
    hf_config: Optional[str] = None


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "pixtral": ModelSpec("mistralai/Pixtral-12B-2409", note="May need trust_remote_code."),
    "llava": ModelSpec("llava-hf/llava-1.5-7b-hf", trust_remote_code=False),
    "llama4": ModelSpec("meta-llama/Llama-4-Maverick-17B-128E-Instruct", note="Gated and very large."),
    "qwen7b": ModelSpec("Qwen/Qwen2.5-VL-7B-Instruct"),
    "qwen72b": ModelSpec("Qwen/Qwen2.5-VL-72B-Instruct", note="Very large."),
    "qwen235b": ModelSpec("Qwen/Qwen3-VL-235B-A22B-Instruct", note="Extremely large."),
}


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "vqav2": DatasetSpec(
        hf_id="HuggingFaceM4/VQAv2",
        split="validation",
        qid_col="question_id",
        image_id_col="image_id",
        question_col="question",
        gt_col="answers",
        image_col="image",
        gt_type="answers",
        caption_cols=("caption_set2", "blip_caption", "caption"),
        output_caption_col="caption_set2",
    ),
    "gqa": DatasetSpec(
        hf_id="lmms-lab/GQA",
        split="testdev",
        qid_col="question_id",
        image_id_col="image_id",
        question_col="question",
        gt_col="answer",
        image_col="image",
        caption_cols=("caption_set2", "caption2", "caption"),
        output_caption_col="caption_set2",
        extra_cols=("ground_truth_full",),
    ),
    "textvqa": DatasetSpec(
        hf_id="lmms-lab/textvqa",
        split="validation",
        qid_col="question_id",
        image_id_col="image_id",
        question_col="question",
        gt_col="answers",
        image_col="image",
        gt_type="semicolon_list",
        caption_cols=("caption_detailed", "caption_confident", "caption"),
        output_caption_col="caption_detailed",
    ),
    "floodnet": DatasetSpec(
        hf_id=None,
        split="test",
        qid_col=None,
        image_id_col="image_id",
        question_col="question",
        gt_col="ground_truth",
        image_col="image",
        caption_cols=("caption1", "caption2", "caption"),
        output_caption_col="caption1",
        extra_cols=("question_type",),
    ),
    "locomo": DatasetSpec(
        hf_id=None,
        split="test",
        qid_col="question_id",
        image_id_col="image_id",
        question_col="question",
        gt_col="gt_answer",
        image_col="image_path",
        caption_cols=("blip_caption", "caption"),
        output_caption_col="blip_caption",
        full_tag="orig",
        jpeg_q1_tag="jpeg_extreme",
        jpeg_q10_tag="jpeg_moderate",
        lpd_prefix="dist_",
        extra_cols=("image_path", "evidence"),
    ),
}


def progress_iter(items: Iterable[Any], **kwargs):
    try:
        from tqdm.auto import tqdm
    except Exception:
        return items
    return tqdm(items, dynamic_ncols=True, **kwargs)


def require_runtime_deps():
    missing = []
    for package in ("torch", "transformers", "PIL", "pandas"):
        try:
            __import__("PIL" if package == "PIL" else package)
        except Exception:
            missing.append(package)
    if missing:
        raise SystemExit(
            "Missing generation dependencies: "
            + ", ".join(missing)
            + ". Install with: pip install torch transformers pillow pandas datasets accelerate"
        )


def get_dataset_spec(args, dataset: str) -> DatasetSpec:
    spec = DATASET_REGISTRY[dataset]
    return DatasetSpec(
        hf_id=args.dataset_id or spec.hf_id,
        hf_config=args.dataset_config or spec.hf_config,
        split=args.split or spec.split,
        qid_col=args.qid_col or spec.qid_col,
        image_id_col=args.image_id_col or spec.image_id_col,
        question_col=args.question_col or spec.question_col,
        gt_col=args.gt_col or spec.gt_col,
        image_col=args.image_col or spec.image_col,
        gt_type=args.gt_type or spec.gt_type,
        caption_cols=tuple(args.caption_col or spec.caption_cols),
        output_caption_col=args.output_caption_col or spec.output_caption_col,
        full_tag=spec.full_tag,
        jpeg_q1_tag=spec.jpeg_q1_tag,
        jpeg_q10_tag=spec.jpeg_q10_tag,
        lpd_prefix=spec.lpd_prefix,
        extra_cols=spec.extra_cols,
    )


def get_model_spec(args, model: str) -> ModelSpec:
    base = MODEL_REGISTRY[model]
    if args.model_id:
        return ModelSpec(args.model_id, trust_remote_code=args.trust_remote_code, note="CLI override")
    return base


def select_pairs(args) -> List[Tuple[str, str]]:
    if args.pairs:
        pairs = []
        for item in args.pairs:
            if ":" not in item:
                raise ValueError(f"Invalid --pairs entry '{item}'. Use dataset:model.")
            dataset, model = item.split(":", 1)
            pairs.append((dataset.strip(), model.strip()))
        return pairs
    pairs = []
    for dataset in args.datasets:
        for model in args.models:
            if dataset == "locomo" and model not in {"pixtral", "qwen7b"} and not args.include_missing_pairs:
                continue
            pairs.append((dataset, model))
    return pairs


def fidelity_tags(spec: DatasetSpec) -> Dict[str, str]:
    return {
        "caption": spec.output_caption_col,
        "resize_32x32": "resize_32x32",
        "jpeg_q1": spec.jpeg_q1_tag,
        "jpeg_q10": spec.jpeg_q10_tag,
        "full": spec.full_tag,
    }


def output_path_for(args, dataset: str, model: str) -> Path:
    out_dir = Path(args.output_dir) / dataset
    name = args.output_name.format(dataset=dataset, model=model)
    return out_dir / name


def load_input_rows(args, dataset: str, spec: DatasetSpec) -> List[Dict[str, Any]]:
    if args.input_csv:
        import pandas as pd

        df = pd.read_csv(args.input_csv, low_memory=False)
        return df.to_dict("records")

    if not spec.hf_id:
        raise ValueError(
            f"{dataset} has no default HF dataset id in the registry. "
            "Pass --dataset_id or --input_csv with an image/image_path column."
        )

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit("Install datasets with: pip install datasets") from exc

    token = os.environ.get(args.hf_token_env) or None
    kwargs: Dict[str, Any] = {
        "split": spec.split,
        "trust_remote_code": args.dataset_trust_remote_code,
    }
    if token:
        kwargs["token"] = token
    if args.local_files_only:
        kwargs["download_mode"] = "reuse_dataset_if_exists"

    if spec.hf_config:
        ds = load_dataset(spec.hf_id, spec.hf_config, **kwargs)
    else:
        ds = load_dataset(spec.hf_id, **kwargs)

    if args.start:
        ds = ds.skip(args.start)
    if args.max_samples is not None:
        ds = ds.take(args.max_samples)
    elif args.end is not None:
        length = max(0, args.end - args.start)
        ds = ds.take(length)
    return [dict(row) for row in ds]


def slice_local_rows(args, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    start = max(0, args.start)
    end = args.end if args.end is not None else len(rows)
    rows = rows[start:end]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    return rows


def normalize_ground_truth(value: Any, gt_type: str) -> str:
    if value is None:
        return ""
    if gt_type == "answers":
        values = []
        if isinstance(value, dict):
            value = value.get("answer") or value.get("answers") or value.get("text") or value
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    item = item.get("answer") or item.get("text") or item.get("raw_answer")
                if item is not None and str(item).strip():
                    values.append(str(item).strip())
        elif value is not None:
            values.append(str(value).strip())
        if not values:
            return ""
        counts: Dict[str, int] = {}
        for item in values:
            counts[item] = counts.get(item, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]
    if gt_type == "semicolon_list":
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            value = value.get("answer") or value.get("answers") or value.get("text") or []
        if isinstance(value, (list, tuple)):
            vals = []
            for item in value:
                if isinstance(item, dict):
                    item = item.get("answer") or item.get("text") or item.get("raw_answer")
                if item is not None and str(item).strip():
                    vals.append(str(item).strip())
            return ";".join(vals)
    if isinstance(value, dict):
        for key in ("answer", "text", "raw_answer"):
            if key in value:
                return str(value[key])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("answer") or first.get("text") or first)
        return str(first)
    return str(value)


def row_value(row: Dict[str, Any], column: Optional[str], default: Any = "") -> Any:
    if not column:
        return default
    return row.get(column, default)


def find_caption(row: Dict[str, Any], spec: DatasetSpec) -> str:
    for col in spec.caption_cols:
        value = row.get(col)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            text = str(value).strip()
            if text:
                return text
    return ""


def load_image(row: Dict[str, Any], spec: DatasetSpec, image_root: Optional[Path]):
    from PIL import Image

    value = row.get(spec.image_col)
    if value is None and "image_path" in row:
        value = row.get("image_path")
    if value is None:
        raise ValueError(f"No image value found in column '{spec.image_col}'")

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        for key in ("path", "bytes", "url"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")

    text = str(value)
    if text.startswith(("http://", "https://")):
        with urllib.request.urlopen(text, timeout=60) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")

    path = Path(text)
    if not path.is_absolute() and image_root is not None:
        path = image_root / path
    return Image.open(path).convert("RGB")


def apply_fidelity(image, fidelity: str):
    if fidelity == "full":
        return image
    if fidelity == "resize_32x32":
        return image.resize((32, 32))
    if fidelity in {"jpeg_q1", "jpeg_q10"}:
        from PIL import Image

        quality = 1 if fidelity == "jpeg_q1" else 10
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality, optimize=False)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(f"Unsupported image fidelity: {fidelity}")


class HFGenerator:
    def __init__(self, model_spec: ModelSpec, args):
        import torch
        import transformers

        token = os.environ.get(args.hf_token_env) or None
        common: Dict[str, Any] = {
            "trust_remote_code": model_spec.trust_remote_code or args.trust_remote_code,
            "local_files_only": args.local_files_only,
        }
        if token:
            common["token"] = token

        self.processor = transformers.AutoProcessor.from_pretrained(model_spec.model_id, **common)
        dtype = self._dtype(args.dtype)
        model_kwargs = dict(common)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if args.device_map:
            model_kwargs["device_map"] = args.device_map
        if args.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        cls_names = ["AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"]
        last_error = None
        for cls_name in cls_names:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                self.model = cls.from_pretrained(model_spec.model_id, **model_kwargs)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Could not load model {model_spec.model_id}: {last_error}") from last_error

        self.model.eval()
        self.args = args
        self.torch = torch
        if not args.device_map and torch.cuda.is_available():
            self.model.to("cuda")

    def _dtype(self, dtype_name: str):
        if dtype_name == "auto":
            return None
        import torch

        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]

    @property
    def device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def build_chat_text(self, prompt: str, image: Any = None) -> str:
        if not hasattr(self.processor, "apply_chat_template"):
            return f"<image>\n{prompt}" if image is not None else prompt
        if image is None:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"<image>\n{prompt}" if image is not None else prompt

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        decoder = getattr(self.processor, "decode", None)
        if decoder is None and hasattr(self.processor, "tokenizer"):
            decoder = self.processor.tokenizer.decode
        if decoder is None:
            return ""
        return decoder(token_ids, skip_special_tokens=skip_special_tokens)

    def prepare_inputs(self, prompt: str, image: Any = None):
        text = self.build_chat_text(prompt, image)
        attempts = []
        if image is None:
            attempts = [
                lambda: self.processor(text=[text], return_tensors="pt", padding=True),
                lambda: self.processor(text, return_tensors="pt"),
            ]
        else:
            attempts = [
                lambda: self.processor(text=[text], images=[image], return_tensors="pt", padding=True),
                lambda: self.processor(images=[image], text=[text], return_tensors="pt", padding=True),
                lambda: self.processor(text=text, images=image, return_tensors="pt"),
            ]
        last_error = None
        for attempt in attempts:
            try:
                inputs = attempt()
                return {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Processor could not prepare inputs: {last_error}") from last_error

    def generate(self, prompt: str, image: Any = None) -> Dict[str, Any]:
        inputs = self.prepare_inputs(prompt, image)
        input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        gen_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if self.args.temperature is not None:
            gen_kwargs["temperature"] = self.args.temperature

        with self.torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        sequence = output.sequences[0]
        gen_ids = sequence[input_len:] if len(sequence) > input_len else sequence
        gen_id_list = gen_ids.tolist() if hasattr(gen_ids, "tolist") else list(gen_ids)
        answer = self.decode(gen_id_list, skip_special_tokens=True).strip()
        token_logprobs, token_rows, top2_margins = self.score_tokens(output, gen_ids)
        mean_logprob = float(sum(token_logprobs) / len(token_logprobs)) if token_logprobs else float("nan")
        mean_margin = float(sum(top2_margins) / len(top2_margins)) if top2_margins else float("nan")
        return {
            "answer": answer,
            "mean_logprob": mean_logprob,
            "token_logprobs": token_rows,
            "top2_margin": mean_margin,
        }

    def score_tokens(self, output, gen_ids) -> Tuple[List[float], List[Dict[str, Any]], List[float]]:
        token_logprobs: List[float] = []
        token_rows: List[Dict[str, Any]] = []
        top2_margins: List[float] = []
        try:
            transition = self.model.compute_transition_scores(
                output.sequences,
                output.scores,
                normalize_logits=True,
            )[0]
            gen_id_list = gen_ids.tolist() if hasattr(gen_ids, "tolist") else list(gen_ids)
            for token_id, logprob in zip(gen_id_list, transition.tolist()):
                token_logprobs.append(float(logprob))
                token_rows.append(
                    {
                        "token": self.decode([token_id], skip_special_tokens=False),
                        "logprob": float(logprob),
                    }
                )
        except Exception:
            pass

        try:
            for scores in output.scores:
                log_probs = self.torch.log_softmax(scores[0].float(), dim=-1)
                top2 = self.torch.topk(log_probs, k=2).values.tolist()
                top2_margins.append(float(top2[0] - top2[1]))
        except Exception:
            pass
        return token_logprobs, token_rows, top2_margins


def base_output_record(row: Dict[str, Any], spec: DatasetSpec, row_id: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {"_voila_row_id": row_id}
    if spec.qid_col:
        record["question_id"] = row_value(row, spec.qid_col, row_id)
    elif "question_id" in row:
        record["question_id"] = row["question_id"]
    if spec.image_id_col:
        record["image_id"] = row_value(row, spec.image_id_col, "")
    record["question"] = row_value(row, spec.question_col, "")
    gt_value = normalize_ground_truth(row_value(row, spec.gt_col, ""), spec.gt_type)
    if spec.gt_col == "gt_answer":
        record["gt_answer"] = gt_value
    elif spec.gt_type == "semicolon_list":
        record["ground_truth_answers"] = gt_value
    else:
        record["ground_truth"] = gt_value
    for col in spec.extra_cols:
        if col in row and col not in record:
            record[col] = row[col]
    caption = find_caption(row, spec)
    if caption:
        record[caption_text_col(spec.output_caption_col)] = caption
    return record


def caption_text_col(tag: str) -> str:
    if tag == "caption_set2":
        return "caption_set2_detailed"
    if tag == "caption1":
        return "caption1_question_based"
    if tag == "caption2":
        return "caption2_detailed"
    if tag == "caption3":
        return "caption3_confident_only"
    return tag


def write_records(path: Path, records: List[Dict[str, Any]]):
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(records).to_csv(tmp, index=False)
    tmp.replace(path)


def complete_record(record: Dict[str, Any], tags: Dict[str, str], fidelities: Sequence[str]) -> bool:
    for fidelity in fidelities:
        tag = tags[fidelity]
        value = record.get(f"ans_{tag}", "")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return False
    return True


def prompt_for_vqa(question: str) -> str:
    return (
        "Answer the visual question with the shortest correct answer. "
        "Do not explain.\nQuestion: "
        + str(question)
        + "\nAnswer:"
    )


def prompt_for_caption_vqa(question: str, caption: str) -> str:
    return (
        "You are answering a visual question using only this image caption.\n"
        f"Caption: {caption}\n"
        f"Question: {question}\n"
        "Answer with the shortest correct answer. Do not explain.\nAnswer:"
    )


def add_generation_result(record: Dict[str, Any], tag: str, prefix: str, result: Dict[str, Any]):
    record[f"ans_{tag}"] = result["answer"]
    record[f"prob_{tag}"] = result["mean_logprob"]
    record[f"{prefix}{tag}"] = json.dumps(result["token_logprobs"], ensure_ascii=False)
    record[f"top2_margin_{tag}"] = result["top2_margin"]


def add_error(record: Dict[str, Any], fidelity: str, exc: Exception):
    record[f"error_{fidelity}"] = f"{type(exc).__name__}: {exc}"


def generate_pair(args, dataset: str, model: str):
    require_runtime_deps()
    import pandas as pd

    spec = get_dataset_spec(args, dataset)
    model_spec = get_model_spec(args, model)
    tags = fidelity_tags(spec)
    output_path = output_path_for(args, dataset, model)
    image_root = Path(args.image_root) if args.image_root else None

    print(f"\n=== generate {dataset}:{model} ===")
    print(f"model_id={model_spec.model_id}")
    print(f"output={output_path}")

    rows = load_input_rows(args, dataset, spec)
    if args.input_csv:
        rows = slice_local_rows(args, rows)

    records: List[Dict[str, Any]] = []
    done_ids = set()
    record_by_id: Dict[int, Dict[str, Any]] = {}
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path, low_memory=False).to_dict("records")
        for record in existing:
            row_id = record.get("_voila_row_id")
            parsed_row_id = None
            try:
                parsed_row_id = int(row_id)
                record_by_id[parsed_row_id] = record
            except Exception:
                pass
            if row_id is not None and complete_record(record, tags, args.fidelities):
                if parsed_row_id is not None:
                    done_ids.add(parsed_row_id)
            records.append(record)
        print(f"resume: loaded {len(records)} records, {len(done_ids)} complete")

    generator = HFGenerator(model_spec, args)
    caption_generator: Optional[HFGenerator] = None
    start_time = time.time()
    iterable = progress_iter(enumerate(rows), total=len(rows), desc=f"{dataset}:{model}")
    since_save = 0

    for local_idx, row in iterable:
        row_id = args.start + local_idx
        if row_id in done_ids:
            continue

        record = record_by_id.get(row_id) or base_output_record(row, spec, row_id)
        question = str(record.get("question", ""))
        caption = str(record.get(caption_text_col(spec.output_caption_col), "") or "")
        if not caption and args.caption_source in {"auto", "blip"} and "caption" in args.fidelities:
            try:
                caption_args = copy.copy(args)
                caption_args.max_new_tokens = args.caption_max_new_tokens
                if caption_generator is None:
                    caption_generator = HFGenerator(
                        ModelSpec(args.caption_model, trust_remote_code=args.trust_remote_code),
                        caption_args,
                    )
                image = load_image(row, spec, image_root)
                caption = caption_generator.generate(args.caption_prompt, image=image)["answer"]
                record[caption_text_col(spec.output_caption_col)] = caption
            except Exception as exc:
                add_error(record, "caption_generation", exc)
                if args.fail_fast:
                    write_records(output_path, records + [record])
                    raise

        for fidelity in args.fidelities:
            tag = tags[fidelity]
            if str(record.get(f"ans_{tag}", "")).strip():
                continue
            try:
                if fidelity == "caption":
                    if args.caption_source == "none":
                        result = generator.generate(prompt_for_vqa(question), image=None)
                    else:
                        caption = str(record.get(caption_text_col(spec.output_caption_col), "") or "")
                        if not caption:
                            raise ValueError("No caption available for caption fidelity")
                        result = generator.generate(prompt_for_caption_vqa(question, caption), image=None)
                else:
                    image = load_image(row, spec, image_root)
                    image = apply_fidelity(image, fidelity)
                    result = generator.generate(prompt_for_vqa(question), image=image)
                add_generation_result(record, tag, spec.lpd_prefix, result)
            except Exception as exc:
                add_error(record, fidelity, exc)
                if args.fail_fast:
                    write_records(output_path, records + [record])
                    raise

        if row_id not in record_by_id:
            records.append(record)
            record_by_id[row_id] = record
        since_save += 1
        if since_save >= args.checkpoint_every:
            write_records(output_path, records)
            since_save = 0

    write_records(output_path, records)
    elapsed = time.time() - start_time
    print(f"wrote {len(records)} rows to {output_path} in {elapsed / 60.0:.2f} min")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VOILA-compatible HF VQA outputs.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY), choices=list(DATASET_REGISTRY))
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY), choices=list(MODEL_REGISTRY))
    parser.add_argument("--pairs", nargs="*", help="Explicit dataset:model pairs. Overrides --datasets/--models.")
    parser.add_argument("--include_missing_pairs", action="store_true", help="Include registry pairs not used in the paper, e.g. all LoCoMo models.")

    parser.add_argument("--output_dir", default=str(ROOT / "generated_hf_files"))
    parser.add_argument("--output_name", default="hf_{dataset}_{model}.csv")
    parser.add_argument("--input_csv", help="Local source CSV for a single dataset run.")
    parser.add_argument("--image_root", help="Root directory for relative image paths in local CSVs.")
    parser.add_argument("--dataset_id", help="Override HF dataset id for a single dataset run.")
    parser.add_argument("--dataset_config", help="Override HF dataset config/subset.")
    parser.add_argument("--split", help="Override HF split.")
    parser.add_argument("--model_id", help="Override HF model id for a single model run.")

    parser.add_argument("--question_col")
    parser.add_argument("--gt_col")
    parser.add_argument("--gt_type", choices=["single", "answers", "semicolon_list"])
    parser.add_argument("--image_col")
    parser.add_argument("--qid_col")
    parser.add_argument("--image_id_col")
    parser.add_argument("--caption_col", nargs="*", help="Caption columns to reuse, in priority order.")
    parser.add_argument("--output_caption_col", help="Answer-column tag for caption fidelity.")
    parser.add_argument("--caption_source", default="auto", choices=["auto", "reuse", "blip", "none"])
    parser.add_argument("--caption_model", default="Salesforce/blip-image-captioning-large")
    parser.add_argument("--caption_prompt", default="Describe the image in one concise sentence.")
    parser.add_argument("--caption_max_new_tokens", type=int, default=48)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail_fast", action="store_true", help="Abort on the first failed row/fidelity.")
    parser.add_argument("--fidelities", nargs="+", default=FIDELITIES, choices=FIDELITIES)

    parser.add_argument("--hf_token_env", default="HF_TOKEN")
    parser.add_argument("--trust_remote_code", action="store_true", help="Force trust_remote_code for model loading.")
    parser.add_argument("--dataset_trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device_map", default="auto", help="Transformers device_map. Use empty string to disable.")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device_map == "":
        args.device_map = None
    pairs = select_pairs(args)
    for dataset, model in pairs:
        generate_pair(args, dataset, model)


if __name__ == "__main__":
    main()
