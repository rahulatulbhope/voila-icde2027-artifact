# Hugging Face VQA Generation

Use `scripts/generate_hf_vqa_outputs.py` when you need to create fresh VOILA
input CSVs by running Hugging Face vision-language models. The main experiment
runner remains offline; it reads the generated CSVs and never calls a VLM.

## What It Produces

For each dataset-model pair, the script writes one CSV with the same style of
columns used by `scripts/run_all_experiments.py`:

- base fields such as `question_id`, `image_id`, `question`, and ground truth
- `ans_<tag>` for every fidelity
- `prob_<tag>` as the mean generated-token log probability
- `log_prob_dist_<tag>` or `dist_<tag>` as token-level log-probability JSON
- `top2_margin_<tag>` as the mean top-1 versus top-2 log-prob margin

The fidelity tags match the paper pipeline:

- VQAv2/GQA/TextVQA/FloodNet: `original`, `resize_32x32`, `jpeg_q1`,
  `jpeg_q10`, and the dataset caption tag
- LoCoMo: `orig`, `resize_32x32`, `jpeg_extreme`, `jpeg_moderate`,
  and `blip_caption`

If the source rows do not already include a caption column, the script can
generate the missing caption with `Salesforce/blip-image-captioning-large`.
This is controlled by `--caption_source`; the default `auto` reuses an existing
caption when present and falls back to BLIP when it is missing.

## Default Pairs

The model registry includes:

- `pixtral`: `mistralai/Pixtral-12B-2409`
- `llava`: `llava-hf/llava-1.5-7b-hf`
- `llama4`: `meta-llama/Llama-4-Maverick-17B-128E-Instruct`
- `qwen7b`: `Qwen/Qwen2.5-VL-7B-Instruct`
- `qwen72b`: `Qwen/Qwen2.5-VL-72B-Instruct`
- `qwen235b`: `Qwen/Qwen3-VL-235B-A22B-Instruct`

Some of these models are gated or very large. Set `HF_TOKEN` before running and
use a server with enough GPU memory.

## Run Examples

Smoke test one pair:

```bash
python scripts/generate_hf_vqa_outputs.py \
  --pairs vqav2:qwen7b \
  --max_samples 25 \
  --output_dir generated_hf_files \
  --checkpoint_every 5
```

Run the paper-style pairs:

```bash
python scripts/generate_hf_vqa_outputs.py \
  --datasets vqav2 gqa textvqa floodnet locomo \
  --models pixtral llava llama4 qwen7b qwen72b qwen235b \
  --output_dir generated_hf_files \
  --checkpoint_every 10 \
  --resume \
  --caption_source auto
```

Run a local dataset CSV with image paths:

```bash
python scripts/generate_hf_vqa_outputs.py \
  --pairs floodnet:qwen7b \
  --input_csv /path/to/floodnet_questions.csv \
  --image_root /path/to/floodnet/images \
  --image_col image_path \
  --question_col question \
  --gt_col ground_truth \
  --output_dir generated_hf_files
```

Use a different Hugging Face dataset or model without editing code:

```bash
python scripts/generate_hf_vqa_outputs.py \
  --pairs gqa:pixtral \
  --dataset_id your-org/your-gqa-dataset \
  --split validation \
  --model_id your-org/your-vlm \
  --question_col question \
  --gt_col answer \
  --image_col image
```

## Checkpointing

The script writes directly to the target CSV after every `--checkpoint_every`
rows. With `--resume` enabled, completed rows are skipped. If a single
fidelity fails, the row records `error_<fidelity>` and the run continues unless
`--fail_fast` is passed.

Generated files are ignored by Git. Store the completed CSVs in Google Drive or
copy the selected files into `files/{dataset}` when you want
`scripts/run_all_experiments.py` to consume them.
