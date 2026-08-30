# PerCom 2027 run handoff

This file is the transferable state for continuing the VOILA PerCom 2027
experiments without relying on Codex chat history.

Last updated: 2026-08-30.

## Workspace

- Root: `D:\VOILA_code`
- Manuscript repo: `D:\VOILA_code\VOILA_Percom`
- PerCom master: `D:\VOILA_code\VOILA_Percom\percom2027.tex`
- Build helper: `D:\VOILA_code\VOILA_Percom\build_percom.ps1`
- Main runner: `D:\VOILA_code\scripts\run_all_experiments.py`
- PerCom systems/byte tooling: `D:\VOILA_code\scripts\percom_experiments.py`
- CellReplay harness: `D:\VOILA_code\scripts\cellreplay`

## Completed implementation

- Added an IEEE PerCom master without modifying the ACL master.
- Added PerCom build switches: `extendedappendix`, `shownotes`, and `submission`.
- Repaired routing to exact utility argmax:
  `argmax_f [p_hat_f(q) - lambda * cost_f]`.
- Retained the old sequential policy only as a labeled legacy ablation.
- Added nested train/calibration/validation/test folds so operating-point
  selection is validation-only.
- Added grouped FloodNet primary splits by `image_id`.
- Added FloodNet leave-template-out stress splits by normalized question
  template.
- Added common `(image_id, question)` population filtering and duplicate checks
  for cross-model aggregation.
- Added query-level records with per-fidelity probabilities, utilities, costs,
  selected fidelity, correctness, fold, dataset, model, calibration, split,
  seed, and cost unit.
- Added calibration comparison among uncalibrated, independent isotonic, and
  shared-slope Platt with fidelity intercepts.
- Added TF-IDF vocabulary audit and selected the current default of 500
  features.
- Added measured-byte tooling, mission-budget MCKP tooling, CPU benchmark
  tooling, CellReplay TCP harness, and tests.

## Completed data/artifacts

Common populations:

- `results\percom2027\common_intersection\floodnet_all_models.csv`
  - 1,448 shared FloodNet image/question rows.
- `results\percom2027\common_intersection\floodnet_all_models_measured.csv`
  - 1,445 shared rows with official image bytes available.
  - Excluded official-image-missing IDs: `7411.JPG`, `7419.JPG`, `7443.JPG`.
- `results\percom2027\common_intersection\vqav2_all_models.csv`
  - 21,388 shared rows.
- `results\percom2027\common_intersection\gqa_all_models.csv`
  - 10,234 shared rows.
- `results\percom2027\common_intersection\textvqa_all_models_clean.csv`
  - 6,750 shared rows.
  - Excludes duplicated key `5813950ba766b4c3 / what year was this coin made?`.

FloodNet official images:

- Archive: `raw_data\FloodNet_full.zip`
- Size: 13,057,835,433 bytes.
- Extracted root: `raw_data\FloodNet_full\FloodNet-Supervised_v1.0`
- Original image counts: train 1,445; val 450; test 448.

Measured FloodNet payload sizes:

- CSV: `results\percom2027_final\systems\floodnet_payload_sizes.csv`
- Summary: `results\percom2027_final\systems\floodnet_payload_sizes.summary.json`
- Payload directory: `raw_data\percom_payloads`
- Median bytes:
  - caption: 188
  - resize_32x32: 2,125
  - jpeg_q1: 213,991
  - jpeg_q10: 432,120
  - full: 5,195,608
- Mean bytes used for routing utility:
  - caption: 281.1501730103806
  - resize_32x32: 2140.81660899654
  - jpeg_q1: 217575.94394463667
  - jpeg_q10: 445479.32387543254
  - full: 5520774.3501730105

Primary byte-cost FloodNet results:

- `results\percom2027_final_bytes\main\floodnet\<model>\voila_results.json`
- `results\percom2027_final_bytes\main\floodnet\<model>\query_records.csv`
- `results\percom2027_final_bytes\main\floodnet\<model>\query_records_enriched.csv`

FloodNet leave-template-out byte-cost stress results:

- `results\percom2027_final_bytes\floodnet_stress\floodnet\<model>\voila_results.json`
- `results\percom2027_final_bytes\floodnet_stress\floodnet\<model>\query_records.csv`

General breadth results, normalized-cost only:

- `results\percom2027_final\main\vqav2\<model>\voila_results.json`
- `results\percom2027_final\main\gqa\<model>\voila_results.json`
- `results\percom2027_final\main\textvqa\<model>\voila_results.json`

Byte-cost calibration and legacy ablations:

- `results\percom2027_final_bytes\ablations\calibration\floodnet\<model>\`
- Legacy query records are named `legacy_query_records.csv` in each model folder.

## Commands used for final byte-cost FloodNet runs

Common measured FloodNet population:

```powershell
python -c "import pandas as pd, json, hashlib; from pathlib import Path; src=Path('results/percom2027/common_intersection/floodnet_all_models.csv'); out=Path('results/percom2027/common_intersection/floodnet_all_models_measured.csv'); missing={'7411.JPG','7419.JPG','7443.JPG'}; df=pd.read_csv(src,dtype=str); clean=df[~df['image_id'].isin(missing)].copy(); out.parent.mkdir(parents=True,exist_ok=True); clean.to_csv(out,index=False); meta={'source_common_csv':str(src.resolve()),'source_common_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'measured_sizes_csv':str(Path('results/percom2027_final/systems/floodnet_payload_sizes.csv').resolve()),'excluded_missing_image_ids':sorted(missing),'source_rows':int(len(df)),'rows':int(len(clean)),'unique_images':int(clean['image_id'].nunique()),'unique_questions':int(clean['question'].nunique())}; out.with_suffix('.summary.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))"
```

Measured-byte FloodNet main:

```powershell
python scripts\run_all_experiments.py --data_dir files --datasets floodnet --models pixtral llava llama4 qwen7b qwen72b qwen235b --experiments main --common_keys_csv results\percom2027\common_intersection\floodnet_all_models_measured.csv --n_cv_folds 5 --bootstrap_samples 1000 --calibration none --cost_unit bytes --cost caption=281.1501730103806 resize_32x32=2140.81660899654 jpeg_q1=217575.94394463667 jpeg_q10=445479.32387543254 full=5520774.3501730105 --lambda_grid 0 1e-10 2e-10 5e-10 1e-9 2e-9 5e-9 1e-8 2e-8 5e-8 1e-7 2e-7 --output_dir results\percom2027_final_bytes --n_jobs 3
```

Measured-byte FloodNet leave-template-out stress:

```powershell
python scripts\run_all_experiments.py --data_dir files --datasets floodnet --models pixtral llava llama4 qwen7b qwen72b qwen235b --experiments floodnet_stress --common_keys_csv results\percom2027\common_intersection\floodnet_all_models_measured.csv --n_cv_folds 5 --bootstrap_samples 1000 --calibration none --cost_unit bytes --cost caption=281.1501730103806 resize_32x32=2140.81660899654 jpeg_q1=217575.94394463667 jpeg_q10=445479.32387543254 full=5520774.3501730105 --lambda_grid 0 1e-10 2e-10 5e-10 1e-9 2e-9 5e-9 1e-8 2e-8 5e-8 1e-7 2e-7 --output_dir results\percom2027_final_bytes --n_jobs 3
```

Byte-cost calibration and legacy ablation:

```powershell
python scripts\run_all_experiments.py --data_dir files --datasets floodnet --models pixtral llava llama4 qwen7b qwen72b qwen235b --experiments ablation_calibration --common_keys_csv results\percom2027\common_intersection\floodnet_all_models_measured.csv --n_cv_folds 5 --bootstrap_samples 1000 --calibration none --cost_unit bytes --cost caption=281.1501730103806 resize_32x32=2140.81660899654 jpeg_q1=217575.94394463667 jpeg_q10=445479.32387543254 full=5520774.3501730105 --lambda_grid 0 1e-10 2e-10 5e-10 1e-9 2e-9 5e-9 1e-8 2e-8 5e-8 1e-7 2e-7 --output_dir results\percom2027_final_bytes --n_jobs 3 --no_save_models
```

Enrich query records:

```powershell
$models = @('pixtral','llava','llama4','qwen7b','qwen72b','qwen235b')
foreach ($m in $models) {
  python scripts\percom_experiments.py enrich --query_records results\percom2027_final_bytes\main\floodnet\$m\query_records.csv --sizes results\percom2027_final\systems\floodnet_payload_sizes.csv --output_csv results\percom2027_final_bytes\main\floodnet\$m\query_records_enriched.csv
}
```

Primary mission-budget run:

```powershell
python scripts\percom_experiments.py mission --query_records results\percom2027_final_bytes\main\floodnet\qwen235b\query_records_enriched.csv --sizes results\percom2027_final\systems\floodnet_payload_sizes.csv --legacy_query_records results\percom2027_final_bytes\ablations\calibration\floodnet\qwen235b\legacy_query_records.csv --output_json results\percom2027_final_bytes\mission\qwen235b.json --repeats 100 --seed 2027 --bucket_bytes 1024
```

The mission command now writes `qwen235b.csv` incrementally and resumes by
skipping already completed `(batch_size, repeat, budget_fraction)` settings.
For safer continuation, run the cheaper settings first:

```powershell
python scripts\percom_experiments.py mission --query_records results\percom2027_final_bytes\main\floodnet\qwen235b\query_records_enriched.csv --sizes results\percom2027_final\systems\floodnet_payload_sizes.csv --legacy_query_records results\percom2027_final_bytes\ablations\calibration\floodnet\qwen235b\legacy_query_records.csv --output_json results\percom2027_final_bytes\mission\qwen235b.json --repeats 100 --seed 2027 --bucket_bytes 1024 --batch_sizes 25 50
```

Then continue the 100-question setting:

```powershell
python scripts\percom_experiments.py mission --query_records results\percom2027_final_bytes\main\floodnet\qwen235b\query_records_enriched.csv --sizes results\percom2027_final\systems\floodnet_payload_sizes.csv --legacy_query_records results\percom2027_final_bytes\ablations\calibration\floodnet\qwen235b\legacy_query_records.csv --output_json results\percom2027_final_bytes\mission\qwen235b.json --repeats 100 --seed 2027 --bucket_bytes 1024 --batch_sizes 100
```

## Current active/next work

As of this handoff, the primary Qwen235B mission-budget run may still be
running or partially complete. Check:

```powershell
Get-ChildItem results\percom2027_final_bytes\mission -Filter qwen235b.*
```

If `qwen235b.csv` exists before `qwen235b.json`, the mission run is partially
complete and the same command can be rerun to resume. The smoke-test file
`qwen235b_smoke.json` is not the final result.

Then regenerate tables:

```powershell
python scripts\run_all_experiments.py --data_dir files --datasets floodnet --models pixtral llava llama4 qwen7b qwen72b qwen235b --experiments tables --output_dir results\percom2027_final_bytes --no_save_models
```

Run tests:

```powershell
python -m py_compile scripts\run_all_experiments.py scripts\percom_experiments.py
pytest tests\test_percom_revision.py
```

Run the CPU benchmark on the final Qwen235B fold-1 artifact:

```powershell
python scripts\percom_experiments.py cpu --artifact model_artifacts\main_auto_nested_validation\floodnet\qwen235b\fold01_GBR_none_tfidf_plus_struct_tfidf500_seed43.joblib --questions_csv results\percom2027_final_bytes\main\floodnet\qwen235b\query_records.csv --output_json results\percom2027_final_bytes\systems\cpu_router_qwen235b_fold01.json --repeats 30 --max_questions 200 --sample_seed 2027
```

Completed bounded CPU benchmark:

- Output: `results\percom2027_final_bytes\systems\cpu_router_qwen235b_fold01.json`
- Artifact size: 237,436 bytes.
- Cold load: 15.69 ms.
- Cold first query: 4.93 ms.
- Warm latency over 200 sampled questions x 30 repeats:
  - p50: 3.72 ms
  - p95: 5.21 ms
  - p99: 5.95 ms
- Peak process memory: 160,157,696 bytes.
- Scope: commodity Intel i7-12700KF CPU reference only.

## CellReplay status

- Repository cloned to `external\cellreplay`.
- Public traces are available from that repository.
- Harness scripts are in `scripts\cellreplay`.
- WSL2 is available, but CellReplay build/install is blocked by sudo password.
- Do not claim CellReplay network/deadline results until setup and trace replay
  have actually run.

Setup command, to run in WSL with a human entering the sudo password:

```powershell
wsl.exe -d Ubuntu -- bash /mnt/d/VOILA_code/scripts/cellreplay/setup_wsl.sh /mnt/d/VOILA_code/external/cellreplay
```

After setup, generate a transfer manifest from the final enriched Qwen235B
records and run `scripts\cellreplay\run_cellreplay_tcp.sh` inside WSL.

## Important caveats

- No new VLM calls have been made or are permitted by the current assumptions.
- Caption generation compute is outside per-query transmission cost because
  captions are assumed generated once and cached at ingestion.
- FloodNet byte-cost routing uses mean bytes as the expected utility cost;
  mission-budget optimization uses per-image measured bytes.
- General VQA-v2/GQA/TextVQA runs are breadth evidence and still use normalized
  legacy costs; do not present their costs as transmitted bytes.
- The official FloodNet archive does not contain three cached images:
  `7411.JPG`, `7419.JPG`, `7443.JPG`.
- Do not claim Jetson, edge energy, carbon, live caption generation, corrupted
  JPEG VQA accuracy, or disaster-specific cellular traces.
