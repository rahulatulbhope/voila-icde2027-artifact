# VOILA PerCom 2027 Revision and Experiment Plan

Last updated: 2026-08-30

Transferability note: the concrete continuation state, commands, completed
artifacts, and caveats are recorded in `PERCOM_2027_RUN_HANDOFF.md`. If this
run is continued in GitHub Copilot or another GPT session, start there before
rerunning experiments.

## Scope and non-negotiable constraints

The PerCom paper centers on FloodNet post-disaster UAV imagery. The operational setting uses a fixed 15-question checklist. Generic captions are generated once during ingestion and cached; caption-generation compute is outside per-query routing and transmission measurements. VQA-v2, GQA, and TextVQA are breadth datasets, and a leave-question-template-out FloodNet evaluation is the query-generalization stress test.

No new VLM inference is permitted. No Jetson, edge-energy, carbon, or live on-device caption-generation result will be claimed. Network results use CellReplay's public cellular field traces as proxies rather than disaster-specific traces. Deadline failures replace the previously proposed corrupted-JPEG/VLM experiment.

PerCom allows nine pages of technical content, including appendices, plus one references-only page in IEEE 10-point two-column US Letter format. The complete appendix remains in the extended working source but is not exempt from the submission limit.

## 1. IEEE format checkpoint

- `VOILA_Percom/percom2027.tex` is the new IEEE master; the ACL master remains independently buildable.
- Class: `\documentclass[10pt,conference,compsocconf]{IEEEtran}`.
- The PerCom master is anonymous, has no custom headers or page numbers, uses the IEEE bibliography style, `subfig` with `caption=false`, and an IEEE-compatible figure wrapper for the existing algorithm.
- Build switches:
  - `extendedappendix`: complete working appendix.
  - `shownotes`: reviewer notes visible or silent.
  - `submission`: anonymous, notes off, extended appendix off.
- `VOILA_Percom/build_percom.ps1` implements `pdflatex -> bibtex -> pdflatex -> pdflatex` without relying on Perl/`latexmk`.
- The extended checkpoint is US Letter, contains all existing content and appendix inputs, and builds without unresolved citations or references.
- Appendix-off submission mode is the rewrite target. Stale appendix pointers must be removed as the PerCom-specific narrative replaces the inherited draft; they must not be hidden with fabricated labels.

## 2. Correctness and reproducibility repair

Primary routing is the exact global decision

\[
f^*(q)=\arg\max_f\left[\hat p_f(q)-\lambda c_f\right],
\]

with deterministic ties resolved toward lower byte cost. The old sequential `lambda/tau` procedure is retained only as `legacy_sequential`.

The experiment runner uses four disjoint group-safe populations inside each outer fold: router training, calibration fitting, operating-point validation, and test. The operating point is the cheapest validation policy reaching at least 95% of the best fixed-fidelity validation accuracy. Test accuracy-cost frontiers are reported but never used for selection.

Split definitions:

- FloodNet primary: five folds grouped by `image_id`; no scene crosses train/test.
- FloodNet stress: five folds grouped by normalized question template; no template crosses train/test.
- General VQA: question-ID grouping, with duplicate image/question counts recorded.

Cross-model results may only be aggregated on the common `(image_id, question)` population. Per-model sample counts and source hashes are mandatory. The current six-model FloodNet intersection is materialized under `results/percom2027/common_intersection/`. The byte-cost FloodNet population is `results/percom2027/common_intersection/floodnet_all_models_measured.csv`; it removes the three cached image IDs absent from the official archive: `7411.JPG`, `7419.JPG`, and `7443.JPG`.

The TF-IDF audit evaluates 500, 1,000, 2,000, and 5,000 features and selects the smallest vocabulary within 0.5 absolute validation-accuracy points of the best configuration, with validation cost reported for matched-cost review. FloodNet's fixed checklist has fewer than 500 unique lexical features, so the current selected default is 500; broader-dataset audits must still be reported separately.

Calibration candidates are uncalibrated scores, independent isotonic calibration, temperature calibration as a legacy comparator, and shared-slope Platt scaling with fidelity-specific intercepts. The primary calibrator is the lowest-validation-regret method whose per-fidelity validation ECE is no more than 0.01 worse than independent isotonic. No cross-fidelity monotonicity or 2-D isotonic construction is used.

Required metrics and records:

- Per-fidelity ECE and Brier score.
- Pairwise fidelity-gain error.
- Selection regret against the cached oracle.
- Cost at matched validation accuracy and the complete test frontier.
- Query-level probabilities, utilities, selected fidelity, cost/bytes, cached correctness, fold, model, and dataset.
- Image-clustered bootstrap 95% intervals on FloodNet; question-level bootstrap elsewhere.
- Run manifests with input/code hashes, seeds, fold definitions, population size, cost profile, policy, and calibration method.

## 3. Measured representations

Download the official 13.1 GB FloodNet archive into ignored `raw_data/`, then use `scripts/percom_experiments.py measure` to recreate and persist the exact transfer payloads:

- cached caption encoded as UTF-8;
- 32x32 image encoded as PNG after the same PIL resize used by cached VLM generation;
- JPEG quality 1 and quality 10 with the same PIL JPEG implementation and `optimize=False`;
- the original source image bytes for full resolution.

Report the distribution, median, and p95 bytes for each fidelity. Transmitted bytes are the primary optimization cost. Normalized historical costs appear only in an appendix comparison. Use the `enrich` command to attach per-fidelity and selected measured bytes to every query record.

## 4. Mission-budget optimization

For batch sizes 25, 50, and 100, evaluate budgets equal to 10%, 25%, 50%, 75%, and 100% of all-full bytes. Repeat 100 seeded batches per setting.

`scripts/percom_experiments.py mission` solves the multiple-choice knapsack problem by dynamic programming with conservative 1 KiB buckets and validates the solver against brute force in unit tests. Each query receives exactly one representation and actual transmitted bytes cannot exceed the budget.

Baselines: budgeted VOILA, best feasible fixed fidelity, legacy sequential policy, value-per-byte greedy upgrades, and the cached oracle. Report effective accuracy, actual bytes, maximum budget violation, and regret against the cached oracle.

## 5. CPU and cellular deadline evaluation

CPU reference:

- Intel i7-12700KF, pinned to one logical CPU.
- Artifact/model size, cold artifact load, cold first query, warmed p50/p95/p99 latency, and peak process memory.
- Describe this as a commodity-CPU reference, not an embedded, energy, or Jetson measurement.

Cellular replay:

- Use WSL2 and the official CellReplay NSDI 2025 repository and traces.
- The public archive supplies T-Mobile driving, stationary-good, and weak-signal traces plus Verizon walking, stationary-good, and crowded traces; report those exact names instead of inventing unavailable carrier/scenario combinations.
- `scripts/cellreplay/` contains the privileged WSL setup step, reliable-TCP sender/receiver, exact-payload manifest workflow, retransmission accounting, and replay wrapper.
- Replay actual selected payload files with CellReplay loss rates 0%, 1%, 5%, and 10%.
- Record payload bytes, TCP completion time, retransmissions, and completion by 0.5/1/2/5/10 second deadlines.
- Join cached VLM correctness only after a complete representation arrives. Report completion rate, conditional accuracy, effective accuracy with misses counted as failures, and abstention rate.
- Compare full resolution, best fixed fidelity, exact-argmax VOILA, and budgeted VOILA.

CellReplay compilation requires the local user's WSL `sudo` password. Until that one privileged setup command is run, the trace archive and harness remain ready but no measured replay result may be claimed.

## 6. Verification

Automated tests cover exact global argmax, lower-byte tie breaking, missing fidelity predictions, infeasible mission budgets, dynamic-program/brute-force equivalence, budget compliance, zero image overlap, zero template overlap, and common cross-model population construction.

Before paper freeze, rerun both LaTeX masters independently, audit anonymity and units, check every table against result hashes, verify no VLM service was called, and confirm the submission PDF uses at most nine technical pages plus one references-only page.

## 7. Schedule

- Aug 29-30: freeze IEEE format checkpoint and correctness tests.
- Aug 30-Sep 1: exact argmax, nested group-safe splits, vocabulary and calibration audits, confidence intervals, common-population reruns.
- Sep 1-2: complete FloodNet download, reconstruct representations, measure bytes.
- Sep 2-3: run 100-repeat mission-budget study.
- Sep 3-5: install CellReplay with user-authorized WSL `sudo`, run CPU/network/deadline studies, register by Sep 4.
- Sep 6: freeze result JSON, plots, and tables.
- Sep 7-8: rewrite around FloodNet UAV/checklist operations; remove LoCoMo and AAAI review notes from submission mode only.
- Sep 9: anonymity, evidence, units, and cached-provenance audit.
- Sep 10: final 9+1-page PDF and upload test.
- Sep 11: submit.

Target technical-page allocation: introduction/motivation 1.25; method 1.75; system/cost model 1; evaluation 1; results 2.5; related work/limitations/conclusion 1.5. Use any remaining space only for essential reproducibility detail.
