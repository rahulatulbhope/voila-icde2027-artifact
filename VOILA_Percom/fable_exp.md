# fable_exp.md — Experiments & Concept Changes for an A+ AAAI Submission

Companion to the dark-green `\fablenote{}` annotations in the .tex files.
The notes cover *writing*; this file covers *new experiments and conceptual
repositioning*. Items are ordered by (impact ÷ effort). "P0" = blocking for
A+ tier, "P1" = strongly recommended, "P2" = nice to have.

---

## A. Concept changes (no new data collection)

### A1. Reframe from NLP paper to resource-bounded-reasoning paper [P0]
AAAI is a broad AI venue. The current EMNLP framing ("question semantics
predict visual needs") should become the *enabling finding* inside a
decision-theoretic frame: **cost-sensitive information acquisition / VOI
under uncertainty**. Concretely:
- Abstract & intro lead with "when should an AI system pay for higher-fidelity
  perception?", cite Howard (1966) VOI and the active feature-acquisition
  literature (Greiner et al. 2002; Shim et al. 2018), then present the
  linguistic predictability result as the surprise that makes the problem
  tractable pre-retrieval.
- Promote the theory (Bayes-optimality of the greedy rule, regret bounded by
  calibration error) from the appendix into a short "Theoretical Guarantees"
  subsection in §3 with one theorem statement (proof stays in appendix).
  AAAI rewards visible theory; EMNLP tolerates it.

### A2. Single scalar metric to rank methods [P0]
Every comparison is currently (Acc, Cost) pairs, which cannot rank methods
when neither dominates — and on VQA-v2 the GPT-4o Router has *higher
accuracy* than VOILA. Adopt one of:
- **AUACC**: area under the accuracy–cost curve as λ sweeps (preferred —
  needs only re-running the VOI rule at ~10 λ values on already-collected
  per-fidelity correctness labels; no new VLM calls).
- Accuracy at matched cost / cost at matched accuracy.
Report it in Table 1 and use it in the feature-ablation table (where the
current "best Pareto in bold" choice is otherwise indefensible).

### A3. Reposition vs. visual-token-compression literature [P1]
Modern VLMs already do adaptive resolution internally (AnyRes in LLaVA-NeXT,
Qwen2-VL dynamic resolution, token pruning/merging). A reviewer WILL ask
"why not just let the VLM downsample?" The answer — those methods still pay
full *retrieval* cost and need the image in hand — must be stated explicitly
in related work and intro, with citations.

---

## B. New experiments (cheap: reuse existing per-fidelity correctness labels)

### B1. Pareto frontier figure with λ-sweep [P0]
Restore the commented-out Pareto figure in results.tex. For each
dataset×model, sweep λ over ~10 values and plot the VOILA curve against all
baseline *points*. This is the single figure that defuses the "10 points
below full-res" objection: it shows VOILA dominates at every operating
point, including the GPT-4o Router's. **Zero new VLM inference needed** —
all per-fidelity correctness labels already exist; only the routing rule
re-runs.

### B2. Statistical significance everywhere [P0]
Bootstrap 95% CIs over evaluation queries for every Acc/Cost cell in
Table 1 and the OOD table; paired bootstrap test for VOILA vs. each
baseline. Pure post-processing of existing prediction logs. AAAI reviewers
treat unbarred tables as a soft reject signal.

### B3. ECE/Brier before–after calibration table, all 6 models [P1]
Listed as "nice to have" in CLAUDE.md; for AAAI it backs the central claim
"calibration is the central enabler." One table: raw GBR vs. isotonic vs.
temperature scaling × 6 models, ECE + downstream routing cost. Reuses
existing held-out scores.

### B4. Log-prob AUC for all six models [P1]
Observation 2 currently cites only the worst model (AUC 0.338). Report the
escalation AUC for all 6 models (one number each) to show the failure is
systematic, not cherry-picked. Already computable from cascade logs.

### B5. Question-keyword → fidelity interpretability table [P1]
Top-k TF-IDF features per fidelity decision with their learned weights
(e.g., "read", "say" → full; "is", "there" → caption). Makes the linguistic
claim *visible* and gives the paper a memorable artifact. Computable from
trained GBRs directly.

### B6. Error-mode qualitative table [P1]
2–3 real example questions for each of the three error modes
(over-escalation / under-escalation / irreducible), with chosen vs. oracle
fidelity. Pure log mining.

---

## C. New experiments (moderate: new VLM inference or training runs)

### C7. Sentence-embedding feature variant of VOILA [P1]
The BERT *Router* baseline exists, but a reviewer will ask the sharper
question: "what if VOILA itself used MiniLM/BERT embeddings instead of
TF-IDF, keeping calibration + VOI?" Run VOILA with embedding features as a
feature-ablation row. If TF-IDF matches embeddings, the "surface linguistic
structure suffices" claim gets much stronger; if embeddings win, you've
found free accuracy. Either outcome improves the paper. (~1 training run
per model×dataset; no new VLM calls.)

### C8. Hallucination analysis at low fidelity [P1]
Quantify *what kind* of errors low fidelity induces: % of caption-fidelity
errors that are fluent-but-ungrounded (language-prior answers) vs.
abstentions. This supplies the *mechanism* for Observation 2 (confident
wrong answers = prior-driven fluency) and connects to the
hallucination literature — a hot AAAI topic. Needs answer-level annotation
on a few hundred sampled errors (GPT-4o-as-judge is acceptable with a
human-verified subsample).

### C9. Cross-VLM predictor transfer [P1]
Train the GBR+calibrator on Model A's correctness labels, deploy for Model
B. If routing transfers across VLMs even partially, the per-VLM-calibration
limitation largely dissolves and the method becomes far more practical
(model-as-a-service). One table: 6×6 transfer matrix on VQA-v2 (36 cheap
routing evaluations; no new VLM inference — labels exist for all 6 models).

### C10. LoCoMo on all six VLMs [P2]
The LoCoMo column currently averages 2 models while the others average 6.
Either run the remaining 4 models or clearly de-emphasize the column.

### C11. Multi-seed variance [P2]
3–5 seeds for GBR training + data splits; report std. Cheap (training is
21 s/config) and closes the "single-run results" objection.

---

## D. Stretch experiments (new capability; only if time permits)

### D12. Video proof-of-concept [P2]
One small experiment selecting frame-rate/resolution for video QA
(e.g., a 200-question subset of a video QA benchmark) would convert the
"modality-agnostic" claim from speculation to evidence and substantially
widen the paper's claimed scope — a classic A+ move at AAAI.

### D13. Continuous-fidelity VOI [P2]
Replace the discrete tier set with JPEG quality as a continuous knob and a
monotone acquisition function. Turns a stated limitation into a
contribution. Risky timeline-wise; do only after P0/P1 items.

### D14. Abstention-aware variant [P2]
When max calibrated probability is below a floor even at full resolution,
predict "unanswerable" instead of retrieving. Connects to selective
prediction, reduces irreducible-failure cost (13–15% of queries), and adds
a clean extra result.

---

## E. Submission mechanics checklist (AAAI-specific)

- [ ] Convert from `acl.sty` to AAAI two-column style (7 pages + refs for
      the main track); re-check all table/figure widths after conversion.
- [ ] AAAI reproducibility checklist (mandatory at submission).
- [ ] Verify all `related_works.tex` citations exist in `example_paper.bib`
      (known gap per CLAUDE.md: antol2015vqa, zhao2008predicting, etc.).
- [ ] Reconcile number inconsistencies: 50–65% vs 50–60% cost reduction;
      inline numbers vs. Table 1 (35.1 vs 35.2 etc.).
- [ ] Silence all annotation macros (`\fablenote`, `\claudenote`,
      `\expnote`, …) before generating the submission PDF.
- [ ] "Results and Discussions" → "Results and Discussion".

---

## Priority summary

| Priority | Items | New VLM compute? |
|---|---|---|
| P0 | A1, A2, B1, B2 | No |
| P1 | A3, B3, B4, B5, B6, C7, C8, C9 | Minimal (C8 judging only) |
| P2 | C10, C11, D12, D13, D14 | Yes |

Doing only the P0 row already addresses the three most likely rejection
reasons: (1) "accuracy drops 10 points," (2) "router baseline is more
accurate," (3) "no significance testing."
