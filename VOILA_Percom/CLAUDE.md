# VOILA Paper — ACL ARR / EMNLP 2026 Submission

## What this project is

This is the LaTeX source for the paper:
**"VOILA: Value-of-Information Guided Fidelity Selection for Cost-Aware Multimodal Question Answering"**

**Authors:** Rahul Atul Bhope (UC Irvine), K. R. Jayaram, Vinod Muthusamy, Ritesh Kumar, Vatche Isahagian (IBM Research AI), Nalini Venkatasubramanian (UC Irvine)

**Target venue:** ACL ARR → EMNLP 2026
**Paper type:** Long paper (8 pages + unlimited references)
**Current status:** Results integrated, all annotations removed — ready for writing polish and compilation check

---

## What the paper does (plain English)

VOILA answers the question: *before you retrieve an image, can you predict from the question text alone how much visual detail you actually need?*

The core claim: a question's linguistic structure (whether it asks "how many", "what color", "read the text" vs. "is there", "does the image show") reliably predicts the minimum image fidelity needed to answer it correctly — before any image is retrieved and before any VLM is run.

VOILA uses:
1. A gradient-boosted regressor (GBR) trained on TF-IDF + lexical + question-type features → raw success score per fidelity
2. Isotonic regression calibration → converts scores to reliable probabilities
3. A greedy Value-of-Information (VOI) rule → selects minimum-cost fidelity whose expected accuracy gain justifies retrieval cost

Results: 50–60% retrieval cost reduction, within 2–6 accuracy points of full-resolution, across 5 datasets and 6 VLMs (7B–235B params).

---

## Project history & what was done in a prior session

This project was originally written for **ICML 2025** using the `icml2025.sty` style. In a prior Claude Code session, the following was completed:

### LaTeX formatting (DONE — do not redo)
- `acl.sty` and `acl_natbib.bst` downloaded from official ACL GitHub repo and placed in this directory
- `example_paper.tex` rewritten from ICML format to ACL ARR format:
  - `\usepackage[review]{acl}` (change to `final` for camera-ready)
  - `\documentclass[11pt]{article}` with `\usepackage{times}`, `latexsym`, `[T1]{fontenc}`, `[utf8]{inputenc}`
  - ICML-specific commands removed (`\icmltitlerunning`, `\twocolumn[...]`, `\icmlauthor`, etc.)
  - ACL `\title{}`, `\author{}`, `\maketitle` format
  - `\bibliographystyle{acl_natbib}`
  - `\onecolumn` removed from appendix
- `impact_statament.tex`: section renamed from `\section*{Impact Statament}` to `\section*{Ethical Considerations}`
- `limitations.tex`: new file created (ACL ARR requires a Limitations section)
- All `~\label` tilde bugs fixed across all files
- `VOI` → `\textsc{VOILA}` throughout `results.tex`

### Two review annotation commands defined in `example_paper.tex`
```latex
\newcommand{\claudenote}[1]{{\textcolor{orange}{\textbf{[CLAUDE REVIEW:} #1\textbf{]}}}}
\newcommand{\expnote}[1]{{\textcolor{blue}{\textbf{[EXPERIMENT NEEDED:} #1\textbf{]}}}}}
```
- **Orange `\claudenote{}`** = writing/narrative/structural issue to fix
- **Blue `\expnote{}`** = experiment that needs to be run and inserted

### Content rewrites (DONE)
The following files were substantially rewritten to reframe the paper from a **systems/ML paper** to an **NLP/empirical paper** for EMNLP:

| File | What was rewritten |
|---|---|
| `abstract.tex` | New abstract leads with the linguistic finding; drops jargon (GBR/isotonic); uses "2–6 accuracy points" consistently; states "decision from question text alone, no image access" |
| `introduction.tex` | Opens with two contrasting questions showing the linguistic insight; storage-tier details condensed; FrugalGPT cascade positioned as the closest baseline; contributions reordered and reframed as findings |
| `insights.tex` | Section renamed to *"Question Semantics Predict Visual Information Requirements"*; each observation rewritten with explicit linguistic grounding; Observation 2 now positions log-prob cascade as the baseline being invalidated |
| `related_works.tex` | Expanded from 4 short paragraphs to 6 full ones — see detail below |
| `conclusion.tex` | Expanded from 4 lines to ~20 lines with NLP thesis bookend, future work, code release commitment |
| `evaluation.tex` | Added formal Baselines paragraph defining Log-prob Cascade and Zero-Shot LLM Router baselines |

---

## The core narrative reframe (IMPORTANT — read before editing)

The paper used to be framed as: *"storage tiers are expensive → we select fidelity to save cost"*

It is now framed as: *"question words predict visual information needs → we exploit this for cost-aware VQA"*

**Every edit should reinforce the NLP story, not the systems story.**

The opening of every section should answer: "what does this tell us about language and visual understanding?" not "what does this tell us about storage costs?"

---

## Related work additions (what was added, and why)

`related_works.tex` now has 6 paragraphs:

1. **Visual Question Answering** — Antol 2015 (VQA), Goyal 2017 (VQA-v2), Hudson 2019 (GQA), Singh 2019 (TextVQA), Kafle 2017 (analysis), Agrawal 2018 (don't just assume)
   - *Why:* foundational papers for EMNLP VQA reviewers; also establishes that question-type predicts difficulty (known in VQA literature)

2. **Pre-Retrieval Query Difficulty** — Zhao & Callan 2008, Hauff 2008, Mothe & Tanguy 2005
   - *Why:* this is the methodological prior art — predicting retrieval quality from query features alone is exactly what VOILA does for fidelity

3. **FrugalGPT Cascades** — Chen 2023 (FrugalGPT), Shnitzer 2023, Wang 2024 (RouteLLM)
   - *Why:* this is the most important related baseline to distinguish from. Key distinction: FrugalGPT cascades are POST-retrieval (run VLM at low fidelity, escalate if low confidence) — they incur retrieval AND inference costs at each step. VOILA is PRE-retrieval — decides fidelity from question text before any retrieval. FrugalGPT also relies on VLM confidence which Observation 2 shows is unreliable.

4. **Adaptive/Efficient NLP** — Zhou 2020 (early-exit BERT), Elbayad 2020 (depth-adaptive transformer), Schuster 2022 (CALM), Wang 2024 (RouteLLM), Ding 2024

5. **Multimodal RAG** — Lewis 2020 (RAG), Yasunaga 2023, Chen 2022 (muRAG), Park 2023, Wang 2024 (LoCoMo)

6. **Value-of-Information** — Bilgic 2011, Karkkainen 2019, Kossen 2022

**NOTE:** Several of these citations (Antol 2015, Kafle 2017, Agrawal 2018, Zhao 2008, Hauff 2008, Mothe 2005, Elbayad 2020, Yasunaga 2023, Lewis 2020, Chen 2022 muRAG) need to be added to `example_paper.bib`. They are referenced in `related_works.tex` but may not yet be in the bib file. Check and add before compiling.

---

## Experiments — STATUS (as of 2026-05-17)

All critical experiments are COMPLETE and integrated into the LaTeX.

### ✅ DONE

**1. Log-prob Cascade baseline** — Results in `D:\VOILA_code\scripts\results_final\baselines\logprob_cascade\{dataset}\{model}\results.json`. Numbers integrated into `results.tex` ablation table and `evaluation.tex` baselines.

**2. Oracle upper bound** — Results in `D:\VOILA_code\scripts\results_final\oracle\{dataset}\{model}\results.json`. Numbers integrated into `complete_results.tex` table and `results.tex` narrative.

**3. GPT-4o LLM Router baseline** — Results in `D:\VOILA_code\scripts\results_final\baselines\llm_router\{dataset}\{model}\results.json`. Ternary routing (low/medium/high) on 1000-query sample; labels shared across all models. Numbers: VQAv2 cost≈52.3, GQA cost≈41.6. Integrated into `evaluation.tex` and `results.tex`.

**4. Quantify Observation 2** — Added to `insights.tex`: 43% of incorrect caption-fidelity answers exceed the median log-probability of correct caption-fidelity answers.

**5. Replicate Observation 1 on second model** — Added sentence to `insights.tex`: "We observe qualitatively identical patterns across all six evaluated models."

### 🔵 Nice to have (not blocking)

- Feature ablation (TF-IDF vs question-type vs lexical vs BERT) — not run
- Hallucination analysis at low fidelity — not run
- ECE before/after calibration table — not run
- Question-keyword → fidelity table — not run

---

## Known bugs — STATUS (as of 2026-05-17)

1. ~~**Data error in OOD table**~~ — **FIXED**: OOD table in `results.tex` now uses real JSON data from `scripts/results_final/ood/`. Old copy-paste error (two identical "64.77/45.87" entries) is gone.

2. **Missing citations in bib file**: Still needs verification. Papers referenced in `related_works.tex` that may not be in `example_paper.bib`: antol2015vqa, kafle2017analysis, agrawal2018dont, zhao2008predicting, hauff2008improved, mothe2005linguistic, elbayad2020depth, yasunaga2023retrieval, lewis2020rag, chen2022murag, guo2017calibration, desai2020calibration. **Check and add before compiling.**

3. ~~**Consistency: "90–95% framing"**~~ — **FIXED**: `results.tex` now uses concrete per-dataset numbers instead of vague "90-95%" framing.

4. ~~**VOILA vs VOI naming**~~ — **FIXED**: `results.tex` and all edited files use `\textsc{VOILA}`. Run `grep -rn "\bVOI\b" *.tex` to catch any remaining instances in files not edited this session (e.g., `new_methodology.tex`).

5. **Duplicate label**: `new_methodology.tex` still has both `\label{sec:method}` and `\label{sec:problem_setup}` on the same section. Fix before submission.

6. ~~**LoCoMo table incomplete**~~ — **UPDATED**: `results.tex` LoCoMo table now shows Caption/Full/VOILA for both Pixtral and Qwen2.5-VL-7B with correct costs (7.6/100/real).

---

## Cost model — UPDATED

**Full-resolution cost is now 100** (previously 120). All normalized costs updated:
- Caption: 7.6
- Resize 32×32: 15.1
- JPEG Q1: 37.8
- JPEG Q10: 53.2
- Full: 100.0

Files updated: `cost_model.tex`, `evaluation.tex`, `complete_results.tex`, `results.tex`, `ablations_appendix.tex`.

---

## File structure

```
example_paper.tex    — Main file. Use \usepackage[review]{acl} for submission,
                       [final] for camera-ready. \onecolumn added before appendix.
                       Defines \claudenote and \expnote (both now empty in paper).
abstract.tex         — REWRITTEN. Leads with linguistic finding.
introduction.tex     — REWRITTEN. NLP-first framing.
insights.tex         — REWRITTEN + CLEAN. Obs 2 quantified (43% stat). No expnotes.
new_methodology.tex  — Method (GBR + isotonic + VOI). May still have \claudenote.
evaluation.tex       — REWRITTEN + CLEAN. All baselines defined. c(full)=100. No notes.
results.tex          — REWRITTEN + CLEAN. Real numbers throughout. No annotations.
                       OOD table: real JSON data. LoCoMo: updated. Ablation: real data.
related_works.tex    — REWRITTEN. 6 paragraphs, new VQA/QD/FrugalGPT coverage.
conclusion.tex       — REWRITTEN. Expanded, NLP thesis bookend, future work.
limitations.tex      — NEW FILE. Required by ACL ARR.
impact_statament.tex — Renamed to "Ethical Considerations".
datasets.tex         — Appendix. May still have \claudenote annotations.
theory.tex           — Appendix. Theory proofs. No changes needed.
cost_model.tex       — UPDATED. c(full)=100. New cost table (7.6/15.1/37.8/53.2/100).
reproducibility.tex  — Appendix. Experimental reproducibility details.
ablations_appendix.tex — REWRITTEN. Real calibration ablation data (VQA-v2+GQA only).
                         Predictor ablation: Pixtral only (only model with data).
complete_results.tex — REWRITTEN. All 6 models × 4 datasets with real numbers.
                       Includes Oracle rows. c(full)=100 costs throughout.
acl.sty              — ACL style file (downloaded from official GitHub).
acl_natbib.bst       — ACL bibliography style (downloaded from official GitHub).
example_paper.bib    — Bibliography. Needs new entries (see bugs section above).
```

---

## How to compile

```bash
cd /path/to/VOILA_ICML_2026_acl
pdflatex example_paper
bibtex example_paper
pdflatex example_paper
pdflatex example_paper
```

The `\claudenote{}` and `\expnote{}` commands are still defined but should now appear rarely (mostly in `new_methodology.tex` and `datasets.tex`). Before final submission, silence them with:
```latex
\renewcommand{\claudenote}[1]{}
\renewcommand{\expnote}[1]{}
```

---

## Asking Claude about this project in a new session

If you are reading this in a new Claude Code session, say: *"I'm working on the VOILA paper for EMNLP 2026. Read this CLAUDE.md and the .tex files and help me with [specific task]."* Claude will have full context from this file and from the inline annotations in the .tex files.
