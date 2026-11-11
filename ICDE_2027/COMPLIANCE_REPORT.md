# ICDE 2027 compliance review

Reviewed against the official ICDE 2027 research-track guidelines and the
latest `jayaramkr/voila-papers` commit `ae78b3fac43e0475d3e2f2ede8f69cd074260b9f`.

## Status: conditionally ready

### Verified

- IEEE conference format: the source uses `IEEEtran` in 10-point conference
  mode and compiles on US Letter paper.
- Length: the rebuilt PDF has 14 pages total. The technical content occupies
  12 pages; references and the AI-generated-content acknowledgement are outside
  the ICDE page limit.
- No appendix: no appendix section appears in the submitted PDF.
- AI disclosure: the acknowledgement names Claude Code and Codex, identifies
  Sections I--X for editorial assistance and Sections III--VII for
  implementation/script assistance, and states that authors reviewed the
  generated code and are responsible for correctness and originality.
- Single-blind review: author names and affiliations are present, consistent
  with ICDE's single-blind policy.
- Build: the direct `pdflatex -> bibtex -> pdflatex -> pdflatex` build succeeds
  with no unresolved citation or reference warnings.
- Visual review: title page, figures, section transitions, acknowledgement, and
  reference pages render without clipping or overlap.

### Not yet verified or incomplete

- Supplemental artifacts: the latest public repository contains the paper
  source and figures but not the `generated_icde` scripts, tables, audit files,
  or placement/OOD outputs referenced in the source comments. ICDE requires
  experiment papers to provide the artifacts needed to reproduce the reported
  results.
- Public artifact URL: a Google Drive file URL has been supplied and recorded
  in the manifest, but public viewer access and the file's contents still need
  to be tested from a clean/private browser session.
- CMT metadata: ORCIDs, conflict-of-interest declarations, track selection,
  concurrent-submission status, and final author order must be checked in CMT
  and with the coauthors.

## Required actions before submission

1. Add the missing ICDE-specific generated artifacts and their producing
   scripts to the public supplemental bundle, or remove the claims that cannot
   be reproduced from the bundle.
2. Test the supplied `google_drive_url` from a private browser session and
   confirm that it contains the complete supplemental bundle.
3. Confirm all CMT metadata and the ICDE novelty/concurrent-submission rules.
