# ICDE 2027 submission review

This checklist is specific to the supplied ICDE manuscript and its artifact
package. It is not a replacement for the official ICDE instructions.

## Verified locally

- The supplied PDF uses a two-column IEEE-style layout.
- The PDF contains 14 pages total.
- The first 12 pages contain the technical paper; the remaining pages contain
  references and the AI-generated-content acknowledgement.
- No appendix heading was found in the supplied PDF.
- The PDF contains an AI-generated-content acknowledgement naming Claude Code
  and Codex and describing editing, grammar, and code/script assistance.
- The paper and the PerCom source remain in separate directories/masters.

## Required before submitting

- Put the real public Google Drive URL in `ICDE_2027/artifact/manifest.json`.
- Download the Drive bundle from a clean machine and run the restore script.
- Verify the restored bundle is readable and the hashes in
  `VOILA_Percom/generated/artifact_manifest.json` still match the checked-in
  paper-facing outputs.
- Test the Drive URL in a private/incognito browser window.
- Enter ORCIDs for every author in CMT.
- Confirm the authors, title, novelty, and concurrent-submission status with
  all coauthors.
- Submit only the PDF to CMT; provide the artifact URL as supplemental
  material according to the ICDE form.

## Current limitation

The edited manuscript was supplied only as a PDF. The package does not claim
that this PDF can be rebuilt from a matching ICDE LaTeX source. If the authors
need a source-rebuildable ICDE branch, the edited `.tex`, bibliography, style
files, and figure sources must be supplied separately.
