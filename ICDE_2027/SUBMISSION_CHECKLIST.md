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
  and Codex, identifying Sections I--X for editorial assistance and
  Sections III--VII for implementation/script assistance.
- Editable paper source is intentionally omitted from the artifact repository.

## Required before submitting

- Confirm the supplied Google Drive file URL is public and remains recorded in
  `ICDE_2027/artifact/manifest.json`.
- Download the Drive bundle from a clean machine and run the restore script.
- Verify the restored bundle is readable and matches the paths in
  `ICDE_2027/artifact/manifest.json`.
- Test the Drive URL in a private/incognito browser window.
- Enter ORCIDs for every author in CMT.
- Confirm the authors, title, novelty, and concurrent-submission status with
  all coauthors.
- Submit only the PDF to CMT; provide the artifact URL as supplemental
  material according to the ICDE form.
- Include the ICDE-specific `generated_icde` scripts, tables, and audit outputs
  referenced by the paper, or remove unsupported claims before submission.

## Current limitation

The ICDE-specific `generated_icde` scripts/tables and audit outputs must be
present in the supplemental bundle, or unsupported claims must be removed
before submission.
