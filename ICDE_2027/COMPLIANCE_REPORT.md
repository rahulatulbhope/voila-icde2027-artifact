# ICDE 2027 compliance review

Reviewed against the official ICDE 2027 research-track guidelines and the
submission PDF packaged in this repository.

## Status: conditionally ready

### Verified

- IEEE conference format: the packaged PDF uses the required two-column
  conference layout on US Letter paper.
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
- PDF integrity: the packaged submission PDF has a recorded SHA-256 hash in
  `artifact/manifest.json`.
- Visual review: title page, figures, section transitions, acknowledgement, and
  reference pages render without clipping or overlap.

### Not yet verified or incomplete

- Supplemental artifacts: the ICDE-specific generated scripts, tables, audit
  files, and placement/OOD outputs still need to be present in the downloaded
  Drive bundle for full reproduction.
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
