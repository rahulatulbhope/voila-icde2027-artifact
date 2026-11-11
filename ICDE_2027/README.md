# VOILA ICDE 2027 submission package

This directory is the ICDE-specific companion to the main VOILA repository.
It keeps the edited ICDE manuscript separate from the IEEE PerCom master and
documents the reproducibility bundle shared through Google Drive.

## Paper

- `paper/icde2027_author_edited.pdf` is the current manuscript built from the
  latest `jayaramkr/voila-papers` ICDE source.
- The editable source and figures are included under `paper/source/`; build it
  with `pwsh -File ICDE_2027/paper/build_icde.ps1`.
- The PDF is 14 pages total: 12 pages of technical content followed by
  references and the AI-generated-content acknowledgement. ICDE 2027 excludes
  references and that acknowledgement from the 12-page limit and prohibits an
  appendix.
- The PDF includes an AI-generated-content acknowledgement. The stale text
  extraction in `C:\Users\Rahul\Downloads\icde2027.txt` contains a TODO, but it
  is not the source used for this package.
- The source snapshot is pinned to GitHub commit `ae78b3f` (2026-11-11).

## Artifact link

ICDE requires supplemental material to be provided through a standard,
openly accessible file-sharing URL. The supplied Drive file link is recorded
in `artifact/manifest.json` and should also be entered in the CMT
supplemental-material field:

```text
https://drive.google.com/file/d/1i6orLyJELF1ZFj-bpSZmMcZqAW60SOV7/view?usp=drive_link
```

The file must be shared as “Anyone with the link” (viewer). If it is a zip
bundle, extract it before using the restore script below. Do not include
credentials, API keys, or private model tokens.

## Google Drive bundle

Upload these repository-relative directories while preserving their paths:

```text
files/
raw_data/
model_artifacts/
scripts/model_artifacts/
results/percom2027/
results/percom2027_final/
results/percom2027_final_bytes/
results/percom2027_submission/
results/cache/
```

These are intentionally ignored by Git because they contain cached model
outputs, fitted artifacts, runtime caches, and large FloodNet imagery. The
checked-in code and paper-facing generated files remain in the repository.

After downloading or syncing the Drive bundle, restore it with:

```powershell
pwsh -File ICDE_2027/artifact/restore_artifacts.ps1 `
  -DriveBundleRoot 'D:\path\to\downloaded\VOILA_ICDE_2027_artifacts'
```

The restore script refuses a missing bundle root and prints the files it
restores. It does not delete existing files.

## Evidence and limits

The package reuses the frozen offline results documented in
`PERCOM_2027_RUN_HANDOFF.md` and the checked-in generated manifest. It does not
claim new VLM, Jetson, edge-energy, carbon, disaster-specific network, or live
caption-generation measurements. The public cellular traces are proxies, as
described in the handoff.

## Submission checklist

- [x] Edited ICDE PDF is preserved with a recorded SHA-256 hash.
- [x] Editable ICDE source and figures are included and rebuild successfully.
- [x] PerCom and ICDE paper masters remain independent.
- [x] Reproducibility bundle paths and restoration procedure are documented.
- [x] Record the supplied Google Drive file URL in the artifact manifest.
- [ ] Confirm all authors' ORCIDs in CMT.
- [ ] Confirm the Drive folder is accessible in an incognito/browser session.
- [ ] Upload the PDF and supplemental URL to the correct ICDE CMT round.
- [ ] Confirm no concurrent-submission or prior-publication restriction is
      violated.
- [ ] Upload the complete artifact bundle, including the ICDE-specific
      `generated_icde` outputs referenced by the paper, or remove claims that
      cannot be reproduced from the uploaded bundle.
