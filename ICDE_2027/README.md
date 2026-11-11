# VOILA ICDE 2027 submission package

This directory is the ICDE-specific companion to the main VOILA repository.
It keeps the edited ICDE manuscript separate from the IEEE PerCom master and
documents the reproducibility bundle that should be shared through a public
Google Drive folder.

## Paper

- `paper/icde2027_author_edited.pdf` is the supplied edited manuscript.
- The PDF is 14 pages total: 12 pages of technical content followed by
  references and the AI-generated-content acknowledgement. ICDE 2027 excludes
  references and that acknowledgement from the 12-page limit and prohibits an
  appendix.
- The PDF includes an AI-generated-content acknowledgement. The stale text
  extraction in `C:\Users\Rahul\Downloads\icde2027.txt` contains a TODO, but it
  is not the source used for this package.
- No LaTeX source for this edited PDF was present in the supplied local files;
  this package therefore preserves the PDF as the submission artifact rather
  than pretending it can be rebuilt from source.

## Artifact link

ICDE requires supplemental material to be provided through a standard,
openly accessible file-sharing URL. The actual Google Drive URL was not
present in the repository or the supplied PDF, so the manifest intentionally
contains `null` until the public folder URL is supplied:

```text
ICDE_2027/artifact/manifest.json
```

Set the folder to “Anyone with the link” (viewer), paste its URL into the
manifest, and use that same URL in the CMT supplemental-material field. Do not
make the folder editable or include credentials, API keys, or private model
tokens.

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
- [x] PerCom and ICDE paper masters remain independent.
- [x] Reproducibility bundle paths and restoration procedure are documented.
- [ ] Replace the manifest's `google_drive_url` with the real public Drive URL.
- [ ] Confirm all authors' ORCIDs in CMT.
- [ ] Confirm the Drive folder is accessible in an incognito/browser session.
- [ ] Upload the PDF and supplemental URL to the correct ICDE CMT round.
- [ ] Confirm no concurrent-submission or prior-publication restriction is
      violated.
