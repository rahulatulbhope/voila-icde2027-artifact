# AGENTS.md

## Scope

These instructions apply to the entire `VOILA_code` repository. The project combines Python experiment code, cached result summaries, a LaTeX paper, and generated paper artifacts.

## Source of truth

- The main offline runner is `scripts/run_all_experiments.py`.
- PerCom systems and artifact generation live in `scripts/percom_experiments.py`.
- The PerCom master is `VOILA_Percom/percom2027.tex`; `VOILA_Percom/build_percom.ps1` is the supported build entry point.
- Read `PERCOM_2027_RUN_HANDOFF.md` and `PerCom_2027_plan.md` before changing PerCom experiments or interpreting existing results.
- Treat `VOILA_Percom/generated/artifact_manifest.json` as the integrity manifest for the checked-in PerCom generated outputs.

## Working rules

1. Preserve unrelated user changes. Do not reset, checkout, or clean the worktree destructively.
2. Do not commit secrets, `.env` files, datasets, downloaded checkpoints, raw imagery, runtime caches, or local `.claude` worktrees. The root `.gitignore` is intentional; review `git status` and `git diff --cached` before committing.
3. Do not claim new VLM, Jetson, edge-energy, carbon, or disaster-specific network measurements unless the experiment and its inputs actually exist. The PerCom plan explicitly limits those claims.
4. Keep the ACL and PerCom paper masters independently buildable. Do not silently rewrite one while repairing the other.
5. When changing routing or evaluation logic, update the relevant tests and run the focused test suite.
6. Keep generated paper-facing files synchronized with the source and record provenance in the manifest or handoff notes when applicable.

## Validation

Run the regression tests with:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Inspect available PerCom commands with:

```powershell
python scripts/percom_experiments.py --help
```

Build the submission PDF with:

```powershell
pwsh -File VOILA_Percom/build_percom.ps1 -Mode submission
```

Only run data-dependent experiments when the external inputs in `inputs/README.md` are available. Avoid regenerating large outputs merely to “refresh” them; use the existing manifest and run handoff first.

## Review checklist

- `git diff` contains only the intended change.
- No ignored secret or data path has been force-added.
- Tests pass, or the final note names the missing dependency/input.
- Paper builds or the reason it cannot build is recorded.
- Generated artifacts and their source/provenance remain consistent.
