# Crucible Docs Index

Crucible has two faces: the **current, working implementation** (the scripts in this repo) and the **target product surface** (the future `crucible` CLI and packaged workflows). Start with the path that matches what you're doing today.

## Choose your path

If you want to **run it right now (repo scripts are truth)**, start here:
- **[CURRENT_IMPLEMENTATION.md](CURRENT_IMPLEMENTATION.md)** — the authoritative flags and commands (auto-generated from `--help`)
- **[QUICKSTART.md](QUICKSTART.md)** — the fastest end-to-end run (baseline train + frozen + organic eval)
- **[PLAYBOOK.md](PLAYBOOK.md)** — the validated experimental sequence (ratio sweeps, step sweeps, decision logic)

If you want to **understand what's happening and why**, go here:
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — modules, dataflow, where each script fits
- **[METRICS.md](METRICS.md)** — the canonical formulas and statistical reporting (single source of truth)
- **[GUARDRAILS.md](GUARDRAILS.md)** — dilution detection, regression thresholds, fingerprints, change control

If you want to **wire in your own data**, start here:
- **[SCHEMAS.md](SCHEMAS.md)** — canonical events, bundles, eval sets, scoreboards, capsules
- **[CONFIG.md](CONFIG.md)** — recommended config layout + how it maps to the current scripts

If you want the **product interface and commercialization view**, go here:
- **[CLI.md](CLI.md)** — the target `crucible` CLI spec (design contract, not current truth)
- **[COMMERCIAL.md](COMMERCIAL.md)** — packaging, licensing, enterprise features, deployment options
- **[README.md](README.md)** — narrative overview and why the two-truth split exists

## The one rule that prevents confusion

When flags or commands disagree: **CURRENT_IMPLEMENTATION.md wins.**
It is regenerated from the scripts' `--help` output, so it cannot drift.

## Fast decision cheat sheet

If you care about **retaining known hard cases**, watch **Frozen Tier3** (frozen eval).
If you care about **generalizing to new hard cases**, watch **Organic Holdout**.

Healthy progress means you're not "buying" one at the expense of the other unless you meant to.

## Maintenance (for contributors)

- After changing any script CLI: regenerate **CURRENT_IMPLEMENTATION.md**
- Before merging doc changes: run the docs linter
- CI automatically checks for drift on every push (`.github/workflows/docs-ci.yml`)

```bash
python tools/extract_script_flags.py
python tools/docs_lint_crucible.py
```

**Optional:** Install the pre-commit hook to catch drift before pushing:
```bash
cp tools/pre-commit-docs.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```
