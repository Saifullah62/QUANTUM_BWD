# QUANTUM_BWD

> **ARCHIVE ONLY**
>
> This repository is frozen for historical reference. All active development has moved to:
>
> **https://github.com/Saifullah62/crucible**
>
> Do not submit PRs or issues here. The crucible repo contains the clean, production-ready harness without generated artifacts.

---

Quantum-inspired embedding model with semantic phase layers and the Crucible evaluation harness.

## Overview

QUANTUM_BWD implements a contrastive learning approach for embedding models with:

- **Semantic Phase Layers**: Quantum-inspired transformations for polysemy disambiguation
- **Lindblad Dynamics**: Decoherence modeling for sense separation
- **Crucible Harness**: Production-grade evaluation with tier-stratified metrics

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with curriculum learning
python -m crucible.train --bundles data/bundles.jsonl --steps 10000

# Evaluate on frozen + organic holdout
python -m crucible.eval --eval evals/frozen.jsonl --results-root results/

# Build production capsule (3 seeds, median selection)
python -m crucible.capsule \
    --bundles data/bundles.jsonl \
    --frozen-eval evals/frozen.jsonl \
    --organic-eval evals/organic.jsonl \
    --output-dir capsules/v1
```

## Project Structure

```
QUANTUM_BWD/
├── qllm/                    # Core model implementation
│   ├── core/                # Config and model architecture
│   ├── layers/              # Semantic phase, Lindblad, qualia layers
│   ├── training/            # Trainer and dataset loaders
│   └── evaluation/          # Benchmarks and analysis
├── scripts/                 # Training and evaluation scripts
│   ├── train_curriculum_v3.py   # Three-pool tier3 training
│   └── ...
├── experiments/             # Experiment runners and capsule builder
│   ├── build_production_capsule.py
│   ├── eval_capsules.py
│   └── ...
├── crucible/                # Module wrapper for CLI
├── paradigm_factory/        # Bundle generation and data collection
├── docs/crucible/           # Full documentation suite
└── tools/                   # Linter and truth extractor
```

## Documentation

Start with the [Crucible Docs Index](docs/crucible/INDEX.md):

| Doc | Purpose |
|-----|---------|
| [CURRENT_IMPLEMENTATION.md](docs/crucible/CURRENT_IMPLEMENTATION.md) | Authoritative CLI flags (auto-generated) |
| [QUICKSTART.md](docs/crucible/QUICKSTART.md) | Fastest end-to-end run |
| [PLAYBOOK.md](docs/crucible/PLAYBOOK.md) | Validated experimental sequence |
| [ARCHITECTURE.md](docs/crucible/ARCHITECTURE.md) | Modules and dataflow |
| [METRICS.md](docs/crucible/METRICS.md) | Canonical formulas (single source of truth) |
| [SCHEMAS.md](docs/crucible/SCHEMAS.md) | Data formats for bundles, evals, scoreboards |

**The one rule**: When flags or commands disagree, **CURRENT_IMPLEMENTATION.md wins**.

## Validated Recipe

The production recipe (R75/15/10 @ 10k) has been validated:

- **Tier3 Ratios**: 75% legacy, 15% organic, 10% expanded
- **Steps**: 10,000 (before overfitting cliff)
- **Seeds**: 3 seeds, select median on organic holdout
- **Metrics**: Track both Frozen Tier3 (retention) and Organic Holdout (generalization)

## Key Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Danger Score | `max(sim(anchor, neg_i)) - sim(anchor, positive)` | How close negatives are to winning |
| Margin | `-danger` | Positive = correct ranking |
| Pass Rate | `count(margin > 0) / total` | Fraction of correct rankings |

See [METRICS.md](docs/crucible/METRICS.md) for complete definitions.

## Development

```bash
# Regenerate CLI docs from --help output
python tools/extract_script_flags.py

# Run docs linter
python tools/docs_lint_crucible.py

# Install pre-commit hook (optional)
cp tools/pre-commit-docs.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

CI automatically checks for documentation drift on every push.

## License

MIT License - see [LICENSE](LICENSE)
