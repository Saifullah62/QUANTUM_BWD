# Crucible

**Contrastive Curriculum and Adversarial Evaluation Harness**

Crucible is a production-grade experimentation harness for teams training embedding models, retrieval systems, and contrastive learners. It solves the silent regression problem: when training metrics improve but real-world performance on hard cases degrades.

## The Problem

Machine learning teams face a dangerous failure mode that standard MLOps tooling does not detect:

1. **Dilution**: Adding more hard examples to training reduces exposure to original "core killers," causing expertise erosion even as aggregate metrics improve
2. **Training Metrics Lie**: Validation loss decreases while generalization to novel adversarial patterns collapses
3. **Pseudo-Killers**: Mined hard examples that appear difficult due to lucky negative draws but are actually easy against most lineups

## The Solution: Two-Truth Evaluation

Crucible separates two fundamentally different questions:

- **Frozen Eval**: "Did we retain mastery of known hard cases?" (Measures retention)
- **Organic Holdout**: "Did we learn to generalize to new hard cases?" (Measures generalization)

Tracking both independently reveals failure modes that either alone would miss.

## Key Features

- **Danger Scoring & Margin Distributions**: Beyond accuracy—track how confidently the model separates positives from negatives
- **Stratified Tier3 Pools**: Separate legacy core killers, vetted expansions, and organic mined killers with explicit ratio controls
- **Multi-Lineup Validation**: Organic miner requires candidates to remain dangerous across multiple independent negative lineups (anti-pseudo-killer)
- **Automated Guardrails**: Dilution detection, regression alerts, overfitting cliff monitoring
- **Pareto Frontier Visualization**: Map trade-offs between retention and generalization

## Quick Start

```bash
# Install
pip install crucible-ml

# Train baseline
crucible train \
  --bundles data/contrastive_bundles.jsonl \
  --steps 10000 \
  --output-dir results/baseline

# Evaluate
crucible eval \
  --checkpoint results/baseline/checkpoint_final.pt \
  --eval-set evals/frozen_eval.jsonl \
  --output results/baseline/scoreboard.json

# Run ratio sweep
crucible sweep ratio \
  --ratios "85_10_5,75_15_10,65_25_10" \
  --seeds 42,123,456 \
  --output-dir results/ratio_sweep

# Generate report
crucible report \
  --results-dir results/ratio_sweep \
  --output dashboard.html
```

## Documentation

- [Documentation Index](INDEX.md) - Start here: which doc should I read?
- [Quickstart Guide](QUICKSTART.md) - Get running in 15 minutes
- [Architecture](ARCHITECTURE.md) - System design and module interfaces
- [Metrics Reference](METRICS.md) - Formal definitions of all metrics
- [CLI Reference](CLI.md) - Complete command documentation
- [Configuration](CONFIG.md) - YAML configuration schema
- [Data Schemas](SCHEMAS.md) - JSON schemas for all data formats
- [Guardrails](GUARDRAILS.md) - Automated checks and warnings
- [Experimental Playbook](PLAYBOOK.md) - Validated operational protocol
- [Commercial Notes](COMMERCIAL.md) - Licensing and enterprise features

## Empirical Validation

Crucible's methodology was validated through extensive experimentation:

| Finding | Result | Implication |
|---------|--------|-------------|
| Dilution regression | 32.5% → 26.2% frozen tier3 | Uncontrolled expansion fails |
| Ratio recovery | 26.2% → 30.0% frozen tier3 | Three-pool mixing works |
| Optimal ratio | R75_15_10 | 75% legacy protects retention |
| Overfitting cliff | 10k→20k: organic 31.5%→28.5% | Early stop at 10k |
| Mining pass rate | ~1.4% | Multi-lineup validation is stringent |

## License

- Core training/eval: Apache 2.0
- Organic miner & enterprise features: Commercial license

## Support

- GitHub Issues: Bug reports and feature requests
- Documentation: This repository
- Enterprise: Contact sales@crucible-ml.com
