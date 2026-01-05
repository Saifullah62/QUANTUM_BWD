# Crucible CLI Reference

Complete command documentation for the Crucible CLI.

> **Note**: This documents the target `crucible` CLI interface. For current implementation,
> use the underlying scripts directly:
> - Training: `python scripts/train_curriculum_v3.py` (or v2)
> - Evaluation: `python experiments/eval_capsules.py`
> - See [ARCHITECTURE.md](ARCHITECTURE.md) for script locations and real flags.

## Global Options

```bash
crucible [--version] [--help] [--config PATH] [--verbose] <command>
```

| Option | Description |
|--------|-------------|
| `--version` | Show version and exit |
| `--help` | Show help message |
| `--config PATH` | Path to config file (default: `./crucible.yaml`) |
| `--verbose, -v` | Increase output verbosity |

## Commands

### crucible init

Initialize a new Crucible project with sample data and configuration.

```bash
crucible init [OPTIONS] OUTPUT_DIR
```

**Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--sample-data` | false | Include sample datasets |
| `--template` | basic | Config template: basic, production, sweep |

**Example**:
```bash
crucible init --sample-data ./my_project
```

**Output**:
```
my_project/
  crucible.yaml       # Configuration file
  events.jsonl        # Sample canonical events (if --sample-data)
  bundles.jsonl       # Sample tiered bundles (if --sample-data)
  frozen_eval.jsonl   # Sample eval set (if --sample-data)
```

---

### crucible train

Train a model with curriculum sampling.

```bash
crucible train [OPTIONS]
```

**Required Options**:
| Option | Description |
|--------|-------------|
| `--bundles PATH` | Path to tiered bundles JSONL |
| `--output-dir PATH` | Directory for checkpoints and logs |

**Training Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--steps INT` | 10000 | Number of training steps |
| `--batch-size INT` | 32 | Batch size |
| `--lr FLOAT` | 1e-3 | Learning rate |
| `--weight-decay FLOAT` | 0.01 | L2 regularization |
| `--seed INT` | 42 | Random seed |
| `--device` | cuda | Device: cuda, cpu |

**Curriculum Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--curriculum` | uniform | Sampling mode: uniform, phased, phased_v2 |
| `--tier-probabilities` | 0.6,0.3,0.1 | Tier1, tier2, tier3 sampling probabilities |
| `--tier3-legacy FLOAT` | 0.75 | Legacy pool ratio within tier3 |
| `--tier3-organic FLOAT` | 0.15 | Organic pool ratio within tier3 |
| `--tier3-expanded FLOAT` | 0.10 | Expanded pool ratio within tier3 |

**Example**:
```bash
crucible train \
  --bundles data/contrastive_bundles.jsonl \
  --output-dir results/baseline \
  --steps 10000 \
  --tier3-legacy 0.75 \
  --tier3-organic 0.15 \
  --tier3-expanded 0.10 \
  --device cuda
```

**Output**:
```
results/baseline/
  checkpoint_final.pt   # Model weights
  train.log            # Training output
  config.yaml          # Frozen config copy
```

---

### crucible eval

Evaluate a checkpoint against an eval set.

```bash
crucible eval [OPTIONS]
```

**Required Options**:
| Option | Description |
|--------|-------------|
| `--checkpoint PATH` | Path to checkpoint file |
| `--eval-set PATH` | Path to eval set JSONL |
| `--output PATH` | Path for scoreboard JSON |

**Optional**:
| Option | Default | Description |
|--------|---------|-------------|
| `--device` | cuda | Device: cuda, cpu |
| `--batch-size INT` | 64 | Inference batch size |
| `--allow-fingerprint-mismatch` | false | Skip fingerprint validation |

**Example**:
```bash
crucible eval \
  --checkpoint results/baseline/checkpoint_final.pt \
  --eval-set evals/frozen_eval_v23.jsonl \
  --output results/baseline/frozen_scoreboard.json
```

**Output Format**:
```
Loaded eval set: evals/frozen_eval_v23.jsonl
EVAL_FINGERPRINT: a1b2c3d4e5f6g7h8|42|3786
  Items: 3786
  Tiers: {tier1: 1666, tier2: 1666, tier3: 454}

Overall: 72.3% pass_rate, margin=0.045
  tier1_easy: 85.2% pass_rate, margin=0.112
  tier2_robust: 68.7% pass_rate, margin=0.031
  tier3_adversarial: 30.0% pass_rate, margin=-0.089

Wrote scoreboard: results/baseline/frozen_scoreboard.json
```

---

### crucible sweep

Run multi-condition experiments with parallel execution.

#### crucible sweep ratio

Sweep across tier3 pool ratios.

```bash
crucible sweep ratio [OPTIONS]
```

**Required Options**:
| Option | Description |
|--------|-------------|
| `--ratios` | Comma-separated ratio specs: "85_10_5,75_15_10" |
| `--output-dir PATH` | Results directory |

**Optional**:
| Option | Default | Description |
|--------|---------|-------------|
| `--seeds` | 42 | Comma-separated seeds |
| `--steps INT` | 10000 | Training steps per condition |
| `--parallel INT` | 1 | Max parallel training jobs |
| `--frozen-eval PATH` | - | Frozen eval set |
| `--organic-eval PATH` | - | Organic holdout eval set |

**Example**:
```bash
crucible sweep ratio \
  --ratios "85_10_5,80_15_5,75_15_10,70_20_10" \
  --seeds 42,123,456 \
  --steps 10000 \
  --frozen-eval evals/frozen_eval_v23.jsonl \
  --organic-eval evals/organic_holdout_v2.jsonl \
  --output-dir results/ratio_sweep
```

**Output Structure**:
```
results/ratio_sweep/
  R85_10_5/
    seed_42/
      checkpoint_final.pt
      train.log
    seed_123/
      ...
  R75_15_10/
    ...
  frozen_scoreboard.json      # Aggregated
  organic_scoreboard.json     # Aggregated
  pareto_analysis.json
```

#### crucible sweep steps

Sweep across training durations.

```bash
crucible sweep steps [OPTIONS]
```

**Required Options**:
| Option | Description |
|--------|-------------|
| `--steps` | Comma-separated step counts: "5000,10000,20000" |
| `--output-dir PATH` | Results directory |

**Example**:
```bash
crucible sweep steps \
  --steps "5000,10000,20000" \
  --seeds 42,123 \
  --output-dir results/step_sweep
```

---

### crucible mine

Mine organic tier3 bundles from canonical events.

```bash
crucible mine [OPTIONS]
```

**Required Options**:
| Option | Description |
|--------|-------------|
| `--events PATH` | Canonical events JSONL |
| `--existing-tier3 PATH` | Existing tier3 for deduplication |
| `--output PATH` | Output path for mined bundles |

**Mining Parameters**:
| Option | Default | Description |
|--------|---------|-------------|
| `--threshold FLOAT` | 0.593 | Danger threshold (calibrated to existing tier3) |
| `--lineups INT` | 10 | Independent lineups per candidate |
| `--min-passes INT` | 3 | Minimum passes to accept |
| `--candidates-per-sense INT` | 15 | Candidates to test per sense |
| `--max-output INT` | 15000 | Maximum bundles to output |

**Example**:
```bash
crucible mine \
  --events data/canonical_events.jsonl \
  --existing-tier3 data/tier3_legacy.jsonl \
  --output data/tier3_organic.jsonl \
  --threshold 0.593 \
  --lineups 10 \
  --min-passes 3
```

**Output**:
```
Mining organic tier3...
Tested: 25,000 candidates
Passed: 352 (1.4%)
Pass distribution: {2: 0, 3: 45, 4: 67, 5: 58, ..., 10: 12}

Wrote: data/tier3_organic.jsonl
Stats: data/tier3_organic_stats.json
```

---

### crucible capsule

Package and manage reproducible experiment artifacts.

#### crucible capsule pack

Create a self-contained reproducible artifact.

```bash
crucible capsule pack [OPTIONS]
```

**Required Options**:
| Option | Description |
|--------|-------------|
| `--source-dir PATH` | Directory containing run artifacts |
| `--output PATH` | Output capsule path (.tar.gz) |

**Optional**:
| Option | Default | Description |
|--------|---------|-------------|
| `--include-bundles` | false | Include training bundles in capsule |
| `--include-eval` | false | Include eval sets in capsule |

**Example**:
```bash
crucible capsule pack \
  --source-dir results/baseline \
  --output releases/baseline_v1.tar.gz
```

**Capsule Contents**:
```
capsule/
  config.yaml           # Frozen configuration
  checkpoint_final.pt   # Model weights
  frozen_scoreboard.json
  organic_scoreboard.json
  train.log
  metadata.json         # Git hash, timestamps, versions
```

#### crucible capsule unpack

Extract and validate a capsule.

```bash
crucible capsule unpack [OPTIONS] CAPSULE_PATH OUTPUT_DIR
```

**Example**:
```bash
crucible capsule unpack releases/baseline_v1.tar.gz ./unpacked
```

#### crucible capsule verify

Verify capsule integrity and reproducibility metadata.

```bash
crucible capsule verify CAPSULE_PATH
```

**Output**:
```
Capsule: releases/baseline_v1.tar.gz
  Created: 2026-01-15T10:30:00Z
  Git hash: a1b2c3d4
  Config hash: SHA256:e5f6g7h8...
  Checkpoint size: 125.3 MB
  Scoreboards: 2 (frozen, organic)
  Status: VALID
```

---

### crucible report

Generate visualizations and reports from experiment results.

```bash
crucible report [OPTIONS]
```

**Required Options**:
| Option | Description |
|--------|-------------|
| `--results-dir PATH` | Directory containing results |
| `--output PATH` | Output file path |

**Report Types**:
| Option | Description |
|--------|-------------|
| `--type summary` | Single-run summary with tier breakdown |
| `--type pareto` | Pareto frontier visualization |
| `--type margin-dist` | Margin distribution histograms |
| `--type step-trajectory` | Metrics vs training steps |
| `--type dilution` | Pool composition and exposure analysis |

**Example**:
```bash
crucible report \
  --results-dir results/ratio_sweep \
  --output reports/ratio_sweep_pareto.html \
  --type pareto
```

---

### crucible config

Configuration management utilities.

#### crucible config validate

Validate a configuration file.

```bash
crucible config validate --config PATH
```

#### crucible config show

Display resolved configuration with defaults.

```bash
crucible config show --config PATH
```

#### crucible config exposure

Calculate expected per-item exposures for a configuration.

```bash
crucible config exposure --config PATH
```

**Output**:
```
Configuration: crucible.yaml
Training: 10000 steps, batch_size=32
Tier3: p=0.10, legacy=0.75, organic=0.15, expanded=0.10

Pool Sizes:
  Legacy: 454 bundles
  Organic: 1037 bundles
  Expanded: 0 bundles

Expected Per-Item Exposures:
  Legacy: 52.9 (HEALTHY)
  Organic: 4.6 (WARNING: low exposure)
  Expanded: N/A

Recommendation: Current config provides healthy legacy exposure.
```

---

### crucible pool

Tier3 pool management utilities.

#### crucible pool merge

Merge multiple tier3 sources with provenance tracking.

```bash
crucible pool merge \
  --legacy PATH \
  --organic PATH \
  --expanded PATH \
  --output PATH
```

#### crucible pool split

Create train/holdout splits.

```bash
crucible pool split \
  --input PATH \
  --holdout-fraction 0.20 \
  --seed 42 \
  --output-train PATH \
  --output-holdout PATH
```

#### crucible pool stats

Display pool statistics.

```bash
crucible pool stats --input PATH
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Data validation error |
| 4 | CUDA/device error |
| 5 | Fingerprint mismatch (eval) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CRUCIBLE_CONFIG` | Default config file path |
| `CRUCIBLE_DEVICE` | Default device (cuda/cpu) |
| `CRUCIBLE_CACHE_DIR` | Cache directory for embeddings |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
