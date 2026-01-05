# Crucible Configuration Reference

YAML configuration schema and validation rules.

> **Note**: These are recommended production defaults. Current scripts may use
> different defaults (e.g., `embed_dim=256` in v2). Check script `--help` for actual defaults.

## Configuration File Structure

```yaml
# crucible.yaml

# Data paths
data:
  bundles: path/to/bundles.jsonl
  frozen_eval: path/to/frozen_eval.jsonl
  organic_eval: path/to/organic_holdout.jsonl  # Optional

# Training parameters
training:
  steps: 10000
  batch_size: 32
  learning_rate: 1.0e-3
  weight_decay: 0.01
  gradient_clip: 1.0
  warmup_steps: 500

# Curriculum sampling
curriculum:
  mode: uniform  # uniform | phased | phased_v2
  tier_probabilities: [0.60, 0.30, 0.10]
  tier_caps: [null, null, null]  # Optional per-tier batch caps
  tier3_ratios:
    legacy: 0.75
    organic: 0.15
    expanded: 0.10

# Model architecture
model:
  embed_dim: 768
  hidden_dim: 1024
  num_layers: 2
  dropout: 0.1

# Experiment configuration
experiment:
  seeds: [42]
  output_dir: results/experiment
  checkpoint_every: null  # Save intermediate checkpoints
  eval_every: null        # Run eval during training

# Device configuration
device:
  type: cuda  # cuda | cpu
  precision: fp32  # fp32 | fp16 | bf16

# Guardrails
guardrails:
  dilution_warning_threshold: 30  # Warn if legacy exposure < 30
  regression_threshold: 0.02      # Alert if frozen tier3 drops > 2%
  overfitting_detection: true     # Monitor organic holdout vs training steps
```

## Section Details

### data

Paths to data files. All paths are relative to config file location unless absolute.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `bundles` | string | Yes | Training bundles JSONL |
| `frozen_eval` | string | Yes | Frozen evaluation set |
| `organic_eval` | string | No | Organic holdout for generalization testing |
| `embeddings_cache` | string | No | Pre-computed embeddings directory |

### training

Core training hyperparameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `steps` | int | 10000 | Total training steps |
| `batch_size` | int | 32 | Samples per batch |
| `learning_rate` | float | 1e-3 | Initial learning rate |
| `weight_decay` | float | 0.01 | L2 regularization strength |
| `gradient_clip` | float | 1.0 | Max gradient norm |
| `warmup_steps` | int | 500 | Linear warmup steps |
| `lr_schedule` | string | constant | Learning rate schedule: constant, linear, cosine |

### curriculum

Curriculum sampling configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | uniform | Sampling strategy |
| `tier_probabilities` | list[float] | [0.6, 0.3, 0.1] | Per-tier sampling weights |
| `tier_caps` | list[int\|null] | [null, null, null] | Max items per tier per batch |
| `tier3_ratios` | object | see below | Three-pool mixing ratios |

#### curriculum.tier3_ratios

Controls distribution of tier3 draws across pools.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `legacy` | float | 0.75 | Fraction to legacy core killers |
| `organic` | float | 0.15 | Fraction to organic mined |
| `expanded` | float | 0.10 | Fraction to vetted expansions |

**Constraint**: `legacy + organic + expanded = 1.0`

#### Curriculum Modes

**uniform**: Fixed tier probabilities throughout training.
```yaml
curriculum:
  mode: uniform
  tier_probabilities: [0.60, 0.30, 0.10]
```

**phased**: Increase tier3 exposure over training.
```yaml
curriculum:
  mode: phased
  phases:
    - until_step: 3000
      tier_probabilities: [0.70, 0.25, 0.05]
    - until_step: 7000
      tier_probabilities: [0.60, 0.30, 0.10]
    - until_step: null  # remainder
      tier_probabilities: [0.50, 0.35, 0.15]
```

**phased_v2**: Smooth interpolation between phases.
```yaml
curriculum:
  mode: phased_v2
  start_probabilities: [0.70, 0.25, 0.05]
  end_probabilities: [0.50, 0.35, 0.15]
  transition_start: 2000
  transition_end: 8000
```

### model

Model architecture parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embed_dim` | int | 768 | Embedding dimension |
| `hidden_dim` | int | 1024 | Hidden layer dimension |
| `num_layers` | int | 2 | Number of encoder layers |
| `dropout` | float | 0.1 | Dropout probability |
| `activation` | string | gelu | Activation function |

### experiment

Experiment orchestration settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seeds` | list[int] | [42] | Random seeds for multi-seed runs |
| `output_dir` | string | required | Base output directory |
| `checkpoint_every` | int\|null | null | Steps between checkpoints |
| `eval_every` | int\|null | null | Steps between evaluations |
| `log_every` | int | 100 | Steps between log entries |

### device

Hardware configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | cuda | Device type: cuda, cpu |
| `precision` | string | fp32 | Numerical precision |
| `cuda_device` | int | 0 | GPU index (multi-GPU) |

### guardrails

Automated safety checks.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dilution_warning_threshold` | float | 30 | Warn if expected legacy exposures below |
| `regression_threshold` | float | 0.02 | Alert threshold for frozen tier3 drop |
| `overfitting_detection` | bool | true | Monitor organic holdout decline |
| `fingerprint_strict` | bool | true | Require exact fingerprint match |

## Configuration Inheritance

Configs can extend base configurations:

```yaml
# production.yaml
extends: base.yaml

training:
  steps: 20000  # Override

experiment:
  seeds: [42, 123, 456]  # Override
```

Resolution order:
1. Default values
2. Base config (`extends`)
3. Current config values
4. CLI overrides

## Environment Variable Substitution

Use `${VAR}` syntax for environment variables:

```yaml
data:
  bundles: ${DATA_ROOT}/bundles.jsonl

experiment:
  output_dir: ${OUTPUT_ROOT}/experiment_${RUN_ID}
```

## Validation Rules

### Required Fields

- `data.bundles`
- `data.frozen_eval`
- `experiment.output_dir`

### Constraints

```python
# Tier probabilities must sum to 1
assert sum(curriculum.tier_probabilities) == 1.0

# Tier3 ratios must sum to 1
assert (curriculum.tier3_ratios.legacy +
        curriculum.tier3_ratios.organic +
        curriculum.tier3_ratios.expanded) == 1.0

# All probabilities/ratios must be non-negative
assert all(p >= 0 for p in curriculum.tier_probabilities)
assert curriculum.tier3_ratios.legacy >= 0
assert curriculum.tier3_ratios.organic >= 0
assert curriculum.tier3_ratios.expanded >= 0

# Steps must be positive
assert training.steps > 0
assert training.batch_size > 0
```

## Example Configurations

### Baseline Config

```yaml
data:
  bundles: data/contrastive_bundles_v25.jsonl
  frozen_eval: evals/frozen_eval_v23.jsonl

training:
  steps: 10000
  batch_size: 32
  learning_rate: 1.0e-3

curriculum:
  mode: uniform
  tier_probabilities: [0.60, 0.30, 0.10]
  tier3_ratios:
    legacy: 0.75
    organic: 0.15
    expanded: 0.10

experiment:
  seeds: [42]
  output_dir: results/baseline
```

### Ratio Sweep Config

```yaml
data:
  bundles: data/contrastive_bundles_v25.jsonl
  frozen_eval: evals/frozen_eval_v23.jsonl
  organic_eval: evals/organic_holdout_v2.jsonl

training:
  steps: 10000
  batch_size: 32

curriculum:
  mode: uniform
  tier_probabilities: [0.60, 0.30, 0.10]
  # tier3_ratios set per-condition by sweep

experiment:
  seeds: [42, 123, 456]
  output_dir: results/ratio_sweep

sweep:
  type: ratio
  conditions:
    - name: R85_10_5
      tier3_ratios: {legacy: 0.85, organic: 0.10, expanded: 0.05}
    - name: R75_15_10
      tier3_ratios: {legacy: 0.75, organic: 0.15, expanded: 0.10}
    - name: R65_25_10
      tier3_ratios: {legacy: 0.65, organic: 0.25, expanded: 0.10}
```

### Production Config

```yaml
data:
  bundles: data/contrastive_bundles_v25_organic.jsonl
  frozen_eval: evals/frozen_eval_v23.jsonl
  organic_eval: evals/organic_holdout_v2.jsonl

training:
  steps: 10000  # Validated sweet spot
  batch_size: 32
  learning_rate: 1.0e-3
  weight_decay: 0.01

curriculum:
  mode: uniform
  tier_probabilities: [0.60, 0.30, 0.10]
  tier3_ratios:
    legacy: 0.75   # Validated optimal
    organic: 0.15
    expanded: 0.10

model:
  embed_dim: 768
  hidden_dim: 1024

experiment:
  seeds: [42, 123, 456]
  output_dir: results/production

guardrails:
  dilution_warning_threshold: 30
  regression_threshold: 0.02
  overfitting_detection: true
```

## CLI Override Syntax

CLI arguments override config file values:

```bash
crucible train \
  --config production.yaml \
  --training.steps 20000 \
  --curriculum.tier3_ratios.legacy 0.80 \
  --experiment.seeds 42,123
```

Nested fields use dot notation. Lists use comma separation.
