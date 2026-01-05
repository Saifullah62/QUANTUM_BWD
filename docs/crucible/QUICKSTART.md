# Crucible Quickstart

Get from installation to a completed baseline run in 15 minutes.

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU for development
- 8GB+ GPU memory for training, 4GB+ for evaluation

## Installation

```bash
pip install crucible-ml

# Verify installation
crucible --version
```

## Step 1: Prepare Your Data

Crucible expects a canonical events corpus in JSONL format:

```json
{"event_id": "evt_001", "sense": "bank#financial", "text": "The bank approved the loan."}
{"event_id": "evt_002", "sense": "bank#financial", "text": "Check your bank balance."}
{"event_id": "evt_003", "sense": "bank#river", "text": "We walked along the river bank."}
```

For this quickstart, we'll use the sample data:

```bash
crucible init --sample-data ./quickstart_data
```

This creates:
- `quickstart_data/events.jsonl` - Sample canonical events
- `quickstart_data/bundles.jsonl` - Pre-generated tiered bundles
- `quickstart_data/frozen_eval.jsonl` - Locked evaluation set
- `quickstart_data/config.yaml` - Starter configuration

## Step 2: Train Baseline Model

```bash
crucible train \
  --config quickstart_data/config.yaml \
  --output-dir results/baseline \
  --device cuda
```

Expected output:
```
Training with seed 42
Loaded 5000 bundles
  tier1_easy: 3500
  tier2_robust: 1000
  tier3_adversarial: 500
Tier3 pools: 400 legacy + 100 organic + 0 expanded
Training: 100%|██████████| 5000/5000 [02:08<00:00, 38.91it/s]
Checkpoint saved: results/baseline/checkpoint_final.pt
```

## Step 3: Evaluate Against Frozen Eval

```bash
crucible eval \
  --checkpoint results/baseline/checkpoint_final.pt \
  --eval-set quickstart_data/frozen_eval.jsonl \
  --output results/baseline/frozen_scoreboard.json \
  --device cuda
```

Expected output:
```
Loaded eval set: quickstart_data/frozen_eval.jsonl
EVAL_FINGERPRINT: a1b2c3d4|42|1000
  Items: 1000
  Tiers: {tier1: 600, tier2: 300, tier3: 100}

Overall: 72.3% pass_rate, margin=-0.012
  tier1_easy: 85.2% pass_rate
  tier2_robust: 68.7% pass_rate
  tier3_adversarial: 31.0% pass_rate

Wrote scoreboard: results/baseline/frozen_scoreboard.json
```

## Step 4: Package as Reproducible Capsule

```bash
crucible capsule pack \
  --source-dir results/baseline \
  --output results/baseline_capsule.tar.gz
```

This bundles:
- `config.yaml` (frozen copy)
- `checkpoint_final.pt`
- `frozen_scoreboard.json`
- `metadata.json` (git hash, timestamps, versions)

## Step 5: Generate Report

```bash
crucible report \
  --results-dir results/baseline \
  --output baseline_report.html \
  --type summary
```

Open `baseline_report.html` in your browser to view margin distributions and tier breakdowns.

## Next Steps

1. **Run a ratio sweep** to find optimal tier3 pool mixing:
   ```bash
   crucible sweep ratio \
     --config quickstart_data/config.yaml \
     --ratios "90_10_0,80_15_5,70_20_10" \
     --output-dir results/ratio_sweep
   ```

2. **Mine organic tier3** to discover new hard cases:
   ```bash
   crucible mine \
     --events quickstart_data/events.jsonl \
     --existing-tier3 quickstart_data/tier3_legacy.jsonl \
     --output quickstart_data/tier3_organic.jsonl
   ```

3. **Run step sweep** to find overfitting cliff:
   ```bash
   crucible sweep steps \
     --steps 5000,10000,20000 \
     --output-dir results/step_sweep
   ```

See [CLI Reference](CLI.md) for complete command documentation.

## Doc QA

Before shipping documentation changes, run the linter to check for consistency:

```bash
python tools/docs_lint_crucible.py
```

This verifies:
- All internal markdown links resolve
- Code fences are properly closed
- Core formulas are single-sourced in METRICS.md
- Documentation file count matches README links

## Troubleshooting

**CUDA out of memory**
```bash
crucible train --batch-size 16 --device cuda  # Reduce batch size
```

**Slow training on CPU**
```bash
crucible train --device cpu --steps 1000  # Reduce steps for testing
```

**Config validation errors**
```bash
crucible config validate --config your_config.yaml
```

## Sample Config Explained

```yaml
# quickstart_data/config.yaml
data:
  bundles: quickstart_data/bundles.jsonl
  frozen_eval: quickstart_data/frozen_eval.jsonl

training:
  steps: 5000
  batch_size: 32
  learning_rate: 1.0e-3

curriculum:
  tier_probabilities: [0.60, 0.30, 0.10]  # tier1, tier2, tier3
  tier3_ratios:
    legacy: 0.80    # 80% of tier3 draws from original core killers
    organic: 0.15   # 15% from mined organic
    expanded: 0.05  # 5% from vetted expansions

experiment:
  seeds: [42]
  output_dir: results/baseline
```

The key insight: `tier3_ratios` controls how much exposure legacy core killers get during training. Higher legacy ratio = better retention of original expertise.
