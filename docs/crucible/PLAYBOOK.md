# Crucible Experimental Playbook

Validated operational protocol for running experiments and avoiding common pitfalls.

## Core Principles

### 1. Two-Truth Evaluation

Every experiment must evaluate against **both**:
- **Frozen Eval**: Measures retention of known hard cases
- **Organic Holdout**: Measures generalization to novel patterns

Either metric alone can be misleading:
- Frozen-only: May miss generalization collapse
- Organic-only: May miss expertise erosion on core killers

### 2. Pareto Thinking

There is rarely a single "best" configuration. Instead, map the trade-off frontier:
- X-axis: Frozen tier3 pass rate (retention)
- Y-axis: Organic holdout pass rate (generalization)

A configuration is optimal if nothing else improves one metric without degrading the other.

### 3. Dilution Awareness

Adding data is not free. Every new bundle in a pool reduces expected exposures for existing items. Always compute per-item expected exposures before expanding pools. See [METRICS.md](METRICS.md#per-item-expected-exposures) for the formula.

## Standard Experiment Sequence

### Phase 1: Baseline Establishment

**Goal**: Establish reproducible baseline with known configuration.

```bash
# 1. Validate configuration
crucible config validate --config baseline.yaml
crucible config exposure --config baseline.yaml

# 2. Train baseline with multiple seeds
crucible train \
  --config baseline.yaml \
  --seeds 42,123,456 \
  --output-dir results/baseline

# 3. Evaluate against both truth sources
crucible eval \
  --results-dir results/baseline \
  --frozen-eval evals/frozen_eval_v23.jsonl \
  --organic-eval evals/organic_holdout_v2.jsonl

# 4. Package as reference capsule
crucible capsule pack \
  --source-dir results/baseline \
  --output releases/baseline_v1.tar.gz
```

**Checklist**:
- [ ] Dilution guardrail passes (exposure ≥ 50)
- [ ] 3+ seeds for statistical significance
- [ ] Both scoreboards generated
- [ ] Capsule archived for future comparison

### Phase 2: Ratio Sweep

**Goal**: Find optimal tier3 pool mixing ratios.

**When to run**: After adding new organic or expanded bundles.

```bash
crucible sweep ratio \
  --config baseline.yaml \
  --ratios "90_10_0,85_10_5,80_15_5,75_15_10,70_20_10,65_25_10" \
  --seeds 42,123,456 \
  --frozen-eval evals/frozen_eval_v23.jsonl \
  --organic-eval evals/organic_holdout_v2.jsonl \
  --output-dir results/ratio_sweep
```

**Analysis**:
1. Plot Pareto frontier (frozen T3 vs organic HO)
2. Identify knee point (best balance)
3. Check for dilution at low legacy ratios

**Expected Findings**:
| Ratio | Typical Frozen T3 | Typical Organic HO | Notes |
|-------|-------------------|-------------------|-------|
| R90_10_0 | Highest | Lowest | Pure legacy, no generalization |
| R75_15_10 | Balanced | Balanced | Validated sweet spot |
| R65_25_10 | Lower | Similar | Dilution risk |

### Phase 3: Step Sweep

**Goal**: Determine optimal training duration and detect overfitting cliff.

**When to run**: After establishing optimal ratios.

```bash
crucible sweep steps \
  --config baseline.yaml \
  --steps "5000,10000,15000,20000" \
  --seeds 42,123 \
  --frozen-eval evals/frozen_eval_v23.jsonl \
  --organic-eval evals/organic_holdout_v2.jsonl \
  --output-dir results/step_sweep
```

**Analysis**:
1. Plot organic holdout vs steps
2. Identify overfitting cliff (where organic starts dropping)
3. Select step count just before cliff

**Expected Pattern**:
```
Steps    Frozen T3    Organic HO    Status
5000     30.2%        31.0%         Baseline
10000    31.9%        31.5%         Improving
15000    32.2%        30.5%         Starting to overfit
20000    32.0%        28.5%         Overfitting confirmed
```

**Recommendation**: Stop at 10k steps if overfitting starts at 15k+.

### Phase 4: Quality Ablation

**Goal**: Determine if quality filtering improves results.

**When to run**: When organic pool has variable quality (different pass counts).

```bash
# Split organic by pass strength
crucible pool filter \
  --input tier3_organic.jsonl \
  --filter "passes >= 8" \
  --output tier3_organic_ultra.jsonl

# Create bundle variants
# Ultra-only: base + ultra organic
# Full: base + all organic

# Run ablation
crucible sweep ablation \
  --conditions "ULTRA:bundles_ultra.jsonl,FULL:bundles_full.jsonl" \
  --seeds 42,123,456 \
  --output-dir results/quality_ablation
```

**Interpretation**:
- **ULTRA > FULL**: Quality matters, filter aggressively
- **ULTRA ≈ FULL**: Saturation is real, volume doesn't hurt
- **ULTRA < FULL**: Volume diversity helps, keep all data

### Phase 5: Production Configuration

**Goal**: Lock in validated production config.

After completing sweeps:

```yaml
# production.yaml
data:
  bundles: data/contrastive_bundles_v25_organic.jsonl
  frozen_eval: evals/frozen_eval_v23.jsonl
  organic_eval: evals/organic_holdout_v2.jsonl

training:
  steps: 10000  # Validated: before overfitting cliff
  batch_size: 32
  learning_rate: 1.0e-3

curriculum:
  mode: uniform
  tier_probabilities: [0.60, 0.30, 0.10]
  tier3_ratios:
    legacy: 0.75   # Validated: Pareto-optimal
    organic: 0.15
    expanded: 0.10

guardrails:
  dilution_warning_threshold: 30
  regression_threshold: 0.02
  baseline_scoreboard: results/baseline/frozen_scoreboard.json
```

## Common Failure Modes

### F1: Uncontrolled Pool Expansion

**Symptom**: Frozen tier3 drops after adding "more hard examples."

**Cause**: Legacy exposure diluted below critical threshold.

**Prevention**:
1. Always compute exposure before expansion
2. Adjust ratios to maintain legacy exposure ≥ 50
3. Gate expansions on quality (multi-lineup validation)

**Example**:
```
Before: 454 legacy, r_legacy=1.0, exposure=70.5
After:  454 legacy + 1000 organic, r_legacy=0.5, exposure=35.2 ← WARNING
```

### F2: Pseudo-Killer Contamination

**Symptom**: Organic holdout doesn't improve despite mining more killers.

**Cause**: Mined examples are pseudo-killers (hard against one lineup, easy against most).

**Prevention**:
1. Use multi-lineup validation (10 lineups, ≥3 passes)
2. Track pass distribution, prefer 8+ pass items
3. Audit sample of accepted candidates manually

**Detection**:
```python
# Pass distribution should show concentration at high pass counts
# Bad: uniform distribution
# Good: right-skewed (most at 7-10 passes)
```

### F3: Overfitting to Frozen Eval

**Symptom**: Frozen tier3 keeps improving, organic holdout stagnates or drops.

**Cause**: Training too long, model memorizes specific patterns.

**Prevention**:
1. Run step sweep to find overfitting cliff
2. Use organic holdout as early stopping signal
3. Consider regularization (dropout, weight decay)

### F4: Eval Set Drift

**Symptom**: Historical comparisons become unreliable.

**Cause**: Eval set modified without versioning.

**Prevention** (see [GUARDRAILS.md](GUARDRAILS.md#frozen-eval-change-control) for full policy):
1. Frozen eval is append-only
2. Version all additions: v23 → v24
3. Record fingerprint changes in changelog
4. Never compare across fingerprints without explicit override

### F5: Seed Sensitivity

**Symptom**: Large variance across seeds, conclusions don't replicate.

**Cause**: Insufficient seeds, unlucky draws.

**Prevention**:
1. Always use 3+ seeds
2. Report mean ± std
3. Require ≥2% separation for significance
4. If variance > 3%, investigate data or training instability

## Decision Trees

### When to Mine More Organics

```
Is organic holdout improving with current pool?
├─ Yes → Keep training, don't mine more
└─ No
   ├─ Is legacy exposure healthy (≥50)?
   │  ├─ Yes → Mine more, saturation may break
   │  └─ No → Increase legacy ratio first
   └─ Is current organic quality high (8+ passes)?
      ├─ Yes → Try longer training or larger model
      └─ No → Quality filter, then mine more
```

### When to Adjust Ratios

```
Is frozen tier3 dropping vs baseline?
├─ Yes
│  ├─ Did pool size increase?
│  │  ├─ Yes → Increase legacy ratio
│  │  └─ No → Check training duration, may be overfitting
│  └─ Is organic holdout also dropping?
│     ├─ Yes → Something fundamentally wrong, investigate
│     └─ No → Classic dilution, increase legacy ratio
└─ No
   └─ Is organic holdout flat?
      ├─ Yes → Try quality filtering or longer training
      └─ No (improving) → Current config is working
```

### When to Train Longer

```
Is organic holdout improving with more steps?
├─ Yes
│  ├─ Is frozen tier3 stable?
│  │  ├─ Yes → Safe to train longer
│  │  └─ No (dropping) → Approaching overfit cliff
│  └─ Continue until organic plateaus
└─ No
   ├─ Is organic holdout dropping?
   │  ├─ Yes → Overfitting, reduce steps
   │  └─ No (flat) → Capacity limited, try larger model
```

## Checklist: Before Pushing to Production

- [ ] Baseline capsule archived
- [ ] Ratio sweep completed, optimal ratio identified
- [ ] Step sweep completed, overfitting cliff mapped
- [ ] Frozen tier3 ≥ baseline (no regression)
- [ ] Organic holdout tracked (generalization preserved)
- [ ] Dilution guardrail passes (legacy exposure ≥ 50)
- [ ] 3+ seeds run, variance within acceptable range
- [ ] Config frozen and version-controlled
- [ ] Capsule packaged with all artifacts

## Experiment Log Template

```markdown
## Experiment: [Name]

**Date**: YYYY-MM-DD
**Goal**: [What are we testing?]
**Hypothesis**: [What do we expect to find?]

### Configuration
- Bundles: [path, version]
- Frozen Eval: [path, fingerprint]
- Organic Eval: [path, fingerprint]
- Ratio: R[legacy]_[organic]_[expanded]
- Steps: [N]
- Seeds: [list]

### Results
| Condition | Frozen T3 | Organic HO | Notes |
|-----------|-----------|------------|-------|
| ... | ...% | ...% | ... |

### Analysis
[What did we learn?]

### Decision
[What do we do next?]

### Artifacts
- Capsule: [path]
- Scoreboards: [paths]
- Logs: [paths]
```
