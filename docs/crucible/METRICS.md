# Crucible Metrics Reference

Formal definitions for all metrics used in Crucible evaluation and monitoring.

## Core Evaluation Metrics

### Danger Score

**Definition**: Measures how close the hardest negative is to the anchor relative to the positive.

```
danger = max(sim(anchor, neg_i)) - sim(anchor, positive)
```

**Interpretation**:
- `danger < 0`: Easy case - positive is closer than all negatives
- `danger ≈ 0`: Boundary case - hardest negative nearly ties the positive
- `danger > 0`: Hard case - at least one negative is closer than the positive

**Usage**: Bundle tier assignment, organic mining threshold calibration.

### Margin

**Definition**: How much closer the positive is than the best negative.

```
margin = sim(anchor, positive) - max(sim(anchor, neg_i))
margin = -danger
```

**Interpretation**:
- `margin > 0`: Correct ranking (pass)
- `margin < 0`: Incorrect ranking (fail)
- `|margin|`: Confidence of the ranking

**Key Percentiles**:
- `Q10`: 10th percentile margin (worst-case performance indicator)
- `Q50`: Median margin (typical performance)
- `Q90`: 90th percentile margin (best-case performance)

### Pass Rate

**Definition**: Fraction of items where the model ranks positive above all negatives.

```
pass_rate = count(margin > 0) / total_items
```

**Note**: Pass rate equals accuracy for single-positive contrastive tasks.

### Mean Rank

**Definition**: Average position of the positive when items are sorted by similarity to anchor.

```
rank_i = 1 + count(sim(anchor, neg_j) > sim(anchor, positive))
mean_rank = mean(rank_i)
```

**Interpretation**:
- `mean_rank = 1.0`: Perfect (positive always ranked first)
- `mean_rank > 1.0`: Some negatives ranked above positive

**Usage**: Measures severity of failures beyond binary pass/fail.

## Tier Assignment

### Tier Definitions

Bundles are assigned to tiers based on danger score percentiles computed over the full corpus:

| Tier | Percentile Range | Danger Score | Description |
|------|------------------|--------------|-------------|
| tier1_easy | 0-70th | Low | Routine cases |
| tier2_robust | 70th-90th | Medium | Challenging but learnable |
| tier3_adversarial | 90th-100th | High | Core killers |

### Tier Assignment Algorithm

```python
def assign_tiers(bundles: List[Bundle], thresholds: Tuple[float, float] = (0.70, 0.90)):
    dangers = [compute_danger(b) for b in bundles]
    sorted_dangers = sorted(dangers)

    t1_cutoff = sorted_dangers[int(len(dangers) * thresholds[0])]
    t2_cutoff = sorted_dangers[int(len(dangers) * thresholds[1])]

    for bundle, danger in zip(bundles, dangers):
        if danger <= t1_cutoff:
            bundle.tier = 'tier1_easy'
        elif danger <= t2_cutoff:
            bundle.tier = 'tier2_robust'
        else:
            bundle.tier = 'tier3_adversarial'
```

## Eval Fingerprinting

### Purpose

Eval fingerprints ensure scoreboards are only compared when they evaluate the same data under the same conditions.

### Fingerprint Components

```json
{
  "source_hash": "SHA256(eval_file_contents)[:16]",
  "seed": 42,
  "item_count": 3786,
  "tier_distribution": {"tier1": 1666, "tier2": 1666, "tier3": 454},
  "version": "v23"
}
```

### Fingerprint String Format

```
EVAL_FINGERPRINT: {source_hash}|{seed}|{item_count}
Example: EVAL_FINGERPRINT: a1b2c3d4e5f6g7h8|42|3786
```

### Comparison Rules

1. **Exact Match Required**: Scoreboards with different fingerprints cannot be directly compared
2. **Override Available**: `--allow-fingerprint-mismatch` flag for exploratory analysis
3. **Version Tracking**: Frozen eval sets are append-only with versioned fingerprints

## Exposure Probability & Dilution Detection

### Per-Item Expected Exposures

The critical metric for detecting dilution is how many times each legacy killer is expected to be seen during training.

**Definitions**:
- `S`: Total training steps
- `B`: Batch size
- `p_tier3`: Probability of sampling a tier3 item per batch slot
- `r_legacy`: Ratio of tier3 draws allocated to legacy pool
- `|L|`: Size of legacy pool (number of legacy bundles)

**Per-Item Expected Exposures**:
```
E[exposures per legacy item] = (S × B × p_tier3 × r_legacy) / |L|
```

**Example Calculation**:
```
S = 10,000 steps
B = 32 batch size
p_tier3 = 0.10 (10% of batch is tier3)
r_legacy = 0.75 (75% of tier3 draws go to legacy)
|L| = 454 legacy bundles

E[exposures] = (10000 × 32 × 0.10 × 0.75) / 454
             = 24000 / 454
             = 52.9 expected exposures per legacy item
```

### Dilution Warning Thresholds

| Expected Exposures | Status | Action |
|-------------------|--------|--------|
| ≥ 50 | Healthy | No action needed |
| 30-50 | Warning | Monitor frozen tier3 closely |
| 10-30 | Critical | Consider increasing legacy ratio |
| < 10 | Severe | Regression highly likely |

### Dilution Detection Formula

```python
def compute_legacy_exposure(
    steps: int,
    batch_size: int,
    tier3_probability: float,
    legacy_ratio: float,
    legacy_pool_size: int
) -> float:
    total_tier3_draws = steps * batch_size * tier3_probability
    legacy_draws = total_tier3_draws * legacy_ratio
    per_item_exposure = legacy_draws / legacy_pool_size
    return per_item_exposure

def check_dilution(exposure: float) -> str:
    if exposure >= 50:
        return "healthy"
    elif exposure >= 30:
        return "warning"
    elif exposure >= 10:
        return "critical"
    else:
        return "severe"
```

## Scoreboard Schema

### TierStats

```python
@dataclass
class TierStats:
    n: int              # Number of items in tier
    correct: int        # Items with positive margin
    accuracy: float     # correct / n
    pass_rate: float    # Same as accuracy for contrastive
    mean_margin: float  # Average margin
    median_margin: float
    q10_margin: float   # 10th percentile
    q90_margin: float   # 90th percentile
    mean_rank: float    # Average rank of positive
```

### Scoreboard

```python
@dataclass
class Scoreboard:
    eval_header: EvalHeader
    overall: TierStats
    by_tier: Dict[str, TierStats]  # tier1_easy, tier2_robust, tier3_adversarial
    timestamp: str
    checkpoint_path: str
```

## Pareto Analysis

### Pareto Frontier Definition

A configuration is Pareto-optimal if no other configuration improves one metric without degrading another.

**Metrics**:
- X-axis: Frozen tier3 pass rate (retention)
- Y-axis: Organic holdout pass rate (generalization)

### Pareto Dominance

Configuration A dominates B if:
```
frozen_A >= frozen_B AND organic_A >= organic_B
AND (frozen_A > frozen_B OR organic_A > organic_B)
```

### Frontier Computation

```python
def compute_pareto_frontier(results: List[Dict]) -> List[Dict]:
    frontier = []
    for r in results:
        dominated = False
        for other in results:
            if (other['frozen_tier3'] >= r['frozen_tier3'] and
                other['organic_holdout'] >= r['organic_holdout'] and
                (other['frozen_tier3'] > r['frozen_tier3'] or
                 other['organic_holdout'] > r['organic_holdout'])):
                dominated = True
                break
        if not dominated:
            frontier.append(r)
    return frontier
```

## Multi-Lineup Validation Metrics

### Pass Count Distribution

For organic mining, candidates are tested across multiple independent lineups:

```
passes = count(danger >= threshold across N lineups)
```

**Pass Strength Categories**:
- Ultra (8-10 passes): Consistently dangerous, highest confidence
- Mid (5-7 passes): Moderately consistent
- Low (2-4 passes): Borderline, may be pseudo-killers

### Mining Pass Rate

```
mining_pass_rate = accepted_candidates / total_candidates_tested
```

Validated mechanism yields ~1.4% pass rate with default parameters (10 lineups, 3 minimum passes).

## Aggregate Experiment Metrics

### Cross-Seed Statistics

When running multiple seeds:

```python
@dataclass
class AggregateStats:
    mean: float
    std: float
    min: float
    max: float
    n_seeds: int
```

### Confidence Intervals

```
CI_95 = mean ± (1.96 × std / sqrt(n_seeds))
```

For 3 seeds, require ≥2% separation for statistical significance at p<0.05.
