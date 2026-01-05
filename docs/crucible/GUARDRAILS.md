# Crucible Guardrails

Automated checks and warnings to prevent silent regressions.

## Overview

Crucible guardrails are automated safety mechanisms that detect dangerous patterns before they cause production regressions. They operate at three stages:

1. **Pre-Training**: Configuration validation and dilution forecasting
2. **During Training**: Quota monitoring and loss anomaly detection
3. **Post-Training**: Regression detection and overfitting alerts

## Pre-Training Guardrails

### G1: Dilution Detector

**Problem**: Adding more tier3 bundles without adjusting ratios reduces per-item exposure for legacy killers, causing expertise erosion.

**Formula**: See [METRICS.md](METRICS.md#per-item-expected-exposures) for the exposure calculation:
`E = (S × B × p_tier3 × r_legacy) / |L|`

**Detection**:
```python
def check_dilution(config: Config, pool_manifest: PoolManifest) -> Alert:
    legacy_exposure = compute_legacy_exposure(
        steps=config.training.steps,
        batch_size=config.training.batch_size,
        tier3_probability=config.curriculum.tier_probabilities[2],
        legacy_ratio=config.curriculum.tier3_ratios.legacy,
        legacy_pool_size=pool_manifest.pools.legacy.count
    )

    if legacy_exposure < 10:
        return Alert.SEVERE, f"Legacy exposure {legacy_exposure:.1f} < 10. Regression highly likely."
    elif legacy_exposure < 30:
        return Alert.CRITICAL, f"Legacy exposure {legacy_exposure:.1f} < 30. Monitor closely."
    elif legacy_exposure < 50:
        return Alert.WARNING, f"Legacy exposure {legacy_exposure:.1f} < 50. Consider increasing ratio."
    else:
        return Alert.OK, f"Legacy exposure {legacy_exposure:.1f}. Healthy."
```

**Thresholds**:
| Expected Exposures | Status | Action |
|-------------------|--------|--------|
| ≥ 50 | OK | Proceed normally |
| 30-50 | WARNING | Monitor frozen tier3 results |
| 10-30 | CRITICAL | Increase legacy ratio or reduce pool |
| < 10 | SEVERE | Block training without override |

**Configuration**:
```yaml
guardrails:
  dilution_warning_threshold: 30
  dilution_block_threshold: 10
  dilution_override: false  # Set true to force proceed
```

### G2: Configuration Validator

**Checks**:
- Tier probabilities sum to 1.0
- Tier3 ratios sum to 1.0
- All paths exist and are readable
- Eval fingerprints match expected versions
- Seeds are deterministic integers

**Example Output**:
```
CONFIG VALIDATION
  [OK] Tier probabilities sum to 1.0 (0.60 + 0.30 + 0.10)
  [OK] Tier3 ratios sum to 1.0 (0.75 + 0.15 + 0.10)
  [OK] Bundles file exists: data/contrastive_bundles_v25.jsonl
  [OK] Frozen eval exists: evals/frozen_eval_v23.jsonl
  [WARN] Organic eval not specified
  [OK] Seeds are valid integers: [42, 123, 456]
```

### G3: Pool Composition Checker

**Checks**:
- Legacy pool not empty
- No duplicate bundle IDs across pools
- All bundles have required metadata fields
- Tier distribution matches expected

```python
def check_pool_composition(pools: Dict[str, List[Bundle]]) -> List[Alert]:
    alerts = []

    if len(pools.get('legacy', [])) == 0:
        alerts.append(Alert.SEVERE, "Legacy pool is empty!")

    all_ids = []
    for pool_name, bundles in pools.items():
        for b in bundles:
            if b.bundle_id in all_ids:
                alerts.append(Alert.ERROR, f"Duplicate bundle_id: {b.bundle_id}")
            all_ids.append(b.bundle_id)

    return alerts
```

## During-Training Guardrails

### G4: Quota Monitor

**Problem**: Tier caps may prevent achieving target tier distribution.

**Detection**: Log actual quota usage per batch.

**Example Output**:
```
Step 1000: quota=11/11/10 (tier1/tier2/tier3), loss=0.234
Step 1100: quota=10/12/10, loss=0.228
```

**Alert Condition**:
```python
def check_quota_utilization(actual: List[int], target: List[float], batch_size: int):
    for i, (a, t) in enumerate(zip(actual, target)):
        expected = batch_size * t
        if a < expected * 0.8:
            return Alert.WARNING, f"Tier{i+1} under-sampled: {a} vs expected {expected:.0f}"
```

### G5: Loss Anomaly Detector

**Problem**: Sudden loss spikes may indicate data issues or gradient explosions.

**Detection**:
```python
def check_loss_anomaly(loss_history: List[float], window: int = 100):
    if len(loss_history) < window:
        return None

    recent = loss_history[-window:]
    mean = sum(recent) / len(recent)
    std = (sum((x - mean)**2 for x in recent) / len(recent)) ** 0.5

    latest = loss_history[-1]
    if latest > mean + 3 * std:
        return Alert.WARNING, f"Loss spike: {latest:.4f} > {mean + 3*std:.4f}"
```

### G6: Gradient Health Monitor

**Checks**:
- Gradient norm within expected range
- No NaN gradients
- No vanishing gradients

```python
def check_gradient_health(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                return Alert.SEVERE, "NaN gradient detected"
            total_norm += p.grad.norm().item() ** 2

    total_norm = total_norm ** 0.5
    if total_norm < 1e-7:
        return Alert.WARNING, f"Vanishing gradient: {total_norm:.2e}"
    if total_norm > 100:
        return Alert.WARNING, f"Exploding gradient: {total_norm:.2f}"
```

## Post-Training Guardrails

### G7: Frozen Tier3 Regression Detector

**Problem**: New training may degrade performance on known hard cases.

**Detection**:
```python
def check_frozen_regression(
    current_scoreboard: Scoreboard,
    baseline_scoreboard: Scoreboard,
    threshold: float = 0.02
) -> Alert:
    current = current_scoreboard.by_tier['tier3_adversarial'].pass_rate
    baseline = baseline_scoreboard.by_tier['tier3_adversarial'].pass_rate
    delta = current - baseline

    if delta < -threshold:
        return Alert.CRITICAL, f"Frozen tier3 regression: {baseline:.1%} → {current:.1%} ({delta:+.1%})"
    elif delta < 0:
        return Alert.WARNING, f"Frozen tier3 slight decline: {delta:+.1%}"
    else:
        return Alert.OK, f"Frozen tier3 stable/improved: {delta:+.1%}"
```

**Configuration**:
```yaml
guardrails:
  regression_threshold: 0.02  # Alert if frozen tier3 drops > 2%
  baseline_scoreboard: results/baseline/frozen_scoreboard.json
```

### G8: Overfitting Cliff Detector

**Problem**: Longer training may improve frozen metrics while collapsing organic holdout.

**Detection** (requires step sweep data):
```python
def check_overfitting_cliff(
    step_results: Dict[int, Tuple[float, float]]  # {steps: (frozen_t3, organic_ho)}
) -> Alert:
    sorted_steps = sorted(step_results.keys())

    for i in range(1, len(sorted_steps)):
        prev_steps = sorted_steps[i-1]
        curr_steps = sorted_steps[i]

        prev_frozen, prev_organic = step_results[prev_steps]
        curr_frozen, curr_organic = step_results[curr_steps]

        frozen_delta = curr_frozen - prev_frozen
        organic_delta = curr_organic - prev_organic

        # Overfitting: frozen improves but organic drops
        if frozen_delta > 0.01 and organic_delta < -0.02:
            return Alert.WARNING, (
                f"Overfitting detected at {curr_steps} steps: "
                f"frozen +{frozen_delta:.1%}, organic {organic_delta:+.1%}"
            )
```

**Empirical Finding**: Validated overfitting cliff at 20k steps where organic holdout dropped from 31.5% to 28.5% while frozen tier3 improved.

### G9: Fingerprint Mismatch Detector

**Problem**: Comparing scoreboards from different eval sets produces meaningless results.

**Detection**:
```python
def check_fingerprint_match(
    scoreboard_a: Scoreboard,
    scoreboard_b: Scoreboard
) -> Alert:
    fp_a = scoreboard_a.eval_header.fingerprint
    fp_b = scoreboard_b.eval_header.fingerprint

    if fp_a != fp_b:
        return Alert.ERROR, (
            f"Fingerprint mismatch: {fp_a} vs {fp_b}. "
            f"Cannot directly compare these scoreboards."
        )
    return Alert.OK, "Fingerprints match"
```

### G10: Margin Distribution Shift Detector

**Problem**: Pass rate may stay stable while margin distribution shifts dangerously.

**Detection**:
```python
def check_margin_distribution(
    current: TierStats,
    baseline: TierStats,
    threshold: float = 0.05
) -> Alert:
    q10_delta = current.q10_margin - baseline.q10_margin

    if q10_delta < -threshold:
        return Alert.WARNING, (
            f"Q10 margin degraded: {baseline.q10_margin:.3f} → {current.q10_margin:.3f}. "
            f"Worst-case performance declining."
        )
```

## Frozen Eval Change Control

### Append-Only Policy

Frozen eval sets are append-only with versioned fingerprints:

1. **Never remove items** from frozen eval
2. **Version all additions**: `eval_v23.jsonl` → `eval_v24.jsonl`
3. **Maintain full history** in version control
4. **Record fingerprint changes** in changelog

### Version Tracking

```json
{
  "eval_versions": [
    {
      "version": "v23",
      "fingerprint": "a1b2c3d4|42|3786",
      "item_count": 3786,
      "created_at": "2026-01-01",
      "changes": "Initial frozen eval"
    },
    {
      "version": "v24",
      "fingerprint": "e5f6g7h8|42|3886",
      "item_count": 3886,
      "created_at": "2026-01-15",
      "changes": "Added 100 new tier3 from mining batch 7"
    }
  ]
}
```

### Comparison Rules

- **Same version**: Direct comparison allowed
- **Different versions**: Requires explicit `--allow-fingerprint-mismatch` flag
- **Breaking changes**: Document in changelog, notify stakeholders

## Alert Levels

| Level | Behavior | Example |
|-------|----------|---------|
| OK | Continue normally | "Legacy exposure 52.9. Healthy." |
| WARNING | Log warning, continue | "Q10 margin slightly degraded" |
| CRITICAL | Require acknowledgment | "Legacy exposure < 30" |
| SEVERE | Block without override | "Legacy pool is empty" |
| ERROR | Block, cannot override | "Fingerprint mismatch" |

## Configuration Reference

```yaml
guardrails:
  # Pre-training
  dilution_warning_threshold: 30
  dilution_block_threshold: 10
  dilution_override: false

  # Post-training
  regression_threshold: 0.02
  baseline_scoreboard: null  # Path to compare against
  overfitting_detection: true

  # Fingerprinting
  fingerprint_strict: true
  allow_fingerprint_mismatch: false

  # Alerting
  alert_on_warning: true
  block_on_critical: false
  block_on_severe: true
```

## CLI Integration

```bash
# Validate config and check guardrails before training
crucible train --config config.yaml --check-guardrails

# Run with guardrail override (use with caution)
crucible train --config config.yaml --guardrail-override dilution

# Compare scoreboards with regression check
crucible eval --checkpoint new.pt --eval-set eval.jsonl \
  --baseline-scoreboard baseline_scoreboard.json \
  --regression-threshold 0.02
```
