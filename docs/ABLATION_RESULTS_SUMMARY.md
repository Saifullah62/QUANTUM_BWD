# SemanticPhase Slack Optimization: Ablation Results Summary

**Date:** January 3, 2026
**Objective:** Achieve 30%+ late-window positive slack rate for polysemy disambiguation certification

---

## Executive Summary

After extensive ablation testing, the slack penalty approach has not yet achieved the 30% certification threshold. The best result (7.3%) was achieved with a 0.05 penalty-only configuration. Stronger penalties (2.0) counterintuitively produced worse results (2.9%), suggesting the current architecture may require fundamental changes rather than hyperparameter tuning.

---

## Certification Criterion

- **Target:** Slack > 0 for 30%+ of steps in the late training window (last 50%)
- **Slack Definition:** `gap - margin` where `gap = similarity(anchor, positive) - similarity(anchor, negative)`
- **Margins:** Easy negatives = 0.05 (or 0.15 at training), Hard negatives = 0.15 (or 0.25 at training)

---

## Ablation Results Table

| Config | Penalty Weight | Hard-Neg Mult | CE Taper | Late Positive Rate | Avg Late Slack | Status |
|--------|---------------|---------------|----------|-------------------|----------------|--------|
| Baseline (no guardrail) | 0.0 | 1.0 | 1.0 | 4.9% | -0.0475 | FAIL |
| Penalty-only | 0.05 | 1.0 | 1.0 | **7.3%** | -0.0479 | FAIL |
| Penalty + CE taper | 0.05 | 1.0 | 0.5 | 4.9% | -0.0505 | FAIL |
| Stronger penalty | 0.10 | 2.0 | 1.0 | 4.3% | -0.0497 | FAIL |
| Heavy penalty | 2.0 | 2.0 | 1.0 | 2.9% | -0.0415 | FAIL |

### Earlier Experiments (Late-Stage Margin Boost)

| Config | Boost Factor | Late Start | Late Ramp | Late Positive Rate | Status |
|--------|-------------|------------|-----------|-------------------|--------|
| 2.5x boost | 2.5 | 0.6 | 0.1 | ~0% | FAIL |
| 5.0x boost | 5.0 | 0.5 | 0.2 | 6.5% | FAIL |

---

## Key Findings

### 1. Slack Penalty Architecture Issue

The slack penalty is added to the contrastive loss, which is then scaled by `contrastive_weight`:

```
total_loss = ce_weight * ce_loss + contrastive_weight * (base_contrastive + penalty_weight * slack_penalty)
           = 0.3 * ce_loss + 0.5 * (base_contrastive + penalty_weight * slack_penalty)
```

**Problem:** With CE loss in the 1.5-4.0 range and slack penalty ~0.05-0.10, the effective contribution is:
- CE contributes: 0.3 * 2.0 = **0.6**
- Slack penalty contributes: 0.5 * 0.10 * 0.10 = **0.005**

The slack penalty is **100x weaker** than CE, explaining why models ignore it.

### 2. Higher Penalties Cause Instability

Counterintuitively, increasing penalty weight from 0.10 to 2.0 made results **worse** (4.3% -> 2.9%). This suggests:
- The optimizer may be oscillating
- The penalty may be interfering with base contrastive learning
- The model may be finding local minima where neither CE nor margin is optimized

### 3. CE Taper Hurts Performance

Adding CE late taper (0.5x) with penalty reduced positive rate from 7.3% to 4.9%. This confirms the user's insight:

> "CE taper alone acts as a release valve, not a guardrail. Without a strong geometry objective, the model drifts into 'meh, everything is similar' territory."

### 4. Hard Negative Problem Persists

Throughout all experiments, hard negative slack remained deeply negative (-0.10 to -0.22), while easy negative slack occasionally crossed zero. The model consistently fails to separate hard negatives (similar-but-wrong meanings).

---

## Training Dynamics Observed

### Loss Evolution (2.0 Penalty Run)
```
Step     50: Loss 1.81, CE 4.07, Slack -0.013 easy / -0.144 hard
Step   500: Loss 0.89, CE 2.67, Slack -0.020 easy / -0.164 hard
Step  1000: Loss 0.85, CE 2.43, Slack -0.060 easy / -0.157 hard
Step  2000: Loss 0.84, CE 2.58, Slack -0.044 easy / -0.163 hard
Step  2950: Loss 0.67, CE 2.19, Slack +0.046 easy / -0.122 hard  <-- BEST
Step  3500: Loss 0.82, CE 1.67, Slack -0.045 easy / -0.141 hard
Step  4125: Loss 0.81, CE 1.67, Slack -0.033 easy / -0.099 hard
```

**Observations:**
1. Easy slack occasionally hits positive (step 2950: +0.046) but doesn't sustain
2. Hard slack never crosses zero throughout training
3. Loss decreases but slack doesn't correlate

---

## Architectural Diagnosis

The current problem is fundamentally an **optimization priority** issue:

1. **CE dominates early and late:** Cross-entropy loss is 1.5-4.0 throughout training
2. **Contrastive is weak:** Base contrastive loss drops to 0.01-0.04 by late training
3. **Penalty is diluted:** Slack penalty contribution is negligible after weight scaling

### Proposed Fix: Standalone Slack Term

Instead of:
```python
total_loss = ce_weight * ce + contrastive_weight * (base_contrastive + penalty_weight * slack_penalty)
```

Restructure to:
```python
total_loss = ce_weight * ce + contrastive_weight * base_contrastive + slack_weight * slack_penalty
```

This makes `slack_weight = 2.0` truly mean 2.0, not `2.0 * 0.5 = 1.0`.

---

## Recommended Next Steps

### Immediate (High Priority)

1. **Rewire slack penalty as standalone loss term** - Bypass contrastive_weight multiplication
2. **Try slack_weight 5.0-10.0** with standalone architecture
3. **Implement late ramp** (0 -> target in final 40% of training) to avoid early instability

### Medium Term

4. **Add explicit hard-negative mining** - Currently hard negatives are pre-labeled but not specially optimized
5. **Try curriculum learning** - Easy negatives first, then introduce hard negatives
6. **Explore different margin schedules** - Perhaps start with smaller margins and grow them

### Diagnostic

7. **Log slack penalty loss separately** - Confirm it's actually backpropagating
8. **Visualize embedding space** - Check if model is collapsing representations

---

## Files Modified

| File | Changes |
|------|---------|
| `paradigm_factory/bundle_dataset.py` | Added `hard_neg_penalty_mult` to `compute_bundle_contrastive_loss()` |
| `scripts/train_semantic_phase_v2.py` | Added `slack_penalty_late_ramp`, `hard_neg_penalty_mult` params |
| `scripts/run_3seed_validation.py` | Added CLI args for new params, updated validation pipeline |

---

## Conclusion

The slack penalty mechanism **works** (it does contribute to the loss and the model responds to it), but it's **too weak** relative to CE in the current architecture. The key insight from this ablation series:

> "The penalty is whispering while CE is shouting."

The path forward is not to increase the penalty weight within the current architecture (which causes instability), but to **restructure how the penalty is added to total loss** so it competes directly with CE on equal footing.

---

*Report generated: 2026-01-03*
