# QLLM SemanticPhase Outcome Roadmap

## The Decisive Metric: Negative-Gap Rate (NGR)

```
NGR = (# of hard-neg pairs where sim(anchor, hard_neg) > sim(anchor, pos)) / total hard pairs
```

**Compute at steps 500 and 1000.** This number tells you which universe you're in.

---

## Three Branches

### Branch A: Optimization Land (NGR < 15%)
**Signal:** Most hard negatives have correct ordering; model just needs tighter margins.

| Metric | Threshold |
|--------|-----------|
| NGR at step 500 | < 15% |
| NGR at step 1000 | < 15% |
| Near-miss rate | > 40% |

**Actions:**
1. Continue current architecture
2. Increase slack weight (try 2.0, 2.5)
3. Extend late-stage training
4. Progressive margin curriculum (0.02 → 0.05 → 0.08)

**Expected outcome:** 3-5 more epochs should push hard slack positive.

---

### Branch B: Representation Land (15% < NGR < 40%)
**Signal:** Significant minority of pairs have inverted ordering. Architecture needs help.

| Metric | Threshold |
|--------|-----------|
| NGR at step 500 | 15-40% |
| Gap collapse rate | > 50% from step 1 |
| Context utilization | Unknown |

**Actions:**
1. **Add context-conditioned projection head**
   - Keep CE in main embedding space
   - Compute slack in projected space conditioned on context
   - Allows separation without fighting CE's clustering

2. **Curriculum isolation**
   - Phase 1: Train only on NGR=0 pairs (correct ordering)
   - Phase 2: Add near-misses (0 < gap < margin)
   - Phase 3: Reintroduce inversions with multi-prototype fallback

3. **Diagnostic:** Check if inversions share surface-cue patterns

**Expected outcome:** NGR should drop to <15% within 2 epochs of Phase 1.

---

### Branch C: Data Land (NGR > 40%)
**Signal:** Nearly half of "hard" pairs are fundamentally confused. Data or task definition issue.

| Metric | Threshold |
|--------|-----------|
| NGR at step 500 | > 40% |
| Inversion stability | Same pairs invert across runs |
| Human ambiguity rate | > 30% on manual review |

**Actions:**
1. **Build killer-negative dossier** (see below)
2. **Manual audit of top 20 inversions**
   - If human-ambiguous → reclassify as "un-certifiable" or require richer context
   - If labeling error → fix bundle definitions
   - If context-dependent → flag for multi-prototype treatment

3. **Tiered certification**
   - Tier 1: Easy + clear hard (gap > 0.10) → full certification
   - Tier 2: Near-misses → probabilistic certification
   - Tier 3: Inversions → require multi-prototype or exclude

4. **Consider multi-prototype per sense cluster**
   - K=4 prototypes per lemma
   - Positive matches best prototype
   - Hard negative competes against same prototype
   - Allows "party#political" and "party#participant" to live in different regions

**Expected outcome:** After audit, NGR on "certifiable" subset should drop to <20%.

---

## Current Status: Branch C (NGR = 46%)

Based on v3 and φ-balanced experiments:
- NGR at late stage: **46%** (53/115 killer negatives inverted)
- Same inversions appear in both runs → stable, not random
- Full sentence context available but not preventing collapse

**Immediate next step:** Build dossier, audit 10 inversions manually.

---

## Early Warning Dashboard

Add this to every training run at steps 500 and 1000:

```python
def compute_early_warning(killers_so_far):
    if len(killers_so_far) == 0:
        return None

    inversions = sum(1 for k in killers_so_far if k['gap'] < 0)
    near_misses = sum(1 for k in killers_so_far if 0 <= k['gap'] < 0.05)
    adequate = sum(1 for k in killers_so_far if k['gap'] >= 0.05)

    ngr = inversions / len(killers_so_far)

    if ngr < 0.15:
        branch = "A (Optimization)"
    elif ngr < 0.40:
        branch = "B (Representation)"
    else:
        branch = "C (Data)"

    print(f"=== EARLY WARNING @ step {step} ===")
    print(f"NGR: {ngr*100:.1f}% → Branch {branch}")
    print(f"  Inversions: {inversions}, Near-misses: {near_misses}, Adequate: {adequate}")

    return branch
```

---

## Decision Tree

```
Step 500: Compute NGR
    │
    ├─ NGR < 15% ──────────────► Branch A: Tune hyperparameters
    │
    ├─ 15% ≤ NGR < 40% ────────► Branch B: Add projection head + curriculum
    │
    └─ NGR ≥ 40% ──────────────► Branch C: Audit data, consider multi-prototype
                                          │
                                          ├─ If audit shows labeling errors → Fix data
                                          │
                                          ├─ If human-ambiguous → Reclassify tier
                                          │
                                          └─ If context-dependent → Multi-prototype
```

---

## Architectural Options (Ranked by Minimal Change)

### Option 1: Context-Conditioned Projection Head (Recommended)
- **Change:** Add small MLP that projects embedding conditioned on context representation
- **Keep:** Main embedding, CE loss, contrastive structure
- **Compute slack in:** Projected space
- **Why it helps:** CE clusters in main space; projection learns to separate for certification

```python
class SenseProjector(nn.Module):
    def __init__(self, dim=768, proj_dim=256):
        self.context_gate = nn.Linear(dim, proj_dim)
        self.projector = nn.Linear(dim, proj_dim)

    def forward(self, embedding, context_repr):
        gate = torch.sigmoid(self.context_gate(context_repr))
        return gate * self.projector(embedding)
```

### Option 2: Multi-Prototype per Lemma
- **Change:** K embeddings per lemma instead of 1
- **Training:** Positive picks best-matching prototype; hard neg competes against same one
- **Why it helps:** Different senses can "park" in different prototypes

### Option 3: Curriculum with Inversion Isolation
- **Change:** Training schedule only
- **Phase 1:** Exclude inverted pairs entirely
- **Phase 2:** Add near-misses once base geometry stable
- **Phase 3:** Attack inversions with specialized loss

---

## Success Criteria

| Metric | Current | Target (Branch A) | Target (Branch B/C) |
|--------|---------|-------------------|---------------------|
| NGR at step 1000 | 46% | < 15% | < 25% after intervention |
| Late-stage positive hard slack | 0.1% | > 30% | > 20% on certifiable subset |
| Final hard slack | -0.15 | > 0.00 | > 0.00 on Tier 1 |
| Killer positive rate | 2.6% | > 25% | > 15% |

---

*Last updated: 2026-01-03*
*Based on: v3 + φ-balanced comparison runs*
