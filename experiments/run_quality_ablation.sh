#!/bin/bash
# Organic Quality Ablation: Ultra-only vs Full organic
# =====================================================
#
# Tests whether organic saturation is real, or whether
# we're diluting with medium-hard noise.
#
# Conditions:
#   ULTRA: R75_15_10 with only 8+ pass organic (365 bundles)
#   FULL:  R75_15_10 with all organic (1,037 bundles)
#
# Usage:
#   nohup ./experiments/run_quality_ablation.sh > results/quality_ablation.log 2>&1 &

set -e

WORKSPACE="/root/QUANTUM_BWD"
RESULTS_DIR="${WORKSPACE}/results/quality_ablation"
TRAINER="${WORKSPACE}/scripts/train_curriculum_v3.py"

# Bundle sets
ULTRA_BUNDLES="${WORKSPACE}/paradigm_factory/v2/bundles_v23/contrastive_bundles_v25_ultra.jsonl"
FULL_BUNDLES="${WORKSPACE}/paradigm_factory/v2/bundles_v23/contrastive_bundles_v25_organic.jsonl"

# Eval sets
FROZEN_EVAL="${WORKSPACE}/evals/eval_v23_contrastive_5k.jsonl"
ORGANIC_EVAL="${WORKSPACE}/evals/organic_holdout_v2_eval.jsonl"

# Fixed ratio (R75_15_10)
LEGACY=0.75
ORGANIC=0.15
EXPANDED=0.10

SEEDS=(42 123 456)
STEPS=5000

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

mkdir -p "$RESULTS_DIR"

log "=============================================="
log "ORGANIC QUALITY ABLATION"
log "=============================================="
log "Conditions: ULTRA (365 organic) vs FULL (1037 organic)"
log "Ratio: R75_15_10 (legacy=$LEGACY, organic=$ORGANIC, expanded=$EXPANDED)"
log "Seeds: ${SEEDS[*]}"
log "Steps: ${STEPS}"

# Phase 1: Training
log ""
log "PHASE 1: TRAINING"
log "=============================================="

# Train ULTRA condition
for seed in "${SEEDS[@]}"; do
    output_dir="${RESULTS_DIR}/ULTRA/seed_${seed}"

    if [ -f "${output_dir}/checkpoint_final.pt" ]; then
        log "[SKIP] ULTRA seed ${seed} already trained"
        continue
    fi

    mkdir -p "$output_dir"
    log "Training ULTRA seed ${seed}..."

    python3 "$TRAINER" \
        --bundles "$ULTRA_BUNDLES" \
        --steps "$STEPS" \
        --seed "$seed" \
        --curriculum uniform \
        --device cuda \
        --tier3-legacy "$LEGACY" \
        --tier3-organic "$ORGANIC" \
        --tier3-expanded "$EXPANDED" \
        --output-dir "$output_dir" \
        2>&1 | tee "${output_dir}/train.log"

    log "[OK] ULTRA seed ${seed} complete"
done

# Train FULL condition
for seed in "${SEEDS[@]}"; do
    output_dir="${RESULTS_DIR}/FULL/seed_${seed}"

    if [ -f "${output_dir}/checkpoint_final.pt" ]; then
        log "[SKIP] FULL seed ${seed} already trained"
        continue
    fi

    mkdir -p "$output_dir"
    log "Training FULL seed ${seed}..."

    python3 "$TRAINER" \
        --bundles "$FULL_BUNDLES" \
        --steps "$STEPS" \
        --seed "$seed" \
        --curriculum uniform \
        --device cuda \
        --tier3-legacy "$LEGACY" \
        --tier3-organic "$ORGANIC" \
        --tier3-expanded "$EXPANDED" \
        --output-dir "$output_dir" \
        2>&1 | tee "${output_dir}/train.log"

    log "[OK] FULL seed ${seed} complete"
done

# Phase 2: Frozen eval
log ""
log "PHASE 2: FROZEN EVAL"
log "=============================================="

python3 "${WORKSPACE}/experiments/eval_capsules.py" \
    --eval "$FROZEN_EVAL" \
    --results-root "$RESULTS_DIR" \
    --device cuda \
    --out "${RESULTS_DIR}/frozen_scoreboard.json"

log "[OK] Frozen eval complete"

# Phase 3: Organic holdout eval
log ""
log "PHASE 3: ORGANIC HOLDOUT EVAL"
log "=============================================="

python3 "${WORKSPACE}/experiments/eval_capsules.py" \
    --eval "$ORGANIC_EVAL" \
    --results-root "$RESULTS_DIR" \
    --device cuda \
    --out "${RESULTS_DIR}/organic_v2_scoreboard.json"

log "[OK] Organic holdout eval complete"

# Phase 4: Summary
log ""
log "=============================================="
log "QUALITY ABLATION RESULTS"
log "=============================================="

python3 << 'PYEOF'
import json

frozen_path = '/root/QUANTUM_BWD/results/quality_ablation/frozen_scoreboard.json'
organic_path = '/root/QUANTUM_BWD/results/quality_ablation/organic_v2_scoreboard.json'

try:
    with open(frozen_path) as f:
        frozen = json.load(f)
    with open(organic_path) as f:
        organic = json.load(f)
except Exception as e:
    print(f"Could not load scoreboards: {e}")
    exit(1)

print()
print("=" * 70)
print("QUALITY ABLATION: ULTRA vs FULL")
print("=" * 70)
print(f"{'Condition':<12} {'Frozen T3':>12} {'Organic HO':>12} {'Delta':>10}")
print("-" * 70)

def get_tier3_score(data, condition):
    capsules = [c for c in data.get('capsules', [])
                if c.get('condition') == condition and 'error' not in c]
    if not capsules:
        return 0, 0
    scores = [c.get('by_tier', {}).get('tier3_adversarial', {}).get('pass_rate', 0)
              for c in capsules]
    mean = sum(scores) / len(scores) * 100
    std = (sum((s*100 - mean)**2 for s in [x/100 for x in [mean]*len(scores)])**0.5) if len(scores) > 1 else 0
    return mean, std

ultra_frozen, _ = get_tier3_score(frozen, 'ULTRA')
full_frozen, _ = get_tier3_score(frozen, 'FULL')

# For organic holdout, all items are tier3
def get_organic_score(data, condition):
    capsules = [c for c in data.get('capsules', [])
                if c.get('condition') == condition and 'error' not in c]
    if not capsules:
        return 0, 0
    scores = [c.get('overall', {}).get('accuracy', 0) for c in capsules]
    mean = sum(scores) / len(scores) * 100
    return mean, 0

ultra_organic, _ = get_organic_score(organic, 'ULTRA')
full_organic, _ = get_organic_score(organic, 'FULL')

delta_frozen = ultra_frozen - full_frozen
delta_organic = ultra_organic - full_organic

print(f"{'ULTRA':<12} {ultra_frozen:>11.1f}% {ultra_organic:>11.1f}%")
print(f"{'FULL':<12} {full_frozen:>11.1f}% {full_organic:>11.1f}%")
print("-" * 70)
print(f"{'DELTA':<12} {delta_frozen:>+11.1f}% {delta_organic:>+11.1f}%")
print()

print("INTERPRETATION:")
if delta_organic > 1.0:
    print("  [WIN] Ultra-only LIFTS organic holdout → bottleneck was noise dilution")
    print("  → Recommend: Use ultra-only organic, or curriculum: ultra early → full later")
elif delta_organic < -1.0:
    print("  [LOSS] Full organic BEATS ultra-only → model benefits from volume diversity")
    print("  → Recommend: Keep full organic, explore longer training")
else:
    print("  [FLAT] No meaningful difference → saturation is real")
    print("  → Recommend: Next lever is training dynamics or model capacity")

if abs(delta_frozen) > 1.0:
    if delta_frozen > 0:
        print(f"  [BONUS] Ultra also improves frozen tier3 by {delta_frozen:.1f}%")
    else:
        print(f"  [COST] Ultra hurts frozen tier3 by {abs(delta_frozen):.1f}%")
else:
    print(f"  [STABLE] Frozen tier3 unchanged (delta={delta_frozen:+.1f}%)")
PYEOF

log ""
log "=============================================="
log "ABLATION COMPLETE"
log "=============================================="
log "Results: ${RESULTS_DIR}/"
log "  - frozen_scoreboard.json"
log "  - organic_v2_scoreboard.json"
