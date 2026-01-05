#!/bin/bash
# Step Sweep: Training Duration vs Capacity
# ==========================================
#
# Tests whether the ~30% organic holdout plateau is:
#   - Training duration limited (more steps = better)
#   - Capacity limited (more steps = flat)
#
# Conditions: 5k, 10k, 20k steps at R75_15_10
#
# Usage:
#   nohup ./experiments/run_step_sweep_overnight.sh > results/step_sweep.log 2>&1 &

set -euo pipefail

ROOT="/root/QUANTUM_BWD"
BUNDLES="${ROOT}/paradigm_factory/v2/bundles_v23/contrastive_bundles_v25_organic.jsonl"
TRAINER="${ROOT}/scripts/train_curriculum_v3.py"
EVAL_FROZEN="${ROOT}/evals/eval_v23_contrastive_5k.jsonl"
EVAL_ORGANIC="${ROOT}/evals/organic_holdout_v2_eval.jsonl"
OUTDIR="${ROOT}/results/step_sweep"

mkdir -p "${OUTDIR}"

# Fixed ratio (R75_15_10)
LEGACY=0.75
ORGANIC=0.15
EXPANDED=0.10

# Sweep parameters
SEEDS=(42 123)
STEPS_LIST=(5000 10000 20000)

CURRIC="uniform"
LR="1e-3"
BS=32

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=============================================="
log "STEP SWEEP: TRAINING DURATION vs CAPACITY"
log "=============================================="
log "Bundles: ${BUNDLES}"
log "Steps: ${STEPS_LIST[*]}"
log "Seeds: ${SEEDS[*]}"
log "Ratio: R75_15_10 (legacy=${LEGACY}, organic=${ORGANIC}, expanded=${EXPANDED})"

# Phase 1: Training
log ""
log "PHASE 1: TRAINING"
log "=============================================="

for steps in "${STEPS_LIST[@]}"; do
    for seed in "${SEEDS[@]}"; do
        cond="S${steps}"
        run="${OUTDIR}/${cond}/seed_${seed}"

        if [ -f "${run}/checkpoint_final.pt" ]; then
            log "[SKIP] ${cond} seed ${seed} already trained"
            continue
        fi

        mkdir -p "${run}"
        log "Training ${cond} seed ${seed}..."

        python3 "${TRAINER}" \
            --bundles "${BUNDLES}" \
            --steps "${steps}" \
            --seed "${seed}" \
            --batch-size "${BS}" \
            --lr "${LR}" \
            --curriculum "${CURRIC}" \
            --device cuda \
            --tier3-legacy "${LEGACY}" \
            --tier3-organic "${ORGANIC}" \
            --tier3-expanded "${EXPANDED}" \
            --output-dir "${run}" \
            2>&1 | tee "${run}/train.log"

        log "[OK] ${cond} seed ${seed} complete"
    done
done

# Phase 2: Frozen eval
log ""
log "PHASE 2: FROZEN EVAL"
log "=============================================="

python3 "${ROOT}/experiments/eval_capsules.py" \
    --eval "${EVAL_FROZEN}" \
    --results-root "${OUTDIR}" \
    --device cuda \
    --out "${OUTDIR}/frozen_scoreboard.json" \
    2>&1 | tee "${OUTDIR}/eval_frozen.log"

log "[OK] Frozen eval complete"

# Phase 3: Organic holdout eval
log ""
log "PHASE 3: ORGANIC HOLDOUT EVAL"
log "=============================================="

python3 "${ROOT}/experiments/eval_capsules.py" \
    --eval "${EVAL_ORGANIC}" \
    --results-root "${OUTDIR}" \
    --device cuda \
    --out "${OUTDIR}/organic_v2_scoreboard.json" \
    2>&1 | tee "${OUTDIR}/eval_organic.log"

log "[OK] Organic holdout eval complete"

# Phase 4: Summary
log ""
log "=============================================="
log "STEP SWEEP RESULTS"
log "=============================================="

python3 << 'PYEOF'
import json

frozen_path = '/root/QUANTUM_BWD/results/step_sweep/frozen_scoreboard.json'
organic_path = '/root/QUANTUM_BWD/results/step_sweep/organic_v2_scoreboard.json'

try:
    with open(frozen_path) as f:
        frozen = json.load(f)
    with open(organic_path) as f:
        organic = json.load(f)
except Exception as e:
    print(f"Could not load scoreboards: {e}")
    exit(1)

def get_tier3_score(data, condition):
    capsules = [c for c in data.get('capsules', [])
                if c.get('condition') == condition and 'error' not in c]
    if not capsules:
        return 0, 0
    scores = [c.get('by_tier', {}).get('tier3_adversarial', {}).get('pass_rate', 0)
              for c in capsules]
    mean = sum(scores) / len(scores) * 100 if scores else 0
    return mean, len(scores)

def get_organic_score(data, condition):
    capsules = [c for c in data.get('capsules', [])
                if c.get('condition') == condition and 'error' not in c]
    if not capsules:
        return 0, 0
    scores = [c.get('overall', {}).get('accuracy', 0) for c in capsules]
    mean = sum(scores) / len(scores) * 100 if scores else 0
    return mean, len(scores)

print()
print("=" * 70)
print("STEP SWEEP: TRAINING DURATION vs CAPACITY")
print("=" * 70)
print(f"{'Steps':<10} {'Frozen T3':>12} {'Organic HO':>12} {'Δ from 5k':>12}")
print("-" * 70)

baseline_frozen = None
baseline_organic = None

for steps in [5000, 10000, 20000]:
    cond = f"S{steps}"
    frozen_score, _ = get_tier3_score(frozen, cond)
    organic_score, _ = get_organic_score(organic, cond)

    if baseline_frozen is None:
        baseline_frozen = frozen_score
        baseline_organic = organic_score
        delta_str = "-"
    else:
        delta_organic = organic_score - baseline_organic
        delta_str = f"{delta_organic:+.1f}%"

    print(f"{steps:<10} {frozen_score:>11.1f}% {organic_score:>11.1f}% {delta_str:>12}")

print("-" * 70)
print()

# Get final scores for interpretation
s5k_organic, _ = get_organic_score(organic, "S5000")
s10k_organic, _ = get_organic_score(organic, "S10000")
s20k_organic, _ = get_organic_score(organic, "S20000")

s5k_frozen, _ = get_tier3_score(frozen, "S5000")
s20k_frozen, _ = get_tier3_score(frozen, "S20000")

lift_10k = s10k_organic - s5k_organic
lift_20k = s20k_organic - s5k_organic
frozen_delta = s20k_frozen - s5k_frozen

print("INTERPRETATION:")
if lift_20k > 2.0:
    print(f"  [WIN] Organic holdout improves with steps (+{lift_20k:.1f}% at 20k)")
    print("  → Plateau was training duration. Consider even longer runs (30k+).")
    if frozen_delta < -2.0:
        print(f"  [COST] But frozen tier3 drops ({frozen_delta:.1f}%). Watch for overfitting.")
    else:
        print(f"  [STABLE] Frozen tier3 holds ({frozen_delta:+.1f}%). Good scaling.")
elif lift_20k < -2.0:
    print(f"  [OVERFIT] Organic holdout DROPS at longer training ({lift_20k:.1f}%)")
    print("  → Model is overfitting. 5k steps may be optimal, or add regularization.")
else:
    print(f"  [FLAT] No meaningful lift from more steps ({lift_20k:+.1f}%)")
    print("  → Plateau is capacity-limited, not duration-limited.")
    print("  → Next lever: model size (embed_dim) or targeted mining.")

if abs(lift_10k - lift_20k) < 1.0 and lift_10k > 1.0:
    print(f"  [DIMINISHING] 10k→20k shows diminishing returns. 10k may be sweet spot.")
PYEOF

log ""
log "=============================================="
log "STEP SWEEP COMPLETE"
log "=============================================="
log "Results: ${OUTDIR}/"
log "  - frozen_scoreboard.json"
log "  - organic_v2_scoreboard.json"
