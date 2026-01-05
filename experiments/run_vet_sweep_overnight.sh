#!/bin/bash
# Overnight Vet Sweep: Test different keep rates
# ===============================================
#
# Tests: 15%, 25%, 35%, 45%, 60% keep rates with 90% core mix
# This determines the optimal vetting threshold for tier3x expansion.
#
# Usage:
#   nohup ./experiments/run_vet_sweep_overnight.sh > results/vet_sweep_nohup.log 2>&1 &

set -e

WORKSPACE="/root/QUANTUM_BWD"
RESULTS_DIR="${WORKSPACE}/results/vet_sweep"

# Source bundles
CORE_BUNDLES="${WORKSPACE}/paradigm_factory/v2/bundles_v23/contrastive_bundles.jsonl"
TIER3X_BUNDLES="${WORKSPACE}/paradigm_factory/v2/bundles_v23/contrastive_bundles_v23_tier3x.jsonl"

# Vetter script
VETTER="${WORKSPACE}/paradigm_factory/v2/vet_tier3x_sbert.py"

# Training script
TRAINER="${WORKSPACE}/scripts/train_curriculum_v2.py"

# Eval set
EVAL_SET="${WORKSPACE}/evals/eval_v23_contrastive_5k.jsonl"

# Sweep parameters
KEEPS=(0.15 0.25 0.35 0.45 0.60)
TIER3_MIX=0.90  # 90% core, 10% vetted
SEEDS=(42 123 456)
STEPS=5000

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

mkdir -p "$RESULTS_DIR"

log "=============================================="
log "OVERNIGHT VET SWEEP"
log "=============================================="
log "Keep rates: ${KEEPS[*]}"
log "Core mix: ${TIER3_MIX}"
log "Seeds: ${SEEDS[*]}"
log "Steps: ${STEPS}"

# Step 1: Create vetted bundle variants for each keep rate
log ""
log "PHASE 1: CREATING VETTED BUNDLES"
log "=============================================="

for keep in "${KEEPS[@]}"; do
    keep_pct=$(printf "%.0f" "$(echo "$keep * 100" | bc)")
    vetted_file="${WORKSPACE}/paradigm_factory/v2/bundles_v23/contrastive_bundles_v23_tier3x_vetted${keep_pct}.jsonl"

    if [ -f "$vetted_file" ]; then
        log "[SKIP] vetted${keep_pct} already exists"
        continue
    fi

    log "Creating vetted${keep_pct} (keep=${keep})..."
    # Use min-margin=-1.0 to keep top bundles by relative margin (SBERT shows negative margins on these hard cases)
    python3 "$VETTER" \
        --bundles "$TIER3X_BUNDLES" \
        --out "$vetted_file" \
        --keep-frac "$keep" \
        --min-margin -1.0

    log "[OK] Created: $vetted_file"
done

# Step 2: Train each condition
log ""
log "PHASE 2: TRAINING CONDITIONS"
log "=============================================="

for keep in "${KEEPS[@]}"; do
    keep_pct=$(printf "%.0f" "$(echo "$keep * 100" | bc)")
    vetted_file="${WORKSPACE}/paradigm_factory/v2/bundles_v23/contrastive_bundles_v23_tier3x_vetted${keep_pct}.jsonl"

    for seed in "${SEEDS[@]}"; do
        cond="vet${keep_pct}"
        output_dir="${RESULTS_DIR}/${cond}/seed_${seed}"

        if [ -f "${output_dir}/checkpoint_final.pt" ]; then
            log "[SKIP] ${cond} seed ${seed} already trained"
            continue
        fi

        mkdir -p "$output_dir"
        log "Training ${cond} seed ${seed}..."

        # Use vetted bundles - they already have tier3 core + filtered expanded
        python3 "$TRAINER" \
            --bundles "$vetted_file" \
            --tier3-mix "$TIER3_MIX" \
            --steps "$STEPS" \
            --seed "$seed" \
            --curriculum uniform \
            --device cuda \
            --output-dir "$output_dir" \
            2>&1 | tee "${output_dir}/train.log"

        log "[OK] Completed ${cond} seed ${seed}"
    done
done

# Step 3: Evaluate all capsules
log ""
log "PHASE 3: EVALUATION"
log "=============================================="

python3 "${WORKSPACE}/experiments/eval_capsules.py" \
    --eval "$EVAL_SET" \
    --results-root "$RESULTS_DIR" \
    --device cuda \
    --out "${RESULTS_DIR}/vet_sweep_scoreboard.json"

log "[OK] Evaluation complete"

# Step 4: Print summary
log ""
log "=============================================="
log "VET SWEEP RESULTS"
log "=============================================="

python3 << 'PYEOF'
import json
import sys

try:
    with open('/root/QUANTUM_BWD/results/vet_sweep/vet_sweep_scoreboard.json') as f:
        data = json.load(f)
except Exception as e:
    print(f"Could not load scoreboard: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("CONDITION COMPARISON")
print("=" * 70)
print(f"{'Condition':<12} {'Accuracy':>10} {'Margin':>10} {'T3 Pass':>10} {'T3 Margin':>12}")
print("-" * 70)

for cond in ['vet15', 'vet25', 'vet35', 'vet45', 'vet60']:
    stats = data.get('summary', {}).get(cond, {})
    if stats:
        acc = stats.get('mean_accuracy', 0) * 100
        margin = stats.get('mean_margin', 0)

        # Get tier3 stats
        capsules = [c for c in data.get('capsules', [])
                    if c.get('condition') == cond and 'error' not in c]
        if capsules:
            t3_passes = [c.get('by_tier', {}).get('tier3', {}).get('pass_rate', 0) for c in capsules]
            t3_margins = [c.get('by_tier', {}).get('tier3', {}).get('median_margin', 0) for c in capsules]
            t3_pass = sum(t3_passes) / len(t3_passes) * 100 if t3_passes else 0
            t3_margin = sum(t3_margins) / len(t3_margins) if t3_margins else 0
        else:
            t3_pass = 0
            t3_margin = 0

        print(f"{cond:<12} {acc:>9.1f}% {margin:>10.4f} {t3_pass:>9.1f}% {t3_margin:>12.4f}")

print("-" * 70)
print()
print("DECISION RULES:")
print("  - Higher T3 Pass + Margin = better vetting threshold")
print("  - Look for the sweet spot: T3 gains without T2 regression")
print("  - Compare to D4 baseline (90/10 with vet35)")
PYEOF

log ""
log "=============================================="
log "SWEEP COMPLETE"
log "=============================================="
log "Results saved to: ${RESULTS_DIR}/vet_sweep_scoreboard.json"
