#!/bin/bash
# Overnight Ratio Sweep: Test tier3 pool allocations
# =================================================
#
# Tests legacy/organic/expanded ratios to find the Pareto frontier:
# - Maximize organic holdout (generalization to new hard patterns)
# - While keeping frozen tier3 above acceptable floor
#
# Ratios tested: 85/10/5, 80/15/5, 75/15/10, 70/20/10, 65/25/10
#
# Usage:
#   nohup ./experiments/run_ratio_sweep_overnight.sh > results/ratio_sweep_nohup.log 2>&1 &

set -e

WORKSPACE="/root/QUANTUM_BWD"
RESULTS_DIR="${WORKSPACE}/results/ratio_sweep"

# Source bundles (v24 with 300 organic tier3 merged)
BUNDLES="${WORKSPACE}/paradigm_factory/v2/bundles_v23/contrastive_bundles_v24_organic.jsonl"

# Training script (v3 with three-pool support)
TRAINER="${WORKSPACE}/scripts/train_curriculum_v3.py"

# Eval sets
FROZEN_EVAL="${WORKSPACE}/evals/eval_v23_contrastive_5k.jsonl"
ORGANIC_EVAL="${WORKSPACE}/evals/organic_holdout_eval.jsonl"

# Sweep parameters
# Format: "legacy_organic_expanded"
RATIOS=("85_10_5" "80_15_5" "75_15_10" "70_20_10" "65_25_10")
SEEDS=(42 123 456)
STEPS=5000

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

mkdir -p "$RESULTS_DIR"

log "=============================================="
log "OVERNIGHT RATIO SWEEP"
log "=============================================="
log "Ratios: ${RATIOS[*]}"
log "Seeds: ${SEEDS[*]}"
log "Steps: ${STEPS}"

# Phase 1: Training
log ""
log "PHASE 1: TRAINING ALL CONDITIONS"
log "=============================================="

for ratio in "${RATIOS[@]}"; do
    # Parse ratio string
    IFS='_' read -r legacy organic expanded <<< "$ratio"

    # Convert to decimals
    legacy_dec=$(echo "scale=2; $legacy/100" | bc)
    organic_dec=$(echo "scale=2; $organic/100" | bc)
    expanded_dec=$(echo "scale=2; $expanded/100" | bc)

    for seed in "${SEEDS[@]}"; do
        cond="R${ratio}"
        output_dir="${RESULTS_DIR}/${cond}/seed_${seed}"

        if [ -f "${output_dir}/checkpoint_final.pt" ]; then
            log "[SKIP] ${cond} seed ${seed} already trained"
            continue
        fi

        mkdir -p "$output_dir"
        log "Training ${cond} seed ${seed} (legacy=${legacy_dec}, organic=${organic_dec}, expanded=${expanded_dec})..."

        python3 "$TRAINER" \
            --bundles "$BUNDLES" \
            --steps "$STEPS" \
            --seed "$seed" \
            --curriculum uniform \
            --device cuda \
            --tier3-legacy "$legacy_dec" \
            --tier3-organic "$organic_dec" \
            --tier3-expanded "$expanded_dec" \
            --output-dir "$output_dir" \
            2>&1 | tee "${output_dir}/train.log"

        log "[OK] Completed ${cond} seed ${seed}"
    done
done

# Phase 2: Evaluation on frozen eval
log ""
log "PHASE 2: FROZEN EVAL"
log "=============================================="

python3 "${WORKSPACE}/experiments/eval_capsules.py" \
    --eval "$FROZEN_EVAL" \
    --results-root "$RESULTS_DIR" \
    --device cuda \
    --out "${RESULTS_DIR}/frozen_scoreboard.json"

log "[OK] Frozen eval complete"

# Phase 3: Evaluation on organic holdout
log ""
log "PHASE 3: ORGANIC HOLDOUT EVAL"
log "=============================================="

python3 "${WORKSPACE}/experiments/eval_capsules.py" \
    --eval "$ORGANIC_EVAL" \
    --results-root "$RESULTS_DIR" \
    --device cuda \
    --out "${RESULTS_DIR}/organic_scoreboard.json"

log "[OK] Organic holdout eval complete"

# Phase 4: Summary
log ""
log "=============================================="
log "RATIO SWEEP RESULTS"
log "=============================================="

python3 << 'PYEOF'
import json

frozen_path = '/root/QUANTUM_BWD/results/ratio_sweep/frozen_scoreboard.json'
organic_path = '/root/QUANTUM_BWD/results/ratio_sweep/organic_scoreboard.json'

try:
    with open(frozen_path) as f:
        frozen = json.load(f)
    with open(organic_path) as f:
        organic = json.load(f)
except Exception as e:
    print(f"Could not load scoreboards: {e}")
    exit(1)

print()
print("=" * 80)
print("PARETO FRONTIER ANALYSIS")
print("=" * 80)
print(f"{'Ratio':<15} {'Legacy':<8} {'Organic':<8} {'Exp':<8} {'Frozen T3':>12} {'Organic HO':>12}")
print("-" * 80)

ratios = ['R85_10_5', 'R80_15_5', 'R75_15_10', 'R70_20_10', 'R65_25_10']

results = []
for ratio in ratios:
    # Parse ratio
    parts = ratio[1:].split('_')
    legacy, org, exp = int(parts[0]), int(parts[1]), int(parts[2])

    # Get frozen tier3
    frozen_capsules = [c for c in frozen.get('capsules', [])
                       if c.get('condition') == ratio and 'error' not in c]
    if frozen_capsules:
        frozen_t3 = sum(c.get('by_tier', {}).get('tier3_adversarial', {}).get('pass_rate', 0)
                       for c in frozen_capsules) / len(frozen_capsules) * 100
    else:
        frozen_t3 = 0

    # Get organic holdout
    organic_capsules = [c for c in organic.get('capsules', [])
                        if c.get('condition') == ratio and 'error' not in c]
    if organic_capsules:
        organic_ho = sum(c.get('by_tier', {}).get('tier3_adversarial', {}).get('pass_rate', 0)
                        for c in organic_capsules) / len(organic_capsules) * 100
    else:
        organic_ho = 0

    results.append((ratio, legacy, org, exp, frozen_t3, organic_ho))
    print(f"{ratio:<15} {legacy:>6}% {org:>6}% {exp:>6}% {frozen_t3:>11.1f}% {organic_ho:>11.1f}%")

print("-" * 80)
print()

# Find Pareto optimal points
print("PARETO ANALYSIS:")
print("  Optimal = highest organic_holdout for each frozen_tier3 level")
print()

# Sort by frozen_t3 descending
results.sort(key=lambda x: x[4], reverse=True)
pareto = []
best_organic = -1
for r in results:
    if r[5] > best_organic:
        pareto.append(r)
        best_organic = r[5]

print("  Pareto-optimal configurations:")
for p in pareto:
    print(f"    {p[0]}: frozen_t3={p[4]:.1f}%, organic_ho={p[5]:.1f}%")

print()
print("RECOMMENDATION:")
print("  Choose based on your floor for frozen_tier3:")
print("    - Floor 30%: pick highest organic_ho above that floor")
print("    - Floor 28%: can push organic ratio higher")
print("    - No floor: maximize organic_ho (highest organic ratio)")
PYEOF

log ""
log "=============================================="
log "SWEEP COMPLETE"
log "=============================================="
log "Results: ${RESULTS_DIR}/"
log "  - frozen_scoreboard.json"
log "  - organic_scoreboard.json"
