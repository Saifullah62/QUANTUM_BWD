#!/bin/bash
# Tier3x A/B Experiment: D1 (v23) vs D2 (v23_tier3x)
# ==================================================
#
# Same curriculum policy (uniform 12/10/10), only data changes.
# Answers: does tier3x improve tier3 pass_rate without dragging tier2?
#
# Usage:
#   ./run_tier3x_ab.sh smoke     # Quick 500-step test
#   ./run_tier3x_ab.sh D1        # Baseline (v23, tier3=454)
#   ./run_tier3x_ab.sh D2        # Tier3x (v23_tier3x, tier3=2704)
#   ./run_tier3x_ab.sh all       # Full A/B comparison
#   ./run_tier3x_ab.sh eval      # Evaluate all capsules

set -e

WORKSPACE="/workspace/QUANTUM_BWD"
RESULTS_DIR="${WORKSPACE}/results/tier3x_ab"
EVAL_SET="${WORKSPACE}/evals/eval_v23_contrastive_5k.jsonl"

# Datasets
D1_BUNDLES="${WORKSPACE}/paradigm_factory/v2/bundles_v23/contrastive_bundles.jsonl"
D2_BUNDLES="${WORKSPACE}/paradigm_factory/v2/bundles_v23/contrastive_bundles_v23_tier3x.jsonl"

# Training params
SEEDS="42 123 456"
STEPS=5000
CURRICULUM="uniform"  # 12/10/10 quotas

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_condition() {
    local cond=$1
    local seed=$2
    local bundles=$3

    local output_dir="${RESULTS_DIR}/${cond}/seed_${seed}"
    mkdir -p "$output_dir"

    log "Running ${cond} seed ${seed}"
    log "  Bundles: $(basename $bundles)"

    python3 "${WORKSPACE}/scripts/train_curriculum_v2.py" \
        --bundles "$bundles" \
        --steps "$STEPS" \
        --seed "$seed" \
        --curriculum "$CURRICULUM" \
        --device cuda \
        --output-dir "$output_dir" \
        2>&1 | tee "${output_dir}/train.log"

    log "Completed ${cond} seed ${seed}"
}

run_all_seeds() {
    local cond=$1
    local bundles=$2

    log "Running all seeds for ${cond}"
    for seed in $SEEDS; do
        run_condition "$cond" "$seed" "$bundles"
    done
}

run_smoke() {
    log "==========================================="
    log "SMOKE TEST: Tier3x A/B Verification"
    log "==========================================="

    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

    # Quick test with D2 (tier3x)
    python3 "${WORKSPACE}/scripts/train_curriculum_v2.py" \
        --bundles "$D2_BUNDLES" \
        --steps 500 \
        --seed 42 \
        --curriculum "$CURRICULUM" \
        --device cuda \
        --output-dir "${RESULTS_DIR}/smoke_test"

    log "Smoke test complete"
}

run_eval() {
    log "==========================================="
    log "EVALUATING ALL CAPSULES"
    log "==========================================="

    python3 "${WORKSPACE}/experiments/eval_capsules.py" \
        --eval "$EVAL_SET" \
        --results-root "$RESULTS_DIR" \
        --device cuda \
        --out "${RESULTS_DIR}/tier3x_ab_scoreboard.json"

    log "Evaluation complete"

    # Print summary
    python3 << 'PYEOF'
import json
with open('/workspace/QUANTUM_BWD/results/tier3x_ab/tier3x_ab_scoreboard.json') as f:
    data = json.load(f)

print('\n' + '='*70)
print('TIER3X A/B EXPERIMENT RESULTS')
print('='*70)
print(f"{'Condition':<12} {'Accuracy':>10} {'Margin':>10} {'T3 Pass':>10} {'T3 Margin':>12}")
print('-'*70)

for cond in ['D1', 'D2']:
    stats = data.get('summary', {}).get(cond, {})
    if stats:
        acc = stats.get('mean_accuracy', 0) * 100
        margin = stats.get('mean_margin', 0)
        t3_pass = stats.get('tier3_pass_rate', 0) * 100
        t3_margin = stats.get('tier3_median_margin', 0)
        print(f"{cond:<12} {acc:>9.1f}% {margin:>10.4f} {t3_pass:>9.1f}% {t3_margin:>12.4f}")

print('-'*70)
print('\nDecision rule:')
print('  - If T3 pass +2-4pp and T2 drop <1pp → ship tier3x')
print('  - If T3 up but T2 collapses → adjust quotas (13/11/8)')
print('  - If no improvement → model saturated, try better positives')
PYEOF
}

run_full() {
    log "==========================================="
    log "FULL TIER3X A/B EXPERIMENT"
    log "==========================================="

    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

    log "Condition D1: v23 baseline (tier3=454)"
    run_all_seeds "D1" "$D1_BUNDLES"

    log "Condition D2: v23_tier3x (tier3=2704)"
    run_all_seeds "D2" "$D2_BUNDLES"

    log "Running evaluation..."
    run_eval

    log "==========================================="
    log "EXPERIMENT COMPLETE"
    log "==========================================="
}

case "${1:-help}" in
    smoke)
        run_smoke
        ;;
    D1)
        run_all_seeds "D1" "$D1_BUNDLES"
        ;;
    D2)
        run_all_seeds "D2" "$D2_BUNDLES"
        ;;
    all|full)
        run_full
        ;;
    eval)
        run_eval
        ;;
    help|*)
        echo "Tier3x A/B Experiment"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  smoke  - Quick verification (500 steps)"
        echo "  D1     - Run baseline (v23, tier3=454)"
        echo "  D2     - Run tier3x (v23_tier3x, tier3=2704)"
        echo "  all    - Full A/B comparison (D1 vs D2)"
        echo "  eval   - Evaluate all capsules"
        echo ""
        echo "Dataset comparison:"
        echo "  D1: 15,113 bundles (tier3=454)"
        echo "  D2: 17,363 bundles (tier3=2,704)"
        echo ""
        echo "Both use uniform curriculum with 11/11/10 quotas (tier3_max=35%)."
        ;;
esac
