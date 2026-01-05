#!/bin/bash
# Curriculum Experiment: C1 vs C2 vs C2b
# =======================================
#
# C1 (baseline): uniform curriculum (31% tier3 throughout)
# C2 (phased): phased curriculum + tier3 cap 20% (starved tier3)
# C2b (phased with teeth): 22/8/2 → 18/9/5 → 14/10/8, cap 25%, tier3 replay
#
# Same seeds, same steps, same eval set - isolates curriculum effects
#
# Usage:
#   ./run_curriculum_experiment.sh smoke    # Quick test
#   ./run_curriculum_experiment.sh C1       # Baseline curriculum
#   ./run_curriculum_experiment.sh C2       # Phased curriculum (too gentle)
#   ./run_curriculum_experiment.sh C2b      # Phased with teeth (recommended)
#   ./run_curriculum_experiment.sh all      # Full C1 vs C2b comparison
#   ./run_curriculum_experiment.sh eval     # Evaluate all capsules

set -e

WORKSPACE="/workspace/QUANTUM_BWD"
RESULTS_DIR="${WORKSPACE}/results/curriculum_exp"
BUNDLES="${WORKSPACE}/paradigm_factory/v2/bundles_v23/contrastive_bundles.jsonl"
EVAL_SET="${WORKSPACE}/evals/eval_v23_contrastive_5k.jsonl"

SEEDS="42 123 456"
STEPS=5000

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_condition() {
    local cond=$1
    local seed=$2
    local curriculum=$3

    local output_dir="${RESULTS_DIR}/${cond}/seed_${seed}"
    mkdir -p "$output_dir"

    log "Running ${cond} seed ${seed} (curriculum=${curriculum})"

    python3 "${WORKSPACE}/scripts/train_curriculum_v2.py" \
        --bundles "$BUNDLES" \
        --steps "$STEPS" \
        --seed "$seed" \
        --curriculum "$curriculum" \
        --device cuda \
        --output-dir "$output_dir" \
        2>&1 | tee "${output_dir}/train.log"

    log "Completed ${cond} seed ${seed}"
}

run_all_seeds() {
    local cond=$1
    local curriculum=$2

    log "Running all seeds for ${cond} (curriculum=${curriculum})"
    for seed in $SEEDS; do
        run_condition "$cond" "$seed" "$curriculum"
    done
}

run_smoke() {
    log "=========================================="
    log "SMOKE TEST: Verifying curriculum training"
    log "=========================================="

    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

    # Quick test with 500 steps
    python3 "${WORKSPACE}/scripts/train_curriculum_v2.py" \
        --bundles "$BUNDLES" \
        --steps 500 \
        --seed 42 \
        --curriculum phased \
        --device cuda \
        --output-dir "${RESULTS_DIR}/smoke_test"

    log "Smoke test complete"
}

run_eval() {
    log "=========================================="
    log "EVALUATING ALL CAPSULES"
    log "=========================================="

    python3 "${WORKSPACE}/experiments/eval_capsules.py" \
        --eval "$EVAL_SET" \
        --results-root "$RESULTS_DIR" \
        --device cuda \
        --out "${RESULTS_DIR}/curriculum_scoreboard.json"

    log "Evaluation complete"
    cat "${RESULTS_DIR}/curriculum_scoreboard.json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('\n' + '='*60)
print('CURRICULUM EXPERIMENT SUMMARY')
print('='*60)
for cond, stats in sorted(data.get('summary', {}).items()):
    print(f\"{cond}: acc={stats['mean_accuracy']:.1%} margin={stats['mean_margin']:.4f}\")
"
}

run_full() {
    log "=========================================="
    log "FULL CURRICULUM EXPERIMENT: C1 vs C2b"
    log "=========================================="

    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

    log "Phase 1: C1 (uniform curriculum - baseline)"
    run_all_seeds "C1" "uniform"

    log "Phase 2: C2b (phased with teeth - proposed)"
    run_all_seeds "C2b" "phased_v2"

    log "Phase 3: Evaluation"
    run_eval

    log "=========================================="
    log "EXPERIMENT COMPLETE"
    log "=========================================="
}

case "${1:-help}" in
    smoke)
        run_smoke
        ;;
    C1)
        run_all_seeds "C1" "uniform"
        ;;
    C2)
        run_all_seeds "C2" "phased"
        ;;
    C2b)
        run_all_seeds "C2b" "phased_v2"
        ;;
    all|full)
        run_full
        ;;
    eval)
        run_eval
        ;;
    single)
        # Single run: ./run_curriculum_experiment.sh single C2b 42 phased_v2
        run_condition "${2:-C2b}" "${3:-42}" "${4:-phased_v2}"
        ;;
    help|*)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  smoke       - Quick verification (500 steps)"
        echo "  C1          - Run baseline (uniform curriculum)"
        echo "  C2          - Run original phased (too gentle)"
        echo "  C2b         - Run phased with teeth (recommended)"
        echo "  all         - Full C1 vs C2b comparison"
        echo "  eval        - Evaluate all capsules"
        echo "  single X Y Z - Single run (condition, seed, curriculum)"
        echo ""
        echo "Recommended sequence:"
        echo "  1. ./run_curriculum_experiment.sh smoke"
        echo "  2. ./run_curriculum_experiment.sh C2b"
        echo "  3. ./run_curriculum_experiment.sh eval"
        ;;
esac
