#!/bin/bash
# QLLM Scaling Experiment Runner (v2)
# Target: RAMP GPU Server (159.89.127.151)
#
# Usage:
#   ./run_scaling_experiment.sh smoke        # Quick integrity check (A, seed 42, short)
#   ./run_scaling_experiment.sh A            # All seeds for condition A
#   ./run_scaling_experiment.sh B            # All seeds for condition B
#   ./run_scaling_experiment.sh C            # All seeds for condition C
#   ./run_scaling_experiment.sh all          # Full A→B→C sequence
#   ./run_scaling_experiment.sh enrichment   # Start enrichment in background

set -e

# Configuration
WORKSPACE="/workspace/QUANTUM_BWD"
RESULTS_DIR="${WORKSPACE}/results/scaling"
BUNDLES_V22="${WORKSPACE}/paradigm_factory/v2/bundles_v22"
BUNDLES_V23="${WORKSPACE}/paradigm_factory/v2/bundles_v23"

SEEDS="42 123 456"
SHORT_EPOCHS=2      # For smoke test
FULL_EPOCHS=10      # For real runs

# Ship blocker thresholds
MIN_TOP1=0.30
MIN_TOP3=0.50
MIN_COHERENCE=0.80

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Get git commit hash
get_git_hash() {
    git -C "$WORKSPACE" rev-parse --short HEAD 2>/dev/null || echo "unknown"
}

# Check GPU
check_gpu() {
    log "Checking GPU..."
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
}

# Create result capsule
create_capsule() {
    local cond=$1
    local seed=$2
    local capsule_dir="${RESULTS_DIR}/${cond}/seed_${seed}"

    mkdir -p "$capsule_dir"

    # Write config snapshot
    cat > "${capsule_dir}/config.json" << EOF
{
    "condition": "$cond",
    "seed": $seed,
    "timestamp": "$(date -Iseconds)",
    "git_hash": "$(get_git_hash)",
    "workspace": "$WORKSPACE"
}
EOF

    echo "$capsule_dir"
}

# Write capsule fingerprint (sha256 of manifest)
finalize_capsule() {
    local capsule_dir=$1

    # Create manifest of all files
    find "$capsule_dir" -type f -name "*.json" -o -name "*.jsonl" | sort | while read f; do
        sha256sum "$f"
    done > "${capsule_dir}/manifest.txt"

    # Compute capsule fingerprint
    local fingerprint=$(sha256sum "${capsule_dir}/manifest.txt" | cut -d' ' -f1 | head -c 16)
    echo "$fingerprint" > "${capsule_dir}/capsule_fingerprint.txt"

    log "CAPSULE FINGERPRINT: $fingerprint"
    log "Capsule written to: $capsule_dir"
}

# Run single experiment
run_condition() {
    local cond=$1
    local seed=$2
    local epochs=${3:-$FULL_EPOCHS}

    log "=========================================="
    log "Running condition $cond with seed $seed (epochs=$epochs)"
    log "=========================================="

    # Create result capsule
    local capsule_dir=$(create_capsule "$cond" "$seed")

    # Set bundle path based on condition
    local bundle_path=""
    local curriculum_config="uniform"

    case $cond in
        A)
            bundle_path="${BUNDLES_V22}/contrastive_bundles.jsonl"
            ;;
        B)
            bundle_path="${BUNDLES_V23}/contrastive_bundles.jsonl"
            curriculum_config="tier_weighted"
            ;;
        C)
            # Combine v2.3 contrastive + adversarial + enriched (if available)
            local combined="${capsule_dir}/combined_bundles.jsonl"
            cat "${BUNDLES_V23}/contrastive_bundles.jsonl" > "$combined"

            if [ -f "${BUNDLES_V23}/adversarial/adversarial_bundles.jsonl" ]; then
                cat "${BUNDLES_V23}/adversarial/adversarial_bundles.jsonl" >> "$combined"
                log "Added adversarial bundles"
            fi

            if [ -f "${BUNDLES_V23}/enrichment/synthetic_enriched_events.jsonl" ]; then
                # Note: enriched events need conversion to bundles
                log "Enriched events available ($(wc -l < ${BUNDLES_V23}/enrichment/synthetic_enriched_events.jsonl) items)"
            fi

            bundle_path="$combined"
            curriculum_config="tier_weighted"
            ;;
        *)
            log "ERROR: Unknown condition $cond"
            exit 1
            ;;
    esac

    # Copy bundle fingerprint
    local bundle_fingerprint=$(head -1 "$bundle_path" | python3 -c "import sys,json,hashlib; d=json.load(sys.stdin); print(hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()[:16])" 2>/dev/null || echo "unknown")
    echo "$bundle_fingerprint" > "${capsule_dir}/bundle_fingerprint.txt"
    log "Bundle fingerprint: $bundle_fingerprint"

    # Run training
    log "Starting training..."

    # Calculate steps: ~2 steps per bundle for smoke, ~10 for full
    local bundle_count=$(wc -l < "$bundle_path")
    local steps=$((bundle_count * epochs / 32))  # batch_size=32
    [ $steps -lt 100 ] && steps=100

    python "${WORKSPACE}/scripts/train_v23_bundles.py" \
        --bundles "$bundle_path" \
        --seed "$seed" \
        --steps "$steps" \
        --device cuda \
        --output-dir "$capsule_dir" \
        2>&1 | tee "${capsule_dir}/train.log"

    # Check training metrics and write gate output
    if [ -f "${capsule_dir}/metrics.json" ]; then
        log "Checking training metrics..."
        python3 << EOF
import json
import sys

with open('${capsule_dir}/metrics.json') as f:
    m = json.load(f)

# Extract key metrics
final_loss = m.get('final_loss', 999)
top1 = m.get('top1_accuracy', 0)
top3 = m.get('top3_accuracy', 0)

# Simple pass criterion: loss below threshold
passed = final_loss < 2.0 and top1 >= $MIN_TOP1

# Write gate output
gate = {
    'final_loss': final_loss,
    'top1': top1,
    'top3': top3,
    'passed': passed,
    'thresholds': {'top1': $MIN_TOP1, 'max_loss': 2.0}
}
with open('${capsule_dir}/gate_output.json', 'w') as f:
    json.dump(gate, f, indent=2)

print(f'Final Loss: {final_loss:.4f}')
print(f'Top-1: {top1:.1%} (min: $MIN_TOP1)')
print(f'Top-3: {top3:.1%}')
print(f'PASSED: {passed}')

sys.exit(0 if passed else 1)
EOF
        local gate_status=$?
        if [ $gate_status -eq 0 ]; then
            log "PASS: Condition $cond seed $seed passed gates"
            echo "PASS" > "${capsule_dir}/status.txt"
        else
            log "FAIL: Condition $cond seed $seed failed gates"
            echo "FAIL" > "${capsule_dir}/status.txt"
        fi
    else
        log "WARNING: No metrics found"
        echo "NO_METRICS" > "${capsule_dir}/status.txt"
    fi

    # Finalize capsule
    finalize_capsule "$capsule_dir"

    log "Completed condition $cond seed $seed"
}

# Smoke test (quick integrity check)
run_smoke() {
    log "=========================================="
    log "SMOKE TEST: Verifying RAMP environment"
    log "=========================================="

    check_gpu

    # Run short A condition with seed 42
    run_condition "A" "42" "$SHORT_EPOCHS"

    log "=========================================="
    log "SMOKE TEST COMPLETE"
    log "=========================================="
    log "If fingerprints and gates match local, proceed with full experiment"
}

# Run all seeds for a condition
run_all_seeds() {
    local cond=$1
    log "Running all seeds for condition $cond"

    for seed in $SEEDS; do
        run_condition "$cond" "$seed"
    done

    log "Completed all seeds for condition $cond"
}

# Run enrichment in background
run_enrichment() {
    log "Starting enrichment in background..."

    nohup python "${WORKSPACE}/paradigm_factory/v2/enrich_singleton_senses.py" \
        --singletons "${BUNDLES_V23}/singleton_senses_todo.jsonl" \
        --events "${WORKSPACE}/paradigm_factory/v2/processed/canonicalized_v21.jsonl" \
        --output "${BUNDLES_V23}/enrichment" \
        > "${WORKSPACE}/logs/enrichment.log" 2>&1 &

    local pid=$!
    echo $pid > "${WORKSPACE}/logs/enrichment.pid"

    log "Enrichment started with PID $pid"
    log "Monitor with: tail -f ${WORKSPACE}/logs/enrichment.log"
}

# Full experiment sequence
run_full() {
    log "=========================================="
    log "FULL SCALING EXPERIMENT"
    log "=========================================="

    check_gpu

    # Run in order: A → B → C
    log "Phase 1: Baseline (Condition A)"
    run_all_seeds "A"

    log "Phase 2: v2.3 Bundles (Condition B)"
    run_all_seeds "B"

    log "Phase 3: v2.3 + Enriched (Condition C)"
    run_all_seeds "C"

    # Aggregate results
    log "Aggregating results..."
    python "${WORKSPACE}/experiments/aggregate_scaling_results.py" "$RESULTS_DIR"

    log "=========================================="
    log "EXPERIMENT COMPLETE"
    log "=========================================="
}

# Main
case "${1:-help}" in
    smoke)
        run_smoke
        ;;
    A|B|C)
        run_all_seeds "$1"
        ;;
    all|full)
        run_full
        ;;
    enrichment)
        run_enrichment
        ;;
    single)
        # Single run: ./run_scaling_experiment.sh single A 42
        run_condition "${2:-A}" "${3:-42}"
        ;;
    help|*)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  smoke       - Quick integrity check (A, seed 42, short epochs)"
        echo "  A|B|C       - Run all seeds for a condition"
        echo "  all         - Full A→B→C sequence"
        echo "  enrichment  - Start enrichment in background"
        echo "  single X Y  - Single run (condition X, seed Y)"
        echo ""
        echo "Recommended sequence:"
        echo "  1. ./run_scaling_experiment.sh smoke       # Verify environment"
        echo "  2. ./run_scaling_experiment.sh enrichment  # Start in background"
        echo "  3. ./run_scaling_experiment.sh all         # Full experiment"
        ;;
esac
