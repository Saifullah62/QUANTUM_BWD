#!/bin/bash
set -e

BUNDLES="/root/QUANTUM_BWD/paradigm_factory/v2/bundles_v23/contrastive_bundles_v25_organic.jsonl"
TRAINER="/root/QUANTUM_BWD/scripts/train_curriculum_v3.py"
RESULTS="/root/QUANTUM_BWD/results/ratio_sweep_v25"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

log "Starting v25 ratio sweep"

# R75_15_10
for seed in 42 123 456; do
    if [ -f "$RESULTS/R75_15_10/seed_$seed/checkpoint_final.pt" ]; then
        log "SKIP R75_15_10 seed $seed"
    else
        log "TRAIN R75_15_10 seed $seed"
        python3 $TRAINER --bundles $BUNDLES --steps 5000 --seed $seed \
            --curriculum uniform --device cuda \
            --tier3-legacy 0.75 --tier3-organic 0.15 --tier3-expanded 0.10 \
            --output-dir $RESULTS/R75_15_10/seed_$seed
    fi
done

# R70_20_10
for seed in 42 123 456; do
    if [ -f "$RESULTS/R70_20_10/seed_$seed/checkpoint_final.pt" ]; then
        log "SKIP R70_20_10 seed $seed"
    else
        log "TRAIN R70_20_10 seed $seed"
        python3 $TRAINER --bundles $BUNDLES --steps 5000 --seed $seed \
            --curriculum uniform --device cuda \
            --tier3-legacy 0.70 --tier3-organic 0.20 --tier3-expanded 0.10 \
            --output-dir $RESULTS/R70_20_10/seed_$seed
    fi
done

# R65_25_10
for seed in 42 123 456; do
    if [ -f "$RESULTS/R65_25_10/seed_$seed/checkpoint_final.pt" ]; then
        log "SKIP R65_25_10 seed $seed"
    else
        log "TRAIN R65_25_10 seed $seed"
        python3 $TRAINER --bundles $BUNDLES --steps 5000 --seed $seed \
            --curriculum uniform --device cuda \
            --tier3-legacy 0.65 --tier3-organic 0.25 --tier3-expanded 0.10 \
            --output-dir $RESULTS/R65_25_10/seed_$seed
    fi
done

log "Training complete, running evals..."

# Frozen eval
python3 /root/QUANTUM_BWD/experiments/eval_capsules.py \
    --eval /root/QUANTUM_BWD/evals/eval_v23_contrastive_5k.jsonl \
    --results-root $RESULTS --device cuda \
    --out $RESULTS/frozen_scoreboard.json

# Organic holdout v2 eval
python3 /root/QUANTUM_BWD/experiments/eval_capsules.py \
    --eval /root/QUANTUM_BWD/evals/organic_holdout_v2_eval.jsonl \
    --results-root $RESULTS --device cuda \
    --out $RESULTS/organic_v2_scoreboard.json

log "V25 SWEEP COMPLETE"
