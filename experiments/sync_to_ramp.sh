#!/bin/bash
# Sync QUANTUM_BWD to RAMP GPU Server
#
# Usage:
#   ./sync_to_ramp.sh          # Sync code + data
#   ./sync_to_ramp.sh code     # Sync code only
#   ./sync_to_ramp.sh results  # Pull results back

set -e

RAMP_HOST="root@159.89.127.151"
RAMP_DIR="/workspace/QUANTUM_BWD"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

sync_code() {
    log "Syncing code to RAMP..."
    rsync -avz --progress \
        --exclude='*.pt' \
        --exclude='*.pth' \
        --exclude='*.ckpt' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='wandb' \
        --exclude='experiments/scaling/*_seed*' \
        "$LOCAL_DIR/" \
        "$RAMP_HOST:$RAMP_DIR/"
    log "Code sync complete"
}

sync_data() {
    log "Syncing data bundles to RAMP..."
    rsync -avz --progress \
        "$LOCAL_DIR/paradigm_factory/v2/bundles_v22/" \
        "$RAMP_HOST:$RAMP_DIR/paradigm_factory/v2/bundles_v22/"

    rsync -avz --progress \
        "$LOCAL_DIR/paradigm_factory/v2/bundles_v23/" \
        "$RAMP_HOST:$RAMP_DIR/paradigm_factory/v2/bundles_v23/"

    log "Data sync complete"
}

pull_results() {
    log "Pulling results from RAMP..."
    rsync -avz --progress \
        "$RAMP_HOST:$RAMP_DIR/experiments/scaling/" \
        "$LOCAL_DIR/experiments/scaling/"
    log "Results pulled"
}

verify_ramp() {
    log "Verifying RAMP connection..."
    ssh "$RAMP_HOST" "nvidia-smi && echo 'RAMP OK'"
}

setup_ramp() {
    log "Setting up RAMP environment..."
    ssh "$RAMP_HOST" "cd $RAMP_DIR && pip install -r requirements.txt"
}

case "${1:-full}" in
    code)
        sync_code
        ;;
    data)
        sync_data
        ;;
    full)
        verify_ramp
        sync_code
        sync_data
        ;;
    results)
        pull_results
        ;;
    setup)
        setup_ramp
        ;;
    verify)
        verify_ramp
        ;;
    *)
        echo "Usage: $0 [code|data|full|results|setup|verify]"
        exit 1
        ;;
esac
