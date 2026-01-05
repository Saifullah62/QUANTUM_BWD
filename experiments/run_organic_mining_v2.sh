#!/bin/bash
# Extended Organic Mining Marathon v2
# ====================================
#
# Goal: Scale organic tier3 inventory from 400 → 1,200+
# Strategy:
#   - Run with min_lineups=2 (slightly more permissive than original 3)
#   - This captures "pretty hard" cases that weren't quite hard enough for the strict gate
#   - Still requires multi-lineup validation (anti-pseudo-killer)
#
# Usage:
#   nohup ./experiments/run_organic_mining_v2.sh > results/organic_mining_v2.log 2>&1 &

set -e

WORKSPACE="/root/QUANTUM_BWD"
MINER="${WORKSPACE}/paradigm_factory/v2/mine_tier3_organic.py"

# Source data
EVENTS="${WORKSPACE}/paradigm_factory/v2/processed/canonicalized_v21.jsonl"
EXISTING="${WORKSPACE}/paradigm_factory/v2/bundles_v23/tier3_organic.jsonl"

# Output
OUT_V2="${WORKSPACE}/paradigm_factory/v2/bundles_v23/tier3_organic_v2.jsonl"
MERGED="${WORKSPACE}/paradigm_factory/v2/bundles_v23/tier3_organic_merged.jsonl"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=============================================="
log "EXTENDED ORGANIC MINING MARATHON v2"
log "=============================================="
log "Target: 800+ additional bundles (min_lineups=2)"

# Phase 1: Mine with relaxed settings
log ""
log "PHASE 1: MINING (min_lineups=2)"
log "=============================================="

python3 "$MINER" \
    --events "$EVENTS" \
    --existing-tier3 "$EXISTING" \
    --out "$OUT_V2" \
    --tier3-threshold 0.593 \
    --lineups-per-candidate 10 \
    --min-lineups-to-pass 2 \
    --candidates-per-sense 15 \
    --max-output 15000

log "[OK] Mining complete"

# Phase 2: Merge v1 + v2
log ""
log "PHASE 2: MERGING"
log "=============================================="

V1_COUNT=$(wc -l < "$EXISTING")
V2_COUNT=$(wc -l < "$OUT_V2")

log "v1 (strict): $V1_COUNT bundles"
log "v2 (relaxed): $V2_COUNT bundles"

# Merge (v1 first, then v2)
cat "$EXISTING" "$OUT_V2" > "$MERGED"

MERGED_COUNT=$(wc -l < "$MERGED")
log "Merged total: $MERGED_COUNT bundles"

# Phase 3: Stats
log ""
log "PHASE 3: PASS DISTRIBUTION"
log "=============================================="

python3 << 'PYEOF'
import json
from collections import Counter

merged_path = '/root/QUANTUM_BWD/paradigm_factory/v2/bundles_v23/tier3_organic_merged.jsonl'

passes = []
with open(merged_path) as f:
    for line in f:
        if line.strip():
            b = json.loads(line)
            md = b.get('metadata', {})
            passes.append(md.get('passes', 0))

dist = Counter(passes)
print(f"Total bundles: {len(passes)}")
print(f"Pass distribution:")
for p in sorted(dist.keys(), reverse=True):
    print(f"  {p}/10: {dist[p]} ({dist[p]/len(passes)*100:.1f}%)")

# Quality tiers
ultra = sum(1 for p in passes if p >= 8)
strong = sum(1 for p in passes if 5 <= p < 8)
moderate = sum(1 for p in passes if 3 <= p < 5)
relaxed = sum(1 for p in passes if p == 2)

print(f"\nQuality tiers:")
print(f"  Ultra (8-10 passes): {ultra}")
print(f"  Strong (5-7 passes): {strong}")
print(f"  Moderate (3-4 passes): {moderate}")
print(f"  Relaxed (2 passes): {relaxed}")
PYEOF

log ""
log "=============================================="
log "MINING COMPLETE"
log "=============================================="
log "Output: $MERGED"
log ""
log "Next steps:"
log "  1. Split merged into train/holdout"
log "  2. Create v25 bundles with expanded organic"
log "  3. Run ratio sweep with larger organic pool"
