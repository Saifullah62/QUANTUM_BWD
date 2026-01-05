# QLLM Scaling Experiment Design

## Infrastructure: RAMP GPU Server
- **Host**: 159.89.127.151
- **GPU**: 1x (48 GB VRAM)
- **CPU**: 8 vCPU
- **RAM**: 64 GB
- **Storage**: 500 GB NVMe (boot), ensure full sync to local repo

## Experiment: Controlled A/B/C Scaling

### Hypothesis
With certified v3.1 baseline and v2.3 bundles (proper tier distribution), scaling
training data and/or model capacity will improve retrieval metrics while maintaining
certification thresholds.

### Conditions

#### Condition A: Baseline v3.1 (Control)
- **Data**: v2.2 bundles (~15,966)
- **Model**: Current SenseHead architecture
- **Training**: Certified hyperparameters
- **Seeds**: 42, 123, 456

#### Condition B: v2.3 Bundles (Data Scaling)
- **Data**: v2.3 bundles (~15,113 + 598 adversarial)
  - tier1_easy: 12,846 (85%)
  - tier2_robust: 1,813 (12%)
  - tier3_adversarial: 454 (3%)
- **Model**: Current SenseHead architecture
- **Training**: Certified hyperparameters
- **Curriculum**: Oversample tier2 (2x), tier3 (4x)
- **Seeds**: 42, 123, 456

#### Condition C: v2.3 + Enriched (Full Data)
- **Data**: v2.3 + singleton enrichment (~40k+ bundles estimated)
- **Model**: Current SenseHead architecture
- **Training**: Certified hyperparameters
- **Curriculum**: Same as B
- **Seeds**: 42, 123, 456

### Metrics (Ship Blockers)
All conditions must pass:
- retrieval_top1 >= 30%
- retrieval_top3 >= 50%
- coherence_accuracy >= 80%

### Primary Metrics
- **Retrieval Top-1 accuracy** (main)
- **Retrieval Top-3 accuracy**
- **Margin distribution** (mean, p5, p95)
- **Tier-stratified accuracy** (tier1 vs tier2 vs tier3)
- **Killer set performance** (tier4 subset)

### Secondary Metrics
- Training stability (loss variance across seeds)
- Convergence speed (epochs to threshold)
- Attention entropy evolution

## Protocol

### Pre-flight Checks
```bash
# Sync repo to RAMP
rsync -avz --exclude='*.pt' --exclude='*.pth' --exclude='__pycache__' \
  /path/to/QUANTUM_BWD/ root@159.89.127.151:/workspace/QUANTUM_BWD/

# Verify GPU
ssh root@159.89.127.151 "nvidia-smi"

# Install dependencies
ssh root@159.89.127.151 "cd /workspace/QUANTUM_BWD && pip install -r requirements.txt"
```

### Run Script
```bash
#!/bin/bash
# run_scaling_experiment.sh

SEEDS="42 123 456"
CONDITIONS="A B C"

for cond in $CONDITIONS; do
    for seed in $SEEDS; do
        echo "Running condition $cond with seed $seed"

        python scripts/run_full_pipeline.py \
            --condition $cond \
            --seed $seed \
            --strict \
            --output experiments/scaling/${cond}_seed${seed}

        # Upload results to local
        scp -r experiments/scaling/${cond}_seed${seed} \
            user@local:/path/to/results/
    done
done
```

### Analysis
1. Aggregate metrics across seeds per condition
2. Statistical significance tests (paired t-test, A vs B, B vs C)
3. Failure mode analysis (which tier fails most?)
4. Attention pattern comparison

## Expected Outcomes

### Condition A (Baseline)
- Known performance from v3.1-certified
- Serves as control for comparison

### Condition B (v2.3 Data)
- Expected: +2-5% retrieval_top1 from adversarial examples
- Risk: May see slight tier1 regression if tier3 is too hard
- Mitigation: Curriculum balancing

### Condition C (Enriched)
- Expected: +5-10% retrieval_top1 from 2.5x more data
- Risk: Synthetic quality may introduce noise
- Mitigation: Verification threshold (0.5 similarity, 0.05 separation gap)

## Timeline
- Sync + setup: 30 min
- Per-condition training: ~2h (3 seeds × 40min each)
- Total runtime: ~7h
- Analysis: 1h

## Contingency
If any condition fails ship blockers:
1. Reduce curriculum oversampling
2. Increase verification thresholds for synthetics
3. Fall back to condition A (certified baseline)
