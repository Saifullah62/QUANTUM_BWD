# Crucible Architecture

## System Overview

Crucible consists of six primary modules forming a directed dataflow pipeline, plus orchestration and reporting layers.

```
Canonical Events → Bundle Generator → Tier3 Pool Manager → Curriculum Sampler
                                                                    ↓
                                                              Training Loop
                                                                    ↓
                                            Evaluation Engine (Frozen + Organic)
                                                                    ↓
                                                 Scoreboard → Report Generator
```

## Module Definitions

### Bundle Generator

**Location**: `paradigm_factory/v2/` (existing repo generator modules)

**Purpose**: Transform canonical events into tiered contrastive bundles.

**Input**:
- Canonical events corpus (JSONL, one event per line)
- Each event has: `event_id`, `sense`, `text`, optional `embedding`

**Output**:
- Tiered contrastive bundles (`contrastive_bundles.jsonl`)
- Each bundle has: anchor, positive, negatives, metadata with tier assignment

**Process**:
1. For each anchor event, sample a positive (same-sense instance)
2. Construct negatives from three sources:
   - Within-lemma confusers (same lemma, different sense)
   - Sibling senses (related senses in hierarchy)
   - Cross-lemma distractors (different lemma, potentially confusing)
3. Compute danger score (see [METRICS.md](METRICS.md#danger-score) for formula)
4. Assign tier based on danger score percentiles:
   - tier1_easy: bottom 70%
   - tier2_robust: 70th-90th percentile
   - tier3_adversarial: top 10%

**Interface**:
```python
def generate_bundles(
    events_path: str,
    output_path: str,
    tier_thresholds: Tuple[float, float] = (0.70, 0.90),
    negatives_config: NegativesConfig = default_config
) -> BundleStats
```

### Tier3 Pool Manager

**Location**: `paradigm_factory/v2/` (conceptual module; implement as needed)

**Purpose**: Maintain separation between legacy, expanded, and organic tier3 pools.

**Input**:
- Multiple tier3 source files with provenance tags
- Legacy core killers (original organically hard examples)
- Expanded killers (synthetically generated, quality-gated)
- Organic killers (mined via multi-lineup validation)

**Output**:
- Merged pool with stratification tags
- Split train/holdout partitions
- Exposure probability calculations

**Responsibilities**:
- Track provenance (source file, mining run, passes count)
- Enforce deduplication by anchor sense
- Compute expected per-item exposures given proposed ratios
- Emit dilution warnings when legacy exposure falls below threshold

**Interface**:
```python
def merge_pools(
    legacy_path: str,
    expanded_path: str,
    organic_path: str,
    output_path: str
) -> PoolManifest

def split_holdout(
    pool_path: str,
    holdout_fraction: float,
    seed: int
) -> Tuple[str, str]  # train_path, holdout_path

def compute_exposure(
    pool_manifest: PoolManifest,
    tier3_ratios: Tier3Ratios,
    steps: int,
    batch_size: int,
    tier3_probability: float
) -> ExposureStats
```

### Curriculum Sampler

**Location**: `scripts/train_curriculum_v3.py` (three-pool ratios) or `scripts/train_curriculum_v2.py` (single tier3-mix)

**Classes**: `TieredBundleDataset`, `Tier3MixRatios`, `CurriculumSampler`

**Purpose**: Implement training-time data loading with explicit ratio controls and tier caps.

**Key Mechanism**: Two-level sampling
1. Sample tier (tier1/tier2/tier3) according to `tier_probabilities`
2. If tier3, sample pool (legacy/organic/expanded) according to `Tier3MixRatios`
3. Enforce per-tier caps per batch to prevent tier domination

**Interface**:
```python
@dataclass
class Tier3MixRatios:
    legacy: float = 0.75
    organic: float = 0.15
    expanded: float = 0.10

    def sample_pool(self) -> str:
        r = random.random()
        if r < self.legacy:
            return "legacy"
        elif r < self.legacy + self.organic:
            return "organic"
        else:
            return "expanded"

class TieredBundleDataset:
    def __init__(
        self,
        bundles_path: str,
        tier_probabilities: List[float],
        tier3_ratios: Tier3MixRatios,
        tier_caps: List[int]
    ):
        # Classify bundles into pools
        # tier3 further split by metadata.source

    def sample_batch(self, batch_size: int) -> List[Bundle]:
        # Two-level sampling with caps
```

**Bundle Classification Logic**:
```python
def classify_tier3(bundle: Bundle) -> str:
    metadata = bundle.get('metadata', {})
    source = metadata.get('source', '')
    expansion = metadata.get('expansion', '')

    if source == 'tier3_organic_miner':
        return 'organic'
    elif expansion == 'tier3x' or source == 'tier3_expansion':
        return 'expanded'
    else:
        return 'legacy'
```

### Training Loop

**Location**: `scripts/train_curriculum_v3.py` (recommended) or `scripts/train_curriculum_v2.py`

**Input**:
- Curriculum sampler (configured with ratios and caps)
- Model architecture
- Optimizer configuration
- Training parameters (steps, LR, weight decay)

**Output**:
- Checkpoint files (`checkpoint_final.pt`, optional intermediate checkpoints)
- Training logs with loss, accuracy, tier quota usage

**Key Parameters** (v3):
- `--steps`: Training duration
- `--lr`: Learning rate
- `--tier3-legacy`, `--tier3-organic`, `--tier3-expanded`: Pool ratios (v3 only)
- `--curriculum`: Sampling mode (uniform, phased, phased_v2)

**Note**: v2 uses `--tier3-mix` for a single ratio; v3 adds explicit three-pool control.

**Training Metrics Logged**:
- Loss per step
- Accuracy per step
- Tier quota usage (e.g., "quota=11/11/10" meaning 11 tier1, 11 tier2, 10 tier3)
- Learning rate (for schedules)

### Evaluation Engine

**Location**: `experiments/eval_capsules.py`

**Purpose**: Run inference over eval sets and produce scoreboards with margin distributions.

**Input**:
- Checkpoint path
- Eval set (JSONL)
- Device configuration

**Output**:
- Scoreboard (JSON) with per-tier and overall metrics
- Eval fingerprint for reproducibility

**Key Mechanism**:
1. Load checkpoint and eval set
2. For each item, compute similarities and margin
3. Aggregate into statistics: pass_rate, mean/median margin, Q10/Q90, mean_rank
4. Stratify by tier
5. Generate fingerprint from source hash + seed + counts

**Interface**:
```python
def evaluate(
    checkpoint_path: str,
    eval_path: str,
    output_path: str,
    device: str = 'cuda'
) -> Scoreboard

@dataclass
class Scoreboard:
    eval_header: EvalHeader
    overall: TierStats
    by_tier: Dict[str, TierStats]

@dataclass
class TierStats:
    n: int
    correct: int
    accuracy: float
    pass_rate: float
    mean_margin: float
    median_margin: float
    q10_margin: float
    q90_margin: float
    mean_rank: float
```

### Organic Miner

**Location**: `paradigm_factory/v2/mine_tier3_organic.py`

**Purpose**: Discover new tier3 bundles from canonical corpus with multi-lineup validation.

**Input**:
- Canonical events corpus
- Existing tier3 (for deduplication)
- Danger threshold (calibrated to existing tier3 distribution)
- Lineup validation parameters

**Output**:
- Mined organic bundles (`tier3_organic.jsonl`)
- Stats summary (candidates tested, pass rate, quality distribution)
- Optional quarantine file for low-pass candidates

**Key Mechanism: Multi-Lineup Validation**

A candidate is not accepted because it appears hard against one negative lineup. It must remain dangerous across multiple independently sampled lineups:

```python
def validate_candidate(
    anchor: Event,
    positive: Event,
    lineups_per_candidate: int = 10,
    min_lineups_to_pass: int = 3,
    danger_threshold: float = 0.593
) -> Tuple[bool, int]:
    passes = 0
    for _ in range(lineups_per_candidate):
        negatives = sample_independent_lineup(anchor)
        danger = compute_danger(anchor, positive, negatives)
        if danger >= danger_threshold:
            passes += 1

    accepted = passes >= min_lineups_to_pass
    return accepted, passes
```

This prevents pseudo-killers: candidates that look hard due to a lucky negative draw but are easy against most lineups. In validation, this mechanism reduced pass rate to ~1.4% and produced consistently dangerous bundles.

**Interface**:
```python
def mine_organic(
    events_path: str,
    existing_tier3_path: str,
    output_path: str,
    threshold: float = 0.593,
    lineups_per_candidate: int = 10,
    min_lineups_to_pass: int = 3,
    candidates_per_sense: int = 15,
    topk_within_lemma: int = 10,
    topk_siblings: int = 5,
    topk_cross_lemma: int = 20,
    max_output: int = 15000
) -> MiningStats
```

### Experiment Runner

**Location**: `experiments/run_*.sh` scripts (to be unified into `crucible sweep`)

**Purpose**: Orchestrate multi-condition sweeps with parallel training and evaluation.

**Types**:

**Ratio Sweep**: Test multiple tier3 pool ratios
- Conditions: e.g., 85/10/5, 80/15/5, 75/15/10, 70/20/10, 65/25/10
- Output: Pareto frontier between frozen tier3 and organic holdout

**Step Sweep**: Test multiple training durations
- Conditions: e.g., 5k, 10k, 20k steps
- Output: Detect overfitting cliff where organic holdout collapses

**Quality Ablation**: Compare pool variants
- Conditions: e.g., ultra-only organic vs full organic
- Output: Determine if quality filtering improves generalization

**Output Structure**:
```
results/ratio_sweep/
  R85_10_5/
    seed_42/
      checkpoint_final.pt
      train.log
      frozen_scoreboard.json
      organic_scoreboard.json
    seed_123/
      ...
  R75_15_10/
    ...
  frozen_scoreboard.json      # Aggregated
  organic_scoreboard.json     # Aggregated
  pareto_analysis.json
```

### Report Generator

**Location**: `crucible report` command

**Purpose**: Generate visualizations from experiment results.

**Report Types**:

- **Pareto Frontier**: Scatter plot of frozen tier3 vs organic holdout
- **Margin Distributions**: Per-tier histograms and box plots
- **Step Trajectory**: Line plot of metrics vs training steps
- **Dilution Monitor**: Pool composition and exposure probabilities over time

**Output**: Static HTML files viewable locally or served via simple web server.

## Dataflow Example

Complete workflow from raw data to production config:

1. **Data Preparation**
   - Team provides canonical events (`canonicalized_v21.jsonl`, ~37k events, ~25k senses)

2. **Bundle Generation**
   - Bundle Generator produces `contrastive_bundles.jsonl` with tier1/tier2/tier3
   - Legacy tier3: 454 core killers

3. **Organic Mining**
   - Organic Miner scans events with multi-lineup validation
   - Produces `tier3_organic.jsonl` (400+ bundles at ~1.4% pass rate)

4. **Pool Management**
   - Tier3 Pool Manager merges legacy (454) + organic (400)
   - Splits organic 80/20 into train (320) and holdout (80)
   - Creates `tier3_organic_train.jsonl` and `organic_holdout_eval.jsonl`

5. **Training**
   - Curriculum Sampler loads merged bundles
   - Configures ratio 75/15/10 (legacy/organic/expanded)
   - Training Loop runs 10k steps
   - Saves `checkpoint_final.pt`

6. **Evaluation**
   - Evaluation Engine runs against frozen eval (`eval_v23_contrastive_5k.jsonl`)
   - Evaluation Engine runs against organic holdout
   - Produces two scoreboards

7. **Analysis**
   - Report Generator creates Pareto frontier and margin distribution charts
   - Guardrails check for dilution, regression, overfitting

8. **Capsule Packaging**
   - `crucible capsule pack` bundles checkpoint, config, scoreboards, metadata
   - Capsule is self-contained and reproducible

## Reproducibility Infrastructure

### Deterministic Seeding

All random operations use explicit seeds derived from config:
- Data shuffling
- Dropout (if enabled)
- Negative sampling in miner
- Train/holdout splits

### Eval Fingerprints

Every scoreboard includes a fingerprint:
```json
{
  "source_hash": "SHA256(eval_file_contents)[:16]",
  "seed": 42,
  "item_count": 3786,
  "tier_distribution": {"tier1": 1666, "tier2": 1666, "tier3": 454}
}
```

Scoreboards with different fingerprints cannot be directly compared without explicit override.

### Capsule Structure

Each run produces a self-contained capsule:
```
capsule/
  config.yaml           # Frozen copy of run config
  checkpoint_final.pt   # Model weights
  train.log             # Training stdout/stderr
  frozen_scoreboard.json
  organic_scoreboard.json
  metadata.json         # Run metadata, git hash, timestamps
```

### Config Logging

Full config YAML saved with every run. Git commit hash recorded if available. All parameters traceable from results to source.
