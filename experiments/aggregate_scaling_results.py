#!/usr/bin/env python3
"""
Aggregate Scaling Experiment Results
=====================================

Collects results from all conditions/seeds and produces:
1. Summary statistics (mean, std across seeds)
2. Statistical significance tests
3. Tier-stratified analysis
4. Recommendation for best condition
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime


@dataclass
class ConditionResult:
    """Results for a single condition."""
    condition: str
    seeds: List[int]

    # Primary metrics (mean ± std)
    top1_mean: float
    top1_std: float
    top3_mean: float
    top3_std: float
    coherence_mean: float
    coherence_std: float

    # Per-seed results
    per_seed: Dict[int, Dict]

    # Ship blocker status
    all_passed: bool
    pass_rate: float


@dataclass
class ExperimentSummary:
    """Full experiment summary."""
    timestamp: str
    conditions: Dict[str, ConditionResult]

    # Best condition
    best_condition: str
    best_top1: float
    improvement_over_baseline: float

    # Statistical tests
    b_vs_a_significant: bool
    c_vs_b_significant: bool


def load_dashboard(path: Path) -> Optional[Dict]:
    """Load eval dashboard JSON."""
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


def extract_metrics(dashboard: Dict) -> Dict:
    """Extract key metrics from dashboard."""
    return {
        'top1': dashboard.get('retrieval_real', {}).get('top1_accuracy', 0),
        'top3': dashboard.get('retrieval_real', {}).get('top3_accuracy', 0),
        'coherence': dashboard.get('coherence', {}).get('accuracy', 0),
        'margin_mean': dashboard.get('retrieval_real', {}).get('margin_mean', 0),
        'margin_p5': dashboard.get('retrieval_real', {}).get('margin_p5', 0),
        'margin_p95': dashboard.get('retrieval_real', {}).get('margin_p95', 0),
    }


def paired_ttest(a: List[float], b: List[float]) -> float:
    """Simple paired t-test, returns p-value."""
    if len(a) != len(b) or len(a) < 2:
        return 1.0

    diffs = [b[i] - a[i] for i in range(len(a))]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    if std_diff == 0:
        return 0.0 if mean_diff != 0 else 1.0

    t_stat = mean_diff / (std_diff / np.sqrt(len(diffs)))

    # Approximate p-value using normal distribution (for small samples)
    from scipy import stats
    try:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(diffs)-1))
    except:
        p_value = 1.0

    return p_value


def aggregate_condition(experiment_dir: Path, condition: str, seeds: List[int]) -> ConditionResult:
    """Aggregate results for a single condition across seeds."""
    per_seed = {}
    top1_vals = []
    top3_vals = []
    coherence_vals = []
    passes = 0

    for seed in seeds:
        seed_dir = experiment_dir / f"{condition}_seed{seed}"
        dashboard_path = seed_dir / "eval_dashboard.json"

        dashboard = load_dashboard(dashboard_path)
        if dashboard:
            metrics = extract_metrics(dashboard)
            per_seed[seed] = metrics
            top1_vals.append(metrics['top1'])
            top3_vals.append(metrics['top3'])
            coherence_vals.append(metrics['coherence'])

            # Check ship blocker
            if metrics['top1'] >= 0.30 and metrics['top3'] >= 0.50 and metrics['coherence'] >= 0.80:
                passes += 1
        else:
            per_seed[seed] = {'error': 'No dashboard found'}

    return ConditionResult(
        condition=condition,
        seeds=seeds,
        top1_mean=np.mean(top1_vals) if top1_vals else 0,
        top1_std=np.std(top1_vals) if top1_vals else 0,
        top3_mean=np.mean(top3_vals) if top3_vals else 0,
        top3_std=np.std(top3_vals) if top3_vals else 0,
        coherence_mean=np.mean(coherence_vals) if coherence_vals else 0,
        coherence_std=np.std(coherence_vals) if coherence_vals else 0,
        per_seed=per_seed,
        all_passed=(passes == len(seeds)),
        pass_rate=passes / len(seeds) if seeds else 0
    )


def aggregate_results(experiment_dir: Path) -> ExperimentSummary:
    """Aggregate all experiment results."""
    experiment_dir = Path(experiment_dir)
    seeds = [42, 123, 456]

    # Aggregate each condition
    conditions = {}
    for cond in ['A', 'B', 'C']:
        conditions[cond] = aggregate_condition(experiment_dir, cond, seeds)

    # Find best condition
    best_cond = max(conditions.keys(), key=lambda c: conditions[c].top1_mean)
    best_top1 = conditions[best_cond].top1_mean
    baseline_top1 = conditions['A'].top1_mean
    improvement = best_top1 - baseline_top1

    # Statistical tests
    try:
        a_top1 = [conditions['A'].per_seed[s]['top1'] for s in seeds if s in conditions['A'].per_seed and 'top1' in conditions['A'].per_seed[s]]
        b_top1 = [conditions['B'].per_seed[s]['top1'] for s in seeds if s in conditions['B'].per_seed and 'top1' in conditions['B'].per_seed[s]]
        c_top1 = [conditions['C'].per_seed[s]['top1'] for s in seeds if s in conditions['C'].per_seed and 'top1' in conditions['C'].per_seed[s]]

        b_vs_a_p = paired_ttest(a_top1, b_top1)
        c_vs_b_p = paired_ttest(b_top1, c_top1)
    except:
        b_vs_a_p = 1.0
        c_vs_b_p = 1.0

    return ExperimentSummary(
        timestamp=datetime.now().isoformat(),
        conditions={k: asdict(v) for k, v in conditions.items()},
        best_condition=best_cond,
        best_top1=best_top1,
        improvement_over_baseline=improvement,
        b_vs_a_significant=(b_vs_a_p < 0.05),
        c_vs_b_significant=(c_vs_b_p < 0.05)
    )


def print_summary(summary: ExperimentSummary):
    """Print human-readable summary."""
    print("=" * 60)
    print(" SCALING EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Timestamp: {summary.timestamp}")
    print()

    print("CONDITION COMPARISON:")
    print("-" * 60)
    print(f"{'Condition':<12} {'Top-1':<18} {'Top-3':<18} {'Coherence':<18} {'Pass':<6}")
    print("-" * 60)

    for cond in ['A', 'B', 'C']:
        c = summary.conditions[cond]
        top1_str = f"{c['top1_mean']:.1%} ± {c['top1_std']:.1%}"
        top3_str = f"{c['top3_mean']:.1%} ± {c['top3_std']:.1%}"
        coh_str = f"{c['coherence_mean']:.1%} ± {c['coherence_std']:.1%}"
        pass_str = "✓" if c['all_passed'] else "✗"
        print(f"{cond:<12} {top1_str:<18} {top3_str:<18} {coh_str:<18} {pass_str:<6}")

    print()
    print("STATISTICAL SIGNIFICANCE:")
    print("-" * 60)
    print(f"  B vs A: {'SIGNIFICANT' if summary.b_vs_a_significant else 'not significant'}")
    print(f"  C vs B: {'SIGNIFICANT' if summary.c_vs_b_significant else 'not significant'}")

    print()
    print("RECOMMENDATION:")
    print("-" * 60)
    print(f"  Best condition: {summary.best_condition}")
    print(f"  Top-1 accuracy: {summary.best_top1:.1%}")
    print(f"  Improvement over baseline: {summary.improvement_over_baseline:+.1%}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_scaling_results.py <experiment_dir>")
        sys.exit(1)

    experiment_dir = Path(sys.argv[1])

    if not experiment_dir.exists():
        print(f"Error: Directory not found: {experiment_dir}")
        sys.exit(1)

    summary = aggregate_results(experiment_dir)

    # Print summary
    print_summary(summary)

    # Save JSON
    output_path = experiment_dir / "experiment_summary.json"
    with open(output_path, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
