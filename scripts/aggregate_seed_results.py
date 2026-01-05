#!/usr/bin/env python3
"""
Aggregate Multi-Seed Results
============================

Compares results across multiple random seeds to assess reproducibility.

Usage:
    python scripts/aggregate_seed_results.py --pattern "proof_ablation_runs/proof_v2_seed*_*/proof_v2_summary.json"
"""

import argparse
import json
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_results(pattern: str) -> List[Dict]:
    """Load all matching result files."""
    results = []
    for path in glob.glob(pattern):
        with open(path, 'r') as f:
            data = json.load(f)
            data['_path'] = path
            results.append(data)
    return results


def aggregate_ablation_results(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across seeds for each ablation."""

    ablation_names = ['baseline', 'semantic_phase', 'lindblad', 'qualia', 'all_paradigms']
    aggregated = {}

    for name in ablation_names:
        best_losses = []
        avg_losses = []
        std_losses = []

        for r in results:
            if 'results' in r and name in r['results']:
                ablation = r['results'][name]
                best_losses.append(ablation.get('best_loss', float('nan')))
                avg_losses.append(ablation.get('avg_last_n_loss', float('nan')))
                std_losses.append(ablation.get('std_last_n_loss', float('nan')))

        if best_losses:
            aggregated[name] = {
                'best_loss': {
                    'mean': np.nanmean(best_losses),
                    'std': np.nanstd(best_losses),
                    'values': best_losses
                },
                'avg_last_n_loss': {
                    'mean': np.nanmean(avg_losses),
                    'std': np.nanstd(avg_losses),
                    'values': avg_losses
                },
                'n_seeds': len(best_losses)
            }

    return aggregated


def compute_deltas(aggregated: Dict) -> Dict[str, Any]:
    """Compute deltas vs baseline for each ablation."""

    baseline_best = aggregated.get('baseline', {}).get('best_loss', {}).get('mean', float('inf'))
    baseline_avg = aggregated.get('baseline', {}).get('avg_last_n_loss', {}).get('mean', float('inf'))

    deltas = {}
    for name, data in aggregated.items():
        best_mean = data['best_loss']['mean']
        avg_mean = data['avg_last_n_loss']['mean']

        delta_best = best_mean - baseline_best
        delta_best_pct = (delta_best / baseline_best) * 100 if baseline_best > 0 else 0

        delta_avg = avg_mean - baseline_avg
        delta_avg_pct = (delta_avg / baseline_avg) * 100 if baseline_avg > 0 else 0

        deltas[name] = {
            'delta_best': delta_best,
            'delta_best_pct': delta_best_pct,
            'delta_avg': delta_avg,
            'delta_avg_pct': delta_avg_pct
        }

    return deltas


def assess_reproducibility(aggregated: Dict) -> Dict[str, Any]:
    """Assess whether results are reproducible across seeds."""

    assessment = {}

    for name, data in aggregated.items():
        # Coefficient of variation (lower = more reproducible)
        best_cv = data['best_loss']['std'] / (data['best_loss']['mean'] + 1e-8)
        avg_cv = data['avg_last_n_loss']['std'] / (data['avg_last_n_loss']['mean'] + 1e-8)

        # Is the effect consistent? (all seeds in same direction)
        if 'baseline' in aggregated and name != 'baseline':
            baseline_vals = aggregated['baseline']['best_loss']['values']
            ablation_vals = data['best_loss']['values']

            if len(baseline_vals) == len(ablation_vals):
                # Check if all seeds show improvement
                improvements = [a < b for a, b in zip(ablation_vals, baseline_vals)]
                consistent = all(improvements) or not any(improvements)
            else:
                consistent = False
        else:
            consistent = True

        assessment[name] = {
            'best_cv': best_cv,
            'avg_cv': avg_cv,
            'is_reproducible': best_cv < 0.1,  # <10% CV is reproducible
            'effect_consistent': consistent
        }

    return assessment


def print_report(results: List[Dict], aggregated: Dict, deltas: Dict, assessment: Dict):
    """Print comprehensive report."""

    print("=" * 70)
    print("  MULTI-SEED REPRODUCIBILITY REPORT")
    print("=" * 70)

    print(f"\nSeeds analyzed: {len(results)}")
    seeds = [r.get('seed', 'unknown') for r in results]
    print(f"Seed values: {seeds}")

    print("\n" + "=" * 70)
    print("  AGGREGATED RESULTS (mean ± std across seeds)")
    print("=" * 70)

    for name in ['baseline', 'semantic_phase', 'lindblad', 'qualia', 'all_paradigms']:
        if name not in aggregated:
            continue

        data = aggregated[name]
        delta = deltas[name]

        print(f"\n{name}:")
        print(f"  Best loss: {data['best_loss']['mean']:.4f} ± {data['best_loss']['std']:.4f} "
              f"({delta['delta_best']:+.4f}, {delta['delta_best_pct']:+.2f}%)")
        print(f"  Avg loss:  {data['avg_last_n_loss']['mean']:.4f} ± {data['avg_last_n_loss']['std']:.4f} "
              f"({delta['delta_avg']:+.4f}, {delta['delta_avg_pct']:+.2f}%)")

    print("\n" + "=" * 70)
    print("  REPRODUCIBILITY ASSESSMENT")
    print("=" * 70)

    for name, assess in assessment.items():
        status = "PASS" if assess['is_reproducible'] else "FAIL"
        consistent = "YES" if assess['effect_consistent'] else "NO"
        print(f"\n{name}:")
        print(f"  CV (best): {assess['best_cv']:.3f} - {status}")
        print(f"  CV (avg):  {assess['avg_cv']:.3f}")
        print(f"  Effect consistent across seeds: {consistent}")

    # Final verdict
    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)

    # Check if all_paradigms consistently beats baseline
    if 'all_paradigms' in aggregated and 'baseline' in aggregated:
        all_best = aggregated['all_paradigms']['best_loss']['values']
        base_best = aggregated['baseline']['best_loss']['values']

        if len(all_best) == len(base_best):
            improvements = [a < b for a, b in zip(all_best, base_best)]
            num_improved = sum(improvements)
            total = len(improvements)

            print(f"\nall_paradigms vs baseline:")
            print(f"  Seeds showing improvement: {num_improved}/{total}")

            if num_improved == total:
                print("\n  >>> REPRODUCIBILITY CONFIRMED <<<")
                print("  All seeds show paradigm training beats baseline!")
            elif num_improved > total / 2:
                print("\n  >>> PARTIAL REPRODUCIBILITY <<<")
                print(f"  {num_improved}/{total} seeds show improvement")
            else:
                print("\n  >>> NOT REPRODUCIBLE <<<")
                print("  Results not consistent across seeds")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument('--pattern', type=str,
                        default='proof_ablation_runs/proof_v2_seed*_*/proof_v2_summary.json',
                        help='Glob pattern for result files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for aggregated JSON')

    args = parser.parse_args()

    print(f"Loading results from: {args.pattern}")
    results = load_results(args.pattern)

    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} result files")

    # Aggregate
    aggregated = aggregate_ablation_results(results)
    deltas = compute_deltas(aggregated)
    assessment = assess_reproducibility(aggregated)

    # Print report
    print_report(results, aggregated, deltas, assessment)

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('proof_ablation_runs') / 'multi_seed_summary.json'

    output_data = {
        'n_seeds': len(results),
        'seeds': [r.get('seed', 'unknown') for r in results],
        'aggregated': aggregated,
        'deltas': deltas,
        'assessment': assessment
    }

    # Convert numpy to python types
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(output_data), f, indent=2)

    print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()
