#!/usr/bin/env python3
"""
Parallel Comparison: v3 vs φ-Balanced Training
==============================================

Runs both v3 (fixed weights) and φ-balanced (adaptive weights) experiments
concurrently for direct comparison.

The key hypothesis: If φ-balanced mode still can't push hard slack above zero,
that's a loud signal the limit is representational or dataset-definitional,
not a tuning problem.

Usage:
    python scripts/run_parallel_comparison.py --bundles data.jsonl --epochs 3 --seeds 42
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the validation runner
import importlib.util
spec = importlib.util.spec_from_file_location("run_3seed_validation",
    os.path.join(os.path.dirname(__file__), "run_3seed_validation.py"))
validation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation_module)

run_single_seed = validation_module.run_single_seed
SeedResult = validation_module.SeedResult


def run_experiment_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for running a single experiment."""
    mode = args.pop('mode')
    seed = args['seed']

    print(f"\n[{mode}] Starting seed {seed}...")

    try:
        result = run_single_seed(**args)
        return {
            'mode': mode,
            'seed': seed,
            'passed': result.passed,
            'best_msr': result.best_msr,
            'final_slack': result.final_slack,
            'slack_positive_rate': result.slack_positive_rate,
            'success': True
        }
    except Exception as e:
        print(f"[{mode}] Seed {seed} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'mode': mode,
            'seed': seed,
            'passed': False,
            'best_msr': 0.0,
            'final_slack': 0.0,
            'slack_positive_rate': 0.0,
            'success': False,
            'error': str(e)
        }


def run_parallel_comparison(
    bundles_path: Path,
    seeds: List[int] = [42],
    epochs: int = 3,
    margin: float = 0.05,
    slack_weight: float = 1.5,
    hard_neg_mult: float = 2.0,
    track_killers: int = 20,
    output_dir: Path = None
):
    """
    Run v3 and φ-balanced experiments in parallel for each seed.

    Returns comparison results showing which mode performs better.
    """
    print("=" * 70)
    print("  PARALLEL COMPARISON: v3 vs φ-BALANCED")
    print("=" * 70)
    print(f"\nBundles: {bundles_path}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print(f"Margin: {margin}")
    print(f"Slack weight (v3): {slack_weight}")
    print(f"Hard-neg mult: {hard_neg_mult}")
    print(f"Killer tracking: {track_killers}")
    print()

    # Build experiment configurations
    experiments = []

    for seed in seeds:
        # v3 experiment (fixed weights, standard late_start=0.6)
        experiments.append({
            'mode': 'v3',
            'bundles_path': bundles_path,
            'seed': seed,
            'epochs': epochs,
            'margin': margin,
            'slack_weight': slack_weight,
            'hard_neg_penalty_mult': hard_neg_mult,
            'late_stage_start': 0.6,
            'late_stage_ramp': 0.35,
            'track_killers': track_killers,
            'killer_log_every': 100,
            'use_phi_balance': False
        })

        # φ-balanced experiment (adaptive weights, nested φ schedule)
        experiments.append({
            'mode': 'phi',
            'bundles_path': bundles_path,
            'seed': seed,
            'epochs': epochs,
            'margin': margin,
            'slack_weight': slack_weight,  # Base weight for φ controller
            'hard_neg_penalty_mult': hard_neg_mult,
            'late_stage_start': 0.618,  # Overridden by φ controller anyway
            'late_stage_ramp': 0.236,
            'track_killers': track_killers,
            'killer_log_every': 100,
            'use_phi_balance': True,
            'phi_ema_decay': 0.99
        })

    # Run experiments sequentially (can be parallelized with mp.Pool if needed)
    # Note: PyTorch/CUDA doesn't always play nice with multiprocessing
    results = []
    for exp in experiments:
        result = run_experiment_worker(exp)
        results.append(result)

    # Aggregate results by mode
    v3_results = [r for r in results if r['mode'] == 'v3']
    phi_results = [r for r in results if r['mode'] == 'phi']

    # Print comparison summary
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)

    print("\n  v3 Mode (Fixed Weights):")
    for r in v3_results:
        status = "[PASS]" if r['passed'] else "[FAIL]"
        print(f"    Seed {r['seed']}: MSR={r['best_msr']*100:.1f}%, "
              f"slack={r['final_slack']:+.4f}, rate={r['slack_positive_rate']*100:.1f}% {status}")

    avg_v3_rate = sum(r['slack_positive_rate'] for r in v3_results) / max(len(v3_results), 1)
    v3_passed = sum(1 for r in v3_results if r['passed'])
    print(f"    Average slack positive rate: {avg_v3_rate*100:.1f}%")
    print(f"    Seeds passed: {v3_passed}/{len(v3_results)}")

    print("\n  φ-Balanced Mode (Adaptive Weights):")
    for r in phi_results:
        status = "[PASS]" if r['passed'] else "[FAIL]"
        print(f"    Seed {r['seed']}: MSR={r['best_msr']*100:.1f}%, "
              f"slack={r['final_slack']:+.4f}, rate={r['slack_positive_rate']*100:.1f}% {status}")

    avg_phi_rate = sum(r['slack_positive_rate'] for r in phi_results) / max(len(phi_results), 1)
    phi_passed = sum(1 for r in phi_results if r['passed'])
    print(f"    Average slack positive rate: {avg_phi_rate*100:.1f}%")
    print(f"    Seeds passed: {phi_passed}/{len(phi_results)}")

    # Verdict
    print("\n  " + "-" * 66)
    if avg_phi_rate > avg_v3_rate + 0.05:
        print("  VERDICT: φ-balanced mode shows improvement over v3")
        print("           → Continue tuning with φ-based approach")
    elif avg_v3_rate > avg_phi_rate + 0.05:
        print("  VERDICT: v3 mode outperforms φ-balanced")
        print("           → Focus on curriculum/data improvements")
    else:
        if avg_phi_rate < 0.10 and avg_v3_rate < 0.10:
            print("  VERDICT: BOTH modes fail to achieve positive hard slack")
            print("           → Limit is representational or dataset-definitional")
            print("           → Focus on: (1) killer neg analysis, (2) data quality, (3) model capacity")
        else:
            print("  VERDICT: Modes perform similarly")
            print("           → Explore other hyperparameters or curriculum approaches")
    print("  " + "-" * 66)

    # Save comparison report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "bundles": str(bundles_path),
            "seeds": seeds,
            "epochs": epochs,
            "v3_results": v3_results,
            "phi_results": phi_results,
            "v3_avg_rate": avg_v3_rate,
            "phi_avg_rate": avg_phi_rate,
            "v3_passed": v3_passed,
            "phi_passed": phi_passed
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to: {report_path}")

    return {
        'v3': v3_results,
        'phi': phi_results,
        'v3_avg_rate': avg_v3_rate,
        'phi_avg_rate': avg_phi_rate
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel v3 vs φ-Balanced Comparison")
    parser.add_argument("--bundles", type=str, required=True, help="Path to bundles JSONL file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds to use (default: just 42 for quick comparison)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs per experiment")
    parser.add_argument("--margin", type=float, default=0.05, help="Certification margin")
    parser.add_argument("--slack-weight", type=float, default=1.5,
                        help="Base slack weight for v3 mode")
    parser.add_argument("--hard-neg-mult", type=float, default=2.0,
                        help="Hard-neg focus multiplier")
    parser.add_argument("--track-killers", type=int, default=20,
                        help="Track K worst hard-neg violations per batch")
    parser.add_argument("--output-dir", type=str, default="checkpoints/comparison",
                        help="Output directory for comparison report")

    args = parser.parse_args()

    run_parallel_comparison(
        bundles_path=Path(args.bundles),
        seeds=args.seeds,
        epochs=args.epochs,
        margin=args.margin,
        slack_weight=args.slack_weight,
        hard_neg_mult=args.hard_neg_mult,
        track_killers=args.track_killers,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
