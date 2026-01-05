#!/usr/bin/env python3
"""
Production Capsule Builder
==========================

Creates a production-ready capsule with the validated recipe:
- R75/15/10 tier3 ratios
- 10k steps (before overfitting cliff)
- 3 seeds, pick median on organic holdout

Usage:
    python experiments/build_production_capsule.py \
        --bundles path/to/bundles.jsonl \
        --frozen-eval path/to/frozen_eval.jsonl \
        --organic-eval path/to/organic_holdout.jsonl \
        --output-dir capsules/production_v1

Output:
    capsules/production_v1/
        manifest.json           # Locked file hashes and fingerprints
        config.yaml             # Frozen configuration
        checkpoint_final.pt     # Median-performer checkpoint
        frozen_scoreboard.json
        organic_scoreboard.json
        seeds/                  # All 3 seed runs for audit
            seed_42/
            seed_123/
            seed_456/
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def run_training(
    bundles: Path,
    output_dir: Path,
    seed: int,
    steps: int = 10000,
    device: str = "cuda"
) -> bool:
    """Run training for one seed."""
    trainer = PROJECT_ROOT / "scripts" / "train_curriculum_v3.py"

    cmd = [
        sys.executable, str(trainer),
        "--bundles", str(bundles),
        "--steps", str(steps),
        "--seed", str(seed),
        "--curriculum", "uniform",
        "--device", device,
        "--tier3-legacy", "0.75",
        "--tier3-organic", "0.15",
        "--tier3-expanded", "0.10",
        "--output-dir", str(output_dir),
    ]

    print(f"  Training seed {seed}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [ERROR] Training failed for seed {seed}")
        print(result.stderr)
        return False

    return True


def run_evaluation(
    results_root: Path,
    eval_path: Path,
    output_path: Path,
    device: str = "cuda"
) -> bool:
    """Run evaluation across all seeds."""
    evaluator = PROJECT_ROOT / "experiments" / "eval_capsules.py"

    cmd = [
        sys.executable, str(evaluator),
        "--eval", str(eval_path),
        "--results-root", str(results_root),
        "--device", device,
        "--out", str(output_path),
    ]

    print(f"  Evaluating against {eval_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [ERROR] Evaluation failed")
        print(result.stderr)
        return False

    return True


def pick_median_checkpoint(
    organic_scoreboard: Path,
    seeds_dir: Path
) -> Tuple[int, Path]:
    """Pick the median-performing seed on organic holdout."""
    with open(organic_scoreboard) as f:
        scores = json.load(f)

    # Extract per-seed organic holdout accuracy
    seed_scores = []
    for capsule in scores.get("capsules", []):
        if "error" in capsule:
            continue
        seed = capsule.get("seed", 0)
        acc = capsule.get("overall", {}).get("accuracy", 0)
        seed_scores.append((seed, acc))

    if len(seed_scores) < 3:
        print(f"  [WARN] Only {len(seed_scores)} valid seeds, expected 3")

    # Sort by accuracy and pick median
    seed_scores.sort(key=lambda x: x[1])
    median_idx = len(seed_scores) // 2
    median_seed, median_acc = seed_scores[median_idx]

    checkpoint = seeds_dir / f"seed_{median_seed}" / "checkpoint_final.pt"
    print(f"  Median performer: seed {median_seed} (organic holdout: {median_acc:.1%})")

    return median_seed, checkpoint


def create_manifest(
    bundles: Path,
    frozen_eval: Path,
    organic_eval: Path,
    checkpoint: Path,
    median_seed: int,
    output_dir: Path
) -> Dict:
    """Create manifest with locked hashes and metadata."""
    manifest = {
        "capsule_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "recipe": {
            "tier3_ratios": {"legacy": 0.75, "organic": 0.15, "expanded": 0.10},
            "steps": 10000,
            "seeds": [42, 123, 456],
            "selected_seed": median_seed,
            "selection_criterion": "median on organic holdout",
        },
        "data": {
            "bundles": {
                "path": str(bundles),
                "hash": compute_file_hash(bundles),
            },
            "frozen_eval": {
                "path": str(frozen_eval),
                "hash": compute_file_hash(frozen_eval),
            },
            "organic_eval": {
                "path": str(organic_eval),
                "hash": compute_file_hash(organic_eval),
            },
        },
        "checkpoint": {
            "path": "checkpoint_final.pt",
            "hash": compute_file_hash(checkpoint) if checkpoint.exists() else None,
        },
    }

    return manifest


def create_config_yaml(output_dir: Path) -> None:
    """Create frozen config.yaml."""
    config = """# Production Configuration (Frozen)
# Generated by build_production_capsule.py

data:
  bundles: ${BUNDLES_PATH}
  frozen_eval: ${FROZEN_EVAL_PATH}
  organic_eval: ${ORGANIC_EVAL_PATH}

training:
  steps: 10000  # Validated: before overfitting cliff
  batch_size: 32
  learning_rate: 1.0e-3

curriculum:
  mode: uniform
  tier_probabilities: [0.60, 0.30, 0.10]
  tier3_ratios:
    legacy: 0.75   # Validated: Pareto-optimal
    organic: 0.15
    expanded: 0.10

guardrails:
  dilution_warning_threshold: 30
  regression_threshold: 0.02
"""
    (output_dir / "config.yaml").write_text(config)


def main():
    parser = argparse.ArgumentParser(description="Build production capsule")
    parser.add_argument("--bundles", required=True, help="Training bundles JSONL")
    parser.add_argument("--frozen-eval", required=True, help="Frozen eval JSONL")
    parser.add_argument("--organic-eval", required=True, help="Organic holdout JSONL")
    parser.add_argument("--output-dir", default="capsules/production_v1",
                        help="Output capsule directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use existing checkpoints")
    args = parser.parse_args()

    bundles = Path(args.bundles)
    frozen_eval = Path(args.frozen_eval)
    organic_eval = Path(args.organic_eval)
    output_dir = Path(args.output_dir)
    seeds_dir = output_dir / "seeds"

    # Validate inputs
    for path, name in [(bundles, "bundles"), (frozen_eval, "frozen_eval"),
                       (organic_eval, "organic_eval")]:
        if not path.exists():
            print(f"ERROR: {name} not found: {path}")
            return 1

    print("=" * 60)
    print("PRODUCTION CAPSULE BUILDER")
    print("=" * 60)
    print(f"Bundles: {bundles}")
    print(f"Frozen Eval: {frozen_eval}")
    print(f"Organic Eval: {organic_eval}")
    print(f"Output: {output_dir}")
    print(f"Recipe: R75/15/10 @ 10k steps, 3 seeds")
    print()

    # Phase 1: Training
    if not args.skip_training:
        print("PHASE 1: TRAINING")
        print("-" * 60)
        seeds_dir.mkdir(parents=True, exist_ok=True)

        for seed in [42, 123, 456]:
            seed_dir = seeds_dir / f"seed_{seed}"
            checkpoint = seed_dir / "checkpoint_final.pt"

            if checkpoint.exists():
                print(f"  [SKIP] seed {seed} already trained")
                continue

            success = run_training(
                bundles=bundles,
                output_dir=seed_dir,
                seed=seed,
                device=args.device
            )
            if not success:
                return 1

        print()

    # Phase 2: Evaluation
    print("PHASE 2: EVALUATION")
    print("-" * 60)

    frozen_scoreboard = output_dir / "frozen_scoreboard.json"
    organic_scoreboard = output_dir / "organic_scoreboard.json"

    if not run_evaluation(seeds_dir, frozen_eval, frozen_scoreboard, args.device):
        return 1

    if not run_evaluation(seeds_dir, organic_eval, organic_scoreboard, args.device):
        return 1

    print()

    # Phase 3: Select median checkpoint
    print("PHASE 3: SELECT MEDIAN CHECKPOINT")
    print("-" * 60)

    median_seed, median_checkpoint = pick_median_checkpoint(
        organic_scoreboard, seeds_dir
    )

    # Copy median checkpoint to capsule root
    final_checkpoint = output_dir / "checkpoint_final.pt"
    shutil.copy(median_checkpoint, final_checkpoint)
    print(f"  Copied to: {final_checkpoint}")
    print()

    # Phase 4: Create manifest
    print("PHASE 4: CREATE MANIFEST")
    print("-" * 60)

    manifest = create_manifest(
        bundles, frozen_eval, organic_eval,
        final_checkpoint, median_seed, output_dir
    )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote: {manifest_path}")

    create_config_yaml(output_dir)
    print(f"  Wrote: {output_dir / 'config.yaml'}")
    print()

    # Summary
    print("=" * 60)
    print("CAPSULE COMPLETE")
    print("=" * 60)
    print(f"Location: {output_dir}")
    print(f"Checkpoint: checkpoint_final.pt (seed {median_seed})")
    print(f"Scoreboards: frozen_scoreboard.json, organic_scoreboard.json")
    print(f"Manifest: manifest.json")
    print()

    # Print scoreboard summary
    with open(frozen_scoreboard) as f:
        frozen = json.load(f)
    with open(organic_scoreboard) as f:
        organic = json.load(f)

    print("RESULTS:")
    for capsule in frozen.get("capsules", []):
        if capsule.get("seed") == median_seed:
            t3 = capsule.get("by_tier", {}).get("tier3_adversarial", {})
            print(f"  Frozen Tier3: {t3.get('pass_rate', 0):.1%}")
            break

    for capsule in organic.get("capsules", []):
        if capsule.get("seed") == median_seed:
            overall = capsule.get("overall", {})
            print(f"  Organic Holdout: {overall.get('accuracy', 0):.1%}")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
