#!/usr/bin/env python3
"""
Tier4 Killer Exporter
=====================

Exports the hardest failures (4-5% very negative cases) as a first-class
dataset slice for curriculum injection during training.

Tier4 criteria:
- Margin < -0.15 (very wrong)
- OR severity > 1.8 (high impact)
- OR rank > 10 (completely lost)

These items get special treatment in training:
- Appear at controlled rate each epoch
- Not drowned out by easy majority
- Tracked separately in metrics

Usage:
  python paradigm_factory/v2/killers/export_tier4_killers.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Tier4Config:
    """Configuration for tier4 extraction."""
    margin_threshold: float = -0.15  # Items with margin below this
    severity_threshold: float = 1.8  # Items with severity above this
    rank_threshold: int = 10         # Items with rank above this
    max_items: int = 100            # Cap on tier4 size
    target_rate: float = 0.05       # Target 5% of total evals


@dataclass
class Tier4Manifest:
    """Manifest for tier4 dataset."""
    version: str
    created: str
    config: Dict
    source_file: str
    total_killers: int
    tier4_count: int
    tier4_rate: float
    criteria_breakdown: Dict
    fingerprint: str


def extract_tier4_killers(
    killers_file: Path,
    output_dir: Path,
    config: Tier4Config = None
) -> Tier4Manifest:
    """
    Extract tier4 killers from a killers file.

    Args:
        killers_file: Path to killers JSONL (from eval harness)
        output_dir: Output directory
        config: Extraction configuration

    Returns:
        Tier4Manifest with extraction metadata
    """
    config = config or Tier4Config()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load killers
    killers = []
    with open(killers_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                killers.append(json.loads(line))

    print(f"Loaded {len(killers)} killers")

    # Classify into tier4
    tier4 = []
    criteria_counts = {
        'margin': 0,
        'severity': 0,
        'rank': 0,
        'multiple': 0
    }

    for killer in killers:
        margin = killer.get('margin', 0)
        severity = killer.get('severity', 0)
        rank = killer.get('best_correct_rank', 1)

        reasons = []

        if margin < config.margin_threshold:
            reasons.append('margin')
            criteria_counts['margin'] += 1

        if severity > config.severity_threshold:
            reasons.append('severity')
            criteria_counts['severity'] += 1

        if rank > config.rank_threshold:
            reasons.append('rank')
            criteria_counts['rank'] += 1

        if len(reasons) > 1:
            criteria_counts['multiple'] += 1

        if reasons:
            # Add tier4 metadata
            killer['_tier4'] = {
                'reasons': reasons,
                'margin': margin,
                'severity': severity,
                'rank': rank,
                'extracted_at': datetime.now().isoformat()
            }
            tier4.append(killer)

    # Sort by severity (worst first)
    tier4.sort(key=lambda x: -x.get('severity', 0))

    # Cap if needed
    if len(tier4) > config.max_items:
        print(f"Capping tier4 from {len(tier4)} to {config.max_items}")
        tier4 = tier4[:config.max_items]

    # Compute rate
    tier4_rate = len(tier4) / len(killers) if killers else 0

    print(f"Tier4 items: {len(tier4)} ({tier4_rate:.1%} of killers)")

    # Write tier4 file
    tier4_file = output_dir / "tier4_killers.jsonl"
    with open(tier4_file, 'w', encoding='utf-8') as f:
        for item in tier4:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Create manifest
    manifest = Tier4Manifest(
        version="1.0",
        created=datetime.now().isoformat(),
        config=asdict(config),
        source_file=str(killers_file),
        total_killers=len(killers),
        tier4_count=len(tier4),
        tier4_rate=tier4_rate,
        criteria_breakdown=criteria_counts,
        fingerprint=f"tier4_v1|n={len(tier4)}|rate={tier4_rate:.1%}|max_sev={tier4[0].get('severity', 0):.2f}" if tier4 else "empty"
    )

    manifest_file = output_dir / "tier4_manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(asdict(manifest), f, indent=2)

    # Write curriculum config (for training integration)
    curriculum_config = {
        "tier4_file": str(tier4_file),
        "injection_rate": config.target_rate,
        "injection_strategy": "fixed_quota_per_batch",
        "quota_per_batch": max(1, int(32 * config.target_rate)),  # Assuming batch_size=32
        "shuffle_per_epoch": True,
        "weight_multiplier": 2.0,  # Higher loss weight for tier4
        "notes": "Tier4 items should appear at controlled rate, not drowned by easy majority"
    }

    curriculum_file = output_dir / "curriculum_config.json"
    with open(curriculum_file, 'w', encoding='utf-8') as f:
        json.dump(curriculum_config, f, indent=2)

    print(f"\nOutputs:")
    print(f"  {tier4_file}")
    print(f"  {manifest_file}")
    print(f"  {curriculum_file}")

    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export tier4 killers")
    parser.add_argument("--input", type=Path,
                       help="Input killers JSONL file")
    parser.add_argument("--output", type=Path,
                       default=PROJECT_ROOT / "paradigm_factory" / "v2" / "killers" / "tier4",
                       help="Output directory")
    parser.add_argument("--margin-threshold", type=float, default=-0.15,
                       help="Margin threshold for tier4")
    parser.add_argument("--severity-threshold", type=float, default=1.8,
                       help="Severity threshold for tier4")
    parser.add_argument("--rank-threshold", type=int, default=10,
                       help="Rank threshold for tier4")
    args = parser.parse_args()

    # Find input file if not specified
    if args.input is None:
        # Look for most recent killers file in dashboard_reports
        reports_dir = PROJECT_ROOT / "paradigm_factory" / "v2" / "dashboard_reports"
        killers_files = sorted(reports_dir.glob("killers_*.jsonl"), reverse=True)
        if killers_files:
            args.input = killers_files[0]
            print(f"Using most recent killers file: {args.input}")
        else:
            print("Error: No killers file found. Run eval harness first.")
            sys.exit(1)

    config = Tier4Config(
        margin_threshold=args.margin_threshold,
        severity_threshold=args.severity_threshold,
        rank_threshold=args.rank_threshold
    )

    print("=" * 60)
    print("TIER4 KILLER EXPORTER")
    print("=" * 60)

    manifest = extract_tier4_killers(args.input, args.output, config)

    print(f"\n{manifest.fingerprint}")


if __name__ == "__main__":
    main()
