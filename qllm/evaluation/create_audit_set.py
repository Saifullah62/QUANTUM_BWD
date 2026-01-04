#!/usr/bin/env python3
"""
Create Frozen Attention Audit Set
=================================

Creates a frozen 50-item audit set for attention pattern analysis.
Samples from bundles to ensure cue_tokens are available.
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def create_audit_set_from_bundles(
    bundles_path: Path,
    output_path: Path,
    n_items: int = 50,
    seed: int = 42
) -> int:
    """
    Create frozen audit set from bundle anchors.

    Args:
        bundles_path: Path to all_bundles.jsonl
        output_path: Output path for audit set
        n_items: Target number of items
        seed: Random seed

    Returns:
        Number of items created
    """
    random.seed(seed)

    # Load bundles
    items = []
    with open(bundles_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                bundle = json.loads(line)
                anchor = bundle.get('anchor', {})

                # Only include if has cue_tokens
                cue_tokens = anchor.get('cue_tokens', [])
                if not cue_tokens:
                    continue

                # Convert to audit format
                audit_item = {
                    "eval_id": bundle.get('id', ''),
                    "lemma": bundle.get('lemma', ''),
                    "query": anchor.get('text', ''),
                    "target_sense_id": anchor.get('sense_id', ''),
                    "sense_gloss": anchor.get('sense_gloss', ''),
                    "cue_tokens": cue_tokens,
                    "cue_tier": _infer_cue_tier(anchor.get('quality', {})),
                    "span": anchor.get('span', {}),
                    "source": anchor.get('source', ''),
                    "difficulty_tier": anchor.get('difficulty_tier', ''),
                }
                items.append(audit_item)

    if not items:
        print(f"Warning: No items with cue_tokens found")
        return 0

    # Group by cue tier and difficulty
    by_tier = {}
    for item in items:
        tier = item.get('cue_tier', 'unknown')
        by_tier.setdefault(tier, []).append(item)

    print(f"Found {len(items)} items with cue_tokens")
    for tier, tier_items in sorted(by_tier.items()):
        print(f"  {tier}: {len(tier_items)}")

    # Sample to get diversity
    selected = []

    # Try to get mix of tiers
    for tier in ['high_cue', 'medium_cue', 'low_cue', 'unknown']:
        tier_items = by_tier.get(tier, [])
        if tier_items:
            # Sample up to 1/3 from each tier
            tier_quota = max(1, n_items // 3)
            sample = random.sample(tier_items, min(tier_quota, len(tier_items)))
            selected.extend(sample)

    # If we need more, sample from all remaining
    if len(selected) < n_items:
        remaining = [i for i in items if i not in selected]
        extra = n_items - len(selected)
        if remaining:
            selected.extend(random.sample(remaining, min(extra, len(remaining))))

    # Trim to exact size
    if len(selected) > n_items:
        selected = random.sample(selected, n_items)

    # Shuffle
    random.shuffle(selected)

    # Also get diversity of lemmas
    lemmas = set(i['lemma'] for i in selected)
    print(f"\nSelected {len(selected)} items from {len(lemmas)} unique lemmas")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Summary by tier
    final_by_tier = {}
    for item in selected:
        tier = item.get('cue_tier', 'unknown')
        final_by_tier[tier] = final_by_tier.get(tier, 0) + 1

    print(f"\nFinal distribution:")
    for tier, count in sorted(final_by_tier.items()):
        print(f"  {tier}: {count}")

    print(f"\nWritten to: {output_path}")
    return len(selected)


def _infer_cue_tier(quality: Dict) -> str:
    """Infer cue tier from quality metadata."""
    cue_strength = quality.get('cue_strength', 0.5)
    if cue_strength >= 0.7:
        return 'high_cue'
    elif cue_strength >= 0.4:
        return 'medium_cue'
    else:
        return 'low_cue'


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create frozen attention audit set")
    parser.add_argument("--bundles", type=Path,
                       default=Path("paradigm_factory/v2/bundles_v22/all_bundles_v22.jsonl"),
                       help="Path to bundles JSONL")
    parser.add_argument("--output", type=Path,
                       default=Path("qllm/evaluation/frozen_audit_set.jsonl"),
                       help="Output path")
    parser.add_argument("--n-items", type=int, default=50,
                       help="Number of items")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Handle relative paths
    project_root = Path(__file__).parent.parent.parent
    bundles_path = project_root / args.bundles if not args.bundles.is_absolute() else args.bundles
    output_path = project_root / args.output if not args.output.is_absolute() else args.output

    create_audit_set_from_bundles(bundles_path, output_path, args.n_items, args.seed)
