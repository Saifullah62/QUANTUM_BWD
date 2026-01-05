#!/usr/bin/env python3
"""
Merge Tier3 Expanded Bundles
============================

Merges original v23 contrastive bundles with expanded tier3 bundles,
with strict signature-based deduplication.

Dedupe signature:
- anchor.text hash (normalized)
- positive.text hash (normalized)
- sorted list of negative.text hashes (normalized)
- lemma + anchor_sense_id

Output: contrastive_bundles_v23_tier3x.jsonl
"""

import json
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse


def normalize_text(text: str) -> str:
    """Normalize text for hashing."""
    return ' '.join(text.lower().split())


def compute_signature(bundle: dict) -> str:
    """
    Compute stable signature for deduplication.

    Components:
    - anchor text hash
    - positive text hash
    - sorted negative text hashes
    - lemma + sense_id
    """
    anchor = bundle.get('anchor', {})
    positive = bundle.get('positive', {})
    negatives = bundle.get('negatives', {})

    # Anchor hash
    anchor_text = normalize_text(anchor.get('text', ''))
    anchor_hash = hashlib.md5(anchor_text.encode()).hexdigest()[:8]

    # Positive hash
    pos_text = normalize_text(positive.get('text', ''))
    pos_hash = hashlib.md5(pos_text.encode()).hexdigest()[:8]

    # Negative hashes (sorted for stability)
    neg_texts = []
    for neg in negatives.get('within_lemma', []):
        neg_texts.append(normalize_text(neg.get('text', '')))
    for neg in negatives.get('cross_lemma', []):
        neg_texts.append(normalize_text(neg.get('text', '')))

    neg_hashes = sorted([hashlib.md5(t.encode()).hexdigest()[:8] for t in neg_texts])
    neg_sig = '-'.join(neg_hashes) if neg_hashes else 'none'

    # Lemma + sense
    lemma = bundle.get('lemma', '').lower()
    sense_id = anchor.get('sense_id', '')[:20]

    return f"{lemma}:{sense_id}:{anchor_hash}:{pos_hash}:{neg_sig}"


def merge_bundles(
    original_path: Path,
    expanded_path: Path,
    output_path: Path
) -> dict:
    """Merge bundles with deduplication."""

    print("=" * 60)
    print("MERGING TIER3 EXPANDED BUNDLES")
    print("=" * 60)

    stats = {
        'original_count': 0,
        'expanded_count': 0,
        'duplicates_skipped': 0,
        'expanded_added': 0,
        'final_count': 0,
        'tier_counts': Counter(),
    }

    # Load original bundles and build signature index
    print(f"\nLoading original bundles from {original_path.name}...")
    seen_signatures = set()
    original_bundles = []

    with open(original_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                bundle = json.loads(line)
                sig = compute_signature(bundle)
                seen_signatures.add(sig)
                original_bundles.append(bundle)
                stats['original_count'] += 1

                tier = bundle.get('metadata', {}).get('difficulty_tier', 'unknown')
                stats['tier_counts'][tier] += 1

    print(f"[OK] Loaded {stats['original_count']} original bundles")
    print(f"     Tier distribution: {dict(stats['tier_counts'])}")

    # Load expanded bundles and dedupe
    print(f"\nLoading expanded tier3 from {expanded_path.name}...")
    expanded_bundles = []

    with open(expanded_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                bundle = json.loads(line)
                stats['expanded_count'] += 1

                sig = compute_signature(bundle)
                if sig in seen_signatures:
                    stats['duplicates_skipped'] += 1
                    continue

                # Mark as expansion
                if 'metadata' not in bundle:
                    bundle['metadata'] = {}
                bundle['metadata']['expansion'] = 'tier3x'
                bundle['metadata']['merged_at'] = datetime.now().isoformat()

                seen_signatures.add(sig)
                expanded_bundles.append(bundle)
                stats['expanded_added'] += 1

                tier = bundle.get('metadata', {}).get('difficulty_tier', 'unknown')
                stats['tier_counts'][tier] += 1

    print(f"[OK] Loaded {stats['expanded_count']} expanded bundles")
    print(f"     Duplicates skipped: {stats['duplicates_skipped']}")
    print(f"     New bundles added: {stats['expanded_added']}")

    # Write merged output
    print(f"\nWriting merged bundles to {output_path.name}...")
    stats['final_count'] = len(original_bundles) + len(expanded_bundles)

    with open(output_path, 'w', encoding='utf-8') as f:
        for bundle in original_bundles:
            f.write(json.dumps(bundle, ensure_ascii=False) + '\n')
        for bundle in expanded_bundles:
            f.write(json.dumps(bundle, ensure_ascii=False) + '\n')

    print(f"[OK] Wrote {stats['final_count']} bundles")

    # Integrity check
    print("\n" + "=" * 60)
    print("INTEGRITY CHECK")
    print("=" * 60)
    print(f"Original bundles:     {stats['original_count']}")
    print(f"Expanded candidates:  {stats['expanded_count']}")
    print(f"Duplicates skipped:   {stats['duplicates_skipped']}")
    print(f"Expanded added:       {stats['expanded_added']}")
    print(f"Final bundle count:   {stats['final_count']}")
    print(f"\nTier distribution in merged file:")
    for tier, count in sorted(stats['tier_counts'].items()):
        print(f"  {tier}: {count}")

    # Verify tier3 count
    tier3_count = stats['tier_counts'].get('tier3_adversarial', 0)
    print(f"\nTier3 total: {tier3_count}")

    if tier3_count >= 2700:
        print("[OK] Tier3 count is in expected range (2700+)")
    else:
        print(f"[!] Warning: Tier3 count ({tier3_count}) lower than expected")

    # Save stats
    stats_path = output_path.parent / 'merge_tier3x_stats.json'
    with open(stats_path, 'w') as f:
        stats['tier_counts'] = dict(stats['tier_counts'])
        json.dump(stats, f, indent=2)
    print(f"\n[OK] Stats saved to {stats_path.name}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge tier3 expanded bundles")
    parser.add_argument('--original', required=True, help='Original contrastive bundles')
    parser.add_argument('--expanded', required=True, help='Expanded tier3 bundles')
    parser.add_argument('--out', required=True, help='Output merged file')
    args = parser.parse_args()

    merge_bundles(
        original_path=Path(args.original),
        expanded_path=Path(args.expanded),
        output_path=Path(args.out)
    )


if __name__ == '__main__':
    main()
