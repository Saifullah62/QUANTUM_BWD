#!/usr/bin/env python3
"""
Build Frozen Eval Set from Bundles
==================================

Creates a deterministic eval set from a bundles file for post-hoc evaluation.
The output is a frozen artifact that can be used to score all capsules against
the same eval items.

Usage:
    python experiments/build_eval_set_from_bundles.py \
        --in paradigm_factory/v2/bundles_v23/contrastive_bundles.jsonl \
        --out evals/eval_v23_contrastive_5k.jsonl \
        --max-items 5000 --seed 42
"""

import argparse
import json
import hashlib
import random
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_bundle_item(b: dict) -> dict | None:
    """
    Extract eval item from bundle, handling v2.2 and v2.3 formats.

    Returns None if bundle is malformed or missing required fields.
    """
    # Get tier from various locations
    metadata = b.get("metadata", {}) if isinstance(b.get("metadata"), dict) else {}
    meta = b.get("meta", {}) if isinstance(b.get("meta"), dict) else {}
    tier = (
        b.get("tier") or
        metadata.get("difficulty_tier") or
        metadata.get("tier") or
        meta.get("tier") or
        meta.get("difficulty_tier")
    )

    # Get anchor/positive (v2.3 format)
    anchor = b.get("anchor") or b.get("anchor_item")
    pos = b.get("positive") or b.get("positive_item")

    if not anchor or not pos:
        return None

    # Get texts
    a_text = anchor.get("text") if isinstance(anchor, dict) else None
    p_text = pos.get("text") if isinstance(pos, dict) else None

    if not a_text or not p_text:
        return None

    # Get negatives - handle both dict and list formats
    negs = b.get("negatives")
    n_texts = []

    if isinstance(negs, dict):
        # v2.3 format: {within_lemma: [...], cross_lemma: [...]}
        for key in ("within_lemma", "cross_lemma", "sibling_negative", "crosslemma_negative"):
            items = negs.get(key, [])
            if isinstance(items, list):
                for n in items:
                    if isinstance(n, dict) and n.get("text"):
                        n_texts.append(n["text"])
    elif isinstance(negs, list):
        # Simple list format
        for n in negs:
            if isinstance(n, dict) and n.get("text"):
                n_texts.append(n["text"])

    # Also check for standalone negative fields
    for key in ("sibling_negative", "sibling_neg", "crosslemma_negative", "cross_neg"):
        neg_item = b.get(key)
        if isinstance(neg_item, dict) and neg_item.get("text"):
            n_texts.append(neg_item["text"])

    if not n_texts:
        return None

    # Get danger score if available
    danger = (
        b.get("danger") or
        b.get("danger_score") or
        metadata.get("danger_score") or
        metadata.get("danger") or
        meta.get("danger_score") or
        meta.get("danger")
    )

    return {
        "id": b.get("id") or b.get("bundle_id") or b.get("anchor", {}).get("sense_id", "unknown"),
        "tier": tier,
        "lemma": b.get("lemma") or anchor.get("lemma"),
        "pos": b.get("pos") or anchor.get("pos"),
        "danger": danger,
        "anchor_text": a_text,
        "positive_text": p_text,
        "negative_texts": n_texts,
        "sense_id": anchor.get("sense_id"),
    }


def main():
    parser = argparse.ArgumentParser(description="Build frozen eval set from bundles")
    parser.add_argument("--in", dest="inp", required=True, help="Input bundles jsonl")
    parser.add_argument("--out", dest="out", required=True, help="Output eval jsonl")
    parser.add_argument("--max-items", type=int, default=5000, help="Cap eval size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--require-tier", action="store_true", help="Drop items missing tier")
    parser.add_argument("--tier-stratified", action="store_true",
                       help="Sample equally from each tier (up to max-items total)")
    args = parser.parse_args()

    random.seed(args.seed)

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load and parse bundles
    items = []
    skipped = 0

    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            item = extract_bundle_item(b)
            if item is None:
                skipped += 1
                continue

            if args.require_tier and item["tier"] is None:
                skipped += 1
                continue

            items.append(item)

    print(f"Loaded {len(items)} valid items from {inp} (skipped {skipped})")

    # Stratified sampling if requested
    if args.tier_stratified:
        by_tier = {}
        for it in items:
            tier = str(it.get("tier", "NA"))
            if tier not in by_tier:
                by_tier[tier] = []
            by_tier[tier].append(it)

        # Sample equally from each tier
        n_tiers = len(by_tier)
        per_tier = args.max_items // n_tiers if n_tiers > 0 else args.max_items

        sampled = []
        for tier, tier_items in sorted(by_tier.items()):
            n = min(per_tier, len(tier_items))
            sampled.extend(random.sample(tier_items, n))
            print(f"  {tier}: {n}/{len(tier_items)} items")

        items = sampled
        random.shuffle(items)
    else:
        # Simple random sample
        if len(items) > args.max_items:
            items = random.sample(items, args.max_items)

    # Count tiers in final set
    tier_counts = {}
    for it in items:
        tier = str(it.get("tier", "NA"))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    # Compute fingerprints
    in_hash = sha256_file(inp)
    content_hash = hashlib.sha256(
        json.dumps(items[:200], sort_keys=True).encode("utf-8")
    ).hexdigest()

    # Write output
    with out.open("w", encoding="utf-8") as f:
        # Header record
        header = {
            "_header": True,
            "source_bundles": str(inp),
            "source_sha256": in_hash,
            "eval_seed": args.seed,
            "eval_count": len(items),
            "tier_counts": tier_counts,
            "content_hash": content_hash[:16],
        }
        f.write(json.dumps(header) + "\n")

        # Items
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\nWrote eval set: {out}")
    print(f"  Items: {len(items)}")
    print(f"  Tiers: {tier_counts}")
    print(f"  Source SHA256: {in_hash[:16]}...")
    print(f"  Content hash: {content_hash[:16]}")


if __name__ == "__main__":
    main()
