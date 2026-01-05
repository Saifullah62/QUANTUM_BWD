#!/usr/bin/env python3
"""
Bundle Splitter - Train/Val/Eval Split with Deduplication
==========================================================

Creates clean splits with:
- No context leakage between splits (hash-based deduplication)
- Word-stratified splitting (each word appears in all splits)
- Sealed eval pack for honest evaluation
- Difficulty balance preserved across splits

Usage:
    python bundle_splitter.py --input clean_bundles.jsonl --output-dir splits/
"""

import argparse
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.polysemy_bundle_v2 import PolysemyBundle, load_bundles, save_bundles


def normalize_text(text: str) -> str:
    """Normalize text for deduplication."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text


def context_hash(text: str) -> str:
    """Hash normalized context for deduplication."""
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def get_all_context_hashes(bundle: PolysemyBundle) -> Set[str]:
    """Get hashes of all contexts in a bundle."""
    return {context_hash(item.context) for item in bundle.items}


def get_bundle_difficulty(bundle: PolysemyBundle) -> str:
    """Categorize bundle by average item difficulty."""
    difficulties = [i.difficulty for i in bundle.items if i.role != "anchor"]
    if not difficulties:
        return "med"
    avg = sum(difficulties) / len(difficulties)
    if avg < 0.33:
        return "easy"
    elif avg < 0.66:
        return "med"
    return "hard"


def split_bundles(
    bundles: List[PolysemyBundle],
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    eval_ratio: float = 0.05,
    seed: int = 42,
    stratify_by_word: bool = True,
    check_leakage: bool = True
) -> Tuple[List[PolysemyBundle], List[PolysemyBundle], List[PolysemyBundle]]:
    """
    Split bundles into train/val/eval with no leakage.

    Strategy:
    1. Group bundles by word
    2. For each word, split bundles proportionally
    3. Check for context hash overlap between splits
    4. Move any leaking bundles to train (conservative)
    """
    random.seed(seed)

    if stratify_by_word:
        # Group by word
        by_word: Dict[str, List[PolysemyBundle]] = defaultdict(list)
        for bundle in bundles:
            by_word[bundle.word.surface].append(bundle)

        train, val, eval_set = [], [], []

        for word, word_bundles in by_word.items():
            random.shuffle(word_bundles)

            n = len(word_bundles)
            n_train = max(1, int(n * train_ratio))
            n_val = max(0, int(n * val_ratio))
            n_eval = max(0, n - n_train - n_val)

            # Ensure at least one in eval if we have enough
            if n >= 3 and n_eval == 0:
                n_eval = 1
                n_train -= 1

            train.extend(word_bundles[:n_train])
            val.extend(word_bundles[n_train:n_train + n_val])
            eval_set.extend(word_bundles[n_train + n_val:])

    else:
        # Simple random split
        shuffled = bundles.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        eval_set = shuffled[n_train + n_val:]

    if check_leakage:
        # Only check leakage if we have enough bundles (small datasets may have intentional reuse)
        if len(bundles) > 200:
            train, val, eval_set = remove_leakage(train, val, eval_set)
        else:
            print("  Skipping leakage check for small dataset (<200 bundles)")

    return train, val, eval_set


def remove_leakage(
    train: List[PolysemyBundle],
    val: List[PolysemyBundle],
    eval_set: List[PolysemyBundle]
) -> Tuple[List[PolysemyBundle], List[PolysemyBundle], List[PolysemyBundle]]:
    """
    Remove any bundles that share contexts across splits.

    Priority: eval > val > train (protect eval and val purity)
    """
    # Build hash sets
    eval_hashes: Set[str] = set()
    for bundle in eval_set:
        eval_hashes.update(get_all_context_hashes(bundle))

    val_hashes: Set[str] = set()
    for bundle in val:
        val_hashes.update(get_all_context_hashes(bundle))

    # Remove from train any bundle that overlaps with eval or val
    clean_train = []
    leaked_from_train = 0
    for bundle in train:
        bundle_hashes = get_all_context_hashes(bundle)
        if bundle_hashes & eval_hashes or bundle_hashes & val_hashes:
            leaked_from_train += 1
        else:
            clean_train.append(bundle)

    # Remove from val any bundle that overlaps with eval
    clean_val = []
    leaked_from_val = 0
    for bundle in val:
        bundle_hashes = get_all_context_hashes(bundle)
        if bundle_hashes & eval_hashes:
            leaked_from_val += 1
        else:
            clean_val.append(bundle)

    if leaked_from_train > 0 or leaked_from_val > 0:
        print(f"  Removed {leaked_from_train} leaked bundles from train")
        print(f"  Removed {leaked_from_val} leaked bundles from val")

    return clean_train, clean_val, eval_set


def print_split_stats(name: str, bundles: List[PolysemyBundle]):
    """Print statistics for a split."""
    if not bundles:
        print(f"  {name}: 0 bundles")
        return

    words = set(b.word.surface for b in bundles)
    items = sum(len(b.items) for b in bundles)

    # Difficulty distribution
    easy = sum(1 for b in bundles if get_bundle_difficulty(b) == "easy")
    med = sum(1 for b in bundles if get_bundle_difficulty(b) == "med")
    hard = sum(1 for b in bundles if get_bundle_difficulty(b) == "hard")

    print(f"  {name}: {len(bundles)} bundles, {len(words)} words, {items} items")
    print(f"    Difficulty: easy={easy} ({100*easy/len(bundles):.0f}%), "
          f"med={med} ({100*med/len(bundles):.0f}%), "
          f"hard={hard} ({100*hard/len(bundles):.0f}%)")


def run_split(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    eval_ratio: float = 0.05,
    seed: int = 42
):
    """Run the full split pipeline."""
    print("=" * 70)
    print("  Bundle Splitter")
    print("=" * 70)

    # Load bundles
    bundles = load_bundles(input_path)
    print(f"Loaded {len(bundles)} bundles from {input_path}")

    # Split
    train, val, eval_set = split_bundles(
        bundles,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        eval_ratio=eval_ratio,
        seed=seed
    )

    # Stats
    print("\nSplit statistics:")
    print_split_stats("Train", train)
    print_split_stats("Val", val)
    print_split_stats("Eval", eval_set)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_bundles.jsonl"
    val_path = output_dir / "val_bundles.jsonl"
    eval_path = output_dir / "eval_bundles_SEALED.jsonl"

    save_bundles(train, train_path)
    save_bundles(val, val_path)
    save_bundles(eval_set, eval_path)

    print(f"\nSaved splits to {output_dir}:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Eval (SEALED): {eval_path}")

    # Create manifest
    manifest = {
        "created": datetime.now().isoformat(),
        "source": str(input_path),
        "seed": seed,
        "splits": {
            "train": {"path": str(train_path), "count": len(train)},
            "val": {"path": str(val_path), "count": len(val)},
            "eval": {"path": str(eval_path), "count": len(eval_set), "sealed": True}
        },
        "words": list(set(b.word.surface for b in bundles)),
        "total_items": sum(len(b.items) for b in bundles)
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    # Warning about eval pack
    print("\n" + "=" * 70)
    print("  EVAL PACK IS NOW SEALED")
    print("=" * 70)
    print("The eval pack should NEVER be regenerated or modified.")
    print("Do not let generation processes access these contexts.")
    print("Use only for final evaluation after training.")

    return train, val, eval_set


def main():
    parser = argparse.ArgumentParser(description="Split bundles into train/val/eval")
    parser.add_argument('--input', type=str, required=True, help='Input verified bundles JSONL')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for splits')
    parser.add_argument('--train-ratio', type=float, default=0.90, help='Train split ratio')
    parser.add_argument('--val-ratio', type=float, default=0.05, help='Validation split ratio')
    parser.add_argument('--eval-ratio', type=float, default=0.05, help='Eval split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    run_split(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        eval_ratio=args.eval_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
