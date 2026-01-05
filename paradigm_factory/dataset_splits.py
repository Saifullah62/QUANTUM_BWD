#!/usr/bin/env python3
"""
Dataset Splitting Tools
=======================

Split polysemy bundles for training/validation/evaluation:

1. BY-WORD SPLIT (recommended for generalization testing)
   - Words are partitioned, not bundles
   - Model never sees test words during training
   - Tests true generalization to new vocabulary

2. BY-BUNDLE SPLIT (standard random split)
   - Bundles randomly shuffled and partitioned
   - Same word may appear in train and test
   - Tests in-distribution performance

Usage:
    # By-word split (recommended)
    python dataset_splits.py --input bundles.jsonl --by-word --train 0.7 --val 0.15 --eval 0.15

    # Standard random split
    python dataset_splits.py --input bundles.jsonl --train 0.8 --val 0.1 --eval 0.1

    # Stratified by difficulty
    python dataset_splits.py --input bundles.jsonl --stratify-difficulty
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.polysemy_bundle_v2 import PolysemyBundle, load_bundles, save_bundles


@dataclass
class SplitManifest:
    """Manifest documenting a dataset split."""
    created_at: str
    seed: int
    split_method: str  # "by_word" or "by_bundle"
    input_file: str
    total_bundles: int
    total_words: int

    train_bundles: int
    train_words: int
    train_file: str

    val_bundles: int
    val_words: int
    val_file: str

    eval_bundles: int
    eval_words: int
    eval_file: str

    train_words_list: List[str] = field(default_factory=list)
    val_words_list: List[str] = field(default_factory=list)
    eval_words_list: List[str] = field(default_factory=list)

    notes: str = ""


def split_by_word(
    bundles: List[PolysemyBundle],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    eval_ratio: float = 0.15,
    seed: int = 42,
    min_bundles_per_word: int = 3
) -> Tuple[List[PolysemyBundle], List[PolysemyBundle], List[PolysemyBundle], Dict]:
    """
    Split bundles by WORD, ensuring no word overlap between splits.

    This tests true generalization: the model must handle words it's never seen.

    Args:
        bundles: List of PolysemyBundle objects
        train_ratio: Fraction of words for training
        val_ratio: Fraction of words for validation
        eval_ratio: Fraction of words for evaluation
        seed: Random seed for reproducibility
        min_bundles_per_word: Words with fewer bundles go to training

    Returns:
        (train_bundles, val_bundles, eval_bundles, metadata)
    """
    random.seed(seed)

    # Group bundles by word
    word_to_bundles = defaultdict(list)
    for bundle in bundles:
        word_to_bundles[bundle.word.surface].append(bundle)

    # Separate words with enough data from those without
    words_with_data = []
    words_sparse = []
    for word, word_bundles in word_to_bundles.items():
        if len(word_bundles) >= min_bundles_per_word:
            words_with_data.append(word)
        else:
            words_sparse.append(word)

    # Shuffle words with enough data
    random.shuffle(words_with_data)

    # Calculate split points
    n_words = len(words_with_data)
    train_end = int(n_words * train_ratio)
    val_end = train_end + int(n_words * val_ratio)

    train_words = set(words_with_data[:train_end])
    val_words = set(words_with_data[train_end:val_end])
    eval_words = set(words_with_data[val_end:])

    # Sparse words go to training
    train_words.update(words_sparse)

    # Collect bundles
    train_bundles = []
    val_bundles = []
    eval_bundles = []

    for word, word_bundles in word_to_bundles.items():
        if word in train_words:
            train_bundles.extend(word_bundles)
        elif word in val_words:
            val_bundles.extend(word_bundles)
        elif word in eval_words:
            eval_bundles.extend(word_bundles)

    # Shuffle within each split
    random.shuffle(train_bundles)
    random.shuffle(val_bundles)
    random.shuffle(eval_bundles)

    metadata = {
        "method": "by_word",
        "train_words": sorted(list(train_words)),
        "val_words": sorted(list(val_words)),
        "eval_words": sorted(list(eval_words)),
        "sparse_words_in_train": sorted(words_sparse)
    }

    return train_bundles, val_bundles, eval_bundles, metadata


def split_by_bundle(
    bundles: List[PolysemyBundle],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    eval_ratio: float = 0.1,
    seed: int = 42,
    stratify_by_word: bool = False
) -> Tuple[List[PolysemyBundle], List[PolysemyBundle], List[PolysemyBundle], Dict]:
    """
    Standard random split by bundle.

    Same word may appear in multiple splits. Tests in-distribution performance.
    """
    random.seed(seed)

    if stratify_by_word:
        # Stratified: ensure proportional representation of each word
        word_to_bundles = defaultdict(list)
        for bundle in bundles:
            word_to_bundles[bundle.word.surface].append(bundle)

        train_bundles = []
        val_bundles = []
        eval_bundles = []

        for word, word_bundles in word_to_bundles.items():
            random.shuffle(word_bundles)
            n = len(word_bundles)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_bundles.extend(word_bundles[:train_end])
            val_bundles.extend(word_bundles[train_end:val_end])
            eval_bundles.extend(word_bundles[val_end:])

        random.shuffle(train_bundles)
        random.shuffle(val_bundles)
        random.shuffle(eval_bundles)

    else:
        # Pure random
        shuffled = bundles.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_bundles = shuffled[:train_end]
        val_bundles = shuffled[train_end:val_end]
        eval_bundles = shuffled[val_end:]

    metadata = {
        "method": "by_bundle",
        "stratified": stratify_by_word
    }

    return train_bundles, val_bundles, eval_bundles, metadata


def split_stratified_difficulty(
    bundles: List[PolysemyBundle],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    eval_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[PolysemyBundle], List[PolysemyBundle], List[PolysemyBundle], Dict]:
    """
    Split with stratification by difficulty level.

    Ensures each split has proportional easy/medium/hard bundles.
    """
    random.seed(seed)

    # Categorize bundles by average difficulty
    easy_bundles = []
    medium_bundles = []
    hard_bundles = []

    for bundle in bundles:
        non_anchor_items = [i for i in bundle.items if i.role != "anchor"]
        if not non_anchor_items:
            medium_bundles.append(bundle)
            continue

        avg_diff = sum(i.difficulty for i in non_anchor_items) / len(non_anchor_items)
        if avg_diff < 0.33:
            easy_bundles.append(bundle)
        elif avg_diff < 0.66:
            medium_bundles.append(bundle)
        else:
            hard_bundles.append(bundle)

    train_bundles = []
    val_bundles = []
    eval_bundles = []

    for difficulty_bundles in [easy_bundles, medium_bundles, hard_bundles]:
        random.shuffle(difficulty_bundles)
        n = len(difficulty_bundles)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_bundles.extend(difficulty_bundles[:train_end])
        val_bundles.extend(difficulty_bundles[train_end:val_end])
        eval_bundles.extend(difficulty_bundles[val_end:])

    random.shuffle(train_bundles)
    random.shuffle(val_bundles)
    random.shuffle(eval_bundles)

    metadata = {
        "method": "stratified_difficulty",
        "easy_count": len(easy_bundles),
        "medium_count": len(medium_bundles),
        "hard_count": len(hard_bundles)
    }

    return train_bundles, val_bundles, eval_bundles, metadata


def create_splits(
    input_path: Path,
    output_dir: Path,
    method: str = "by_word",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    eval_ratio: float = 0.15,
    seed: int = 42
) -> SplitManifest:
    """
    Create train/val/eval splits and save to output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load bundles
    bundles = load_bundles(input_path)
    print(f"Loaded {len(bundles)} bundles from {input_path}")

    # Get unique words
    all_words = set(b.word.surface for b in bundles)
    print(f"Unique words: {len(all_words)}")

    # Perform split
    if method == "by_word":
        train, val, eval_set, metadata = split_by_word(
            bundles, train_ratio, val_ratio, eval_ratio, seed
        )
    elif method == "stratified_difficulty":
        train, val, eval_set, metadata = split_stratified_difficulty(
            bundles, train_ratio, val_ratio, eval_ratio, seed
        )
    else:  # by_bundle
        train, val, eval_set, metadata = split_by_bundle(
            bundles, train_ratio, val_ratio, eval_ratio, seed
        )

    # Save splits
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    eval_path = output_dir / "eval_SEALED.jsonl"  # SEALED indicates don't peek

    save_bundles(train, train_path)
    save_bundles(val, val_path)
    save_bundles(eval_set, eval_path)

    # Create manifest
    manifest = SplitManifest(
        created_at=datetime.now().isoformat(),
        seed=seed,
        split_method=method,
        input_file=str(input_path),
        total_bundles=len(bundles),
        total_words=len(all_words),
        train_bundles=len(train),
        train_words=len(set(b.word.surface for b in train)),
        train_file=str(train_path),
        val_bundles=len(val),
        val_words=len(set(b.word.surface for b in val)),
        val_file=str(val_path),
        eval_bundles=len(eval_set),
        eval_words=len(set(b.word.surface for b in eval_set)),
        eval_file=str(eval_path),
        train_words_list=metadata.get("train_words", []),
        val_words_list=metadata.get("val_words", []),
        eval_words_list=metadata.get("eval_words", []),
        notes=f"Method: {method}, Ratios: {train_ratio}/{val_ratio}/{eval_ratio}"
    )

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "created_at": manifest.created_at,
            "seed": manifest.seed,
            "split_method": manifest.split_method,
            "input_file": manifest.input_file,
            "total_bundles": manifest.total_bundles,
            "total_words": manifest.total_words,
            "train": {
                "bundles": manifest.train_bundles,
                "words": manifest.train_words,
                "file": manifest.train_file,
                "word_list": manifest.train_words_list[:50]  # First 50 for reference
            },
            "val": {
                "bundles": manifest.val_bundles,
                "words": manifest.val_words,
                "file": manifest.val_file,
                "word_list": manifest.val_words_list
            },
            "eval": {
                "bundles": manifest.eval_bundles,
                "words": manifest.eval_words,
                "file": manifest.eval_file,
                "word_list": manifest.eval_words_list
            },
            "notes": manifest.notes
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  DATASET SPLITS CREATED ({method})")
    print(f"{'='*60}")
    print(f"\n  Split method: {method}")
    print(f"  Seed: {seed}")
    print(f"\n  Train: {manifest.train_bundles:5d} bundles, {manifest.train_words:3d} words")
    print(f"  Val:   {manifest.val_bundles:5d} bundles, {manifest.val_words:3d} words")
    print(f"  Eval:  {manifest.eval_bundles:5d} bundles, {manifest.eval_words:3d} words")

    if method == "by_word":
        print(f"\n  Word overlap check:")
        train_words_set = set(b.word.surface for b in train)
        val_words_set = set(b.word.surface for b in val)
        eval_words_set = set(b.word.surface for b in eval_set)

        train_val_overlap = train_words_set & val_words_set
        train_eval_overlap = train_words_set & eval_words_set
        val_eval_overlap = val_words_set & eval_words_set

        print(f"    Train & Val:  {len(train_val_overlap)} words {'OK' if len(train_val_overlap) == 0 else 'OVERLAP'}")
        print(f"    Train & Eval: {len(train_eval_overlap)} words {'OK' if len(train_eval_overlap) == 0 else 'OVERLAP'}")
        print(f"    Val & Eval:   {len(val_eval_overlap)} words {'OK' if len(val_eval_overlap) == 0 else 'OVERLAP'}")

    print(f"\n  Files saved to: {output_dir}")
    print(f"    - train.jsonl")
    print(f"    - val.jsonl")
    print(f"    - eval_SEALED.jsonl")
    print(f"    - manifest.json")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Dataset Splitting Tools")
    parser.add_argument("--input", type=str, required=True, help="Input bundles file (JSONL)")
    parser.add_argument("--output-dir", type=str, default="paradigm_factory/output/splits",
                        help="Output directory for splits")
    parser.add_argument("--by-word", action="store_true",
                        help="Split by word (no word overlap between splits)")
    parser.add_argument("--stratify-difficulty", action="store_true",
                        help="Stratify by difficulty level")
    parser.add_argument("--train", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--eval", type=float, default=0.15, help="Evaluation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train + args.val + args.eval
    if abs(total_ratio - 1.0) > 0.01:
        parser.error(f"Ratios must sum to 1.0, got {total_ratio}")

    # Determine method
    if args.by_word:
        method = "by_word"
    elif args.stratify_difficulty:
        method = "stratified_difficulty"
    else:
        method = "by_bundle"

    create_splits(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        method=method,
        train_ratio=args.train,
        val_ratio=args.val,
        eval_ratio=args.eval,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
