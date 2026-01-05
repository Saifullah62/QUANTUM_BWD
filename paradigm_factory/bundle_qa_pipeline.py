#!/usr/bin/env python3
"""
Bundle QA Pipeline: staging → verifier → train_ready
=====================================================

A quality assurance pipeline for polysemy bundles:

1. STAGING: Raw bundles from swarm generation land here
2. VERIFIER: Validates bundles, rejects weak ones with reasons
3. TRAIN_READY: Verified bundles ready for training

Usage:
    # Run full pipeline on a raw bundle file
    python bundle_qa_pipeline.py --input raw_bundles.jsonl --output-dir paradigm_factory/output/qa

    # Just verify bundles (no staging)
    python bundle_qa_pipeline.py --verify-only --input raw_bundles.jsonl

    # Show statistics on existing train_ready bundles
    python bundle_qa_pipeline.py --stats --train-ready-dir paradigm_factory/output/qa/train_ready
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.polysemy_bundle_v2 import (
    PolysemyBundle, load_bundles, save_bundles, compute_cue_difficulty
)
from paradigm_factory.dataset_characterization import (
    LEAKAGE_PATTERNS, audit_hard_negatives, audit_answer_leakage
)
import re


@dataclass
class VerificationResult:
    """Result of verifying a single bundle."""
    bundle_id: str
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 0.0  # 0-1, higher = better quality


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    total_input: int = 0
    staged: int = 0
    verified: int = 0
    rejected: int = 0
    train_ready: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    difficulty_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    words_covered: int = 0
    senses_covered: int = 0


class BundleVerifier:
    """Validates bundles against quality criteria."""

    def __init__(
        self,
        min_items: int = 3,
        require_anchor: bool = True,
        require_positive: bool = True,
        require_negative: bool = True,
        min_difficulty_spread: float = 0.1,
        max_duplicate_contexts: int = 0,
        min_context_length: int = 20,
        max_context_length: int = 300,
        check_leakage: bool = True,
        max_leakage_items: int = 1  # Max items with leakage patterns allowed
    ):
        self.min_items = min_items
        self.require_anchor = require_anchor
        self.require_positive = require_positive
        self.require_negative = require_negative
        self.min_difficulty_spread = min_difficulty_spread
        self.max_duplicate_contexts = max_duplicate_contexts
        self.min_context_length = min_context_length
        self.max_context_length = max_context_length
        self.check_leakage = check_leakage
        self.max_leakage_items = max_leakage_items

    def verify(self, bundle: PolysemyBundle) -> VerificationResult:
        """Run all checks on a bundle."""
        result = VerificationResult(
            bundle_id=bundle.bundle_id,
            passed=True,
            checks={},
            reasons=[],
            warnings=[]
        )

        # Check 1: Minimum items
        if len(bundle.items) < self.min_items:
            result.checks["min_items"] = False
            result.reasons.append(f"Too few items ({len(bundle.items)} < {self.min_items})")
            result.passed = False
        else:
            result.checks["min_items"] = True

        # Check 2: Has anchor
        has_anchor = any(i.role == "anchor" for i in bundle.items)
        if self.require_anchor and not has_anchor:
            result.checks["has_anchor"] = False
            result.reasons.append("Missing anchor item")
            result.passed = False
        else:
            result.checks["has_anchor"] = has_anchor

        # Check 3: Has positive
        has_positive = any(i.role == "positive" for i in bundle.items)
        if self.require_positive and not has_positive:
            result.checks["has_positive"] = False
            result.reasons.append("Missing positive item")
            result.passed = False
        else:
            result.checks["has_positive"] = has_positive

        # Check 4: Has negative
        has_negative = any(i.role in ("negative", "hard_negative") for i in bundle.items)
        if self.require_negative and not has_negative:
            result.checks["has_negative"] = False
            result.reasons.append("Missing negative/hard_negative item")
            result.passed = False
        else:
            result.checks["has_negative"] = has_negative

        # Check 5: Difficulty spread
        difficulties = [i.difficulty for i in bundle.items]
        diff_spread = max(difficulties) - min(difficulties) if difficulties else 0
        if diff_spread < self.min_difficulty_spread:
            result.checks["difficulty_spread"] = False
            result.reasons.append(f"Low difficulty spread ({diff_spread:.2f} < {self.min_difficulty_spread})")
            result.passed = False
        else:
            result.checks["difficulty_spread"] = True

        # Check 6: No duplicate contexts
        contexts = [i.context for i in bundle.items]
        unique_contexts = set(contexts)
        duplicates = len(contexts) - len(unique_contexts)
        if duplicates > self.max_duplicate_contexts:
            result.checks["no_duplicates"] = False
            result.reasons.append(f"Has {duplicates} duplicate context(s)")
            result.passed = False
        else:
            result.checks["no_duplicates"] = True

        # Check 7: Context length
        context_issues = []
        for item in bundle.items:
            ctx_len = len(item.context)
            if ctx_len < self.min_context_length:
                context_issues.append(f"'{item.context[:30]}...' too short ({ctx_len})")
            elif ctx_len > self.max_context_length:
                context_issues.append(f"'{item.context[:30]}...' too long ({ctx_len})")

        if context_issues:
            result.checks["context_length"] = False
            result.warnings.extend(context_issues)  # Warnings, not rejections
        else:
            result.checks["context_length"] = True

        # Check 8: Word appears in context
        word = bundle.word.surface.lower()
        missing_word = []
        for item in bundle.items:
            if word not in item.context.lower():
                missing_word.append(f"'{item.context[:30]}...' missing '{word}'")

        if missing_word:
            result.checks["word_in_context"] = False
            result.reasons.append(f"Word missing from {len(missing_word)} context(s)")
            result.passed = False
        else:
            result.checks["word_in_context"] = True

        # Check 9: Answer leakage detection
        if self.check_leakage:
            leakage_items = 0
            leakage_patterns_found = []
            for item in bundle.items:
                for pattern in LEAKAGE_PATTERNS:
                    if re.search(pattern, item.context, re.IGNORECASE):
                        leakage_items += 1
                        leakage_patterns_found.append(pattern[:20])
                        break  # Only count once per item

            if leakage_items > self.max_leakage_items:
                result.checks["no_leakage"] = False
                result.reasons.append(f"Answer leakage in {leakage_items} item(s)")
                result.passed = False
            elif leakage_items > 0:
                result.checks["no_leakage"] = True  # Pass but warn
                result.warnings.append(f"Potential leakage in {leakage_items} item(s)")
            else:
                result.checks["no_leakage"] = True

        # Calculate quality score (0-1)
        passed_checks = sum(1 for v in result.checks.values() if v)
        total_checks = len(result.checks)
        result.score = passed_checks / total_checks if total_checks > 0 else 0.0

        return result

    def verify_batch(self, bundles: List[PolysemyBundle]) -> Tuple[List[PolysemyBundle], List[Tuple[PolysemyBundle, VerificationResult]]]:
        """Verify a batch of bundles. Returns (passed, failed) lists."""
        passed = []
        failed = []

        for bundle in bundles:
            result = self.verify(bundle)
            if result.passed:
                passed.append(bundle)
            else:
                failed.append((bundle, result))

        return passed, failed


class BundleQAPipeline:
    """Orchestrates the staging → verifier → train_ready pipeline."""

    def __init__(self, output_dir: Path, verifier: BundleVerifier = None):
        self.output_dir = Path(output_dir)
        self.staging_dir = self.output_dir / "staging"
        self.verified_dir = self.output_dir / "verified"
        self.train_ready_dir = self.output_dir / "train_ready"
        self.rejected_dir = self.output_dir / "rejected"

        self.verifier = verifier or BundleVerifier()

        # Create directories
        for d in [self.staging_dir, self.verified_dir, self.train_ready_dir, self.rejected_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def stage(self, bundles: List[PolysemyBundle], batch_name: str = None) -> Path:
        """Stage bundles for verification."""
        if batch_name is None:
            batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        staged_path = self.staging_dir / f"{batch_name}.jsonl"
        save_bundles(bundles, staged_path)

        return staged_path

    def verify_staged(self, staged_path: Path) -> Tuple[Path, Path, PipelineStats]:
        """Verify staged bundles, output to verified and rejected."""
        bundles = load_bundles(staged_path)
        stats = PipelineStats(total_input=len(bundles), staged=len(bundles))

        passed, failed = self.verifier.verify_batch(bundles)

        # Save verified bundles
        batch_name = staged_path.stem
        verified_path = self.verified_dir / f"{batch_name}_verified.jsonl"
        save_bundles(passed, verified_path)
        stats.verified = len(passed)

        # Save rejected bundles with reasons
        rejected_path = self.rejected_dir / f"{batch_name}_rejected.json"
        rejected_data = []
        for bundle, result in failed:
            rejected_data.append({
                "bundle_id": bundle.bundle_id,
                "word": bundle.word.surface,
                "reasons": result.reasons,
                "warnings": result.warnings,
                "checks": result.checks,
                "score": result.score
            })
            for reason in result.reasons:
                # Extract reason category
                category = reason.split("(")[0].strip()
                stats.rejection_reasons[category] += 1

        stats.rejected = len(failed)

        with open(rejected_path, "w") as f:
            json.dump(rejected_data, f, indent=2)

        return verified_path, rejected_path, stats

    def promote_to_train_ready(self, verified_path: Path, split_by_word: bool = False) -> Tuple[Path, PipelineStats]:
        """Promote verified bundles to train_ready."""
        bundles = load_bundles(verified_path)
        stats = PipelineStats(verified=len(bundles))

        if split_by_word:
            # Group by word for later by-word splitting
            by_word = defaultdict(list)
            for bundle in bundles:
                by_word[bundle.word.surface].append(bundle)

            for word, word_bundles in by_word.items():
                word_path = self.train_ready_dir / f"{word}_bundles.jsonl"
                save_bundles(word_bundles, word_path)

            stats.words_covered = len(by_word)
        else:
            # Single file
            batch_name = verified_path.stem.replace("_verified", "")
            train_path = self.train_ready_dir / f"{batch_name}_train_ready.jsonl"
            save_bundles(bundles, train_path)

        stats.train_ready = len(bundles)

        # Collect difficulty stats
        for bundle in bundles:
            for item in bundle.items:
                if item.difficulty < 0.33:
                    stats.difficulty_distribution["easy"] += 1
                elif item.difficulty < 0.66:
                    stats.difficulty_distribution["medium"] += 1
                else:
                    stats.difficulty_distribution["hard"] += 1

        # Count unique senses
        senses = set()
        for bundle in bundles:
            for item in bundle.items:
                senses.add(item.sense_id)
        stats.senses_covered = len(senses)
        stats.words_covered = len(set(b.word.surface for b in bundles))

        return self.train_ready_dir, stats

    def run_full_pipeline(self, input_path: Path, batch_name: str = None) -> PipelineStats:
        """Run the complete pipeline on input bundles."""
        print(f"\n{'='*60}")
        print("  BUNDLE QA PIPELINE")
        print(f"{'='*60}")

        # Load input
        bundles = load_bundles(input_path)
        print(f"\nInput: {len(bundles)} bundles from {input_path}")

        # Stage
        print("\n[1/3] STAGING...")
        staged_path = self.stage(bundles, batch_name)
        print(f"  Staged to: {staged_path}")

        # Verify
        print("\n[2/3] VERIFYING...")
        verified_path, rejected_path, verify_stats = self.verify_staged(staged_path)
        print(f"  Verified: {verify_stats.verified}")
        print(f"  Rejected: {verify_stats.rejected}")
        if verify_stats.rejection_reasons:
            print("  Rejection breakdown:")
            for reason, count in sorted(verify_stats.rejection_reasons.items(), key=lambda x: -x[1]):
                print(f"    - {reason}: {count}")

        # Promote
        print("\n[3/3] PROMOTING TO TRAIN_READY...")
        train_dir, promote_stats = self.promote_to_train_ready(verified_path)
        print(f"  Train-ready: {promote_stats.train_ready} bundles")
        print(f"  Words covered: {promote_stats.words_covered}")
        print(f"  Senses covered: {promote_stats.senses_covered}")

        # Final stats
        final_stats = PipelineStats(
            total_input=len(bundles),
            staged=len(bundles),
            verified=verify_stats.verified,
            rejected=verify_stats.rejected,
            train_ready=promote_stats.train_ready,
            rejection_reasons=verify_stats.rejection_reasons,
            difficulty_distribution=promote_stats.difficulty_distribution,
            words_covered=promote_stats.words_covered,
            senses_covered=promote_stats.senses_covered
        )

        print(f"\n{'='*60}")
        print("  PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"  {final_stats.total_input} -> {final_stats.verified} verified -> {final_stats.train_ready} train-ready")
        if final_stats.difficulty_distribution:
            total_items = sum(final_stats.difficulty_distribution.values())
            print(f"\n  Difficulty distribution ({total_items} items):")
            for diff, count in sorted(final_stats.difficulty_distribution.items()):
                pct = 100 * count / total_items if total_items > 0 else 0
                print(f"    {diff}: {count} ({pct:.1f}%)")

        return final_stats


def show_train_ready_stats(train_ready_dir: Path):
    """Show statistics on train_ready bundles."""
    train_ready_dir = Path(train_ready_dir)

    if not train_ready_dir.exists():
        print(f"Directory not found: {train_ready_dir}")
        return

    all_bundles = []
    for jsonl_file in train_ready_dir.glob("*.jsonl"):
        all_bundles.extend(load_bundles(jsonl_file))

    if not all_bundles:
        print("No bundles found in train_ready directory")
        return

    print(f"\n{'='*60}")
    print("  TRAIN-READY DATASET STATS")
    print(f"{'='*60}")
    print(f"\nTotal bundles: {len(all_bundles)}")

    # Words
    words = defaultdict(int)
    for b in all_bundles:
        words[b.word.surface] += 1
    print(f"Unique words: {len(words)}")

    # Senses
    senses = set()
    for b in all_bundles:
        for item in b.items:
            senses.add(item.sense_id)
    print(f"Unique senses: {len(senses)}")

    # Items and roles
    role_counts = defaultdict(int)
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    total_items = 0

    for b in all_bundles:
        for item in b.items:
            total_items += 1
            role_counts[item.role] += 1
            if item.difficulty < 0.33:
                difficulty_counts["easy"] += 1
            elif item.difficulty < 0.66:
                difficulty_counts["medium"] += 1
            else:
                difficulty_counts["hard"] += 1

    print(f"\nTotal items: {total_items}")
    print(f"Avg items per bundle: {total_items / len(all_bundles):.1f}")

    print("\nRole distribution:")
    for role, count in sorted(role_counts.items()):
        pct = 100 * count / total_items
        print(f"  {role}: {count} ({pct:.1f}%)")

    print("\nDifficulty distribution:")
    for diff, count in sorted(difficulty_counts.items()):
        pct = 100 * count / total_items
        print(f"  {diff}: {count} ({pct:.1f}%)")

    # Top words
    print(f"\nTop 10 words by bundle count:")
    for word, count in sorted(words.items(), key=lambda x: -x[1])[:10]:
        print(f"  {word}: {count} bundles")


def main():
    parser = argparse.ArgumentParser(description="Bundle QA Pipeline")
    parser.add_argument("--input", type=str, help="Input bundles file (JSONL)")
    parser.add_argument("--output-dir", type=str, default="paradigm_factory/output/qa",
                        help="Output directory for pipeline")
    parser.add_argument("--batch-name", type=str, help="Name for this batch")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run verification, don't stage or promote")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats on existing train_ready bundles")
    parser.add_argument("--train-ready-dir", type=str,
                        help="Train-ready directory for --stats")

    args = parser.parse_args()

    if args.stats:
        train_dir = args.train_ready_dir or f"{args.output_dir}/train_ready"
        show_train_ready_stats(Path(train_dir))
        return

    if not args.input:
        parser.error("--input is required unless using --stats")

    pipeline = BundleQAPipeline(Path(args.output_dir))

    if args.verify_only:
        # Just verify, don't stage/promote
        bundles = load_bundles(Path(args.input))
        verifier = BundleVerifier()
        passed, failed = verifier.verify_batch(bundles)

        print(f"\nVerification Results:")
        print(f"  Passed: {len(passed)}")
        print(f"  Failed: {len(failed)}")

        if failed:
            print("\nRejection reasons:")
            reasons = defaultdict(int)
            for _, result in failed:
                for reason in result.reasons:
                    reasons[reason.split("(")[0].strip()] += 1
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"  - {reason}: {count}")
    else:
        # Full pipeline
        stats = pipeline.run_full_pipeline(
            Path(args.input),
            batch_name=args.batch_name
        )


if __name__ == "__main__":
    main()
