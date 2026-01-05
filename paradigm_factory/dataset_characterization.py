#!/usr/bin/env python3
"""
Dataset Characterization Tools
==============================

Comprehensive analysis of polysemy bundle datasets:
- Word/sense coverage analysis
- Difficulty distribution
- Contrastive geometry quality
- Minimal pair statistics
- Balance and diversity metrics

Usage:
    python dataset_characterization.py --input bundles.jsonl
    python dataset_characterization.py --input bundles.jsonl --output-report report.json
    python dataset_characterization.py --compare file1.jsonl file2.jsonl
"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.polysemy_bundle_v2 import PolysemyBundle, load_bundles

# Common leakage patterns to detect
LEAKAGE_PATTERNS = [
    # Explicit sense definitions in context
    r'\b(meaning|sense|definition)\s+(of|is|:)',
    r'\bin\s+the\s+sense\s+of\b',
    r'\bas\s+in\s+"[^"]+"\b',
    # Parenthetical explanations
    r'\([^)]*sense[^)]*\)',
    r'\([^)]*meaning[^)]*\)',
    # Quotes around the target word (often indicates meta-usage)
    r'"[^"]{1,20}"',  # Short quoted phrases
    # Dictionary-style definitions
    r'\b(noun|verb|adjective)\s*[:\-]\s*',
    # Example markers
    r'\b(e\.g\.|i\.e\.|for\s+example|such\s+as)\b',
]


@dataclass
class HardNegativeAudit:
    """Audit results for hard negative quality."""
    total_bundles_audited: int = 0
    bundles_with_valid_hard_negs: int = 0
    bundles_with_invalid_hard_negs: int = 0  # Hard neg NOT closer than easy neg

    # Similarity statistics (text-based proxy)
    avg_hard_neg_similarity: float = 0.0  # To anchor
    avg_easy_neg_similarity: float = 0.0  # To anchor
    hard_easier_than_easy_count: int = 0  # Mislabeled count

    # Flagged bundles
    flagged_bundle_ids: List[str] = field(default_factory=list)

    passed: bool = True  # Overall audit pass/fail
    issues: List[str] = field(default_factory=list)


@dataclass
class LeakageAudit:
    """Audit results for answer leakage detection."""
    total_items_scanned: int = 0
    items_with_leakage: int = 0
    leakage_rate: float = 0.0

    # Types of leakage found
    leakage_by_type: Dict[str, int] = field(default_factory=dict)

    # Repeated cue phrases (same phrase appears in multiple contexts for same sense)
    repeated_cue_phrases: List[Tuple[str, int]] = field(default_factory=list)

    # Flagged items
    flagged_items: List[Dict] = field(default_factory=list)

    passed: bool = True
    issues: List[str] = field(default_factory=list)


@dataclass
class WordAnalysis:
    """Analysis for a single word."""
    word: str
    bundle_count: int = 0
    sense_count: int = 0
    senses: List[str] = field(default_factory=list)
    item_count: int = 0
    avg_items_per_bundle: float = 0.0
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    role_distribution: Dict[str, int] = field(default_factory=dict)
    has_hard_negatives: bool = False
    hard_negative_count: int = 0


@dataclass
class DifficultyAnalysis:
    """Difficulty distribution analysis."""
    easy_count: int = 0  # < 0.33
    medium_count: int = 0  # 0.33-0.66
    hard_count: int = 0  # >= 0.66
    total: int = 0
    easy_pct: float = 0.0
    medium_pct: float = 0.0
    hard_pct: float = 0.0
    balance_score: float = 0.0  # 1.0 = perfectly balanced (33/33/33)


@dataclass
class ContrastiveGeometryAnalysis:
    """Analysis of contrastive structure."""
    total_bundles: int = 0
    bundles_with_positives: int = 0
    bundles_with_negatives: int = 0
    bundles_with_hard_negatives: int = 0
    avg_positives_per_bundle: float = 0.0
    avg_negatives_per_bundle: float = 0.0
    avg_hard_negatives_per_bundle: float = 0.0
    complete_bundles: int = 0  # Has anchor + positive + negative + hard_negative


@dataclass
class DatasetReport:
    """Complete dataset characterization report."""
    # Basic stats
    total_bundles: int = 0
    total_items: int = 0
    unique_words: int = 0
    unique_senses: int = 0
    avg_bundles_per_word: float = 0.0
    avg_items_per_bundle: float = 0.0

    # Coverage
    word_coverage: Dict[str, int] = field(default_factory=dict)  # word -> bundle count
    sense_coverage: Dict[str, int] = field(default_factory=dict)  # sense -> item count

    # Difficulty
    difficulty: DifficultyAnalysis = field(default_factory=DifficultyAnalysis)

    # Contrastive geometry
    geometry: ContrastiveGeometryAnalysis = field(default_factory=ContrastiveGeometryAnalysis)

    # Role distribution
    role_distribution: Dict[str, int] = field(default_factory=dict)

    # Per-word analysis
    word_analyses: List[WordAnalysis] = field(default_factory=list)

    # Audits
    hard_negative_audit: Optional[HardNegativeAudit] = None
    leakage_audit: Optional[LeakageAudit] = None

    # Quality metrics
    quality_score: float = 0.0  # 0-1, composite quality metric
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def analyze_difficulty(bundles: List[PolysemyBundle]) -> DifficultyAnalysis:
    """Analyze difficulty distribution across all items."""
    analysis = DifficultyAnalysis()

    for bundle in bundles:
        for item in bundle.items:
            if item.role == "anchor":
                continue  # Skip anchors for difficulty analysis
            analysis.total += 1
            if item.difficulty < 0.33:
                analysis.easy_count += 1
            elif item.difficulty < 0.66:
                analysis.medium_count += 1
            else:
                analysis.hard_count += 1

    if analysis.total > 0:
        analysis.easy_pct = 100.0 * analysis.easy_count / analysis.total
        analysis.medium_pct = 100.0 * analysis.medium_count / analysis.total
        analysis.hard_pct = 100.0 * analysis.hard_count / analysis.total

        # Balance score: how close to 33/33/33
        target = analysis.total / 3
        deviations = [
            abs(analysis.easy_count - target),
            abs(analysis.medium_count - target),
            abs(analysis.hard_count - target)
        ]
        max_deviation = analysis.total  # Worst case: all in one bucket
        actual_deviation = sum(deviations)
        analysis.balance_score = 1.0 - (actual_deviation / max_deviation) if max_deviation > 0 else 0.0

    return analysis


def analyze_contrastive_geometry(bundles: List[PolysemyBundle]) -> ContrastiveGeometryAnalysis:
    """Analyze contrastive structure of bundles."""
    analysis = ContrastiveGeometryAnalysis(total_bundles=len(bundles))

    total_positives = 0
    total_negatives = 0
    total_hard_negatives = 0

    for bundle in bundles:
        has_pos = False
        has_neg = False
        has_hard_neg = False

        for item in bundle.items:
            if item.role == "positive":
                total_positives += 1
                has_pos = True
            elif item.role == "negative":
                total_negatives += 1
                has_neg = True
            elif item.role == "hard_negative":
                total_hard_negatives += 1
                has_hard_neg = True

        if has_pos:
            analysis.bundles_with_positives += 1
        if has_neg:
            analysis.bundles_with_negatives += 1
        if has_hard_neg:
            analysis.bundles_with_hard_negatives += 1
        if has_pos and (has_neg or has_hard_neg):
            analysis.complete_bundles += 1

    if len(bundles) > 0:
        analysis.avg_positives_per_bundle = total_positives / len(bundles)
        analysis.avg_negatives_per_bundle = total_negatives / len(bundles)
        analysis.avg_hard_negatives_per_bundle = total_hard_negatives / len(bundles)

    return analysis


def analyze_word(word: str, bundles: List[PolysemyBundle]) -> WordAnalysis:
    """Analyze bundles for a single word."""
    analysis = WordAnalysis(word=word, bundle_count=len(bundles))

    senses = set()
    total_items = 0
    difficulty_dist = {"easy": 0, "medium": 0, "hard": 0}
    role_dist = defaultdict(int)

    for bundle in bundles:
        for item in bundle.items:
            total_items += 1
            senses.add(item.sense_id)
            role_dist[item.role] += 1

            if item.role != "anchor":
                if item.difficulty < 0.33:
                    difficulty_dist["easy"] += 1
                elif item.difficulty < 0.66:
                    difficulty_dist["medium"] += 1
                else:
                    difficulty_dist["hard"] += 1

            if item.role == "hard_negative":
                analysis.has_hard_negatives = True
                analysis.hard_negative_count += 1

    analysis.sense_count = len(senses)
    analysis.senses = sorted(list(senses))
    analysis.item_count = total_items
    analysis.avg_items_per_bundle = total_items / len(bundles) if bundles else 0
    analysis.difficulty_distribution = difficulty_dist
    analysis.role_distribution = dict(role_dist)

    return analysis


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute simple text similarity as proxy for embedding similarity.
    Uses Jaccard similarity on word n-grams.
    """
    def get_ngrams(text: str, n: int = 2) -> set:
        words = text.lower().split()
        if len(words) < n:
            return set(words)
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    ngrams1 = get_ngrams(text1)
    ngrams2 = get_ngrams(text2)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


def audit_hard_negatives(bundles: List[PolysemyBundle]) -> HardNegativeAudit:
    """
    Audit that hard negatives are actually harder than easy negatives.

    A hard negative should be MORE similar to the anchor (textually/structurally)
    than an easy negative, because it's designed to be confusable.
    """
    import re

    audit = HardNegativeAudit()

    hard_neg_sims = []
    easy_neg_sims = []

    for bundle in bundles:
        # Find anchor
        anchor_item = None
        hard_negs = []
        easy_negs = []

        for item in bundle.items:
            if item.role == "anchor":
                anchor_item = item
            elif item.role == "hard_negative":
                hard_negs.append(item)
            elif item.role == "negative" and item.difficulty < 0.33:
                easy_negs.append(item)

        if not anchor_item or not hard_negs:
            continue

        audit.total_bundles_audited += 1
        anchor_text = anchor_item.context

        # Compute similarities
        bundle_hard_sims = [compute_text_similarity(anchor_text, hn.context) for hn in hard_negs]
        bundle_easy_sims = [compute_text_similarity(anchor_text, en.context) for en in easy_negs] if easy_negs else []

        hard_neg_sims.extend(bundle_hard_sims)
        easy_neg_sims.extend(bundle_easy_sims)

        # Check if hard negatives are actually harder (more similar to anchor)
        if bundle_hard_sims and bundle_easy_sims:
            avg_hard = sum(bundle_hard_sims) / len(bundle_hard_sims)
            avg_easy = sum(bundle_easy_sims) / len(bundle_easy_sims)

            if avg_hard < avg_easy:
                # Hard negative is LESS similar than easy negative - mislabeled!
                audit.bundles_with_invalid_hard_negs += 1
                audit.hard_easier_than_easy_count += 1
                audit.flagged_bundle_ids.append(bundle.bundle_id)
            else:
                audit.bundles_with_valid_hard_negs += 1

    # Compute averages
    if hard_neg_sims:
        audit.avg_hard_neg_similarity = sum(hard_neg_sims) / len(hard_neg_sims)
    if easy_neg_sims:
        audit.avg_easy_neg_similarity = sum(easy_neg_sims) / len(easy_neg_sims)

    # Determine pass/fail
    if audit.total_bundles_audited > 0:
        invalid_rate = audit.bundles_with_invalid_hard_negs / audit.total_bundles_audited
        if invalid_rate > 0.2:  # More than 20% mislabeled
            audit.passed = False
            audit.issues.append(f"{invalid_rate*100:.1f}% of hard negatives are easier than easy negatives")

        if audit.avg_hard_neg_similarity <= audit.avg_easy_neg_similarity:
            audit.passed = False
            audit.issues.append(
                f"Avg hard_neg similarity ({audit.avg_hard_neg_similarity:.3f}) <= "
                f"avg easy_neg similarity ({audit.avg_easy_neg_similarity:.3f})"
            )

    return audit


def audit_answer_leakage(bundles: List[PolysemyBundle]) -> LeakageAudit:
    """
    Scan for answer leakage patterns in contexts.

    Detects:
    - Explicit sense definitions
    - Repeated cue phrases across contexts
    - Giveaway patterns that let the model shortcut
    """
    import re

    audit = LeakageAudit()

    # Track cue phrases per sense
    sense_cue_phrases: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for bundle in bundles:
        word = bundle.word.surface.lower()

        for item in bundle.items:
            audit.total_items_scanned += 1
            context = item.context

            leakage_found = []

            # Check each leakage pattern
            for i, pattern in enumerate(LEAKAGE_PATTERNS):
                matches = re.findall(pattern, context, re.IGNORECASE)
                if matches:
                    pattern_name = f"pattern_{i}"
                    leakage_found.append(pattern_name)
                    audit.leakage_by_type[pattern_name] = audit.leakage_by_type.get(pattern_name, 0) + 1

            # Extract potential cue phrases (words near the target word)
            # This detects if the same disambiguating phrase appears repeatedly
            words = context.lower().split()
            try:
                word_idx = next(i for i, w in enumerate(words) if word in w)
                # Get 2-word window around target
                window_start = max(0, word_idx - 2)
                window_end = min(len(words), word_idx + 3)
                window_phrase = " ".join(words[window_start:window_end])
                sense_cue_phrases[item.sense_id][window_phrase] += 1
            except StopIteration:
                pass

            if leakage_found:
                audit.items_with_leakage += 1
                if len(audit.flagged_items) < 20:  # Limit stored examples
                    audit.flagged_items.append({
                        "bundle_id": bundle.bundle_id,
                        "word": word,
                        "context": context[:100],
                        "patterns": leakage_found
                    })

    # Find repeated cue phrases (same phrase appears 3+ times for same sense)
    for sense_id, phrases in sense_cue_phrases.items():
        for phrase, count in phrases.items():
            if count >= 3 and len(phrase.split()) >= 3:
                audit.repeated_cue_phrases.append((phrase, count))

    # Sort by frequency
    audit.repeated_cue_phrases.sort(key=lambda x: -x[1])
    audit.repeated_cue_phrases = audit.repeated_cue_phrases[:10]  # Top 10

    # Compute leakage rate
    if audit.total_items_scanned > 0:
        audit.leakage_rate = audit.items_with_leakage / audit.total_items_scanned

    # Determine pass/fail
    if audit.leakage_rate > 0.1:  # More than 10% leakage
        audit.passed = False
        audit.issues.append(f"High leakage rate: {audit.leakage_rate*100:.1f}% of items have leakage patterns")

    if len(audit.repeated_cue_phrases) > 5:
        audit.passed = False
        audit.issues.append(f"Found {len(audit.repeated_cue_phrases)} repeated cue phrases (potential shortcuts)")

    return audit


def characterize_dataset(bundles: List[PolysemyBundle], run_audits: bool = True) -> DatasetReport:
    """Generate complete characterization report for a dataset."""
    report = DatasetReport()

    # Basic stats
    report.total_bundles = len(bundles)
    report.total_items = sum(len(b.items) for b in bundles)

    # Word and sense coverage
    word_bundles = defaultdict(list)
    for bundle in bundles:
        word_bundles[bundle.word.surface].append(bundle)
        report.word_coverage[bundle.word.surface] = report.word_coverage.get(bundle.word.surface, 0) + 1
        for item in bundle.items:
            report.sense_coverage[item.sense_id] = report.sense_coverage.get(item.sense_id, 0) + 1
            report.role_distribution[item.role] = report.role_distribution.get(item.role, 0) + 1

    report.unique_words = len(word_bundles)
    report.unique_senses = len(report.sense_coverage)
    report.avg_bundles_per_word = report.total_bundles / report.unique_words if report.unique_words > 0 else 0
    report.avg_items_per_bundle = report.total_items / report.total_bundles if report.total_bundles > 0 else 0

    # Difficulty analysis
    report.difficulty = analyze_difficulty(bundles)

    # Contrastive geometry
    report.geometry = analyze_contrastive_geometry(bundles)

    # Per-word analysis
    for word, word_bundle_list in word_bundles.items():
        report.word_analyses.append(analyze_word(word, word_bundle_list))

    # Run audits if requested
    if run_audits:
        report.hard_negative_audit = audit_hard_negatives(bundles)
        report.leakage_audit = audit_answer_leakage(bundles)

        # Add audit issues to main issues list
        if report.hard_negative_audit and not report.hard_negative_audit.passed:
            report.issues.extend(report.hard_negative_audit.issues)
            report.recommendations.append("Review flagged hard negatives and regenerate if needed")

        if report.leakage_audit and not report.leakage_audit.passed:
            report.issues.extend(report.leakage_audit.issues)
            report.recommendations.append("Clean contexts with leakage patterns")

    # Quality scoring
    quality_factors = []

    # Factor 1: Difficulty balance (target: 33/33/33)
    quality_factors.append(report.difficulty.balance_score)

    # Factor 2: Completeness (bundles with full contrastive structure)
    completeness = report.geometry.complete_bundles / report.geometry.total_bundles if report.geometry.total_bundles > 0 else 0
    quality_factors.append(completeness)

    # Factor 3: Hard negative coverage
    hard_neg_coverage = report.geometry.bundles_with_hard_negatives / report.geometry.total_bundles if report.geometry.total_bundles > 0 else 0
    quality_factors.append(hard_neg_coverage)

    # Factor 4: Word diversity (entropy-based)
    word_counts = list(report.word_coverage.values())
    if word_counts and sum(word_counts) > 0:
        probs = [c / sum(word_counts) for c in word_counts]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(len(word_counts)) if len(word_counts) > 1 else 1
        word_diversity = entropy / max_entropy if max_entropy > 0 else 0
        quality_factors.append(word_diversity)

    report.quality_score = sum(quality_factors) / len(quality_factors) if quality_factors else 0

    # Generate issues and recommendations
    if report.difficulty.balance_score < 0.6:
        if report.difficulty.easy_pct > 50:
            report.issues.append("Dataset skews easy (>50%)")
            report.recommendations.append("Add more hard examples with minimal cues")
        elif report.difficulty.hard_pct > 50:
            report.issues.append("Dataset skews hard (>50%)")
            report.recommendations.append("Add more easy examples with clear cues")

    if hard_neg_coverage < 0.3:
        report.issues.append(f"Low hard_negative coverage ({100*hard_neg_coverage:.1f}%)")
        report.recommendations.append("Generate more hard negatives (similar surface, different sense)")

    if report.avg_items_per_bundle < 3.5:
        report.issues.append(f"Low items per bundle ({report.avg_items_per_bundle:.1f})")
        report.recommendations.append("Add more positive/negative examples per bundle")

    # Check for underrepresented words
    low_coverage_words = [w for w, c in report.word_coverage.items() if c < 5]
    if len(low_coverage_words) > report.unique_words * 0.2:
        report.issues.append(f"{len(low_coverage_words)} words have <5 bundles")
        report.recommendations.append("Generate more bundles for underrepresented words")

    return report


def print_report(report: DatasetReport, verbose: bool = False):
    """Print a formatted report to console."""
    print("\n" + "=" * 70)
    print("  DATASET CHARACTERIZATION REPORT")
    print("=" * 70)

    print(f"\n  BASIC STATS")
    print(f"  -----------")
    print(f"  Total bundles:        {report.total_bundles:,}")
    print(f"  Total items:          {report.total_items:,}")
    print(f"  Unique words:         {report.unique_words}")
    print(f"  Unique senses:        {report.unique_senses}")
    print(f"  Avg bundles/word:     {report.avg_bundles_per_word:.1f}")
    print(f"  Avg items/bundle:     {report.avg_items_per_bundle:.1f}")

    print(f"\n  ROLE DISTRIBUTION")
    print(f"  -----------------")
    for role, count in sorted(report.role_distribution.items()):
        pct = 100 * count / report.total_items if report.total_items > 0 else 0
        print(f"  {role:15s}  {count:6,} ({pct:5.1f}%)")

    print(f"\n  DIFFICULTY DISTRIBUTION")
    print(f"  -----------------------")
    d = report.difficulty
    print(f"  Easy (<0.33):    {d.easy_count:6,} ({d.easy_pct:5.1f}%)")
    print(f"  Medium:          {d.medium_count:6,} ({d.medium_pct:5.1f}%)")
    print(f"  Hard (>=0.66):   {d.hard_count:6,} ({d.hard_pct:5.1f}%)")
    print(f"  Balance score:   {d.balance_score:.3f} (1.0 = perfect 33/33/33)")

    print(f"\n  CONTRASTIVE GEOMETRY")
    print(f"  --------------------")
    g = report.geometry
    print(f"  Bundles with positives:       {g.bundles_with_positives:5,} ({100*g.bundles_with_positives/g.total_bundles:.1f}%)")
    print(f"  Bundles with negatives:       {g.bundles_with_negatives:5,} ({100*g.bundles_with_negatives/g.total_bundles:.1f}%)")
    print(f"  Bundles with hard_negatives:  {g.bundles_with_hard_negatives:5,} ({100*g.bundles_with_hard_negatives/g.total_bundles:.1f}%)")
    print(f"  Complete bundles:             {g.complete_bundles:5,} ({100*g.complete_bundles/g.total_bundles:.1f}%)")
    print(f"  Avg positives/bundle:         {g.avg_positives_per_bundle:.2f}")
    print(f"  Avg negatives/bundle:         {g.avg_negatives_per_bundle:.2f}")
    print(f"  Avg hard_negatives/bundle:    {g.avg_hard_negatives_per_bundle:.2f}")

    # Hard Negative Audit
    if report.hard_negative_audit:
        hn = report.hard_negative_audit
        print(f"\n  HARD NEGATIVE AUDIT")
        print(f"  -------------------")
        print(f"  Bundles audited:    {hn.total_bundles_audited}")
        print(f"  Valid hard negs:    {hn.bundles_with_valid_hard_negs}")
        print(f"  Invalid hard negs:  {hn.bundles_with_invalid_hard_negs}")
        print(f"  Avg hard_neg sim:   {hn.avg_hard_neg_similarity:.3f}")
        print(f"  Avg easy_neg sim:   {hn.avg_easy_neg_similarity:.3f}")
        status = "PASS" if hn.passed else "FAIL"
        print(f"  Status: {status}")
        if hn.issues:
            for issue in hn.issues:
                print(f"    ! {issue}")

    # Leakage Audit
    if report.leakage_audit:
        la = report.leakage_audit
        print(f"\n  ANSWER LEAKAGE AUDIT")
        print(f"  --------------------")
        print(f"  Items scanned:      {la.total_items_scanned}")
        print(f"  Items with leakage: {la.items_with_leakage} ({la.leakage_rate*100:.1f}%)")
        if la.leakage_by_type:
            print(f"  Leakage patterns found:")
            for pattern, count in sorted(la.leakage_by_type.items(), key=lambda x: -x[1])[:5]:
                print(f"    {pattern}: {count}")
        if la.repeated_cue_phrases:
            print(f"  Top repeated cue phrases:")
            for phrase, count in la.repeated_cue_phrases[:3]:
                print(f"    '{phrase}' ({count}x)")
        status = "PASS" if la.passed else "FAIL"
        print(f"  Status: {status}")

    print(f"\n  QUALITY ASSESSMENT")
    print(f"  ------------------")
    print(f"  Overall quality score: {report.quality_score:.3f}")

    if report.issues:
        print(f"\n  Issues:")
        for issue in report.issues:
            print(f"    ! {issue}")

    if report.recommendations:
        print(f"\n  Recommendations:")
        for rec in report.recommendations:
            print(f"    > {rec}")

    if verbose and report.word_analyses:
        print(f"\n  TOP 10 WORDS BY BUNDLE COUNT")
        print(f"  ----------------------------")
        sorted_words = sorted(report.word_analyses, key=lambda w: -w.bundle_count)[:10]
        for wa in sorted_words:
            print(f"  {wa.word:15s}  bundles={wa.bundle_count:3d}  senses={wa.sense_count:2d}  hard_neg={wa.hard_negative_count}")


def save_report(report: DatasetReport, output_path: Path):
    """Save report to JSON file."""
    # Convert to serializable format
    data = {
        "total_bundles": report.total_bundles,
        "total_items": report.total_items,
        "unique_words": report.unique_words,
        "unique_senses": report.unique_senses,
        "avg_bundles_per_word": report.avg_bundles_per_word,
        "avg_items_per_bundle": report.avg_items_per_bundle,
        "difficulty": {
            "easy_count": report.difficulty.easy_count,
            "medium_count": report.difficulty.medium_count,
            "hard_count": report.difficulty.hard_count,
            "easy_pct": report.difficulty.easy_pct,
            "medium_pct": report.difficulty.medium_pct,
            "hard_pct": report.difficulty.hard_pct,
            "balance_score": report.difficulty.balance_score
        },
        "geometry": {
            "bundles_with_positives": report.geometry.bundles_with_positives,
            "bundles_with_negatives": report.geometry.bundles_with_negatives,
            "bundles_with_hard_negatives": report.geometry.bundles_with_hard_negatives,
            "complete_bundles": report.geometry.complete_bundles,
            "avg_positives_per_bundle": report.geometry.avg_positives_per_bundle,
            "avg_negatives_per_bundle": report.geometry.avg_negatives_per_bundle,
            "avg_hard_negatives_per_bundle": report.geometry.avg_hard_negatives_per_bundle
        },
        "role_distribution": report.role_distribution,
        "quality_score": report.quality_score,
        "issues": report.issues,
        "recommendations": report.recommendations,
        "word_coverage": report.word_coverage,
        "top_words": [
            {"word": wa.word, "bundles": wa.bundle_count, "senses": wa.sense_count}
            for wa in sorted(report.word_analyses, key=lambda w: -w.bundle_count)[:20]
        ]
    }

    # Add audit results if available
    if report.hard_negative_audit:
        hn = report.hard_negative_audit
        data["hard_negative_audit"] = {
            "total_bundles_audited": hn.total_bundles_audited,
            "bundles_with_valid_hard_negs": hn.bundles_with_valid_hard_negs,
            "bundles_with_invalid_hard_negs": hn.bundles_with_invalid_hard_negs,
            "avg_hard_neg_similarity": hn.avg_hard_neg_similarity,
            "avg_easy_neg_similarity": hn.avg_easy_neg_similarity,
            "passed": hn.passed,
            "issues": hn.issues,
            "flagged_bundle_ids": hn.flagged_bundle_ids[:20]  # Top 20
        }

    if report.leakage_audit:
        la = report.leakage_audit
        data["leakage_audit"] = {
            "total_items_scanned": la.total_items_scanned,
            "items_with_leakage": la.items_with_leakage,
            "leakage_rate": la.leakage_rate,
            "leakage_by_type": la.leakage_by_type,
            "repeated_cue_phrases": la.repeated_cue_phrases,
            "passed": la.passed,
            "issues": la.issues,
            "flagged_items": la.flagged_items[:10]  # Top 10
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nReport saved to: {output_path}")


def compare_datasets(paths: List[str]):
    """Compare multiple datasets side by side."""
    reports = []
    for path in paths:
        bundles = load_bundles(Path(path))
        report = characterize_dataset(bundles)
        reports.append((Path(path).name, report))

    print("\n" + "=" * 80)
    print("  DATASET COMPARISON")
    print("=" * 80)

    # Header
    header = f"{'Metric':<30}"
    for name, _ in reports:
        header += f"{name[:15]:>15}"
    print(header)
    print("-" * 80)

    # Metrics
    metrics = [
        ("Total bundles", lambda r: f"{r.total_bundles:,}"),
        ("Total items", lambda r: f"{r.total_items:,}"),
        ("Unique words", lambda r: f"{r.unique_words}"),
        ("Unique senses", lambda r: f"{r.unique_senses}"),
        ("Avg items/bundle", lambda r: f"{r.avg_items_per_bundle:.1f}"),
        ("Easy %", lambda r: f"{r.difficulty.easy_pct:.1f}%"),
        ("Medium %", lambda r: f"{r.difficulty.medium_pct:.1f}%"),
        ("Hard %", lambda r: f"{r.difficulty.hard_pct:.1f}%"),
        ("Difficulty balance", lambda r: f"{r.difficulty.balance_score:.3f}"),
        ("Hard neg coverage", lambda r: f"{100*r.geometry.bundles_with_hard_negatives/r.geometry.total_bundles:.1f}%"),
        ("Quality score", lambda r: f"{r.quality_score:.3f}"),
    ]

    for metric_name, metric_fn in metrics:
        row = f"{metric_name:<30}"
        for _, report in reports:
            row += f"{metric_fn(report):>15}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Dataset Characterization Tools")
    parser.add_argument("--input", type=str, help="Input bundles file (JSONL)")
    parser.add_argument("--output-report", type=str, help="Output JSON report path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--compare", nargs="+", help="Compare multiple dataset files")
    parser.add_argument("--no-audits", action="store_true",
                        help="Skip hard-negative and leakage audits (faster)")
    parser.add_argument("--audit-only", action="store_true",
                        help="Run only the audits (skip basic characterization)")

    args = parser.parse_args()

    if args.compare:
        compare_datasets(args.compare)
        return

    if not args.input:
        parser.error("--input is required")

    # Load bundles
    bundles = load_bundles(Path(args.input))

    if args.audit_only:
        # Just run audits
        print(f"\nRunning audits on {len(bundles)} bundles...")

        print("\n" + "=" * 60)
        print("  HARD NEGATIVE AUDIT")
        print("=" * 60)
        hn_audit = audit_hard_negatives(bundles)
        print(f"  Bundles audited:    {hn_audit.total_bundles_audited}")
        print(f"  Valid hard negs:    {hn_audit.bundles_with_valid_hard_negs}")
        print(f"  Invalid hard negs:  {hn_audit.bundles_with_invalid_hard_negs}")
        print(f"  Avg hard_neg sim:   {hn_audit.avg_hard_neg_similarity:.3f}")
        print(f"  Avg easy_neg sim:   {hn_audit.avg_easy_neg_similarity:.3f}")
        status = "PASS" if hn_audit.passed else "FAIL"
        print(f"  Status: {status}")
        if hn_audit.issues:
            for issue in hn_audit.issues:
                print(f"    ! {issue}")

        print("\n" + "=" * 60)
        print("  ANSWER LEAKAGE AUDIT")
        print("=" * 60)
        leak_audit = audit_answer_leakage(bundles)
        print(f"  Items scanned:      {leak_audit.total_items_scanned}")
        print(f"  Items with leakage: {leak_audit.items_with_leakage} ({leak_audit.leakage_rate*100:.1f}%)")
        status = "PASS" if leak_audit.passed else "FAIL"
        print(f"  Status: {status}")
        if leak_audit.issues:
            for issue in leak_audit.issues:
                print(f"    ! {issue}")

        return

    # Full characterization
    report = characterize_dataset(bundles, run_audits=not args.no_audits)

    # Print report
    print_report(report, verbose=args.verbose)

    # Save if requested
    if args.output_report:
        save_report(report, Path(args.output_report))


if __name__ == "__main__":
    main()
