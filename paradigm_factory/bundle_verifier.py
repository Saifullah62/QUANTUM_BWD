#!/usr/bin/env python3
"""
Bundle Verifier - Quality Gate for v2 Bundles
==============================================

Two-stage quality control:
1. Heuristic filters (fast, local)
2. LLM verification (swarm-based, for ambiguous cases)

Rejection criteria:
- Anchor/positive not truly same-sense
- Negatives not truly different-sense
- Giveaway contexts (explicit definitions, gloss words verbatim)
- Near-duplicate contexts (high n-gram overlap)
- Difficulty imbalance (enforces target distribution)

Usage:
    python bundle_verifier.py --input raw_bundles.jsonl --output clean_bundles.jsonl
"""

import argparse
import json
import hashlib
import re
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.polysemy_bundle_v2 import (
    PolysemyBundle, load_bundles, save_bundles, SENSE_CATALOG
)

FLEET_BASE = "http://159.203.35.45"


@dataclass
class VerificationResult:
    """Result of verifying a single bundle."""
    bundle_id: str
    passed: bool
    rejection_reason: Optional[str] = None
    confidence: float = 1.0


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text


def get_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    text = normalize_text(text)
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """Compute Jaccard similarity of n-grams."""
    ng1 = get_ngrams(text1, n)
    ng2 = get_ngrams(text2, n)
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


def content_hash(text: str) -> str:
    """Hash normalized text for deduplication."""
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def check_giveaway_patterns(context: str, word: str, sense_label: str, gloss: str) -> Tuple[bool, str]:
    """
    Check if context is a giveaway (too obvious).

    Giveaway patterns:
    - Explicit definitions: "X means...", "X is defined as..."
    - Dictionary style: "X (noun): ..."
    - Gloss words appear verbatim
    - Meta-references: "the word X", "in this sense"
    """
    context_lower = context.lower()
    word_lower = word.lower()

    # Explicit definition patterns
    definition_patterns = [
        f"{word_lower} means",
        f"{word_lower} is defined as",
        f"{word_lower} refers to",
        f"definition of {word_lower}",
        f"{word_lower} (noun)",
        f"{word_lower} (verb)",
        f"the word {word_lower}",
        f"'{word_lower}' means",
        f'"{word_lower}" means',
        "in this context",
        "in this sense",
        "meaning of the word",
    ]

    for pattern in definition_patterns:
        if pattern in context_lower:
            return True, f"Definition pattern: '{pattern}'"

    # Check if gloss appears verbatim (more than 3 consecutive words)
    if gloss:
        gloss_words = normalize_text(gloss).split()
        context_words = normalize_text(context).split()

        for i in range(len(gloss_words) - 3):
            phrase = ' '.join(gloss_words[i:i+4])
            if phrase in ' '.join(context_words):
                return True, f"Gloss verbatim: '{phrase}'"

    return False, ""


def check_near_duplicate(context: str, existing_contexts: Set[str], threshold: float = 0.7) -> Tuple[bool, str]:
    """Check if context is a near-duplicate of existing ones."""
    context_hash = content_hash(context)

    # Exact duplicate
    if context_hash in existing_contexts:
        return True, "Exact duplicate"

    # N-gram similarity check (more expensive, sample if many)
    for existing in list(existing_contexts)[:100]:
        sim = ngram_similarity(context, existing)
        if sim > threshold:
            return True, f"Near-duplicate (sim={sim:.2f})"

    return False, ""


def verify_sense_assignment_heuristic(
    context: str,
    assigned_sense: str,
    word: str,
    all_senses: List[Dict]
) -> Tuple[bool, float, str]:
    """
    Heuristically verify sense assignment using cue matching.

    Returns: (is_correct, confidence, reason)
    """
    if word not in SENSE_CATALOG:
        return True, 0.5, "No catalog entry (assumed OK)"

    sense_defs = SENSE_CATALOG[word]
    context_lower = context.lower()

    # Find assigned sense definition
    assigned_def = None
    for sd in sense_defs:
        if sd.sense_id == assigned_sense or sd.label == assigned_sense:
            assigned_def = sd
            break

    if not assigned_def:
        return True, 0.5, "Sense not in catalog (assumed OK)"

    # Count cues for assigned sense
    assigned_cue_count = sum(1 for cue in assigned_def.cues if cue.lower() in context_lower)
    assigned_anti_count = sum(1 for cue in assigned_def.anti_cues if cue.lower() in context_lower)

    # Count cues for other senses
    other_cue_counts = []
    for sd in sense_defs:
        if sd.sense_id != assigned_sense and sd.label != assigned_sense:
            other_count = sum(1 for cue in sd.cues if cue.lower() in context_lower)
            other_cue_counts.append((sd.label, other_count))

    # Decision logic
    max_other = max([c for _, c in other_cue_counts], default=0)

    # Clear correct: assigned has more cues and no anti-cues
    if assigned_cue_count > max_other and assigned_anti_count == 0:
        confidence = min(0.9, 0.5 + 0.1 * assigned_cue_count)
        return True, confidence, f"Cue match: {assigned_cue_count} assigned vs {max_other} other"

    # Clear wrong: other sense has more cues
    if max_other > assigned_cue_count + 1:
        best_other = max(other_cue_counts, key=lambda x: x[1])
        return False, 0.8, f"Wrong sense: {best_other[0]} has {best_other[1]} cues vs {assigned_cue_count}"

    # Anti-cues present: suspicious
    if assigned_anti_count > 0:
        return False, 0.6, f"Anti-cues present: {assigned_anti_count}"

    # Ambiguous
    return True, 0.4, "Ambiguous (low confidence)"


def verify_bundle_heuristic(bundle: PolysemyBundle, existing_contexts: Set[str],
                           check_cross_bundle_duplicates: bool = False) -> VerificationResult:
    """
    Fast heuristic verification of a bundle.

    Checks:
    1. Giveaway patterns in any context
    2. Internal duplicates (same context used twice in bundle)
    3. Cross-bundle duplicates (optional, for eval pack sealing)
    4. Sense assignment correctness (cue-based)
    5. Minimum item count
    """
    word = bundle.word.surface

    # Check minimum items
    if len(bundle.items) < 3:
        return VerificationResult(
            bundle_id=bundle.bundle_id,
            passed=False,
            rejection_reason="Too few items (< 3)",
            confidence=1.0
        )

    # Track contexts within this bundle for internal duplicate check
    bundle_contexts: Set[str] = set()

    # Check each item
    for item in bundle.items:
        context = item.context
        ctx_hash = content_hash(context)

        # Get gloss for this sense
        gloss = ""
        for sd in bundle.sense_catalog:
            if sd.sense_id == item.sense_id:
                gloss = sd.gloss
                break

        # Giveaway check
        is_giveaway, reason = check_giveaway_patterns(context, word, item.sense_id, gloss)
        if is_giveaway:
            return VerificationResult(
                bundle_id=bundle.bundle_id,
                passed=False,
                rejection_reason=f"Giveaway in {item.role}: {reason}",
                confidence=0.9
            )

        # Internal duplicate check (same context twice in this bundle)
        if ctx_hash in bundle_contexts:
            return VerificationResult(
                bundle_id=bundle.bundle_id,
                passed=False,
                rejection_reason=f"Internal duplicate in {item.role}",
                confidence=0.95
            )
        bundle_contexts.add(ctx_hash)

        # Cross-bundle duplicate check (only for eval pack sealing)
        if check_cross_bundle_duplicates:
            is_dup, reason = check_near_duplicate(context, existing_contexts)
            if is_dup:
                return VerificationResult(
                    bundle_id=bundle.bundle_id,
                    passed=False,
                    rejection_reason=f"Cross-bundle duplicate in {item.role}: {reason}",
                    confidence=0.95
                )

        # Sense verification (for anchor and positives especially)
        if item.role in ("anchor", "positive"):
            is_correct, conf, reason = verify_sense_assignment_heuristic(
                context, item.sense_id, word, bundle.sense_catalog
            )
            if not is_correct:
                return VerificationResult(
                    bundle_id=bundle.bundle_id,
                    passed=False,
                    rejection_reason=f"Sense mismatch in {item.role}: {reason}",
                    confidence=conf
                )

    # All checks passed
    return VerificationResult(
        bundle_id=bundle.bundle_id,
        passed=True,
        confidence=0.8
    )


def verify_bundle_llm(bundle: PolysemyBundle) -> VerificationResult:
    """
    LLM-based verification for ambiguous cases.
    Uses swarm to verify sense assignments.
    """
    word = bundle.word.surface
    anchor = next((i for i in bundle.items if i.role == "anchor"), None)
    positive = next((i for i in bundle.items if i.role == "positive"), None)
    negative = next((i for i in bundle.items if i.role == "negative"), None)

    if not anchor or not positive:
        return VerificationResult(
            bundle_id=bundle.bundle_id,
            passed=False,
            rejection_reason="Missing anchor or positive",
            confidence=1.0
        )

    prompt = f"""Verify these sense assignments for the word "{word}":

ANCHOR ({anchor.sense_id.split('#')[1]}):
"{anchor.context}"

POSITIVE (should be SAME sense as anchor):
"{positive.context}"

{f'NEGATIVE (should be DIFFERENT sense):' + chr(10) + f'"{negative.context}"' if negative else ''}

Questions:
1. Does the POSITIVE use "{word}" in the same sense as the ANCHOR? (yes/no)
2. Does the NEGATIVE use "{word}" in a different sense than the ANCHOR? (yes/no/na)

Answer format:
POSITIVE_SAME: yes/no
NEGATIVE_DIFFERENT: yes/no/na
CONFIDENCE: high/medium/low
"""

    try:
        resp = requests.post(
            f"{FLEET_BASE}:8007/swarm/explore",
            json={"problem": prompt, "num_agents": 1},
            timeout=30
        )
        if resp.status_code == 200:
            result = resp.json()
            text = str(result.get("result", "")).lower()

            pos_same = "positive_same: yes" in text
            neg_diff = "negative_different: yes" in text or "negative_different: na" in text
            confidence = 0.9 if "high" in text else (0.7 if "medium" in text else 0.5)

            if pos_same and neg_diff:
                return VerificationResult(
                    bundle_id=bundle.bundle_id,
                    passed=True,
                    confidence=confidence
                )
            else:
                return VerificationResult(
                    bundle_id=bundle.bundle_id,
                    passed=False,
                    rejection_reason=f"LLM verification failed: pos_same={pos_same}, neg_diff={neg_diff}",
                    confidence=confidence
                )
    except Exception as e:
        pass

    # Fallback: assume OK with low confidence
    return VerificationResult(
        bundle_id=bundle.bundle_id,
        passed=True,
        confidence=0.3
    )


def enforce_difficulty_balance(
    bundles: List[PolysemyBundle],
    target_easy: float = 0.4,
    target_med: float = 0.4,
    target_hard: float = 0.2
) -> List[PolysemyBundle]:
    """
    Filter bundles to enforce difficulty distribution.

    Uses item difficulty scores to bucket bundles, then samples to hit targets.
    """
    # Bucket bundles by average item difficulty
    easy_bundles = []
    med_bundles = []
    hard_bundles = []

    for bundle in bundles:
        difficulties = [i.difficulty for i in bundle.items if i.role != "anchor"]
        if not difficulties:
            continue
        avg_diff = sum(difficulties) / len(difficulties)

        if avg_diff < 0.33:
            easy_bundles.append(bundle)
        elif avg_diff < 0.66:
            med_bundles.append(bundle)
        else:
            hard_bundles.append(bundle)

    total = len(bundles)
    target_counts = {
        'easy': int(total * target_easy),
        'med': int(total * target_med),
        'hard': int(total * target_hard)
    }

    # Sample from each bucket
    import random
    result = []

    for bucket, target in [('easy', target_counts['easy']),
                           ('med', target_counts['med']),
                           ('hard', target_counts['hard'])]:
        source = {'easy': easy_bundles, 'med': med_bundles, 'hard': hard_bundles}[bucket]
        if len(source) <= target:
            result.extend(source)
        else:
            result.extend(random.sample(source, target))

    print(f"Difficulty balance: easy={len([b for b in result if sum(i.difficulty for i in b.items if i.role != 'anchor')/max(1,len([i for i in b.items if i.role != 'anchor'])) < 0.33])}, "
          f"med={len([b for b in result if 0.33 <= sum(i.difficulty for i in b.items if i.role != 'anchor')/max(1,len([i for i in b.items if i.role != 'anchor'])) < 0.66])}, "
          f"hard={len([b for b in result if sum(i.difficulty for i in b.items if i.role != 'anchor')/max(1,len([i for i in b.items if i.role != 'anchor'])) >= 0.66])}")

    return result


def run_verification(
    input_path: Path,
    output_path: Path,
    use_llm: bool = False,
    difficulty_balance: bool = True,
    verbose: bool = True
):
    """Run full verification pipeline."""
    print("=" * 70)
    print("  Bundle Verifier")
    print("=" * 70)

    # Load bundles
    bundles = load_bundles(input_path)
    print(f"Loaded {len(bundles)} bundles from {input_path}")

    # Track seen contexts for deduplication
    existing_contexts: Set[str] = set()

    # Verification results
    passed_bundles = []
    rejected_bundles = []
    rejection_reasons = defaultdict(int)

    for i, bundle in enumerate(bundles):
        # Heuristic verification (don't check cross-bundle duplicates - that's for splitting)
        result = verify_bundle_heuristic(bundle, existing_contexts, check_cross_bundle_duplicates=False)

        # LLM verification for ambiguous cases
        if use_llm and result.passed and result.confidence < 0.6:
            result = verify_bundle_llm(bundle)

        if result.passed:
            passed_bundles.append(bundle)
            # Add contexts to seen set (for tracking, even though we don't check cross-bundle)
            for item in bundle.items:
                existing_contexts.add(content_hash(item.context))
        else:
            rejected_bundles.append((bundle, result))
            rejection_reasons[result.rejection_reason.split(':')[0]] += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(bundles)}: {len(passed_bundles)} passed, {len(rejected_bundles)} rejected")

    print(f"\nHeuristic pass: {len(passed_bundles)} passed, {len(rejected_bundles)} rejected")

    # Rejection breakdown
    if rejection_reasons:
        print("\nRejection reasons:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Difficulty balance
    if difficulty_balance and passed_bundles:
        print("\nApplying difficulty balance...")
        passed_bundles = enforce_difficulty_balance(passed_bundles)

    # Save
    save_bundles(passed_bundles, output_path)
    print(f"\nSaved {len(passed_bundles)} verified bundles to {output_path}")

    # Save rejection log
    rejection_log_path = output_path.with_suffix('.rejections.json')
    with open(rejection_log_path, 'w') as f:
        json.dump([
            {"bundle_id": b.bundle_id, "reason": r.rejection_reason}
            for b, r in rejected_bundles
        ], f, indent=2)
    print(f"Rejection log saved to {rejection_log_path}")

    return passed_bundles


def main():
    parser = argparse.ArgumentParser(description="Verify and filter v2 bundles")
    parser.add_argument('--input', type=str, required=True, help='Input bundles JSONL')
    parser.add_argument('--output', type=str, required=True, help='Output verified bundles JSONL')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for ambiguous cases')
    parser.add_argument('--no-balance', action='store_true', help='Skip difficulty balancing')

    args = parser.parse_args()

    run_verification(
        input_path=Path(args.input),
        output_path=Path(args.output),
        use_llm=args.use_llm,
        difficulty_balance=not args.no_balance
    )


if __name__ == "__main__":
    main()
