#!/usr/bin/env python3
"""
Paradigm Factory - Merge Factory Output to Training JSONL
==========================================================

Converts factory polysemy examples into training-compatible format:
- Creates same-sense positive pairs
- Creates cross-sense negative pairs (hard negatives)
- Merges into existing training JSONL or creates new shard
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime


def extract_sentences_from_context(raw_context: str, word: str) -> List[str]:
    """Extract actual sentences from swarm exploration output."""
    sentences = []

    # The context may be a JSON string of exploration results
    try:
        if raw_context.startswith('['):
            explorations = json.loads(raw_context)
            for exp in explorations:
                solution = exp.get('full_solution', '')
                # Extract sentences from the solution text
                sentences.extend(extract_sentences_from_text(solution, word))
        else:
            sentences.extend(extract_sentences_from_text(raw_context, word))
    except json.JSONDecodeError:
        sentences.extend(extract_sentences_from_text(raw_context, word))

    # Dedupe and filter
    seen = set()
    result = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in seen and len(s_clean) > 15:
            seen.add(s_clean)
            result.append(s_clean)

    return result[:5]  # Max 5 per sense


def extract_sentences_from_text(text: str, word: str) -> List[str]:
    """Extract sentences containing the target word."""
    sentences = []

    # Split on sentence boundaries
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)

    for sent in raw_sentences:
        sent = sent.strip()
        # Clean up numbering, bullets, etc.
        sent = re.sub(r'^[\d\.\-\*\#]+\s*', '', sent)
        sent = sent.strip()

        # Check if word is present (case-insensitive)
        if word.lower() in sent.lower() and 15 < len(sent) < 200:
            sentences.append(sent)

    return sentences


def create_training_pairs(
    examples: List[Dict],
    include_hard_negatives: bool = True
) -> List[Dict]:
    """Create training pairs from polysemy examples."""

    # Group by word and sense
    by_word = defaultdict(lambda: defaultdict(list))

    for ex in examples:
        word = ex['word']
        sense = ex['sense']
        sentences = extract_sentences_from_context(ex['context'], word)
        for sent in sentences:
            by_word[word][sense].append(sent)

    training_items = []

    for word, senses in by_word.items():
        sense_list = list(senses.keys())

        for i, sense in enumerate(sense_list):
            sentences = senses[sense]

            # Create same-sense positive pairs
            for j in range(len(sentences)):
                for k in range(j + 1, len(sentences)):
                    training_items.append({
                        "input_text": f"Context A: {sentences[j]}\nContext B: {sentences[k]}\n\nDo these use '{word}' in the same sense?",
                        "output_text": f"Yes, both use '{word}' in the {sense} sense. The meaning is consistent.",
                        "paradigm": "semantic_phase",
                        "subtype": "polysemy_positive",
                        "metadata": {
                            "word": word,
                            "sense": sense,
                            "pair_type": "same_sense",
                            "expected_phase": "aligned"
                        }
                    })

            # Create cross-sense negative pairs (hard negatives)
            if include_hard_negatives:
                for other_sense in sense_list[i+1:]:
                    other_sentences = senses[other_sense]
                    for sent_a in sentences[:2]:  # Limit pairs
                        for sent_b in other_sentences[:2]:
                            training_items.append({
                                "input_text": f"Context A: {sent_a}\nContext B: {sent_b}\n\nDo these use '{word}' in the same sense?",
                                "output_text": f"No, these use '{word}' in different senses: '{sense}' vs '{other_sense}'.",
                                "paradigm": "semantic_phase",
                                "subtype": "polysemy_negative",
                                "metadata": {
                                    "word": word,
                                    "sense_a": sense,
                                    "sense_b": other_sense,
                                    "pair_type": "cross_sense",
                                    "expected_phase": "misaligned"
                                }
                            })

    return training_items


def merge_to_training(
    polysemy_path: Path,
    output_path: Path,
    existing_data_path: Optional[Path] = None
) -> Tuple[int, int]:
    """Merge factory output into training JSONL."""

    # Load factory polysemy examples
    examples = []
    with open(polysemy_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} polysemy examples from factory")

    # Create training pairs
    training_items = create_training_pairs(examples)
    print(f"Created {len(training_items)} training pairs")

    # Count by type
    positives = sum(1 for t in training_items if t['subtype'] == 'polysemy_positive')
    negatives = sum(1 for t in training_items if t['subtype'] == 'polysemy_negative')
    print(f"  Positive (same-sense): {positives}")
    print(f"  Negative (cross-sense): {negatives}")

    # Merge with existing data if provided
    existing_count = 0
    if existing_data_path and existing_data_path.exists():
        with open(existing_data_path, 'r') as f:
            existing_lines = f.readlines()
        existing_count = len(existing_lines)
        print(f"Loaded {existing_count} existing training examples")
    else:
        existing_lines = []

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        # Write existing data first
        for line in existing_lines:
            f.write(line if line.endswith('\n') else line + '\n')

        # Write new training items
        for item in training_items:
            f.write(json.dumps(item) + '\n')

    total = existing_count + len(training_items)
    print(f"\nWrote {total} total examples to {output_path}")
    print(f"  ({existing_count} existing + {len(training_items)} new)")

    return len(training_items), total


def main():
    parser = argparse.ArgumentParser(description="Merge factory output to training JSONL")
    parser.add_argument('--polysemy', type=str, required=True,
                        help='Path to polysemy JSONL from factory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for merged training JSONL')
    parser.add_argument('--existing', type=str, default=None,
                        help='Existing training JSONL to merge with')

    args = parser.parse_args()

    print("=" * 60)
    print("  PARADIGM FACTORY - MERGE TO TRAINING")
    print("=" * 60)

    new_count, total = merge_to_training(
        polysemy_path=Path(args.polysemy),
        output_path=Path(args.output),
        existing_data_path=Path(args.existing) if args.existing else None
    )

    print("\nDone! Your PolysemyAwareSampler will now have richer contrastive material.")


if __name__ == "__main__":
    main()
