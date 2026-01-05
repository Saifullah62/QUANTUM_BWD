#!/usr/bin/env python3
"""
Paradigm Factory - Phase B: Embedding QA & Clustering
======================================================

Takes generated polysemy examples and:
1. Embeds each example using Fleet :8001/embed
2. Deduplicates near-identical examples
3. Clusters by word to find sense separation
4. Assigns difficulty scores based on embedding geometry
5. Stores in Fleet :8001/store for retrieval

Hard negatives that are close to positives in embedding space = valuable
Easy negatives that are far away = less useful for training
"""

import argparse
import json
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

# Fleet service base URL
FLEET_BASE = "http://159.203.35.45"

# Thresholds
DEDUPE_THRESHOLD = 0.95  # Cosine similarity above this = duplicate
HARD_NEGATIVE_THRESHOLD = 0.7  # Similarity above this = hard negative


def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from Fleet embedding box."""
    try:
        resp = requests.post(
            f"{FLEET_BASE}:8001/embed",
            json={"text": text},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
    except requests.RequestException as e:
        print(f"  Warning: Embedding failed: {e}")
        return None


def store_example(example: Dict, embedding: List[float], namespace: str = "polysemy") -> bool:
    """Store example with embedding in Fleet store."""
    try:
        resp = requests.post(
            f"{FLEET_BASE}:8001/store",
            json={
                "id": example["text_hash"],
                "text": example["context"],
                "namespace": namespace,
                "metadata": {
                    "word": example["word"],
                    "sense": example["sense"],
                    "subtype": example["subtype"],
                    "paradigm": example["paradigm"],
                    "difficulty": example.get("difficulty", 0.0)
                }
            },
            timeout=30
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"  Warning: Store failed: {e}")
        return False


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def deduplicate(examples: List[Dict], embeddings: List[List[float]],
                threshold: float = DEDUPE_THRESHOLD) -> Tuple[List[Dict], List[List[float]]]:
    """Remove near-duplicate examples based on embedding similarity."""
    if not examples:
        return [], []

    kept_examples = [examples[0]]
    kept_embeddings = [embeddings[0]]

    for ex, emb in zip(examples[1:], embeddings[1:]):
        is_duplicate = False
        for kept_emb in kept_embeddings:
            if cosine_similarity(emb, kept_emb) > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_examples.append(ex)
            kept_embeddings.append(emb)

    return kept_examples, kept_embeddings


def compute_difficulty_scores(
    examples: List[Dict],
    embeddings: List[List[float]]
) -> List[Dict]:
    """
    Assign difficulty scores based on embedding geometry.

    For hard negatives: higher score if close to positives of different sense
    For positives: higher score if close to negatives
    """
    # Group by word
    by_word = defaultdict(list)
    for ex, emb in zip(examples, embeddings):
        by_word[ex["word"]].append((ex, emb))

    scored_examples = []

    for word, items in by_word.items():
        positives = [(ex, emb) for ex, emb in items if ex["subtype"] == "positive"]
        negatives = [(ex, emb) for ex, emb in items if ex["subtype"] != "positive"]

        for ex, emb in items:
            if ex["subtype"] == "positive":
                # Difficulty = max similarity to any negative of different sense
                max_sim = 0.0
                for neg_ex, neg_emb in negatives:
                    if neg_ex["sense"] != ex["sense"]:
                        sim = cosine_similarity(emb, neg_emb)
                        max_sim = max(max_sim, sim)
                ex["difficulty"] = max_sim

            else:  # negative
                # Difficulty = max similarity to any positive of different sense
                max_sim = 0.0
                for pos_ex, pos_emb in positives:
                    if pos_ex["sense"] != ex["sense"]:
                        sim = cosine_similarity(emb, pos_emb)
                        max_sim = max(max_sim, sim)
                ex["difficulty"] = max_sim

            scored_examples.append(ex)

    return scored_examples


def analyze_sense_separation(
    examples: List[Dict],
    embeddings: List[List[float]]
) -> Dict[str, Dict]:
    """
    Analyze how well senses are separated in embedding space.
    Good separation = easier to learn; poor separation = harder but more valuable.
    """
    by_word = defaultdict(lambda: defaultdict(list))

    for ex, emb in zip(examples, embeddings):
        if ex["subtype"] == "positive":
            by_word[ex["word"]][ex["sense"]].append(emb)

    analysis = {}

    for word, sense_embeddings in by_word.items():
        senses = list(sense_embeddings.keys())
        if len(senses) < 2:
            continue

        # Compute inter-sense similarity (lower = better separation)
        inter_sims = []
        for i, sense1 in enumerate(senses):
            for sense2 in senses[i+1:]:
                for emb1 in sense_embeddings[sense1]:
                    for emb2 in sense_embeddings[sense2]:
                        inter_sims.append(cosine_similarity(emb1, emb2))

        # Compute intra-sense similarity (higher = better coherence)
        intra_sims = []
        for sense, embs in sense_embeddings.items():
            if len(embs) >= 2:
                for i, emb1 in enumerate(embs):
                    for emb2 in embs[i+1:]:
                        intra_sims.append(cosine_similarity(emb1, emb2))

        analysis[word] = {
            "num_senses": len(senses),
            "inter_sense_sim_mean": float(np.mean(inter_sims)) if inter_sims else 0,
            "inter_sense_sim_std": float(np.std(inter_sims)) if inter_sims else 0,
            "intra_sense_sim_mean": float(np.mean(intra_sims)) if intra_sims else 0,
            "separation_score": float(np.mean(intra_sims) - np.mean(inter_sims))
                if intra_sims and inter_sims else 0
        }

    return analysis


def process_polysemy_batch(
    input_path: Path,
    output_path: Optional[Path] = None,
    store_to_fleet: bool = False,
    namespace: str = "polysemy"
) -> Tuple[List[Dict], Dict]:
    """
    Process a batch of polysemy examples:
    1. Load and embed
    2. Deduplicate
    3. Score difficulty
    4. Analyze separation
    5. Optionally store to Fleet
    """
    # Load examples
    print(f"Loading examples from {input_path}...")
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} examples")

    # Embed all
    print("Generating embeddings...")
    embeddings = []
    for i, ex in enumerate(examples):
        if i % 10 == 0:
            print(f"  {i}/{len(examples)}")
        emb = get_embedding(ex["context"])
        if emb:
            embeddings.append(emb)
        else:
            embeddings.append([0.0] * 384)  # Placeholder

    # Deduplicate
    print("Deduplicating...")
    original_count = len(examples)
    examples, embeddings = deduplicate(examples, embeddings)
    print(f"  Removed {original_count - len(examples)} duplicates")

    # Score difficulty
    print("Computing difficulty scores...")
    examples = compute_difficulty_scores(examples, embeddings)

    # Analyze separation
    print("Analyzing sense separation...")
    analysis = analyze_sense_separation(examples, embeddings)

    # Store to Fleet if requested
    if store_to_fleet:
        print(f"Storing to Fleet namespace '{namespace}'...")
        stored = 0
        for ex, emb in zip(examples, embeddings):
            if store_example(ex, emb, namespace):
                stored += 1
        print(f"  Stored {stored}/{len(examples)} examples")

    # Save processed output
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved processed examples to {output_path}")

        # Save analysis
        analysis_path = output_path.with_suffix('.analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Saved analysis to {analysis_path}")

    return examples, analysis


def main():
    parser = argparse.ArgumentParser(description="Embed and cluster polysemy examples")
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL from Phase A')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSONL path (default: input_processed.jsonl)')
    parser.add_argument('--store', action='store_true',
                        help='Store to Fleet embedding store')
    parser.add_argument('--namespace', type=str, default='polysemy',
                        help='Fleet store namespace')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix('.processed.jsonl')

    print("=" * 60)
    print("  PARADIGM FACTORY - EMBEDDING QA & CLUSTERING")
    print("=" * 60)

    examples, analysis = process_polysemy_batch(
        input_path=input_path,
        output_path=output_path,
        store_to_fleet=args.store,
        namespace=args.namespace
    )

    # Summary
    print("\n" + "=" * 60)
    print("  PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Final examples: {len(examples)}")
    print(f"Words analyzed: {len(analysis)}")

    # Best/worst separation
    if analysis:
        sorted_words = sorted(analysis.items(),
                              key=lambda x: x[1]["separation_score"],
                              reverse=True)
        print("\nBest sense separation:")
        for word, stats in sorted_words[:3]:
            print(f"  {word}: {stats['separation_score']:.3f}")
        print("\nWorst sense separation (hardest to learn):")
        for word, stats in sorted_words[-3:]:
            print(f"  {word}: {stats['separation_score']:.3f}")

    # Difficulty distribution
    difficulties = [ex.get("difficulty", 0) for ex in examples]
    if difficulties:
        print(f"\nDifficulty distribution:")
        print(f"  Mean: {np.mean(difficulties):.3f}")
        print(f"  Std:  {np.std(difficulties):.3f}")
        print(f"  Hard (>0.7): {len([d for d in difficulties if d > 0.7])}")


if __name__ == "__main__":
    main()
