#!/usr/bin/env python3
"""
Paradigm Factory - Scaled Polysemy Pipeline
=============================================

High-signal contrastive bundle generation:
1. Swarm-expands lexicon with tree pattern (sense normalization)
2. Embedding-based verification (sense separation)
3. Hard negative manufacturing via challenge endpoint
4. Namespace storage for deduplication and continuous improvement

Target: Thousands of high-signal bundles → tens of thousands of contrastive opportunities
"""

import argparse
import json
import requests
import hashlib
import random
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import time

# Fleet service endpoints
FLEET_BASE = "http://159.203.35.45"
SWARM_URL = f"{FLEET_BASE}:8007"
EMBED_URL = f"{FLEET_BASE}:8001"

# Quality thresholds
SENSE_SIMILARITY_THRESHOLD = 0.85  # Above this = senses are duplicates
CONTEXT_SEPARATION_THRESHOLD = 0.75  # Below this = senses don't separate
HARD_NEGATIVE_MIN_SIMILARITY = 0.70  # Hard negatives should be close but different

# Seed words - core polysemous vocabulary
SEED_WORDS = [
    "bank", "spring", "bat", "crane", "bark", "bow", "bass", "lead",
    "match", "seal", "light", "rock", "ring", "well", "run", "bear",
    "fair", "palm", "tire", "date", "fan", "jam", "mine", "port",
    "watch", "clip", "nail", "pound", "train", "wave", "present",
]


@dataclass
class SenseBundle:
    """A complete sense bundle for a polysemous word."""
    word: str
    sense_id: str
    gloss: str
    contexts: List[str]
    hard_negatives: List[Dict]  # Each with {text, wrong_sense, similarity}
    embedding: Optional[List[float]] = None
    difficulty: float = 0.0


def call_swarm(endpoint: str, payload: Dict, timeout: int = 90) -> Optional[Dict]:
    """Call a Fleet swarm endpoint."""
    url = f"{SWARM_URL}/swarm/{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  Warning: Swarm {endpoint} failed: {e}")
        return None


def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from Fleet embedding box."""
    try:
        resp = requests.post(
            f"{EMBED_URL}/embed",
            json={"text": text},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
    except requests.RequestException:
        return None


def store_to_namespace(item: Dict, namespace: str) -> bool:
    """Store item to Fleet embedding namespace."""
    try:
        resp = requests.post(
            f"{EMBED_URL}/store",
            json={
                "id": item.get("hash", hashlib.md5(item["text"].encode()).hexdigest()[:12]),
                "text": item["text"],
                "namespace": namespace,
                "metadata": item.get("metadata", {})
            },
            timeout=30
        )
        resp.raise_for_status()
        return True
    except requests.RequestException:
        return False


def search_namespace(text: str, namespace: str, top_k: int = 5) -> List[Dict]:
    """Search namespace for similar items (for deduplication)."""
    try:
        resp = requests.post(
            f"{EMBED_URL}/search",
            json={"query": text, "namespace": namespace, "top_k": top_k},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except requests.RequestException:
        return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity."""
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# =============================================================================
# STEP 1: Lexicon Expansion via Tree Pattern
# =============================================================================

def expand_lexicon_tree(seed_words: List[str], max_senses: int = 4) -> Dict[str, List[str]]:
    """Use swarm/tree to expand words into normalized sense inventories."""
    word_senses = {}

    for word in seed_words:
        prompt = f"""Identify the distinct meanings of the word "{word}".

Rules:
1. List 2-{max_senses} clearly different meanings only
2. Each meaning should be a short, distinct gloss (3-6 words max)
3. Use neutral, concrete language (avoid politically loaded contexts)
4. Format: one meaning per line, no numbering

Example for "bank":
financial institution
river edge
to tilt an aircraft

Now list meanings for "{word}":"""

        result = call_swarm("tree", {
            "problem": prompt
        })

        senses = []
        if result and result.get("status") == "SUCCESS":
            # Parse the final answer from tree response
            best = result.get("final_answer", result.get("best_solution", ""))
            for line in str(best).split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Handle multiple formats:
                # 1. Simple one-liner: "financial institution"
                # 2. Numbered with bold: "1. **Name**: Description"
                # 3. Numbered without bold: "1. Name: Description"

                # Check for numbered format
                if any(line.startswith(f"{i}.") for i in range(1, 5)):
                    line = line.lstrip("0123456789. ")
                    if "**" in line:
                        match = re.search(r'\*\*([^*]+)\*\*', line)
                        if match:
                            meaning = match.group(1).strip().lower()
                            if meaning and 3 < len(meaning) < 50:
                                senses.append(meaning)
                                continue
                    if ":" in line:
                        meaning = line.split(":")[0].strip().lower()
                        if meaning and 3 < len(meaning) < 50:
                            senses.append(meaning)
                            continue
                    # Just a numbered line without special formatting
                    if 3 < len(line) < 50:
                        senses.append(line.lower())
                else:
                    # Simple one-liner (no numbering)
                    # Skip common prefixes/suffixes
                    if line.lower().startswith(("the", "a ", "an ", "example", "now", "here")):
                        continue
                    if 3 < len(line) < 50 and word.lower() not in line.lower():
                        senses.append(line.lower())

        # Fallback if swarm didn't produce useful output
        if len(senses) < 2:
            print(f"  Warning: Tree didn't produce senses for '{word}', using fallback")
            senses = [f"meaning {i+1}" for i in range(2)]

        word_senses[word] = senses[:max_senses]
        print(f"  {word}: {len(word_senses[word])} senses")
        time.sleep(0.5)  # Rate limiting

    return word_senses


# =============================================================================
# STEP 2: Embedding-Based Verification
# =============================================================================

def verify_sense_separation(
    word: str,
    senses: List[str]
) -> Tuple[List[str], Dict[str, List[float]]]:
    """Verify senses are distinct via embedding similarity."""
    # Embed each gloss
    embeddings = {}
    for sense in senses:
        emb = get_embedding(f"{word}: {sense}")
        if emb:
            embeddings[sense] = emb

    if len(embeddings) < 2:
        return senses, embeddings

    # Check for near-duplicates
    unique_senses = []
    for sense in senses:
        if sense not in embeddings:
            continue

        is_duplicate = False
        for other in unique_senses:
            if other in embeddings:
                sim = cosine_similarity(embeddings[sense], embeddings[other])
                if sim > SENSE_SIMILARITY_THRESHOLD:
                    print(f"    Filtering duplicate sense: '{sense}' ≈ '{other}' (sim={sim:.3f})")
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_senses.append(sense)

    return unique_senses, embeddings


def generate_contexts_for_sense(
    word: str,
    sense: str,
    num_contexts: int = 3
) -> List[str]:
    """Generate concrete, neutral contexts for a word sense."""
    prompt = f"""Generate {num_contexts} short, natural sentences using "{word}" in the sense of "{sense}".

Rules:
1. Keep sentences 10-25 words, concrete and descriptive
2. Use neutral, everyday contexts (avoid politics, violence, controversy)
3. Make the meaning unambiguous from context
4. Vary the sentence structure

Return only the sentences, one per line."""

    result = call_swarm("explore", {
        "problem": prompt,
        "num_agents": 2
    })

    contexts = []
    if result:
        text = str(result.get("synthesis", result.get("explorations", "")))
        for line in text.split("\n"):
            line = line.strip()
            # Clean up common prefixes
            line = line.lstrip("0123456789.-)*# ")
            line = line.strip('"\'')
            if word.lower() in line.lower() and 10 < len(line) < 150:
                contexts.append(line)

    return contexts[:num_contexts]


# =============================================================================
# STEP 3: Hard Negative Manufacturing
# =============================================================================

def manufacture_hard_negative(
    word: str,
    correct_sense: str,
    correct_context: str,
    wrong_sense: str
) -> Optional[Dict]:
    """Use swarm/challenge to create a deceptive wrong-sense context."""
    prompt = f"""Given this sentence using "{word}" meaning "{correct_sense}":
"{correct_context}"

Create a similar-looking sentence where "{word}" instead means "{wrong_sense}".
The new sentence should:
1. Be grammatically correct and natural
2. Differ by only a few words from the original
3. Clearly use the different meaning
4. Be a plausible "hard negative" that could confuse a model

Return only the new sentence."""

    result = call_swarm("challenge", {
        "problem": prompt,
        "solution": correct_context,
        "intensity": "moderate"
    })

    if not result:
        return None

    negative_text = str(result.get("challenge", result.get("result", "")))
    # Clean up
    for line in negative_text.split("\n"):
        line = line.strip().strip('"\'')
        if word.lower() in line.lower() and 10 < len(line) < 150:
            return {
                "text": line,
                "wrong_sense": wrong_sense,
                "correct_sense": correct_sense
            }

    return None


def score_hard_negative(
    positive_context: str,
    negative: Dict
) -> float:
    """Score how "hard" a negative is based on embedding similarity."""
    pos_emb = get_embedding(positive_context)
    neg_emb = get_embedding(negative["text"])

    if pos_emb and neg_emb:
        return cosine_similarity(pos_emb, neg_emb)
    return 0.0


# =============================================================================
# STEP 4: Full Pipeline
# =============================================================================

def generate_polysemy_bundles(
    seed_words: List[str],
    contexts_per_sense: int = 3,
    hard_negatives_per_sense: int = 2,
    output_path: Optional[Path] = None,
    namespace: str = "polysemy_v1"
) -> List[SenseBundle]:
    """Generate complete sense bundles for all seed words."""

    print("=" * 60)
    print("  SCALED POLYSEMY PIPELINE")
    print("=" * 60)
    print(f"Seed words: {len(seed_words)}")
    print(f"Contexts per sense: {contexts_per_sense}")
    print(f"Hard negatives per sense: {hard_negatives_per_sense}")
    print()

    # Step 1: Expand lexicon
    print("STEP 1: Expanding lexicon via tree pattern...")
    word_senses = expand_lexicon_tree(seed_words)
    total_senses = sum(len(s) for s in word_senses.values())
    print(f"  Generated {total_senses} senses for {len(word_senses)} words\n")

    bundles = []

    for word, senses in word_senses.items():
        print(f"\nProcessing '{word}'...")

        # Step 2: Verify sense separation
        print("  Step 2: Verifying sense separation...")
        verified_senses, sense_embeddings = verify_sense_separation(word, senses)
        print(f"    Kept {len(verified_senses)}/{len(senses)} distinct senses")

        if len(verified_senses) < 2:
            print(f"    Skipping '{word}' - insufficient distinct senses")
            continue

        # Generate contexts and hard negatives for each sense
        for sense in verified_senses:
            sense_id = hashlib.md5(f"{word}:{sense}".encode()).hexdigest()[:8]

            # Step 3a: Generate contexts
            print(f"  Step 3a: Generating contexts for '{sense}'...")
            contexts = generate_contexts_for_sense(word, sense, contexts_per_sense)
            print(f"    Generated {len(contexts)} contexts")

            if not contexts:
                continue

            # Step 3b: Manufacture hard negatives
            print(f"  Step 3b: Manufacturing hard negatives...")
            hard_negatives = []
            other_senses = [s for s in verified_senses if s != sense]

            for other_sense in other_senses[:2]:  # Limit to 2 other senses
                for ctx in contexts[:1]:  # Use first context as template
                    negative = manufacture_hard_negative(word, sense, ctx, other_sense)
                    if negative:
                        negative["similarity"] = score_hard_negative(ctx, negative)
                        if negative["similarity"] >= HARD_NEGATIVE_MIN_SIMILARITY:
                            hard_negatives.append(negative)
                            print(f"    Hard negative (sim={negative['similarity']:.3f}): {negative['text'][:50]}...")

            # Create bundle
            bundle = SenseBundle(
                word=word,
                sense_id=sense_id,
                gloss=sense,
                contexts=contexts,
                hard_negatives=hard_negatives,
                embedding=sense_embeddings.get(sense),
                difficulty=sum(n.get("similarity", 0) for n in hard_negatives) / max(len(hard_negatives), 1)
            )
            bundles.append(bundle)

            # Step 4: Store to namespace
            for i, ctx in enumerate(contexts):
                store_to_namespace({
                    "text": ctx,
                    "hash": f"{sense_id}_pos_{i}",
                    "metadata": {
                        "word": word,
                        "sense_id": sense_id,
                        "gloss": sense,
                        "subtype": "positive"
                    }
                }, f"{namespace}/{word}")

            for i, neg in enumerate(hard_negatives):
                store_to_namespace({
                    "text": neg["text"],
                    "hash": f"{sense_id}_neg_{i}",
                    "metadata": {
                        "word": word,
                        "sense_id": sense_id,
                        "gloss": neg["wrong_sense"],
                        "subtype": "hard_negative",
                        "similarity": neg.get("similarity", 0)
                    }
                }, f"{namespace}/{word}")

        time.sleep(1)  # Rate limiting between words

    # Save output
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for bundle in bundles:
                f.write(json.dumps(asdict(bundle)) + "\n")
        print(f"\nSaved {len(bundles)} bundles to {output_path}")

    return bundles


def bundles_to_training(bundles: List[SenseBundle], output_path: Path):
    """Convert bundles to training JSONL format."""
    training_items = []

    # Group bundles by word for cross-sense pairs
    by_word = defaultdict(list)
    for bundle in bundles:
        by_word[bundle.word].append(bundle)

    for word, word_bundles in by_word.items():
        for bundle in word_bundles:
            # Same-sense positive pairs
            for i, ctx1 in enumerate(bundle.contexts):
                for ctx2 in bundle.contexts[i+1:]:
                    training_items.append({
                        "input_text": f"Context A: {ctx1}\nContext B: {ctx2}\n\nDo these use '{word}' in the same sense?",
                        "output_text": f"Yes, both use '{word}' in the sense of '{bundle.gloss}'. The meaning is consistent.",
                        "paradigm": "semantic_phase",
                        "subtype": "polysemy_positive",
                        "metadata": {
                            "word": word,
                            "sense": bundle.gloss,
                            "sense_id": bundle.sense_id,
                            "pair_type": "same_sense",
                            "expected_phase": "aligned"
                        }
                    })

            # Hard negative pairs
            for neg in bundle.hard_negatives:
                for ctx in bundle.contexts[:2]:
                    training_items.append({
                        "input_text": f"Context A: {ctx}\nContext B: {neg['text']}\n\nDo these use '{word}' in the same sense?",
                        "output_text": f"No, these use '{word}' in different senses: '{bundle.gloss}' vs '{neg['wrong_sense']}'.",
                        "paradigm": "semantic_phase",
                        "subtype": "polysemy_hard_negative",
                        "metadata": {
                            "word": word,
                            "sense_a": bundle.gloss,
                            "sense_b": neg["wrong_sense"],
                            "pair_type": "hard_negative",
                            "similarity": neg.get("similarity", 0),
                            "expected_phase": "misaligned"
                        }
                    })

    # Save
    with open(output_path, 'w') as f:
        for item in training_items:
            f.write(json.dumps(item) + "\n")

    print(f"Generated {len(training_items)} training pairs")
    return training_items


def main():
    parser = argparse.ArgumentParser(description="Scaled polysemy pipeline")
    parser.add_argument('--words', type=str, nargs='+', default=None,
                        help='Specific words to generate (default: all seed words)')
    parser.add_argument('--num-words', type=int, default=10,
                        help='Number of words to process (default: 10)')
    parser.add_argument('--contexts-per-sense', type=int, default=3,
                        help='Contexts per sense')
    parser.add_argument('--hard-negatives', type=int, default=2,
                        help='Hard negatives per sense')
    parser.add_argument('--output', type=str,
                        default=f'paradigm_factory/output/scaled_bundles_{datetime.now().strftime("%Y%m%d")}.jsonl')
    parser.add_argument('--training-output', type=str, default=None,
                        help='Also generate training JSONL')
    parser.add_argument('--namespace', type=str, default='polysemy_v1',
                        help='Namespace for storage')

    args = parser.parse_args()

    words = args.words if args.words else SEED_WORDS[:args.num_words]

    bundles = generate_polysemy_bundles(
        seed_words=words,
        contexts_per_sense=args.contexts_per_sense,
        hard_negatives_per_sense=args.hard_negatives,
        output_path=Path(args.output),
        namespace=args.namespace
    )

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Words processed: {len(set(b.word for b in bundles))}")
    print(f"Sense bundles: {len(bundles)}")
    print(f"Total contexts: {sum(len(b.contexts) for b in bundles)}")
    print(f"Total hard negatives: {sum(len(b.hard_negatives) for b in bundles)}")

    # Generate training format if requested
    if args.training_output:
        training_items = bundles_to_training(bundles, Path(args.training_output))
        print(f"Training pairs: {len(training_items)}")


if __name__ == "__main__":
    main()
