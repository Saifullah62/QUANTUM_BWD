#!/usr/bin/env python3
"""
Paradigm Factory - Phase C: Lindblad Noise Twin Generator
==========================================================

Generates realistic noise twins for Lindblad invariance training:
1. Semantic-preserving corruptions that mimic real-world input degradation
2. Scored by embedding similarity to ensure meaning is preserved
3. Filtered to remove twins that drift too far semantically

Corruption types:
- Punctuation removal/variation
- Casing drift
- Repeated phrases
- Minor word reordering
- Filler insertion
- Tail truncation
- Keyboard-neighbor typos
- OCR-style errors
"""

import argparse
import json
import random
import re
import string
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict

# Fleet service base URL
FLEET_BASE = "http://159.203.35.45"

# Minimum similarity for a valid twin (meaning preserved)
MIN_SIMILARITY_THRESHOLD = 0.8

# Keyboard neighbor map for typo generation
KEYBOARD_NEIGHBORS = {
    'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfxc',
    'e': 'rdsw', 'f': 'rtgdcv', 'g': 'tyhfvb', 'h': 'yujgbn',
    'i': 'uojk', 'j': 'uikhm', 'k': 'ioljm', 'l': 'opk',
    'm': 'njk', 'n': 'bhjm', 'o': 'iplk', 'p': 'ol',
    'q': 'wa', 'r': 'etdf', 's': 'wedxza', 't': 'ryfg',
    'u': 'yihj', 'v': 'cfgb', 'w': 'qeas', 'x': 'zsdc',
    'y': 'tugh', 'z': 'asx'
}


@dataclass
class LindbladTwin:
    """A Lindblad noise twin example."""
    original_text: str
    noisy_text: str
    corruption_type: str
    corruption_params: Dict
    paradigm: str = "lindblad"
    similarity_score: float = 0.0  # Set by embedding check
    text_hash: str = ""

    def __post_init__(self):
        if not self.text_hash:
            combined = self.original_text + self.noisy_text
            self.text_hash = hashlib.md5(combined.encode()).hexdigest()[:12]


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
    except requests.RequestException:
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity."""
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# =============================================================================
# CORRUPTION FUNCTIONS
# =============================================================================

def remove_punctuation(text: str, prob: float = 0.5) -> Tuple[str, Dict]:
    """Remove some or all punctuation."""
    result = []
    removed = 0
    for char in text:
        if char in string.punctuation and random.random() < prob:
            removed += 1
            continue
        result.append(char)
    return ''.join(result), {"removed_count": removed, "prob": prob}


def case_drift(text: str, mode: str = "random") -> Tuple[str, Dict]:
    """Apply casing changes."""
    if mode == "lower":
        return text.lower(), {"mode": "lower"}
    elif mode == "upper":
        return text.upper(), {"mode": "upper"}
    elif mode == "random":
        result = []
        for char in text:
            if random.random() < 0.1:
                char = char.swapcase()
            result.append(char)
        return ''.join(result), {"mode": "random"}
    elif mode == "title":
        return text.title(), {"mode": "title"}
    return text, {"mode": "none"}


def repeat_phrase(text: str) -> Tuple[str, Dict]:
    """Randomly repeat a phrase (common in speech/transcription)."""
    words = text.split()
    if len(words) < 4:
        return text, {"repeated": False}

    # Pick a random phrase of 2-4 words to repeat
    phrase_len = min(random.randint(2, 4), len(words) - 1)
    start_idx = random.randint(0, len(words) - phrase_len)
    phrase = words[start_idx:start_idx + phrase_len]

    # Insert the repeat
    insert_idx = start_idx + phrase_len
    words = words[:insert_idx] + phrase + words[insert_idx:]

    return ' '.join(words), {"repeated": True, "phrase_len": phrase_len}


def minor_reorder(text: str) -> Tuple[str, Dict]:
    """Swap adjacent words (common in speech/informal text)."""
    words = text.split()
    if len(words) < 3:
        return text, {"swapped": False}

    # Pick random adjacent pair to swap
    idx = random.randint(0, len(words) - 2)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]

    return ' '.join(words), {"swapped": True, "position": idx}


def insert_filler(text: str) -> Tuple[str, Dict]:
    """Insert common filler words (um, uh, like, you know)."""
    fillers = ["um", "uh", "like", "you know", "basically", "actually", "I mean"]
    words = text.split()
    if len(words) < 3:
        return text, {"inserted": False}

    filler = random.choice(fillers)
    insert_idx = random.randint(1, len(words) - 1)
    words.insert(insert_idx, filler)

    return ' '.join(words), {"inserted": True, "filler": filler, "position": insert_idx}


def truncate_tail(text: str, ratio: float = 0.2) -> Tuple[str, Dict]:
    """Truncate the end of the text."""
    words = text.split()
    keep = max(int(len(words) * (1 - ratio)), 3)
    truncated = words[:keep]
    return ' '.join(truncated), {"truncated_words": len(words) - keep, "ratio": ratio}


def keyboard_typo(text: str, prob: float = 0.05) -> Tuple[str, Dict]:
    """Introduce keyboard neighbor typos."""
    result = []
    typos = 0
    for char in text:
        if char.lower() in KEYBOARD_NEIGHBORS and random.random() < prob:
            neighbors = KEYBOARD_NEIGHBORS[char.lower()]
            typo_char = random.choice(neighbors)
            if char.isupper():
                typo_char = typo_char.upper()
            result.append(typo_char)
            typos += 1
        else:
            result.append(char)
    return ''.join(result), {"typo_count": typos, "prob": prob}


def ocr_errors(text: str, prob: float = 0.03) -> Tuple[str, Dict]:
    """Introduce OCR-style character confusions."""
    ocr_confusions = {
        'l': '1I|', 'I': '1l|', '1': 'lI|',
        'O': '0Q', '0': 'OQ', 'o': '0',
        'S': '5$', '5': 'S$',
        'rn': 'm', 'm': 'rn',
        'cl': 'd', 'd': 'cl',
        'B': '8', '8': 'B',
    }
    result = text
    errors = 0
    for orig, replacements in ocr_confusions.items():
        if orig in result and random.random() < prob:
            replacement = random.choice(replacements)
            result = result.replace(orig, replacement, 1)
            errors += 1
    return result, {"error_count": errors, "prob": prob}


def whitespace_noise(text: str) -> Tuple[str, Dict]:
    """Add/remove whitespace irregularly."""
    # Multiple spaces
    text = re.sub(r' ', lambda m: ' ' * random.randint(1, 3) if random.random() < 0.2 else ' ', text)
    # Occasional missing spaces
    text = re.sub(r' ', lambda m: '' if random.random() < 0.05 else ' ', text)
    return text.strip(), {"applied": True}


# All corruption functions
CORRUPTION_FUNCTIONS: Dict[str, Callable] = {
    "punctuation_removal": lambda t: remove_punctuation(t, prob=random.uniform(0.3, 0.8)),
    "case_drift": lambda t: case_drift(t, mode=random.choice(["lower", "random", "title"])),
    "repeat_phrase": repeat_phrase,
    "minor_reorder": minor_reorder,
    "insert_filler": insert_filler,
    "truncate_tail": lambda t: truncate_tail(t, ratio=random.uniform(0.1, 0.3)),
    "keyboard_typo": lambda t: keyboard_typo(t, prob=random.uniform(0.03, 0.08)),
    "ocr_errors": lambda t: ocr_errors(t, prob=random.uniform(0.02, 0.05)),
    "whitespace_noise": whitespace_noise,
}


def apply_corruption(text: str, corruption_type: Optional[str] = None) -> LindbladTwin:
    """Apply a corruption to text and return a twin."""
    if corruption_type is None:
        corruption_type = random.choice(list(CORRUPTION_FUNCTIONS.keys()))

    func = CORRUPTION_FUNCTIONS[corruption_type]
    noisy_text, params = func(text)

    return LindbladTwin(
        original_text=text,
        noisy_text=noisy_text,
        corruption_type=corruption_type,
        corruption_params=params
    )


def apply_compound_corruption(text: str, num_corruptions: int = 2) -> LindbladTwin:
    """Apply multiple corruptions in sequence."""
    corruption_types = random.sample(list(CORRUPTION_FUNCTIONS.keys()), num_corruptions)
    current_text = text
    all_params = {}

    for ctype in corruption_types:
        func = CORRUPTION_FUNCTIONS[ctype]
        current_text, params = func(current_text)
        all_params[ctype] = params

    return LindbladTwin(
        original_text=text,
        noisy_text=current_text,
        corruption_type="compound",
        corruption_params={"types": corruption_types, "details": all_params}
    )


def score_twin_similarity(twin: LindbladTwin, use_fleet: bool = True) -> float:
    """Score how well the twin preserves meaning."""
    if use_fleet:
        emb_orig = get_embedding(twin.original_text)
        emb_noisy = get_embedding(twin.noisy_text)
        if emb_orig and emb_noisy:
            return cosine_similarity(emb_orig, emb_noisy)

    # Fallback: simple text overlap
    orig_words = set(twin.original_text.lower().split())
    noisy_words = set(twin.noisy_text.lower().split())
    if not orig_words:
        return 0.0
    return len(orig_words & noisy_words) / len(orig_words)


def generate_lindblad_batch(
    source_texts: List[str],
    twins_per_text: int = 3,
    use_compound: bool = True,
    score_with_embeddings: bool = True,
    min_similarity: float = MIN_SIMILARITY_THRESHOLD,
    output_path: Optional[Path] = None,
    seed: int = 42
) -> List[LindbladTwin]:
    """
    Generate Lindblad twins for a batch of source texts.
    """
    random.seed(seed)
    twins = []

    for i, text in enumerate(source_texts):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(source_texts)}")

        text_twins = []

        # Generate single-corruption twins
        for _ in range(twins_per_text // 2 + 1):
            twin = apply_corruption(text)
            text_twins.append(twin)

        # Generate compound-corruption twins
        if use_compound:
            for _ in range(twins_per_text // 2):
                twin = apply_compound_corruption(text, num_corruptions=2)
                text_twins.append(twin)

        # Score and filter
        valid_twins = []
        for twin in text_twins:
            if twin.noisy_text.strip() and twin.noisy_text != twin.original_text:
                twin.similarity_score = score_twin_similarity(twin, use_fleet=score_with_embeddings)
                if twin.similarity_score >= min_similarity:
                    valid_twins.append(twin)

        twins.extend(valid_twins[:twins_per_text])

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for twin in twins:
                f.write(json.dumps(asdict(twin)) + "\n")
        print(f"\nSaved {len(twins)} twins to {output_path}")

    return twins


def main():
    parser = argparse.ArgumentParser(description="Generate Lindblad noise twins")
    parser.add_argument('--input', type=str, required=True,
                        help='Input file with source texts (one per line or JSONL)')
    parser.add_argument('--output', type=str,
                        default=f'paradigm_factory/output/lindblad_twins_{datetime.now().strftime("%Y%m%d")}.jsonl',
                        help='Output JSONL path')
    parser.add_argument('--twins-per-text', type=int, default=3,
                        help='Number of twins per source text')
    parser.add_argument('--no-compound', action='store_true',
                        help='Skip compound corruptions')
    parser.add_argument('--no-embedding-score', action='store_true',
                        help='Skip embedding-based similarity scoring')
    parser.add_argument('--min-similarity', type=float, default=0.8,
                        help='Minimum similarity threshold')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Load source texts
    source_texts = []
    input_path = Path(args.input)

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try to parse as JSON, else use as plain text
            try:
                data = json.loads(line)
                text = data.get("context", data.get("text", line))
            except json.JSONDecodeError:
                text = line
            source_texts.append(text)

    print("=" * 60)
    print("  PARADIGM FACTORY - LINDBLAD TWIN GENERATOR")
    print("=" * 60)
    print(f"Source texts: {len(source_texts)}")
    print(f"Twins per text: {args.twins_per_text}")
    print(f"Compound corruptions: {not args.no_compound}")
    print(f"Embedding scoring: {not args.no_embedding_score}")
    print(f"Min similarity: {args.min_similarity}")

    twins = generate_lindblad_batch(
        source_texts=source_texts,
        twins_per_text=args.twins_per_text,
        use_compound=not args.no_compound,
        score_with_embeddings=not args.no_embedding_score,
        min_similarity=args.min_similarity,
        output_path=Path(args.output),
        seed=args.seed
    )

    # Summary
    print("\n" + "=" * 60)
    print("  GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total twins: {len(twins)}")

    # By corruption type
    from collections import Counter
    type_counts = Counter(t.corruption_type for t in twins)
    print("\nBy corruption type:")
    for ctype, count in type_counts.most_common():
        print(f"  {ctype}: {count}")

    # Similarity distribution
    sims = [t.similarity_score for t in twins]
    if sims:
        import numpy as np
        print(f"\nSimilarity scores:")
        print(f"  Mean: {np.mean(sims):.3f}")
        print(f"  Min:  {np.min(sims):.3f}")
        print(f"  Max:  {np.max(sims):.3f}")


if __name__ == "__main__":
    main()
