#!/usr/bin/env python3
"""
Paradigm Factory - Phase A: Polysemy Candidate Generator
=========================================================

Uses Fleet swarm services to generate:
1. Polysemous word contexts across different senses
2. Hard negatives (lexically similar but wrong sense)
3. Minimal-pair edits for precise sense boundaries

Endpoints used:
- :8007/swarm/explore - Multi-agent context generation
- :8007/swarm/challenge - Hard negative refinement
- :8007/swarm/refine - Minimal-pair polishing
"""

import argparse
import json
import requests
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time

# Fleet service base URL
FLEET_BASE = "http://159.203.35.45"

# Target polysemous words with their senses
# Expanded set for richer contrastive training
DEFAULT_WORD_SENSES = {
    # Original core set
    "bank": ["financial institution", "river edge", "to tilt (aircraft)"],
    "spring": ["season", "water source", "coiled metal", "to jump"],
    "bat": ["flying mammal", "sports equipment"],
    "crane": ["bird", "construction equipment", "to stretch neck"],
    "bark": ["tree covering", "dog sound", "sailing ship"],
    "bow": ["weapon", "front of ship", "to bend", "ribbon decoration"],
    "bass": ["fish", "low frequency sound", "musical instrument"],
    "lead": ["metal", "to guide", "electrical wire"],
    "match": ["fire starter", "competition", "to correspond"],
    "seal": ["marine animal", "to close", "stamp/emblem"],
    "light": ["illumination", "not heavy", "to ignite"],
    "rock": ["stone", "music genre", "to sway"],
    "ring": ["jewelry", "sound", "circular shape", "boxing area"],
    "well": ["water source", "healthy", "adequately"],

    # Extended set - common polysemous words
    "run": ["to move quickly", "to operate", "a sequence", "in stockings"],
    "bear": ["animal", "to carry", "to endure"],
    "fair": ["carnival", "just/equitable", "light-colored", "moderately good"],
    "palm": ["tree", "hand part", "to conceal"],
    "tire": ["wheel rubber", "to exhaust", "attire/clothing"],
    "date": ["calendar day", "romantic meeting", "fruit"],
    "fan": ["cooling device", "enthusiast", "to spread out"],
    "jam": ["fruit preserve", "traffic congestion", "to force/wedge"],
    "mine": ["excavation", "belonging to me", "explosive device"],
    "port": ["harbor", "wine", "computer connection", "left side of ship"],
    "watch": ["timepiece", "to observe", "guard duty"],
    "clip": ["fastener", "video segment", "to cut", "magazine for ammo"],
    "nail": ["finger covering", "metal fastener", "to accomplish"],
    "pound": ["currency", "unit of weight", "to strike repeatedly"],
    "train": ["vehicle", "to teach", "dress trail", "series of connected things"],
    "wave": ["ocean motion", "hand gesture", "hair curl", "surge/increase"],
    "present": ["gift", "current time", "to show", "in attendance"],
    "pitcher": ["container", "baseball player"],
    "organ": ["body part", "musical instrument", "organization"],
    "bat": ["flying mammal", "sports equipment", "to flutter eyelashes"],
    "scale": ["weight device", "fish covering", "musical notes", "to climb"],
    "sentence": ["prison time", "group of words", "to condemn"],
    "bill": ["invoice", "proposed law", "bird beak", "paper money"],
    "cabinet": ["furniture", "government body", "case for display"],
    "pen": ["writing tool", "enclosure for animals", "to write"],
    "pool": ["swimming area", "game", "shared resource", "puddle"],
    "mouse": ["rodent", "computer device", "shy person"],
    "letter": ["alphabet character", "written message", "varsity award"],
    "kind": ["type/sort", "gentle/caring"],
    "board": ["wooden plank", "committee", "to enter vehicle", "meals"],
    "check": ["verify", "payment", "pattern", "chess move"],
    "coat": ["garment", "layer of paint", "animal fur"],
    "cold": ["temperature", "illness", "emotionally distant"],
    "crash": ["collision", "market decline", "to sleep", "loud noise"],
    "draft": ["preliminary version", "military selection", "air current", "beer type"],
    "drive": ["operate vehicle", "motivation", "computer storage", "path"],
    "drop": ["liquid bead", "fall", "decrease", "to release"],
    "file": ["folder", "tool", "line of people", "to submit"],
    "ground": ["earth surface", "electrical connection", "coffee state", "basis"],
    "head": ["body part", "leader", "top", "to move toward"],
    "issue": ["problem", "edition", "to distribute", "offspring"],
    "key": ["lock opener", "music scale", "crucial", "island"],
    "left": ["direction", "departed", "remaining"],
    "lie": ["recline", "untruth", "position"],
    "log": ["wood piece", "record", "to enter data"],
    "lot": ["parking area", "destiny", "large amount", "auction item"],
    "note": ["written message", "musical tone", "to observe"],
    "object": ["thing", "purpose", "to protest"],
    "order": ["sequence", "command", "religious group", "to request"],
    "park": ["green space", "to leave vehicle", "gear position"],
    "pass": ["to go by", "mountain route", "ticket", "throw"],
    "pipe": ["tube", "smoking device", "to transport", "musical instrument"],
    "plant": ["vegetation", "factory", "to place", "to establish"],
    "plot": ["story plan", "land area", "to scheme", "graph"],
    "power": ["strength", "electricity", "authority", "math exponent"],
    "press": ["news media", "to push", "printing machine", "closet"],
    "project": ["plan", "to extend", "to throw", "housing development"],
    "race": ["competition", "ethnicity", "to hurry"],
    "record": ["achievement", "audio disc", "to document"],
    "right": ["correct", "direction", "entitlement", "politically conservative"],
    "round": ["circular", "ammunition", "period of time", "to complete"],
    "second": ["time unit", "ordinal number", "to support", "inferior"],
    "set": ["collection", "to place", "solid", "stage background"],
    "shot": ["bullet", "photograph", "attempt", "injection"],
    "sign": ["symbol", "to write name", "indication", "zodiac"],
    "sound": ["noise", "healthy", "body of water", "to measure depth"],
    "space": ["area", "outer space", "gap", "to arrange"],
    "spot": ["location", "stain", "to notice", "advertising slot"],
    "stage": ["platform", "phase", "theater", "to organize"],
    "stand": ["to be upright", "booth", "position", "to tolerate"],
    "star": ["celestial body", "celebrity", "rating symbol", "to feature"],
    "state": ["condition", "government region", "to declare"],
    "stick": ["branch", "to adhere", "hockey equipment", "remote area"],
    "store": ["shop", "to keep", "supply"],
    "strike": ["to hit", "labor action", "bowling score", "military attack"],
    "study": ["to learn", "research", "room", "to examine"],
    "suit": ["clothing", "lawsuit", "card type", "to be appropriate"],
    "swing": ["playground equipment", "to move", "music style", "attempt"],
    "table": ["furniture", "to postpone", "data grid"],
    "tank": ["container", "military vehicle", "to fail"],
    "tear": ["rip", "eye drop"],
    "throw": ["to toss", "blanket", "pottery process"],
    "tie": ["neckwear", "to fasten", "equal score", "connection"],
    "tip": ["end point", "gratuity", "advice", "to tilt"],
    "track": ["path", "to follow", "music recording", "athletic field"],
    "trip": ["journey", "to stumble", "drug experience"],
    "trunk": ["tree part", "elephant nose", "luggage storage", "torso"],
    "type": ["category", "to keyboard", "printing font"],
    "volume": ["book", "loudness", "space amount", "series"],
    "yard": ["unit of length", "outdoor area", "ship pole"],
}


@dataclass
class PolysemyExample:
    """A single polysemy training example."""
    word: str
    sense: str
    context: str
    paradigm: str = "semantic_phase"
    subtype: str = "positive"  # positive, negative, hard_negative
    text_hash: str = ""
    difficulty: float = 0.0  # Set later by embedding analysis

    def __post_init__(self):
        if not self.text_hash:
            self.text_hash = hashlib.md5(self.context.encode()).hexdigest()[:12]


def call_swarm(endpoint: str, payload: Dict, timeout: int = 60) -> Optional[Dict]:
    """Call a Fleet swarm endpoint."""
    url = f"{FLEET_BASE}:8007/swarm/{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  Warning: Swarm call failed: {e}")
        return None


def generate_sense_contexts(word: str, sense: str, num_contexts: int = 5) -> List[str]:
    """
    Generate diverse contexts for a word in a specific sense.
    Uses swarm/explore for multi-agent diversity.
    """
    prompt = f"""Generate {num_contexts} diverse, natural sentences that use the word "{word}"
in the sense of "{sense}". Each sentence should:
1. Clearly demonstrate this specific meaning
2. Be grammatically correct and natural
3. Vary in structure and context
4. Be 10-30 words long

Return ONLY the sentences, one per line, no numbering."""

    result = call_swarm("explore", {
        "problem": prompt,
        "num_agents": 3
    })

    if not result:
        return []

    # Parse response - agents return in different formats
    contexts = []
    response_text = str(result.get("result", result.get("explorations", "")))

    for line in response_text.split("\n"):
        line = line.strip()
        # Skip empty lines and common prefixes
        if line and not line.startswith(("1.", "2.", "3.", "-", "*", "Agent")):
            # Remove quotes if present
            line = line.strip('"\'')
            if word.lower() in line.lower() and len(line) > 20:
                contexts.append(line)

    return contexts[:num_contexts]


def generate_hard_negatives(word: str, correct_sense: str, wrong_sense: str,
                            positive_context: str) -> List[str]:
    """
    Generate hard negatives: sentences that look similar but use wrong sense.
    Uses swarm/challenge to stress-test and find edge cases.
    """
    prompt = f"""Given this sentence using "{word}" in the sense of "{correct_sense}":
"{positive_context}"

Generate a similar-looking sentence that instead uses "{word}" in the sense of "{wrong_sense}".
The sentence should:
1. Be structurally similar to the original
2. Be natural and grammatically correct
3. Clearly use the different sense
4. Be a "hard negative" - easily confused with the original

Return ONLY the new sentence."""

    result = call_swarm("challenge", {
        "problem": f"Create hard negative for polysemy disambiguation",
        "solution": positive_context,
        "intensity": "moderate"
    })

    if not result:
        return []

    # Also try refine for minimal-pair generation
    refine_result = call_swarm("refine", {
        "task": prompt,
        "iterations": 1
    })

    negatives = []
    for r in [result, refine_result]:
        if r:
            text = str(r.get("result", r.get("refined_output", "")))
            for line in text.split("\n"):
                line = line.strip().strip('"\'')
                if line and word.lower() in line.lower() and len(line) > 15:
                    negatives.append(line)

    return negatives[:2]


def generate_polysemy_batch(
    word_senses: Dict[str, List[str]],
    contexts_per_sense: int = 3,
    generate_negatives: bool = True,
    output_path: Optional[Path] = None
) -> List[PolysemyExample]:
    """
    Generate a batch of polysemy examples for all words/senses.
    """
    examples = []

    for word, senses in word_senses.items():
        print(f"\nGenerating for '{word}' ({len(senses)} senses)...")

        word_examples = []

        for sense in senses:
            print(f"  Sense: {sense}")

            # Generate positive contexts
            contexts = generate_sense_contexts(word, sense, contexts_per_sense)
            print(f"    Generated {len(contexts)} positive contexts")

            for ctx in contexts:
                ex = PolysemyExample(
                    word=word,
                    sense=sense,
                    context=ctx,
                    subtype="positive"
                )
                word_examples.append(ex)

            # Generate hard negatives (using other senses)
            if generate_negatives and len(contexts) > 0 and len(senses) > 1:
                other_senses = [s for s in senses if s != sense]
                for other_sense in other_senses[:1]:  # Limit to 1 other sense
                    for ctx in contexts[:1]:  # Limit negatives per positive
                        negatives = generate_hard_negatives(
                            word, sense, other_sense, ctx
                        )
                        for neg_ctx in negatives:
                            neg_ex = PolysemyExample(
                                word=word,
                                sense=other_sense,  # The sense it actually uses
                                context=neg_ctx,
                                subtype="hard_negative"
                            )
                            word_examples.append(neg_ex)

                print(f"    Generated {len([e for e in word_examples if e.subtype == 'hard_negative'])} hard negatives")

        examples.extend(word_examples)

        # Rate limiting - be nice to the swarm
        time.sleep(1)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex)) + "\n")
        print(f"\nSaved {len(examples)} examples to {output_path}")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate polysemy training data")
    parser.add_argument('--words', type=str, nargs='+', default=None,
                        help='Specific words to generate (default: all)')
    parser.add_argument('--contexts-per-sense', type=int, default=3,
                        help='Number of positive contexts per sense')
    parser.add_argument('--no-negatives', action='store_true',
                        help='Skip hard negative generation')
    parser.add_argument('--output', type=str,
                        default=f'paradigm_factory/output/polysemy_{datetime.now().strftime("%Y%m%d")}.jsonl',
                        help='Output JSONL path')

    args = parser.parse_args()

    # Filter words if specified
    word_senses = DEFAULT_WORD_SENSES
    if args.words:
        word_senses = {w: s for w, s in DEFAULT_WORD_SENSES.items() if w in args.words}

    print("=" * 60)
    print("  PARADIGM FACTORY - POLYSEMY GENERATOR")
    print("=" * 60)
    print(f"Words: {list(word_senses.keys())}")
    print(f"Contexts per sense: {args.contexts_per_sense}")
    print(f"Generate negatives: {not args.no_negatives}")
    print(f"Output: {args.output}")

    examples = generate_polysemy_batch(
        word_senses=word_senses,
        contexts_per_sense=args.contexts_per_sense,
        generate_negatives=not args.no_negatives,
        output_path=Path(args.output)
    )

    # Summary
    print("\n" + "=" * 60)
    print("  GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total examples: {len(examples)}")
    print(f"Positives: {len([e for e in examples if e.subtype == 'positive'])}")
    print(f"Hard negatives: {len([e for e in examples if e.subtype == 'hard_negative'])}")
    print(f"Unique words: {len(set(e.word for e in examples))}")


if __name__ == "__main__":
    main()
