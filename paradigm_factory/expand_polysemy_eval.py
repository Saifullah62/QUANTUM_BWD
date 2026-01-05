#!/usr/bin/env python3
"""
Expand polysemy eval pack to 300+ items using Fleet swarm.
Generates hard negatives for log-prob preference testing.
"""

import requests
import json
from pathlib import Path
import time

FLEET_BASE = "http://159.203.35.45:8007"

# Polysemous words with their senses
WORDS_SENSES = {
    "bank": ["financial institution", "river edge", "to tilt aircraft"],
    "spring": ["season", "water source", "coiled metal", "to jump"],
    "bat": ["flying mammal", "sports equipment"],
    "crane": ["bird", "construction equipment", "to stretch neck"],
    "bark": ["tree covering", "dog sound", "sailing ship"],
    "light": ["illumination", "not heavy", "to ignite"],
    "rock": ["stone", "music genre", "to sway"],
    "ring": ["jewelry", "sound", "circular shape", "boxing area"],
    "run": ["to move quickly", "to operate", "a sequence"],
    "bear": ["animal", "to carry", "to endure"],
    "fair": ["carnival", "just/equitable", "light-colored"],
    "palm": ["tree", "hand part", "to conceal"],
    "date": ["calendar day", "romantic meeting", "fruit"],
    "fan": ["cooling device", "enthusiast", "to spread out"],
    "watch": ["timepiece", "to observe", "guard duty"],
    "train": ["vehicle", "to teach", "dress trail"],
    "wave": ["ocean motion", "hand gesture", "hair curl"],
    "present": ["gift", "current time", "to show"],
    "scale": ["weight device", "fish covering", "to climb"],
    "match": ["fire starter", "competition", "to correspond"],
    "seal": ["marine animal", "to close", "stamp/emblem"],
    "lead": ["metal", "to guide", "electrical wire"],
    "well": ["water source", "healthy", "adequately"],
    "bow": ["weapon", "front of ship", "to bend", "ribbon decoration"],
    "bass": ["fish", "low frequency sound", "musical instrument"],
    "check": ["verify", "payment", "pattern", "chess move"],
    "key": ["lock opener", "music scale", "crucial", "island"],
    "letter": ["alphabet character", "written message"],
    "nail": ["finger covering", "metal fastener", "to accomplish"],
    "plant": ["vegetation", "factory", "to place"],
}


def generate_contexts(word: str, sense: str, num_contexts: int = 3) -> list:
    """Generate contexts for a word in a specific sense."""
    import re

    prompt = f"""Generate {num_contexts} natural sentences using "{word}" in the sense of "{sense}".
Each sentence should clearly demonstrate this specific meaning.
Return ONLY the sentences, one per line, no numbering."""

    try:
        resp = requests.post(
            f"{FLEET_BASE}/swarm/explore",
            json={"problem": prompt, "num_agents": 2},
            timeout=45
        )
        if resp.ok:
            result = resp.json()

            # Extract from explorations list
            explorations = result.get("explorations", [])
            all_text = ""
            for exp in explorations:
                all_text += " " + str(exp.get("full_solution", ""))

            # Also check synthesis
            all_text += " " + str(result.get("synthesis", ""))

            # Find sentences containing the word
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', all_text)

            contexts = []
            word_lower = word.lower()
            for sent in sentences:
                sent = sent.strip()
                # Skip meta-commentary
                if any(skip in sent.lower() for skip in ['approach', 'strategy', 'explorer', 'solution:', 'sentence 1', 'sentence 2']):
                    continue
                # Check if word is present and sentence is reasonable length
                if word_lower in sent.lower() and 20 < len(sent) < 250:
                    # Clean up
                    sent = re.sub(r'^\*+\s*', '', sent)
                    sent = re.sub(r'\*+$', '', sent)
                    if sent and sent[0].isupper():  # Starts with capital
                        contexts.append(sent)

            # Dedupe
            seen = set()
            unique = []
            for c in contexts:
                c_clean = c.lower()[:50]
                if c_clean not in seen:
                    seen.add(c_clean)
                    unique.append(c)

            return unique[:num_contexts]
    except Exception as e:
        print(f"  Error: {e}")

    return []


def main():
    print("=" * 60)
    print("  EXPANDING POLYSEMY EVAL PACK")
    print("=" * 60)

    all_items = []

    for word, senses in WORDS_SENSES.items():
        print(f"\n{word} ({len(senses)} senses)...")

        for correct_sense in senses:
            contexts = generate_contexts(word, correct_sense, num_contexts=3)
            print(f"  {correct_sense}: {len(contexts)} contexts")

            for ctx in contexts:
                # Hard negatives: other senses of the same word
                distractors = [s for s in senses if s != correct_sense]

                item = {
                    "word": word,
                    "context_with_blank": ctx,
                    "correct_sense": correct_sense,
                    "distractor_senses": distractors[:2],
                    "difficulty": "medium",
                    "item_type": "polysemy_logprob"
                }
                all_items.append(item)

        # Rate limit
        time.sleep(0.5)

    print(f"\n\nGenerated {len(all_items)} polysemy items")

    # Load existing lindblad items
    existing_pack = Path("paradigm_factory/output/20260102/eval_pack_20260102.json")
    lindblad_items = []
    if existing_pack.exists():
        with open(existing_pack) as f:
            existing = json.load(f)
            lindblad_items = existing.get("lindblad_items", [])
            print(f"Loaded {len(lindblad_items)} existing Lindblad items")

    # Save expanded pack
    output = {
        "name": "polysemy_expanded_300",
        "polysemy_items": all_items,
        "lindblad_items": lindblad_items,
        "summary": {
            "total_polysemy": len(all_items),
            "total_lindblad": len(lindblad_items),
            "words": len(WORDS_SENSES)
        }
    }

    output_path = Path("paradigm_factory/output/polysemy_expanded.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
