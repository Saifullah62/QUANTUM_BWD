"""
Bundle Generator Runner
=======================

Generates v2 bundles from scraped raw events.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paradigm_factory.v2.sense_inventory.load_inventory import load_inventory
from paradigm_factory.v2.bundle_generator import BundleGenerator


def run_bundle_generator():
    """Generate v2 bundles from pilot scrape data."""

    print("=" * 60)
    print("V2 BUNDLE GENERATOR")
    print("=" * 60)

    # Load sense inventory
    print("\n1. Loading sense inventory...")
    inventory = load_inventory()
    print(f"   Loaded {len(inventory)} lemmas")

    # Convert to dict format
    inv_dict = {}
    for key, lemma_inv in inventory.items():
        inv_dict[key] = {
            'lemma': lemma_inv.lemma,
            'pos': lemma_inv.pos,
            'senses': [
                {
                    'sense_id': s.sense_id,
                    'gloss': s.gloss,
                    'domain': s.domain,
                    'cues': s.cues,
                    'anti_cues': s.anti_cues,
                    'pos': s.pos
                }
                for s in lemma_inv.senses
            ]
        }

    # Check for raw events
    events_path = Path(__file__).parent / "raw_events" / "pilot_wikipedia.jsonl"
    if not events_path.exists():
        print(f"   ERROR: Raw events not found at {events_path}")
        print("   Run pilot scrape first: python run_pilot_scrape.py")
        return

    # Create bundle generator
    print("\n2. Creating bundle generator...")
    output_dir = Path(__file__).parent / "bundles"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = BundleGenerator(
        sense_inventory=inv_dict,
        output_dir=output_dir
    )

    # Generate bundles
    print("\n3. Generating bundles...")
    bundles = generator.generate_bundles(
        events_path=events_path,
        max_bundles_per_lemma=30  # Target 30 per lemma for pilot
    )

    # Save bundles
    print("\n4. Saving bundles...")
    output_path = generator.save_bundles(bundles, "pilot_bundles_v2.jsonl")

    # Print statistics
    print("\n" + "=" * 60)
    print("BUNDLE GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal bundles: {len(bundles)}")

    if bundles:
        # Bundles per lemma
        lemma_counts = defaultdict(int)
        for b in bundles:
            lemma_counts[b.word['lemma']] += 1

        print("\nBundles per lemma:")
        for lemma, count in sorted(lemma_counts.items()):
            print(f"  {lemma}: {count}")

        # Average items per bundle
        avg_items = sum(len(b.items) for b in bundles) / len(bundles)
        print(f"\nAvg items per bundle: {avg_items:.1f}")

        # Hard negative stats
        hard_neg_counts = defaultdict(int)
        for b in bundles:
            for item in b.items:
                if item.negative_type:
                    hard_neg_counts[item.negative_type] += 1

        print("\nNegative type distribution:")
        for neg_type, count in sorted(hard_neg_counts.items()):
            print(f"  {neg_type}: {count}")

        # Style distribution
        styles = defaultdict(int)
        for b in bundles:
            for item in b.items:
                styles[item.style] += 1

        print("\nStyle distribution:")
        for style, count in sorted(styles.items(), key=lambda x: -x[1]):
            print(f"  {style}: {count}")

        # Show sample bundle
        print("\n" + "-" * 40)
        print("Sample bundle:")
        print("-" * 40)
        sample = bundles[0]
        print(f"Bundle ID: {sample.bundle_id}")
        print(f"Lemma: {sample.word['lemma']}")
        print(f"Items: {len(sample.items)}")
        for item in sample.items:
            print(f"  - [{item.role}] {item.sense_id}")
            print(f"    Context: {item.context[:80]}...")
            if item.negative_type:
                print(f"    Type: {item.negative_type}, Overlap: {item.lexical_overlap:.2f}")

    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    run_bundle_generator()
