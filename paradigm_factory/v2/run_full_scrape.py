"""
Full Scrape Runner
==================

Runs a comprehensive scrape on all lemmas in the sense inventory.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paradigm_factory.v2.sense_inventory.load_inventory import load_inventory
from paradigm_factory.v2.scrapers.wikipedia_scraper import WikipediaScraper


def run_full_scrape():
    """Run a comprehensive scrape on all lemmas."""

    print("=" * 60)
    print("V2 FULL SCRAPE")
    print("=" * 60)

    # Load inventory
    print("\n1. Loading sense inventory...")
    inventory = load_inventory()
    print(f"   Loaded {len(inventory)} lemmas")

    # Get all lemmas
    lemmas = [inv.lemma for inv in inventory.values()]
    print(f"   Lemmas: {', '.join(lemmas)}")

    # Convert to dict format expected by scraper
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

    # Create scraper
    print("\n2. Creating Wikipedia scraper...")
    output_dir = Path(__file__).parent / "raw_events"
    output_dir.mkdir(parents=True, exist_ok=True)

    scraper = WikipediaScraper(
        sense_inventory=inv_dict,
        output_dir=output_dir,
        rate_limit=0.3,  # Faster for full scrape
        min_context_tokens=15,
        max_context_tokens=50
    )

    # Run scrape
    print(f"\n3. Scraping all {len(lemmas)} lemmas...")
    output_path = scraper.scrape_and_save(
        lemmas=lemmas,
        max_per_sense=50,  # More events per sense for diversity
        output_file="full_wikipedia.jsonl"
    )

    # Show results
    print("\n4. Results:")
    if output_path.exists():
        with open(output_path) as f:
            events = [json.loads(line) for line in f]

        print(f"   Total events: {len(events)}")

        # Count by lemma
        lemma_counts = defaultdict(int)
        for e in events:
            lemma_counts[e['lemma']] += 1

        print("\n   Events per lemma:")
        for lemma, count in sorted(lemma_counts.items()):
            print(f"     {lemma}: {count}")

        # Count by sense
        sense_counts = defaultdict(int)
        for e in events:
            sense_counts[e['sense_id']] += 1

        print("\n   Events per sense:")
        for sid, count in sorted(sense_counts.items()):
            print(f"     {sid}: {count}")

        # Style distribution
        styles = defaultdict(int)
        for e in events:
            styles[e['quality']['style']] += 1

        print("\n   Style distribution:")
        for style, count in sorted(styles.items(), key=lambda x: -x[1]):
            print(f"     {style}: {count}")

    print("\n" + "=" * 60)
    print("FULL SCRAPE COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    run_full_scrape()
