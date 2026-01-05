"""
Pilot Scrape Runner
===================

Runs a pilot Wikipedia scrape to test the v2 bundle infrastructure.
"""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paradigm_factory.v2.sense_inventory.load_inventory import load_inventory
from paradigm_factory.v2.scrapers.base_scraper import (
    BaseScraper, RawUsageEvent, SourceInfo, ContextInfo, SignalInfo, QualityInfo
)
from paradigm_factory.v2.scrapers.wikipedia_scraper import WikipediaScraper


def run_pilot_scrape():
    """Run a pilot scrape with a few lemmas."""

    print("=" * 60)
    print("V2 PILOT SCRAPE")
    print("=" * 60)

    # Load inventory
    print("\n1. Loading sense inventory...")
    inventory = load_inventory()
    print(f"   Loaded {len(inventory)} lemmas")

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
        rate_limit=0.5,  # Be polite to Wikipedia
        min_context_tokens=15,
        max_context_tokens=50
    )

    # Select test lemmas
    test_lemmas = ['bank', 'spring', 'run']
    print(f"\n3. Scraping lemmas: {test_lemmas}")

    # Run scrape
    output_path = scraper.scrape_and_save(
        lemmas=test_lemmas,
        max_per_sense=20,  # Limited for pilot
        output_file="pilot_wikipedia.jsonl"
    )

    # Show results
    print("\n4. Results:")
    if output_path.exists():
        with open(output_path) as f:
            events = [json.loads(line) for line in f]

        print(f"   Total events: {len(events)}")

        # Count by sense
        sense_counts = {}
        for e in events:
            sid = e['sense_id']
            sense_counts[sid] = sense_counts.get(sid, 0) + 1

        print("\n   Events per sense:")
        for sid, count in sorted(sense_counts.items()):
            print(f"     {sid}: {count}")

        # Show a sample event
        if events:
            print("\n   Sample event:")
            sample = events[0]
            print(f"     Lemma: {sample['lemma']}")
            print(f"     Sense: {sample['sense_id']}")
            print(f"     Context: {sample['context']['sentence'][:100]}...")
            print(f"     Cue hits: {sample['signals']['cue_hits']}")
    else:
        print("   No output file generated")

    print("\n" + "=" * 60)
    print("PILOT SCRAPE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_pilot_scrape()
