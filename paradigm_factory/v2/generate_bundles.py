"""
Bundle Generator
================

Generates training bundles from processed v2.1 events.
Creates contrastive pairs for semantic phase training.
"""

import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import itertools


class BundleGenerator:
    """Generates training bundles from v2.1 events."""

    def __init__(self, events: List[Dict], seed: int = 42):
        self.events = events
        self.seed = seed
        random.seed(seed)

        # Index events by lemma and sense
        self.by_lemma: Dict[str, List[Dict]] = defaultdict(list)
        self.by_sense: Dict[str, List[Dict]] = defaultdict(list)
        self.by_pos: Dict[str, List[Dict]] = defaultdict(list)

        self._index_events()

    def _index_events(self):
        """Build indices for efficient bundle generation."""
        for event in self.events:
            lemma = event.get('lemma', '').lower()
            sense_id = event.get('sense_id', '')
            pos = event.get('pos', 'noun')

            if lemma:
                self.by_lemma[lemma].append(event)
            if sense_id:
                self.by_sense[sense_id].append(event)
            if pos:
                self.by_pos[pos].append(event)

    def create_contrastive_bundle(self, anchor_event: Dict) -> Optional[Dict]:
        """Create a contrastive bundle with anchor, positive, and negatives."""
        lemma = anchor_event.get('lemma', '').lower()
        anchor_sense = anchor_event.get('sense_id', '')

        if not lemma or lemma not in self.by_lemma:
            return None

        # Get all events for this lemma
        lemma_events = self.by_lemma[lemma]

        if len(lemma_events) < 2:
            return None

        # Find positive (same sense, different text)
        same_sense = [e for e in lemma_events
                      if e.get('sense_id') == anchor_sense and e != anchor_event]

        # Find negatives (different sense, same lemma)
        diff_sense = [e for e in lemma_events
                      if e.get('sense_id') != anchor_sense]

        # Need at least one positive OR one negative to make a useful bundle
        if not same_sense and not diff_sense:
            return None

        # Select positive
        positive = random.choice(same_sense) if same_sense else None

        # Select negatives (up to 3)
        negatives = random.sample(diff_sense, min(3, len(diff_sense))) if diff_sense else []

        # Create bundle
        bundle = {
            'id': hashlib.md5(f"{anchor_event['id']}:{self.seed}".encode()).hexdigest()[:12],
            'lemma': lemma,
            'anchor': self._create_bundle_item(anchor_event),
            'positive': self._create_bundle_item(positive) if positive else None,
            'negatives': [self._create_bundle_item(n) for n in negatives],
            'metadata': {
                'anchor_sense': anchor_sense,
                'num_senses': len(set(e.get('sense_id') for e in lemma_events)),
                'total_examples': len(lemma_events),
                'bundle_type': 'contrastive'
            }
        }

        return bundle

    def create_discrimination_bundle(self, lemma: str) -> Optional[Dict]:
        """Create a sense discrimination bundle for a lemma."""
        if lemma not in self.by_lemma:
            return None

        lemma_events = self.by_lemma[lemma]

        # Group by sense
        by_sense = defaultdict(list)
        for event in lemma_events:
            sense_id = event.get('sense_id', 'unknown')
            by_sense[sense_id].append(event)

        if len(by_sense) < 2:
            return None

        # Create bundle with examples from each sense
        senses = []
        for sense_id, events in by_sense.items():
            # Sample up to 3 examples per sense
            sample = random.sample(events, min(3, len(events)))
            sense_entry = {
                'sense_id': sense_id,
                'gloss': events[0].get('sense_gloss', ''),
                'examples': [self._create_bundle_item(e) for e in sample]
            }
            senses.append(sense_entry)

        bundle = {
            'id': hashlib.md5(f"{lemma}:disc:{self.seed}".encode()).hexdigest()[:12],
            'lemma': lemma,
            'senses': senses,
            'metadata': {
                'num_senses': len(senses),
                'total_examples': sum(len(s['examples']) for s in senses),
                'bundle_type': 'discrimination'
            }
        }

        return bundle

    def _create_bundle_item(self, event: Dict) -> Dict:
        """Create a bundle item from an event."""
        return {
            'text': event.get('text', ''),
            'span': event.get('span', {}),
            'sense_id': event.get('sense_id', ''),
            'sense_gloss': event.get('sense_gloss', ''),
            'pos': event.get('pos', ''),
            'cue_tokens': event.get('cue_tokens', []),
            'source_domain': event.get('source', {}).get('domain', '') if isinstance(event.get('source'), dict) else '',
            'quality_score': event.get('quality', {}).get('cue_strength', 0.5) if isinstance(event.get('quality'), dict) else 0.5
        }

    def generate_all_bundles(self,
                             max_contrastive: int = None,
                             max_discrimination: int = None) -> Tuple[List[Dict], List[Dict]]:
        """Generate all bundles."""
        contrastive_bundles = []
        discrimination_bundles = []

        # Get polysemous lemmas (multiple senses)
        polysemous_lemmas = [
            lemma for lemma, events in self.by_lemma.items()
            if len(set(e.get('sense_id') for e in events)) >= 2
        ]

        print(f"  Polysemous lemmas: {len(polysemous_lemmas)}")

        # Generate contrastive bundles
        print("  Generating contrastive bundles...")
        all_events = list(self.events)
        random.shuffle(all_events)

        for event in all_events:
            if max_contrastive and len(contrastive_bundles) >= max_contrastive:
                break

            bundle = self.create_contrastive_bundle(event)
            if bundle:
                contrastive_bundles.append(bundle)

        # Generate discrimination bundles
        print("  Generating discrimination bundles...")
        random.shuffle(polysemous_lemmas)

        for lemma in polysemous_lemmas:
            if max_discrimination and len(discrimination_bundles) >= max_discrimination:
                break

            bundle = self.create_discrimination_bundle(lemma)
            if bundle:
                discrimination_bundles.append(bundle)

        return contrastive_bundles, discrimination_bundles


def main():
    """Main bundle generation function."""
    print("=" * 70)
    print("BUNDLE GENERATION")
    print("=" * 70)

    # Load processed events
    processed_dir = Path("paradigm_factory/v2/processed")
    events_file = processed_dir / "merged_v21.jsonl"

    print(f"\n--- Loading events from {events_file} ---")
    events = []
    with open(events_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    print(f"  Loaded: {len(events)} events")

    # Create generator
    generator = BundleGenerator(events, seed=42)

    print(f"\n--- Indexed data ---")
    print(f"  Unique lemmas: {len(generator.by_lemma)}")
    print(f"  Unique senses: {len(generator.by_sense)}")
    print(f"  POS distribution: {dict((k, len(v)) for k, v in generator.by_pos.items())}")

    # Generate bundles
    print(f"\n--- Generating bundles ---")
    contrastive, discrimination = generator.generate_all_bundles(
        max_contrastive=50000,
        max_discrimination=10000
    )

    print(f"\n  Contrastive bundles: {len(contrastive)}")
    print(f"  Discrimination bundles: {len(discrimination)}")

    # Analyze bundles
    print(f"\n--- Bundle Analysis ---")

    # Contrastive stats
    with_positive = sum(1 for b in contrastive if b.get('positive'))
    avg_negatives = sum(len(b.get('negatives', [])) for b in contrastive) / len(contrastive) if contrastive else 0

    print(f"\n  Contrastive bundles:")
    print(f"    With positive example: {with_positive} ({100*with_positive/len(contrastive):.1f}%)" if contrastive else "")
    print(f"    Average negatives per bundle: {avg_negatives:.2f}")

    # Discrimination stats
    avg_senses = sum(len(b.get('senses', [])) for b in discrimination) / len(discrimination) if discrimination else 0
    avg_examples = sum(b['metadata'].get('total_examples', 0) for b in discrimination) / len(discrimination) if discrimination else 0

    print(f"\n  Discrimination bundles:")
    print(f"    Average senses per bundle: {avg_senses:.2f}")
    print(f"    Average examples per bundle: {avg_examples:.2f}")

    # Save bundles
    output_dir = Path("paradigm_factory/v2/bundles")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Saving bundles ---")

    # Save contrastive bundles
    contrastive_file = output_dir / "contrastive_bundles.jsonl"
    with open(contrastive_file, 'w', encoding='utf-8') as f:
        for bundle in contrastive:
            f.write(json.dumps(bundle, ensure_ascii=False) + '\n')
    print(f"  Saved: {contrastive_file} ({len(contrastive)} bundles)")

    # Save discrimination bundles
    discrimination_file = output_dir / "discrimination_bundles.jsonl"
    with open(discrimination_file, 'w', encoding='utf-8') as f:
        for bundle in discrimination:
            f.write(json.dumps(bundle, ensure_ascii=False) + '\n')
    print(f"  Saved: {discrimination_file} ({len(discrimination)} bundles)")

    # Save combined for training
    combined_file = output_dir / "all_bundles.jsonl"
    all_bundles = contrastive + discrimination
    random.shuffle(all_bundles)

    with open(combined_file, 'w', encoding='utf-8') as f:
        for bundle in all_bundles:
            f.write(json.dumps(bundle, ensure_ascii=False) + '\n')
    print(f"  Saved: {combined_file} ({len(all_bundles)} bundles)")

    # Save statistics
    stats = {
        'timestamp': datetime.now().isoformat(),
        'source_events': len(events),
        'contrastive_bundles': len(contrastive),
        'discrimination_bundles': len(discrimination),
        'total_bundles': len(all_bundles),
        'unique_lemmas': len(generator.by_lemma),
        'unique_senses': len(generator.by_sense),
        'contrastive_with_positive': with_positive,
        'avg_negatives_per_bundle': avg_negatives,
        'avg_senses_per_discrimination': avg_senses
    }

    stats_file = output_dir / "bundle_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_file}")

    print("\n" + "=" * 70)
    print("BUNDLE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n  Total bundles: {len(all_bundles)}")
    print(f"  Contrastive: {len(contrastive)}")
    print(f"  Discrimination: {len(discrimination)}")
    print(f"  Output directory: {output_dir}")

    return all_bundles


if __name__ == "__main__":
    bundles = main()
