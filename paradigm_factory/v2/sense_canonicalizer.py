"""
Sense ID Canonicalizer
======================

Enforces a "sense identity contract" across multiple source taxonomies.
Prevents namespace collisions between WordNet, Wiktionary, Wikipedia, and custom sources.

Schema: {source}:{lemma}.{pos}.{sense_num}
Examples:
  - wordnet:bank.noun.01
  - wiktionary:bank.noun.3
  - wikipedia:bank#financial_institution
  - custom:bank#legal
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CanonicalSense:
    """A canonicalized sense with source provenance."""
    canonical_id: str          # Full namespaced ID
    source: str                # Source taxonomy (wordnet, wiktionary, wikipedia, custom)
    lemma: str
    pos: str
    sense_key: str             # Original sense identifier from source
    gloss: str
    related_ids: List[str] = field(default_factory=list)  # Cross-references


class SenseCanonicalizer:
    """Canonicalizes sense IDs across multiple taxonomies."""

    # Source prefixes
    SOURCES = {
        'wordnet': 'wn',
        'wiktionary': 'wk',
        'wikipedia': 'wp',
        'custom': 'cx',
        'unknown': 'uk'
    }

    # POS normalization
    POS_MAP = {
        'noun': 'n',
        'verb': 'v',
        'adjective': 'adj',
        'adverb': 'adv',
        'preposition': 'prep',
        'conjunction': 'conj',
        'pronoun': 'pron',
        'interjection': 'intj',
        'n': 'n',
        'v': 'v',
        'adj': 'adj',
        'adv': 'adv',
        'a': 'adj',
        's': 'adj',  # WordNet satellite adjective
        'r': 'adv',  # WordNet adverb
    }

    def __init__(self):
        self.sense_registry: Dict[str, CanonicalSense] = {}
        self.collision_log: List[Dict] = []
        self.alias_map: Dict[str, str] = {}  # old_id -> canonical_id

    def detect_source(self, event: Dict) -> str:
        """Detect the source taxonomy from an event."""
        source_info = event.get('source', {})
        if isinstance(source_info, dict):
            domain = source_info.get('domain', '').lower()
            url = source_info.get('url', '').lower()
        else:
            domain = ''
            url = ''

        sense_id = event.get('sense_id', '').lower()

        # WordNet patterns
        if 'wordnet' in domain or 'wordnet' in url:
            return 'wordnet'
        if re.match(r'^[a-z_]+\.[nvars]\.\d+$', sense_id):
            return 'wordnet'

        # Wiktionary patterns
        if 'wiktionary' in domain:
            return 'wiktionary'
        if re.match(r'^[a-z]+\.(noun|verb|adjective|adverb)\.\d+$', sense_id):
            return 'wiktionary'

        # Wikipedia patterns
        if 'wikipedia' in domain or 'en.wikipedia' in url:
            return 'wikipedia'
        if '#' in sense_id and not sense_id.startswith(('wn:', 'wk:', 'wp:', 'cx:')):
            return 'wikipedia'

        # Custom/scraped
        if any(x in domain for x in ['uscourts', 'consumerfinance', 'rcog', 'dfpi']):
            return 'custom'

        return 'unknown'

    def normalize_pos(self, pos: str) -> str:
        """Normalize POS tag to standard form."""
        return self.POS_MAP.get(pos.lower(), 'n')

    def extract_sense_key(self, sense_id: str, source: str) -> str:
        """Extract the sense key from various formats."""
        # Already namespaced
        if ':' in sense_id and sense_id.split(':')[0] in self.SOURCES.values():
            return sense_id.split(':', 1)[1]

        # WordNet format: lemma.pos.num
        if source == 'wordnet':
            match = re.match(r'^([a-z_]+)\.([nvars])\.(\d+)$', sense_id.lower())
            if match:
                return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"

        # Wiktionary format: lemma.pos.num
        if source == 'wiktionary':
            match = re.match(r'^([a-z]+)\.(noun|verb|adjective|adverb)\.(\d+)$', sense_id.lower())
            if match:
                pos_short = self.normalize_pos(match.group(2))
                return f"{match.group(1)}.{pos_short}.{match.group(3)}"

        # Wikipedia/custom format: lemma#context
        if '#' in sense_id:
            return sense_id.lower().replace(' ', '_')

        # Fallback: hash-based key
        return hashlib.md5(sense_id.encode()).hexdigest()[:8]

    def canonicalize(self, event: Dict) -> str:
        """Generate canonical sense ID for an event."""
        source = self.detect_source(event)
        source_prefix = self.SOURCES.get(source, 'uk')

        lemma = event.get('lemma', '').lower().replace(' ', '_')
        pos = self.normalize_pos(event.get('pos', 'noun'))
        original_sense_id = event.get('sense_id', '')

        sense_key = self.extract_sense_key(original_sense_id, source)

        # Build canonical ID
        canonical_id = f"{source_prefix}:{lemma}.{pos}.{sense_key}"

        # Clean up any double dots or invalid chars
        canonical_id = re.sub(r'\.+', '.', canonical_id)
        canonical_id = re.sub(r'[^a-z0-9:._#-]', '', canonical_id)

        # Track alias
        if original_sense_id and original_sense_id != canonical_id:
            self.alias_map[original_sense_id] = canonical_id

        return canonical_id

    def process_event(self, event: Dict) -> Dict:
        """Process an event and add canonical sense ID."""
        canonical_id = self.canonicalize(event)

        # Store original for debugging
        event['_original_sense_id'] = event.get('sense_id', '')
        event['sense_id'] = canonical_id

        # Add source prefix to related fields
        source = self.detect_source(event)
        event['_sense_source'] = source

        return event

    def detect_collisions(self, events: List[Dict]) -> List[Dict]:
        """Detect potential sense ID collisions."""
        collisions = []

        # Group by canonical ID
        by_canonical: Dict[str, List[Dict]] = defaultdict(list)
        for event in events:
            canonical_id = event.get('sense_id', '')
            by_canonical[canonical_id].append(event)

        # Check for collisions (same ID, different glosses)
        for canonical_id, group in by_canonical.items():
            glosses = set(e.get('sense_gloss', '')[:100] for e in group)
            if len(glosses) > 1:
                collisions.append({
                    'canonical_id': canonical_id,
                    'count': len(group),
                    'glosses': list(glosses)[:5],
                    'sources': list(set(e.get('_sense_source', '') for e in group))
                })

        self.collision_log = collisions
        return collisions

    def build_sense_inventory(self, events: List[Dict]) -> Dict[str, List[CanonicalSense]]:
        """Build a sense inventory grouped by lemma."""
        inventory: Dict[str, List[CanonicalSense]] = defaultdict(list)

        seen_senses: Set[str] = set()

        for event in events:
            canonical_id = event.get('sense_id', '')
            if canonical_id in seen_senses:
                continue

            seen_senses.add(canonical_id)

            sense = CanonicalSense(
                canonical_id=canonical_id,
                source=event.get('_sense_source', 'unknown'),
                lemma=event.get('lemma', ''),
                pos=event.get('pos', 'noun'),
                sense_key=event.get('_original_sense_id', ''),
                gloss=event.get('sense_gloss', '')
            )

            inventory[sense.lemma.lower()].append(sense)

        return dict(inventory)


def canonicalize_dataset(input_file: Path, output_file: Path) -> Dict:
    """Canonicalize all sense IDs in a dataset."""
    canonicalizer = SenseCanonicalizer()

    print(f"Loading {input_file}...")
    events = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    print(f"  Loaded {len(events)} events")

    # Process all events
    print("Canonicalizing sense IDs...")
    processed = []
    source_counts = defaultdict(int)

    for event in events:
        processed_event = canonicalizer.process_event(event)
        processed.append(processed_event)
        source_counts[processed_event.get('_sense_source', 'unknown')] += 1

    print(f"  Source distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {source}: {count}")

    # Detect collisions
    print("\nDetecting collisions...")
    collisions = canonicalizer.detect_collisions(processed)
    print(f"  Found {len(collisions)} potential collision groups")

    if collisions:
        print("  Top collision examples:")
        for c in collisions[:5]:
            print(f"    {c['canonical_id']}: {c['count']} events, {len(c['glosses'])} glosses")

    # Build inventory
    print("\nBuilding sense inventory...")
    inventory = canonicalizer.build_sense_inventory(processed)
    total_senses = sum(len(senses) for senses in inventory.values())
    print(f"  {len(inventory)} lemmas, {total_senses} unique senses")

    # Save processed events
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for event in processed:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')

    # Save inventory
    inventory_file = output_file.parent / "sense_inventory.json"
    inventory_data = {
        lemma: [
            {
                'canonical_id': s.canonical_id,
                'source': s.source,
                'pos': s.pos,
                'gloss': s.gloss
            }
            for s in senses
        ]
        for lemma, senses in inventory.items()
    }

    with open(inventory_file, 'w', encoding='utf-8') as f:
        json.dump(inventory_data, f, indent=2)
    print(f"  Saved inventory to {inventory_file}")

    return {
        'total_events': len(processed),
        'source_distribution': dict(source_counts),
        'collisions': len(collisions),
        'unique_lemmas': len(inventory),
        'unique_senses': total_senses,
        'alias_count': len(canonicalizer.alias_map)
    }


if __name__ == "__main__":
    input_file = Path("paradigm_factory/v2/processed/merged_v21.jsonl")
    output_file = Path("paradigm_factory/v2/processed/canonicalized_v21.jsonl")

    stats = canonicalize_dataset(input_file, output_file)

    print("\n" + "=" * 60)
    print("CANONICALIZATION COMPLETE")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
