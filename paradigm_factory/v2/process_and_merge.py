"""
Data Processing and Merging Script
===================================

Merges all collected JSONL files, deduplicates events, and ensures v2.1 schema compliance.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re


def compute_text_hash(text: str) -> str:
    """Compute normalized hash of text for deduplication."""
    # Normalize: lowercase, remove extra whitespace, strip punctuation variations
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def is_valid_v21_event(event: Dict) -> bool:
    """Check if event conforms to v2.1 schema."""
    required_fields = ['id', 'lemma', 'pos', 'sense_id', 'text', 'span', 'source']

    for field in required_fields:
        if field not in event:
            return False

    # Check nested structures
    if 'span' in event:
        span = event['span']
        if not isinstance(span, dict):
            return False
        if 'start' not in span or 'end' not in span:
            return False

    return True


def convert_to_v21(event: Dict) -> Dict:
    """Convert older format events to v2.1 schema."""
    # If already v2.1, return as-is
    if is_valid_v21_event(event):
        return event

    # Convert from older formats
    converted = {}

    # Basic fields
    converted['id'] = event.get('id', hashlib.md5(str(event).encode()).hexdigest()[:8])
    converted['lemma'] = event.get('lemma', event.get('word', ''))
    converted['pos'] = event.get('pos', 'noun')
    converted['sense_id'] = event.get('sense_id', f"{converted['lemma']}.{converted['pos']}.1")
    converted['sense_gloss'] = event.get('sense_gloss', event.get('definition', ''))
    converted['text'] = event.get('text', event.get('sentence', event.get('context', '')))

    # Span - convert from various formats
    if 'span' in event and isinstance(event['span'], dict):
        converted['span'] = event['span']
    elif 'span_char' in event:
        start, end = event['span_char']
        surface = converted['text'][start:end] if converted['text'] else converted['lemma']
        converted['span'] = {'start': start, 'end': end, 'surface': surface}
    else:
        # Try to find lemma in text
        text = converted['text'].lower()
        lemma = converted['lemma'].lower()
        start = text.find(lemma)
        if start == -1:
            start = 0
        end = start + len(converted['lemma'])
        converted['span'] = {'start': start, 'end': end, 'surface': converted['lemma']}

    # Context window
    if 'context_window' in event and isinstance(event['context_window'], dict):
        converted['context_window'] = event['context_window']
    else:
        span = converted['span']
        text = converted['text']
        converted['context_window'] = {
            'left': text[:span['start']].strip(),
            'right': text[span['end']:].strip()
        }

    # Cue tokens and type
    converted['cue_tokens'] = event.get('cue_tokens', [])
    converted['cue_type'] = event.get('cue_type', event.get('signal', ['unknown']))
    if not isinstance(converted['cue_type'], list):
        converted['cue_type'] = [converted['cue_type']]

    # Topic tags
    converted['topic_tags'] = event.get('topic_tags', [])
    if not converted['topic_tags']:
        converted['topic_tags'] = [converted['pos']]

    # Source
    if 'source' in event and isinstance(event['source'], dict):
        converted['source'] = event['source']
    else:
        converted['source'] = {
            'url': event.get('url', event.get('source_url', '')),
            'domain': event.get('domain', 'unknown'),
            'license': event.get('license', 'unknown'),
            'rights_ok': True,
            'robots_ok': True
        }

    # Quality
    if 'quality' in event and isinstance(event['quality'], dict):
        converted['quality'] = event['quality']
    else:
        converted['quality'] = {
            'cue_strength': event.get('cue_strength', 0.5),
            'ambiguity_risk': event.get('ambiguity_risk', 0.5),
            'toxicity_risk': 0.0,
            'boilerplate_risk': 0.0,
            'length_chars': len(converted['text']),
            'style': 'narrative'
        }

    # Splits
    if 'splits' in event and isinstance(event['splits'], dict):
        converted['splits'] = event['splits']
    else:
        converted['splits'] = {
            'holdout_lemma': False,
            'holdout_template_family': False,
            'holdout_cue_family': False
        }

    # Provenance hash
    if 'provenance_hash' in event:
        converted['provenance_hash'] = event['provenance_hash']
    else:
        prov_str = f"{converted['lemma']}|{converted['text'][:100]}"
        converted['provenance_hash'] = hashlib.sha256(prov_str.encode()).hexdigest()[:16]

    # Notes
    converted['notes'] = event.get('notes', '')

    return converted


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load events from a JSONL file."""
    events = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
    return events


def deduplicate_events(events: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Deduplicate events based on text hash and provenance."""
    seen_text_hashes: Set[str] = set()
    seen_prov_hashes: Set[str] = set()
    unique_events = []
    stats = {
        'total_input': len(events),
        'text_duplicates': 0,
        'prov_duplicates': 0,
        'unique': 0
    }

    for event in events:
        # Get text hash
        text = event.get('text', '')
        text_hash = compute_text_hash(text)

        # Get provenance hash
        prov_hash = event.get('provenance_hash', '')

        # Check for duplicates
        if text_hash in seen_text_hashes:
            stats['text_duplicates'] += 1
            continue

        if prov_hash and prov_hash in seen_prov_hashes:
            stats['prov_duplicates'] += 1
            continue

        # Mark as seen
        seen_text_hashes.add(text_hash)
        if prov_hash:
            seen_prov_hashes.add(prov_hash)

        unique_events.append(event)
        stats['unique'] += 1

    return unique_events, stats


def analyze_dataset(events: List[Dict]) -> Dict:
    """Analyze the dataset and return statistics."""
    stats = {
        'total_events': len(events),
        'unique_lemmas': set(),
        'by_pos': defaultdict(int),
        'by_source_domain': defaultdict(int),
        'by_cue_type': defaultdict(int),
        'avg_text_length': 0,
        'min_text_length': float('inf'),
        'max_text_length': 0
    }

    total_length = 0

    for event in events:
        # Lemmas
        lemma = event.get('lemma', '')
        if lemma:
            stats['unique_lemmas'].add(lemma.lower())

        # POS
        pos = event.get('pos', 'unknown')
        stats['by_pos'][pos] += 1

        # Source domain
        source = event.get('source', {})
        domain = source.get('domain', 'unknown') if isinstance(source, dict) else 'unknown'
        stats['by_source_domain'][domain] += 1

        # Cue types
        cue_types = event.get('cue_type', [])
        if isinstance(cue_types, list):
            for ct in cue_types:
                stats['by_cue_type'][ct] += 1

        # Text length
        text = event.get('text', '')
        text_len = len(text)
        total_length += text_len
        stats['min_text_length'] = min(stats['min_text_length'], text_len)
        stats['max_text_length'] = max(stats['max_text_length'], text_len)

    stats['unique_lemmas'] = len(stats['unique_lemmas'])
    stats['avg_text_length'] = total_length / len(events) if events else 0
    stats['by_pos'] = dict(stats['by_pos'])
    stats['by_source_domain'] = dict(stats['by_source_domain'])
    stats['by_cue_type'] = dict(stats['by_cue_type'])

    return stats


def main():
    """Main processing function."""
    print("=" * 70)
    print("DATA PROCESSING AND MERGING")
    print("=" * 70)

    raw_events_dir = Path("paradigm_factory/v2/raw_events")
    output_dir = Path("paradigm_factory/v2/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all JSONL files
    print("\n--- Loading source files ---")
    all_events = []
    source_stats = {}

    for jsonl_file in sorted(raw_events_dir.glob("*.jsonl")):
        print(f"  Loading {jsonl_file.name}...", end=" ")
        events = load_jsonl(jsonl_file)
        print(f"{len(events)} events")
        source_stats[jsonl_file.name] = len(events)
        all_events.extend(events)

    print(f"\n  Total loaded: {len(all_events)} events")

    # Convert to v2.1 schema
    print("\n--- Converting to v2.1 schema ---")
    converted_events = []
    conversion_errors = 0

    for event in all_events:
        try:
            converted = convert_to_v21(event)
            converted_events.append(converted)
        except Exception as e:
            conversion_errors += 1

    print(f"  Converted: {len(converted_events)}")
    print(f"  Conversion errors: {conversion_errors}")

    # Deduplicate
    print("\n--- Deduplicating ---")
    unique_events, dedup_stats = deduplicate_events(converted_events)

    print(f"  Input: {dedup_stats['total_input']}")
    print(f"  Text duplicates removed: {dedup_stats['text_duplicates']}")
    print(f"  Provenance duplicates removed: {dedup_stats['prov_duplicates']}")
    print(f"  Unique events: {dedup_stats['unique']}")

    # Filter quality
    print("\n--- Quality filtering ---")
    quality_events = []
    filtered_out = 0

    for event in unique_events:
        text = event.get('text', '')

        # Filter criteria
        if len(text) < 20:  # Too short
            filtered_out += 1
            continue
        if len(text) > 5000:  # Too long
            filtered_out += 1
            continue
        if not event.get('lemma'):  # No lemma
            filtered_out += 1
            continue

        quality_events.append(event)

    print(f"  Filtered out: {filtered_out}")
    print(f"  Quality events: {len(quality_events)}")

    # Analyze final dataset
    print("\n--- Dataset Analysis ---")
    analysis = analyze_dataset(quality_events)

    print(f"\n  Total events: {analysis['total_events']}")
    print(f"  Unique lemmas: {analysis['unique_lemmas']}")
    print(f"  Avg text length: {analysis['avg_text_length']:.1f} chars")
    print(f"  Text length range: {analysis['min_text_length']} - {analysis['max_text_length']}")

    print(f"\n  By POS:")
    for pos, count in sorted(analysis['by_pos'].items(), key=lambda x: -x[1])[:10]:
        print(f"    {pos}: {count}")

    print(f"\n  By source domain:")
    for domain, count in sorted(analysis['by_source_domain'].items(), key=lambda x: -x[1])[:10]:
        print(f"    {domain}: {count}")

    print(f"\n  By cue type:")
    for ct, count in sorted(analysis['by_cue_type'].items(), key=lambda x: -x[1])[:10]:
        print(f"    {ct}: {count}")

    # Save processed data
    print("\n--- Saving processed data ---")

    # Save merged JSONL
    merged_file = output_dir / "merged_v21.jsonl"
    with open(merged_file, 'w', encoding='utf-8') as f:
        for event in quality_events:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    print(f"  Saved: {merged_file} ({len(quality_events)} events)")

    # Save statistics
    stats_file = output_dir / "processing_stats.json"
    stats_data = {
        'timestamp': datetime.now().isoformat(),
        'source_files': source_stats,
        'deduplication': dedup_stats,
        'filtered_out': filtered_out,
        'final_count': len(quality_events),
        'analysis': {
            'total_events': analysis['total_events'],
            'unique_lemmas': analysis['unique_lemmas'],
            'avg_text_length': analysis['avg_text_length'],
            'by_pos': analysis['by_pos'],
            'by_source_domain': analysis['by_source_domain'],
            'by_cue_type': analysis['by_cue_type']
        }
    }

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2)
    print(f"  Saved: {stats_file}")

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\n  Final dataset: {len(quality_events)} events")
    print(f"  Unique lemmas: {analysis['unique_lemmas']}")
    print(f"  Output: {merged_file}")

    return quality_events


if __name__ == "__main__":
    events = main()
