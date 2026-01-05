"""
Bundle Format Converter
=======================

Converts existing v2.0 bundles to v2.1 format.
Keeps original files intact.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def convert_item_v20_to_v21(item: Dict, lemma: str) -> Dict:
    """Convert a v2.0 item to v2.1 format."""

    context = item.get('context', '')

    # Find target span (approximate - find the lemma in context)
    context_lower = context.lower()
    lemma_lower = lemma.lower()
    start = context_lower.find(lemma_lower)
    if start == -1:
        start = 0
    end = start + len(lemma)

    # Determine negative type from role and hardness
    negative_type = ""
    if item['role'] in ('negative', 'hard_negative'):
        hardness = item.get('hardness', 'easy')
        if hardness == 'hard':
            negative_type = 'hard_lexical'  # Assume lexical since we don't have embedding info
        else:
            negative_type = 'easy'

    # Extract disambiguators from context (simple word tokenization minus stopwords)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'on', 'it', 'that'}
    words = context.lower().split()
    disambiguators = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:') not in stopwords and len(w) > 2][:8]

    return {
        "item_id": item.get('item_id', ''),
        "role": item['role'],
        "sense_id": item['sense_id'],
        "context": context,
        "context_window": context,  # Same as context in v2.0
        "target_span": [start, end],
        "disambiguators": disambiguators,
        "source_event_id": item.get('item_id', ''),
        "difficulty": item.get('difficulty', 0.5),
        "hardness": item.get('hardness', 'medium'),
        "style": "narrative",  # Default - v2.0 didn't track style
        "negative_type": negative_type,
        "lexical_overlap": 0.0,
        "embedding_similarity": 0.0,
        "confusion_reason": item.get('notes', '')
    }


def convert_pairings_v20_to_v21(pairings: Dict) -> Dict:
    """Convert v2.0 pairings to v2.1 format."""

    negatives = pairings.get('negatives', [])
    hard_negatives = pairings.get('hard_negatives', [])

    return {
        "anchor_item_id": pairings.get('anchor_item_id', ''),
        "positives": pairings.get('positives', []),
        "negatives": {
            "easy": negatives,
            "medium": [],
            "hard_lexical": hard_negatives,
            "hard_embedding": []
        }
    }


def convert_contrastive_targets_v20_to_v21(targets: Dict) -> Dict:
    """Convert v2.0 contrastive targets to v2.1 format."""

    margin = targets.get('margin', {})

    return {
        "margins": {
            "positive": 0.05,
            "easy_negative": margin.get('positive_vs_negative', 0.15),
            "medium_negative": 0.10,
            "hard_negative": margin.get('positive_vs_hard_negative', 0.25)
        }
    }


def convert_simple_bundle(bundle: Dict) -> Dict:
    """Convert a simple/older format bundle to v2.1."""

    word = bundle.get('word', '')
    if isinstance(word, str):
        lemma = word
    else:
        lemma = word.get('lemma', '')

    items_data = bundle.get('items', {})
    metadata = bundle.get('metadata', {})
    target_sense = metadata.get('target_sense', 's1')

    # Create items list from simple format
    items = []

    # Map sense_catalog keys to proper sense_ids
    sense_catalog_data = bundle.get('sense_catalog', {})
    sense_map = {}
    for key, gloss in sense_catalog_data.items():
        sense_id = f"{lemma}#{key}"
        sense_map[key] = sense_id

    # Anchor
    if 'anchor' in items_data:
        items.append({
            "item_id": f"anchor_{bundle.get('bundle_id', '')}",
            "role": "anchor",
            "sense_id": sense_map.get(target_sense, f"{lemma}#{target_sense}"),
            "context": items_data['anchor'],
            "context_window": items_data['anchor'],
            "target_span": [0, len(lemma)],
            "disambiguators": [],
            "source_event_id": bundle.get('bundle_id', ''),
            "difficulty": metadata.get('difficulty_score', 0.5),
            "hardness": "medium",
            "style": "narrative",
            "negative_type": "",
            "lexical_overlap": 0.0,
            "embedding_similarity": 0.0,
            "confusion_reason": ""
        })

    # Positive
    if 'positive' in items_data:
        items.append({
            "item_id": f"positive_{bundle.get('bundle_id', '')}",
            "role": "positive",
            "sense_id": sense_map.get(target_sense, f"{lemma}#{target_sense}"),
            "context": items_data['positive'],
            "context_window": items_data['positive'],
            "target_span": [0, len(lemma)],
            "disambiguators": [],
            "source_event_id": bundle.get('bundle_id', ''),
            "difficulty": metadata.get('difficulty_score', 0.5),
            "hardness": "medium",
            "style": "narrative",
            "negative_type": "",
            "lexical_overlap": 0.0,
            "embedding_similarity": 0.0,
            "confusion_reason": ""
        })

    # Negative
    if 'negative' in items_data:
        items.append({
            "item_id": f"negative_{bundle.get('bundle_id', '')}",
            "role": "negative",
            "sense_id": f"{lemma}#other",
            "context": items_data['negative'],
            "context_window": items_data['negative'],
            "target_span": [0, len(lemma)],
            "disambiguators": [],
            "source_event_id": bundle.get('bundle_id', ''),
            "difficulty": metadata.get('difficulty_score', 0.5),
            "hardness": "easy",
            "style": "narrative",
            "negative_type": "easy",
            "lexical_overlap": 0.0,
            "embedding_similarity": 0.0,
            "confusion_reason": ""
        })

    # Hard negative
    if 'hard_negative' in items_data:
        items.append({
            "item_id": f"hard_negative_{bundle.get('bundle_id', '')}",
            "role": "hard_negative",
            "sense_id": f"{lemma}#other",
            "context": items_data['hard_negative'],
            "context_window": items_data['hard_negative'],
            "target_span": [0, len(lemma)],
            "disambiguators": [],
            "source_event_id": bundle.get('bundle_id', ''),
            "difficulty": metadata.get('difficulty_score', 0.5),
            "hardness": "hard",
            "style": "narrative",
            "negative_type": "hard_lexical",
            "lexical_overlap": 0.0,
            "embedding_similarity": 0.0,
            "confusion_reason": ""
        })

    # Build sense catalog
    sense_catalog = []
    for key, gloss in sense_catalog_data.items():
        sense_catalog.append({
            "sense_id": sense_map.get(key, f"{lemma}#{key}"),
            "label": key,
            "gloss": gloss if isinstance(gloss, str) else str(gloss),
            "cues": [],
            "anti_cues": []
        })

    return {
        "schema_version": "2.1",
        "bundle_id": bundle.get('bundle_id', ''),
        "paradigm": "polysemy",
        "word": {
            "lemma": lemma,
            "pos": "NOUN",
            "language": "en"
        },
        "sense_catalog": sense_catalog,
        "items": items,
        "pairings": {
            "anchor_item_id": f"anchor_{bundle.get('bundle_id', '')}",
            "positives": [f"positive_{bundle.get('bundle_id', '')}"],
            "negatives": {
                "easy": [f"negative_{bundle.get('bundle_id', '')}"] if 'negative' in items_data else [],
                "medium": [],
                "hard_lexical": [f"hard_negative_{bundle.get('bundle_id', '')}"] if 'hard_negative' in items_data else [],
                "hard_embedding": []
            }
        },
        "contrastive_targets": {
            "margins": {
                "positive": 0.05,
                "easy_negative": 0.15,
                "medium_negative": 0.10,
                "hard_negative": 0.25
            }
        },
        "provenance": {
            "generator": "bundle_converter",
            "source": "semanticphase_v2",
            "converted_from": "simple_format",
            "conversion_timestamp": datetime.utcnow().isoformat() + 'Z',
            "quality_score": 1.0 - metadata.get('difficulty_score', 0.5)
        }
    }


def convert_bundle_v20_to_v21(bundle: Dict) -> Dict:
    """Convert a v2.0 bundle to v2.1 format."""

    word = bundle.get('word', {})
    lemma = word.get('lemma', '')

    # Convert items
    items = [convert_item_v20_to_v21(item, lemma) for item in bundle.get('items', [])]

    # Convert pairings
    pairings = convert_pairings_v20_to_v21(bundle.get('pairings', {}))

    # Convert contrastive targets
    contrastive_targets = convert_contrastive_targets_v20_to_v21(bundle.get('contrastive_targets', {}))

    # Update provenance
    provenance = bundle.get('provenance', {})
    provenance['converted_from'] = bundle.get('schema_version', '2.0')
    provenance['conversion_timestamp'] = datetime.utcnow().isoformat() + 'Z'

    return {
        "schema_version": "2.1",
        "bundle_id": bundle.get('bundle_id', ''),
        "paradigm": "polysemy",
        "word": {
            "lemma": lemma,
            "pos": word.get('pos', 'NOUN').upper(),
            "language": word.get('language', 'en')
        },
        "sense_catalog": bundle.get('sense_catalog', []),
        "items": items,
        "pairings": pairings,
        "contrastive_targets": contrastive_targets,
        "provenance": provenance
    }


def convert_file(input_path: Path, output_path: Path) -> int:
    """Convert a JSONL file from v2.0 to v2.1 format."""

    count = 0

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            try:
                bundle = json.loads(line)
                converted = convert_bundle_v20_to_v21(bundle)
                fout.write(json.dumps(converted) + '\n')
                count += 1
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error converting bundle: {e}")
                continue

    return count


def main():
    """Convert existing bundle files to v2.1 format."""

    print("=" * 60)
    print("BUNDLE FORMAT CONVERTER (v2.0 -> v2.1)")
    print("=" * 60)

    # Files to convert
    input_dir = Path("paradigm_factory/output")
    output_dir = Path("paradigm_factory/v2/bundles_converted")
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_convert = [
        "verified_bundles.jsonl",
        "local_bundles.jsonl",
        "bundles_30k.jsonl",
        "semanticphase_v2_bundles.jsonl"
    ]

    total_converted = 0

    for filename in files_to_convert:
        input_path = input_dir / filename
        if not input_path.exists():
            print(f"\nSkipping {filename} (not found)")
            continue

        output_path = output_dir / f"v21_{filename}"

        print(f"\nConverting {filename}...")
        count = convert_file(input_path, output_path)
        print(f"  Converted {count} bundles -> {output_path}")
        total_converted += count

    print("\n" + "=" * 60)
    print(f"CONVERSION COMPLETE: {total_converted} bundles converted")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("Original files remain unchanged.")


if __name__ == "__main__":
    main()
