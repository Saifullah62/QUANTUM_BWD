"""
Sense Inventory Loader
======================

Loads pre-built sense inventory from JSON.
This is faster and more reproducible than building from WordNet at runtime.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SenseEntry:
    """A single sense definition."""
    sense_id: str
    gloss: str
    domain: str
    cues: Dict[str, List[str]]
    anti_cues: Dict[str, List[str]]
    pos: str = "NOUN"


@dataclass
class LemmaInventory:
    """All senses for a lemma."""
    lemma: str
    pos: str
    senses: List[SenseEntry]


def load_inventory(
    inventory_path: Path = None
) -> Dict[str, LemmaInventory]:
    """
    Load the sense inventory from JSON.

    Returns:
        Dict mapping 'lemma_POS' to LemmaInventory
    """
    if inventory_path is None:
        inventory_path = Path(__file__).parent / "polysemous_senses.json"

    with open(inventory_path, 'r') as f:
        data = json.load(f)

    inventory = {}

    for lemma, entry in data.items():
        pos = entry.get('pos', 'NOUN')
        senses = []

        for sense_data in entry.get('senses', []):
            sense = SenseEntry(
                sense_id=sense_data['sense_id'],
                gloss=sense_data['gloss'],
                domain=sense_data['domain'],
                cues=sense_data.get('cues', {'keywords': [], 'collocates': []}),
                anti_cues=sense_data.get('anti_cues', {'keywords': [], 'collocates': []}),
                pos=pos
            )
            senses.append(sense)

        lemma_inv = LemmaInventory(
            lemma=lemma,
            pos=pos,
            senses=senses
        )

        key = f"{lemma}_{pos}"
        inventory[key] = lemma_inv

    return inventory


def get_inventory_stats(inventory: Dict[str, LemmaInventory]) -> Dict:
    """Get statistics about the inventory."""
    total_senses = sum(len(inv.senses) for inv in inventory.values())
    domains = set()
    for inv in inventory.values():
        for sense in inv.senses:
            domains.add(sense.domain)

    return {
        'num_lemmas': len(inventory),
        'total_senses': total_senses,
        'avg_senses_per_lemma': total_senses / len(inventory) if inventory else 0,
        'domains': list(domains)
    }


def export_to_jsonl(
    inventory: Dict[str, LemmaInventory],
    output_path: Path
) -> None:
    """Export inventory to JSONL format for scraper consumption."""
    with open(output_path, 'w') as f:
        for key, inv in inventory.items():
            entry = {
                'lemma': inv.lemma,
                'pos': inv.pos,
                'senses': [
                    {
                        'sense_id': s.sense_id,
                        'gloss': s.gloss,
                        'domain': s.domain,
                        'cues': s.cues,
                        'anti_cues': s.anti_cues,
                        'pos': s.pos
                    }
                    for s in inv.senses
                ]
            }
            f.write(json.dumps(entry) + '\n')


def main():
    """Test the inventory loader."""
    inventory = load_inventory()
    stats = get_inventory_stats(inventory)

    print("=" * 60)
    print("SENSE INVENTORY STATS")
    print("=" * 60)
    print(f"Lemmas: {stats['num_lemmas']}")
    print(f"Total senses: {stats['total_senses']}")
    print(f"Avg senses per lemma: {stats['avg_senses_per_lemma']:.1f}")
    print(f"Domains: {', '.join(sorted(stats['domains']))}")
    print()

    # Show example
    print("Example - 'bank':")
    bank_inv = inventory.get('bank_NOUN')
    if bank_inv:
        for sense in bank_inv.senses:
            print(f"  {sense.sense_id}")
            print(f"    Gloss: {sense.gloss}")
            print(f"    Domain: {sense.domain}")
            print(f"    Cue keywords: {sense.cues.get('keywords', [])[:5]}")
            print()

    # Export to JSONL
    output_dir = Path(__file__).parent
    output_path = output_dir / "sense_inventory.jsonl"
    export_to_jsonl(inventory, output_path)
    print(f"Exported to: {output_path}")


if __name__ == "__main__":
    main()
