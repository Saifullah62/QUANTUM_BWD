"""Simple test of bundle generation."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import json
from pathlib import Path
from collections import defaultdict

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("Starting test...", flush=True)

from paradigm_factory.v2.sense_inventory.load_inventory import load_inventory
from paradigm_factory.v2.bundle_generator import BundleGenerator

print("Loading inventory...", flush=True)
inventory = load_inventory()
print(f"Loaded {len(inventory)} lemmas", flush=True)

# Convert inventory
inv_dict = {}
for key, lemma_inv in inventory.items():
    inv_dict[key] = {
        'lemma': lemma_inv.lemma,
        'pos': lemma_inv.pos,
        'senses': [
            {'sense_id': s.sense_id, 'gloss': s.gloss, 'domain': s.domain,
             'cues': s.cues, 'anti_cues': s.anti_cues, 'pos': s.pos}
            for s in lemma_inv.senses
        ]
    }

events_path = Path(__file__).parent / "raw_events" / "pilot_wikipedia.jsonl"
output_dir = Path(__file__).parent / "bundles"
output_dir.mkdir(parents=True, exist_ok=True)

print("Creating generator...", flush=True)
generator = BundleGenerator(sense_inventory=inv_dict, output_dir=output_dir)

print("Generating bundles...", flush=True)
try:
    bundles = generator.generate_bundles(events_path=events_path, max_bundles_per_lemma=20)
    print(f"Generated {len(bundles)} bundles", flush=True)

    output_path = generator.save_bundles(bundles, "pilot_bundles_v2.jsonl")
    print(f"Saved to: {output_path}", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("Done!", flush=True)
