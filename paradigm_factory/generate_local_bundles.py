#!/usr/bin/env python3
"""
Local Bundle Generator - No Swarm Required
===========================================

Generates bundles from curated context templates for immediate testing.
Use this while waiting for swarm generation, or as a baseline dataset.

Creates ~500 bundles across 8 core words with known-good sense assignments.
"""

import random
from pathlib import Path
from datetime import datetime

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.polysemy_bundle_v2 import (
    create_bundle, save_bundles, SENSE_CATALOG
)

# Curated context templates for each word/sense
# These are hand-verified to have correct sense assignments
CURATED_CONTEXTS = {
    "present": {
        "present#gift": [
            "She wrapped the birthday present carefully with colorful paper.",
            "The present was hidden under the Christmas tree.",
            "He bought her a present for their anniversary.",
            "The children eagerly opened their presents on Christmas morning.",
            "I need to find a suitable present for my mother's birthday.",
            "The present came in a beautiful gift box with a ribbon.",
            "She received many presents at her graduation party.",
            "He forgot to bring a present to the wedding.",
            "The present was exactly what she had been wishing for.",
            "They exchanged presents during the holiday celebration.",
        ],
        "present#now": [
            "At present, sales are rising faster than expected.",
            "The present situation requires immediate attention.",
            "We must focus on the present rather than dwelling on the past.",
            "The present moment is all we truly have.",
            "At present, there are no plans to expand the facility.",
            "The present economic conditions favor investment.",
            "She lives very much in the present.",
            "The present state of affairs is concerning.",
            "At present, we cannot confirm the reports.",
            "The present administration has different priorities.",
        ],
        "present#verb": [
            "The CEO will present the quarterly results tomorrow.",
            "She will present her research findings at the conference.",
            "The lawyer presented compelling evidence to the jury.",
            "He presented his case with great eloquence.",
            "The committee will present its recommendations next week.",
            "She presented a persuasive argument for change.",
            "The artist will present her new collection in Paris.",
            "He presented the award to the winner.",
            "The teacher presented the concept clearly.",
            "She presented herself as a capable leader.",
        ],
        "present#attendance": [
            "All members were present at the emergency meeting.",
            "The teacher marked who was present on the roster.",
            "Everyone present agreed with the proposal.",
            "Only five students were present for the exam.",
            "Those present voted unanimously.",
            "She was present when the announcement was made.",
            "All shareholders present approved the merger.",
            "The witnesses present confirmed the story.",
            "How many people were present at the event?",
            "Everyone present signed the petition.",
        ],
    },
    "bank": {
        "bank#financial": [
            "I deposited my paycheck at the bank this morning.",
            "The bank approved my loan application yesterday.",
            "She works as a teller at the local bank downtown.",
            "The bank's interest rates have increased recently.",
            "He opened a savings account at the bank.",
            "The bank was robbed early this morning.",
            "She transferred money between her bank accounts.",
            "The bank offers excellent customer service.",
            "He waited in line at the bank for an hour.",
            "The central bank announced new monetary policy.",
        ],
        "bank#river": [
            "We sat on the grassy bank watching the river flow.",
            "The deer came down to drink at the river bank.",
            "Wildflowers grew along the bank of the stream.",
            "The children played on the sandy bank.",
            "The bank was eroding due to the strong current.",
            "He fished from the bank of the lake.",
            "Trees lined both banks of the river.",
            "The flood waters overflowed the banks.",
            "She walked along the bank looking for shells.",
            "The steep bank made access to the water difficult.",
        ],
        "bank#tilt": [
            "The plane banked sharply to the left.",
            "The pilot banked the aircraft to avoid the storm.",
            "The motorcycle banked into the curve at high speed.",
            "The bird banked gracefully on the wind currents.",
            "He banked the car around the tight corner.",
            "The glider banked smoothly over the valley.",
            "The helicopter banked to give passengers a better view.",
            "She banked the kayak to navigate the rapids.",
            "The jet banked steeply during the air show.",
            "The drone banked to capture the panoramic shot.",
        ],
    },
    "spring": {
        "spring#season": [
            "The flowers bloom beautifully in spring.",
            "Spring is my favorite time of year.",
            "We planted the garden in early spring.",
            "The birds return from migration each spring.",
            "Spring brings warmer weather and longer days.",
            "The spring semester starts in January.",
            "Cherry blossoms appear in late spring.",
            "Spring cleaning is an annual tradition.",
            "The spring thaw caused flooding.",
            "She was born in the spring of 1990.",
        ],
        "spring#water": [
            "We hiked to the mountain spring for fresh water.",
            "The natural spring provided cold, clear water.",
            "Hot springs are popular tourist attractions.",
            "The spring fed into a small stream.",
            "She filled her bottle from the spring.",
            "The village was built around a natural spring.",
            "Mineral springs were believed to have healing properties.",
            "The spring had been used for centuries.",
            "Fresh water bubbled up from the underground spring.",
            "The spring never dried up, even in drought.",
        ],
        "spring#coil": [
            "The spring in the mattress was broken.",
            "The old clock needed a new spring mechanism.",
            "He replaced the worn spring in the door.",
            "The spring provides tension in the device.",
            "She compressed the spring and released it.",
            "The toy car was powered by a wound spring.",
            "The spring had lost its elasticity.",
            "A coiled spring stores mechanical energy.",
            "The spring snapped under the pressure.",
            "He wound the spring tightly.",
        ],
        "spring#jump": [
            "The cat sprang onto the counter.",
            "He sprang to his feet when the alarm sounded.",
            "She sprang forward to catch the falling vase.",
            "The deer sprang over the fence.",
            "He sprang into action immediately.",
            "The athlete sprang from the starting block.",
            "She sprang up to answer the door.",
            "The lion sprang at its prey.",
            "He sprang back in surprise.",
            "The dancer sprang gracefully across the stage.",
        ],
    },
    "wave": {
        "wave#ocean": [
            "A huge wave crashed against the rocks.",
            "The surfer rode the wave expertly.",
            "Waves pounded the shore during the storm.",
            "The wave knocked her off her feet.",
            "He watched the waves roll in from the beach.",
            "The boat rocked on the gentle waves.",
            "Massive waves threatened the coastal town.",
            "The wave carried the shell onto the beach.",
            "She loved the sound of waves at night.",
            "The wave crested and broke near the pier.",
        ],
        "wave#gesture": [
            "He returned the wave with a smile.",
            "She gave a friendly wave as she left.",
            "The queen waved to the crowd from the balcony.",
            "He waved goodbye from the train window.",
            "She waved her hand to get attention.",
            "The children waved excitedly at the parade.",
            "He waved hello across the parking lot.",
            "She waved frantically to signal for help.",
            "The president waved to supporters.",
            "He gave a casual wave of acknowledgment.",
        ],
        "wave#hair": [
            "Her natural waves framed her face beautifully.",
            "The stylist added waves to her straight hair.",
            "His hair had a slight wave to it.",
            "She used a curling iron to create waves.",
            "Beach waves are a popular hairstyle.",
            "The waves in her hair were perfectly defined.",
            "He styled his waves with gel.",
            "Her waves bounced as she walked.",
            "The wave pattern suited her face shape.",
            "Loose waves gave her a romantic look.",
        ],
        "wave#surge": [
            "A wave of enthusiasm swept through the crowd.",
            "The crime wave concerned local residents.",
            "A heat wave gripped the region for weeks.",
            "A wave of refugees crossed the border.",
            "The new wave of technology changed everything.",
            "A wave of nausea washed over him.",
            "The first wave of protesters arrived at noon.",
            "A wave of panic spread through the market.",
            "She felt a wave of relief.",
            "The second wave of the pandemic was worse.",
        ],
    },
    "bark": {
        "bark#tree": [
            "The bark of the old oak was deeply furrowed.",
            "She peeled bark from the birch tree.",
            "The bark protected the tree from insects.",
            "He carved his initials into the bark.",
            "Rough bark covered the ancient trunk.",
            "The bark was smooth and silvery.",
            "Beetles had damaged the bark.",
            "Medicine was extracted from the bark.",
            "The bark peeled off in strips.",
            "Cork comes from tree bark.",
        ],
        "bark#dog": [
            "The dog's bark echoed through the night.",
            "A sharp bark alerted us to the intruder.",
            "Her bark is worse than her bite.",
            "The puppy's bark was surprisingly loud.",
            "We heard barking from next door.",
            "The dog's bark startled the cat.",
            "His bark woke the whole neighborhood.",
            "The guard dog's bark was intimidating.",
            "She recognized her dog's bark instantly.",
            "Constant barking annoyed the neighbors.",
        ],
    },
    "light": {
        "light#illumination": [
            "The light from the lamp illuminated the room.",
            "Morning light streamed through the window.",
            "She turned on the light to read.",
            "The light was too bright for his eyes.",
            "Candlelight created a romantic atmosphere.",
            "The light faded as the sun set.",
            "Natural light is best for photography.",
            "The light reflected off the water.",
            "She worked by the light of her laptop.",
            "The lighthouse beam swept across the sea.",
        ],
        "light#weight": [
            "The package was surprisingly light.",
            "She packed light for the weekend trip.",
            "The material is light but durable.",
            "He felt light on his feet after losing weight.",
            "Aluminum is a light metal.",
            "The light fabric was perfect for summer.",
            "Her suitcase was light enough to carry.",
            "The cake has a light, fluffy texture.",
            "Feathers are extremely light.",
            "The light construction made it portable.",
        ],
        "light#ignite": [
            "He lit the candles for dinner.",
            "She lit the fire to warm the cabin.",
            "The match wouldn't light in the wind.",
            "He lit his cigarette and inhaled.",
            "They lit fireworks for the celebration.",
            "She lit the gas stove carefully.",
            "The fire was lit at sunset.",
            "He struck a match to light the way.",
            "She lit incense for meditation.",
            "The bonfire was lit at midnight.",
        ],
    },
    "match": {
        "match#fire": [
            "He struck a match to light the candle.",
            "The box of matches was nearly empty.",
            "She carefully lit a match in the dark.",
            "The match flared and then went out.",
            "He always carried matches for emergencies.",
            "Safety matches only ignite on the box.",
            "The damp matches wouldn't light.",
            "She used the last match to start the fire.",
            "He tossed the spent match away.",
            "A single match can start a forest fire.",
        ],
        "match#competition": [
            "The tennis match lasted three hours.",
            "They won the championship match.",
            "The boxing match ended in a knockout.",
            "She watched the soccer match on TV.",
            "The chess match was intense.",
            "He lost the match in the final set.",
            "The wrestling match drew a large crowd.",
            "The match was postponed due to rain.",
            "She trained hard for the upcoming match.",
            "The match ended in a draw.",
        ],
        "match#correspond": [
            "The colors don't quite match.",
            "Her skills match the job requirements.",
            "Find the items that match.",
            "The fingerprints match the suspect.",
            "His story doesn't match the evidence.",
            "These socks don't match.",
            "The DNA samples match perfectly.",
            "Her accessories match her dress.",
            "The results match our predictions.",
            "These pieces match together.",
        ],
    },
    "bass": {
        "bass#fish": [
            "He caught a large bass in the lake.",
            "Bass fishing is popular in this region.",
            "The bass weighed nearly ten pounds.",
            "Striped bass migrate along the coast.",
            "She ordered the grilled sea bass.",
            "Largemouth bass are prized by anglers.",
            "The bass put up quite a fight.",
            "Bass are found in both fresh and salt water.",
            "He released the bass back into the water.",
            "The restaurant serves excellent bass.",
        ],
        "bass#music": [
            "The bass guitar provided the rhythm.",
            "He plays bass in a rock band.",
            "The bass was too loud in the mix.",
            "She sang in a deep bass voice.",
            "Turn up the bass on the speakers.",
            "The bass line was catchy.",
            "He bought a new bass amplifier.",
            "The double bass is a classical instrument.",
            "The bass notes were rich and deep.",
            "She adjusted the bass and treble.",
        ],
    },
}


def generate_local_bundles(output_path: Path, bundles_per_sense: int = 5, seed: int = 42):
    """Generate bundles from curated contexts."""
    random.seed(seed)

    print("=" * 70)
    print("  Local Bundle Generator (No Swarm)")
    print("=" * 70)

    bundles = []
    bundle_idx = 0

    for word, senses_dict in CURATED_CONTEXTS.items():
        if word not in SENSE_CATALOG:
            continue

        sense_defs = SENSE_CATALOG[word]
        sense_ids = list(senses_dict.keys())

        print(f"\n{word}: {len(sense_ids)} senses")

        for anchor_sense_id in sense_ids:
            anchor_contexts = senses_dict[anchor_sense_id]

            # Generate multiple bundles per sense
            for b in range(min(bundles_per_sense, len(anchor_contexts) - 1)):
                # Select anchor and positives
                anchor_context = anchor_contexts[b]
                positive_contexts = [(anchor_sense_id, anchor_contexts[(b + 1) % len(anchor_contexts)])]

                # Add more positives if available
                if len(anchor_contexts) > b + 2:
                    positive_contexts.append((anchor_sense_id, anchor_contexts[(b + 2) % len(anchor_contexts)]))

                # Select negatives from other senses
                negative_contexts = []
                hard_negative_contexts = []

                for other_sense_id, other_contexts in senses_dict.items():
                    if other_sense_id == anchor_sense_id:
                        continue

                    # Regular negative
                    neg_idx = b % len(other_contexts)
                    negative_contexts.append((other_sense_id, other_contexts[neg_idx]))

                    # Hard negative (structurally similar)
                    if len(other_contexts) > neg_idx + 1:
                        hard_negative_contexts.append(
                            (other_sense_id, other_contexts[(neg_idx + 1) % len(other_contexts)],
                             "Different sense, similar structure")
                        )

                # Create bundle
                try:
                    bundle = create_bundle(
                        word=word,
                        anchor_sense_id=anchor_sense_id,
                        anchor_context=anchor_context,
                        positive_contexts=positive_contexts[:2],
                        negative_contexts=negative_contexts[:2],
                        hard_negative_contexts=hard_negative_contexts[:1],
                        bundle_index=bundle_idx,
                        seed=seed + bundle_idx
                    )
                    bundles.append(bundle)
                    bundle_idx += 1
                except Exception as e:
                    print(f"  Warning: Failed to create bundle: {e}")

        print(f"  Generated {bundle_idx} bundles so far")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_bundles(bundles, output_path)

    print(f"\n{'=' * 70}")
    print(f"  Generated {len(bundles)} bundles")
    print(f"  Words: {len(set(b.word.surface for b in bundles))}")
    print(f"  Items: {sum(len(b.items) for b in bundles)}")
    print(f"  Saved to: {output_path}")
    print("=" * 70)

    return bundles


if __name__ == "__main__":
    output_path = Path("paradigm_factory/output/local_bundles.jsonl")
    generate_local_bundles(output_path, bundles_per_sense=5)
