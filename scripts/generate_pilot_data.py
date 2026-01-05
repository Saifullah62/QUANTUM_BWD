#!/usr/bin/env python3
"""
Pilot Dataset Generator for QLLM Paradigm Validation
=====================================================

Creates a small, surgical dataset that makes each paradigm objective fire reliably.

This is NOT a million-row dataset. It's a focused proof-of-concept that should show
clear paradigm-specific separation in ablation studies.

Paradigm-Specific Requirements:
-------------------------------
1. POLYSEMY (SemanticPhase):
   - Same surface form with clearly different senses
   - Contexts that make sense unambiguous
   - Hard negatives: similar contexts, different senses
   - Win: separable phase basins across senses, tight within sense

2. LINDBLAD (Noise Invariance):
   - Meaning-preserving paraphrases
   - Structured surface-level corruption
   - Win: degradation curve - baseline falls off, Lindblad stays stable

3. RETROCAUSAL (Backward Inference):
   - Effect→cause where forward heuristics fail
   - Cause cannot be copied from effect
   - Hypothesis revision with later evidence
   - Win: improved reasoning while leak stays clean

4. QUALIA (Qualitative Calibration):
   - Certainty ↔ entropy/calibration signals
   - Novelty ↔ surprisal patterns
   - Coherence ↔ internal agreement
   - Win: channels correlate with measurable properties

Usage:
    python scripts/generate_pilot_data.py --output data/pilot_paradigm_data.jsonl
    python scripts/generate_pilot_data.py --paradigm polysemy --count 200
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict


@dataclass
class PilotExample:
    """Single training example with paradigm metadata."""
    input_text: str
    output_text: str
    paradigm: str
    subtype: str  # e.g., "polysemy_positive", "noise_clean", "effect_cause"
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return asdict(self)


class PolysemyGenerator:
    """
    Generate polysemy pairs for contrastive phase learning.

    Key insight: We need the SAME token to appear in contexts where:
    - Same sense → phases should ALIGN (positive pairs)
    - Different sense → phases should SEPARATE (negative pairs)
    """

    # Core polysemy words with clearly distinct senses
    POLYSEMY_WORDS = {
        'bank': {
            'financial': [
                "I deposited my paycheck at the bank this morning.",
                "The bank approved my loan application yesterday.",
                "She works as a teller at the local bank downtown.",
                "The bank's interest rates have increased recently.",
                "I need to visit the bank to open a new account.",
            ],
            'river': [
                "We sat on the grassy bank watching the river flow.",
                "The boat was tied to a post on the bank.",
                "Fish were jumping near the muddy bank of the stream.",
                "Children played along the sandy bank of the creek.",
                "The deer came down to drink at the river bank.",
            ]
        },
        'light': {
            'illumination': [
                "The light from the window brightened the room.",
                "She turned on the light to read her book.",
                "The morning light filtered through the curtains.",
                "A single light bulb hung from the ceiling.",
                "The light was too dim to see clearly.",
            ],
            'weight': [
                "The suitcase was surprisingly light to carry.",
                "She packed light for the weekend trip.",
                "The material is light but very durable.",
                "He preferred a light lunch before the meeting.",
                "The dancer moved with light, graceful steps.",
            ]
        },
        'spring': {
            'season': [
                "The flowers bloom beautifully in spring.",
                "Spring brings warmer weather and longer days.",
                "We plan our garden planting every spring.",
                "The birds return from migration each spring.",
                "Spring cleaning is a tradition in our house.",
            ],
            'coil': [
                "The spring in the mattress was broken.",
                "He replaced the door spring last week.",
                "The spring mechanism makes the toy jump.",
                "A metal spring held the lid tightly closed.",
                "The old clock needed a new spring.",
            ],
            'water': [
                "Fresh water bubbled up from the natural spring.",
                "We hiked to the mountain spring for water.",
                "The spring provided clean drinking water.",
                "Hot springs are popular tourist destinations.",
                "The spring dried up during the drought.",
            ]
        },
        'bark': {
            'dog': [
                "The dog began to bark at the stranger.",
                "I heard a loud bark from across the yard.",
                "Some dogs bark more than others.",
                "The puppy's bark was surprisingly loud.",
                "A sudden bark startled the sleeping cat.",
            ],
            'tree': [
                "The bark of the oak tree was rough and thick.",
                "She collected pieces of bark for her art project.",
                "Beetles had damaged the tree's bark.",
                "The bark peeled off the birch tree easily.",
                "Ancient carvings marked the bark of the old elm.",
            ]
        },
        'bass': {
            'fish': [
                "He caught a large bass in the lake.",
                "Bass fishing is popular in this region.",
                "The bass was too small to keep.",
                "Largemouth bass thrive in warm waters.",
                "We grilled the fresh bass for dinner.",
            ],
            'music': [
                "The bass guitar added depth to the song.",
                "She plays bass in a jazz band.",
                "Turn up the bass on the speakers.",
                "The bass notes resonated through the hall.",
                "He practiced bass lines for hours.",
            ]
        },
        'match': {
            'fire': [
                "He struck a match to light the candle.",
                "Keep matches away from children.",
                "The match flickered and went out.",
                "She used a match to start the campfire.",
                "A single match was enough to ignite the paper.",
            ],
            'competition': [
                "The tennis match lasted three hours.",
                "They won the championship match decisively.",
                "It was an exciting match from start to finish.",
                "The match ended in a surprising upset.",
                "Both teams prepared intensely for the match.",
            ],
            'pair': [
                "These socks don't match each other.",
                "Find the card that matches this one.",
                "The curtains match the sofa perfectly.",
                "Her shoes match her handbag.",
                "Try to match the colors correctly.",
            ]
        },
        'wave': {
            'ocean': [
                "A huge wave crashed against the rocks.",
                "The wave carried the surfer toward shore.",
                "Children jumped over the small waves.",
                "Ocean waves eroded the coastline gradually.",
                "The wave pulled sand back into the sea.",
            ],
            'gesture': [
                "She gave a friendly wave from across the room.",
                "He returned the wave with a smile.",
                "A quick wave goodbye and she was gone.",
                "The parade included people waving flags.",
                "I saw her wave but couldn't reach her in time.",
            ],
            'physics': [
                "Sound travels as a wave through air.",
                "Light exhibits both wave and particle properties.",
                "The earthquake sent seismic waves outward.",
                "Radio wave frequencies vary by station.",
                "The wave pattern showed clear interference.",
            ]
        },
        'present': {
            'gift': [
                "She wrapped the birthday present carefully.",
                "The present was hidden under the tree.",
                "He bought a present for his mother.",
                "Opening presents is the best part of holidays.",
                "The present came with a beautiful ribbon.",
            ],
            'time': [
                "We should focus on the present moment.",
                "The present situation requires immediate action.",
                "Living in the present brings peace.",
                "At present, we have no further information.",
                "The present circumstances are challenging.",
            ],
            'attend': [
                "All members must be present at the meeting.",
                "Please confirm you will be present tomorrow.",
                "Those present voted unanimously.",
                "She was not present during the discussion.",
                "Only present members can participate.",
            ]
        }
    }

    # Hard negatives: similar context words but different sense
    HARD_NEGATIVES = {
        'bank': {
            'financial_looks_like_river': [
                "The bank was located right by the waterfront downtown.",  # financial
                "I walked along the bank to reach the ATM.",  # financial (sounds like river)
            ],
            'river_looks_like_financial': [
                "The bank held our deposits of silt and sediment.",  # river (sounds financial)
                "We made a withdrawal from the bank of the stream.",  # river (uses financial words)
            ]
        },
        'light': {
            'illumination_looks_like_weight': [
                "The light touch of the morning sun woke me.",  # illumination (uses 'touch')
                "She carried the light source to the heavy table.",  # illumination
            ],
            'weight_looks_like_illumination': [
                "The bright package was surprisingly light.",  # weight (uses 'bright')
                "She switched to a light load in the sunny room.",  # weight
            ]
        }
    }

    def generate_positive_pairs(self, count_per_word: int = 10) -> List[PilotExample]:
        """
        Generate same-sense pairs (should have aligned phases).
        """
        examples = []

        for word, senses in self.POLYSEMY_WORDS.items():
            for sense_name, contexts in senses.items():
                # Create pairs within the same sense
                for i, ctx1 in enumerate(contexts):
                    for ctx2 in contexts[i+1:]:
                        if len(examples) >= count_per_word * len(self.POLYSEMY_WORDS):
                            break

                        examples.append(PilotExample(
                            input_text=f"Context A: {ctx1}\nContext B: {ctx2}\n\nDo these use '{word}' in the same sense?",
                            output_text=f"Yes, both use '{word}' in the {sense_name} sense. The meaning is consistent.",
                            paradigm='semantic_phase',
                            subtype='polysemy_positive',
                            metadata={
                                'word': word,
                                'sense': sense_name,
                                'pair_type': 'same_sense',
                                'expected_phase': 'aligned'
                            }
                        ))

        return examples[:count_per_word * len(self.POLYSEMY_WORDS)]

    def generate_negative_pairs(self, count_per_word: int = 10) -> List[PilotExample]:
        """
        Generate different-sense pairs (should have separated phases).
        """
        examples = []

        for word, senses in self.POLYSEMY_WORDS.items():
            sense_names = list(senses.keys())
            for i, sense1 in enumerate(sense_names):
                for sense2 in sense_names[i+1:]:
                    ctx1_list = senses[sense1]
                    ctx2_list = senses[sense2]

                    for ctx1 in ctx1_list[:2]:
                        for ctx2 in ctx2_list[:2]:
                            examples.append(PilotExample(
                                input_text=f"Context A: {ctx1}\nContext B: {ctx2}\n\nDo these use '{word}' in the same sense?",
                                output_text=f"No, '{word}' has different meanings. Context A uses the {sense1} sense, while Context B uses the {sense2} sense.",
                                paradigm='semantic_phase',
                                subtype='polysemy_negative',
                                metadata={
                                    'word': word,
                                    'sense1': sense1,
                                    'sense2': sense2,
                                    'pair_type': 'different_sense',
                                    'expected_phase': 'separated'
                                }
                            ))

        random.shuffle(examples)
        return examples[:count_per_word * len(self.POLYSEMY_WORDS)]

    def generate_hard_negatives(self, count: int = 20) -> List[PilotExample]:
        """
        Generate hard negatives where context seems similar but sense differs.
        """
        examples = []

        for word, categories in self.HARD_NEGATIVES.items():
            for category, contexts in categories.items():
                for ctx in contexts:
                    actual_sense = category.split('_')[0]  # e.g., 'financial' from 'financial_looks_like_river'
                    looks_like = category.split('_')[-1]   # e.g., 'river'

                    examples.append(PilotExample(
                        input_text=f"What sense of '{word}' is used here?\n\n\"{ctx}\"",
                        output_text=f"Despite contextual cues that might suggest the {looks_like} sense, this sentence uses '{word}' in the {actual_sense} sense.",
                        paradigm='semantic_phase',
                        subtype='polysemy_hard_negative',
                        metadata={
                            'word': word,
                            'actual_sense': actual_sense,
                            'deceptive_sense': looks_like,
                            'difficulty': 'hard'
                        }
                    ))

        return examples[:count]

    def generate_all(self, total_count: int = 200) -> List[PilotExample]:
        """Generate balanced polysemy dataset."""
        positives = self.generate_positive_pairs(total_count // 4)
        negatives = self.generate_negative_pairs(total_count // 4)
        hard_negs = self.generate_hard_negatives(total_count // 4)

        all_examples = positives + negatives + hard_negs
        random.shuffle(all_examples)
        return all_examples[:total_count]


class LindbladGenerator:
    """
    Generate meaning-preserving paraphrases with structured corruption.

    Key insight: The win condition is a DEGRADATION CURVE.
    Baseline should fall off as noise increases.
    Lindblad-trained models should degrade smoothly.
    """

    # Base sentences with multiple paraphrase levels
    PARAPHRASE_SETS = [
        {
            'canonical': "The scientist discovered a new species of butterfly in the rainforest.",
            'paraphrases': [
                "A researcher found a previously unknown butterfly species in tropical jungle.",
                "New butterfly discovered by scientist in rainforest region.",
                "In the jungle, a scientist came across a butterfly species never seen before.",
                "A novel butterfly species was identified by a researcher in the rainforest.",
            ]
        },
        {
            'canonical': "The company announced record profits despite economic challenges.",
            'paraphrases': [
                "Record earnings were reported by the firm amid tough economic conditions.",
                "Despite economic difficulties, the corporation achieved unprecedented profits.",
                "The business declared its highest-ever profits even with economic headwinds.",
                "Amid economic turbulence, the company posted record-breaking profit figures.",
            ]
        },
        {
            'canonical': "Students protested against the university's decision to raise tuition fees.",
            'paraphrases': [
                "University students demonstrated in opposition to increased tuition costs.",
                "Protests erupted among students over the school's tuition hike decision.",
                "The university's fee increase sparked student demonstrations.",
                "Students staged protests against rising tuition at the university.",
            ]
        },
        {
            'canonical': "The ancient manuscript was carefully restored by expert conservators.",
            'paraphrases': [
                "Expert restorers painstakingly repaired the old manuscript.",
                "Conservation specialists meticulously restored the ancient document.",
                "The historical manuscript underwent careful restoration by experts.",
                "Skilled conservators worked to restore the centuries-old manuscript.",
            ]
        },
        {
            'canonical': "Climate change is causing glaciers to melt at an unprecedented rate.",
            'paraphrases': [
                "Glaciers are melting faster than ever due to climate change.",
                "Global warming is driving record-speed glacier melt.",
                "The rate of glacier melting has reached historic highs from climate change.",
                "Unprecedented glacier melt rates are linked to changing climate patterns.",
            ]
        },
        {
            'canonical': "The chef created an innovative dish using local ingredients.",
            'paraphrases': [
                "A creative new dish was crafted by the chef using locally-sourced produce.",
                "Using ingredients from the area, the chef invented an original dish.",
                "The chef's innovative creation featured locally-grown ingredients.",
                "Local produce inspired the chef's creative new culinary creation.",
            ]
        },
        {
            'canonical': "The rescue team saved three hikers stranded on the mountain.",
            'paraphrases': [
                "Three stranded hikers were rescued from the mountain by the rescue team.",
                "Rescue workers saved a trio of hikers who were stuck on the mountainside.",
                "The search team successfully rescued three hikers trapped on the mountain.",
                "Mountain rescue saved three people who had become stranded while hiking.",
            ]
        },
        {
            'canonical': "The museum acquired a rare painting from the Renaissance period.",
            'paraphrases': [
                "A rare Renaissance painting was purchased by the museum.",
                "The museum added a rare painting from the Renaissance era to its collection.",
                "A Renaissance-era painting of exceptional rarity joined the museum's holdings.",
                "The museum obtained an unusual painting dating to the Renaissance.",
            ]
        }
    ]

    @staticmethod
    def _add_typos(text: str) -> str:
        """Add realistic typos."""
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < 0.05 and chars[i].isalpha():
                if random.random() < 0.5:
                    # Swap with adjacent
                    if i < len(chars) - 1:
                        chars[i], chars[i+1] = chars[i+1], chars[i]
                else:
                    # Replace with nearby key
                    nearby = 'qwertyuiopasdfghjklzxcvbnm'
                    chars[i] = random.choice(nearby)
        return ''.join(chars)

    @staticmethod
    def _swap_adjacent_words(text: str) -> str:
        """Swap some adjacent words."""
        words = text.split()
        for i in range(len(words) - 1):
            if random.random() < 0.1:
                words[i], words[i+1] = words[i+1], words[i]
        return ' '.join(words)

    def _get_corruption_funcs(self):
        """Get corruption functions."""
        return {
            'typo': self._add_typos,
            'word_drop': lambda s: ' '.join(w for i, w in enumerate(s.split()) if random.random() > 0.15),
            'word_swap': self._swap_adjacent_words,
            'case_noise': lambda s: ''.join(c.upper() if random.random() > 0.85 else c.lower() if random.random() > 0.9 else c for c in s),
            'punctuation': lambda s: s.replace('.', '').replace(',', '').replace('!', '').replace('?', ''),
        }

    def generate_clean_pairs(self, count: int = 50) -> List[PilotExample]:
        """Generate meaning-equivalent pairs without corruption."""
        examples = []

        for pset in self.PARAPHRASE_SETS:
            canonical = pset['canonical']
            for para in pset['paraphrases']:
                examples.append(PilotExample(
                    input_text=f"Text A: {canonical}\nText B: {para}\n\nDo these sentences convey the same meaning?",
                    output_text="Yes, both sentences express the same core meaning despite different wording.",
                    paradigm='lindblad',
                    subtype='paraphrase_clean',
                    metadata={
                        'canonical': canonical,
                        'paraphrase': para,
                        'noise_level': 0.0,
                        'expected_basin': 'same'
                    }
                ))

        random.shuffle(examples)
        return examples[:count]

    def generate_noisy_pairs(self, count: int = 100) -> List[PilotExample]:
        """Generate pairs with increasing noise levels."""
        examples = []
        noise_levels = [0.1, 0.2, 0.3, 0.5, 0.7]
        corruption_funcs = self._get_corruption_funcs()

        for pset in self.PARAPHRASE_SETS:
            canonical = pset['canonical']
            for para in pset['paraphrases']:
                for noise_level in noise_levels:
                    # Apply multiple corruption types based on noise level
                    corrupted = para
                    for ctype, cfunc in corruption_funcs.items():
                        if random.random() < noise_level:
                            corrupted = cfunc(corrupted)

                    examples.append(PilotExample(
                        input_text=f"Text A: {canonical}\nText B: {corrupted}\n\nDespite surface differences, do these have the same meaning?",
                        output_text="Yes, the core meaning is preserved despite surface-level noise and variations.",
                        paradigm='lindblad',
                        subtype=f'paraphrase_noisy_{noise_level}',
                        metadata={
                            'canonical': canonical,
                            'original_paraphrase': para,
                            'corrupted': corrupted,
                            'noise_level': noise_level,
                            'expected_basin': 'same',
                            'corruptions_applied': list(corruption_funcs.keys())
                        }
                    ))

        random.shuffle(examples)
        return examples[:count]

    def generate_different_meaning(self, count: int = 30) -> List[PilotExample]:
        """Generate pairs that look similar but have different meanings."""
        examples = []

        similar_but_different = [
            ("The bank was closed for the holiday.", "The bank was eroded by the flooding."),
            ("She found the movie quite moving.", "She found the furniture quite heavy."),
            ("The plant grew rapidly in spring.", "The plant manufactured cars rapidly."),
            ("He left the room in tears.", "He left the paper in pieces."),
            ("The coach trained the team.", "The coach arrived at the station."),
        ]

        for sent1, sent2 in similar_but_different:
            examples.append(PilotExample(
                input_text=f"Text A: {sent1}\nText B: {sent2}\n\nDo these sentences have the same meaning?",
                output_text="No, these sentences have different meanings despite surface similarities.",
                paradigm='lindblad',
                subtype='different_meaning',
                metadata={
                    'text1': sent1,
                    'text2': sent2,
                    'expected_basin': 'different'
                }
            ))

        return examples[:count]

    def generate_all(self, total_count: int = 200) -> List[PilotExample]:
        """Generate balanced Lindblad dataset."""
        clean = self.generate_clean_pairs(total_count // 3)
        noisy = self.generate_noisy_pairs(total_count // 2)
        different = self.generate_different_meaning(total_count // 6)

        all_examples = clean + noisy + different
        random.shuffle(all_examples)
        return all_examples[:total_count]


class RetrocausalGenerator:
    """
    Generate effect→cause examples where forward heuristics fail.

    Key insight: The cause CANNOT be copied from the effect.
    The model must reason backward while keeping leak metrics clean.
    """

    # Effect→Cause pairs where effect doesn't contain the cause
    EFFECT_CAUSE_PAIRS = [
        {
            'effect': "The streets were wet and there were puddles everywhere.",
            'cause': "It had rained heavily during the night.",
            'red_herrings': ["The sprinklers were on.", "A water main broke.", "Morning dew collected."],
            'inference_type': 'physical'
        },
        {
            'effect': "The stock price dropped sharply at market open.",
            'cause': "The company announced disappointing quarterly earnings after hours.",
            'red_herrings': ["Investors panicked randomly.", "The CEO resigned.", "A competitor launched a product."],
            'inference_type': 'economic'
        },
        {
            'effect': "She was smiling and couldn't stop talking about her day.",
            'cause': "She had just received a job offer she'd been hoping for.",
            'red_herrings': ["She had coffee.", "The weather was nice.", "She saw a friend."],
            'inference_type': 'emotional'
        },
        {
            'effect': "The plant's leaves were yellowing and wilting.",
            'cause': "The plant hadn't been watered in two weeks.",
            'red_herrings': ["Too much sunlight.", "Pest infestation.", "Nutrient deficiency."],
            'inference_type': 'biological'
        },
        {
            'effect': "The house was unusually quiet when she arrived home.",
            'cause': "Her family had planned a surprise party and was hiding.",
            'red_herrings': ["Everyone was asleep.", "They had gone out.", "The power was out."],
            'inference_type': 'social'
        },
        {
            'effect': "The car wouldn't start despite a full tank of gas.",
            'cause': "The battery had died from leaving the lights on overnight.",
            'red_herrings': ["Bad fuel.", "Engine failure.", "Starter motor broken."],
            'inference_type': 'mechanical'
        },
        {
            'effect': "The meeting was postponed to next week.",
            'cause': "A key stakeholder had a scheduling conflict that couldn't be resolved.",
            'red_herrings': ["Bad weather.", "Technical issues.", "Holiday observed."],
            'inference_type': 'procedural'
        },
        {
            'effect': "The bread came out of the oven flat and dense.",
            'cause': "The yeast was expired and didn't activate during proofing.",
            'red_herrings': ["Oven too hot.", "Wrong flour type.", "Kneaded too long."],
            'inference_type': 'culinary'
        },
        {
            'effect': "The email was never received by the intended recipient.",
            'cause': "A typo in the email address sent it to the wrong person.",
            'red_herrings': ["Spam filter.", "Server down.", "Inbox full."],
            'inference_type': 'technical'
        },
        {
            'effect': "The team lost the championship game in the final seconds.",
            'cause': "A strategic timeout call was mismanaged, leaving them unprepared.",
            'red_herrings': ["Key player injured.", "Bad referee call.", "Opponent too strong."],
            'inference_type': 'strategic'
        }
    ]

    # Hypothesis revision examples - initial guess must be updated
    REVISION_EXAMPLES = [
        {
            'initial_observation': "The door was wide open when I arrived.",
            'initial_hypothesis': "Someone forgot to close it.",
            'new_evidence': "I then noticed muddy paw prints leading inside and heard barking.",
            'revised_conclusion': "The dog had pushed the door open to get inside.",
        },
        {
            'initial_observation': "The project deadline was suddenly moved up by a week.",
            'initial_hypothesis': "Management made an arbitrary decision.",
            'new_evidence': "Later I learned a major client had moved their launch date earlier.",
            'revised_conclusion': "The deadline change was driven by external client requirements.",
        },
        {
            'initial_observation': "My colleague seemed upset during the meeting.",
            'initial_hypothesis': "The project feedback was too harsh.",
            'new_evidence': "After the meeting, she mentioned receiving difficult personal news that morning.",
            'revised_conclusion': "Her mood was due to personal circumstances, not the meeting content.",
        },
    ]

    def generate_effect_cause(self, count: int = 80) -> List[PilotExample]:
        """Generate effect→cause inference examples."""
        examples = []

        for pair in self.EFFECT_CAUSE_PAIRS:
            # Main effect→cause
            examples.append(PilotExample(
                input_text=f"Observation: {pair['effect']}\n\nWhat is the most likely cause of this observation?",
                output_text=f"The most likely cause is: {pair['cause']}\n\nThis explains the observation because the effect logically follows from this cause.",
                paradigm='retrocausal',
                subtype='effect_to_cause',
                metadata={
                    'effect': pair['effect'],
                    'cause': pair['cause'],
                    'inference_type': pair['inference_type'],
                    'forward_copyable': False  # Cause NOT in effect
                }
            ))

            # With red herrings
            all_options = [pair['cause']] + pair['red_herrings']
            random.shuffle(all_options)
            options_text = '\n'.join(f"{i+1}. {opt}" for i, opt in enumerate(all_options))
            correct_idx = all_options.index(pair['cause']) + 1

            examples.append(PilotExample(
                input_text=f"Observation: {pair['effect']}\n\nPossible causes:\n{options_text}\n\nWhich cause is most likely?",
                output_text=f"Option {correct_idx} is correct: {pair['cause']}\n\nThe other options are plausible but less likely given the specific details of the observation.",
                paradigm='retrocausal',
                subtype='effect_to_cause_mcq',
                metadata={
                    'effect': pair['effect'],
                    'correct_cause': pair['cause'],
                    'red_herrings': pair['red_herrings'],
                    'inference_type': pair['inference_type']
                }
            ))

        random.shuffle(examples)
        return examples[:count]

    def generate_hypothesis_revision(self, count: int = 30) -> List[PilotExample]:
        """Generate examples requiring hypothesis revision."""
        examples = []

        for rev in self.REVISION_EXAMPLES:
            examples.append(PilotExample(
                input_text=f"""Initial observation: {rev['initial_observation']}
Initial hypothesis: {rev['initial_hypothesis']}
New evidence: {rev['new_evidence']}

How should the hypothesis be revised?""",
                output_text=f"Revised conclusion: {rev['revised_conclusion']}\n\nThe new evidence requires updating our initial hypothesis because it provides a more complete picture of the situation.",
                paradigm='retrocausal',
                subtype='hypothesis_revision',
                metadata={
                    'initial_hypothesis': rev['initial_hypothesis'],
                    'revised_conclusion': rev['revised_conclusion'],
                    'requires_revision': True
                }
            ))

        return examples[:count]

    def generate_all(self, total_count: int = 150) -> List[PilotExample]:
        """Generate balanced retrocausal dataset."""
        effect_cause = self.generate_effect_cause(int(total_count * 0.7))
        revision = self.generate_hypothesis_revision(int(total_count * 0.3))

        all_examples = effect_cause + revision
        random.shuffle(all_examples)
        return all_examples[:total_count]


class QualiaGenerator:
    """
    Generate examples with qualitative signals that correlate with measurable properties.

    Key insight: We don't need perfect labels.
    We need CONSISTENT correlations:
    - Certainty ↔ entropy/calibration
    - Novelty ↔ surprisal
    - Coherence ↔ internal agreement
    """

    # High certainty examples (clear, definitive statements)
    HIGH_CERTAINTY = [
        "Water freezes at 0 degrees Celsius at standard pressure.",
        "The Earth orbits the Sun once per year.",
        "2 + 2 = 4 in standard arithmetic.",
        "Paris is the capital of France.",
        "Mammals are warm-blooded vertebrates.",
    ]

    # Low certainty examples (hedged, uncertain statements)
    LOW_CERTAINTY = [
        "The meeting might be rescheduled, though we're not sure yet.",
        "Perhaps the results could indicate some correlation, maybe.",
        "It's somewhat possible that the trend will continue, arguably.",
        "There may or may not be a connection between these factors.",
        "The outcome remains uncertain pending further analysis.",
    ]

    # High novelty examples (surprising, unexpected)
    HIGH_NOVELTY = [
        "Scientists discovered that octopi can edit their own RNA on the fly.",
        "A black hole was found that shouldn't exist according to current models.",
        "The ancient manuscript contained a previously unknown language.",
        "Researchers found that plants can 'hear' and respond to sounds.",
        "A new species was discovered that doesn't fit any known classification.",
    ]

    # Low novelty examples (expected, routine)
    LOW_NOVELTY = [
        "The sun rose in the east this morning as usual.",
        "Traffic was heavy during rush hour as expected.",
        "The quarterly report showed results in line with projections.",
        "Water flows downhill following the path of least resistance.",
        "The meeting started at the scheduled time.",
    ]

    # High coherence examples (internally consistent)
    HIGH_COHERENCE = [
        "The argument followed logically: A leads to B, B leads to C, therefore A leads to C.",
        "All evidence pointed to the same conclusion without contradiction.",
        "The theory explains observations at multiple scales consistently.",
        "Each piece of the puzzle fit perfectly with the others.",
        "The data supported the hypothesis from every angle we examined.",
    ]

    # Low coherence examples (contradictory, confusing)
    LOW_COHERENCE = [
        "The report said sales increased but also that revenue declined simultaneously.",
        "On one hand it's clear, but on the other hand it's completely uncertain.",
        "The witness claimed to be elsewhere but also present at the scene.",
        "The policy aims to reduce spending while increasing all budget items.",
        "The study found both that X causes Y and that Y prevents X.",
    ]

    def generate_certainty_examples(self, count: int = 40) -> List[PilotExample]:
        """Generate examples with clear certainty signals."""
        examples = []

        for text in self.HIGH_CERTAINTY:
            examples.append(PilotExample(
                input_text=f"Statement: {text}\n\nRate the certainty of this statement.",
                output_text="This statement expresses high certainty. It is definitive, clear, and leaves no room for ambiguity.",
                paradigm='qualia',
                subtype='certainty_high',
                metadata={'certainty_level': 'high', 'expected_entropy': 'low'}
            ))

        for text in self.LOW_CERTAINTY:
            examples.append(PilotExample(
                input_text=f"Statement: {text}\n\nRate the certainty of this statement.",
                output_text="This statement expresses low certainty. It uses hedging language and leaves significant ambiguity.",
                paradigm='qualia',
                subtype='certainty_low',
                metadata={'certainty_level': 'low', 'expected_entropy': 'high'}
            ))

        random.shuffle(examples)
        return examples[:count]

    def generate_novelty_examples(self, count: int = 40) -> List[PilotExample]:
        """Generate examples with clear novelty signals."""
        examples = []

        for text in self.HIGH_NOVELTY:
            examples.append(PilotExample(
                input_text=f"Information: {text}\n\nRate how novel or surprising this information is.",
                output_text="This is highly novel information. It presents unexpected findings that challenge existing understanding.",
                paradigm='qualia',
                subtype='novelty_high',
                metadata={'novelty_level': 'high', 'expected_surprisal': 'high'}
            ))

        for text in self.LOW_NOVELTY:
            examples.append(PilotExample(
                input_text=f"Information: {text}\n\nRate how novel or surprising this information is.",
                output_text="This is not novel. It describes expected, routine occurrences that match prior expectations.",
                paradigm='qualia',
                subtype='novelty_low',
                metadata={'novelty_level': 'low', 'expected_surprisal': 'low'}
            ))

        random.shuffle(examples)
        return examples[:count]

    def generate_coherence_examples(self, count: int = 40) -> List[PilotExample]:
        """Generate examples with clear coherence signals."""
        examples = []

        for text in self.HIGH_COHERENCE:
            examples.append(PilotExample(
                input_text=f"Passage: {text}\n\nRate the internal coherence of this passage.",
                output_text="This passage shows high coherence. All elements align logically without contradiction.",
                paradigm='qualia',
                subtype='coherence_high',
                metadata={'coherence_level': 'high', 'internal_agreement': 'strong'}
            ))

        for text in self.LOW_COHERENCE:
            examples.append(PilotExample(
                input_text=f"Passage: {text}\n\nRate the internal coherence of this passage.",
                output_text="This passage shows low coherence. It contains contradictions or logically inconsistent elements.",
                paradigm='qualia',
                subtype='coherence_low',
                metadata={'coherence_level': 'low', 'internal_agreement': 'weak'}
            ))

        random.shuffle(examples)
        return examples[:count]

    def generate_all(self, total_count: int = 150) -> List[PilotExample]:
        """Generate balanced qualia dataset."""
        certainty = self.generate_certainty_examples(total_count // 3)
        novelty = self.generate_novelty_examples(total_count // 3)
        coherence = self.generate_coherence_examples(total_count // 3)

        all_examples = certainty + novelty + coherence
        random.shuffle(all_examples)
        return all_examples[:total_count]


def generate_pilot_dataset(
    output_path: str,
    polysemy_count: int = 200,
    lindblad_count: int = 200,
    retrocausal_count: int = 150,
    qualia_count: int = 150
) -> Path:
    """Generate complete pilot dataset for paradigm validation."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PILOT DATASET GENERATION")
    print("  Surgical data for paradigm validation")
    print("=" * 60)

    all_examples = []

    # Polysemy
    print(f"\n[1/4] Generating polysemy examples ({polysemy_count})...")
    polysemy_gen = PolysemyGenerator()
    polysemy_examples = polysemy_gen.generate_all(polysemy_count)
    all_examples.extend(polysemy_examples)
    print(f"       Generated {len(polysemy_examples)} polysemy examples")

    # Lindblad
    print(f"\n[2/4] Generating Lindblad examples ({lindblad_count})...")
    lindblad_gen = LindbladGenerator()
    lindblad_examples = lindblad_gen.generate_all(lindblad_count)
    all_examples.extend(lindblad_examples)
    print(f"       Generated {len(lindblad_examples)} Lindblad examples")

    # Retrocausal
    print(f"\n[3/4] Generating retrocausal examples ({retrocausal_count})...")
    retrocausal_gen = RetrocausalGenerator()
    retrocausal_examples = retrocausal_gen.generate_all(retrocausal_count)
    all_examples.extend(retrocausal_examples)
    print(f"       Generated {len(retrocausal_examples)} retrocausal examples")

    # Qualia
    print(f"\n[4/4] Generating qualia examples ({qualia_count})...")
    qualia_gen = QualiaGenerator()
    qualia_examples = qualia_gen.generate_all(qualia_count)
    all_examples.extend(qualia_examples)
    print(f"       Generated {len(qualia_examples)} qualia examples")

    # Shuffle and write
    random.shuffle(all_examples)

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example.to_dict()) + '\n')

    # Summary
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total examples: {len(all_examples)}")
    print(f"  Output file: {output_path}")

    # Breakdown by paradigm
    paradigm_counts = {}
    subtype_counts = {}
    for ex in all_examples:
        paradigm_counts[ex.paradigm] = paradigm_counts.get(ex.paradigm, 0) + 1
        key = f"{ex.paradigm}/{ex.subtype}"
        subtype_counts[key] = subtype_counts.get(key, 0) + 1

    print("\n  By paradigm:")
    for paradigm, count in sorted(paradigm_counts.items()):
        print(f"    {paradigm}: {count}")

    print("\n  By subtype:")
    for subtype, count in sorted(subtype_counts.items()):
        print(f"    {subtype}: {count}")

    print("=" * 60)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate pilot dataset for QLLM paradigm validation")
    parser.add_argument('--output', type=str, default='./data/pilot_paradigm_data.jsonl',
                        help='Output path for dataset')
    parser.add_argument('--paradigm', type=str, default=None,
                        choices=['polysemy', 'lindblad', 'retrocausal', 'qualia'],
                        help='Generate only one paradigm')
    parser.add_argument('--count', type=int, default=200,
                        help='Examples per paradigm (or total if --paradigm specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    random.seed(args.seed)

    if args.paradigm:
        # Single paradigm
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generators = {
            'polysemy': PolysemyGenerator,
            'lindblad': LindbladGenerator,
            'retrocausal': RetrocausalGenerator,
            'qualia': QualiaGenerator
        }

        gen = generators[args.paradigm]()
        examples = gen.generate_all(args.count)

        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + '\n')

        print(f"Generated {len(examples)} {args.paradigm} examples -> {output_path}")
    else:
        # Full dataset
        generate_pilot_dataset(
            args.output,
            polysemy_count=args.count,
            lindblad_count=args.count,
            retrocausal_count=int(args.count * 0.75),
            qualia_count=int(args.count * 0.75)
        )


if __name__ == "__main__":
    main()
