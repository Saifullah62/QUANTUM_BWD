"""
Evaluation Generators
=====================

Two sanity evals for semantic retrieval:

1. Multi-Sense Retrieval Eval:
   "Does retrieval pick the right meaning under ambiguity?"
   - Mini-corpora with passages containing multiple senses
   - Score: correct sense ranked top-1/top-3?
   - Stratified by cue strength

2. Multi-Step Coherence Eval:
   "Does an agent stay consistent when same word appears under two meanings?"
   - Two/three-step scripts forcing sense A then sense B
   - Check state coherence across steps

Also: Eval-0 Golden Canary (frozen regression gate)
"""

import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RetrievalEvalItem:
    """A single retrieval evaluation item."""
    query_id: str
    query_text: str
    query_span: Dict
    query_sense_id: str
    query_lemma: str
    query_cue_strength: float
    candidate_passages: List[Dict]  # Mixed senses
    correct_passage_ids: List[str]


@dataclass
class CoherenceEvalItem:
    """A multi-step coherence evaluation item."""
    eval_id: str
    lemma: str
    steps: List[Dict]  # Each step has sense_id, context, expected_behavior
    coherence_checks: List[Dict]  # Points where we verify consistency


class EvalGenerator:
    """Generates evaluation sets from the v2.2 data."""

    def __init__(self, events: List[Dict], seed: int = 42):
        self.events = events
        self.seed = seed
        random.seed(seed)

        # Build indices
        self.by_lemma: Dict[str, List[Dict]] = defaultdict(list)
        self.by_sense: Dict[str, List[Dict]] = defaultdict(list)

        for event in events:
            lemma = event.get('lemma', '').lower()
            sense_id = event.get('sense_id', '')
            if lemma:
                self.by_lemma[lemma].append(event)
            if sense_id:
                self.by_sense[sense_id].append(event)

    def _get_cue_strength(self, event: Dict) -> float:
        """Extract cue strength from event."""
        quality = event.get('quality', {})
        if isinstance(quality, dict):
            return quality.get('cue_strength', 0.5)
        return 0.5

    def _get_cue_tier(self, cue_strength: float) -> str:
        """Classify cue strength into tier."""
        if cue_strength >= 0.7:
            return "high_cue"
        elif cue_strength >= 0.4:
            return "medium_cue"
        else:
            return "low_cue"

    def generate_retrieval_eval(self, num_items: int = 500) -> List[Dict]:
        """
        Generate multi-sense retrieval evaluation items.

        Each item:
        - Query context with target lemma
        - Candidate passages with MIXED senses (some correct, some wrong)
        - Task: rank correct-sense passages above wrong-sense passages
        """
        eval_items = []

        # Focus on highly polysemous lemmas
        polysemous_lemmas = [
            (lemma, events) for lemma, events in self.by_lemma.items()
            if len(set(e.get('sense_id') for e in events)) >= 3
        ]

        random.shuffle(polysemous_lemmas)

        for lemma, lemma_events in polysemous_lemmas:
            if len(eval_items) >= num_items:
                break

            # Group by sense
            by_sense = defaultdict(list)
            for event in lemma_events:
                sense_id = event.get('sense_id', '')
                by_sense[sense_id].append(event)

            # Need multiple senses with multiple examples
            valid_senses = [(sid, events) for sid, events in by_sense.items()
                           if len(events) >= 2]

            if len(valid_senses) < 2:
                continue

            # Create eval item for each sense
            for target_sense_id, target_events in valid_senses[:3]:
                if len(eval_items) >= num_items:
                    break

                # Select query
                query_event = random.choice(target_events)
                query_cue = self._get_cue_strength(query_event)

                # Build candidate pool
                candidates = []

                # Add correct-sense passages (excluding query)
                correct_ids = []
                for event in target_events:
                    if event != query_event:
                        cid = hashlib.md5(event.get('text', '').encode()).hexdigest()[:8]
                        candidates.append({
                            'passage_id': cid,
                            'text': event.get('text', ''),
                            'span': event.get('span', {}),
                            'sense_id': event.get('sense_id', ''),
                            'is_correct_sense': True
                        })
                        correct_ids.append(cid)

                # Add wrong-sense passages
                for other_sense_id, other_events in by_sense.items():
                    if other_sense_id == target_sense_id:
                        continue
                    for event in other_events[:2]:  # Limit per sense
                        cid = hashlib.md5(event.get('text', '').encode()).hexdigest()[:8]
                        candidates.append({
                            'passage_id': cid,
                            'text': event.get('text', ''),
                            'span': event.get('span', {}),
                            'sense_id': event.get('sense_id', ''),
                            'is_correct_sense': False
                        })

                if not correct_ids or len(candidates) < 3:
                    continue

                # Shuffle candidates
                random.shuffle(candidates)

                eval_item = {
                    'eval_id': hashlib.md5(f"{lemma}:{target_sense_id}:{self.seed}".encode()).hexdigest()[:12],
                    'eval_type': 'multi_sense_retrieval',
                    'lemma': lemma,
                    'query': {
                        'text': query_event.get('text', ''),
                        'span': query_event.get('span', {}),
                        'sense_id': target_sense_id,
                        'cue_strength': query_cue,
                        'cue_tier': self._get_cue_tier(query_cue)
                    },
                    'candidates': candidates,
                    'correct_passage_ids': correct_ids,
                    'num_correct': len(correct_ids),
                    'num_distractors': len(candidates) - len(correct_ids),
                    'scoring': {
                        'metric': 'correct_sense_in_top_k',
                        'k_values': [1, 3, 5]
                    }
                }

                eval_items.append(eval_item)

        # Stratify by cue tier
        tier_distribution = defaultdict(int)
        for item in eval_items:
            tier = item['query']['cue_tier']
            tier_distribution[tier] += 1

        print(f"  Retrieval eval stratification:")
        for tier, count in sorted(tier_distribution.items()):
            print(f"    {tier}: {count} ({100*count/len(eval_items):.1f}%)")

        return eval_items

    def generate_coherence_eval(self, num_items: int = 200) -> List[Dict]:
        """
        Generate multi-step coherence evaluation items.

        Each item:
        - 2-3 step workflow using same lemma in different senses
        - Step 1 forces Sense A, Step 2 forces Sense B
        - Check: Does system correctly track sense shift?
        """
        eval_items = []

        # Need lemmas with very distinct senses
        polysemous_lemmas = [
            (lemma, events) for lemma, events in self.by_lemma.items()
            if len(set(e.get('sense_id') for e in events)) >= 2
        ]

        random.shuffle(polysemous_lemmas)

        for lemma, lemma_events in polysemous_lemmas:
            if len(eval_items) >= num_items:
                break

            # Group by sense
            by_sense = defaultdict(list)
            for event in lemma_events:
                sense_id = event.get('sense_id', '')
                by_sense[sense_id].append(event)

            sense_list = [(sid, events) for sid, events in by_sense.items()
                         if len(events) >= 1]

            if len(sense_list) < 2:
                continue

            # Pick two distinct senses
            random.shuffle(sense_list)
            sense_a_id, sense_a_events = sense_list[0]
            sense_b_id, sense_b_events = sense_list[1]

            sense_a_example = random.choice(sense_a_events)
            sense_b_example = random.choice(sense_b_events)

            # Create workflow steps
            steps = [
                {
                    'step_num': 1,
                    'context': sense_a_example.get('text', ''),
                    'span': sense_a_example.get('span', {}),
                    'expected_sense': sense_a_id,
                    'sense_gloss': sense_a_example.get('sense_gloss', ''),
                    'instruction': f"Process this context where '{lemma}' means: {sense_a_example.get('sense_gloss', '')[:100]}"
                },
                {
                    'step_num': 2,
                    'context': sense_b_example.get('text', ''),
                    'span': sense_b_example.get('span', {}),
                    'expected_sense': sense_b_id,
                    'sense_gloss': sense_b_example.get('sense_gloss', ''),
                    'instruction': f"Now process this context where '{lemma}' has shifted to mean: {sense_b_example.get('sense_gloss', '')[:100]}"
                }
            ]

            # Optional third step returning to first sense
            if len(sense_a_events) > 1 and random.random() > 0.5:
                sense_a_alt = [e for e in sense_a_events if e != sense_a_example]
                if sense_a_alt:
                    alt_example = random.choice(sense_a_alt)
                    steps.append({
                        'step_num': 3,
                        'context': alt_example.get('text', ''),
                        'span': alt_example.get('span', {}),
                        'expected_sense': sense_a_id,
                        'sense_gloss': alt_example.get('sense_gloss', ''),
                        'instruction': f"Return to original sense: '{lemma}' meaning {alt_example.get('sense_gloss', '')[:100]}"
                    })

            # Coherence checks
            coherence_checks = []
            for i in range(len(steps)):
                coherence_checks.append({
                    'check_point': f"after_step_{i+1}",
                    'expected_active_sense': steps[i]['expected_sense'],
                    'check_type': 'sense_consistency',
                    'description': f"Verify system has correctly identified '{lemma}' as sense: {steps[i]['expected_sense']}"
                })

                if i > 0:
                    # Check for sense shift detection
                    prev_sense = steps[i-1]['expected_sense']
                    curr_sense = steps[i]['expected_sense']
                    if prev_sense != curr_sense:
                        coherence_checks.append({
                            'check_point': f"transition_{i}_to_{i+1}",
                            'from_sense': prev_sense,
                            'to_sense': curr_sense,
                            'check_type': 'sense_transition',
                            'description': f"Verify system detected sense shift from {prev_sense} to {curr_sense}"
                        })

            eval_item = {
                'eval_id': hashlib.md5(f"{lemma}:coherence:{self.seed}:{len(eval_items)}".encode()).hexdigest()[:12],
                'eval_type': 'multi_step_coherence',
                'lemma': lemma,
                'num_steps': len(steps),
                'num_sense_shifts': sum(1 for i in range(1, len(steps))
                                        if steps[i]['expected_sense'] != steps[i-1]['expected_sense']),
                'steps': steps,
                'coherence_checks': coherence_checks,
                'scoring': {
                    'metric': 'coherence_accuracy',
                    'checks': ['per_step_sense_match', 'transition_detection', 'final_state_correct']
                }
            }

            eval_items.append(eval_item)

        return eval_items

    def create_golden_canary(self, num_items: int = 200) -> Dict:
        """
        Create Eval-0: The frozen golden canary regression gate.

        This set is NEVER modified and serves as the regression test.
        If any change makes this worse, stop.
        """
        # Sample diverse, high-quality items
        eval_items = []

        # Balance across difficulty tiers
        tier_targets = {
            'high_cue': num_items // 3,
            'medium_cue': num_items // 3,
            'low_cue': num_items - 2 * (num_items // 3)
        }

        tier_items = defaultdict(list)

        for lemma, events in self.by_lemma.items():
            senses = set(e.get('sense_id') for e in events)
            if len(senses) < 2:
                continue

            for event in events:
                cue = self._get_cue_strength(event)
                tier = self._get_cue_tier(cue)

                if len(tier_items[tier]) < tier_targets.get(tier, 0):
                    tier_items[tier].append({
                        'text': event.get('text', ''),
                        'lemma': lemma,
                        'span': event.get('span', {}),
                        'sense_id': event.get('sense_id', ''),
                        'sense_gloss': event.get('sense_gloss', ''),
                        'cue_strength': cue,
                        'cue_tier': tier
                    })

        # Combine
        for tier, items in tier_items.items():
            eval_items.extend(items)

        random.shuffle(eval_items)

        canary = {
            'version': 'eval_0_frozen',
            'created': datetime.now().isoformat(),
            'frozen': True,
            'description': 'Golden canary regression gate - NEVER MODIFY',
            'purpose': 'If any change makes performance on this set worse, stop and investigate',
            'num_items': len(eval_items),
            'tier_distribution': {tier: len(items) for tier, items in tier_items.items()},
            'items': eval_items,
            'checksum': hashlib.sha256(json.dumps(eval_items, sort_keys=True).encode()).hexdigest()
        }

        return canary


def main():
    """Generate all evaluation sets."""
    print("=" * 70)
    print("EVALUATION SET GENERATION")
    print("=" * 70)

    # Load canonicalized events
    processed_dir = Path("paradigm_factory/v2/processed")
    events_file = processed_dir / "canonicalized_v21.jsonl"

    print(f"\nLoading events from {events_file}...")
    events = []
    with open(events_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    print(f"  Loaded: {len(events)} events")

    # Create generator
    generator = EvalGenerator(events, seed=42)

    # Create output directory
    eval_dir = Path("paradigm_factory/v2/evals")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Generate Eval-0 Golden Canary
    print("\n--- Creating Eval-0 Golden Canary ---")
    canary = generator.create_golden_canary(num_items=300)
    canary_file = eval_dir / "eval_0_golden_canary_FROZEN.json"
    with open(canary_file, 'w', encoding='utf-8') as f:
        json.dump(canary, f, indent=2)
    print(f"  Saved: {canary_file}")
    print(f"  Items: {canary['num_items']}")
    print(f"  Checksum: {canary['checksum'][:16]}...")
    print(f"  Tier distribution: {canary['tier_distribution']}")

    # Generate Multi-Sense Retrieval Eval
    print("\n--- Generating Multi-Sense Retrieval Eval ---")
    retrieval_items = generator.generate_retrieval_eval(num_items=500)
    retrieval_file = eval_dir / "eval_multi_sense_retrieval.jsonl"
    with open(retrieval_file, 'w', encoding='utf-8') as f:
        for item in retrieval_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved: {retrieval_file}")
    print(f"  Items: {len(retrieval_items)}")

    # Generate Multi-Step Coherence Eval
    print("\n--- Generating Multi-Step Coherence Eval ---")
    coherence_items = generator.generate_coherence_eval(num_items=200)
    coherence_file = eval_dir / "eval_multi_step_coherence.jsonl"
    with open(coherence_file, 'w', encoding='utf-8') as f:
        for item in coherence_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved: {coherence_file}")
    print(f"  Items: {len(coherence_items)}")

    # Summary stats
    stats = {
        'timestamp': datetime.now().isoformat(),
        'eval_0_golden_canary': {
            'file': str(canary_file),
            'num_items': canary['num_items'],
            'checksum': canary['checksum'],
            'frozen': True
        },
        'multi_sense_retrieval': {
            'file': str(retrieval_file),
            'num_items': len(retrieval_items)
        },
        'multi_step_coherence': {
            'file': str(coherence_file),
            'num_items': len(coherence_items)
        }
    }

    stats_file = eval_dir / "eval_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n  Eval-0 Golden Canary: {canary['num_items']} items (FROZEN)")
    print(f"  Multi-Sense Retrieval: {len(retrieval_items)} items")
    print(f"  Multi-Step Coherence: {len(coherence_items)} items")
    print(f"\n  Output: {eval_dir}")


if __name__ == "__main__":
    main()
