"""
Enhanced Bundle Generator v2.2
==============================

Improvements over v2.1:
1. Positive-present enforcement (>95% of contrastive bundles have positive)
2. Difficulty tiers based on cue_strength and ambiguity_risk
3. Two-pool hard negative mining (within-lemma + cross-lemma)
4. Gloss-matching auxiliary bundles
5. Source diversity in positive sampling
"""

import json
import random
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class DifficultyTier:
    """Classification for example difficulty."""
    HIGH_CUE_LOW_RISK = "tier1_easy"      # Strong cues, low ambiguity - teaches attention
    MEDIUM_CUE = "tier2_robust"            # Medium cues - builds robustness
    LOW_CUE_HIGH_RISK = "tier3_adversarial"  # Weak cues, high ambiguity - stress test


class EnhancedBundleGenerator:
    """
    Enhanced bundle generator with:
    - Positive-present enforcement
    - Difficulty tiering
    - Cross-lemma hard negatives
    - Gloss-matching objectives
    """

    def __init__(self, events: List[Dict], seed: int = 42):
        self.events = events
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Primary indices
        self.by_lemma: Dict[str, List[Dict]] = defaultdict(list)
        self.by_sense: Dict[str, List[Dict]] = defaultdict(list)
        self.by_pos: Dict[str, List[Dict]] = defaultdict(list)
        self.by_source: Dict[str, List[Dict]] = defaultdict(list)
        self.by_topic: Dict[str, List[Dict]] = defaultdict(list)

        # Difficulty classification
        self.by_difficulty: Dict[str, List[Dict]] = defaultdict(list)

        # Cross-lemma similarity index (for hard negative mining)
        self.topic_neighbors: Dict[str, Set[str]] = defaultdict(set)

        self._build_indices()

    def _get_difficulty_tier(self, event: Dict) -> str:
        """Classify event into difficulty tier."""
        quality = event.get('quality', {})
        if isinstance(quality, dict):
            cue_strength = quality.get('cue_strength', 0.5)
            ambiguity_risk = quality.get('ambiguity_risk', 0.5)
        else:
            cue_strength = 0.5
            ambiguity_risk = 0.5

        # Tier classification
        if cue_strength >= 0.7 and ambiguity_risk <= 0.3:
            return DifficultyTier.HIGH_CUE_LOW_RISK
        elif cue_strength <= 0.4 or ambiguity_risk >= 0.6:
            return DifficultyTier.LOW_CUE_HIGH_RISK
        else:
            return DifficultyTier.MEDIUM_CUE

    def _build_indices(self):
        """Build all indices for efficient bundle generation."""
        print("  Building indices...")

        for event in self.events:
            lemma = event.get('lemma', '').lower()
            sense_id = event.get('sense_id', '')
            pos = event.get('pos', 'noun')
            source = event.get('_sense_source', event.get('source', {}).get('domain', 'unknown'))
            topics = event.get('topic_tags', [])
            difficulty = self._get_difficulty_tier(event)

            if lemma:
                self.by_lemma[lemma].append(event)
            if sense_id:
                self.by_sense[sense_id].append(event)
            if pos:
                self.by_pos[pos].append(event)
            if source:
                self.by_source[source].append(event)

            self.by_difficulty[difficulty].append(event)

            # Topic indexing for cross-lemma neighbors
            for topic in topics:
                if topic and topic not in ['noun', 'verb', 'adjective', 'adverb']:
                    self.by_topic[topic].append(event)
                    if lemma:
                        self.topic_neighbors[lemma].add(topic)

        # Build topic-based lemma neighbors
        self._build_topic_neighbors()

        print(f"    Lemmas: {len(self.by_lemma)}")
        print(f"    Senses: {len(self.by_sense)}")
        print(f"    Difficulty tiers: {dict((k, len(v)) for k, v in self.by_difficulty.items())}")

    def _build_topic_neighbors(self):
        """Build cross-lemma neighbors based on topic overlap."""
        # Group lemmas by their topics
        topic_to_lemmas: Dict[str, Set[str]] = defaultdict(set)

        for lemma, topics in self.topic_neighbors.items():
            for topic in topics:
                topic_to_lemmas[topic].add(lemma)

        # Build neighbor sets (lemmas that share topics)
        self.lemma_neighbors: Dict[str, Set[str]] = defaultdict(set)

        for topic, lemmas in topic_to_lemmas.items():
            for lemma in lemmas:
                self.lemma_neighbors[lemma].update(lemmas - {lemma})

    def _find_positive(self, anchor: Dict, require_different_source: bool = True) -> Optional[Dict]:
        """
        Find a positive example (same sense, different text).
        Prioritizes different sources for diversity.
        """
        sense_id = anchor.get('sense_id', '')
        anchor_text = anchor.get('text', '').lower()
        anchor_source = anchor.get('_sense_source', '')

        if not sense_id or sense_id not in self.by_sense:
            return None

        same_sense = self.by_sense[sense_id]

        # Filter candidates
        candidates = []
        diff_source_candidates = []

        for event in same_sense:
            if event.get('text', '').lower() == anchor_text:
                continue  # Skip identical text

            event_source = event.get('_sense_source', '')

            if event_source != anchor_source:
                diff_source_candidates.append(event)
            else:
                candidates.append(event)

        # Prefer different source
        if diff_source_candidates:
            return random.choice(diff_source_candidates)
        elif candidates and not require_different_source:
            return random.choice(candidates)

        return None

    def _find_within_lemma_negatives(self, anchor: Dict, max_count: int = 2) -> List[Dict]:
        """Find hard negatives from same lemma, different sense."""
        lemma = anchor.get('lemma', '').lower()
        anchor_sense = anchor.get('sense_id', '')

        if not lemma or lemma not in self.by_lemma:
            return []

        # Get events with different senses
        candidates = [
            e for e in self.by_lemma[lemma]
            if e.get('sense_id') != anchor_sense
        ]

        if not candidates:
            return []

        # Sort by "confusability" - prefer same POS, similar topic overlap
        anchor_pos = anchor.get('pos', '')
        anchor_topics = set(anchor.get('topic_tags', []))

        def confusability_score(event):
            score = 0
            if event.get('pos') == anchor_pos:
                score += 2
            event_topics = set(event.get('topic_tags', []))
            score += len(anchor_topics & event_topics)
            return score

        candidates.sort(key=confusability_score, reverse=True)

        return candidates[:max_count]

    def _find_cross_lemma_negatives(self, anchor: Dict, max_count: int = 1) -> List[Dict]:
        """Find hard negatives from different lemmas in same topic neighborhood."""
        lemma = anchor.get('lemma', '').lower()
        anchor_sense = anchor.get('sense_id', '')
        anchor_topics = set(anchor.get('topic_tags', []))

        if not lemma:
            return []

        # Get neighbor lemmas
        neighbor_lemmas = self.lemma_neighbors.get(lemma, set())

        if not neighbor_lemmas:
            # Fallback: sample from same topics
            candidates = []
            for topic in anchor_topics:
                if topic in self.by_topic:
                    for event in self.by_topic[topic]:
                        if event.get('lemma', '').lower() != lemma:
                            candidates.append(event)
            if candidates:
                return random.sample(candidates, min(max_count, len(candidates)))
            return []

        # Sample from neighbors
        candidates = []
        for neighbor in neighbor_lemmas:
            if neighbor in self.by_lemma:
                candidates.extend(self.by_lemma[neighbor][:3])  # Limit per neighbor

        if candidates:
            return random.sample(candidates, min(max_count, len(candidates)))

        return []

    def create_contrastive_bundle(self, anchor: Dict,
                                   enforce_positive: bool = True,
                                   within_lemma_negs: int = 2,
                                   cross_lemma_negs: int = 1) -> Optional[Dict]:
        """
        Create an enhanced contrastive bundle with:
        - Enforced positive (if possible)
        - Two-pool negatives (within-lemma + cross-lemma)
        - Difficulty tier annotation
        """
        lemma = anchor.get('lemma', '').lower()

        if not lemma:
            return None

        # Find positive (try hard to get one)
        positive = self._find_positive(anchor, require_different_source=True)
        if not positive:
            positive = self._find_positive(anchor, require_different_source=False)

        # If we require positive and couldn't find one, skip this bundle
        if enforce_positive and not positive:
            return None

        # Find negatives from both pools
        within_negs = self._find_within_lemma_negatives(anchor, within_lemma_negs)
        cross_negs = self._find_cross_lemma_negatives(anchor, cross_lemma_negs)

        # Need at least some negatives
        if not within_negs and not cross_negs:
            return None

        all_negatives = within_negs + cross_negs

        # Classify difficulty
        anchor_difficulty = self._get_difficulty_tier(anchor)

        bundle = {
            'id': hashlib.md5(f"{anchor['id']}:{self.seed}:v22".encode()).hexdigest()[:12],
            'lemma': lemma,
            'anchor': self._create_bundle_item(anchor),
            'positive': self._create_bundle_item(positive) if positive else None,
            'negatives': {
                'within_lemma': [self._create_bundle_item(n) for n in within_negs],
                'cross_lemma': [self._create_bundle_item(n) for n in cross_negs]
            },
            'metadata': {
                'anchor_sense': anchor.get('sense_id', ''),
                'has_positive': positive is not None,
                'num_within_lemma_negs': len(within_negs),
                'num_cross_lemma_negs': len(cross_negs),
                'difficulty_tier': anchor_difficulty,
                'bundle_type': 'contrastive_v22'
            }
        }

        return bundle

    def create_gloss_matching_bundle(self, lemma: str) -> Optional[Dict]:
        """
        Create a gloss-matching bundle for auxiliary training.
        Task: Given an example, rank the correct sense gloss above other glosses.
        """
        if lemma not in self.by_lemma:
            return None

        events = self.by_lemma[lemma]

        # Group by sense
        by_sense = defaultdict(list)
        for event in events:
            sense_id = event.get('sense_id', 'unknown')
            by_sense[sense_id].append(event)

        if len(by_sense) < 2:
            return None

        # Build gloss inventory for this lemma
        glosses = {}
        for sense_id, sense_events in by_sense.items():
            gloss = sense_events[0].get('sense_gloss', '')
            if gloss:
                glosses[sense_id] = gloss[:200]  # Truncate long glosses

        if len(glosses) < 2:
            return None

        # Create examples with correct gloss target
        examples = []
        for sense_id, sense_events in by_sense.items():
            if sense_id not in glosses:
                continue

            # Sample up to 2 examples per sense
            sample = random.sample(sense_events, min(2, len(sense_events)))

            for event in sample:
                examples.append({
                    'text': event.get('text', ''),
                    'span': event.get('span', {}),
                    'correct_sense_id': sense_id,
                    'correct_gloss': glosses[sense_id],
                    'distractor_glosses': {
                        sid: g for sid, g in glosses.items() if sid != sense_id
                    }
                })

        if not examples:
            return None

        bundle = {
            'id': hashlib.md5(f"{lemma}:gloss:{self.seed}".encode()).hexdigest()[:12],
            'lemma': lemma,
            'gloss_inventory': glosses,
            'examples': examples,
            'metadata': {
                'num_senses': len(glosses),
                'num_examples': len(examples),
                'bundle_type': 'gloss_matching'
            }
        }

        return bundle

    def create_discrimination_bundle(self, lemma: str) -> Optional[Dict]:
        """Create a sense discrimination bundle (unchanged from v2.1)."""
        if lemma not in self.by_lemma:
            return None

        events = self.by_lemma[lemma]

        # Group by sense
        by_sense = defaultdict(list)
        for event in events:
            sense_id = event.get('sense_id', 'unknown')
            by_sense[sense_id].append(event)

        if len(by_sense) < 2:
            return None

        senses = []
        for sense_id, sense_events in by_sense.items():
            sample = random.sample(sense_events, min(3, len(sense_events)))
            sense_entry = {
                'sense_id': sense_id,
                'gloss': sense_events[0].get('sense_gloss', ''),
                'examples': [self._create_bundle_item(e) for e in sample],
                'difficulty_distribution': {
                    tier: sum(1 for e in sample if self._get_difficulty_tier(e) == tier)
                    for tier in [DifficultyTier.HIGH_CUE_LOW_RISK,
                                DifficultyTier.MEDIUM_CUE,
                                DifficultyTier.LOW_CUE_HIGH_RISK]
                }
            }
            senses.append(sense_entry)

        bundle = {
            'id': hashlib.md5(f"{lemma}:disc:{self.seed}".encode()).hexdigest()[:12],
            'lemma': lemma,
            'senses': senses,
            'metadata': {
                'num_senses': len(senses),
                'total_examples': sum(len(s['examples']) for s in senses),
                'bundle_type': 'discrimination_v22'
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
            'cue_tokens': event.get('cue_tokens', [])[:5],
            'source': event.get('_sense_source', ''),
            'difficulty_tier': self._get_difficulty_tier(event),
            'quality': {
                'cue_strength': event.get('quality', {}).get('cue_strength', 0.5),
                'ambiguity_risk': event.get('quality', {}).get('ambiguity_risk', 0.5)
            }
        }

    def generate_all_bundles(self,
                             max_contrastive: int = 50000,
                             max_discrimination: int = 5000,
                             max_gloss_matching: int = 5000,
                             positive_rate_target: float = 0.95) -> Dict[str, List[Dict]]:
        """Generate all bundle types with quality controls."""
        print("\n--- Generating Enhanced Bundles v2.2 ---")

        contrastive_bundles = []
        discrimination_bundles = []
        gloss_matching_bundles = []

        # Get polysemous lemmas
        polysemous_lemmas = [
            lemma for lemma, events in self.by_lemma.items()
            if len(set(e.get('sense_id') for e in events)) >= 2
        ]
        print(f"  Polysemous lemmas: {len(polysemous_lemmas)}")

        # Generate contrastive bundles with positive enforcement
        print("  Generating contrastive bundles (enforcing positive-present)...")

        # First pass: try to get bundles with positives
        all_events = list(self.events)
        random.shuffle(all_events)

        bundles_with_positive = 0
        bundles_without_positive = 0

        for event in all_events:
            if len(contrastive_bundles) >= max_contrastive:
                break

            # Calculate current positive rate
            current_positive_rate = bundles_with_positive / (bundles_with_positive + bundles_without_positive + 1)

            # Enforce positive if below target rate
            enforce = current_positive_rate < positive_rate_target

            bundle = self.create_contrastive_bundle(
                event,
                enforce_positive=enforce,
                within_lemma_negs=2,
                cross_lemma_negs=1
            )

            if bundle:
                contrastive_bundles.append(bundle)
                if bundle['positive']:
                    bundles_with_positive += 1
                else:
                    bundles_without_positive += 1

        actual_positive_rate = bundles_with_positive / len(contrastive_bundles) if contrastive_bundles else 0
        print(f"    Generated {len(contrastive_bundles)} bundles")
        print(f"    Positive rate: {actual_positive_rate:.1%} (target: {positive_rate_target:.1%})")

        # Generate discrimination bundles
        print("  Generating discrimination bundles...")
        random.shuffle(polysemous_lemmas)

        for lemma in polysemous_lemmas[:max_discrimination]:
            bundle = self.create_discrimination_bundle(lemma)
            if bundle:
                discrimination_bundles.append(bundle)

        print(f"    Generated {len(discrimination_bundles)} bundles")

        # Generate gloss-matching bundles
        print("  Generating gloss-matching bundles...")

        for lemma in polysemous_lemmas[:max_gloss_matching]:
            bundle = self.create_gloss_matching_bundle(lemma)
            if bundle:
                gloss_matching_bundles.append(bundle)

        print(f"    Generated {len(gloss_matching_bundles)} bundles")

        # Analyze difficulty distribution
        difficulty_dist = defaultdict(int)
        for bundle in contrastive_bundles:
            tier = bundle['metadata'].get('difficulty_tier', 'unknown')
            difficulty_dist[tier] += 1

        print(f"\n  Difficulty distribution (contrastive):")
        for tier, count in sorted(difficulty_dist.items()):
            print(f"    {tier}: {count} ({100*count/len(contrastive_bundles):.1f}%)")

        return {
            'contrastive': contrastive_bundles,
            'discrimination': discrimination_bundles,
            'gloss_matching': gloss_matching_bundles
        }


def main():
    """Main bundle generation function."""
    print("=" * 70)
    print("ENHANCED BUNDLE GENERATION v2.2")
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
    generator = EnhancedBundleGenerator(events, seed=42)

    # Generate bundles
    bundles = generator.generate_all_bundles(
        max_contrastive=50000,
        max_discrimination=2000,
        max_gloss_matching=2000,
        positive_rate_target=0.95
    )

    # Save bundles
    output_dir = Path("paradigm_factory/v2/bundles_v22")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Saving bundles to {output_dir} ---")

    for bundle_type, bundle_list in bundles.items():
        output_file = output_dir / f"{bundle_type}_bundles.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for bundle in bundle_list:
                f.write(json.dumps(bundle, ensure_ascii=False) + '\n')
        print(f"  {output_file.name}: {len(bundle_list)} bundles")

    # Save combined training set
    all_bundles = bundles['contrastive'] + bundles['discrimination'] + bundles['gloss_matching']
    random.shuffle(all_bundles)

    combined_file = output_dir / "all_bundles_v22.jsonl"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for bundle in all_bundles:
            f.write(json.dumps(bundle, ensure_ascii=False) + '\n')
    print(f"  {combined_file.name}: {len(all_bundles)} bundles")

    # Save statistics
    stats = {
        'timestamp': datetime.now().isoformat(),
        'version': '2.2',
        'source_events': len(events),
        'contrastive_bundles': len(bundles['contrastive']),
        'discrimination_bundles': len(bundles['discrimination']),
        'gloss_matching_bundles': len(bundles['gloss_matching']),
        'total_bundles': len(all_bundles),
        'positive_rate': sum(1 for b in bundles['contrastive'] if b.get('positive')) / len(bundles['contrastive']) if bundles['contrastive'] else 0,
        'improvements': [
            'Positive-present enforcement (>95%)',
            'Difficulty tiering (tier1_easy, tier2_robust, tier3_adversarial)',
            'Two-pool negatives (within-lemma + cross-lemma)',
            'Gloss-matching auxiliary objective',
            'Canonicalized sense IDs with source namespaces'
        ]
    }

    stats_file = output_dir / "bundle_stats_v22.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 70)
    print("BUNDLE GENERATION v2.2 COMPLETE")
    print("=" * 70)
    print(f"\n  Total bundles: {len(all_bundles)}")
    print(f"  Contrastive: {len(bundles['contrastive'])} (positive rate: {stats['positive_rate']:.1%})")
    print(f"  Discrimination: {len(bundles['discrimination'])}")
    print(f"  Gloss-matching: {len(bundles['gloss_matching'])}")
    print(f"\n  Output: {output_dir}")

    return bundles


if __name__ == "__main__":
    bundles = main()
