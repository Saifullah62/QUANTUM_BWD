#!/usr/bin/env python3
"""
Tier3 Expansion Pipeline
=========================

Expands tier3 from ~454 to 2k-5k bundles by generating variants
around proven hard anchors, then re-scoring to keep only true tier3.

Three controlled moves:
1. Extract tier3 seeds (proven hard bundles)
2. Generate positive variants (sense-preserving transforms)
3. Generate negative variants (diversified sibling + cue-aligned cross-lemma)
4. Re-score with danger and keep only tier3-eligible

Usage:
    python paradigm_factory/v2/expand_tier3.py \
        --bundles paradigm_factory/v2/bundles_v23/contrastive_bundles.jsonl \
        --events paradigm_factory/v2/canonicalized_v21.jsonl \
        --out paradigm_factory/v2/bundles_v23/tier3_expanded.jsonl \
        --max-per-seed 6
"""

import json
import hashlib
import random
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Set, Optional
import argparse

# Stopwords for cue extraction
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don',
    'now', 'and', 'but', 'or', 'because', 'until', 'while', 'if', 'that',
    'this', 'it', 'its', 'he', 'she', 'they', 'them', 'his', 'her', 'their',
    'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your'
}

# Simple synonym pairs for micro-variants
SYNONYM_PAIRS = [
    ('big', 'large'), ('small', 'little'), ('quick', 'fast'), ('slow', 'gradual'),
    ('happy', 'pleased'), ('sad', 'unhappy'), ('good', 'excellent'), ('bad', 'poor'),
    ('start', 'begin'), ('end', 'finish'), ('make', 'create'), ('use', 'utilize'),
    ('show', 'display'), ('find', 'discover'), ('get', 'obtain'), ('give', 'provide'),
    ('old', 'ancient'), ('new', 'recent'), ('hard', 'difficult'), ('easy', 'simple'),
    ('important', 'significant'), ('different', 'distinct'), ('similar', 'alike'),
]

# Disambiguation clauses by POS
DISAMBIGUATION_CLAUSES = {
    'noun': [
        'in this context,', 'specifically,', 'as a concept,', 'in particular,',
        'notably,', 'essentially,', 'fundamentally,'
    ],
    'verb': [
        'in action,', 'when performed,', 'as an action,', 'actively,',
        'when doing so,', 'in practice,'
    ],
    'adjective': [
        'as a quality,', 'in description,', 'characteristically,',
        'when describing,', 'as a property,'
    ],
    'adverb': [
        'in manner,', 'as a modifier,', 'descriptively,'
    ]
}


@dataclass
class ExpansionStats:
    """Statistics for tier3 expansion."""
    timestamp: str = ""
    tier3_seeds: int = 0
    positive_variants_generated: int = 0
    positive_variants_kept: int = 0
    negative_variants_added: int = 0
    candidates_assembled: int = 0
    candidates_above_threshold: int = 0
    final_expanded_bundles: int = 0
    diversity_filtered: int = 0
    original_tier3_threshold: float = 0.0


class Tier3Expander:
    """Expands tier3 bundles through controlled variant generation."""

    def __init__(
        self,
        bundles_path: Path,
        events_path: Path,
        output_path: Path,
        max_per_seed: int = 12,
        min_token_diff: int = 2,
        candidates_per_positive: int = 4,
        seed: int = 42
    ):
        self.bundles_path = bundles_path
        self.events_path = events_path
        self.output_path = output_path
        self.max_per_seed = max_per_seed
        self.min_token_diff = min_token_diff
        self.candidates_per_positive = candidates_per_positive
        self.rng_seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Data containers
        self.bundles: List[Dict] = []
        self.tier3_seeds: List[Dict] = []
        self.events_by_sense: Dict[str, List[Dict]] = defaultdict(list)
        self.events_by_lemma: Dict[str, List[Dict]] = defaultdict(list)

        # Embedding cache
        self._encoder = None
        self.embeddings: Dict[str, np.ndarray] = {}

        # Stats
        self.stats = ExpansionStats()

    def _get_encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print("[OK] Loaded sentence-transformers encoder")
            except ImportError:
                print("[!] sentence-transformers not available, using fallback")
                self._encoder = "fallback"
        return self._encoder

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to embedding."""
        if text in self.embeddings:
            return self.embeddings[text]

        encoder = self._get_encoder()
        if encoder == "fallback":
            h = hashlib.md5(text.encode()).hexdigest()
            emb = np.array([int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)])
        else:
            emb = encoder.encode(text, show_progress_bar=False)

        self.embeddings[text] = emb
        return emb

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def load_data(self):
        """Load bundles and events."""
        print("=" * 60)
        print("TIER3 EXPANSION PIPELINE")
        print("=" * 60)
        print("\nLoading data...")

        # Load bundles
        danger_scores = []
        with open(self.bundles_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    bundle = json.loads(line)
                    self.bundles.append(bundle)

                    # Track danger scores for threshold calculation
                    meta = bundle.get('metadata', {})
                    danger = meta.get('danger_score', 0)
                    danger_scores.append(danger)

                    # Extract tier3 seeds
                    tier = meta.get('difficulty_tier', '')
                    if tier == 'tier3_adversarial':
                        self.tier3_seeds.append(bundle)

        # Calculate tier3 threshold (top 3% of danger scores)
        if danger_scores:
            self.tier3_threshold = np.percentile(danger_scores, 97)
        else:
            self.tier3_threshold = 0.0

        print(f"[OK] Loaded {len(self.bundles)} bundles")
        print(f"[OK] Found {len(self.tier3_seeds)} tier3 seeds")
        print(f"[OK] Tier3 threshold (p97): {self.tier3_threshold:.4f}")

        self.stats.tier3_seeds = len(self.tier3_seeds)
        self.stats.original_tier3_threshold = self.tier3_threshold

        # Load events for additional mining
        if self.events_path.exists():
            with open(self.events_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        sense_id = event.get('sense_id', '')
                        lemma = event.get('lemma', '').lower()
                        self.events_by_sense[sense_id].append(event)
                        self.events_by_lemma[lemma].append(event)
            print(f"[OK] Loaded events for {len(self.events_by_lemma)} lemmas")

    def _extract_content_words(self, text: str, lemma: str) -> Set[str]:
        """Extract content words from text (excluding stopwords and lemma)."""
        words = re.findall(r'\b\w+\b', text.lower())
        content = {w for w in words if w not in STOPWORDS and w != lemma.lower()}
        return content

    def _generate_positive_variants(self, seed: Dict) -> List[Dict]:
        """
        Generate sense-preserving positive variants.

        Transforms:
        1. Add disambiguation clause
        2. Swap non-cue synonyms
        3. Minor tense/structure variants
        """
        anchor = seed.get('anchor', {})
        positive = seed.get('positive', {})
        lemma = seed.get('lemma', '').lower()
        pos = seed.get('pos', 'noun')
        sense_id = anchor.get('sense_id', '')

        anchor_text = anchor.get('text', '')
        positive_text = positive.get('text', '')

        variants = []
        seen_texts = {positive_text.lower()}

        # Get anchor embedding for verification
        anchor_emb = self._encode(anchor_text)

        # Transform 1: Add disambiguation clauses (increased from 3 to 6)
        clauses = DISAMBIGUATION_CLAUSES.get(pos, DISAMBIGUATION_CLAUSES['noun'])
        for clause in clauses[:6]:
            # Try prepending
            variant_text = f"{clause} {positive_text}"
            if variant_text.lower() not in seen_texts:
                variant = self._verify_positive_variant(
                    variant_text, anchor_emb, sense_id, lemma
                )
                if variant:
                    variants.append(variant)
                    seen_texts.add(variant_text.lower())

        # Transform 2: Synonym swaps (1-2 swaps max)
        words = positive_text.split()
        for old_word, new_word in SYNONYM_PAIRS:
            if old_word in positive_text.lower() and old_word != lemma:
                variant_text = re.sub(
                    rf'\b{old_word}\b',
                    new_word,
                    positive_text,
                    flags=re.IGNORECASE
                )
                if variant_text.lower() not in seen_texts:
                    variant = self._verify_positive_variant(
                        variant_text, anchor_emb, sense_id, lemma
                    )
                    if variant:
                        variants.append(variant)
                        seen_texts.add(variant_text.lower())

        # Transform 3: Use other same-sense examples as positives (increased from 5 to 15)
        same_sense_events = self.events_by_sense.get(sense_id, [])
        for event in same_sense_events[:15]:
            event_text = event.get('text', '')
            if event_text.lower() not in seen_texts and event_text != anchor_text:
                variant = self._verify_positive_variant(
                    event_text, anchor_emb, sense_id, lemma
                )
                if variant:
                    variants.append(variant)
                    seen_texts.add(event_text.lower())

        self.stats.positive_variants_generated += len(variants) + 1  # +1 for original

        return variants

    def _verify_positive_variant(
        self,
        text: str,
        anchor_emb: np.ndarray,
        sense_id: str,
        lemma: str
    ) -> Optional[Dict]:
        """
        Verify a positive variant passes quality checks.

        Checks:
        1. Embedding similarity to anchor >= 0.5
        2. Contains the lemma (or close variant)
        """
        # Check lemma presence (allow minor variations)
        text_lower = text.lower()
        if lemma.lower() not in text_lower:
            # Check for lemma variants (plural, verb forms)
            lemma_variants = [lemma, lemma + 's', lemma + 'es', lemma + 'ed', lemma + 'ing']
            if not any(v in text_lower for v in lemma_variants):
                return None

        # Check embedding similarity (relaxed threshold for expansion)
        text_emb = self._encode(text)
        sim = self._cosine_sim(anchor_emb, text_emb)

        if sim < 0.3:  # Relaxed from 0.5 to allow more diversity
            return None

        self.stats.positive_variants_kept += 1

        return {
            'text': text,
            'sense_id': sense_id,
            'similarity_to_anchor': sim,
            'source': 'tier3_expansion'
        }

    def _generate_negative_variants(self, seed: Dict) -> Dict[str, List[Dict]]:
        """
        Generate diversified negative variants.

        1. Top 5 sibling confusers (same lemma, different sense)
        2. Cue-aligned cross-lemma neighbors
        """
        anchor = seed.get('anchor', {})
        lemma = seed.get('lemma', '').lower()
        pos = seed.get('pos', 'noun')
        anchor_text = anchor.get('text', '')
        anchor_sense = anchor.get('sense_id', '')

        anchor_emb = self._encode(anchor_text)
        anchor_cues = self._extract_content_words(anchor_text, lemma)

        siblings = []
        cross_lemma = []

        # Get sibling confusers (same lemma, different sense)
        lemma_events = self.events_by_lemma.get(lemma, [])
        sibling_candidates = []

        for event in lemma_events:
            event_sense = event.get('sense_id', '')
            if event_sense != anchor_sense and event.get('pos') == pos:
                event_text = event.get('text', '')
                event_emb = self._encode(event_text)
                sim = self._cosine_sim(anchor_emb, event_emb)
                sibling_candidates.append((event, sim))

        # Sort by similarity (highest = most confusable)
        sibling_candidates.sort(key=lambda x: -x[1])

        # Take top 20 unique senses (increased from 5)
        seen_senses = set()
        for event, sim in sibling_candidates:
            sense = event.get('sense_id', '')
            if sense not in seen_senses and len(siblings) < 20:
                siblings.append({
                    'text': event.get('text', ''),
                    'sense_id': sense,
                    'sense_gloss': event.get('sense_gloss', ''),
                    'similarity': sim,
                    'type': 'sibling_confuser'
                })
                seen_senses.add(sense)

        # Get cross-lemma confusers with cue overlap
        for other_lemma, other_events in self.events_by_lemma.items():
            if other_lemma == lemma:
                continue

            for event in other_events[:10]:  # Sample from each lemma
                if event.get('pos') != pos:
                    continue

                event_text = event.get('text', '')
                event_cues = self._extract_content_words(event_text, other_lemma)

                # Require at least 1 cue overlap
                cue_overlap = anchor_cues & event_cues
                if len(cue_overlap) < 1:
                    continue

                event_emb = self._encode(event_text)
                sim = self._cosine_sim(anchor_emb, event_emb)

                if sim > 0.3:  # Reasonably similar (relaxed for more negatives)
                    cross_lemma.append({
                        'text': event_text,
                        'sense_id': event.get('sense_id', ''),
                        'sense_gloss': event.get('sense_gloss', ''),
                        'similarity': sim,
                        'cue_overlap': list(cue_overlap),
                        'type': 'cross_lemma_confuser'
                    })

        # Sort cross-lemma by similarity and take top 50 (increased from 5)
        cross_lemma.sort(key=lambda x: -x['similarity'])
        cross_lemma = cross_lemma[:50]

        self.stats.negative_variants_added += len(siblings) + len(cross_lemma)

        return {
            'siblings': siblings,
            'cross_lemma': cross_lemma
        }

    def _compute_danger(
        self,
        anchor_emb: np.ndarray,
        positive_emb: np.ndarray,
        negative_embs: List[np.ndarray]
    ) -> float:
        """
        Compute danger score.

        danger = max(neg_similarities) - pos_similarity
        Higher = harder (negatives closer than positive)
        """
        pos_sim = self._cosine_sim(anchor_emb, positive_emb)

        if not negative_embs:
            return -1.0  # No negatives = easy

        neg_sims = [self._cosine_sim(anchor_emb, neg_emb) for neg_emb in negative_embs]
        max_neg_sim = max(neg_sims)

        return max_neg_sim - pos_sim

    def _assemble_candidates(self, seed: Dict) -> List[Dict]:
        """
        Assemble candidate bundles from seed + variants.

        For each positive variant, generate multiple candidates by sampling
        different "villain lineups" from the negative pool. This multiplies
        candidate volume without touching sense integrity.
        """
        anchor = seed.get('anchor', {})
        original_positive = seed.get('positive', {})
        lemma = seed.get('lemma', '')
        pos = seed.get('pos', '')

        anchor_text = anchor.get('text', '')
        anchor_emb = self._encode(anchor_text)

        # Generate variants
        positive_variants = self._generate_positive_variants(seed)
        negative_variants = self._generate_negative_variants(seed)

        siblings = negative_variants['siblings']
        cross_lemma = negative_variants['cross_lemma']

        candidates = []
        seen_signatures = set()  # Track (pos_text, neg_signature) to avoid duplicates

        # Include original positive + new positives
        all_positives = [original_positive] + positive_variants

        for pos_variant in all_positives:
            pos_text = pos_variant.get('text', '')
            pos_emb = self._encode(pos_text)

            # Generate multiple candidates per positive with different negative samples
            for _ in range(self.candidates_per_positive):
                # Sample negatives: 4 siblings + 3 cross-lemma (total ~7-10)
                n_sibs = min(4, len(siblings))
                n_cross = min(3, len(cross_lemma))

                if n_sibs == 0 and n_cross == 0:
                    continue

                sampled_sibs = random.sample(siblings, n_sibs) if n_sibs > 0 else []
                sampled_cross = random.sample(cross_lemma, n_cross) if n_cross > 0 else []

                all_negs = sampled_sibs + sampled_cross
                if not all_negs:
                    continue

                # Create negative signature to prevent duplicate villain lineups
                sib_ids = tuple(sorted(n.get('sense_id', n.get('text', '')[:20]) for n in sampled_sibs))
                cross_ids = tuple(sorted(n.get('sense_id', n.get('text', '')[:20]) for n in sampled_cross))
                neg_signature = (sib_ids, cross_ids)

                # Skip if we've seen this (positive, negative_set) combo
                combo_signature = (pos_text[:50], neg_signature)
                if combo_signature in seen_signatures:
                    continue
                seen_signatures.add(combo_signature)

                neg_embs = [self._encode(n['text']) for n in all_negs]
                danger = self._compute_danger(anchor_emb, pos_emb, neg_embs)

                # Include neg_signature in ID for uniqueness
                id_str = f"{anchor_text}:{pos_text}:{sib_ids}:{cross_ids}:{self.rng_seed}"
                candidate = {
                    'id': hashlib.md5(id_str.encode()).hexdigest()[:16],
                    'lemma': lemma,
                    'pos': pos,
                    'anchor': {
                        'text': anchor_text,
                        'sense_id': anchor.get('sense_id', ''),
                        'sense_gloss': anchor.get('sense_gloss', ''),
                    },
                    'positive': {
                        'text': pos_text,
                        'sense_id': pos_variant.get('sense_id', anchor.get('sense_id', '')),
                        'source': pos_variant.get('source', 'original'),
                    },
                    'negatives': {
                        'within_lemma': sampled_sibs,
                        'cross_lemma': sampled_cross,
                    },
                    'metadata': {
                        'difficulty_tier': 'tier3_adversarial',
                        'danger_score': danger,
                        'source': 'tier3_expansion',
                        'original_seed_id': seed.get('id', ''),
                        'timestamp': datetime.now().isoformat(),
                    }
                }
                candidates.append(candidate)

        self.stats.candidates_assembled += len(candidates)
        return candidates

    def _filter_by_danger(self, candidates: List[Dict], threshold_factor: float = 0.8) -> List[Dict]:
        """
        Keep candidates above adjusted tier3 threshold.

        Uses threshold_factor to allow borderline-tier3 candidates (default 0.8x tier3).
        Labels them appropriately based on danger score.
        """
        adjusted_threshold = self.tier3_threshold * threshold_factor
        kept = []

        for c in candidates:
            danger = c.get('metadata', {}).get('danger_score', 0)
            if danger >= adjusted_threshold:
                # Label based on actual danger level
                if danger >= self.tier3_threshold:
                    tier_label = 'tier3_adversarial'
                else:
                    tier_label = 'tier3_expanded'  # Borderline tier3
                c['metadata']['difficulty_tier'] = tier_label
                c['metadata']['expansion_threshold'] = adjusted_threshold
                kept.append(c)

        self.stats.candidates_above_threshold = len(kept)
        return kept

    def _enforce_diversity(self, candidates: List[Dict]) -> List[Dict]:
        """
        Enforce diversity constraints.

        - Max bundles per original seed
        - Unique (positive, negative_signature) pairs
        - Min token difference between positives (when same negative set)
        """
        by_seed = defaultdict(list)
        for c in candidates:
            seed_id = c.get('metadata', {}).get('original_seed_id', 'unknown')
            by_seed[seed_id].append(c)

        filtered = []
        for seed_id, seed_candidates in by_seed.items():
            # Sort by danger (highest first)
            seed_candidates.sort(key=lambda x: -x.get('metadata', {}).get('danger_score', 0))

            # Track (pos_text_hash, neg_signature) to allow same positive with different negatives
            kept_combos = set()
            kept_pos_texts = {}  # pos_text -> count, to limit repetition

            for c in seed_candidates:
                if len([x for x in filtered if x.get('metadata', {}).get('original_seed_id') == seed_id]) >= self.max_per_seed:
                    break

                pos_text = c.get('positive', {}).get('text', '')

                # Create negative signature from sense_ids
                negs = c.get('negatives', {})
                within_ids = tuple(sorted(n.get('sense_id', '')[:20] for n in negs.get('within_lemma', [])))
                cross_ids = tuple(sorted(n.get('sense_id', '')[:20] for n in negs.get('cross_lemma', [])))
                neg_signature = (within_ids, cross_ids)

                # Check for duplicate (positive, negative) combo
                combo = (pos_text[:50], neg_signature)
                if combo in kept_combos:
                    continue

                # Limit same positive to max 6 different negative lineups (increased from 3)
                if kept_pos_texts.get(pos_text, 0) >= 6:
                    continue

                kept_combos.add(combo)
                kept_pos_texts[pos_text] = kept_pos_texts.get(pos_text, 0) + 1
                filtered.append(c)

        self.stats.diversity_filtered = len(candidates) - len(filtered)
        return filtered

    def expand(self, threshold_factor: float = 0.8):
        """Run the full expansion pipeline."""
        print("\n" + "=" * 60)
        print("EXPANDING TIER3")
        print("=" * 60)

        adjusted_threshold = self.tier3_threshold * threshold_factor
        print(f"Original tier3 threshold: {self.tier3_threshold:.4f}")
        print(f"Adjusted threshold ({threshold_factor:.0%}): {adjusted_threshold:.4f}")

        all_candidates = []

        print(f"\nProcessing {len(self.tier3_seeds)} tier3 seeds...")
        for i, seed in enumerate(self.tier3_seeds):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(self.tier3_seeds)} seeds...")

            candidates = self._assemble_candidates(seed)
            all_candidates.extend(candidates)

        print(f"\n[OK] Generated {len(all_candidates)} candidate bundles")

        # Filter by danger threshold
        print("\nFiltering by danger threshold...")
        filtered = self._filter_by_danger(all_candidates, threshold_factor)
        print(f"[OK] {len(filtered)} candidates above adjusted threshold")

        # Enforce diversity
        print("\nEnforcing diversity constraints...")
        final = self._enforce_diversity(filtered)
        print(f"[OK] {len(final)} bundles after diversity filtering")

        self.stats.final_expanded_bundles = len(final)

        return final

    def save(self, bundles: List[Dict]):
        """Save expanded bundles."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            for bundle in bundles:
                f.write(json.dumps(bundle, ensure_ascii=False) + '\n')

        # Save stats
        self.stats.timestamp = datetime.now().isoformat()
        stats_path = self.output_path.parent / 'tier3_expansion_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(asdict(self.stats), f, indent=2)

        print(f"\n[OK] Saved {len(bundles)} expanded bundles to {self.output_path}")
        print(f"[OK] Stats saved to {stats_path}")

    def print_summary(self):
        """Print expansion summary."""
        print("\n" + "=" * 60)
        print("EXPANSION SUMMARY")
        print("=" * 60)
        print(f"Original tier3 seeds:       {self.stats.tier3_seeds}")
        print(f"Tier3 danger threshold:     {self.stats.original_tier3_threshold:.4f}")
        print(f"Positive variants kept:     {self.stats.positive_variants_kept}")
        print(f"Negative variants added:    {self.stats.negative_variants_added}")
        print(f"Candidates assembled:       {self.stats.candidates_assembled}")
        print(f"Above tier3 threshold:      {self.stats.candidates_above_threshold}")
        print(f"Diversity filtered:         {self.stats.diversity_filtered}")
        print(f"Final expanded bundles:     {self.stats.final_expanded_bundles}")
        print(f"Expansion factor:           {self.stats.final_expanded_bundles / max(1, self.stats.tier3_seeds):.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Expand tier3 bundles")
    parser.add_argument('--bundles', required=True, help='Input bundles JSONL')
    parser.add_argument('--events', required=True, help='Events JSONL for mining')
    parser.add_argument('--out', required=True, help='Output expanded bundles')
    parser.add_argument('--max-per-seed', type=int, default=12, help='Max bundles per seed')
    parser.add_argument('--min-token-diff', type=int, default=2, help='Min token diff for diversity')
    parser.add_argument('--candidates-per-positive', type=int, default=4,
                        help='Number of negative samples per positive (multiplies candidates)')
    parser.add_argument('--threshold-factor', type=float, default=0.8,
                        help='Tier3 threshold multiplier (0.8 = 80%% of tier3, allows borderline)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    expander = Tier3Expander(
        bundles_path=Path(args.bundles),
        events_path=Path(args.events),
        output_path=Path(args.out),
        max_per_seed=args.max_per_seed,
        min_token_diff=args.min_token_diff,
        candidates_per_positive=args.candidates_per_positive,
        seed=args.seed
    )

    expander.load_data()
    expanded = expander.expand(threshold_factor=args.threshold_factor)
    expander.save(expanded)
    expander.print_summary()


if __name__ == '__main__':
    main()
