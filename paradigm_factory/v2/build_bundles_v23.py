#!/usr/bin/env python3
"""
V2.3 Bundle Generator - Strict Contrastive Bundle Construction
==============================================================

Builds on v2.2 with three non-negotiable constraints:
1. Every bundle MUST have a positive (or be explicitly flagged/excluded)
2. Every bundle MUST have within-lemma sibling negative (same POS)
3. Every bundle MUST have cross-lemma negative by embedding proximity

Cross-lemma mining uses two passes:
- Pass 1: Build embedding cache for all (sense_id, example)
- Pass 2: ANN index for nearest-neighbor retrieval across lemmas

Inputs:
- canonicalized_v21.jsonl (processed events)
- sense_inventory.json (lemma -> senses mapping)
- tier4_killers.jsonl (must-include hard cases)

Outputs:
- bundles_v23/contrastive_bundles.jsonl
- bundles_v23/discrimination_bundles.jsonl
- bundles_v23/gloss_matching_bundles.jsonl
- bundles_v23/all_bundles_v23.jsonl
- bundles_v23/bundle_stats_v23.json
- Fingerprint: bundles|v23|hash=...|lemmas=...|senses=...|sib=...|xlem=...
"""

import json
import hashlib
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Set
import sys

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BundleStats:
    """Statistics for the bundle build."""
    version: str = "2.3"
    timestamp: str = ""

    # Input stats
    source_events: int = 0
    unique_lemmas: int = 0
    unique_senses: int = 0

    # Bundle counts
    contrastive_bundles: int = 0
    discrimination_bundles: int = 0
    gloss_matching_bundles: int = 0
    total_bundles: int = 0

    # Constraint satisfaction
    positive_rate: float = 0.0
    sibling_negative_rate: float = 0.0  # Target: >95%
    cross_lemma_negative_rate: float = 0.0  # Target: >90%

    # Negatives per bundle
    avg_within_lemma_negs: float = 0.0
    avg_cross_lemma_negs: float = 0.0

    # Tier distribution
    tier_distribution: Dict[str, int] = field(default_factory=dict)

    # Exclusions (detailed for enrichment TODO)
    excluded_no_positive: int = 0
    excluded_no_sibling: int = 0
    singleton_senses: int = 0  # Senses with only 1 example
    single_sense_lemmas: int = 0  # Lemmas with only 1 sense

    # Yield analysis (why aren't we at 100%?)
    total_anchors_tried: int = 0
    yield_rate: float = 0.0  # bundles / anchors_tried

    # Fingerprint
    content_hash: str = ""
    fingerprint: str = ""


class EmbeddingCache:
    """
    Manages embeddings for cross-lemma similarity mining.

    Uses the same encoder as the retrieval eval scorer for consistency.
    Maintains bidirectional mapping: event_id <-> row_id for ANN queries.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path
        self.embeddings: Dict[str, np.ndarray] = {}
        self.example_ids: List[str] = []  # Ordered list of event IDs
        self.example_data: Dict[str, Dict] = {}
        self._encoder = None

        # Bidirectional index mappings (built after build_index)
        self.event_id_to_row: Dict[str, int] = {}  # event_id -> matrix row
        self.row_to_event_id: Dict[int, str] = {}  # matrix row -> event_id

    def _get_encoder(self):
        """Lazy load the encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print("[OK] Loaded sentence-transformers encoder")
            except ImportError:
                print("[!] sentence-transformers not available, using fallback")
                self._encoder = "fallback"
        return self._encoder

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        encoder = self._get_encoder()
        if encoder == "fallback":
            # Simple fallback: hash-based pseudo-embedding
            h = hashlib.md5(text.encode()).hexdigest()
            return np.array([int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)])
        else:
            return encoder.encode(text, show_progress_bar=False)

    def add_example(self, example_id: str, text: str, metadata: Dict):
        """Add an example to the cache."""
        if example_id not in self.embeddings:
            self.embeddings[example_id] = self._encode_text(text)
            self.example_ids.append(example_id)
            self.example_data[example_id] = metadata

    def build_index(self):
        """Build the similarity index and bidirectional mappings."""
        # For production, use FAISS. For now, brute force is fine for ~40k items
        self.embedding_matrix = np.stack([
            self.embeddings[eid] for eid in self.example_ids
        ]) if self.example_ids else np.array([])

        # Build bidirectional index mappings
        self.event_id_to_row = {eid: i for i, eid in enumerate(self.example_ids)}
        self.row_to_event_id = {i: eid for i, eid in enumerate(self.example_ids)}

        print(f"[OK] Built embedding index with {len(self.example_ids)} examples")
        print(f"     Index mappings: {len(self.event_id_to_row)} event_id <-> row")

    def find_nearest_cross_lemma(
        self,
        query_id: str,
        query_lemma: str,
        query_pos: str,
        k: int = 10,
        same_pos_only: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest examples from OTHER lemmas.

        Args:
            query_id: ID of the query example
            query_lemma: Lemma to exclude
            query_pos: POS to match (if same_pos_only)
            k: Number of neighbors
            same_pos_only: If True, only return same-POS matches

        Returns:
            List of (example_id, similarity) tuples
        """
        if query_id not in self.embeddings:
            return []

        query_emb = self.embeddings[query_id]

        # Compute similarities
        if len(self.embedding_matrix) == 0:
            return []

        similarities = np.dot(self.embedding_matrix, query_emb)

        # Filter and rank
        candidates = []
        for idx, (eid, sim) in enumerate(zip(self.example_ids, similarities)):
            meta = self.example_data.get(eid, {})

            # Exclude same lemma
            if meta.get('lemma') == query_lemma:
                continue

            # Exclude self
            if eid == query_id:
                continue

            # POS filter
            if same_pos_only and meta.get('pos') != query_pos:
                continue

            candidates.append((eid, float(sim)))

        # Sort by similarity (descending) and take top k
        candidates.sort(key=lambda x: -x[1])
        return candidates[:k]

    def compute_pool_similarities(
        self,
        query_id: str,
        candidate_ids: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Compute similarities between query and a specific candidate pool.

        For sibling selection, this is more efficient than the full ANN search
        since we already know the candidates we care about.

        Args:
            query_id: ID of the anchor
            candidate_ids: List of candidate event IDs to score

        Returns:
            List of (candidate_id, similarity) sorted by similarity descending
        """
        if query_id not in self.embeddings:
            return []

        query_emb = self.embeddings[query_id]

        results = []
        for cand_id in candidate_ids:
            if cand_id in self.embeddings:
                cand_emb = self.embeddings[cand_id]
                sim = float(np.dot(query_emb, cand_emb))
                results.append((cand_id, sim))

        # Sort by similarity descending (highest = most similar = hardest negative)
        results.sort(key=lambda x: -x[1])
        return results

    def save(self, path: Path):
        """Save cache to disk."""
        # Save metadata only, recompute embeddings on load
        data = {
            'example_ids': self.example_ids,
            'example_data': self.example_data
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: Path) -> bool:
        """Load cache from disk."""
        if not path.exists():
            return False
        with open(path, 'r') as f:
            data = json.load(f)
        self.example_ids = data['example_ids']
        self.example_data = data['example_data']
        return True


class BundleBuilderV23:
    """
    V2.3 Bundle Builder with strict constraints.
    """

    def __init__(
        self,
        events_path: Path,
        inventory_path: Path,
        output_dir: Path,
        tier4_path: Optional[Path] = None,
        seed: int = 42
    ):
        self.events_path = events_path
        self.inventory_path = inventory_path
        self.output_dir = output_dir
        self.tier4_path = tier4_path
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Data containers
        self.events: List[Dict] = []
        self.inventory: Dict[str, List[Dict]] = {}
        self.tier4_lemmas: Set[str] = set()

        # Index structures - key is (lemma, pos, sense_id) for fast same-sense lookup
        self.by_sense: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
        self.by_lemma_pos: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        self.events_by_lemma: Dict[str, List[Dict]] = defaultdict(list)
        self.events_by_sense: Dict[str, List[Dict]] = defaultdict(list)
        self.senses_by_lemma: Dict[str, List[str]] = defaultdict(list)

        # Singleton senses log
        self.singleton_log: List[Dict] = []

        # Embedding cache
        self.emb_cache = EmbeddingCache()

        # Stats
        self.stats = BundleStats()

    def load_data(self):
        """Load all input data."""
        print("=" * 60)
        print("Loading input data...")
        print("=" * 60)

        # Load events
        with open(self.events_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    self.events.append(event)

                    lemma = event.get('lemma', '').lower()
                    pos = event.get('pos', 'noun')
                    sense_id = event.get('sense_id', '')

                    # Build indices
                    sense_key = (lemma, pos, sense_id)
                    self.by_sense[sense_key].append(event)
                    self.by_lemma_pos[(lemma, pos)].append(event)
                    self.events_by_lemma[lemma].append(event)
                    self.events_by_sense[sense_id].append(event)

        print(f"[OK] Loaded {len(self.events)} events")
        print(f"    Unique (lemma,pos,sense) keys: {len(self.by_sense)}")
        print(f"    Unique (lemma,pos) keys: {len(self.by_lemma_pos)}")
        self.stats.source_events = len(self.events)

        # Load inventory
        with open(self.inventory_path, 'r', encoding='utf-8') as f:
            self.inventory = json.load(f)

        # Build senses_by_lemma index
        for lemma, senses in self.inventory.items():
            for sense in senses:
                sense_id = sense.get('canonical_id', '')
                if sense_id:
                    self.senses_by_lemma[lemma].append(sense_id)

        print(f"[OK] Loaded inventory with {len(self.inventory)} lemmas")
        self.stats.unique_lemmas = len(self.inventory)
        self.stats.unique_senses = sum(len(s) for s in self.inventory.values())

        # Load tier4 must-include
        if self.tier4_path and self.tier4_path.exists():
            with open(self.tier4_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        self.tier4_lemmas.add(item.get('lemma', ''))
            print(f"[OK] Loaded {len(self.tier4_lemmas)} tier4 must-include lemmas")

    def build_embedding_cache(self):
        """Build embedding cache for cross-lemma mining."""
        print("\n" + "=" * 60)
        print("Building embedding cache (Pass 1)...")
        print("=" * 60)

        for i, event in enumerate(self.events):
            example_id = event.get('id', f"event_{i}")
            text = event.get('text', '')

            metadata = {
                'lemma': event.get('lemma', ''),
                'pos': event.get('pos', ''),
                'sense_id': event.get('sense_id', ''),
                'difficulty_tier': event.get('quality', {}).get('difficulty_tier', 'tier1_easy')
            }

            self.emb_cache.add_example(example_id, text, metadata)

            if (i + 1) % 5000 == 0:
                print(f"  Cached {i + 1}/{len(self.events)} examples...")

        # Build index
        self.emb_cache.build_index()

    def _get_sibling_negatives(
        self,
        anchor_event: Dict,
        exclude_sense: str,
        max_negs: int = 3,
        samples_per_sense: int = 25
    ) -> List[Dict]:
        """
        Get within-lemma sibling negatives (same POS, different sense).

        Uses by_lemma_pos index with per-sense sampling cap for efficiency.
        For large lemmas (thousands of events), samples up to 25 per sibling sense
        before computing similarities, ensuring "dangerous" siblings are found
        without blowing compute.

        Constraint: MUST return at least 1 if lemma has multiple senses.
        """
        lemma = anchor_event.get('lemma', '').lower()
        pos = anchor_event.get('pos', 'noun')
        anchor_id = anchor_event.get('id', '')

        # Look up same (lemma, pos) pool
        pool = self.by_lemma_pos.get((lemma, pos), [])

        # Group siblings by sense_id (excluding anchor sense)
        siblings_by_sense: Dict[str, List[Dict]] = defaultdict(list)
        for e in pool:
            e_sense = e.get('sense_id', '')
            if e_sense != exclude_sense and e.get('id') != anchor_id:
                siblings_by_sense[e_sense].append(e)

        if not siblings_by_sense:
            return []

        # Sample up to samples_per_sense from each sibling sense
        # This caps compute for lemmas with thousands of events
        sampled_candidates = []
        for sense_id, sense_events in siblings_by_sense.items():
            if len(sense_events) > samples_per_sense:
                sampled = random.sample(sense_events, samples_per_sense)
            else:
                sampled = sense_events
            sampled_candidates.extend(sampled)

        if not sampled_candidates:
            return []

        # Use embedding similarity to find "dangerous" (closest) siblings
        if anchor_id in self.emb_cache.embeddings and len(sampled_candidates) > max_negs:
            # Get candidate IDs
            candidate_ids = [e.get('id', '') for e in sampled_candidates]

            # Compute similarities using the pool method
            scored = self.emb_cache.compute_pool_similarities(anchor_id, candidate_ids)

            if scored:
                # Map back from IDs to events
                id_to_event = {e.get('id', ''): e for e in sampled_candidates}
                selected = []
                for cand_id, sim in scored[:max_negs]:
                    if cand_id in id_to_event:
                        selected.append(id_to_event[cand_id])
                sampled_candidates = selected
            else:
                # Fallback to random
                sampled_candidates = random.sample(sampled_candidates, min(max_negs, len(sampled_candidates)))
        else:
            # No embeddings or small pool - random sample
            sampled_candidates = random.sample(sampled_candidates, min(max_negs, len(sampled_candidates)))

        # Format as negatives
        return [self._format_negative(e) for e in sampled_candidates]

    def _get_cross_lemma_negatives(
        self,
        anchor_event: Dict,
        max_negs: int = 3
    ) -> List[Dict]:
        """
        Get cross-lemma negatives using embedding proximity.

        Uses ANN to find nearest examples from OTHER lemmas.
        """
        example_id = anchor_event.get('id', '')
        lemma = anchor_event.get('lemma', '')
        pos = anchor_event.get('pos', '')

        # Find nearest neighbors from other lemmas
        neighbors = self.emb_cache.find_nearest_cross_lemma(
            example_id, lemma, pos,
            k=max_negs * 2,  # Get extra to filter
            same_pos_only=True
        )

        negatives = []
        for neighbor_id, similarity in neighbors[:max_negs]:
            # Get the original event
            meta = self.emb_cache.example_data.get(neighbor_id, {})
            neighbor_sense = meta.get('sense_id', '')

            # Find the actual event
            for event in self.events_by_sense.get(neighbor_sense, []):
                if event.get('id') == neighbor_id:
                    neg = self._format_negative(event)
                    neg['_similarity'] = similarity  # For debugging
                    negatives.append(neg)
                    break

        return negatives

    def _format_negative(self, event: Dict) -> Dict:
        """Format an event as a negative example."""
        return {
            "text": event.get('text', ''),
            "span": event.get('span', {}),
            "sense_id": event.get('sense_id', ''),
            "sense_gloss": event.get('sense_gloss', ''),
            "pos": event.get('pos', ''),
            "cue_tokens": event.get('cue_tokens', [])[:5],
            "source": event.get('source', {}).get('domain', 'unknown') if isinstance(event.get('source'), dict) else 'unknown',
            "difficulty_tier": event.get('quality', {}).get('difficulty_tier', 'tier1_easy') if isinstance(event.get('quality'), dict) else 'tier1_easy',
            "quality": {
                "cue_strength": event.get('quality', {}).get('cue_strength', 0.5) if isinstance(event.get('quality'), dict) else 0.5,
                "ambiguity_risk": event.get('quality', {}).get('ambiguity_risk', 0.3) if isinstance(event.get('quality'), dict) else 0.3
            }
        }

    def _format_anchor_or_positive(self, event: Dict) -> Dict:
        """Format an event as anchor or positive."""
        source = event.get('source', {})
        source_str = source.get('domain', 'unknown') if isinstance(source, dict) else 'unknown'

        quality = event.get('quality', {})
        if not isinstance(quality, dict):
            quality = {}

        return {
            "text": event.get('text', ''),
            "span": event.get('span', {}),
            "sense_id": event.get('sense_id', ''),
            "sense_gloss": event.get('sense_gloss', ''),
            "pos": event.get('pos', ''),
            "cue_tokens": event.get('cue_tokens', [])[:5],
            "source": source_str,
            "difficulty_tier": quality.get('difficulty_tier', 'tier1_easy'),
            "quality": {
                "cue_strength": quality.get('cue_strength', 0.5),
                "ambiguity_risk": quality.get('ambiguity_risk', 0.3)
            }
        }

    def _get_positive(self, anchor_event: Dict) -> Optional[Dict]:
        """Get a positive example (same sense, different text).

        Uses (lemma, pos, sense_id) key for precise same-sense lookup.
        Returns None if this is a singleton sense (only 1 example).
        """
        lemma = anchor_event.get('lemma', '').lower()
        pos = anchor_event.get('pos', 'noun')
        sense_id = anchor_event.get('sense_id', '')
        anchor_id = anchor_event.get('id', '')
        anchor_text = anchor_event.get('text', '')

        # Look up same-sense pool
        sense_key = (lemma, pos, sense_id)
        pool = self.by_sense.get(sense_key, [])

        # Need at least 2 examples to have a positive
        if len(pool) < 2:
            # Log singleton for potential enrichment
            self.singleton_log.append({
                "reason": "singleton_sense",
                "event_id": anchor_id,
                "sense_id": sense_id,
                "lemma": lemma
            })
            return None

        # Select a different example from the same sense
        candidates = [e for e in pool if e.get('id') != anchor_id and e.get('text') != anchor_text]

        if not candidates:
            return None

        return self._format_anchor_or_positive(random.choice(candidates))

    def _compute_danger_score(
        self,
        anchor: Dict,
        positive: Dict,
        sibling_negs: List[Dict],
        cross_lemma_negs: List[Dict],
        anchor_event: Dict
    ) -> float:
        """
        Compute danger score for percentile-based tiering.

        danger = max(sim(anchor,sibling_neg), sim(anchor,cross_neg)) - sim(anchor,positive)

        Higher danger = harder bundle (closer negatives, further positives).
        """
        anchor_id = anchor_event.get('id', '')

        # Get anchor embedding
        if anchor_id not in self.emb_cache.embeddings:
            return 0.0
        anchor_emb = self.emb_cache.embeddings[anchor_id]

        # Compute max negative similarity
        max_neg_sim = 0.0

        # Check sibling negatives
        for neg in sibling_negs:
            neg_text = neg.get('text', '')
            # Find the event ID for this negative
            for sense_events in self.events_by_sense.values():
                for e in sense_events:
                    if e.get('text', '') == neg_text:
                        neg_id = e.get('id', '')
                        if neg_id in self.emb_cache.embeddings:
                            sim = float(np.dot(anchor_emb, self.emb_cache.embeddings[neg_id]))
                            max_neg_sim = max(max_neg_sim, sim)
                        break

        # Check cross-lemma negatives (already have _similarity)
        for neg in cross_lemma_negs:
            sim = neg.get('_similarity', 0.0)
            max_neg_sim = max(max_neg_sim, sim)

        # Compute positive similarity (approximation: use text-based lookup)
        pos_text = positive.get('text', '')
        pos_sim = 0.5  # Default

        for sense_events in self.events_by_sense.values():
            for e in sense_events:
                if e.get('text', '') == pos_text:
                    pos_id = e.get('id', '')
                    if pos_id in self.emb_cache.embeddings:
                        pos_sim = float(np.dot(anchor_emb, self.emb_cache.embeddings[pos_id]))
                    break

        # danger = max_neg_sim - pos_sim
        # Higher = negatives closer than positive = more dangerous
        danger = max_neg_sim - pos_sim

        return danger

    def _determine_tier(
        self,
        anchor: Dict,
        sibling_negs: List[Dict],
        cross_lemma_negs: List[Dict]
    ) -> str:
        """
        Determine difficulty tier based on cue strength and confusability.

        NOTE: This is the initial tier assignment. Final tiering uses percentile-based
        danger scoring in a post-processing pass.

        - tier1_easy: High cue strength, low confusability
        - tier2_robust: Medium cue strength or some confusability
        - tier3_adversarial: Low cue strength or high confusability
        """
        cue_strength = anchor.get('quality', {}).get('cue_strength', 0.5)

        # Check confusability from cross-lemma similarities
        max_similarity = 0.0
        for neg in cross_lemma_negs:
            sim = neg.get('_similarity', 0.0)
            max_similarity = max(max_similarity, sim)

        # Tier logic (initial assignment)
        if cue_strength >= 0.7 and max_similarity < 0.7:
            return "tier1_easy"
        elif cue_strength >= 0.4 or max_similarity < 0.85:
            return "tier2_robust"
        else:
            return "tier3_adversarial"

    def build_contrastive_bundles(self, max_bundles: int = 50000) -> List[Dict]:
        """Build contrastive bundles with strict constraints.

        Iterates over events (like v2.2) to maximize bundle count.
        Enforces:
        1. Positive MUST exist (same sense, different text)
        2. Sibling negative MUST exist for multi-sense lemmas
        3. Cross-lemma negative by embedding proximity
        """
        print("\n" + "=" * 60)
        print("Building contrastive bundles (Pass 2)...")
        print("=" * 60)

        bundles = []
        stats = {
            'total_anchors': 0,
            'has_positive': 0,
            'has_sibling': 0,
            'has_cross_lemma': 0,
            'excluded_no_positive': 0,
            'excluded_no_sibling': 0,
            'single_sense_lemmas': 0,
            'tiers': defaultdict(int)
        }

        # Shuffle events for variety
        shuffled_events = list(self.events)
        random.shuffle(shuffled_events)

        # Track used (anchor_text, sense_id) pairs to avoid duplicates
        used_pairs = set()

        for anchor_event in shuffled_events:
            if len(bundles) >= max_bundles:
                break

            stats['total_anchors'] += 1

            lemma = anchor_event.get('lemma', '')
            sense_id = anchor_event.get('sense_id', '')
            pos = anchor_event.get('pos', '')
            anchor_text = anchor_event.get('text', '')

            # Skip if we've already used this anchor
            pair_key = (anchor_text[:50], sense_id)
            if pair_key in used_pairs:
                continue

            # Get positive (MUST exist for v2.3)
            positive = self._get_positive(anchor_event)
            if positive is None:
                stats['excluded_no_positive'] += 1
                continue
            stats['has_positive'] += 1

            # Check if lemma has multiple senses
            lemma_senses = self.senses_by_lemma.get(lemma, [])
            lemma_has_multiple_senses = len(set(lemma_senses)) > 1

            # Get sibling negatives (MANDATORY for multi-sense lemmas)
            sibling_negs = self._get_sibling_negatives(anchor_event, sense_id)

            if lemma_has_multiple_senses and not sibling_negs:
                # Can't satisfy sibling constraint for multi-sense lemma
                stats['excluded_no_sibling'] += 1
                continue

            if not lemma_has_multiple_senses:
                stats['single_sense_lemmas'] += 1
                # Single-sense lemmas stay tier1-only (no sibling possible)

            if sibling_negs:
                stats['has_sibling'] += 1

            # Get cross-lemma negatives (by embedding proximity)
            cross_lemma_negs = self._get_cross_lemma_negatives(anchor_event)
            if cross_lemma_negs:
                stats['has_cross_lemma'] += 1

            # Determine tier
            anchor_formatted = self._format_anchor_or_positive(anchor_event)
            tier = self._determine_tier(anchor_formatted, sibling_negs, cross_lemma_negs)
            stats['tiers'][tier] += 1

            # Compute danger score for percentile-based tiering
            danger = self._compute_danger_score(
                anchor_formatted, positive, sibling_negs, cross_lemma_negs, anchor_event
            )

            # Build bundle
            bundle = {
                "id": hashlib.md5(f"{sense_id}_{anchor_event.get('id', '')}_{len(bundles)}".encode()).hexdigest()[:12],
                "lemma": lemma,
                "anchor": anchor_formatted,
                "positive": positive,
                "negatives": {
                    "within_lemma": sibling_negs,
                    "cross_lemma": cross_lemma_negs
                },
                "metadata": {
                    "anchor_sense": sense_id,
                    "anchor_event_id": anchor_event.get('id', ''),
                    "has_positive": True,
                    "num_within_lemma_negs": len(sibling_negs),
                    "num_cross_lemma_negs": len(cross_lemma_negs),
                    "difficulty_tier": tier,  # Initial tier (will be updated)
                    "danger_score": danger,
                    "bundle_type": "contrastive_v23"
                }
            }

            bundles.append(bundle)
            used_pairs.add(pair_key)

            if len(bundles) % 2000 == 0:
                print(f"  Built {len(bundles)} bundles...")

        # Post-processing: Percentile-based tier assignment
        # Top 3% = tier3_adversarial, next 12% = tier2_robust, rest = tier1_easy
        print(f"\n  Applying percentile-based tier assignment...")

        danger_scores = [b['metadata']['danger_score'] for b in bundles]
        if danger_scores:
            p97 = np.percentile(danger_scores, 97)  # Top 3% threshold
            p85 = np.percentile(danger_scores, 85)  # Top 15% threshold

            # Reset tier counts
            stats['tiers'] = defaultdict(int)

            for b in bundles:
                danger = b['metadata']['danger_score']
                if danger >= p97:
                    new_tier = "tier3_adversarial"
                elif danger >= p85:
                    new_tier = "tier2_robust"
                else:
                    new_tier = "tier1_easy"

                b['metadata']['difficulty_tier'] = new_tier
                stats['tiers'][new_tier] += 1

            print(f"    Danger score thresholds: tier3 >= {p97:.3f}, tier2 >= {p85:.3f}")
            print(f"    New tier distribution: {dict(stats['tiers'])}")

        # Update stats
        # Note: All bundles have positives by construction, so rate is 100%
        self.stats.contrastive_bundles = len(bundles)
        self.stats.positive_rate = 1.0  # All bundles have positives (enforced)

        # Sibling rate: bundles with sibling / (bundles from multi-sense lemmas)
        multi_sense_bundles = len(bundles) - stats['single_sense_lemmas']
        self.stats.sibling_negative_rate = stats['has_sibling'] / multi_sense_bundles if multi_sense_bundles > 0 else 1.0

        self.stats.cross_lemma_negative_rate = stats['has_cross_lemma'] / len(bundles) if bundles else 0
        self.stats.excluded_no_positive = stats['excluded_no_positive']
        self.stats.excluded_no_sibling = stats['excluded_no_sibling']
        self.stats.tier_distribution = dict(stats['tiers'])

        # Compute average negatives
        if bundles:
            self.stats.avg_within_lemma_negs = np.mean([
                b['metadata']['num_within_lemma_negs'] for b in bundles
            ])
            self.stats.avg_cross_lemma_negs = np.mean([
                b['metadata']['num_cross_lemma_negs'] for b in bundles
            ])

        # Track additional stats for yield analysis
        self.stats.total_anchors_tried = stats['total_anchors']
        self.stats.yield_rate = len(bundles) / stats['total_anchors'] if stats['total_anchors'] > 0 else 0
        self.stats.single_sense_lemmas = stats['single_sense_lemmas']
        self.stats.singleton_senses = len(self.singleton_log)

        print(f"\n[OK] Built {len(bundles)} contrastive bundles")
        print(f"    Events processed: {stats['total_anchors']}")
        print(f"    Yield rate: {self.stats.yield_rate:.1%}")
        print(f"    Excluded (no positive): {stats['excluded_no_positive']}")
        print(f"    Excluded (no sibling): {stats['excluded_no_sibling']}")
        print(f"    Single-sense lemma bundles: {stats['single_sense_lemmas']}")
        print(f"    Multi-sense bundles with sibling: {stats['has_sibling']}")
        print(f"    Positive rate: 100% (enforced)")
        print(f"    Sibling-neg rate (multi-sense): {self.stats.sibling_negative_rate:.1%}")
        print(f"    Cross-lemma-neg rate: {self.stats.cross_lemma_negative_rate:.1%}")
        print(f"    Tier distribution: {dict(stats['tiers'])}")

        return bundles

    def build_discrimination_bundles(self, contrastive: List[Dict]) -> List[Dict]:
        """Build discrimination bundles (sense boundary tests)."""
        print("\n[..] Building discrimination bundles...")

        bundles = []
        seen_pairs = set()

        for cb in contrastive[:2000]:  # Sample from contrastive
            lemma = cb['lemma']
            anchor_sense = cb['anchor']['sense_id']

            for neg in cb['negatives']['within_lemma']:
                neg_sense = neg['sense_id']
                pair_key = tuple(sorted([anchor_sense, neg_sense]))

                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                bundle = {
                    "id": hashlib.md5(f"disc_{pair_key}".encode()).hexdigest()[:12],
                    "lemma": lemma,
                    "sense_a": anchor_sense,
                    "sense_b": neg_sense,
                    "example_a": cb['anchor']['text'],
                    "example_b": neg['text'],
                    "metadata": {
                        "bundle_type": "discrimination_v23"
                    }
                }
                bundles.append(bundle)

        self.stats.discrimination_bundles = len(bundles)
        print(f"[OK] Built {len(bundles)} discrimination bundles")
        return bundles

    def build_gloss_bundles(self, contrastive: List[Dict]) -> List[Dict]:
        """Build gloss-matching bundles."""
        print("[..] Building gloss-matching bundles...")

        bundles = []
        seen_senses = set()

        for cb in contrastive:
            sense_id = cb['anchor']['sense_id']
            if sense_id in seen_senses:
                continue
            seen_senses.add(sense_id)

            gloss = cb['anchor'].get('sense_gloss', '')
            if not gloss:
                continue

            bundle = {
                "id": hashlib.md5(f"gloss_{sense_id}".encode()).hexdigest()[:12],
                "sense_id": sense_id,
                "gloss": gloss,
                "example": cb['anchor']['text'],
                "lemma": cb['lemma'],
                "metadata": {
                    "bundle_type": "gloss_matching_v23"
                }
            }
            bundles.append(bundle)

        self.stats.gloss_matching_bundles = len(bundles)
        print(f"[OK] Built {len(bundles)} gloss-matching bundles")
        return bundles

    def compute_content_hash(self, bundles: List[Dict]) -> str:
        """Compute deterministic content hash for all bundles."""
        # Sort bundles by ID for determinism
        sorted_bundles = sorted(bundles, key=lambda x: x.get('id', ''))

        # Hash the content
        content = json.dumps(sorted_bundles, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def build(self):
        """Run the full build pipeline."""
        print("\n" + "#" * 60)
        print(" V2.3 BUNDLE GENERATOR")
        print("#" * 60)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.load_data()

        # Build embedding cache
        self.build_embedding_cache()

        # Build bundles
        contrastive = self.build_contrastive_bundles()
        discrimination = self.build_discrimination_bundles(contrastive)
        gloss = self.build_gloss_bundles(contrastive)

        # Combine all bundles
        all_bundles = contrastive + discrimination + gloss
        self.stats.total_bundles = len(all_bundles)

        # Compute content hash
        self.stats.content_hash = self.compute_content_hash(all_bundles)[:16]

        # Generate fingerprint
        self.stats.fingerprint = (
            f"bundles|v23|hash={self.stats.content_hash}|"
            f"lemmas={self.stats.unique_lemmas}|senses={self.stats.unique_senses}|"
            f"sib={self.stats.sibling_negative_rate:.0%}|xlem={self.stats.cross_lemma_negative_rate:.0%}|"
            f"tier3={self.stats.tier_distribution.get('tier3_adversarial', 0)}"
        )

        self.stats.timestamp = datetime.now().isoformat()

        # Write outputs
        print("\n" + "=" * 60)
        print("Writing outputs...")
        print("=" * 60)

        # Contrastive bundles
        contrastive_path = self.output_dir / "contrastive_bundles.jsonl"
        with open(contrastive_path, 'w', encoding='utf-8') as f:
            for b in contrastive:
                f.write(json.dumps(b, ensure_ascii=False) + '\n')
        print(f"[OK] {contrastive_path}")

        # Discrimination bundles
        disc_path = self.output_dir / "discrimination_bundles.jsonl"
        with open(disc_path, 'w', encoding='utf-8') as f:
            for b in discrimination:
                f.write(json.dumps(b, ensure_ascii=False) + '\n')
        print(f"[OK] {disc_path}")

        # Gloss bundles
        gloss_path = self.output_dir / "gloss_matching_bundles.jsonl"
        with open(gloss_path, 'w', encoding='utf-8') as f:
            for b in gloss:
                f.write(json.dumps(b, ensure_ascii=False) + '\n')
        print(f"[OK] {gloss_path}")

        # All bundles
        all_path = self.output_dir / "all_bundles_v23.jsonl"
        with open(all_path, 'w', encoding='utf-8') as f:
            for b in all_bundles:
                f.write(json.dumps(b, ensure_ascii=False) + '\n')
        print(f"[OK] {all_path}")

        # Stats
        stats_path = self.output_dir / "bundle_stats_v23.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.stats), f, indent=2)
        print(f"[OK] {stats_path}")

        # Write singleton log for enrichment TODO
        if self.singleton_log:
            singleton_path = self.output_dir / "singleton_senses_todo.jsonl"
            # Deduplicate by sense_id
            seen_senses = set()
            unique_singletons = []
            for entry in self.singleton_log:
                sense_id = entry.get('sense_id', '')
                if sense_id not in seen_senses:
                    seen_senses.add(sense_id)
                    unique_singletons.append(entry)

            with open(singleton_path, 'w', encoding='utf-8') as f:
                for entry in unique_singletons:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"[OK] {singleton_path} ({len(unique_singletons)} unique senses for enrichment)")

        # Summary
        print("\n" + "=" * 60)
        print("BUILD SUMMARY")
        print("=" * 60)
        print(f"  Total bundles: {self.stats.total_bundles}")
        print(f"  Contrastive: {self.stats.contrastive_bundles}")
        print(f"  Discrimination: {self.stats.discrimination_bundles}")
        print(f"  Gloss-matching: {self.stats.gloss_matching_bundles}")
        print(f"")
        print(f"  Positive rate: {self.stats.positive_rate:.1%}")
        print(f"  Sibling-neg rate: {self.stats.sibling_negative_rate:.1%} (target: >95%)")
        print(f"  Cross-lemma-neg rate: {self.stats.cross_lemma_negative_rate:.1%} (target: >90%)")
        print(f"")
        print(f"  Tier distribution: {self.stats.tier_distribution}")
        print(f"")
        print(f"  FINGERPRINT: {self.stats.fingerprint}")

        # Detailed yield analysis telemetry
        print("\n" + "=" * 60)
        print("YIELD ANALYSIS (Why not 100%?)")
        print("=" * 60)
        total_tried = self.stats.total_anchors_tried
        total_bundles = self.stats.contrastive_bundles
        excluded_positive = self.stats.excluded_no_positive
        excluded_sibling = self.stats.excluded_no_sibling

        print(f"  Anchors tried: {total_tried}")
        print(f"  Bundles created: {total_bundles} ({self.stats.yield_rate:.1%} yield)")
        print(f"")
        print(f"  EXCLUSION BREAKDOWN:")
        print(f"    No positive (singleton senses): {excluded_positive} ({100*excluded_positive/total_tried:.1f}%)" if total_tried > 0 else "    No positive: 0")
        print(f"    No sibling (multi-sense req): {excluded_sibling} ({100*excluded_sibling/total_tried:.1f}%)" if total_tried > 0 else "    No sibling: 0")
        duplicates = total_tried - total_bundles - excluded_positive - excluded_sibling
        if duplicates > 0:
            print(f"    Duplicates skipped: ~{duplicates} ({100*duplicates/total_tried:.1f}%)")
        print(f"")
        print(f"  TO IMPROVE YIELD:")
        if excluded_positive > 0:
            print(f"    - Add examples for {len(set(e.get('sense_id') for e in self.singleton_log))} singleton senses")
            print(f"      (see singleton_senses_todo.jsonl)")
        if self.stats.sibling_negative_rate < 0.95:
            gap = 0.95 - self.stats.sibling_negative_rate
            print(f"    - Sibling coverage gap: {gap:.1%} below 95% target")
        if self.stats.cross_lemma_negative_rate < 0.90:
            gap = 0.90 - self.stats.cross_lemma_negative_rate
            print(f"    - Cross-lemma coverage gap: {gap:.1%} below 90% target")

        return self.stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build v2.3 bundles")
    parser.add_argument("--events", type=Path,
                       default=Path("paradigm_factory/v2/processed/canonicalized_v21.jsonl"))
    parser.add_argument("--inventory", type=Path,
                       default=Path("paradigm_factory/v2/processed/sense_inventory.json"))
    parser.add_argument("--tier4", type=Path,
                       default=Path("paradigm_factory/v2/killers/tier4/tier4_killers.jsonl"))
    parser.add_argument("--output", type=Path,
                       default=Path("paradigm_factory/v2/bundles_v23"))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent

    events_path = project_root / args.events if not args.events.is_absolute() else args.events
    inventory_path = project_root / args.inventory if not args.inventory.is_absolute() else args.inventory
    tier4_path = project_root / args.tier4 if not args.tier4.is_absolute() else args.tier4
    output_dir = project_root / args.output if not args.output.is_absolute() else args.output

    builder = BundleBuilderV23(
        events_path=events_path,
        inventory_path=inventory_path,
        output_dir=output_dir,
        tier4_path=tier4_path if tier4_path.exists() else None,
        seed=args.seed
    )

    stats = builder.build()

    print(f"\n{stats.fingerprint}")


if __name__ == "__main__":
    main()
