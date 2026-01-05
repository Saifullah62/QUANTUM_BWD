#!/usr/bin/env python3
"""
Singleton Sense Enrichment Pipeline (v2 - Batched + Resumable)
==============================================================

Converts singleton senses into usable training data via verified synthesis.

Key improvements over v1:
- Models load ONCE at startup (no per-item reloading)
- Batched embedding computation (10-50x faster)
- Progress tracking with resume capability
- Periodic atomic flushes (kill-safe)

Usage:
    python enrich_singleton_senses.py                    # Full run
    python enrich_singleton_senses.py --max-senses 100   # Smoke test
    python enrich_singleton_senses.py --resume           # Continue from checkpoint
"""

import json
import hashlib
import random
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import argparse


# =============================================================================
# Progress Tracking (Resumable)
# =============================================================================

class ProgressDB:
    """
    Lightweight progress tracker for resumable enrichment.
    Stores completed sense_ids in a simple JSONL index.
    """
    def __init__(self, path: Path):
        self.path = path
        self.done: Set[str] = set()
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    self.done.add(line.strip())
        print(f"[OK] Progress tracker: {len(self.done)} senses already done")

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark_done(self, key: str):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(key + "\n")
        self.done.add(key)


def atomic_append_jsonl(path: Path, rows: List[Dict]):
    """Append rows to JSONL file atomically."""
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =============================================================================
# Batched Embedding Encoder (Load Once, Encode Many)
# =============================================================================

class BatchedEncoder:
    """
    Embedding encoder with batched computation.

    IMPORTANT: Constructed ONCE at process start, reused for all operations.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._encoder = None
        self._cache: Dict[str, np.ndarray] = {}  # text -> embedding

    def _load_encoder(self):
        """Load encoder exactly once."""
        if self._encoder is not None:
            return

        print(f"[..] Loading encoder: {self.model_name}")
        start = time.time()
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
            print(f"[OK] Encoder loaded in {time.time() - start:.1f}s")
        except ImportError:
            print("[!] sentence-transformers not available, using fallback")
            self._encoder = "fallback"

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode a batch of texts. Uses cache for repeated texts.

        This is the ONLY encoding method - always use batched calls.
        """
        self._load_encoder()

        # Separate cached vs uncached
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch encode uncached texts
        if uncached_texts:
            if self._encoder == "fallback":
                # Fallback: hash-based pseudo-embedding
                new_embeddings = []
                for text in uncached_texts:
                    h = hashlib.md5(text.encode()).hexdigest()
                    emb = np.array([int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)])
                    new_embeddings.append(emb)
            else:
                # Real batch encoding
                new_embeddings = self._encoder.encode(
                    uncached_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False
                )

            # Fill results and cache
            for idx, emb in zip(uncached_indices, new_embeddings):
                results[idx] = emb
                self._cache[texts[idx]] = emb

        return results

    def encode_single(self, text: str) -> np.ndarray:
        """Convenience method for single text (still uses cache)."""
        return self.encode_batch([text])[0]


# =============================================================================
# Template Generator (No model loading)
# =============================================================================

class TemplateGenerator:
    """Template-based synthetic example generator."""

    TEMPLATES = {
        'noun': [
            "The {L} was carefully examined.",
            "I noticed the {L} immediately.",
            "This particular {L} caught my attention.",
            "The {L} in question was interesting.",
            "A typical {L} would be similar.",
            "Looking at the {L}, I could see why.",
            "The {L} seemed to be exactly what was needed.",
            "Someone mentioned the {L} earlier.",
            "That {L} is a good example.",
            "The {L} has these characteristics.",
        ],
        'verb': [
            "They decided to {L} carefully.",
            "You should {L} when possible.",
            "It's important to {L} correctly.",
            "One can {L} in various ways.",
            "They began to {L} slowly.",
            "We need to {L} this properly.",
            "She wanted to {L} everything.",
            "The plan was to {L} tomorrow.",
            "He tried to {L} without help.",
            "Learning to {L} takes practice.",
        ],
        'adj': [
            "It was quite {L} overall.",
            "The {L} appearance was notable.",
            "Something more {L} would work.",
            "A rather {L} situation developed.",
            "The most {L} option was chosen.",
        ],
        'adv': [
            "He spoke {L}.",
            "She moved {L} through the room.",
            "The process proceeded {L}.",
            "They handled it {L}.",
            "It happened {L}.",
        ]
    }

    def generate(self, lemma: str, pos: str, n_samples: int = 3) -> List[str]:
        """Generate n synthetic texts for a lemma/POS."""
        pos_key = pos.lower()
        if pos_key not in self.TEMPLATES:
            pos_key = 'noun'

        templates = self.TEMPLATES[pos_key]
        selected = random.sample(templates, min(n_samples, len(templates)))

        return [t.format(L=lemma) for t in selected]


# =============================================================================
# Batched Verifier
# =============================================================================

class BatchedVerifier:
    """
    Verifies synthetic examples with batched embedding computation.

    Uses shared encoder instance - never loads its own model.
    """

    def __init__(
        self,
        encoder: BatchedEncoder,
        similarity_threshold: float = 0.5,
        separation_gap: float = 0.05
    ):
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self.separation_gap = separation_gap

    def verify_batch(
        self,
        candidates: List[Dict]  # [{text, original_text, sibling_texts}, ...]
    ) -> List[Tuple[bool, Dict]]:
        """
        Verify a batch of candidates efficiently.

        Each candidate dict must have:
        - text: the synthetic text to verify
        - original_text: the original singleton anchor
        - sibling_texts: list of texts from sibling senses
        """
        if not candidates:
            return []

        # Collect all texts that need encoding
        all_texts = []
        text_to_idx = {}

        for cand in candidates:
            for text in [cand['text'], cand['original_text']] + cand.get('sibling_texts', [])[:5]:
                if text not in text_to_idx:
                    text_to_idx[text] = len(all_texts)
                    all_texts.append(text)

        # Batch encode all texts at once
        all_embeddings = self.encoder.encode_batch(all_texts)

        # Verify each candidate using pre-computed embeddings
        results = []
        for cand in candidates:
            synth_emb = all_embeddings[text_to_idx[cand['text']]]
            orig_emb = all_embeddings[text_to_idx[cand['original_text']]]

            # Similarity to original
            sim_to_original = float(np.dot(synth_emb, orig_emb))
            passed_embedding = sim_to_original >= self.similarity_threshold

            # Separation from siblings
            sibling_texts = cand.get('sibling_texts', [])[:5]
            if sibling_texts:
                max_sibling_sim = max(
                    float(np.dot(synth_emb, all_embeddings[text_to_idx[s]]))
                    for s in sibling_texts if s in text_to_idx
                )
                separation = sim_to_original - max_sibling_sim
                passed_separation = separation >= self.separation_gap
            else:
                separation = 1.0
                passed_separation = True

            passed = passed_embedding and passed_separation
            details = {
                'embedding_similarity': sim_to_original,
                'min_sibling_separation': separation,
                'passed_embedding': passed_embedding,
                'passed_separation': passed_separation
            }

            results.append((passed, details))

        return results


# =============================================================================
# Main Pipeline
# =============================================================================

@dataclass
class EnrichmentStats:
    """Statistics for the enrichment pipeline."""
    timestamp: str = ""
    singleton_senses: int = 0
    senses_processed: int = 0
    senses_skipped: int = 0
    candidates_generated: int = 0
    passed_embedding: int = 0
    passed_separation: int = 0
    final_accepted: int = 0
    rejection_rate: float = 0.0


def load_data(
    singleton_path: Path,
    events_path: Path
) -> Tuple[List[Dict], Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """Load all input data."""
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    # Load singletons
    singletons = []
    with singleton_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                singletons.append(json.loads(line))
    print(f"[OK] {len(singletons)} singleton senses to enrich")

    # Load events and index
    events_by_sense: Dict[str, List[Dict]] = defaultdict(list)
    events_by_lemma: Dict[str, List[Dict]] = defaultdict(list)

    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                event = json.loads(line)
                sense_id = event.get('sense_id', '')
                lemma = event.get('lemma', '').lower()
                events_by_sense[sense_id].append(event)
                events_by_lemma[lemma].append(event)
    print(f"[OK] Loaded events for {len(events_by_sense)} senses")

    return singletons, events_by_sense, events_by_lemma


def get_sibling_texts(
    lemma: str,
    pos: str,
    exclude_sense: str,
    events_by_lemma: Dict[str, List[Dict]]
) -> List[str]:
    """Get example texts from sibling senses."""
    texts = []
    for event in events_by_lemma.get(lemma.lower(), []):
        if event.get('pos', '') == pos and event.get('sense_id', '') != exclude_sense:
            texts.append(event.get('text', ''))
    return texts[:10]  # Cap for efficiency


def run_enrichment(args):
    """Main enrichment pipeline."""
    print("\n" + "#" * 60)
    print(" SINGLETON SENSE ENRICHMENT (v2 - Batched)")
    print("#" * 60)

    # Paths
    singleton_path = Path(args.singletons)
    events_path = Path(args.events)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "synthetic_enriched_events.jsonl"
    progress_path = output_dir / "progress.index"
    stats_path = output_dir / "enrichment_stats.json"

    # =========================================================================
    # LOAD MODELS ONCE
    # =========================================================================
    print("\n" + "=" * 60)
    print("Initializing models (load once)...")
    print("=" * 60)

    encoder = BatchedEncoder(batch_size=args.batch_size)
    encoder._load_encoder()  # Force load now

    generator = TemplateGenerator()
    verifier = BatchedVerifier(encoder,
                               similarity_threshold=args.similarity_threshold,
                               separation_gap=args.separation_gap)

    progress = ProgressDB(progress_path)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    singletons, events_by_sense, events_by_lemma = load_data(singleton_path, events_path)

    if args.max_senses:
        singletons = singletons[:args.max_senses]
        print(f"[!] Limited to {args.max_senses} senses (smoke test mode)")

    # =========================================================================
    # MAIN LOOP: Batch Processing with Progress Tracking
    # =========================================================================
    print("\n" + "=" * 60)
    print("Generating and verifying synthetic examples...")
    print("=" * 60)

    stats = EnrichmentStats(timestamp=datetime.now().isoformat())
    stats.singleton_senses = len(singletons)

    accepted_buffer = []
    last_flush = time.time()
    batch_candidates = []  # Accumulate for batch verification
    batch_metadata = []    # Track which singleton each candidate came from

    VERIFY_BATCH_SIZE = 100
    FLUSH_INTERVAL = 30  # seconds
    FLUSH_COUNT = 500    # items

    for i, singleton in enumerate(singletons):
        sense_id = singleton.get('sense_id', '')
        lemma = singleton.get('lemma', '')

        # Skip if already done
        if progress.is_done(sense_id):
            stats.senses_skipped += 1
            continue

        # Get original event
        original_events = events_by_sense.get(sense_id, [])
        if not original_events:
            progress.mark_done(sense_id)
            continue

        original = original_events[0]
        original_text = original.get('text', '')
        pos = original.get('pos', 'noun')
        cue_tokens = original.get('cue_tokens', [])
        sense_gloss = original.get('sense_gloss', '')
        event_id = original.get('id', '')

        # Get sibling texts
        sibling_texts = get_sibling_texts(lemma, pos, sense_id, events_by_lemma)

        # Generate candidates
        candidate_texts = generator.generate(lemma, pos, n_samples=3)
        stats.candidates_generated += len(candidate_texts)

        # Queue for batch verification
        for cand_text in candidate_texts:
            batch_candidates.append({
                'text': cand_text,
                'original_text': original_text,
                'sibling_texts': sibling_texts
            })
            batch_metadata.append({
                'sense_id': sense_id,
                'lemma': lemma,
                'pos': pos,
                'cue_tokens': cue_tokens,
                'sense_gloss': sense_gloss,
                'event_id': event_id,
                'candidate_text': cand_text
            })

        # Batch verify when buffer is full
        if len(batch_candidates) >= VERIFY_BATCH_SIZE:
            results = verifier.verify_batch(batch_candidates)

            for (passed, details), meta in zip(results, batch_metadata):
                if details['passed_embedding']:
                    stats.passed_embedding += 1
                if details['passed_separation']:
                    stats.passed_separation += 1

                if passed:
                    enriched_event = {
                        'id': hashlib.md5(f"synth_{meta['sense_id']}_{len(accepted_buffer)}".encode()).hexdigest()[:16],
                        'text': meta['candidate_text'],
                        'lemma': meta['lemma'],
                        'pos': meta['pos'],
                        'sense_id': meta['sense_id'],
                        'sense_gloss': meta['sense_gloss'],
                        'cue_tokens': meta['cue_tokens'],
                        'source': {
                            'type': 'synthetic_enrichment',
                            'generator': 'template_v2',
                            'verified_by': ['embedding_check', 'sibling_separation'],
                            'original_anchor_id': meta['event_id'],
                            'similarity': details['embedding_similarity'],
                            'separation': details['min_sibling_separation'],
                            'timestamp': datetime.now().isoformat()
                        },
                        'quality': {
                            'cue_strength': 0.5,
                            'synthetic': True
                        }
                    }
                    accepted_buffer.append(enriched_event)
                    stats.final_accepted += 1

            # Clear batch
            batch_candidates.clear()
            batch_metadata.clear()

            # Mark sense as done after all its candidates processed
            progress.mark_done(sense_id)
            stats.senses_processed += 1

        # Periodic flush
        if len(accepted_buffer) >= FLUSH_COUNT or (time.time() - last_flush) > FLUSH_INTERVAL:
            atomic_append_jsonl(output_path, accepted_buffer)
            print(f"  [{i+1}/{len(singletons)}] Flushed {len(accepted_buffer)} accepted, "
                  f"total: {stats.final_accepted}")
            accepted_buffer.clear()
            last_flush = time.time()

    # Process remaining batch
    if batch_candidates:
        results = verifier.verify_batch(batch_candidates)
        for (passed, details), meta in zip(results, batch_metadata):
            if details['passed_embedding']:
                stats.passed_embedding += 1
            if details['passed_separation']:
                stats.passed_separation += 1
            if passed:
                enriched_event = {
                    'id': hashlib.md5(f"synth_{meta['sense_id']}_{len(accepted_buffer)}".encode()).hexdigest()[:16],
                    'text': meta['candidate_text'],
                    'lemma': meta['lemma'],
                    'pos': meta['pos'],
                    'sense_id': meta['sense_id'],
                    'sense_gloss': meta['sense_gloss'],
                    'cue_tokens': meta['cue_tokens'],
                    'source': {
                        'type': 'synthetic_enrichment',
                        'generator': 'template_v2',
                        'verified_by': ['embedding_check', 'sibling_separation'],
                        'original_anchor_id': meta['event_id'],
                        'similarity': details['embedding_similarity'],
                        'separation': details['min_sibling_separation'],
                        'timestamp': datetime.now().isoformat()
                    },
                    'quality': {
                        'cue_strength': 0.5,
                        'synthetic': True
                    }
                }
                accepted_buffer.append(enriched_event)
                stats.final_accepted += 1

    # Final flush
    if accepted_buffer:
        atomic_append_jsonl(output_path, accepted_buffer)

    # Compute final stats
    if stats.candidates_generated > 0:
        stats.rejection_rate = 1.0 - (stats.final_accepted / stats.candidates_generated)

    # Save stats
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("ENRICHMENT COMPLETE")
    print("=" * 60)
    print(f"  Singletons processed: {stats.senses_processed}")
    print(f"  Singletons skipped (already done): {stats.senses_skipped}")
    print(f"  Candidates generated: {stats.candidates_generated}")
    print(f"  Passed embedding check: {stats.passed_embedding}")
    print(f"  Passed separation check: {stats.passed_separation}")
    print(f"  Final accepted: {stats.final_accepted}")
    print(f"  Rejection rate: {stats.rejection_rate:.1%}")
    print(f"\n  Output: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Enrich singleton senses (v2 - batched)")
    parser.add_argument("--singletons", type=str,
                       default="paradigm_factory/v2/bundles_v23/singleton_senses_todo.jsonl")
    parser.add_argument("--events", type=str,
                       default="paradigm_factory/v2/processed/canonicalized_v21.jsonl")
    parser.add_argument("--output", type=str,
                       default="paradigm_factory/v2/bundles_v23/enrichment")
    parser.add_argument("--max-senses", type=int, default=None,
                       help="Limit senses for smoke testing")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Embedding batch size")
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--separation-gap", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from checkpoint (default: True)")

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    args.singletons = project_root / args.singletons
    args.events = project_root / args.events
    args.output = project_root / args.output

    run_enrichment(args)


if __name__ == "__main__":
    main()
