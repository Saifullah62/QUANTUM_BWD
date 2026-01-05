#!/usr/bin/env python3
"""
Adversarial Bundle Miner
========================

Mines for the hardest cases in the bundle dataset:
1. Surface-form traps: Same lemma, highly similar surface forms, different senses
2. Sibling confusion: Sibling senses with highest embedding similarity
3. Cross-lemma confusers: Different lemmas but semantically very similar

Outputs dedicated adversarial_bundles.jsonl for curriculum injection.
"""

import json
import hashlib
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Set
from datetime import datetime


@dataclass
class AdversarialStats:
    """Statistics for adversarial mining."""
    timestamp: str = ""

    # Mining results
    surface_form_traps: int = 0
    sibling_confusers: int = 0
    cross_lemma_confusers: int = 0
    total_adversarial: int = 0

    # Coverage
    lemmas_with_adversarial: int = 0
    senses_with_adversarial: int = 0

    # Quality metrics
    avg_sibling_similarity: float = 0.0
    avg_cross_lemma_similarity: float = 0.0


class AdversarialMiner:
    """Mines adversarial hard negatives from existing bundles."""

    def __init__(
        self,
        bundles_path: Path,
        events_path: Path,
        output_dir: Path,
        seed: int = 42
    ):
        self.bundles_path = bundles_path
        self.events_path = events_path
        self.output_dir = output_dir
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.bundles: List[Dict] = []
        self.events: List[Dict] = []
        self.events_by_lemma: Dict[str, List[Dict]] = defaultdict(list)
        self.events_by_sense: Dict[str, List[Dict]] = defaultdict(list)

        # Embedding cache
        self._encoder = None
        self.embeddings: Dict[str, np.ndarray] = {}

        self.stats = AdversarialStats()

    def _get_encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print("[OK] Loaded encoder")
            except ImportError:
                print("[!] sentence-transformers not available")
                self._encoder = "fallback"
        return self._encoder

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to embedding."""
        encoder = self._get_encoder()
        if encoder == "fallback":
            h = hashlib.md5(text.encode()).hexdigest()
            return np.array([int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)])
        return encoder.encode(text, show_progress_bar=False)

    def load_data(self):
        """Load bundles and events."""
        print("=" * 60)
        print("Loading data...")
        print("=" * 60)

        # Load bundles
        with open(self.bundles_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.bundles.append(json.loads(line))
        print(f"[OK] Loaded {len(self.bundles)} bundles")

        # Load events
        with open(self.events_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    self.events.append(event)
                    lemma = event.get('lemma', '').lower()
                    sense_id = event.get('sense_id', '')
                    self.events_by_lemma[lemma].append(event)
                    self.events_by_sense[sense_id].append(event)
        print(f"[OK] Loaded {len(self.events)} events")

    def mine_surface_form_traps(self, max_bundles: int = 500) -> List[Dict]:
        """
        Mine surface-form traps: same lemma, similar text patterns, different senses.

        These are cases where the surface context is very similar but the meaning differs.
        """
        print("\n" + "=" * 60)
        print("Mining surface-form traps...")
        print("=" * 60)

        traps = []
        seen = set()

        # Group bundles by lemma
        by_lemma = defaultdict(list)
        for b in self.bundles:
            if b.get('metadata', {}).get('bundle_type') == 'contrastive_v23':
                lemma = b.get('lemma', '')
                by_lemma[lemma].append(b)

        # For each lemma with multiple bundles, find surface-similar pairs
        for lemma, lemma_bundles in by_lemma.items():
            if len(lemma_bundles) < 2:
                continue

            # Get unique senses
            senses = {}
            for b in lemma_bundles:
                sense_id = b.get('anchor', {}).get('sense_id', '')
                if sense_id and sense_id not in senses:
                    senses[sense_id] = b

            if len(senses) < 2:
                continue

            # Compare all sense pairs
            sense_ids = list(senses.keys())
            for i, s1 in enumerate(sense_ids):
                for s2 in sense_ids[i+1:]:
                    b1 = senses[s1]
                    b2 = senses[s2]

                    text1 = b1.get('anchor', {}).get('text', '')
                    text2 = b2.get('anchor', {}).get('text', '')

                    # Skip if already seen this pair
                    pair_key = tuple(sorted([s1, s2]))
                    if pair_key in seen:
                        continue

                    # Compute text similarity (embedding-based)
                    if text1 not in self.embeddings:
                        self.embeddings[text1] = self._encode(text1)
                    if text2 not in self.embeddings:
                        self.embeddings[text2] = self._encode(text2)

                    sim = float(np.dot(self.embeddings[text1], self.embeddings[text2]))

                    # High similarity (>0.7) = surface-form trap
                    if sim > 0.7:
                        trap = {
                            "id": hashlib.md5(f"sft_{pair_key}".encode()).hexdigest()[:12],
                            "type": "surface_form_trap",
                            "lemma": lemma,
                            "sense_a": s1,
                            "sense_b": s2,
                            "text_a": text1,
                            "text_b": text2,
                            "gloss_a": b1.get('anchor', {}).get('sense_gloss', ''),
                            "gloss_b": b2.get('anchor', {}).get('sense_gloss', ''),
                            "surface_similarity": sim,
                            "metadata": {
                                "bundle_type": "adversarial_surface_trap",
                                "difficulty_tier": "tier3_adversarial"
                            }
                        }
                        traps.append(trap)
                        seen.add(pair_key)

                        if len(traps) >= max_bundles:
                            break

                if len(traps) >= max_bundles:
                    break

            if len(traps) >= max_bundles:
                break

        # Sort by similarity (highest first)
        traps.sort(key=lambda x: -x['surface_similarity'])

        self.stats.surface_form_traps = len(traps)
        print(f"[OK] Found {len(traps)} surface-form traps")
        if traps:
            print(f"    Similarity range: {traps[-1]['surface_similarity']:.3f} - {traps[0]['surface_similarity']:.3f}")

        return traps[:max_bundles]

    def mine_sibling_confusers(self, max_bundles: int = 500) -> List[Dict]:
        """
        Mine sibling confusers: same lemma, same POS, different sense,
        highest embedding similarity between examples.
        """
        print("\n" + "=" * 60)
        print("Mining sibling confusers...")
        print("=" * 60)

        confusers = []
        seen = set()
        similarities = []

        # For each bundle with within-lemma negatives
        for b in self.bundles:
            if b.get('metadata', {}).get('bundle_type') != 'contrastive_v23':
                continue

            anchor = b.get('anchor', {})
            anchor_text = anchor.get('text', '')
            anchor_sense = anchor.get('sense_id', '')
            lemma = b.get('lemma', '')

            within_lemma_negs = b.get('negatives', {}).get('within_lemma', [])
            if not within_lemma_negs:
                continue

            # Encode anchor
            if anchor_text not in self.embeddings:
                self.embeddings[anchor_text] = self._encode(anchor_text)

            # Find most similar sibling
            best_neg = None
            best_sim = -1

            for neg in within_lemma_negs:
                neg_text = neg.get('text', '')
                neg_sense = neg.get('sense_id', '')

                pair_key = tuple(sorted([anchor_sense, neg_sense]))
                if pair_key in seen:
                    continue

                if neg_text not in self.embeddings:
                    self.embeddings[neg_text] = self._encode(neg_text)

                sim = float(np.dot(self.embeddings[anchor_text], self.embeddings[neg_text]))

                if sim > best_sim:
                    best_sim = sim
                    best_neg = neg

            if best_neg and best_sim > 0.6:  # Threshold for "confusing"
                pair_key = tuple(sorted([anchor_sense, best_neg.get('sense_id', '')]))
                if pair_key not in seen:
                    confuser = {
                        "id": hashlib.md5(f"sib_{pair_key}_{len(confusers)}".encode()).hexdigest()[:12],
                        "type": "sibling_confuser",
                        "lemma": lemma,
                        "anchor_sense": anchor_sense,
                        "confuser_sense": best_neg.get('sense_id', ''),
                        "anchor_text": anchor_text,
                        "confuser_text": best_neg.get('text', ''),
                        "anchor_gloss": anchor.get('sense_gloss', ''),
                        "confuser_gloss": best_neg.get('sense_gloss', ''),
                        "sibling_similarity": best_sim,
                        "metadata": {
                            "bundle_type": "adversarial_sibling_confuser",
                            "difficulty_tier": "tier3_adversarial"
                        }
                    }
                    confusers.append(confuser)
                    seen.add(pair_key)
                    similarities.append(best_sim)

            if len(confusers) >= max_bundles:
                break

        # Sort by similarity (highest first)
        confusers.sort(key=lambda x: -x['sibling_similarity'])

        self.stats.sibling_confusers = len(confusers)
        self.stats.avg_sibling_similarity = np.mean(similarities) if similarities else 0.0

        print(f"[OK] Found {len(confusers)} sibling confusers")
        if confusers:
            print(f"    Similarity range: {confusers[-1]['sibling_similarity']:.3f} - {confusers[0]['sibling_similarity']:.3f}")
            print(f"    Average similarity: {self.stats.avg_sibling_similarity:.3f}")

        return confusers[:max_bundles]

    def mine_cross_lemma_confusers(self, max_bundles: int = 500) -> List[Dict]:
        """
        Mine cross-lemma confusers: different lemmas but very similar meaning.
        """
        print("\n" + "=" * 60)
        print("Mining cross-lemma confusers...")
        print("=" * 60)

        confusers = []
        similarities = []
        seen = set()

        for b in self.bundles:
            if b.get('metadata', {}).get('bundle_type') != 'contrastive_v23':
                continue

            anchor = b.get('anchor', {})
            anchor_text = anchor.get('text', '')
            anchor_sense = anchor.get('sense_id', '')
            anchor_lemma = b.get('lemma', '')

            cross_lemma_negs = b.get('negatives', {}).get('cross_lemma', [])
            if not cross_lemma_negs:
                continue

            # Encode anchor
            if anchor_text not in self.embeddings:
                self.embeddings[anchor_text] = self._encode(anchor_text)

            # Find most similar cross-lemma
            for neg in cross_lemma_negs:
                neg_text = neg.get('text', '')
                neg_sense = neg.get('sense_id', '')

                pair_key = tuple(sorted([anchor_sense, neg_sense]))
                if pair_key in seen:
                    continue

                if neg_text not in self.embeddings:
                    self.embeddings[neg_text] = self._encode(neg_text)

                sim = float(np.dot(self.embeddings[anchor_text], self.embeddings[neg_text]))

                # Very high similarity (>0.85) = cross-lemma confuser
                if sim > 0.85:
                    confuser = {
                        "id": hashlib.md5(f"xlm_{pair_key}_{len(confusers)}".encode()).hexdigest()[:12],
                        "type": "cross_lemma_confuser",
                        "anchor_lemma": anchor_lemma,
                        "anchor_sense": anchor_sense,
                        "confuser_sense": neg_sense,
                        "anchor_text": anchor_text,
                        "confuser_text": neg_text,
                        "cross_lemma_similarity": sim,
                        "metadata": {
                            "bundle_type": "adversarial_cross_lemma",
                            "difficulty_tier": "tier3_adversarial"
                        }
                    }
                    confusers.append(confuser)
                    seen.add(pair_key)
                    similarities.append(sim)

                if len(confusers) >= max_bundles:
                    break

            if len(confusers) >= max_bundles:
                break

        # Sort by similarity (highest first)
        confusers.sort(key=lambda x: -x['cross_lemma_similarity'])

        self.stats.cross_lemma_confusers = len(confusers)
        self.stats.avg_cross_lemma_similarity = np.mean(similarities) if similarities else 0.0

        print(f"[OK] Found {len(confusers)} cross-lemma confusers")
        if confusers:
            print(f"    Similarity range: {confusers[-1]['cross_lemma_similarity']:.3f} - {confusers[0]['cross_lemma_similarity']:.3f}")
            print(f"    Average similarity: {self.stats.avg_cross_lemma_similarity:.3f}")

        return confusers[:max_bundles]

    def mine(self):
        """Run the full mining pipeline."""
        print("\n" + "#" * 60)
        print(" ADVERSARIAL BUNDLE MINER")
        print("#" * 60)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.load_data()

        # Mine adversarial cases
        surface_traps = self.mine_surface_form_traps()
        sibling_confusers = self.mine_sibling_confusers()
        cross_lemma_confusers = self.mine_cross_lemma_confusers()

        # Combine all
        all_adversarial = surface_traps + sibling_confusers + cross_lemma_confusers
        self.stats.total_adversarial = len(all_adversarial)

        # Compute coverage
        lemmas = set()
        senses = set()
        for a in all_adversarial:
            if 'lemma' in a:
                lemmas.add(a['lemma'])
            if 'anchor_lemma' in a:
                lemmas.add(a['anchor_lemma'])
            if 'sense_a' in a:
                senses.add(a['sense_a'])
            if 'sense_b' in a:
                senses.add(a['sense_b'])
            if 'anchor_sense' in a:
                senses.add(a['anchor_sense'])
            if 'confuser_sense' in a:
                senses.add(a['confuser_sense'])

        self.stats.lemmas_with_adversarial = len(lemmas)
        self.stats.senses_with_adversarial = len(senses)
        self.stats.timestamp = datetime.now().isoformat()

        # Write outputs
        print("\n" + "=" * 60)
        print("Writing outputs...")
        print("=" * 60)

        # All adversarial bundles
        adversarial_path = self.output_dir / "adversarial_bundles.jsonl"
        with open(adversarial_path, 'w', encoding='utf-8') as f:
            for a in all_adversarial:
                f.write(json.dumps(a, ensure_ascii=False) + '\n')
        print(f"[OK] {adversarial_path}")

        # By type
        for name, bundles in [
            ("surface_form_traps.jsonl", surface_traps),
            ("sibling_confusers.jsonl", sibling_confusers),
            ("cross_lemma_confusers.jsonl", cross_lemma_confusers)
        ]:
            path = self.output_dir / name
            with open(path, 'w', encoding='utf-8') as f:
                for b in bundles:
                    f.write(json.dumps(b, ensure_ascii=False) + '\n')
            print(f"[OK] {path} ({len(bundles)} items)")

        # Stats
        stats_path = self.output_dir / "adversarial_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.stats), f, indent=2)
        print(f"[OK] {stats_path}")

        # Summary
        print("\n" + "=" * 60)
        print("ADVERSARIAL MINING SUMMARY")
        print("=" * 60)
        print(f"  Surface-form traps: {self.stats.surface_form_traps}")
        print(f"  Sibling confusers: {self.stats.sibling_confusers}")
        print(f"  Cross-lemma confusers: {self.stats.cross_lemma_confusers}")
        print(f"  Total adversarial: {self.stats.total_adversarial}")
        print(f"")
        print(f"  Lemmas covered: {self.stats.lemmas_with_adversarial}")
        print(f"  Senses covered: {self.stats.senses_with_adversarial}")

        return self.stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mine adversarial hard negatives")
    parser.add_argument("--bundles", type=Path,
                       default=Path("paradigm_factory/v2/bundles_v23/contrastive_bundles.jsonl"))
    parser.add_argument("--events", type=Path,
                       default=Path("paradigm_factory/v2/processed/canonicalized_v21.jsonl"))
    parser.add_argument("--output", type=Path,
                       default=Path("paradigm_factory/v2/bundles_v23/adversarial"))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    bundles_path = project_root / args.bundles if not args.bundles.is_absolute() else args.bundles
    events_path = project_root / args.events if not args.events.is_absolute() else args.events
    output_dir = project_root / args.output if not args.output.is_absolute() else args.output

    miner = AdversarialMiner(
        bundles_path=bundles_path,
        events_path=events_path,
        output_dir=output_dir,
        seed=args.seed
    )

    miner.mine()


if __name__ == "__main__":
    main()
