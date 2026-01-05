"""
Bundle Generator v2.1
=====================

Assembles raw usage events into training bundles with adversarial hard negatives.
Implements both lexical and embedding-based confusability mining.

Schema v2.1 compliant format.
"""

import json
import uuid
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Set, Tuple
import hashlib

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class BundleItem:
    """A single item in a bundle (v2.1 schema)."""
    item_id: str
    role: str  # anchor, positive, negative, hard_negative
    sense_id: str
    text: str
    span: Dict  # {start, end, surface}
    context_window: Dict  # {left, right}
    cue_tokens: List[str]
    cue_type: List[str]
    source_event_id: str
    difficulty: float
    hardness: str  # easy, medium, hard
    style: str
    negative_type: str = ""  # easy, medium, hard_lexical, hard_embedding
    lexical_overlap: float = 0.0
    embedding_similarity: float = 0.0
    confusion_reason: str = ""


@dataclass
class Bundle:
    """A complete training bundle (v2.1 schema)."""
    schema_version: str = "2.1"
    bundle_id: str = ""
    paradigm: str = "polysemy"
    word: Dict = field(default_factory=dict)
    sense_catalog: List[Dict] = field(default_factory=list)
    items: List[BundleItem] = field(default_factory=list)
    pairings: Dict = field(default_factory=dict)
    contrastive_targets: Dict = field(default_factory=dict)
    provenance: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "paradigm": self.paradigm,
            "word": self.word,
            "sense_catalog": self.sense_catalog,
            "items": [asdict(item) for item in self.items],
            "pairings": self.pairings,
            "contrastive_targets": self.contrastive_targets,
            "provenance": self.provenance
        }


class BundleGenerator:
    """Generate bundles from raw usage events (v2.1 schema)."""

    def __init__(
        self,
        sense_inventory: Dict,
        embedder=None,  # Optional: sentence transformer for embedding similarity
        output_dir: Path = None
    ):
        self.sense_inventory = sense_inventory
        self.embedder = embedder
        self.output_dir = output_dir or Path("paradigm_factory/v2/bundles")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build lookup structures
        self.lemma_senses = self._build_lemma_lookup()

        # Cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {} if NUMPY_AVAILABLE else {}

    def _build_lemma_lookup(self) -> Dict[str, List[Dict]]:
        """Build lookup from lemma to senses."""
        lookup = {}
        for entry in self.sense_inventory.values():
            lemma = entry.get('lemma') if isinstance(entry, dict) else entry.lemma
            senses = entry.get('senses', []) if isinstance(entry, dict) else entry.senses
            lookup[lemma] = senses
        return lookup

    def load_raw_events(self, events_path: Path) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Load and index raw events by lemma and sense.
        Returns: {lemma: {sense_id: [events]}}
        """
        events_by_sense = defaultdict(lambda: defaultdict(list))

        with open(events_path) as f:
            for line in f:
                event = json.loads(line)
                lemma = event['lemma']
                sense_id = event['sense_id']
                events_by_sense[lemma][sense_id].append(event)

        # Print stats
        total_events = sum(
            sum(len(evts) for evts in senses.values())
            for senses in events_by_sense.values()
        )
        print(f"Loaded {total_events} events across {len(events_by_sense)} lemmas")

        return dict(events_by_sense)

    def compute_lexical_overlap(self, ctx1: str, ctx2: str) -> float:
        """Compute lexical overlap between two contexts."""
        words1 = set(ctx1.lower().split())
        words2 = set(ctx2.lower().split())

        # Remove very common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'on'}
        words1 -= stopwords
        words2 -= stopwords

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def compute_ngram_overlap(self, ctx1: str, ctx2: str, n: int = 2) -> float:
        """Compute n-gram overlap between two contexts."""
        def get_ngrams(text, n):
            words = text.lower().split()
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

        ng1 = get_ngrams(ctx1, n)
        ng2 = get_ngrams(ctx2, n)

        if not ng1 or not ng2:
            return 0.0

        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)

        return intersection / union if union > 0 else 0.0

    def compute_embedding_similarity(self, ctx1: str, ctx2: str) -> float:
        """Compute embedding similarity between two contexts."""
        if not self.embedder or not NUMPY_AVAILABLE:
            return 0.0

        # Check cache
        h1 = hashlib.md5(ctx1.encode()).hexdigest()
        h2 = hashlib.md5(ctx2.encode()).hexdigest()

        if h1 not in self.embedding_cache:
            self.embedding_cache[h1] = self.embedder.encode(ctx1)
        if h2 not in self.embedding_cache:
            self.embedding_cache[h2] = self.embedder.encode(ctx2)

        emb1 = self.embedding_cache[h1]
        emb2 = self.embedding_cache[h2]

        # Cosine similarity
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(sim)

    def _get_event_text(self, event: Dict) -> str:
        """Get the text from an event (handles both v2.0 and v2.1 formats)."""
        if 'text' in event:
            return event['text']
        elif 'context' in event:
            ctx = event['context']
            if isinstance(ctx, dict):
                return ctx.get('sentence', ctx.get('full_window', ''))
            return str(ctx)
        return ''

    def _get_event_style(self, event: Dict) -> str:
        """Get the style from an event (handles both formats)."""
        quality = event.get('quality', {})
        if isinstance(quality, dict):
            return quality.get('style', 'narrative')
        return 'narrative'

    def select_positive(
        self,
        anchor_event: Dict,
        same_sense_events: List[Dict]
    ) -> Optional[Dict]:
        """Select a positive example for the anchor."""
        anchor_text = self._get_event_text(anchor_event)
        anchor_hash = hashlib.md5(anchor_text.lower().encode()).hexdigest()
        anchor_style = self._get_event_style(anchor_event)

        candidates = []
        for event in same_sense_events:
            event_text = self._get_event_text(event)
            ctx_hash = hashlib.md5(event_text.lower().encode()).hexdigest()

            if ctx_hash == anchor_hash:
                continue

            # Check for sufficient difference
            overlap = self.compute_lexical_overlap(anchor_text, event_text)
            if overlap < 0.8:  # Not a near-duplicate
                # Prefer different style
                event_style = self._get_event_style(event)
                style_diff = anchor_style != event_style
                candidates.append((event, overlap, style_diff))

        if not candidates:
            return None

        # Sort by style difference (prefer different), then by moderate overlap
        candidates.sort(key=lambda x: (-x[2], abs(x[1] - 0.3)))
        return candidates[0][0]

    def select_easy_negative(
        self,
        anchor_event: Dict,
        other_sense_events: Dict[str, List[Dict]]
    ) -> Optional[Tuple[Dict, float]]:
        """Select an easy negative (low overlap, different sense)."""
        anchor_text = self._get_event_text(anchor_event)

        best = None
        best_score = float('inf')

        for sense_id, events in other_sense_events.items():
            for event in events[:20]:  # Sample
                event_text = self._get_event_text(event)
                overlap = self.compute_lexical_overlap(anchor_text, event_text)

                # Easy negatives have low overlap
                if overlap < 0.5 and overlap < best_score:
                    best_score = overlap
                    best = (event, overlap)

        return best

    def select_hard_negatives_lexical(
        self,
        anchor_event: Dict,
        other_sense_events: Dict[str, List[Dict]],
        n: int = 2
    ) -> List[Tuple[Dict, float, str]]:
        """Select hard negatives based on lexical confusability."""
        anchor_text = self._get_event_text(anchor_event)

        candidates = []

        for sense_id, events in other_sense_events.items():
            for event in events:
                event_text = self._get_event_text(event)

                # Lexical overlap
                word_overlap = self.compute_lexical_overlap(anchor_text, event_text)
                ngram_overlap = self.compute_ngram_overlap(anchor_text, event_text)

                # Combined score
                score = word_overlap * 0.6 + ngram_overlap * 0.4

                if score > 0.1:  # Threshold for "confusable"
                    # Identify shared features causing confusion
                    shared = self._identify_shared_features(anchor_text, event_text)
                    candidates.append((event, score, f"shared_features:{shared}"))

        # Sort by score descending (most confusable first)
        candidates.sort(key=lambda x: -x[1])
        return candidates[:n]

    def select_hard_negatives_embedding(
        self,
        anchor_event: Dict,
        other_sense_events: Dict[str, List[Dict]],
        n: int = 2
    ) -> List[Tuple[Dict, float, str]]:
        """Select hard negatives based on embedding similarity."""
        if not self.embedder:
            return []

        anchor_text = self._get_event_text(anchor_event)

        candidates = []

        for sense_id, events in other_sense_events.items():
            for event in events:
                event_text = self._get_event_text(event)
                sim = self.compute_embedding_similarity(anchor_text, event_text)

                if sim > 0.7:  # Threshold for "embedding confusable"
                    candidates.append((event, sim, "embedding_nearest_neighbor"))

        candidates.sort(key=lambda x: -x[1])
        return candidates[:n]

    def _identify_shared_features(self, ctx1: str, ctx2: str) -> str:
        """Identify what features are shared between contexts."""
        words1 = set(ctx1.lower().split())
        words2 = set(ctx2.lower().split())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'on', 'it', 'that'}

        shared = (words1 & words2) - stopwords
        return ','.join(list(shared)[:5])

    def compute_difficulty(self, event: Dict) -> float:
        """Compute difficulty score for an event (v2.1 format)."""
        quality = event.get('quality', {})

        # Factors that increase difficulty
        difficulty = 0.5

        # Shorter contexts are harder
        text = self._get_event_text(event)
        ctx_len = len(text.split())
        if ctx_len < 20:
            difficulty += 0.1
        elif ctx_len > 35:
            difficulty -= 0.1

        # Fewer cue tokens = harder
        n_cues = len(event.get('cue_tokens', []))
        if n_cues <= 2:
            difficulty += 0.15
        elif n_cues >= 5:
            difficulty -= 0.1

        # Lower cue strength = harder
        cue_strength = quality.get('cue_strength', 0.5)
        if cue_strength < 0.3:
            difficulty += 0.1
        elif cue_strength > 0.7:
            difficulty -= 0.1

        # Higher ambiguity risk = harder
        ambiguity_risk = quality.get('ambiguity_risk', 0.0)
        difficulty += ambiguity_risk * 0.2

        # Boilerplate = easier (but we filter these)
        if quality.get('boilerplate_risk', 0.0) > 0.3:
            difficulty -= 0.2

        return max(0.1, min(0.9, difficulty))

    def create_bundle_item(
        self,
        event: Dict,
        role: str,
        negative_type: str = "",
        lexical_overlap: float = 0.0,
        embedding_sim: float = 0.0,
        confusion_reason: str = ""
    ) -> BundleItem:
        """Create a BundleItem from a raw event (v2.1 format)."""
        difficulty = self.compute_difficulty(event)

        # Determine hardness
        if difficulty < 0.35:
            hardness = "easy"
        elif difficulty < 0.65:
            hardness = "medium"
        else:
            hardness = "hard"

        # Get event ID (handle both formats)
        event_id = event.get('id', event.get('event_id', str(uuid.uuid4())))

        # Get text and span
        text = self._get_event_text(event)
        span = event.get('span', {'start': 0, 'end': 0, 'surface': ''})
        if isinstance(span, list):
            span = {'start': span[0], 'end': span[1], 'surface': text[span[0]:span[1]] if len(span) >= 2 else ''}

        # Get context window
        context_window = event.get('context_window', {'left': '', 'right': ''})
        if isinstance(context_window, str):
            context_window = {'left': '', 'right': context_window}

        # Get cue tokens and types
        cue_tokens = event.get('cue_tokens', [])
        cue_types = event.get('cue_type', [])

        # Fallback for old format
        if not cue_tokens and 'signals' in event:
            cue_tokens = event['signals'].get('disambiguators', [])
            cue_types = ['context'] * len(cue_tokens)

        return BundleItem(
            item_id=f"{role}_{event_id[:8]}",
            role=role,
            sense_id=event['sense_id'],
            text=text,
            span=span,
            context_window=context_window,
            cue_tokens=cue_tokens,
            cue_type=cue_types,
            source_event_id=event_id,
            difficulty=difficulty,
            hardness=hardness,
            style=self._get_event_style(event),
            negative_type=negative_type,
            lexical_overlap=lexical_overlap,
            embedding_similarity=embedding_sim,
            confusion_reason=confusion_reason
        )

    def build_bundle(
        self,
        lemma: str,
        anchor_event: Dict,
        positive_event: Dict,
        easy_neg: Tuple[Dict, float],
        hard_negs_lex: List[Tuple[Dict, float, str]],
        hard_negs_emb: List[Tuple[Dict, float, str]]
    ) -> Bundle:
        """Build a complete bundle."""

        bundle_id = str(uuid.uuid4())[:12]

        # Get sense catalog
        senses = self.lemma_senses.get(lemma, [])
        sense_catalog = [
            {
                "sense_id": s.get('sense_id', ''),
                "label": s.get('label', ''),
                "gloss": s.get('gloss', ''),
                "cues": s.get('cues', {}).get('keywords', [])[:5],
                "anti_cues": s.get('anti_cues', {}).get('keywords', [])[:5]
            }
            for s in senses
        ]

        # Create items
        items = []

        # Anchor
        anchor_item = self.create_bundle_item(anchor_event, "anchor")
        items.append(anchor_item)

        # Positive
        pos_item = self.create_bundle_item(positive_event, "positive")
        items.append(pos_item)

        # Easy negative
        if easy_neg:
            neg_event, overlap = easy_neg
            neg_item = self.create_bundle_item(
                neg_event, "negative",
                negative_type="easy",
                lexical_overlap=overlap
            )
            items.append(neg_item)

        # Hard negatives (lexical)
        for neg_event, overlap, reason in hard_negs_lex:
            neg_item = self.create_bundle_item(
                neg_event, "hard_negative",
                negative_type="hard_lexical",
                lexical_overlap=overlap,
                confusion_reason=reason
            )
            items.append(neg_item)

        # Hard negatives (embedding)
        for neg_event, sim, reason in hard_negs_emb:
            neg_item = self.create_bundle_item(
                neg_event, "hard_negative",
                negative_type="hard_embedding",
                embedding_similarity=sim,
                confusion_reason=reason
            )
            items.append(neg_item)

        # Build pairings
        pairings = {
            "anchor_item_id": anchor_item.item_id,
            "positives": [pos_item.item_id],
            "negatives": {
                "easy": [i.item_id for i in items if i.negative_type == "easy"],
                "medium": [],
                "hard_lexical": [i.item_id for i in items if i.negative_type == "hard_lexical"],
                "hard_embedding": [i.item_id for i in items if i.negative_type == "hard_embedding"]
            }
        }

        return Bundle(
            bundle_id=bundle_id,
            word={"lemma": lemma, "pos": anchor_event.get('pos', 'noun'), "language": "en"},
            sense_catalog=sense_catalog,
            items=items,
            pairings=pairings,
            contrastive_targets={
                "margins": {
                    "positive": 0.05,
                    "easy_negative": 0.20,
                    "medium_negative": 0.10,
                    "hard_negative": 0.03
                }
            },
            provenance={
                "generator": "bundle_generator_v2.1",
                "generation_timestamp": datetime.utcnow().isoformat() + "Z",
                "sense_inventory_version": "polysemous_senses_v1",
                "quality_score": self._compute_bundle_quality(items)
            }
        )

    def _compute_bundle_quality(self, items: List[BundleItem]) -> float:
        """Compute overall quality score for a bundle."""
        if not items:
            return 0.0

        # Factors
        n_items = len(items)
        n_hard_neg = len([i for i in items if 'hard' in i.negative_type])
        styles = set(i.style for i in items)

        score = 0.5

        # More items = better
        score += min(0.2, n_items * 0.03)

        # More hard negatives = better
        score += min(0.15, n_hard_neg * 0.05)

        # Style diversity = better
        score += min(0.15, len(styles) * 0.05)

        return round(score, 2)

    def passes_bundle_quality_gate(self, bundle: Bundle) -> bool:
        """Check if bundle passes quality gate 2."""

        items = bundle.items
        if len(items) < 3:  # Need at least anchor + positive + 1 negative
            return False

        # Check anchor/positive same sense
        anchor = next((i for i in items if i.role == "anchor"), None)
        positive = next((i for i in items if i.role == "positive"), None)

        if not anchor or not positive:
            return False

        if anchor.sense_id != positive.sense_id:
            return False

        # Check positive is sufficiently different from anchor
        if anchor.source_event_id == positive.source_event_id:
            return False

        # Check at least one negative (easy or hard)
        all_negs = [i for i in items if i.role in ("negative", "hard_negative")]
        if not all_negs:
            return False

        # Check negatives have different sense
        for neg in all_negs:
            if neg.sense_id == anchor.sense_id:
                return False

        return True

    def generate_bundles(
        self,
        events_path: Path,
        max_bundles_per_lemma: int = 50
    ) -> List[Bundle]:
        """Generate bundles from raw events."""

        events_by_sense = self.load_raw_events(events_path)

        bundles = []

        for lemma, sense_events in events_by_sense.items():
            print(f"\nProcessing '{lemma}'...")

            sense_ids = list(sense_events.keys())
            if len(sense_ids) < 2:
                print(f"  Skipping - only {len(sense_ids)} sense(s)")
                continue

            bundle_count = 0

            # For each sense, create bundles
            for anchor_sense_id, anchor_events in sense_events.items():
                if bundle_count >= max_bundles_per_lemma:
                    break

                # Other senses for negatives
                other_senses = {
                    sid: evts for sid, evts in sense_events.items()
                    if sid != anchor_sense_id
                }

                for anchor_event in anchor_events:
                    if bundle_count >= max_bundles_per_lemma:
                        break

                    # Select positive
                    positive = self.select_positive(anchor_event, anchor_events)
                    if not positive:
                        continue

                    # Select negatives
                    easy_neg = self.select_easy_negative(anchor_event, other_senses)
                    hard_lex = self.select_hard_negatives_lexical(anchor_event, other_senses, n=2)
                    hard_emb = self.select_hard_negatives_embedding(anchor_event, other_senses, n=1)

                    # Need at least easy neg or hard neg
                    if not easy_neg and not hard_lex and not hard_emb:
                        continue

                    # Build bundle
                    bundle = self.build_bundle(
                        lemma, anchor_event, positive,
                        easy_neg, hard_lex, hard_emb
                    )

                    # Quality check
                    if self.passes_bundle_quality_gate(bundle):
                        bundles.append(bundle)
                        bundle_count += 1

            print(f"  Generated {bundle_count} bundles for '{lemma}'")

        return bundles

    def save_bundles(self, bundles: List[Bundle], filename: str = "bundles_v2.jsonl") -> Path:
        """Save bundles to JSONL file."""
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            for bundle in bundles:
                f.write(json.dumps(bundle.to_dict()) + '\n')

        print(f"\nSaved {len(bundles)} bundles to {output_path}")
        return output_path


def main():
    """Test bundle generation."""
    # Load sense inventory
    inv_path = Path("paradigm_factory/v2/sense_inventory/sense_inventory.jsonl")
    if not inv_path.exists():
        print("Sense inventory not found.")
        return

    inventory = {}
    with open(inv_path) as f:
        for line in f:
            entry = json.loads(line)
            key = f"{entry['lemma']}_{entry['pos']}"
            inventory[key] = entry

    # Load raw events
    events_path = Path("paradigm_factory/v2/raw_events/wikipedia_test.jsonl")
    if not events_path.exists():
        print("Raw events not found. Run scraper first.")
        return

    # Generate bundles
    generator = BundleGenerator(sense_inventory=inventory)
    bundles = generator.generate_bundles(events_path, max_bundles_per_lemma=20)

    # Save
    output = generator.save_bundles(bundles, "bundles_v2_test.jsonl")

    # Stats
    print("\n" + "=" * 60)
    print("  Bundle Statistics")
    print("=" * 60)
    print(f"  Total bundles: {len(bundles)}")

    if bundles:
        avg_items = sum(len(b.items) for b in bundles) / len(bundles)
        print(f"  Avg items per bundle: {avg_items:.1f}")

        styles = defaultdict(int)
        for b in bundles:
            for item in b.items:
                styles[item.style] += 1
        print(f"  Style distribution: {dict(styles)}")


if __name__ == "__main__":
    main()
