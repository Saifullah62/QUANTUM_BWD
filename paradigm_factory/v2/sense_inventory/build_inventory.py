"""
Sense Inventory Builder
=======================

Builds a canonical sense inventory from WordNet with cue/anti-cue bags.
Outputs sense_inventory.jsonl with one entry per lemma.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, asdict
import hashlib

try:
    from nltk.corpus import wordnet as wn
    from nltk.corpus import wordnet_ic
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('wordnet_ic', quiet=True)
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    print("Warning: NLTK not available. Install with: pip install nltk")


@dataclass
class SenseCues:
    """Cue and anti-cue bags for a sense."""
    keywords: List[str]
    collocates: List[str]
    prepositions: List[str]
    argument_patterns: List[str]


@dataclass
class SenseEntry:
    """A single sense for a lemma."""
    sense_id: str
    label: str
    gloss: str
    definition_source: str
    cues: Dict
    anti_cues: Dict
    domain: str
    frequency_rank: int
    synset_id: Optional[str] = None
    examples: List[str] = None
    hypernyms: List[str] = None
    hyponyms: List[str] = None


@dataclass
class LemmaInventory:
    """Complete sense inventory for a lemma."""
    lemma: str
    pos: str
    senses: List[SenseEntry]
    confusable_pairs: List[Dict]


class SenseInventoryBuilder:
    """Builds sense inventory from WordNet."""

    # Common prepositions by semantic role
    PREP_PATTERNS = {
        'location': ['at', 'in', 'on', 'near', 'by', 'beside'],
        'direction': ['to', 'from', 'into', 'toward', 'through'],
        'instrument': ['with', 'using', 'by'],
        'recipient': ['to', 'for'],
        'source': ['from', 'of'],
        'temporal': ['at', 'during', 'after', 'before'],
    }

    # Domain keywords for classification
    DOMAIN_KEYWORDS = {
        'finance': ['money', 'bank', 'loan', 'credit', 'invest', 'account', 'deposit', 'interest', 'fund'],
        'geography': ['river', 'mountain', 'land', 'water', 'shore', 'coast', 'terrain', 'slope'],
        'technology': ['computer', 'software', 'data', 'system', 'digital', 'network', 'device'],
        'biology': ['cell', 'organism', 'species', 'gene', 'protein', 'tissue', 'organ'],
        'law': ['court', 'legal', 'judge', 'law', 'contract', 'statute', 'right'],
        'music': ['note', 'melody', 'rhythm', 'instrument', 'song', 'chord', 'tune'],
        'sports': ['game', 'play', 'team', 'score', 'match', 'win', 'player'],
        'food': ['eat', 'cook', 'taste', 'meal', 'dish', 'ingredient', 'recipe'],
        'medicine': ['doctor', 'patient', 'treatment', 'disease', 'symptom', 'medicine'],
        'construction': ['build', 'structure', 'material', 'foundation', 'wall', 'floor'],
    }

    # High-value polysemous lemmas to prioritize
    PRIORITY_LEMMAS = [
        # Noun/verb ambiguity
        'bank', 'run', 'play', 'set', 'light', 'spring', 'fall', 'match', 'break', 'change',
        'charge', 'check', 'clear', 'close', 'contract', 'cool', 'cover', 'cross', 'cut', 'deal',
        'draft', 'draw', 'drive', 'drop', 'express', 'fair', 'fast', 'file', 'fine', 'fire',
        'fit', 'flat', 'fly', 'fold', 'form', 'frame', 'free', 'ground', 'hand', 'head',
        'hold', 'iron', 'jam', 'judge', 'key', 'kind', 'lead', 'lean', 'leave', 'left',
        'level', 'lie', 'lift', 'line', 'live', 'lock', 'log', 'long', 'mark', 'mine',
        'miss', 'model', 'move', 'note', 'object', 'order', 'pack', 'park', 'pass', 'patch',
        'pick', 'pitch', 'plant', 'plot', 'point', 'pool', 'pop', 'post', 'pound', 'present',
        'press', 'produce', 'project', 'prompt', 'pump', 'punch', 'rank', 'rate', 'record', 'report',
        'rest', 'ring', 'rock', 'roll', 'root', 'round', 'row', 'rule', 'saw', 'scale',
        'seal', 'season', 'second', 'sentence', 'serve', 'shade', 'shape', 'share', 'sheet', 'shift',
        'ship', 'shoot', 'shop', 'show', 'sign', 'sink', 'skip', 'slip', 'snap', 'sound',
        'space', 'spare', 'spell', 'split', 'spot', 'spread', 'square', 'staff', 'stage', 'stake',
        'stamp', 'stand', 'star', 'start', 'state', 'stay', 'step', 'stick', 'stock', 'stop',
        'store', 'storm', 'story', 'strain', 'stream', 'stress', 'stretch', 'strike', 'strip', 'stroke',
        'study', 'subject', 'suit', 'supply', 'support', 'survey', 'switch', 'table', 'take', 'tap',
        'target', 'taste', 'tear', 'term', 'test', 'tie', 'tip', 'tire', 'top', 'touch',
        'track', 'trade', 'train', 'transfer', 'transport', 'trap', 'treat', 'trial', 'trick', 'trip',
        'trust', 'turn', 'type', 'view', 'voice', 'walk', 'watch', 'wave', 'wear', 'wind',
        'work', 'wrap', 'yard',
    ]

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("paradigm_factory/v2/sense_inventory")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inventory = {}

    def extract_keywords_from_gloss(self, gloss: str) -> List[str]:
        """Extract potential cue keywords from a gloss."""
        # Remove common words and extract content words
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'also', 'now', 'or', 'and', 'but', 'if',
            'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'it',
            'its', 'something', 'someone', 'anything', 'anyone', 'nothing',
            'one', 'two', 'first', 'second', 'used', 'make', 'made', 'thing',
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-z]+\b', gloss.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        return list(set(keywords))[:10]  # Top 10 unique

    def infer_domain(self, gloss: str, keywords: List[str]) -> str:
        """Infer domain from gloss and keywords."""
        gloss_lower = gloss.lower()
        all_words = set(keywords + gloss_lower.split())

        best_domain = 'general'
        best_score = 0

        for domain, domain_kw in self.DOMAIN_KEYWORDS.items():
            score = len(set(domain_kw) & all_words)
            if score > best_score:
                best_score = score
                best_domain = domain

        return best_domain

    def get_collocates_from_examples(self, examples: List[str], lemma: str) -> List[str]:
        """Extract collocates from example sentences."""
        collocates = []
        lemma_lower = lemma.lower()

        for ex in examples:
            ex_lower = ex.lower()
            # Find 2-3 word windows around the lemma
            words = ex_lower.split()
            for i, w in enumerate(words):
                if lemma_lower in w:
                    # Get surrounding context
                    start = max(0, i - 2)
                    end = min(len(words), i + 3)
                    window = words[start:end]
                    colloc = ' '.join(window)
                    if colloc not in collocates:
                        collocates.append(colloc)

        return collocates[:5]

    def build_sense_entry(
        self,
        lemma: str,
        synset,
        rank: int,
        all_senses: List
    ) -> SenseEntry:
        """Build a SenseEntry from a WordNet synset."""
        # Create sense ID
        sense_label = synset.name().split('.')[0]
        # Make label more readable
        label = sense_label.replace('_', ' ')
        sense_id = f"{lemma}#{sense_label}"

        # Get gloss and examples
        gloss = synset.definition()
        examples = synset.examples()

        # Extract keywords from gloss
        keywords = self.extract_keywords_from_gloss(gloss)

        # Get hypernyms/hyponyms for additional keywords
        hypernyms = [h.name().split('.')[0] for h in synset.hypernyms()[:3]]
        hyponyms = [h.name().split('.')[0] for h in synset.hyponyms()[:3]]

        # Extend keywords with hypernym/hyponym info
        all_keywords = list(set(keywords + hypernyms + hyponyms))

        # Get collocates from examples
        collocates = self.get_collocates_from_examples(examples, lemma)

        # Infer domain
        domain = self.infer_domain(gloss, all_keywords)

        # Build anti-cues from OTHER senses
        anti_cue_keywords = []
        for other in all_senses:
            if other.name() != synset.name():
                other_kw = self.extract_keywords_from_gloss(other.definition())
                # Anti-cues are keywords unique to other senses
                for kw in other_kw:
                    if kw not in all_keywords and kw not in anti_cue_keywords:
                        anti_cue_keywords.append(kw)

        return SenseEntry(
            sense_id=sense_id,
            label=label,
            gloss=gloss,
            definition_source="wordnet",
            cues={
                "keywords": all_keywords[:10],
                "collocates": collocates,
                "prepositions": [],  # Will be filled by usage analysis
                "argument_patterns": []
            },
            anti_cues={
                "keywords": anti_cue_keywords[:10],
                "collocates": []
            },
            domain=domain,
            frequency_rank=rank,
            synset_id=synset.name(),
            examples=examples[:3],
            hypernyms=hypernyms,
            hyponyms=hyponyms
        )

    def compute_confusability(
        self,
        sense_a: SenseEntry,
        sense_b: SenseEntry
    ) -> Dict:
        """Compute confusability between two senses."""
        # Keyword overlap
        kw_a = set(sense_a.cues.get('keywords', []))
        kw_b = set(sense_b.cues.get('keywords', []))
        shared_kw = kw_a & kw_b
        total_kw = kw_a | kw_b

        kw_overlap = len(shared_kw) / len(total_kw) if total_kw else 0

        # Collocate overlap
        col_a = set(sense_a.cues.get('collocates', []))
        col_b = set(sense_b.cues.get('collocates', []))
        shared_col = col_a & col_b

        # Domain similarity
        domain_same = sense_a.domain == sense_b.domain

        # Overall confusability score
        score = kw_overlap * 0.5 + (0.3 if domain_same else 0) + (0.2 * len(shared_col) / 5)

        return {
            "sense_a": sense_a.sense_id,
            "sense_b": sense_b.sense_id,
            "confusability_score": round(score, 3),
            "shared_keywords": list(shared_kw),
            "shared_collocates": list(shared_col),
            "same_domain": domain_same,
            "differentiating_features": self._get_differentiators(sense_a, sense_b)
        }

    def _get_differentiators(self, sense_a: SenseEntry, sense_b: SenseEntry) -> List[str]:
        """Identify what differentiates two senses."""
        diffs = []

        # Different domains
        if sense_a.domain != sense_b.domain:
            diffs.append(f"domain:{sense_a.domain}_vs_{sense_b.domain}")

        # Unique keywords
        kw_a = set(sense_a.cues.get('keywords', []))
        kw_b = set(sense_b.cues.get('keywords', []))
        unique_a = kw_a - kw_b
        unique_b = kw_b - kw_a
        if unique_a:
            diffs.append(f"unique_to_a:{list(unique_a)[:3]}")
        if unique_b:
            diffs.append(f"unique_to_b:{list(unique_b)[:3]}")

        return diffs

    def build_lemma_inventory(self, lemma: str, pos: str = 'n') -> Optional[LemmaInventory]:
        """Build complete inventory for a lemma."""
        if not WORDNET_AVAILABLE:
            return None

        # Get all synsets for this lemma+POS
        pos_map = {'n': wn.NOUN, 'v': wn.VERB, 'a': wn.ADJ, 'r': wn.ADV}
        wn_pos = pos_map.get(pos.lower(), wn.NOUN)

        synsets = wn.synsets(lemma, pos=wn_pos)

        if not synsets:
            return None

        # Build sense entries
        senses = []
        for rank, syn in enumerate(synsets, 1):
            entry = self.build_sense_entry(lemma, syn, rank, synsets)
            senses.append(entry)

        # Compute confusable pairs (only for senses with overlap)
        confusable = []
        for i, sa in enumerate(senses):
            for sb in senses[i+1:]:
                pair = self.compute_confusability(sa, sb)
                if pair["confusability_score"] > 0.1:  # Only include if somewhat confusable
                    confusable.append(pair)

        # Sort by confusability (most confusable first)
        confusable.sort(key=lambda x: -x["confusability_score"])

        pos_full = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 'r': 'ADV'}.get(pos.lower(), 'NOUN')

        return LemmaInventory(
            lemma=lemma,
            pos=pos_full,
            senses=senses,
            confusable_pairs=confusable[:5]  # Top 5 confusable pairs
        )

    def build_full_inventory(
        self,
        lemmas: List[str] = None,
        pos_list: List[str] = None,
        min_senses: int = 2
    ) -> Dict[str, LemmaInventory]:
        """Build inventory for multiple lemmas."""
        if lemmas is None:
            lemmas = self.PRIORITY_LEMMAS
        if pos_list is None:
            pos_list = ['n', 'v']  # Nouns and verbs by default

        inventory = {}

        for lemma in lemmas:
            for pos in pos_list:
                inv = self.build_lemma_inventory(lemma, pos)
                if inv and len(inv.senses) >= min_senses:
                    key = f"{lemma}_{inv.pos}"
                    inventory[key] = inv
                    print(f"  {key}: {len(inv.senses)} senses, {len(inv.confusable_pairs)} confusable pairs")

        self.inventory = inventory
        return inventory

    def save_inventory(self, filename: str = "sense_inventory.jsonl"):
        """Save inventory to JSONL file."""
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            for key, inv in self.inventory.items():
                # Convert to dict, handling nested dataclasses
                record = {
                    "lemma": inv.lemma,
                    "pos": inv.pos,
                    "senses": [asdict(s) for s in inv.senses],
                    "confusable_pairs": inv.confusable_pairs
                }
                f.write(json.dumps(record) + '\n')

        print(f"\nSaved {len(self.inventory)} lemma inventories to {output_path}")
        return output_path

    def get_stats(self) -> Dict:
        """Get statistics about the inventory."""
        total_senses = sum(len(inv.senses) for inv in self.inventory.values())
        total_confusable = sum(len(inv.confusable_pairs) for inv in self.inventory.values())

        domains = defaultdict(int)
        for inv in self.inventory.values():
            for sense in inv.senses:
                domains[sense.domain] += 1

        return {
            "total_lemmas": len(self.inventory),
            "total_senses": total_senses,
            "total_confusable_pairs": total_confusable,
            "avg_senses_per_lemma": total_senses / len(self.inventory) if self.inventory else 0,
            "domain_distribution": dict(domains)
        }


def main():
    """Build and save the sense inventory."""
    print("=" * 60)
    print("  Building Sense Inventory from WordNet")
    print("=" * 60)

    builder = SenseInventoryBuilder()

    # Build for priority lemmas
    print("\nProcessing priority lemmas...")
    inventory = builder.build_full_inventory(
        lemmas=builder.PRIORITY_LEMMAS[:100],  # Start with first 100
        pos_list=['n', 'v'],
        min_senses=2
    )

    # Save
    output_path = builder.save_inventory()

    # Print stats
    stats = builder.get_stats()
    print("\n" + "=" * 60)
    print("  Inventory Statistics")
    print("=" * 60)
    print(f"  Total lemmas: {stats['total_lemmas']}")
    print(f"  Total senses: {stats['total_senses']}")
    print(f"  Confusable pairs: {stats['total_confusable_pairs']}")
    print(f"  Avg senses/lemma: {stats['avg_senses_per_lemma']:.1f}")
    print("\n  Domain distribution:")
    for domain, count in sorted(stats['domain_distribution'].items(), key=lambda x: -x[1]):
        print(f"    {domain}: {count}")


if __name__ == "__main__":
    main()
