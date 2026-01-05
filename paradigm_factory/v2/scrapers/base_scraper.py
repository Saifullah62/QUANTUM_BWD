"""
Base Scraper Framework (v2.1 Schema)
====================================

Abstract base class for sense-labeled usage event scrapers.
Each scraper implementation handles a specific data source.

Schema v2.1 compliant format.
"""

import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Generator, Set
import hashlib


@dataclass
class SpanInfo:
    """Character span of the target word."""
    start: int
    end: int
    surface: str


@dataclass
class ContextWindow:
    """Context window around the target word."""
    left: str
    right: str


@dataclass
class SourceInfo:
    """Provenance information for a scraped example."""
    url: str
    domain: str
    license: str
    rights_ok: bool = True
    robots_ok: bool = True


@dataclass
class QualityInfo:
    """Quality indicators for the example."""
    cue_strength: float = 0.0
    ambiguity_risk: float = 0.0
    toxicity_risk: float = 0.0
    boilerplate_risk: float = 0.0
    length_chars: int = 0
    style: str = "narrative"  # news, narrative, instructional, dialogue, technical, forum


@dataclass
class SplitsInfo:
    """Holdout tracking for train/test splits."""
    holdout_lemma: bool = False
    holdout_template_family: bool = False
    holdout_cue_family: bool = False


@dataclass
class RawUsageEvent:
    """A sense-labeled usage event - the atomic unit we collect (v2.1 schema)."""
    id: str
    lemma: str
    pos: str
    sense_id: str
    sense_gloss: str
    text: str
    span: SpanInfo
    context_window: ContextWindow
    cue_tokens: List[str]
    cue_type: List[str]
    topic_tags: List[str]
    source: SourceInfo
    quality: QualityInfo
    splits: SplitsInfo
    provenance_hash: str
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "lemma": self.lemma,
            "pos": self.pos,
            "sense_id": self.sense_id,
            "sense_gloss": self.sense_gloss,
            "text": self.text,
            "span": asdict(self.span),
            "context_window": asdict(self.context_window),
            "cue_tokens": self.cue_tokens,
            "cue_type": self.cue_type,
            "topic_tags": self.topic_tags,
            "source": asdict(self.source),
            "quality": asdict(self.quality),
            "splits": asdict(self.splits),
            "provenance_hash": self.provenance_hash,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RawUsageEvent":
        """Create from dictionary."""
        return cls(
            id=d["id"],
            lemma=d["lemma"],
            pos=d["pos"],
            sense_id=d["sense_id"],
            sense_gloss=d["sense_gloss"],
            text=d["text"],
            span=SpanInfo(**d["span"]),
            context_window=ContextWindow(**d["context_window"]),
            cue_tokens=d.get("cue_tokens", []),
            cue_type=d.get("cue_type", []),
            topic_tags=d.get("topic_tags", []),
            source=SourceInfo(**d["source"]),
            quality=QualityInfo(**d["quality"]),
            splits=SplitsInfo(**d.get("splits", {})),
            provenance_hash=d.get("provenance_hash", ""),
            notes=d.get("notes", "")
        )


class BaseScraper(ABC):
    """Abstract base class for scrapers."""

    # Common stopwords for filtering
    STOPWORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'also', 'now', 'or', 'and', 'but', 'if',
        'that', 'which', 'who', 'this', 'it', 'its', 'they', 'their', 'them',
        'he', 'she', 'him', 'her', 'we', 'us', 'our', 'you', 'your', 'i', 'me', 'my',
    }

    # Template patterns to detect (these are low-quality)
    TEMPLATE_PATTERNS = [
        r'^in terms of\s',
        r'^regarding\s',
        r'^speaking of\s',
        r'^with respect to\s',
        r'^as for\s',
        r'^concerning\s',
        r'^when it comes to\s',
        r'^\w+ refers to\s',
        r'^\w+ is defined as\s',
        r'^\w+ means\s',
    ]

    # Definition-like patterns
    DEFINITION_PATTERNS = [
        r'is a type of',
        r'is a kind of',
        r'refers to',
        r'is defined as',
        r'means the same as',
        r': a \w+ that',
        r'^\w+: ',
        r'\(\w+\)',  # parenthetical definitions
    ]

    # Cue type categories
    CUE_TYPE_MAPPING = {
        # Financial domain
        'money': 'finance_object', 'account': 'finance_object', 'deposit': 'finance_action',
        'loan': 'finance_object', 'interest': 'finance_concept', 'savings': 'finance_object',
        'credit': 'finance_object', 'financial': 'finance_concept', 'banking': 'finance_action',
        # Geography domain
        'river': 'geography', 'shore': 'geography', 'stream': 'geography',
        'water': 'geography', 'flow': 'geography', 'erosion': 'geography',
        # Season domain
        'season': 'temporal', 'bloom': 'nature', 'flower': 'nature',
        'warm': 'sensory', 'growth': 'nature', 'renewal': 'abstract',
        # Mechanical domain
        'coil': 'mechanical', 'bounce': 'mechanical', 'metal': 'material',
        'tension': 'mechanical', 'elastic': 'material',
        # General
        'jump': 'action', 'leap': 'action', 'run': 'action', 'walk': 'action',
    }

    def __init__(
        self,
        sense_inventory: Dict,
        output_dir: Path = None,
        min_context_tokens: int = 15,
        max_context_tokens: int = 50,
        rate_limit: float = 0.5
    ):
        self.sense_inventory = sense_inventory
        self.output_dir = output_dir or Path("paradigm_factory/v2/raw_events")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_context_tokens = min_context_tokens
        self.max_context_tokens = max_context_tokens
        self.rate_limit = rate_limit

        # Build lemma -> sense lookup
        self.lemma_senses = self._build_lemma_lookup()

        # Track seen contexts to avoid duplicates
        self.seen_contexts: Set[str] = set()

    def _build_lemma_lookup(self) -> Dict[str, List[Dict]]:
        """Build lookup from lemma to sense definitions."""
        lookup = {}
        for entry in self.sense_inventory.values():
            lemma = entry.lemma if hasattr(entry, 'lemma') else entry.get('lemma')
            senses = entry.senses if hasattr(entry, 'senses') else entry.get('senses', [])

            if lemma not in lookup:
                lookup[lemma] = []

            for sense in senses:
                if hasattr(sense, 'sense_id'):
                    lookup[lemma].append({
                        'sense_id': sense.sense_id,
                        'gloss': sense.gloss,
                        'cues': sense.cues,
                        'anti_cues': sense.anti_cues,
                        'domain': sense.domain
                    })
                else:
                    lookup[lemma].append(sense)

        return lookup

    def generate_event_id(self) -> str:
        """Generate a unique event ID."""
        return str(uuid.uuid4())

    def tokenize_simple(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return text.split()

    def find_target_span(self, text: str, lemma: str) -> Optional[SpanInfo]:
        """Find the character span of the target lemma."""
        text_lower = text.lower()
        lemma_lower = lemma.lower()

        # Find character span
        char_start = text_lower.find(lemma_lower)
        if char_start == -1:
            # Try with word boundaries
            match = re.search(rf'\b{re.escape(lemma_lower)}\b', text_lower)
            if match:
                char_start = match.start()
            else:
                return None

        char_end = char_start + len(lemma)
        surface = text[char_start:char_end]

        return SpanInfo(start=char_start, end=char_end, surface=surface)

    def extract_context_window(
        self,
        text: str,
        span: SpanInfo,
        window_tokens: int = 20
    ) -> ContextWindow:
        """Extract a context window around the target lemma."""
        left_text = text[:span.start].strip()
        right_text = text[span.end:].strip()

        # Limit to window_tokens on each side
        left_tokens = left_text.split()[-window_tokens:]
        right_tokens = right_text.split()[:window_tokens]

        return ContextWindow(
            left=' '.join(left_tokens),
            right=' '.join(right_tokens)
        )

    def extract_cue_tokens(
        self,
        context: str,
        lemma: str,
        sense_cues: Dict = None
    ) -> tuple:
        """Extract cue tokens and their types from context."""
        tokens = self.tokenize_simple(context.lower())
        lemma_lower = lemma.lower()

        # Remove stopwords and the lemma itself
        content_words = [
            t.strip('.,!?;:') for t in tokens
            if t.strip('.,!?;:') not in self.STOPWORDS
            and t.strip('.,!?;:') != lemma_lower
            and len(t.strip('.,!?;:')) > 2
        ]

        cue_tokens = []
        cue_types = []

        # Check against sense cues if provided
        if sense_cues:
            keywords = set(sense_cues.get('keywords', []))
            for word in content_words:
                if word in keywords:
                    cue_tokens.append(word)
                    cue_types.append(self.CUE_TYPE_MAPPING.get(word, 'domain_specific'))

        # Add other content words as context cues
        for word in content_words[:8]:
            if word not in cue_tokens:
                cue_tokens.append(word)
                cue_types.append(self.CUE_TYPE_MAPPING.get(word, 'context'))

        return cue_tokens[:10], cue_types[:10]

    def compute_cue_strength(
        self,
        cue_tokens: List[str],
        sense_cues: Dict
    ) -> float:
        """Compute strength of cue signal (0.0 to 1.0)."""
        if not sense_cues:
            return 0.0

        keywords = set(sense_cues.get('keywords', []))
        if not keywords:
            return 0.0

        matches = sum(1 for t in cue_tokens if t in keywords)
        return min(1.0, matches / max(1, len(keywords) * 0.5))

    def compute_ambiguity_risk(
        self,
        context: str,
        anti_cues: Dict
    ) -> float:
        """Compute risk of sense ambiguity (0.0 to 1.0)."""
        if not anti_cues:
            return 0.0

        context_lower = context.lower()
        anti_keywords = anti_cues.get('keywords', [])

        matches = sum(1 for k in anti_keywords if k in context_lower)
        return min(1.0, matches * 0.3)

    def compute_boilerplate_risk(self, text: str) -> float:
        """Compute risk of boilerplate/template content (0.0 to 1.0)."""
        text_lower = text.lower().strip()

        risk = 0.0
        for pattern in self.TEMPLATE_PATTERNS:
            if re.match(pattern, text_lower, re.IGNORECASE):
                risk += 0.3

        for pattern in self.DEFINITION_PATTERNS:
            if re.search(pattern, text_lower):
                risk += 0.2

        return min(1.0, risk)

    def compute_provenance_hash(self, text: str, url: str, span: SpanInfo) -> str:
        """Compute SHA256 hash for provenance tracking."""
        data = f"{text}|{url}|{span.start}:{span.end}"
        return hashlib.sha256(data.encode()).hexdigest()

    def detect_style(self, text: str) -> str:
        """Detect the writing style of the text."""
        text_lower = text.lower()

        # Dialogue markers
        if '"' in text or "'" in text or 'said' in text_lower or 'asked' in text_lower:
            return 'dialogue'

        # Technical markers
        if any(w in text_lower for w in ['function', 'method', 'parameter', 'algorithm', 'system', 'process']):
            return 'technical'

        # Instructional markers
        if any(w in text_lower for w in ['how to', 'step', 'first', 'then', 'finally', 'should', 'must']):
            return 'instructional'

        # Forum/informal markers
        if any(w in text_lower for w in ['lol', 'btw', 'imo', 'anyone', '?', '!']):
            return 'forum'

        # News markers
        if any(w in text_lower for w in ['announced', 'reported', 'according to', 'officials', 'yesterday']):
            return 'news'

        # Default to narrative
        return 'narrative'

    def get_topic_tags(self, sense: Dict) -> List[str]:
        """Get topic tags from sense domain."""
        domain = sense.get('domain', 'general')
        tags = [domain]

        # Add related tags
        domain_tags = {
            'finance': ['banking', 'money', 'economics'],
            'geography': ['nature', 'environment', 'physical'],
            'nature': ['environment', 'biology', 'outdoors'],
            'sports': ['athletics', 'competition', 'physical'],
            'mechanics': ['physics', 'engineering', 'machines'],
            'computing': ['technology', 'software', 'digital'],
        }

        if domain in domain_tags:
            tags.extend(domain_tags[domain][:1])  # Add one related tag

        return tags

    def context_hash(self, context: str) -> str:
        """Generate hash of context for deduplication."""
        normalized = ' '.join(context.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def passes_quality_gate(self, event: RawUsageEvent) -> bool:
        """Check if event passes quality gate."""
        q = event.quality

        # Length checks
        token_count = len(event.text.split())
        if token_count < self.min_context_tokens:
            return False
        if token_count > self.max_context_tokens:
            return False

        # Quality score checks
        if q.boilerplate_risk > 0.5:
            return False
        if q.ambiguity_risk > 0.5:
            return False
        if q.cue_strength < 0.1:
            return False  # Need at least some cue signal

        # Deduplication
        ctx_hash = self.context_hash(event.text)
        if ctx_hash in self.seen_contexts:
            return False
        self.seen_contexts.add(ctx_hash)

        return True

    @abstractmethod
    def scrape(
        self,
        lemmas: List[str],
        max_per_sense: int = 100
    ) -> Generator[RawUsageEvent, None, None]:
        """
        Scrape usage events for given lemmas.
        Must be implemented by subclasses.
        """
        pass

    def scrape_and_save(
        self,
        lemmas: List[str],
        max_per_sense: int = 100,
        output_file: str = "raw_events.jsonl"
    ) -> Path:
        """Scrape and save to JSONL file."""
        output_path = self.output_dir / output_file

        count = 0
        passed = 0

        with open(output_path, 'w') as f:
            for event in self.scrape(lemmas, max_per_sense):
                count += 1
                if self.passes_quality_gate(event):
                    f.write(json.dumps(event.to_dict()) + '\n')
                    passed += 1

                if count % 100 == 0:
                    print(f"  Processed {count} events, {passed} passed quality gate")

        print(f"\nSaved {passed}/{count} events to {output_path}")
        return output_path


class MultiSourceScraper:
    """Orchestrates multiple scrapers and merges results."""

    def __init__(self, scrapers: List[BaseScraper], output_dir: Path = None):
        self.scrapers = scrapers
        self.output_dir = output_dir or Path("paradigm_factory/v2/raw_events")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scrape_all(
        self,
        lemmas: List[str],
        max_per_sense: int = 100
    ) -> Path:
        """Run all scrapers and merge results."""
        all_events = []

        for scraper in self.scrapers:
            print(f"\nRunning {scraper.__class__.__name__}...")
            for event in scraper.scrape(lemmas, max_per_sense):
                if scraper.passes_quality_gate(event):
                    all_events.append(event)

        # Save merged results
        output_path = self.output_dir / "merged_raw_events.jsonl"
        with open(output_path, 'w') as f:
            for event in all_events:
                f.write(json.dumps(event.to_dict()) + '\n')

        print(f"\nMerged {len(all_events)} events to {output_path}")
        return output_path
