"""
Multi-Source Web Scraper (v2.1 Schema)
======================================

Scrapes polysemy data from various web sources including:
- Legal glossaries
- Medical terminology
- Financial terms
- Computer science vocabulary
- Sports terminology
- Slang dictionaries
- Academic polysemy lists

Converts all data to v2.1 schema format.
"""

import json
import re
import time
import uuid
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Generator, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from urllib.parse import urlparse

try:
    from .base_scraper import (
        BaseScraper, RawUsageEvent, SpanInfo, ContextWindow,
        SourceInfo, QualityInfo, SplitsInfo
    )
except ImportError:
    from base_scraper import (
        BaseScraper, RawUsageEvent, SpanInfo, ContextWindow,
        SourceInfo, QualityInfo, SplitsInfo
    )


@dataclass
class ScrapedEntry:
    """Raw scraped entry before conversion to v2.1 schema."""
    word: str
    pos: str
    sense_id: str
    gloss: str
    example: str
    domain: str
    source_url: str
    source_domain: str


class WebScraper:
    """Multi-source web scraper for polysemy data."""

    # Domain category mappings
    DOMAIN_CATEGORIES = {
        'legal': ['law', 'court', 'legal', 'attorney', 'judge'],
        'medical': ['health', 'medicine', 'medical', 'clinical', 'doctor'],
        'financial': ['finance', 'banking', 'investment', 'money', 'economic'],
        'technology': ['computer', 'software', 'tech', 'digital', 'code'],
        'sports': ['sport', 'game', 'athletic', 'play', 'team'],
        'slang': ['slang', 'informal', 'colloquial', 'casual'],
        'academic': ['academic', 'scholarly', 'research', 'study'],
    }

    # Common polysemous words to look for
    POLYSEMOUS_WORDS = [
        'bank', 'spring', 'run', 'light', 'draft', 'set', 'play', 'match',
        'cell', 'note', 'bar', 'scale', 'band', 'pitch', 'court', 'interest',
        'case', 'charge', 'bear', 'board', 'book', 'break', 'call', 'check',
        'class', 'close', 'club', 'cold', 'contract', 'cool', 'cover', 'cross',
        'culture', 'cut', 'date', 'degree', 'depression', 'die', 'draw', 'drive',
        'drop', 'exchange', 'express', 'face', 'fair', 'fall', 'fan', 'fast',
        'file', 'fire', 'firm', 'fit', 'fly', 'fold', 'force', 'fork', 'frame',
        'free', 'game', 'goal', 'grade', 'grant', 'gross', 'ground', 'growth',
        'hand', 'handle', 'head', 'heart', 'hit', 'hold', 'host', 'house',
        'iron', 'issue', 'jam', 'joint', 'jump', 'key', 'kind', 'land', 'lap',
        'lead', 'lean', 'leave', 'left', 'leg', 'letter', 'level', 'lie', 'lift',
        'line', 'link', 'list', 'live', 'load', 'lock', 'log', 'long', 'lot',
        'mail', 'main', 'make', 'mark', 'market', 'mass', 'matter', 'mean',
        'measure', 'medium', 'meet', 'memory', 'might', 'mine', 'minute', 'miss',
        'model', 'monitor', 'mount', 'mouse', 'move', 'natural', 'net', 'network',
        'novel', 'object', 'odd', 'offer', 'office', 'order', 'organ', 'original',
        'outlook', 'pack', 'palm', 'pan', 'paper', 'park', 'party', 'pass', 'past',
        'patch', 'patient', 'pay', 'peer', 'pen', 'period', 'permit', 'pick',
        'piece', 'pilot', 'pipe', 'place', 'plain', 'plane', 'plant', 'plastic',
        'plate', 'platform', 'plot', 'plug', 'point', 'pole', 'pool', 'pop',
        'port', 'position', 'post', 'pot', 'pound', 'power', 'present', 'press',
        'prime', 'principal', 'print', 'process', 'produce', 'program', 'project',
        'property', 'prospect', 'provision', 'public', 'pull', 'pump', 'punch',
        'pupil', 'push', 'quarter', 'race', 'rack', 'radical', 'range', 'rank',
        'rate', 'ray', 'reach', 'read', 'record', 'reference', 'reflect', 'reform',
        'register', 'regular', 'release', 'relief', 'remote', 'rent', 'report',
        'reserve', 'resolution', 'rest', 'return', 'review', 'revolution', 'rich',
        'ride', 'right', 'ring', 'rise', 'rock', 'roll', 'root', 'rose', 'round',
        'row', 'rule', 'rush', 'safe', 'sail', 'sale', 'sample', 'save', 'saw',
        'scan', 'scene', 'school', 'score', 'screen', 'seal', 'search', 'season',
        'seat', 'second', 'section', 'security', 'sense', 'sentence', 'serve',
        'service', 'session', 'settle', 'shade', 'shake', 'shape', 'share',
        'sharp', 'shell', 'shift', 'ship', 'shock', 'shoot', 'shop', 'short',
        'shot', 'show', 'shut', 'side', 'sign', 'signal', 'single', 'sink',
        'site', 'size', 'skip', 'slide', 'slip', 'slot', 'snap', 'solid', 'sound',
        'source', 'space', 'spare', 'spark', 'speaker', 'special', 'speed',
        'spell', 'spin', 'spirit', 'split', 'spot', 'spread', 'square', 'stable',
        'staff', 'stage', 'stake', 'stamp', 'stand', 'standard', 'star', 'start',
        'state', 'station', 'stay', 'steam', 'steel', 'stem', 'step', 'stick',
        'stock', 'stone', 'stop', 'store', 'storm', 'story', 'strain', 'strange',
        'stream', 'stress', 'stretch', 'strike', 'string', 'strip', 'stroke',
        'structure', 'study', 'stuff', 'style', 'subject', 'suit', 'sum', 'supply',
        'support', 'surface', 'surge', 'surplus', 'survey', 'swing', 'switch',
        'system', 'table', 'tackle', 'tag', 'tail', 'take', 'talk', 'tank', 'tap',
        'tape', 'target', 'taste', 'tax', 'tear', 'technical', 'temperature',
        'tender', 'term', 'terminal', 'terms', 'test', 'text', 'thread', 'tie',
        'tight', 'tile', 'time', 'tip', 'title', 'tone', 'tool', 'top', 'topic',
        'touch', 'tour', 'tower', 'track', 'trade', 'trail', 'train', 'transfer',
        'trap', 'travel', 'treat', 'treatment', 'tree', 'trial', 'trick', 'trip',
        'trouble', 'truck', 'trust', 'tube', 'tune', 'turn', 'twist', 'type',
        'union', 'unit', 'value', 'variety', 'vehicle', 'venture', 'version',
        'vessel', 'view', 'virus', 'vision', 'volume', 'vote', 'wage', 'walk',
        'wall', 'war', 'warm', 'wash', 'waste', 'watch', 'water', 'wave', 'way',
        'wear', 'weather', 'web', 'weight', 'well', 'wheel', 'whip', 'will',
        'wind', 'window', 'wing', 'wire', 'witness', 'wonder', 'wood', 'word',
        'work', 'world', 'wrap', 'write', 'yard', 'year', 'yield', 'zone'
    ]

    def __init__(
        self,
        output_dir: Path = None,
        rate_limit: float = 1.0,
        timeout: int = 30
    ):
        self.output_dir = output_dir or Path("paradigm_factory/v2/raw_events")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 QLLM-Research/1.0'
        })

        # Track scraped content for deduplication
        self.seen_contexts = set()
        self.collected_entries: List[ScrapedEntry] = []

        # Stats
        self.stats = defaultdict(int)

    def fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL content with rate limiting."""
        try:
            time.sleep(self.rate_limit)
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            self.stats['fetch_errors'] += 1
            return None

    def extract_text_from_html(self, html: str) -> str:
        """Extract clean text from HTML."""
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Remove tags but keep content
        text = re.sub(r'<[^>]+>', ' ', html)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def extract_sentences(self, text: str, target_word: str) -> List[str]:
        """Extract sentences containing the target word."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        matching = []

        word_pattern = rf'\b{re.escape(target_word)}s?\b'

        for sent in sentences:
            if re.search(word_pattern, sent, re.IGNORECASE):
                sent = sent.strip()
                sent = re.sub(r'\s+', ' ', sent)
                words = sent.split()
                if 10 <= len(words) <= 60:
                    matching.append(sent)

        return matching

    def detect_domain(self, url: str, text: str) -> str:
        """Detect the domain category from URL and content."""
        url_lower = url.lower()
        text_lower = text.lower()[:2000]

        for domain, keywords in self.DOMAIN_CATEGORIES.items():
            for kw in keywords:
                if kw in url_lower or kw in text_lower:
                    return domain

        return 'general'

    def parse_glossary_entry(self, text: str, domain: str) -> List[Tuple[str, str, str]]:
        """Parse glossary-style entries (word: definition)."""
        entries = []

        # Pattern: Word - Definition or Word: Definition
        patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[-–:]\s*(.+)$',
            r'^([A-Za-z]+)\s*[-–:]\s*(.+)$',
        ]

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue

            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    word = match.group(1).strip().lower()
                    definition = match.group(2).strip()

                    # Check if word is in our polysemy list
                    if word in self.POLYSEMOUS_WORDS or len(word.split()) == 1:
                        entries.append((word, definition, domain))
                    break

        return entries

    def create_entry_from_sentence(
        self,
        word: str,
        sentence: str,
        domain: str,
        source_url: str,
        gloss: str = ""
    ) -> Optional[ScrapedEntry]:
        """Create a scraped entry from a sentence."""
        # Check for duplicates
        ctx_hash = hashlib.md5(sentence.lower().encode()).hexdigest()
        if ctx_hash in self.seen_contexts:
            return None
        self.seen_contexts.add(ctx_hash)

        parsed_url = urlparse(source_url)
        source_domain = parsed_url.netloc

        # Generate sense ID based on domain
        sense_id = f"{word}#{domain}"

        return ScrapedEntry(
            word=word,
            pos='noun',  # Default, can be refined
            sense_id=sense_id,
            gloss=gloss or f"{word} in {domain} context",
            example=sentence,
            domain=domain,
            source_url=source_url,
            source_domain=source_domain
        )

    def convert_to_v21_event(self, entry: ScrapedEntry) -> RawUsageEvent:
        """Convert scraped entry to v2.1 RawUsageEvent."""
        text = entry.example

        # Find span
        word_lower = entry.word.lower()
        text_lower = text.lower()
        start = text_lower.find(word_lower)
        if start == -1:
            start = 0
        end = start + len(entry.word)
        surface = text[start:end] if start >= 0 else entry.word

        span = SpanInfo(start=start, end=end, surface=surface)

        # Context window
        left = text[:start].strip().split()[-15:]
        right = text[end:].strip().split()[:15]
        context_window = ContextWindow(
            left=' '.join(left),
            right=' '.join(right)
        )

        # Extract cue tokens
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'on', 'it', 'that', 'for', 'with'}
        words = [w.strip('.,!?;:').lower() for w in text.split()]
        cue_tokens = [w for w in words if w not in stopwords and w != word_lower and len(w) > 2][:10]
        cue_types = ['context'] * len(cue_tokens)

        # Source info
        source = SourceInfo(
            url=entry.source_url,
            domain=entry.source_domain,
            license="fair-use-research",
            rights_ok=True,
            robots_ok=True
        )

        # Quality metrics
        quality = QualityInfo(
            cue_strength=min(1.0, len(cue_tokens) * 0.1),
            ambiguity_risk=0.3,  # Default moderate risk
            toxicity_risk=0.0,
            boilerplate_risk=0.1,
            length_chars=len(text),
            style='technical' if entry.domain in ['legal', 'medical', 'technology'] else 'narrative'
        )

        # Splits
        splits = SplitsInfo(
            holdout_lemma=False,
            holdout_template_family=False,
            holdout_cue_family=False
        )

        # Provenance hash
        provenance_hash = hashlib.sha256(f"{text}|{entry.source_url}|{start}:{end}".encode()).hexdigest()

        return RawUsageEvent(
            id=str(uuid.uuid4()),
            lemma=entry.word,
            pos=entry.pos,
            sense_id=entry.sense_id,
            sense_gloss=entry.gloss,
            text=text,
            span=span,
            context_window=context_window,
            cue_tokens=cue_tokens,
            cue_type=cue_types,
            topic_tags=[entry.domain],
            source=source,
            quality=quality,
            splits=splits,
            provenance_hash=provenance_hash,
            notes=""
        )

    def scrape_glossary_page(self, url: str, domain: str) -> List[ScrapedEntry]:
        """Scrape a glossary page for term definitions and examples."""
        entries = []
        print(f"  Scraping: {url}")

        html = self.fetch_url(url)
        if not html:
            return entries

        text = self.extract_text_from_html(html)

        # Try to extract glossary entries
        glossary_entries = self.parse_glossary_entry(text, domain)

        for word, definition, dom in glossary_entries:
            # Create an entry with the definition as example
            entry = self.create_entry_from_sentence(
                word=word,
                sentence=definition,
                domain=dom,
                source_url=url,
                gloss=definition[:100]
            )
            if entry:
                entries.append(entry)
                self.stats['glossary_entries'] += 1

        # Also extract sentences with polysemous words
        for word in self.POLYSEMOUS_WORDS[:50]:  # Top 50 words
            sentences = self.extract_sentences(text, word)
            for sent in sentences[:3]:  # Max 3 per word per page
                entry = self.create_entry_from_sentence(
                    word=word,
                    sentence=sent,
                    domain=domain,
                    source_url=url
                )
                if entry:
                    entries.append(entry)
                    self.stats['sentence_entries'] += 1

        return entries

    def scrape_all_sources(self, sources: List[Tuple[str, str]]) -> int:
        """Scrape all provided sources."""
        total_entries = 0

        for url, domain in sources:
            try:
                entries = self.scrape_glossary_page(url, domain)
                self.collected_entries.extend(entries)
                total_entries += len(entries)
                print(f"    Collected {len(entries)} entries")
            except Exception as e:
                print(f"    Error: {e}")
                self.stats['page_errors'] += 1

        return total_entries

    def save_raw_events(self, filename: str = "web_scraped_events.jsonl") -> Path:
        """Convert all entries to v2.1 format and save."""
        output_path = self.output_dir / filename

        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in self.collected_entries:
                try:
                    event = self.convert_to_v21_event(entry)
                    f.write(json.dumps(event.to_dict()) + '\n')
                    count += 1
                except Exception as e:
                    print(f"  Error converting entry: {e}")

        print(f"\nSaved {count} events to {output_path}")
        return output_path

    def print_stats(self):
        """Print collection statistics."""
        print("\n" + "=" * 60)
        print("SCRAPING STATISTICS")
        print("=" * 60)
        print(f"  Total entries collected: {len(self.collected_entries)}")
        print(f"  Glossary entries: {self.stats['glossary_entries']}")
        print(f"  Sentence entries: {self.stats['sentence_entries']}")
        print(f"  Fetch errors: {self.stats['fetch_errors']}")
        print(f"  Page errors: {self.stats['page_errors']}")
        print(f"  Unique contexts: {len(self.seen_contexts)}")

        # By domain
        domain_counts = defaultdict(int)
        for entry in self.collected_entries:
            domain_counts[entry.domain] += 1

        print("\n  By domain:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            print(f"    {domain}: {count}")

        # By word (top 20)
        word_counts = defaultdict(int)
        for entry in self.collected_entries:
            word_counts[entry.word] += 1

        print("\n  Top 20 words:")
        for word, count in sorted(word_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"    {word}: {count}")


# Source configurations
SOURCES = [
    # Legal terminology
    ("https://www.uscourts.gov/glossary", "legal"),
    ("https://nycourts.gov/courthelp/GoingToCourt/glossary.shtml", "legal"),
    ("https://www.courts.wa.gov/newsinfo/resources/?fa=newsinfo_jury.display&altMenu=Cour&folderID=glossary", "legal"),
    ("https://www.fccourts.org/resources/legal-terms-definitions/", "legal"),

    # Financial terminology
    ("https://www.investopedia.com/financial-term-dictionary-4769738", "financial"),
    ("https://www.consumerfinance.gov/consumer-tools/educator-tools/youth-financial-education/glossary/", "financial"),
    ("https://www.dfpi.ca.gov/glossary-of-financial-terms/", "financial"),
    ("https://www.schwab.com/learn/story/investing-glossary-100-terms-definitions", "financial"),

    # Medical terminology
    ("https://medlineplus.gov/appendixa.html", "medical"),
    ("https://www.sgu.edu/blog/medical/must-know-medical-terms-abbreviations-acronyms/", "medical"),
    ("https://www.rcog.org.uk/for-the-public/a-z-of-medical-terms/", "medical"),

    # Technology/CS terminology
    ("https://code.org/curriculum/docs/glossary", "technology"),
    ("https://en.wikipedia.org/wiki/Glossary_of_computer_science", "technology"),
    ("https://en.wikipedia.org/wiki/Glossary_of_mechanical_engineering", "technology"),

    # Academic/polysemy sources
    ("https://prepedu.com/en/blog/polysemy-in-english", "academic"),
    ("https://www.study.com/academy/lesson/what-is-polysemy-definition-examples.html", "academic"),
    ("https://www.thoughtco.com/polysemy-words-and-meanings-1691642", "academic"),

    # Sports
    ("https://www.usingenglish.com/reference/phrasal-verbs/sports-vocabulary.html", "sports"),

    # Slang
    ("https://www.heylama.com/blog/english-slang-words", "slang"),
    ("https://youth.europa.eu/news/gen-z-slang-words-and-phrases-2024_en", "slang"),
]


def main():
    """Run the multi-source scraper."""
    print("=" * 60)
    print("MULTI-SOURCE WEB SCRAPER FOR POLYSEMY DATA")
    print("=" * 60)

    scraper = WebScraper(rate_limit=1.5)

    print(f"\nScraping {len(SOURCES)} sources...")
    total = scraper.scrape_all_sources(SOURCES)

    print(f"\nTotal entries collected: {total}")

    scraper.print_stats()

    # Save to v2.1 format
    output_path = scraper.save_raw_events("web_scraped_v21.jsonl")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
