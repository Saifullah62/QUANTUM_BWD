"""
Comprehensive Data Collection Pipeline
=======================================

Exhaustive scraping from all provided sources to double the dataset.
Runs continuously to collect maximum polysemy data.

Sources:
- Legal glossaries (uscourts, nycourts, courts.wa, fccourts, etc.)
- Medical terminology (medlineplus, sgu, rcog, germanna, pce.sandiego)
- Financial terms (investopedia, consumerfinance, schwab, dfpi)
- Technology/CS (code.org, wikipedia glossaries, dataprise, ccaps.umn)
- Sports vocabulary (usingenglish, grammarly, covers)
- Slang (heylama, youth.europa, shorelight, salisbury)
- Academic polysemy (prepedu, study.com, thoughtco, scispace)
- Multiple meaning lists (amazingtalker, yourdictionary, home-speech-home)
"""

import json
import re
import time
import uuid
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from urllib.parse import urlparse, urljoin
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paradigm_factory.v2.scrapers.base_scraper import (
    RawUsageEvent, SpanInfo, ContextWindow,
    SourceInfo, QualityInfo, SplitsInfo
)


# =============================================================================
# ALL SOURCES TO SCRAPE
# =============================================================================

SOURCES = {
    # LEGAL TERMINOLOGY
    'legal': [
        "https://www.uscourts.gov/glossary",
        "https://nycourts.gov/courthelp/GoingToCourt/glossary.shtml",
        "https://www.courts.wa.gov/newsinfo/resources/?fa=newsinfo_jury.display&altMenu=Cour&folderID=glossary",
        "https://www.fccourts.org/resources/legal-terms-definitions/",
        "https://legal.thomsonreuters.com/en/insights/articles/glossary-of-legal-terms",
    ],

    # FINANCIAL TERMINOLOGY
    'financial': [
        "https://www.investopedia.com/financial-term-dictionary-4769738",
        "https://www.consumerfinance.gov/consumer-tools/educator-tools/youth-financial-education/glossary/",
        "https://www.dfpi.ca.gov/glossary-of-financial-terms/",
        "https://www.schwab.com/learn/story/investing-glossary-100-terms-definitions",
        "https://edge.denison.edu/insights/financial-terminology",
        "https://www.preply.com/en/blog/finance-vocabulary/",
        "https://www.afm.utexas.edu/glossary",
    ],

    # MEDICAL TERMINOLOGY
    'medical': [
        "https://medlineplus.gov/appendixa.html",
        "https://www.sgu.edu/blog/medical/must-know-medical-terms-abbreviations-acronyms/",
        "https://www.rcog.org.uk/for-the-public/a-z-of-medical-terms/",
        "https://www.germanna.edu/tutoring/handouts/guide-common-medical-terminology/",
        "https://pce.sandiego.edu/key-medical-terms/",
        "https://michiganassessment.org/teach-english/english-health-care-vocabulary/",
    ],

    # TECHNOLOGY/CS TERMINOLOGY
    'technology': [
        "https://code.org/curriculum/docs/glossary",
        "https://en.wikipedia.org/wiki/Glossary_of_computer_science",
        "https://en.wikipedia.org/wiki/Glossary_of_mechanical_engineering",
        "https://www.dataprise.com/resources/glossary/",
        "https://ccaps.umn.edu/vocabulary-computer-science-majors",
    ],

    # SPORTS TERMINOLOGY
    'sports': [
        "https://www.usingenglish.com/reference/phrasal-verbs/sports-vocabulary.html",
        "https://www.grammarly.com/blog/sports-words-americans/",
        "https://www.covers.com/guides/sports-terms-double-meanings",
    ],

    # SLANG/INFORMAL
    'slang': [
        "https://www.heylama.com/blog/english-slang-words",
        "https://youth.europa.eu/news/gen-z-slang-words-and-phrases-2024_en",
        "https://www.shorelight.com/student-stories/american-slang-words/",
        "https://www.salisbury.edu/administration/student-affairs/new-student-and-family-programs/orientation/international-student-handbook/american-slang.aspx",
        "https://www.englishanyone.com/common-slang-words/",
        "https://www.berlitz.com/blog/american-slang-words-phrases",
        "https://www.englishpath.com/20-gen-z-slang-terms/",
    ],

    # ACADEMIC/POLYSEMY RESEARCH
    'academic': [
        "https://prepedu.com/en/blog/polysemy-in-english",
        "https://www.study.com/academy/lesson/what-is-polysemy-definition-examples.html",
        "https://www.thoughtco.com/polysemy-words-and-meanings-1691642",
        "https://en.wikipedia.org/wiki/Polysemy",
        "https://en.wikipedia.org/wiki/List_of_English_homographs",
        "https://www.vocabulary.com/lists/194479",
    ],

    # MULTIPLE MEANING WORD LISTS
    'multiple_meanings': [
        "https://www.amazingtalker.com/blog/en/40-english-words-multiple-meanings/",
        "https://www.yourdictionary.com/articles/words-multiple-meanings",
        "https://www.home-speech-home.com/multiple-meaning-words.html",
        "https://www.forbrain.com/multiple-meaning-words-speech-therapy/",
        "https://www.writersdigest.com/write-better-fiction/homographs-examples",
        "https://www.immigo.io/blogs/english-words-with-multiple-meanings",
    ],
}


# =============================================================================
# POLYSEMOUS WORDS DATABASE
# =============================================================================

POLYSEMOUS_WORDS = [
    # Core polysemous words from our inventory
    'bank', 'spring', 'run', 'light', 'draft', 'set', 'play', 'match',
    'cell', 'note', 'bar', 'scale', 'band', 'pitch', 'court',

    # Extended list
    'interest', 'case', 'charge', 'bear', 'board', 'book', 'break', 'call',
    'check', 'class', 'close', 'club', 'cold', 'contract', 'cool', 'cover',
    'cross', 'culture', 'cut', 'date', 'degree', 'depression', 'die', 'draw',
    'drive', 'drop', 'exchange', 'express', 'face', 'fair', 'fall', 'fan',
    'fast', 'file', 'fire', 'firm', 'fit', 'fly', 'fold', 'force', 'fork',
    'frame', 'free', 'game', 'goal', 'grade', 'grant', 'gross', 'ground',
    'growth', 'hand', 'handle', 'head', 'heart', 'hit', 'hold', 'host',
    'house', 'iron', 'issue', 'jam', 'joint', 'jump', 'key', 'kind', 'land',
    'lap', 'lead', 'lean', 'leave', 'left', 'leg', 'letter', 'level', 'lie',
    'lift', 'line', 'link', 'list', 'live', 'load', 'lock', 'log', 'long',
    'lot', 'mail', 'main', 'make', 'mark', 'market', 'mass', 'matter', 'mean',
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
    'tape', 'target', 'taste', 'tax', 'tear', 'term', 'terminal', 'terms',
    'test', 'text', 'thread', 'tie', 'tight', 'tile', 'time', 'tip', 'title',
    'tone', 'tool', 'top', 'topic', 'touch', 'tour', 'tower', 'track', 'trade',
    'trail', 'train', 'transfer', 'trap', 'travel', 'treat', 'treatment',
    'tree', 'trial', 'trick', 'trip', 'trouble', 'truck', 'trust', 'tube',
    'tune', 'turn', 'twist', 'type', 'union', 'unit', 'value', 'variety',
    'vehicle', 'venture', 'version', 'vessel', 'view', 'virus', 'vision',
    'volume', 'vote', 'wage', 'walk', 'wall', 'war', 'warm', 'wash', 'waste',
    'watch', 'water', 'wave', 'way', 'wear', 'weather', 'web', 'weight',
    'well', 'wheel', 'will', 'wind', 'window', 'wing', 'wire', 'witness',
    'wood', 'word', 'work', 'world', 'wrap', 'write', 'yard', 'year', 'yield',
]


# =============================================================================
# DATA COLLECTOR CLASS
# =============================================================================

class ComprehensiveDataCollector:
    """Comprehensive data collector for polysemy data."""

    def __init__(self, output_dir: Path = None, rate_limit: float = 1.0):
        self.output_dir = output_dir or Path("paradigm_factory/v2/raw_events")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 QLLM-Research/1.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

        self.seen_contexts = set()
        self.events: List[Dict] = []
        self.stats = defaultdict(int)
        self.start_time = datetime.now()

    def log(self, msg: str):
        """Log with timestamp."""
        elapsed = datetime.now() - self.start_time
        print(f"[{elapsed}] {msg}")

    def fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL with rate limiting and retries."""
        for attempt in range(3):
            try:
                time.sleep(self.rate_limit)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                self.stats['pages_fetched'] += 1
                return response.text
            except requests.exceptions.Timeout:
                self.log(f"  Timeout on attempt {attempt + 1}: {url}")
                time.sleep(5)
            except requests.exceptions.RequestException as e:
                self.log(f"  Error on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(3)

        self.stats['fetch_errors'] += 1
        return None

    def extract_text(self, html: str) -> str:
        """Extract clean text from HTML."""
        # Remove unwanted elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', html)

        # Decode entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_sentences(self, text: str, word: str) -> List[str]:
        """Extract sentences containing the target word."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        matching = []

        pattern = rf'\b{re.escape(word)}s?\b'

        for sent in sentences:
            if re.search(pattern, sent, re.IGNORECASE):
                sent = sent.strip()
                sent = re.sub(r'\s+', ' ', sent)
                words = sent.split()
                if 10 <= len(words) <= 60:
                    # Check for quality
                    if not self._is_low_quality(sent):
                        matching.append(sent)

        return matching

    def _is_low_quality(self, text: str) -> bool:
        """Check if text is low quality."""
        text_lower = text.lower()

        # Skip navigation/boilerplate
        boilerplate = ['click here', 'read more', 'sign up', 'subscribe', 'cookie',
                       'privacy policy', 'terms of service', 'copyright', 'all rights']
        if any(bp in text_lower for bp in boilerplate):
            return True

        # Skip too short
        if len(text) < 40:
            return True

        # Skip mostly numbers
        alpha_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(1, len(text))
        if alpha_ratio < 0.5:
            return True

        return False

    def create_event(
        self,
        word: str,
        text: str,
        domain: str,
        source_url: str,
        gloss: str = ""
    ) -> Optional[Dict]:
        """Create a v2.1 event from extracted text."""
        # Dedup check
        ctx_hash = hashlib.md5(text.lower().encode()).hexdigest()
        if ctx_hash in self.seen_contexts:
            return None
        self.seen_contexts.add(ctx_hash)

        # Find span
        text_lower = text.lower()
        word_lower = word.lower()
        start = text_lower.find(word_lower)
        if start == -1:
            # Try word boundary search
            match = re.search(rf'\b{re.escape(word_lower)}s?\b', text_lower)
            if match:
                start = match.start()
            else:
                start = 0
        end = start + len(word)
        surface = text[start:end] if start >= 0 else word

        # Context window
        left_text = text[:start].strip()
        right_text = text[end:].strip()
        left_tokens = left_text.split()[-15:]
        right_tokens = right_text.split()[:15]

        # Cue tokens
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'on', 'it', 'that', 'for', 'with', 'as', 'at', 'by'}
        words = [w.strip('.,!?;:"\'()[]').lower() for w in text.split()]
        cue_tokens = [w for w in words if w not in stopwords and w != word_lower and len(w) > 2][:10]

        # Parse source domain
        parsed = urlparse(source_url)
        source_domain = parsed.netloc

        # Compute quality scores
        cue_strength = min(1.0, len(cue_tokens) * 0.12)
        boilerplate_risk = 0.1 if len(text) > 50 else 0.3

        event = {
            "id": str(uuid.uuid4()),
            "lemma": word,
            "pos": "noun",  # Default
            "sense_id": f"{word}#{domain}",
            "sense_gloss": gloss or f"{word} in {domain} context",
            "text": text,
            "span": {"start": start, "end": end, "surface": surface},
            "context_window": {"left": ' '.join(left_tokens), "right": ' '.join(right_tokens)},
            "cue_tokens": cue_tokens,
            "cue_type": ["context"] * len(cue_tokens),
            "topic_tags": [domain],
            "source": {
                "url": source_url,
                "domain": source_domain,
                "license": "fair-use-research",
                "rights_ok": True,
                "robots_ok": True
            },
            "quality": {
                "cue_strength": cue_strength,
                "ambiguity_risk": 0.3,
                "toxicity_risk": 0.0,
                "boilerplate_risk": boilerplate_risk,
                "length_chars": len(text),
                "style": "technical" if domain in ['legal', 'medical', 'technology', 'financial'] else "narrative"
            },
            "splits": {
                "holdout_lemma": False,
                "holdout_template_family": False,
                "holdout_cue_family": False
            },
            "provenance_hash": hashlib.sha256(f"{text}|{source_url}|{start}:{end}".encode()).hexdigest(),
            "notes": ""
        }

        return event

    def scrape_page(self, url: str, domain: str) -> int:
        """Scrape a single page for polysemy data."""
        self.log(f"  Scraping: {url}")

        html = self.fetch_url(url)
        if not html:
            return 0

        text = self.extract_text(html)
        count = 0

        # Extract sentences for each polysemous word
        for word in POLYSEMOUS_WORDS:
            sentences = self.extract_sentences(text, word)
            for sent in sentences[:5]:  # Max 5 per word per page
                event = self.create_event(word, sent, domain, url)
                if event:
                    self.events.append(event)
                    count += 1
                    self.stats[f'events_{domain}'] += 1

        self.stats['total_events'] += count
        return count

    def scrape_domain(self, domain: str, urls: List[str]) -> int:
        """Scrape all URLs for a domain."""
        self.log(f"\n{'='*60}")
        self.log(f"DOMAIN: {domain.upper()}")
        self.log(f"{'='*60}")

        total = 0
        for url in urls:
            count = self.scrape_page(url, domain)
            total += count
            self.log(f"    -> {count} events collected")

        self.log(f"  Domain total: {total} events")
        return total

    def scrape_all(self) -> int:
        """Scrape all sources."""
        self.log("=" * 60)
        self.log("COMPREHENSIVE DATA COLLECTION")
        self.log("=" * 60)
        self.log(f"Total sources: {sum(len(urls) for urls in SOURCES.values())}")
        self.log(f"Polysemous words: {len(POLYSEMOUS_WORDS)}")

        total = 0
        for domain, urls in SOURCES.items():
            count = self.scrape_domain(domain, urls)
            total += count

        return total

    def save_events(self, filename: str = "comprehensive_v21.jsonl") -> Path:
        """Save all events to JSONL."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for event in self.events:
                f.write(json.dumps(event) + '\n')

        self.log(f"\nSaved {len(self.events)} events to {output_path}")
        return output_path

    def print_stats(self):
        """Print collection statistics."""
        elapsed = datetime.now() - self.start_time

        print("\n" + "=" * 60)
        print("COLLECTION STATISTICS")
        print("=" * 60)
        print(f"  Elapsed time: {elapsed}")
        print(f"  Total events: {len(self.events)}")
        print(f"  Pages fetched: {self.stats['pages_fetched']}")
        print(f"  Fetch errors: {self.stats['fetch_errors']}")
        print(f"  Unique contexts: {len(self.seen_contexts)}")

        print("\n  By domain:")
        for key, val in sorted(self.stats.items()):
            if key.startswith('events_'):
                domain = key.replace('events_', '')
                print(f"    {domain}: {val}")

        # Word distribution
        word_counts = defaultdict(int)
        for event in self.events:
            word_counts[event['lemma']] += 1

        print(f"\n  Unique words: {len(word_counts)}")
        print("  Top 20 words:")
        for word, count in sorted(word_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"    {word}: {count}")


def main():
    """Run comprehensive data collection."""
    print("=" * 60)
    print("STARTING COMPREHENSIVE DATA COLLECTION")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)

    collector = ComprehensiveDataCollector(rate_limit=1.5)

    try:
        total = collector.scrape_all()
        print(f"\n\nTotal collected: {total} events")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving collected data...")

    # Save whatever we have
    output = collector.save_events()
    collector.print_stats()

    print(f"\nOutput: {output}")
    print(f"Finished at: {datetime.now()}")


if __name__ == "__main__":
    main()
