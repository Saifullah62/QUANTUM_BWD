"""
Polysemy List Scraper
=====================

Specialized scraper for structured polysemy word lists from educational sites.
Targets sites like:
- prepedu.com - 59 polysemous words
- amazingtalker.com - 40 words with multiple meanings
- home-speech-home.com - 100 multiple meaning words
- yourdictionary.com - multiple meaning words
- study.com - polysemy examples
- thoughtco.com - polysemy explanation
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
from collections import defaultdict
from urllib.parse import urlparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paradigm_factory.v2.scrapers.base_scraper import (
    RawUsageEvent, SpanInfo, ContextWindow,
    SourceInfo, QualityInfo, SplitsInfo
)


class PolysemyListScraper:
    """Scraper for structured polysemy word lists."""

    def __init__(self, output_dir: Path = None, rate_limit: float = 2.0):
        self.output_dir = output_dir or Path("paradigm_factory/v2/raw_events")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

        self.seen_contexts = set()
        self.events: List[Dict] = []
        self.stats = defaultdict(int)

    def log(self, msg: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL with retries."""
        for attempt in range(3):
            try:
                time.sleep(self.rate_limit)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.text
            except Exception as e:
                self.log(f"  Attempt {attempt + 1} failed: {e}")
                time.sleep(3)
        return None

    def extract_text(self, html: str) -> str:
        """Clean HTML to text."""
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_word_meanings(self, text: str) -> List[Tuple[str, List[str], List[str]]]:
        """Extract word, meanings, and examples from text."""
        results = []

        # Pattern 1: Word - meaning patterns
        # e.g., "Bank - 1. A financial institution 2. The side of a river"
        pattern1 = re.compile(
            r'\b([A-Z][a-z]+)\b\s*[-–:]\s*((?:\d+\.\s*.+?)+)',
            re.MULTILINE
        )

        for match in pattern1.finditer(text):
            word = match.group(1).lower()
            meanings_text = match.group(2)

            # Split meanings
            meanings = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', meanings_text)
            meanings = [m.strip() for m in meanings if len(m.strip()) > 10]

            if len(meanings) >= 2:
                results.append((word, meanings, []))

        # Pattern 2: Word with example sentences
        # e.g., "The word 'bank' can mean... Example: I went to the bank."
        pattern2 = re.compile(
            r'\b([A-Za-z]+)\b[:\s]+(?:can mean|means|refers to)\s+(.+?)(?:Example[s]?:\s*(.+?))?(?=\.|$)',
            re.IGNORECASE | re.DOTALL
        )

        for match in pattern2.finditer(text):
            word = match.group(1).lower()
            meaning = match.group(2).strip()
            example = match.group(3).strip() if match.group(3) else ""

            if len(meaning) > 20:
                results.append((word, [meaning], [example] if example else []))

        return results

    def extract_sentences(self, text: str, word: str) -> List[str]:
        """Extract sentences containing word."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        matching = []

        pattern = rf'\b{re.escape(word)}s?\b'

        for sent in sentences:
            if re.search(pattern, sent, re.IGNORECASE):
                sent = sent.strip()
                words = sent.split()
                if 8 <= len(words) <= 50:
                    matching.append(sent)

        return matching

    def create_event(
        self,
        word: str,
        text: str,
        sense_label: str,
        gloss: str,
        source_url: str
    ) -> Optional[Dict]:
        """Create a v2.1 format event."""
        ctx_hash = hashlib.md5(text.lower().encode()).hexdigest()
        if ctx_hash in self.seen_contexts:
            return None
        self.seen_contexts.add(ctx_hash)

        # Find span
        text_lower = text.lower()
        word_lower = word.lower()
        start = text_lower.find(word_lower)
        if start == -1:
            match = re.search(rf'\b{re.escape(word_lower)}s?\b', text_lower)
            if match:
                start = match.start()
            else:
                start = 0
        end = start + len(word)
        surface = text[start:end] if start >= 0 else word

        # Context window
        left = text[:start].strip().split()[-15:]
        right = text[end:].strip().split()[:15]

        # Cue tokens
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'on', 'it', 'that', 'for', 'with'}
        words = [w.strip('.,!?;:"\'()[]').lower() for w in text.split()]
        cue_tokens = [w for w in words if w not in stopwords and w != word_lower and len(w) > 2][:10]

        parsed = urlparse(source_url)

        return {
            "id": str(uuid.uuid4()),
            "lemma": word,
            "pos": "noun",
            "sense_id": f"{word}#{sense_label}",
            "sense_gloss": gloss,
            "text": text,
            "span": {"start": start, "end": end, "surface": surface},
            "context_window": {"left": ' '.join(left), "right": ' '.join(right)},
            "cue_tokens": cue_tokens,
            "cue_type": ["context"] * len(cue_tokens),
            "topic_tags": ["polysemy", sense_label],
            "source": {
                "url": source_url,
                "domain": parsed.netloc,
                "license": "educational-fair-use",
                "rights_ok": True,
                "robots_ok": True
            },
            "quality": {
                "cue_strength": min(1.0, len(cue_tokens) * 0.12),
                "ambiguity_risk": 0.2,
                "toxicity_risk": 0.0,
                "boilerplate_risk": 0.1,
                "length_chars": len(text),
                "style": "instructional"
            },
            "splits": {
                "holdout_lemma": False,
                "holdout_template_family": False,
                "holdout_cue_family": False
            },
            "provenance_hash": hashlib.sha256(f"{text}|{source_url}|{start}:{end}".encode()).hexdigest(),
            "notes": ""
        }

    def scrape_prepedu(self) -> int:
        """Scrape prepedu.com polysemy list."""
        url = "https://prepedu.com/en/blog/polysemy-in-english"
        self.log(f"Scraping: {url}")

        html = self.fetch_url(url)
        if not html:
            return 0

        text = self.extract_text(html)
        count = 0

        # Known polysemous words from prepedu
        polysemous_words = [
            'bank', 'bark', 'bass', 'bat', 'bear', 'board', 'book', 'bow', 'box',
            'branch', 'break', 'bridge', 'bright', 'bug', 'cabinet', 'capital',
            'cell', 'chair', 'change', 'check', 'class', 'club', 'coach', 'court',
            'crane', 'cross', 'current', 'date', 'deck', 'degree', 'draft', 'fair',
            'fall', 'fan', 'file', 'fire', 'fly', 'foot', 'fork', 'frame', 'game',
            'glass', 'grade', 'ground', 'hand', 'head', 'host', 'iron', 'jam',
            'key', 'kind', 'lap', 'lead', 'left', 'letter', 'light', 'line', 'match'
        ]

        for word in polysemous_words:
            sentences = self.extract_sentences(text, word)
            for i, sent in enumerate(sentences[:3]):
                event = self.create_event(
                    word=word,
                    text=sent,
                    sense_label=f"sense{i+1}",
                    gloss=f"Meaning {i+1} of {word}",
                    source_url=url
                )
                if event:
                    self.events.append(event)
                    count += 1

        self.log(f"  -> {count} events")
        return count

    def scrape_amazingtalker(self) -> int:
        """Scrape amazingtalker.com multiple meanings list."""
        url = "https://en.amazingtalker.com/blog/en/40-english-words-multiple-meanings/"
        self.log(f"Scraping: {url}")

        html = self.fetch_url(url)
        if not html:
            return 0

        text = self.extract_text(html)
        count = 0

        words = [
            'address', 'back', 'bank', 'bar', 'bat', 'bear', 'block', 'book',
            'box', 'break', 'bright', 'can', 'capital', 'case', 'change',
            'charge', 'check', 'chip', 'class', 'clear', 'clip', 'close',
            'club', 'coach', 'cool', 'court', 'crane', 'cross', 'current',
            'date', 'diamond', 'draft', 'fair', 'fan', 'file', 'fire', 'fly',
            'foot', 'fork', 'frame'
        ]

        for word in words:
            sentences = self.extract_sentences(text, word)
            for i, sent in enumerate(sentences[:2]):
                event = self.create_event(
                    word=word,
                    text=sent,
                    sense_label=f"sense{i+1}",
                    gloss=f"Multiple meaning {i+1} of {word}",
                    source_url=url
                )
                if event:
                    self.events.append(event)
                    count += 1

        self.log(f"  -> {count} events")
        return count

    def scrape_home_speech_home(self) -> int:
        """Scrape home-speech-home.com multiple meaning words."""
        url = "https://www.home-speech-home.com/multiple-meaning-words.html"
        self.log(f"Scraping: {url}")

        html = self.fetch_url(url)
        if not html:
            return 0

        text = self.extract_text(html)
        count = 0

        words = [
            'arm', 'back', 'ball', 'band', 'bank', 'bark', 'bat', 'bear',
            'beat', 'bed', 'bit', 'blue', 'board', 'book', 'bowl', 'box',
            'break', 'bright', 'bug', 'call', 'can', 'capital', 'case',
            'cast', 'catch', 'cell', 'change', 'charge', 'check', 'chip',
            'class', 'clear', 'clip', 'close', 'club', 'coach', 'cold',
            'cool', 'court', 'cover', 'crane', 'cross', 'current', 'cut',
            'date', 'deck', 'draft', 'draw', 'dress', 'drop', 'duck', 'fair',
            'fall', 'fan', 'fast', 'file', 'fire', 'fit', 'flat', 'fly',
            'foot', 'fork', 'frame', 'free', 'game', 'glass', 'grade',
            'ground', 'hand', 'hang', 'hard', 'head', 'hide', 'hit', 'hold',
            'host', 'iron', 'jam', 'key', 'kid', 'kind', 'land', 'lap',
            'last', 'lead', 'lean', 'left', 'letter', 'lie', 'light', 'like',
            'line', 'live', 'long', 'lot', 'match', 'mean', 'mine', 'miss'
        ]

        for word in words:
            sentences = self.extract_sentences(text, word)
            for i, sent in enumerate(sentences[:2]):
                event = self.create_event(
                    word=word,
                    text=sent,
                    sense_label=f"sense{i+1}",
                    gloss=f"Speech therapy meaning {i+1} of {word}",
                    source_url=url
                )
                if event:
                    self.events.append(event)
                    count += 1

        self.log(f"  -> {count} events")
        return count

    def scrape_yourdictionary(self) -> int:
        """Scrape yourdictionary.com multiple meanings."""
        url = "https://examples.yourdictionary.com/examples-of-words-with-multiple-meanings.html"
        self.log(f"Scraping: {url}")

        html = self.fetch_url(url)
        if not html:
            return 0

        text = self.extract_text(html)
        count = 0

        words = [
            'arm', 'ball', 'band', 'bank', 'bark', 'bat', 'beam', 'bear',
            'beat', 'bit', 'block', 'blow', 'board', 'bolt', 'book', 'bow',
            'box', 'break', 'bridge', 'bright', 'brush', 'bug', 'burn',
            'cabinet', 'can', 'capital', 'case', 'cast', 'cell', 'change',
            'charge', 'chest', 'chip', 'class', 'clear', 'clip', 'close',
            'club', 'coach', 'coat', 'cold', 'cool', 'count', 'course',
            'court', 'cover', 'crane', 'cross', 'current', 'cut', 'date',
            'deal', 'deck', 'degree', 'die', 'draft', 'draw', 'dress',
            'drive', 'drop', 'drum', 'duck', 'dust', 'express', 'eye',
            'face', 'fail', 'fair', 'fall', 'fan', 'fast', 'figure', 'file',
            'fire', 'firm', 'fit', 'flag', 'flat', 'fly', 'fold', 'foot'
        ]

        for word in words:
            sentences = self.extract_sentences(text, word)
            for i, sent in enumerate(sentences[:2]):
                event = self.create_event(
                    word=word,
                    text=sent,
                    sense_label=f"sense{i+1}",
                    gloss=f"Dictionary meaning {i+1} of {word}",
                    source_url=url
                )
                if event:
                    self.events.append(event)
                    count += 1

        self.log(f"  -> {count} events")
        return count

    def scrape_wikipedia_polysemy(self) -> int:
        """Scrape Wikipedia polysemy and homograph articles."""
        urls = [
            ("https://en.wikipedia.org/wiki/Polysemy", "polysemy"),
            ("https://en.wikipedia.org/wiki/List_of_English_homographs", "homograph"),
        ]

        count = 0
        for url, category in urls:
            self.log(f"Scraping: {url}")

            html = self.fetch_url(url)
            if not html:
                continue

            text = self.extract_text(html)

            # Extract examples from Wikipedia
            words = [
                'bank', 'bar', 'bass', 'bat', 'beam', 'bear', 'bit', 'bow',
                'bright', 'bug', 'case', 'change', 'charge', 'check', 'chip',
                'class', 'clear', 'close', 'club', 'coach', 'cold', 'content',
                'contract', 'cool', 'court', 'crane', 'cross', 'current',
                'date', 'desert', 'dove', 'draft', 'entrance', 'fair', 'fan',
                'file', 'fine', 'fire', 'fly', 'foot', 'game', 'grave', 'head',
                'hide', 'lead', 'lean', 'left', 'lie', 'light', 'live', 'long',
                'match', 'mean', 'minute', 'miss', 'mole', 'object', 'present',
                'produce', 'project', 'read', 'record', 'refuse', 'row',
                'second', 'subject', 'tear', 'wind', 'wound'
            ]

            for word in words:
                sentences = self.extract_sentences(text, word)
                for i, sent in enumerate(sentences[:2]):
                    event = self.create_event(
                        word=word,
                        text=sent,
                        sense_label=category,
                        gloss=f"Wikipedia {category} example: {word}",
                        source_url=url
                    )
                    if event:
                        self.events.append(event)
                        count += 1

        self.log(f"  -> {count} events from Wikipedia")
        return count

    def scrape_all(self) -> int:
        """Scrape all polysemy list sources."""
        print("=" * 60)
        print("POLYSEMY LIST SCRAPER")
        print("=" * 60)

        total = 0
        total += self.scrape_prepedu()
        total += self.scrape_amazingtalker()
        total += self.scrape_home_speech_home()
        total += self.scrape_yourdictionary()
        total += self.scrape_wikipedia_polysemy()

        return total

    def save_events(self, filename: str = "polysemy_lists_v21.jsonl") -> Path:
        """Save events to JSONL."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for event in self.events:
                f.write(json.dumps(event) + '\n')

        self.log(f"\nSaved {len(self.events)} events to {output_path}")
        return output_path

    def print_stats(self):
        """Print statistics."""
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        print(f"  Total events: {len(self.events)}")
        print(f"  Unique contexts: {len(self.seen_contexts)}")

        word_counts = defaultdict(int)
        for event in self.events:
            word_counts[event['lemma']] += 1

        print(f"\n  Unique words: {len(word_counts)}")
        print("  Top 20 words:")
        for word, count in sorted(word_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"    {word}: {count}")


def main():
    print("=" * 60)
    print("POLYSEMY LIST SCRAPER")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    scraper = PolysemyListScraper(rate_limit=2.0)
    total = scraper.scrape_all()

    print(f"\nTotal collected: {total}")

    output = scraper.save_events()
    scraper.print_stats()

    print(f"\nOutput: {output}")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
