"""
Wikipedia Scraper (v2.1 Schema)
===============================

Scrapes sense-labeled usage events from Wikipedia articles.
Uses disambiguation pages and category-based retrieval to find relevant articles.

Schema v2.1 compliant format.
"""

import json
import re
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Generator
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


class WikipediaScraper(BaseScraper):
    """Scrape usage events from Wikipedia."""

    BASE_URL = "https://en.wikipedia.org/w/api.php"

    # Domain -> category mappings for targeted scraping
    DOMAIN_CATEGORIES = {
        'finance': ['Category:Finance', 'Category:Banking', 'Category:Economics'],
        'geography': ['Category:Geography', 'Category:Rivers', 'Category:Landforms'],
        'technology': ['Category:Technology', 'Category:Computing', 'Category:Software'],
        'biology': ['Category:Biology', 'Category:Anatomy', 'Category:Organisms'],
        'law': ['Category:Law', 'Category:Legal_terms', 'Category:Courts'],
        'music': ['Category:Music', 'Category:Musical_instruments', 'Category:Songs'],
        'sports': ['Category:Sports', 'Category:Games', 'Category:Athletes'],
        'medicine': ['Category:Medicine', 'Category:Diseases', 'Category:Medical_terms'],
    }

    def __init__(
        self,
        sense_inventory: Dict,
        output_dir: Path = None,
        rate_limit: float = 0.5,  # Seconds between requests
        **kwargs
    ):
        super().__init__(sense_inventory, output_dir, rate_limit=rate_limit, **kwargs)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'QLLM-BundleScraper/1.0 (research project)'
        })

    def _api_request(self, params: Dict) -> Optional[Dict]:
        """Make a request to Wikipedia API with rate limiting."""
        params['format'] = 'json'

        try:
            time.sleep(self.rate_limit)
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  API error: {e}")
            return None

    def search_articles(self, query: str, limit: int = 20) -> List[str]:
        """Search for articles matching a query."""
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'srlimit': limit,
            'srprop': 'snippet'
        }

        result = self._api_request(params)
        if not result:
            return []

        titles = []
        for item in result.get('query', {}).get('search', []):
            titles.append(item['title'])

        return titles

    def get_article_text(self, title: str) -> Optional[str]:
        """Get plain text content of an article."""
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain'
        }

        result = self._api_request(params)
        if not result:
            return None

        pages = result.get('query', {}).get('pages', {})
        for page in pages.values():
            return page.get('extract', '')

        return None

    def get_category_articles(self, category: str, limit: int = 50) -> List[str]:
        """Get articles in a category."""
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': category,
            'cmtype': 'page',
            'cmlimit': limit
        }

        result = self._api_request(params)
        if not result:
            return []

        return [
            m['title']
            for m in result.get('query', {}).get('categorymembers', [])
        ]

    def extract_sentences(self, text: str, lemma: str) -> List[str]:
        """Extract sentences containing the target lemma."""
        if not text:
            return []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter to sentences containing the lemma
        lemma_pattern = rf'\b{re.escape(lemma)}\b'
        matching = []

        for sent in sentences:
            if re.search(lemma_pattern, sent, re.IGNORECASE):
                # Basic cleanup
                sent = sent.strip()
                sent = re.sub(r'\s+', ' ', sent)

                # Skip very short or very long
                words = sent.split()
                if 10 <= len(words) <= 60:
                    matching.append(sent)

        return matching

    def classify_sense(
        self,
        sentence: str,
        lemma: str,
        senses: List[Dict]
    ) -> Optional[Dict]:
        """
        Attempt to classify which sense is used in a sentence.
        Returns the sense with highest cue match score.
        """
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())

        best_sense = None
        best_score = 0

        for sense in senses:
            score = 0

            # Check cue keywords
            cue_kw = set(sense.get('cues', {}).get('keywords', []))
            cue_hits = cue_kw & sentence_words
            score += len(cue_hits) * 2

            # Check cue collocates
            for colloc in sense.get('cues', {}).get('collocates', []):
                if colloc.lower() in sentence_lower:
                    score += 3

            # Penalty for anti-cue hits
            anti_kw = set(sense.get('anti_cues', {}).get('keywords', []))
            anti_hits = anti_kw & sentence_words
            score -= len(anti_hits) * 3

            # Check anti-cue collocates
            for colloc in sense.get('anti_cues', {}).get('collocates', []):
                if colloc.lower() in sentence_lower:
                    score -= 4

            if score > best_score:
                best_score = score
                best_sense = sense

        # Require a minimum confidence
        if best_score >= 2:
            return best_sense

        return None

    def create_event(
        self,
        sentence: str,
        lemma: str,
        sense: Dict,
        article_title: str
    ) -> Optional[RawUsageEvent]:
        """Create a RawUsageEvent from a sentence and sense (v2.1 schema)."""

        # Find target span
        span = self.find_target_span(sentence, lemma)
        if not span:
            return None

        # Extract context window
        context_window = self.extract_context_window(sentence, span, window_tokens=15)

        # Extract cue tokens and types
        cue_tokens, cue_types = self.extract_cue_tokens(
            sentence, lemma,
            sense.get('cues', {})
        )

        # Get topic tags
        topic_tags = self.get_topic_tags(sense)

        # Build source info
        url = f"https://en.wikipedia.org/wiki/{article_title.replace(' ', '_')}"
        source = SourceInfo(
            url=url,
            domain="wikipedia.org",
            license="CC-BY-SA-3.0",
            rights_ok=True,
            robots_ok=True
        )

        # Compute quality metrics
        cue_strength = self.compute_cue_strength(cue_tokens, sense.get('cues', {}))
        ambiguity_risk = self.compute_ambiguity_risk(sentence, sense.get('anti_cues', {}))
        boilerplate_risk = self.compute_boilerplate_risk(sentence)
        style = self.detect_style(sentence)

        quality = QualityInfo(
            cue_strength=cue_strength,
            ambiguity_risk=ambiguity_risk,
            toxicity_risk=0.0,  # Wikipedia is generally safe
            boilerplate_risk=boilerplate_risk,
            length_chars=len(sentence),
            style=style
        )

        # Build splits info (default for now)
        splits = SplitsInfo(
            holdout_lemma=False,
            holdout_template_family=False,
            holdout_cue_family=False
        )

        # Compute provenance hash
        provenance_hash = self.compute_provenance_hash(sentence, url, span)

        return RawUsageEvent(
            id=self.generate_event_id(),
            lemma=lemma,
            pos=sense.get('pos', 'noun').lower(),
            sense_id=sense.get('sense_id', f"{lemma}#unknown"),
            sense_gloss=sense.get('gloss', ''),
            text=sentence,
            span=span,
            context_window=context_window,
            cue_tokens=cue_tokens,
            cue_type=cue_types,
            topic_tags=topic_tags,
            source=source,
            quality=quality,
            splits=splits,
            provenance_hash=provenance_hash,
            notes=""
        )

    def scrape_lemma(
        self,
        lemma: str,
        senses: List[Dict],
        max_per_sense: int = 100
    ) -> Generator[RawUsageEvent, None, None]:
        """Scrape events for a single lemma."""

        # Track counts per sense
        sense_counts = {s['sense_id']: 0 for s in senses}

        # Strategy 1: Search for lemma directly
        articles = self.search_articles(lemma, limit=30)

        # Strategy 2: Search with domain keywords
        for sense in senses:
            domain = sense.get('domain', 'general')
            cue_kw = sense.get('cues', {}).get('keywords', [])[:2]
            if cue_kw:
                query = f"{lemma} {' '.join(cue_kw)}"
                articles.extend(self.search_articles(query, limit=10))

        # Deduplicate
        articles = list(set(articles))
        print(f"    Found {len(articles)} articles for '{lemma}'")

        # Process articles
        for title in articles:
            # Check if we have enough for all senses
            if all(c >= max_per_sense for c in sense_counts.values()):
                break

            text = self.get_article_text(title)
            if not text:
                continue

            sentences = self.extract_sentences(text, lemma)

            for sent in sentences:
                # Classify the sense
                sense = self.classify_sense(sent, lemma, senses)
                if not sense:
                    continue

                # Check quota
                sid = sense['sense_id']
                if sense_counts[sid] >= max_per_sense:
                    continue

                # Create event
                event = self.create_event(sent, lemma, sense, title)
                if event:
                    sense_counts[sid] += 1
                    yield event

    def scrape(
        self,
        lemmas: List[str],
        max_per_sense: int = 100
    ) -> Generator[RawUsageEvent, None, None]:
        """Scrape usage events for given lemmas."""

        for lemma in lemmas:
            if lemma not in self.lemma_senses:
                print(f"  Skipping '{lemma}' - not in sense inventory")
                continue

            senses = self.lemma_senses[lemma]
            print(f"\n  Scraping '{lemma}' ({len(senses)} senses)...")

            for event in self.scrape_lemma(lemma, senses, max_per_sense):
                yield event


def main():
    """Test the Wikipedia scraper."""
    # Load sense inventory
    inv_path = Path("paradigm_factory/v2/sense_inventory/sense_inventory.jsonl")
    if not inv_path.exists():
        print("Sense inventory not found. Run build_inventory.py first.")
        return

    inventory = {}
    with open(inv_path) as f:
        for line in f:
            entry = json.loads(line)
            key = f"{entry['lemma']}_{entry['pos']}"
            inventory[key] = entry

    print(f"Loaded {len(inventory)} lemma inventories")

    # Create scraper
    scraper = WikipediaScraper(
        sense_inventory=inventory,
        min_context_tokens=15,
        max_context_tokens=50
    )

    # Test with a few lemmas
    test_lemmas = ['bank', 'spring', 'run']
    output = scraper.scrape_and_save(
        lemmas=test_lemmas,
        max_per_sense=20,
        output_file="wikipedia_test.jsonl"
    )

    print(f"\nOutput saved to: {output}")


if __name__ == "__main__":
    main()
