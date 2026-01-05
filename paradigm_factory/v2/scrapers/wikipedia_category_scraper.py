"""
Wikipedia Category Scraper
==========================

Scrapes articles from Wikipedia categories related to:
- English words with multiple meanings
- Polysemy
- Homonyms and homographs
- Domain-specific terminology
"""

import json
import re
import time
import uuid
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
import sys
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from paradigm_factory.v2.scrapers.base_scraper import (
        RawUsageEvent, SpanInfo, ContextWindow,
        SourceInfo, QualityInfo, SplitsInfo
    )
except ImportError:
    # Fallback: define minimal classes inline if imports fail
    from dataclasses import dataclass

    @dataclass
    class SpanInfo:
        start: int
        end: int
        surface: str

    @dataclass
    class ContextWindow:
        left: str
        right: str

    @dataclass
    class SourceInfo:
        url: str
        domain: str
        license: str
        rights_ok: bool = True
        robots_ok: bool = True

    @dataclass
    class QualityInfo:
        cue_strength: float = 0.0
        ambiguity_risk: float = 0.0
        toxicity_risk: float = 0.0
        boilerplate_risk: float = 0.0
        length_chars: int = 0
        style: str = "narrative"

    @dataclass
    class SplitsInfo:
        holdout_lemma: bool = False
        holdout_template_family: bool = False
        holdout_cue_family: bool = False


class WikipediaCategoryScraper:
    """Scrapes Wikipedia articles from relevant categories."""

    # Categories to scrape
    CATEGORIES = [
        "English_words_with_multiple_meanings",
        "Homonyms",
        "Polysemy",
        "English_homographs",
        "English_heteronyms",
        "Semantics",
        "Lexical_semantics",
        "Legal_terminology",
        "Medical_terminology",
        "Financial_terminology",
        "Computing_terminology",
        "Sports_terminology",
        "Musical_terminology",
        "Military_terminology",
        "Nautical_terms",
        "Aviation_terminology",
        "Scientific_terminology",
        "Botanical_nomenclature",
        "Anatomical_terms",
        "Psychological_terminology",
        "Philosophical_terminology",
    ]

    # Additional specific pages to scrape
    PAGES = [
        "List_of_English_words_with_dual_French_and_Anglo-Saxon_variations",
        "List_of_medical_roots,_suffixes_and_prefixes",
        "Glossary_of_legal_terms",
        "Glossary_of_medical_terms_related_to_communications_disorders",
        "Glossary_of_economics",
        "Glossary_of_banking",
        "Glossary_of_computer_hardware_terms",
        "Glossary_of_computer_science",
        "Glossary_of_artificial_intelligence",
        "Glossary_of_machine_learning",
        "Glossary_of_physics",
        "Glossary_of_chemistry_terms",
        "Glossary_of_biology",
        "Glossary_of_ecology",
        "Glossary_of_astronomy",
        "Glossary_of_meteorology",
        "Glossary_of_genetics",
        "Glossary_of_psychology",
        "Glossary_of_psychiatry",
        "Glossary_of_philosophy",
        "Glossary_of_music_terminology",
        "Glossary_of_musical_instruments",
        "Glossary_of_architecture",
        "Glossary_of_nautical_terms",
        "Glossary_of_aviation",
        "Glossary_of_military_terms",
        "Glossary_of_sports_terms",
        "Glossary_of_basketball",
        "Glossary_of_baseball",
        "Glossary_of_American_football",
        "Glossary_of_association_football_terms",
        "Glossary_of_cricket_terms",
        "Glossary_of_tennis_terms",
        "Glossary_of_poker_terms",
        "Glossary_of_chess",
        "Glossary_of_brewing",
        "Glossary_of_winemaking_terms",
        "Glossary_of_cooking",
        "Glossary_of_textile_manufacturing",
        "Glossary_of_rail_terminology",
        "Glossary_of_entomology_terms",
        "Glossary_of_botanical_terms",
    ]

    def __init__(self, output_dir: Path = None, rate_limit: float = 0.5):
        self.output_dir = output_dir or Path("paradigm_factory/v2/raw_events")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'QLLM-DataCollector/1.0 (Research project; contact: research@example.com)'
        })

        self.api_url = "https://en.wikipedia.org/w/api.php"
        self.events: List[Dict] = []
        self.processed_pages: Set[str] = set()

    def get_category_members(self, category: str, limit: int = 500) -> List[str]:
        """Get all articles in a category."""
        members = []
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": min(limit, 500),
            "cmtype": "page",
            "format": "json"
        }

        try:
            response = self.session.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            for member in data.get("query", {}).get("categorymembers", []):
                members.append(member["title"])

        except Exception as e:
            print(f"  Error getting category {category}: {e}")

        return members

    def get_page_content(self, title: str) -> Optional[Dict]:
        """Get page content and extract text."""
        if title in self.processed_pages:
            return None

        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|categories",
            "explaintext": True,
            "exsectionformat": "plain",
            "format": "json",
            "formatversion": "2"
        }

        try:
            response = self.session.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", [])
            if pages and "extract" in pages[0]:
                self.processed_pages.add(title)
                return {
                    "title": title,
                    "extract": pages[0]["extract"],
                    "categories": [c["title"] for c in pages[0].get("categories", [])]
                }
        except Exception as e:
            print(f"  Error fetching {title}: {e}")

        return None

    def extract_terms_from_glossary(self, page_content: Dict) -> List[Dict]:
        """Extract term definitions from glossary-style pages."""
        terms = []
        text = page_content.get("extract", "")
        title = page_content.get("title", "")

        # Pattern for glossary entries (Term: Definition or Term - Definition)
        patterns = [
            r'^([A-Z][a-z]+(?:\s+[a-z]+)?)\s*[-–:]\s*(.{20,}?)(?=\n[A-Z]|\n\n|\Z)',
            r'\n([A-Z][a-z]+(?:\s+[a-z]+)?)\s*[-–:]\s*(.{20,}?)(?=\n[A-Z]|\n\n|\Z)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                term = match.group(1).strip()
                definition = match.group(2).strip()

                if len(term) > 2 and len(definition) > 20:
                    terms.append({
                        "term": term.lower(),
                        "definition": definition[:500],
                        "source_page": title
                    })

        return terms

    def create_event_from_term(self, term_data: Dict, domain: str) -> Dict:
        """Create a v2.1 event from a term definition."""
        term = term_data["term"]
        definition = term_data["definition"]
        source_page = term_data["source_page"]

        # Create context text
        context_text = f"{term.title()}: {definition}"

        # Find term position
        term_start = 0
        term_end = len(term)

        # Generate unique ID
        event_id = str(uuid.uuid4())[:8]

        # Create provenance hash
        prov_str = f"{term}|{definition[:50]}|{source_page}"
        prov_hash = hashlib.sha256(prov_str.encode()).hexdigest()[:16]

        # Infer POS from context (simple heuristic)
        pos = "noun"
        if definition.lower().startswith("to "):
            pos = "verb"
        elif definition.lower().startswith("relating to") or definition.lower().startswith("of or"):
            pos = "adjective"

        event = {
            "id": event_id,
            "lemma": term,
            "pos": pos,
            "sense_id": f"{term}.{pos}.{domain}",
            "sense_gloss": definition[:100] + "..." if len(definition) > 100 else definition,
            "text": context_text,
            "span": {
                "start": term_start,
                "end": term_end,
                "surface": term.title()
            },
            "context_window": {
                "left": "",
                "right": context_text[term_end:].strip()
            },
            "cue_tokens": [],
            "cue_type": ["definition", "glossary"],
            "topic_tags": ["wikipedia", "glossary", domain],
            "source": {
                "url": f"https://en.wikipedia.org/wiki/{source_page.replace(' ', '_')}",
                "domain": "wikipedia.org",
                "license": "CC-BY-SA-3.0",
                "rights_ok": True,
                "robots_ok": True
            },
            "quality": {
                "cue_strength": 0.85,
                "ambiguity_risk": 0.1,
                "toxicity_risk": 0.0,
                "boilerplate_risk": 0.1,
                "length_chars": len(context_text),
                "style": "encyclopedic"
            },
            "splits": {
                "holdout_lemma": False,
                "holdout_template_family": False,
                "holdout_cue_family": False
            },
            "provenance_hash": prov_hash,
            "notes": f"From Wikipedia glossary: {source_page}"
        }

        return event

    def scrape_glossary_page(self, title: str, domain: str) -> List[Dict]:
        """Scrape a glossary page and extract events."""
        events = []

        page_content = self.get_page_content(title)
        if not page_content:
            return []

        terms = self.extract_terms_from_glossary(page_content)

        for term_data in terms:
            event = self.create_event_from_term(term_data, domain)
            events.append(event)

        return events

    def run(self) -> List[Dict]:
        """Run the category scraper."""
        print(f"{'='*60}")
        print("WIKIPEDIA CATEGORY SCRAPER")
        print(f"{'='*60}")
        print(f"Categories: {len(self.CATEGORIES)}")
        print(f"Direct pages: {len(self.PAGES)}")

        start_time = datetime.now()

        # First, scrape direct glossary pages
        print(f"\n--- Scraping glossary pages ---")
        for i, page in enumerate(self.PAGES):
            domain = page.split("_of_")[-1].replace("_", " ") if "_of_" in page else "general"
            print(f"  [{i+1}/{len(self.PAGES)}] {page}...", end=" ")

            try:
                events = self.scrape_glossary_page(page, domain)
                self.events.extend(events)
                print(f"{len(events)} terms")
            except Exception as e:
                print(f"Error: {e}")

            time.sleep(self.rate_limit)

        # Then, get articles from categories
        print(f"\n--- Scraping categories ---")
        for i, category in enumerate(self.CATEGORIES):
            print(f"  [{i+1}/{len(self.CATEGORIES)}] Category:{category}...")

            try:
                members = self.get_category_members(category, limit=100)
                print(f"    Found {len(members)} pages")

                for j, member in enumerate(members[:50]):  # Limit per category
                    if member.startswith("Category:") or member.startswith("Template:"):
                        continue

                    page_content = self.get_page_content(member)
                    if page_content:
                        terms = self.extract_terms_from_glossary(page_content)
                        for term_data in terms:
                            event = self.create_event_from_term(term_data, category.lower())
                            self.events.append(event)

                    time.sleep(self.rate_limit * 0.5)

            except Exception as e:
                print(f"    Error: {e}")

            time.sleep(self.rate_limit)

        # Save events
        output_file = self.output_dir / "wikipedia_categories.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')

        elapsed = datetime.now() - start_time
        print(f"\n{'='*60}")
        print(f"WIKIPEDIA CATEGORY SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"  Pages processed: {len(self.processed_pages)}")
        print(f"  Total events: {len(self.events)}")
        print(f"  Elapsed time: {elapsed}")
        print(f"  Saved to: {output_file}")

        return self.events


if __name__ == "__main__":
    scraper = WikipediaCategoryScraper()
    events = scraper.run()
