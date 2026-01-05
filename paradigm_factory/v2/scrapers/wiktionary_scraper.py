"""
Wiktionary Scraper
==================

Scrapes word senses and usage examples from Wiktionary.
Uses the MediaWiki API to get structured word data.
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
from dataclasses import asdict
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


class WiktionaryScraper:
    """Scrapes word senses and examples from Wiktionary."""

    POLYSEMOUS_WORDS = [
        # Core polysemous words with many senses
        "bank", "set", "run", "strike", "match", "fall", "spring", "well",
        "fair", "fine", "light", "right", "left", "watch", "plant", "head",
        "court", "case", "file", "lead", "mean", "kind", "type", "class",
        "order", "degree", "state", "power", "force", "field", "frame",
        "point", "line", "plane", "space", "time", "mind", "body", "heart",
        "hand", "arm", "foot", "leg", "eye", "face", "back", "front",
        "top", "bottom", "side", "end", "edge", "corner", "center", "middle",
        "part", "piece", "bit", "lot", "deal", "matter", "thing", "way",
        "place", "ground", "base", "root", "branch", "stem", "trunk", "leaf",
        "rock", "stone", "sand", "dust", "earth", "water", "fire", "air",
        "wave", "current", "flow", "stream", "pool", "channel", "bridge",
        "board", "card", "paper", "sheet", "page", "block", "box", "ring",
        "ball", "wheel", "gear", "chain", "net", "web", "grid", "table",
        "chair", "bed", "desk", "door", "window", "wall", "floor", "ceiling",
        "bar", "club", "band", "company", "party", "group", "team", "crew",
        "cast", "staff", "board", "council", "court", "commission", "committee",
        "draft", "draw", "drive", "drop", "dump", "express", "fair", "fast",
        "figure", "file", "film", "fire", "fit", "fix", "flag", "flat",
        "float", "flush", "fold", "follow", "force", "form", "frame", "free",
        "front", "full", "game", "gas", "gate", "gauge", "gear", "general",
        "give", "glass", "go", "grade", "grain", "grant", "grave", "green",
        "ground", "guard", "guide", "gun", "handle", "hang", "harbor", "hard",
        "head", "heart", "heat", "heavy", "hide", "high", "hit", "hold",
        "hook", "host", "hot", "house", "iron", "issue", "jack", "jam",
        "joint", "jump", "keep", "key", "kick", "kind", "lap", "last",
        "launch", "lay", "lead", "lean", "leave", "level", "license", "lie",
        "lift", "light", "like", "limit", "line", "link", "list", "live",
        "load", "lock", "log", "long", "look", "loop", "loose", "lot",
        "low", "major", "make", "mark", "mass", "master", "match", "matter",
        "mean", "measure", "medium", "meet", "memory", "mine", "minor", "mint",
        "miss", "mix", "model", "mold", "monitor", "mount", "move", "nail",
        "natural", "nature", "neck", "needle", "negative", "net", "neutral",
        "note", "notice", "object", "odd", "office", "open", "operation",
        "orange", "order", "organ", "original", "outlet", "overall", "pack",
        "pad", "page", "paint", "pair", "pan", "panel", "paper", "parallel",
        "park", "part", "party", "pass", "past", "patch", "path", "pattern",
        "pay", "peak", "pen", "pension", "period", "permit", "phase", "pick",
        "picture", "piece", "pile", "pilot", "pin", "pipe", "pitch", "place",
        "plain", "plan", "plane", "plant", "plastic", "plate", "platform",
        "play", "plot", "plug", "pocket", "point", "pole", "pool", "pop",
        "port", "position", "positive", "post", "pot", "pound", "power",
        "practice", "present", "press", "pressure", "price", "primary", "prime",
        "print", "private", "process", "produce", "program", "project", "promise",
        "proper", "property", "proportion", "proposal", "prospect", "protection",
        "provision", "public", "pull", "pump", "punch", "purpose", "push",
        "put", "quarter", "race", "rack", "radio", "rail", "rain", "raise",
        "range", "rank", "rate", "raw", "reach", "read", "ready", "real",
        "reason", "record", "red", "reference", "reflect", "reform", "refuse",
        "regular", "relate", "release", "relief", "remain", "remote", "rent",
        "repair", "repeat", "report", "represent", "request", "reserve", "reset",
        "resolution", "resort", "resource", "rest", "result", "return", "review",
        "revolution", "rich", "ride", "right", "ring", "rise", "risk", "river",
        "road", "rock", "roll", "roof", "room", "root", "rope", "rough",
        "round", "row", "royal", "rule", "run", "rush", "safe", "sail",
        "sale", "salt", "sample", "satellite", "sauce", "save", "scale", "scan",
        "scene", "school", "scope", "score", "screen", "seal", "search", "season",
        "seat", "second", "section", "sector", "secure", "seed", "select", "self",
        "sense", "sentence", "separate", "series", "service", "session", "set",
        "settle", "shade", "shadow", "shaft", "shake", "shape", "share", "sharp",
        "shed", "sheet", "shelf", "shell", "shelter", "shift", "ship", "shock",
        "shoot", "shop", "shore", "short", "shot", "shoulder", "show", "shut",
        "sick", "side", "sight", "sign", "signal", "silver", "simple", "single",
        "sink", "site", "size", "skill", "skin", "sky", "sleep", "slide",
        "slip", "slope", "slot", "slow", "small", "smart", "smell", "smoke",
        "smooth", "snap", "snow", "soft", "soil", "solid", "solution", "sound",
        "source", "south", "space", "spare", "spark", "speaker", "special",
        "speed", "spell", "spend", "spin", "spirit", "split", "spot", "spread",
        "spring", "square", "squeeze", "stable", "staff", "stage", "stake",
        "stamp", "stand", "standard", "star", "start", "state", "station",
        "status", "stay", "steel", "step", "stick", "still", "stock", "stomach",
        "stone", "stop", "store", "storm", "story", "straight", "strain",
        "strange", "stream", "street", "strength", "stress", "stretch", "strike",
        "string", "strip", "stroke", "strong", "structure", "struggle", "study",
        "stuff", "style", "subject", "substance", "succeed", "such", "sugar",
        "suit", "summer", "sun", "supply", "support", "surface", "surprise",
        "surround", "survey", "survive", "suspect", "suspend", "sweet", "swim",
        "swing", "switch", "symbol", "system", "table", "tail", "take", "talk",
        "tank", "tap", "tape", "target", "task", "taste", "tax", "team",
        "tear", "technical", "technique", "telephone", "television", "tell",
        "temperature", "temporary", "tend", "term", "terminal", "territory",
        "test", "text", "thick", "thin", "thing", "think", "threat", "throw",
        "ticket", "tie", "tight", "time", "tip", "tire", "title", "toe",
        "tone", "tongue", "tool", "tooth", "top", "total", "touch", "tough",
        "tour", "tower", "town", "track", "trade", "traffic", "trail", "train",
        "transfer", "transform", "transition", "transport", "trap", "travel",
        "treat", "tree", "trend", "trial", "trick", "trigger", "trip", "trouble",
        "truck", "trust", "truth", "try", "tube", "turn", "twist", "type",
        "understanding", "unit", "universe", "university", "upper", "urban", "use",
        "user", "usual", "value", "variety", "vehicle", "version", "victim",
        "view", "village", "vision", "visit", "voice", "volume", "vote", "wage",
        "wait", "wake", "walk", "wall", "want", "war", "warm", "warning",
        "wash", "waste", "watch", "water", "wave", "way", "weak", "wealth",
        "weapon", "wear", "weather", "web", "wedding", "week", "weight", "well",
        "west", "wheel", "white", "whole", "wide", "wild", "will", "wind",
        "window", "wine", "wing", "winter", "wire", "wish", "witness", "woman",
        "wonder", "wood", "word", "work", "worker", "world", "worry", "worth",
        "wrap", "write", "wrong", "yard", "year", "yellow", "young", "youth", "zero"
    ]

    def __init__(self, output_dir: Path = None, rate_limit: float = 1.0):
        self.output_dir = output_dir or Path("paradigm_factory/v2/raw_events")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'QLLM-DataCollector/1.0 (Research project; contact: research@example.com)'
        })

        self.api_url = "https://en.wiktionary.org/w/api.php"
        self.events: List[Dict] = []
        self.processed_words: Set[str] = set()

    def get_word_page(self, word: str) -> Optional[str]:
        """Fetch the wikitext for a word's page."""
        params = {
            "action": "query",
            "titles": word,
            "prop": "revisions",
            "rvprop": "content",
            "format": "json",
            "formatversion": "2"
        }

        try:
            response = self.session.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", [])
            if pages and "revisions" in pages[0]:
                return pages[0]["revisions"][0]["content"]
        except Exception as e:
            print(f"  Error fetching {word}: {e}")

        return None

    def parse_english_section(self, wikitext: str) -> str:
        """Extract the English section from wikitext."""
        # Find the ==English== section
        match = re.search(r'==English==\n(.*?)(?=\n==[^=]|\Z)', wikitext, re.DOTALL)
        if match:
            return match.group(1)
        return ""

    def parse_definitions(self, english_section: str, word: str) -> List[Dict]:
        """Parse definitions and examples from the English section."""
        definitions = []

        # Find POS sections (Noun, Verb, Adjective, etc.)
        pos_pattern = r'===\s*(Noun|Verb|Adjective|Adverb|Preposition|Conjunction|Pronoun|Interjection)\s*===\n(.*?)(?=\n===|\Z)'

        for pos_match in re.finditer(pos_pattern, english_section, re.DOTALL):
            pos = pos_match.group(1).lower()
            pos_content = pos_match.group(2)

            # Find numbered definitions
            def_pattern = r'#\s*([^#\n]+)'
            for i, def_match in enumerate(re.finditer(def_pattern, pos_content)):
                def_text = def_match.group(1).strip()

                # Clean up wiki markup
                def_text = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', def_text)
                def_text = re.sub(r"'''([^']+)'''", r'\1', def_text)
                def_text = re.sub(r"''([^']+)''", r'\1', def_text)
                def_text = re.sub(r'\{\{[^}]+\}\}', '', def_text)
                def_text = def_text.strip()

                if len(def_text) > 10:  # Minimum definition length
                    definitions.append({
                        'word': word,
                        'pos': pos,
                        'sense_num': i + 1,
                        'definition': def_text
                    })

        return definitions

    def create_event_from_definition(self, definition: Dict, source_url: str) -> Optional[Dict]:
        """Create a v2.1 RawUsageEvent from a definition."""
        word = definition['word']
        pos = definition['pos']
        sense_num = definition['sense_num']
        def_text = definition['definition']

        # Create a context text using the definition
        context_text = f"The word '{word}' means: {def_text}"

        # Find word position in context
        word_lower = word.lower()
        text_lower = context_text.lower()
        word_start = text_lower.find(word_lower)

        if word_start == -1:
            word_start = context_text.find("'") + 1

        word_end = word_start + len(word)

        # Generate unique ID
        event_id = str(uuid.uuid4())[:8]

        # Create provenance hash
        prov_str = f"{word}|{pos}|{sense_num}|{def_text[:50]}"
        prov_hash = hashlib.sha256(prov_str.encode()).hexdigest()[:16]

        # Create v2.1 event
        event = {
            "id": event_id,
            "lemma": word,
            "pos": pos,
            "sense_id": f"{word}.{pos}.{sense_num}",
            "sense_gloss": def_text[:100] + "..." if len(def_text) > 100 else def_text,
            "text": context_text,
            "span": {
                "start": word_start,
                "end": word_end,
                "surface": word
            },
            "context_window": {
                "left": context_text[:word_start].strip(),
                "right": context_text[word_end:].strip()
            },
            "cue_tokens": [],
            "cue_type": ["definition"],
            "topic_tags": ["wiktionary", "definition", pos],
            "source": {
                "url": source_url,
                "domain": "wiktionary.org",
                "license": "CC-BY-SA-3.0",
                "rights_ok": True,
                "robots_ok": True
            },
            "quality": {
                "cue_strength": 0.9,
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
            "notes": f"Definition {sense_num} for {word} ({pos})"
        }

        return event

    def scrape_word(self, word: str) -> List[Dict]:
        """Scrape all senses for a single word."""
        if word in self.processed_words:
            return []

        self.processed_words.add(word)
        events = []

        wikitext = self.get_word_page(word)
        if not wikitext:
            return []

        english_section = self.parse_english_section(wikitext)
        if not english_section:
            return []

        definitions = self.parse_definitions(english_section, word)
        source_url = f"https://en.wiktionary.org/wiki/{word}"

        for defn in definitions:
            event = self.create_event_from_definition(defn, source_url)
            if event:
                events.append(event)

        return events

    def run(self, max_words: int = None) -> List[Dict]:
        """Run the scraper on all polysemous words."""
        words = self.POLYSEMOUS_WORDS[:max_words] if max_words else self.POLYSEMOUS_WORDS

        print(f"Starting Wiktionary scraper...")
        print(f"Words to scrape: {len(words)}")

        start_time = datetime.now()

        for i, word in enumerate(words):
            print(f"  [{i+1}/{len(words)}] Scraping '{word}'...", end=" ")

            try:
                word_events = self.scrape_word(word)
                self.events.extend(word_events)
                print(f"{len(word_events)} senses")
            except Exception as e:
                print(f"Error: {e}")

            time.sleep(self.rate_limit)

            # Progress update every 50 words
            if (i + 1) % 50 == 0:
                elapsed = datetime.now() - start_time
                print(f"\n  Progress: {i+1}/{len(words)} words, {len(self.events)} events, elapsed: {elapsed}\n")

        # Save events
        output_file = self.output_dir / "wiktionary_senses.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')

        elapsed = datetime.now() - start_time
        print(f"\n{'='*60}")
        print(f"WIKTIONARY SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"  Total words scraped: {len(self.processed_words)}")
        print(f"  Total events: {len(self.events)}")
        print(f"  Elapsed time: {elapsed}")
        print(f"  Saved to: {output_file}")

        return self.events


if __name__ == "__main__":
    scraper = WiktionaryScraper()
    events = scraper.run()
