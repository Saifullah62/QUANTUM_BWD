"""
WordNet Scraper
===============

Uses NLTK's WordNet interface to extract word senses and example sentences.
WordNet is one of the most comprehensive lexical databases of English.
"""

import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
from dataclasses import dataclass

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("Installing NLTK...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'nltk', '-q'])
    import nltk
    from nltk.corpus import wordnet as wn


# Dataclass definitions for event schema
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


class WordNetScraper:
    """Extracts word senses and examples from WordNet."""

    # High-polysemy words to focus on
    POLYSEMOUS_WORDS = [
        # Most polysemous in WordNet
        "break", "cut", "run", "take", "set", "get", "go", "make", "put", "turn",
        "head", "line", "point", "time", "way", "work", "hand", "part", "case", "place",
        "right", "back", "light", "sound", "call", "fall", "play", "strike", "pass", "draw",
        "drive", "stand", "hold", "catch", "leave", "stop", "keep", "check", "cover", "change",
        "close", "open", "clear", "fire", "charge", "field", "figure", "spring", "stock", "match",
        "bank", "bar", "base", "bass", "bat", "bear", "beat", "bit", "block", "blow",
        "board", "body", "book", "bore", "bound", "box", "break", "bridge", "brush", "burn",
        "can", "capital", "cast", "catch", "cell", "chain", "chair", "chance", "change", "charge",
        "check", "class", "clean", "clear", "clip", "club", "coach", "coat", "code", "cold",
        "color", "company", "contract", "cook", "cool", "copy", "count", "course", "court", "cover",
        "crane", "crash", "credit", "cross", "crown", "current", "cut", "date", "deal", "deck",
        "degree", "desert", "design", "detail", "direct", "dish", "dock", "doctor", "dog", "down",
        "draft", "draw", "dress", "drill", "drink", "drive", "drop", "drum", "dry", "duck",
        "dump", "duty", "ear", "earth", "edge", "effect", "end", "engine", "equal", "exchange",
        "express", "eye", "face", "fair", "fall", "fan", "farm", "fast", "father", "favor",
        "feel", "field", "fight", "figure", "file", "fill", "film", "find", "fine", "finish",
        "fire", "firm", "fish", "fit", "fix", "flag", "flash", "flat", "flight", "float",
        "floor", "flow", "fly", "focus", "fold", "follow", "foot", "force", "fork", "form",
        "frame", "free", "freeze", "front", "fruit", "fuel", "function", "game", "gas", "gate",
        "gauge", "gear", "general", "give", "glass", "grade", "grain", "grant", "green", "ground",
        "group", "grow", "guard", "guide", "gun", "hand", "handle", "hang", "harbor", "hard",
        "head", "hear", "heart", "heat", "heavy", "heel", "help", "hide", "high", "hit",
        "hold", "hole", "home", "hook", "hope", "horn", "horse", "host", "hot", "house",
        "iron", "issue", "jack", "jam", "job", "joint", "judge", "jump", "keep", "key",
        "kick", "kind", "king", "kitchen", "knee", "land", "lap", "last", "late", "launch",
        "law", "lay", "lead", "lean", "learn", "leave", "left", "leg", "lemon", "letter",
        "level", "lie", "lift", "light", "like", "limit", "line", "link", "lip", "list",
        "live", "load", "lock", "log", "long", "look", "loop", "loose", "lose", "lot",
        "love", "low", "mail", "main", "major", "make", "man", "mark", "market", "mass",
        "master", "match", "matter", "mean", "measure", "meet", "member", "memory", "mind", "mine",
        "minor", "mint", "miss", "mix", "model", "money", "monitor", "month", "mood", "moon",
        "morning", "mother", "motion", "motor", "mount", "mouse", "mouth", "move", "movie", "music",
        "nail", "name", "nature", "neck", "needle", "net", "network", "neutral", "news", "night",
        "note", "notice", "number", "nurse", "object", "ocean", "odd", "offer", "office", "oil",
        "open", "operation", "order", "organ", "original", "other", "out", "outline", "over", "own",
        "pack", "page", "paint", "pair", "palm", "pan", "panel", "paper", "park", "part",
        "party", "pass", "past", "patch", "path", "pattern", "pay", "peak", "pen", "people",
        "period", "person", "phase", "phone", "pick", "picture", "piece", "pile", "pilot", "pin",
        "pipe", "pitch", "place", "plain", "plan", "plane", "plant", "plastic", "plate", "platform",
        "play", "plot", "plug", "pocket", "point", "pole", "pool", "poor", "pop", "port",
        "position", "post", "pot", "pound", "power", "practice", "present", "press", "pressure", "price",
        "prime", "print", "private", "process", "produce", "program", "project", "promise", "proof", "property",
        "prospect", "protection", "public", "pull", "pump", "punch", "purple", "purpose", "push", "put",
        "quality", "quarter", "question", "quick", "quiet", "race", "rack", "radio", "rail", "rain",
        "raise", "range", "rank", "rate", "raw", "reach", "read", "ready", "real", "reason",
        "record", "red", "reference", "reflect", "reform", "region", "regular", "relate", "release", "relief",
        "remain", "remote", "rent", "repair", "repeat", "report", "represent", "request", "reserve", "reset",
        "resolution", "resort", "resource", "rest", "result", "return", "review", "revolution", "rich", "ride",
        "right", "ring", "rise", "risk", "river", "road", "rock", "roll", "roof", "room",
        "root", "rope", "rough", "round", "row", "royal", "rule", "run", "rush", "safe",
        "sail", "sale", "salt", "same", "sample", "sand", "save", "scale", "scene", "school",
        "science", "score", "screen", "sea", "seal", "search", "season", "seat", "second", "secret",
        "section", "secure", "see", "seed", "seek", "select", "self", "sell", "send", "sense",
        "sentence", "separate", "series", "serve", "service", "session", "set", "settle", "shade", "shadow",
        "shaft", "shake", "shape", "share", "sharp", "shed", "sheet", "shelf", "shell", "shelter",
        "shift", "shine", "ship", "shock", "shoe", "shoot", "shop", "shore", "short", "shot",
        "shoulder", "show", "shut", "sick", "side", "sight", "sign", "signal", "silver", "simple",
        "single", "sink", "sister", "sit", "site", "situation", "size", "skill", "skin", "sky",
        "sleep", "slide", "slip", "slope", "slow", "small", "smart", "smell", "smile", "smoke",
        "smooth", "snap", "snow", "soft", "soil", "solid", "solution", "solve", "some", "son",
        "song", "sound", "source", "south", "space", "spare", "speak", "special", "speed", "spell",
        "spend", "spin", "spirit", "split", "spot", "spread", "spring", "square", "squeeze", "stable"
    ]

    POS_MAP = {
        'n': 'noun',
        'v': 'verb',
        'a': 'adjective',
        's': 'adjective',  # satellite adjective
        'r': 'adverb'
    }

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("paradigm_factory/v2/raw_events")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Download WordNet if needed
        try:
            wn.synsets('test')
        except LookupError:
            print("Downloading WordNet...")
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

        self.events: List[Dict] = []
        self.processed_words: Set[str] = set()

    def get_word_senses(self, word: str) -> List[Dict]:
        """Get all senses for a word from WordNet."""
        senses = []

        for synset in wn.synsets(word):
            # Get lemma names for this synset
            lemmas = synset.lemmas()

            for lemma in lemmas:
                if lemma.name().lower() == word.lower():
                    # This synset includes our word
                    pos_code = synset.pos()
                    pos = self.POS_MAP.get(pos_code, pos_code)

                    sense_data = {
                        'word': word,
                        'synset_name': synset.name(),
                        'pos': pos,
                        'definition': synset.definition(),
                        'examples': synset.examples(),
                        'lemma_count': lemma.count() if lemma.count() else 0,
                        'hypernyms': [h.name() for h in synset.hypernyms()[:3]],
                        'hyponyms': [h.name() for h in synset.hyponyms()[:3]]
                    }
                    senses.append(sense_data)
                    break

        return senses

    def create_event_from_sense(self, sense_data: Dict) -> Dict:
        """Create a v2.1 event from a WordNet sense."""
        word = sense_data['word']
        synset_name = sense_data['synset_name']
        pos = sense_data['pos']
        definition = sense_data['definition']
        examples = sense_data.get('examples', [])

        # Create context text
        if examples:
            # Use the first example sentence
            context_text = examples[0]
        else:
            # Create from definition
            context_text = f"{word.title()}: {definition}"

        # Find word position in context
        word_lower = word.lower()
        text_lower = context_text.lower()
        word_start = text_lower.find(word_lower)

        if word_start == -1:
            # Word not found directly, create synthetic context
            context_text = f"The term {word} refers to {definition}"
            word_start = context_text.lower().find(word_lower)

        word_end = word_start + len(word)
        surface = context_text[word_start:word_end]

        # Generate unique ID
        event_id = str(uuid.uuid4())[:8]

        # Create provenance hash
        prov_str = f"{word}|{synset_name}|{definition[:50]}"
        prov_hash = hashlib.sha256(prov_str.encode()).hexdigest()[:16]

        # Parse synset name for sense number
        parts = synset_name.split('.')
        sense_num = parts[2] if len(parts) > 2 else "01"

        event = {
            "id": event_id,
            "lemma": word,
            "pos": pos,
            "sense_id": synset_name,
            "sense_gloss": definition[:100] + "..." if len(definition) > 100 else definition,
            "text": context_text,
            "span": {
                "start": word_start,
                "end": word_end,
                "surface": surface
            },
            "context_window": {
                "left": context_text[:word_start].strip(),
                "right": context_text[word_end:].strip()
            },
            "cue_tokens": sense_data.get('hypernyms', [])[:2],
            "cue_type": ["definition", "wordnet"],
            "topic_tags": ["wordnet", pos] + sense_data.get('hypernyms', [])[:1],
            "source": {
                "url": f"https://wordnet.princeton.edu/",
                "domain": "wordnet.princeton.edu",
                "license": "WordNet-3.0",
                "rights_ok": True,
                "robots_ok": True
            },
            "quality": {
                "cue_strength": 0.95,
                "ambiguity_risk": 0.05,
                "toxicity_risk": 0.0,
                "boilerplate_risk": 0.0,
                "length_chars": len(context_text),
                "style": "encyclopedic" if not examples else "narrative"
            },
            "splits": {
                "holdout_lemma": False,
                "holdout_template_family": False,
                "holdout_cue_family": False
            },
            "provenance_hash": prov_hash,
            "notes": f"WordNet synset: {synset_name}, usage frequency: {sense_data.get('lemma_count', 0)}"
        }

        return event

    def scrape_word(self, word: str) -> List[Dict]:
        """Scrape all senses for a single word."""
        if word in self.processed_words:
            return []

        self.processed_words.add(word)
        events = []

        senses = self.get_word_senses(word)

        for sense in senses:
            event = self.create_event_from_sense(sense)
            events.append(event)

        return events

    def run(self, max_words: int = None) -> List[Dict]:
        """Run the scraper on all polysemous words."""
        words = self.POLYSEMOUS_WORDS[:max_words] if max_words else self.POLYSEMOUS_WORDS

        print(f"{'='*60}")
        print("WORDNET SENSE EXTRACTION")
        print(f"{'='*60}")
        print(f"Words to process: {len(words)}")

        start_time = datetime.now()

        for i, word in enumerate(words):
            word_events = self.scrape_word(word)
            self.events.extend(word_events)

            if (i + 1) % 50 == 0:
                elapsed = datetime.now() - start_time
                print(f"  [{i+1}/{len(words)}] Processed, {len(self.events)} total senses, elapsed: {elapsed}")

        # Save events
        output_file = self.output_dir / "wordnet_senses.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')

        elapsed = datetime.now() - start_time

        # Count by POS
        pos_counts = {}
        for event in self.events:
            pos = event.get('pos', 'unknown')
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        print(f"\n{'='*60}")
        print("WORDNET EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Total words processed: {len(self.processed_words)}")
        print(f"  Total senses extracted: {len(self.events)}")
        print(f"  Elapsed time: {elapsed}")
        print(f"\n  By POS:")
        for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
            print(f"    {pos}: {count}")
        print(f"\n  Saved to: {output_file}")

        return self.events


if __name__ == "__main__":
    scraper = WordNetScraper()
    events = scraper.run()
