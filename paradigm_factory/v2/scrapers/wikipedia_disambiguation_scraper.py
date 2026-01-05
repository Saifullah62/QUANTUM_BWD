"""
Wikipedia Disambiguation Page Scraper
======================================

Scrapes Wikipedia disambiguation pages to collect polysemy examples.
These pages explicitly list different meanings of the same word.
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
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class WikipediaDisambiguationScraper:
    """Scrape Wikipedia disambiguation pages for polysemy data."""

    BASE_URL = "https://en.wikipedia.org/w/api.php"

    # Words that have disambiguation pages
    DISAMBIGUATION_WORDS = [
        # Core polysemous words
        'bank', 'spring', 'run', 'light', 'draft', 'set', 'play', 'match',
        'cell', 'note', 'bar', 'scale', 'band', 'pitch', 'court',

        # Extended list - common disambiguation pages
        'ace', 'action', 'address', 'age', 'agent', 'aim', 'air', 'alarm',
        'alpha', 'anchor', 'angle', 'ant', 'apple', 'arc', 'arch', 'arm',
        'arrow', 'atlas', 'attack', 'aurora', 'axis', 'back', 'badge', 'ball',
        'base', 'basin', 'bass', 'bat', 'bay', 'beam', 'bear', 'beat', 'bell',
        'belt', 'bend', 'beta', 'bit', 'blade', 'blast', 'block', 'blow',
        'board', 'bolt', 'bond', 'bone', 'book', 'boom', 'boot', 'bow', 'box',
        'branch', 'brand', 'break', 'bridge', 'brush', 'buck', 'buffer', 'bug',
        'bull', 'bump', 'burn', 'burst', 'bus', 'bush', 'buzz', 'cab', 'cage',
        'cake', 'call', 'cam', 'camp', 'can', 'cap', 'cape', 'card', 'cargo',
        'case', 'cast', 'cat', 'catch', 'cave', 'chain', 'chamber', 'chance',
        'change', 'channel', 'charge', 'chart', 'chase', 'check', 'chip',
        'circle', 'circuit', 'city', 'claim', 'class', 'claw', 'clay', 'clear',
        'click', 'cliff', 'climb', 'clip', 'clock', 'close', 'cloud', 'club',
        'coach', 'coal', 'coast', 'coat', 'code', 'coil', 'cold', 'collar',
        'column', 'comb', 'command', 'compact', 'compass', 'compound', 'cone',
        'conflict', 'connection', 'console', 'contact', 'content', 'contract',
        'control', 'cook', 'cool', 'copy', 'cord', 'core', 'cork', 'corn',
        'corner', 'count', 'counter', 'country', 'course', 'cover', 'crack',
        'craft', 'crane', 'crash', 'cream', 'credit', 'creek', 'crew', 'cross',
        'crown', 'crush', 'crystal', 'cube', 'cup', 'current', 'curve', 'cut',
        'cycle', 'dam', 'dance', 'dark', 'dash', 'data', 'date', 'dawn', 'deal',
        'deck', 'deep', 'degree', 'delta', 'den', 'desert', 'design', 'desk',
        'diamond', 'dice', 'die', 'digit', 'dimension', 'dip', 'disc', 'disk',
        'display', 'dive', 'dock', 'dodge', 'dog', 'dome', 'door', 'dot',
        'double', 'dove', 'down', 'draft', 'dragon', 'drain', 'draw', 'dream',
        'dress', 'drift', 'drill', 'drink', 'drive', 'drop', 'drum', 'duck',
        'dump', 'dust', 'eagle', 'ear', 'earth', 'east', 'echo', 'eclipse',
        'edge', 'effect', 'egg', 'element', 'empire', 'end', 'engine', 'equal',
        'error', 'escape', 'event', 'exchange', 'express', 'eye', 'face',
        'factor', 'fair', 'falcon', 'fall', 'fan', 'farm', 'fast', 'fate',
        'fault', 'fear', 'feature', 'feed', 'field', 'figure', 'file', 'film',
        'filter', 'final', 'fine', 'finger', 'fire', 'firm', 'first', 'fish',
        'fist', 'fit', 'fix', 'flag', 'flame', 'flash', 'flat', 'fleet',
        'flesh', 'flight', 'flip', 'float', 'flood', 'floor', 'flow', 'flower',
        'fly', 'focus', 'fog', 'fold', 'folk', 'food', 'foot', 'force', 'ford',
        'forest', 'forge', 'fork', 'form', 'fort', 'fortune', 'forward',
        'foundation', 'fountain', 'fox', 'frame', 'free', 'freeze', 'fresh',
        'front', 'frost', 'fruit', 'fuel', 'function', 'fund', 'fury', 'fuse',
        'future', 'gain', 'galaxy', 'game', 'gap', 'garden', 'gas', 'gate',
        'gear', 'gem', 'gene', 'ghost', 'giant', 'gift', 'glass', 'globe',
        'glory', 'glow', 'goal', 'gold', 'golf', 'good', 'grace', 'grade',
        'grain', 'grand', 'grant', 'grape', 'graph', 'grass', 'grave', 'gray',
        'green', 'grid', 'grip', 'ground', 'group', 'grove', 'growth', 'guard',
        'guess', 'guide', 'gulf', 'gun', 'hacker', 'hail', 'hair', 'half',
        'hall', 'hammer', 'hand', 'handle', 'hang', 'harbor', 'hard', 'harm',
        'harvest', 'hatch', 'hawk', 'head', 'heart', 'heat', 'heaven', 'hedge',
        'heel', 'height', 'hell', 'hero', 'hide', 'high', 'hill', 'hinge',
        'hit', 'hive', 'hold', 'hole', 'hollow', 'home', 'honey', 'hood',
        'hook', 'hope', 'horizon', 'horn', 'horse', 'host', 'hot', 'hour',
        'house', 'hull', 'hunt', 'hurricane', 'ice', 'icon', 'idea', 'image',
        'impact', 'index', 'input', 'insight', 'instance', 'interest', 'iron',
        'island', 'issue', 'item', 'ivy', 'jack', 'jacket', 'jade', 'jam',
        'jar', 'jaw', 'jazz', 'jet', 'jewel', 'job', 'joint', 'joke', 'journal',
        'journey', 'joy', 'judge', 'juice', 'jump', 'jungle', 'junior', 'jury',
        'justice', 'key', 'kick', 'kid', 'kill', 'kind', 'king', 'kiss', 'kit',
        'kitchen', 'kite', 'knee', 'knife', 'knight', 'knock', 'knot', 'label',
        'lab', 'lake', 'lamp', 'land', 'lane', 'lap', 'laser', 'last', 'latch',
        'launch', 'law', 'layer', 'lead', 'leaf', 'league', 'lean', 'leap',
        'learn', 'left', 'leg', 'legend', 'lemon', 'lens', 'leopard', 'lesson',
        'letter', 'level', 'lever', 'liberty', 'license', 'lid', 'lie', 'life',
        'lift', 'light', 'lightning', 'lime', 'limit', 'line', 'link', 'lion',
        'lip', 'list', 'live', 'load', 'loan', 'local', 'lock', 'log', 'logic',
        'long', 'loop', 'lord', 'loss', 'lot', 'love', 'low', 'luck', 'lunch',
        'lung', 'machine', 'magic', 'magnet', 'mail', 'main', 'major', 'make',
        'mall', 'man', 'map', 'marble', 'march', 'margin', 'mark', 'market',
        'mars', 'mask', 'mass', 'master', 'match', 'matrix', 'matter', 'max',
        'maze', 'meal', 'mean', 'measure', 'meat', 'medal', 'media', 'medium',
        'meet', 'melt', 'member', 'memory', 'mentor', 'menu', 'mercy', 'merge',
        'merit', 'mesh', 'message', 'metal', 'method', 'micro', 'middle',
        'might', 'mile', 'milk', 'mill', 'mind', 'mine', 'mint', 'minute',
        'mirror', 'miss', 'mission', 'mist', 'mix', 'mobile', 'mock', 'mode',
        'model', 'module', 'mold', 'moment', 'monitor', 'monkey', 'moon',
        'moral', 'morning', 'morph', 'mortar', 'motion', 'motor', 'mount',
        'mouse', 'mouth', 'move', 'mud', 'mule', 'muscle', 'music', 'nail',
        'name', 'native', 'nature', 'navy', 'neck', 'needle', 'nerve', 'nest',
        'net', 'network', 'neutral', 'news', 'night', 'node', 'noise', 'noon',
        'normal', 'north', 'nose', 'note', 'notice', 'novel', 'nucleus',
        'number', 'nurse', 'nut', 'oak', 'object', 'ocean', 'odd', 'offer',
        'office', 'oil', 'olive', 'omega', 'open', 'opera', 'option', 'oracle',
        'orange', 'orbit', 'order', 'organ', 'origin', 'output', 'outside',
        'oval', 'oven', 'over', 'owl', 'pack', 'pad', 'page', 'pain', 'paint',
        'pair', 'palace', 'palm', 'pan', 'panel', 'panic', 'paper', 'parade',
        'parallel', 'park', 'part', 'party', 'pass', 'passage', 'passion',
        'past', 'paste', 'patch', 'path', 'patient', 'pattern', 'pause', 'pay',
        'peace', 'peak', 'pearl', 'pen', 'penalty', 'penny', 'pentagon',
        'pepper', 'perfect', 'period', 'permit', 'person', 'pet', 'phase',
        'phoenix', 'phone', 'photo', 'piano', 'pick', 'picture', 'piece',
        'pier', 'pig', 'pike', 'pile', 'pilot', 'pin', 'pine', 'pink', 'pipe',
        'pit', 'pitch', 'pixel', 'pizza', 'place', 'plain', 'plan', 'plane',
        'planet', 'plant', 'plasma', 'plastic', 'plate', 'platform', 'play',
        'plaza', 'plug', 'plum', 'plus', 'pocket', 'point', 'poison', 'poker',
        'polar', 'pole', 'police', 'polish', 'poll', 'pond', 'pool', 'pop',
        'port', 'portal', 'pose', 'position', 'post', 'pot', 'potato', 'pound',
        'pour', 'powder', 'power', 'practice', 'praise', 'prayer', 'premium',
        'present', 'preserve', 'press', 'pressure', 'price', 'pride', 'priest',
        'prime', 'prince', 'princess', 'principal', 'print', 'prison', 'prize',
        'probe', 'problem', 'process', 'produce', 'profile', 'profit',
        'program', 'progress', 'project', 'promise', 'proof', 'property',
        'prophet', 'proposal', 'prospect', 'protect', 'protein', 'protest',
        'proud', 'prove', 'provision', 'public', 'pulse', 'pump', 'punch',
        'pupil', 'puppet', 'purple', 'purpose', 'push', 'puzzle', 'pyramid',
        'quality', 'quarter', 'queen', 'quest', 'question', 'quick', 'quiet',
        'rabbit', 'race', 'rack', 'radar', 'radical', 'radio', 'rage', 'raid',
        'rail', 'rain', 'rainbow', 'raise', 'ram', 'ranch', 'random', 'range',
        'rank', 'rapid', 'rare', 'rate', 'ratio', 'raw', 'ray', 'reach',
        'react', 'read', 'real', 'reason', 'rebel', 'recall', 'record', 'red',
        'reef', 'reference', 'reflect', 'reform', 'refresh', 'refuge',
        'register', 'regular', 'reign', 'relay', 'release', 'relief',
        'religion', 'remote', 'rent', 'repair', 'repeat', 'reply', 'report',
        'republic', 'request', 'rescue', 'reserve', 'reset', 'resist',
        'resolution', 'resort', 'resource', 'response', 'rest', 'restore',
        'result', 'return', 'reveal', 'revenge', 'reverse', 'review',
        'revolution', 'reward', 'rhythm', 'ribbon', 'rice', 'rich', 'ride',
        'ridge', 'rifle', 'right', 'ring', 'riot', 'rise', 'risk', 'rival',
        'river', 'road', 'roar', 'rob', 'robot', 'rock', 'rocket', 'rod',
        'role', 'roll', 'romance', 'roof', 'room', 'root', 'rope', 'rose',
        'rough', 'round', 'route', 'row', 'royal', 'rubber', 'ruby', 'ruin',
        'rule', 'run', 'rush', 'rust', 'sacred', 'sacrifice', 'sad', 'safe',
        'sail', 'saint', 'sake', 'salt', 'sample', 'sand', 'satellite', 'sauce',
        'save', 'saw', 'scale', 'scan', 'scandal', 'scar', 'scene', 'scent',
        'schedule', 'scheme', 'school', 'science', 'scope', 'score', 'scout',
        'scratch', 'scream', 'screen', 'script', 'scroll', 'sea', 'seal',
        'search', 'season', 'seat', 'second', 'secret', 'section', 'sector',
        'secure', 'seed', 'segment', 'select', 'self', 'sell', 'sense',
        'sentence', 'sequence', 'series', 'serve', 'service', 'session', 'set',
        'settle', 'setup', 'seven', 'shade', 'shadow', 'shaft', 'shake',
        'shame', 'shape', 'share', 'shark', 'sharp', 'sheep', 'sheet', 'shelf',
        'shell', 'shelter', 'shield', 'shift', 'shine', 'ship', 'shirt',
        'shock', 'shoe', 'shoot', 'shop', 'shore', 'short', 'shot', 'shoulder',
        'show', 'shower', 'shrub', 'shut', 'sick', 'side', 'sight', 'sign',
        'signal', 'silence', 'silk', 'silver', 'simple', 'sin', 'single',
        'sink', 'sir', 'sister', 'site', 'size', 'skeleton', 'sketch', 'skill',
        'skin', 'skip', 'skull', 'sky', 'slam', 'slang', 'slave', 'sleep',
        'slice', 'slide', 'slim', 'slip', 'slope', 'slot', 'slow', 'small',
        'smart', 'smell', 'smile', 'smoke', 'smooth', 'snake', 'snap', 'snow',
        'soap', 'social', 'socket', 'soft', 'software', 'soil', 'solar',
        'soldier', 'solid', 'solution', 'son', 'song', 'sort', 'soul', 'sound',
        'source', 'south', 'space', 'span', 'spare', 'spark', 'speak',
        'speaker', 'special', 'spectrum', 'speech', 'speed', 'spell', 'spend',
        'sphere', 'spider', 'spike', 'spin', 'spirit', 'splash', 'split',
        'spoke', 'sponsor', 'sport', 'spot', 'spray', 'spread', 'spring',
        'spy', 'square', 'squeeze', 'stable', 'stack', 'staff', 'stage',
        'stake', 'stamp', 'stand', 'standard', 'star', 'start', 'state',
        'station', 'status', 'stay', 'steady', 'steam', 'steel', 'steep',
        'stem', 'step', 'stick', 'still', 'stock', 'stomach', 'stone', 'stop',
        'store', 'storm', 'story', 'straight', 'strain', 'strange', 'stranger',
        'strategy', 'straw', 'stream', 'street', 'strength', 'stress',
        'stretch', 'strike', 'string', 'strip', 'stroke', 'strong', 'structure',
        'struggle', 'student', 'studio', 'study', 'stuff', 'style', 'subject',
        'submit', 'substance', 'success', 'sudden', 'suffer', 'sugar', 'suit',
        'sum', 'summer', 'summit', 'sun', 'super', 'supply', 'support',
        'suppose', 'surface', 'surge', 'surplus', 'surprise', 'surrender',
        'surround', 'survey', 'survival', 'suspect', 'suspend', 'swallow',
        'swamp', 'swap', 'swear', 'sweat', 'sweep', 'sweet', 'swell', 'swift',
        'swim', 'swing', 'switch', 'sword', 'symbol', 'symptom', 'system',
        'table', 'tablet', 'tackle', 'tag', 'tail', 'take', 'tale', 'talent',
        'talk', 'tall', 'tank', 'tap', 'tape', 'target', 'task', 'taste',
        'tax', 'tea', 'teach', 'team', 'tear', 'tech', 'technique', 'teen',
        'telephone', 'temple', 'tempo', 'tender', 'tennis', 'tension', 'tent',
        'term', 'terminal', 'terms', 'terrain', 'terror', 'test', 'text',
        'texture', 'thank', 'theater', 'theme', 'theory', 'therapy', 'thick',
        'thin', 'thing', 'think', 'third', 'thought', 'thread', 'threat',
        'three', 'throat', 'throne', 'throw', 'thrust', 'thumb', 'thunder',
        'tick', 'ticket', 'tide', 'tie', 'tiger', 'tight', 'tile', 'till',
        'timber', 'time', 'tin', 'tip', 'tire', 'tissue', 'title', 'toast',
        'tobacco', 'today', 'toe', 'token', 'toll', 'tomato', 'tomb', 'tone',
        'tongue', 'tool', 'tooth', 'top', 'topic', 'torch', 'tornado', 'total',
        'touch', 'tough', 'tour', 'tournament', 'tower', 'town', 'toy', 'trace',
        'track', 'trade', 'tradition', 'traffic', 'trail', 'train', 'trait',
        'transfer', 'transform', 'transit', 'transition', 'translate', 'trap',
        'trash', 'travel', 'treasure', 'treat', 'treatment', 'treaty', 'tree',
        'trend', 'trial', 'triangle', 'tribe', 'tribute', 'trick', 'trigger',
        'trim', 'trio', 'trip', 'triumph', 'troop', 'trophy', 'trouble',
        'truck', 'true', 'trump', 'trunk', 'trust', 'truth', 'tube', 'tumor',
        'tune', 'tunnel', 'turkey', 'turn', 'turtle', 'tutor', 'twin', 'twist',
        'type', 'uncle', 'underground', 'understand', 'uniform', 'union',
        'unique', 'unit', 'unity', 'universe', 'unknown', 'update', 'upgrade',
        'upper', 'urban', 'urge', 'usage', 'use', 'user', 'usual', 'utility',
        'vacation', 'valley', 'value', 'vampire', 'van', 'vanilla', 'variable',
        'variety', 'various', 'vase', 'vast', 'vector', 'vegetable', 'vehicle',
        'vein', 'velocity', 'velvet', 'vendor', 'venture', 'venue', 'verb',
        'verdict', 'version', 'verse', 'vertical', 'vessel', 'veteran',
        'vibration', 'vice', 'victim', 'victory', 'video', 'view', 'village',
        'vine', 'vintage', 'violin', 'virtual', 'virtue', 'virus', 'vision',
        'visit', 'visual', 'vital', 'vivid', 'vocabulary', 'voice', 'volcano',
        'volume', 'volunteer', 'vote', 'voyage', 'wage', 'wagon', 'wait',
        'wake', 'walk', 'wall', 'wallet', 'wander', 'want', 'war', 'ward',
        'warm', 'warn', 'warrant', 'warrior', 'wash', 'waste', 'watch', 'water',
        'wave', 'wax', 'way', 'weak', 'wealth', 'weapon', 'wear', 'weather',
        'web', 'wedding', 'weed', 'week', 'weekend', 'weight', 'weird',
        'welcome', 'welfare', 'well', 'west', 'wet', 'whale', 'wheat', 'wheel',
        'whistle', 'white', 'whole', 'wide', 'wife', 'wild', 'will', 'win',
        'wind', 'window', 'wine', 'wing', 'winner', 'winter', 'wire', 'wisdom',
        'wise', 'wish', 'witch', 'witness', 'wizard', 'wolf', 'woman', 'wonder',
        'wood', 'wool', 'word', 'work', 'worker', 'workshop', 'world', 'worm',
        'worry', 'worship', 'worst', 'worth', 'wound', 'wrap', 'wreck', 'write',
        'wrong', 'yard', 'year', 'yellow', 'yesterday', 'yield', 'young',
        'youth', 'zebra', 'zero', 'zone', 'zoo'
    ]

    def __init__(self, output_dir: Path = None, rate_limit: float = 0.5):
        self.output_dir = output_dir or Path("paradigm_factory/v2/raw_events")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'QLLM-BundleScraper/2.0 (research project; polysemy data collection)'
        })

        self.seen_contexts = set()
        self.events: List[Dict] = []
        self.stats = defaultdict(int)

    def log(self, msg: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def api_request(self, params: Dict) -> Optional[Dict]:
        """Make Wikipedia API request."""
        params['format'] = 'json'
        try:
            time.sleep(self.rate_limit)
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"  API error: {e}")
            return None

    def get_disambiguation_page(self, word: str) -> Optional[str]:
        """Get disambiguation page content for a word."""
        # Try disambiguation page first
        params = {
            'action': 'query',
            'titles': f"{word.title()} (disambiguation)",
            'prop': 'extracts',
            'explaintext': True,
        }

        result = self.api_request(params)
        if result:
            pages = result.get('query', {}).get('pages', {})
            for page in pages.values():
                if 'extract' in page and len(page['extract']) > 100:
                    return page['extract']

        # Try regular article with disambiguation
        params['titles'] = word.title()
        result = self.api_request(params)
        if result:
            pages = result.get('query', {}).get('pages', {})
            for page in pages.values():
                if 'extract' in page:
                    return page['extract']

        return None

    def search_for_word(self, word: str, limit: int = 10) -> List[str]:
        """Search for articles mentioning the word."""
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': word,
            'srlimit': limit,
        }

        result = self.api_request(params)
        if not result:
            return []

        return [item['title'] for item in result.get('query', {}).get('search', [])]

    def get_article_extract(self, title: str) -> Optional[str]:
        """Get article extract."""
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain',
        }

        result = self.api_request(params)
        if not result:
            return None

        pages = result.get('query', {}).get('pages', {})
        for page in pages.values():
            return page.get('extract', '')

        return None

    def extract_sentences(self, text: str, word: str) -> List[str]:
        """Extract sentences containing the word."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        matching = []

        pattern = rf'\b{re.escape(word)}s?\b'

        for sent in sentences:
            if re.search(pattern, sent, re.IGNORECASE):
                sent = sent.strip()
                sent = re.sub(r'\s+', ' ', sent)
                words = sent.split()
                if 12 <= len(words) <= 50:
                    matching.append(sent)

        return matching

    def extract_sense_labels(self, text: str, word: str) -> List[Tuple[str, str]]:
        """Extract sense labels and descriptions from disambiguation page."""
        senses = []

        # Pattern for disambiguation entries like "Bank (geography), ..."
        pattern = rf'{re.escape(word.title())}\s+\(([^)]+)\)[,\s]*([^.]+\.)?'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            label = match.group(1).strip()
            desc = match.group(2).strip() if match.group(2) else ""
            senses.append((label, desc))

        # Also extract from "may refer to:" patterns
        refer_pattern = rf'{re.escape(word)}.*?may refer to:(.+?)(?=\n\n|\Z)'
        match = re.search(refer_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            items = match.group(1).split('\n')
            for item in items:
                item = item.strip()
                if item and len(item) > 5:
                    # Extract label from parentheses or first few words
                    label_match = re.search(r'\(([^)]+)\)', item)
                    if label_match:
                        senses.append((label_match.group(1), item[:100]))

        return senses[:10]  # Limit to 10 senses

    def create_event(
        self,
        word: str,
        text: str,
        sense_label: str,
        gloss: str,
        source_title: str
    ) -> Optional[Dict]:
        """Create v2.1 event."""
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

        source_url = f"https://en.wikipedia.org/wiki/{source_title.replace(' ', '_')}"

        return {
            "id": str(uuid.uuid4()),
            "lemma": word,
            "pos": "noun",
            "sense_id": f"{word}#{sense_label.lower().replace(' ', '_')}",
            "sense_gloss": gloss[:200] if gloss else f"{word} ({sense_label})",
            "text": text,
            "span": {"start": start, "end": end, "surface": surface},
            "context_window": {"left": ' '.join(left), "right": ' '.join(right)},
            "cue_tokens": cue_tokens,
            "cue_type": ["context"] * len(cue_tokens),
            "topic_tags": ["wikipedia", sense_label.lower()],
            "source": {
                "url": source_url,
                "domain": "wikipedia.org",
                "license": "CC-BY-SA-3.0",
                "rights_ok": True,
                "robots_ok": True
            },
            "quality": {
                "cue_strength": min(1.0, len(cue_tokens) * 0.12),
                "ambiguity_risk": 0.2,
                "toxicity_risk": 0.0,
                "boilerplate_risk": 0.05,
                "length_chars": len(text),
                "style": "encyclopedic"
            },
            "splits": {
                "holdout_lemma": False,
                "holdout_template_family": False,
                "holdout_cue_family": False
            },
            "provenance_hash": hashlib.sha256(f"{text}|{source_url}|{start}:{end}".encode()).hexdigest(),
            "notes": ""
        }

    def scrape_word(self, word: str) -> int:
        """Scrape disambiguation data for a single word."""
        count = 0

        # Get disambiguation page
        disam_text = self.get_disambiguation_page(word)
        if disam_text:
            # Extract sense labels
            senses = self.extract_sense_labels(disam_text, word)
            self.stats['senses_found'] += len(senses)

            # Get examples from disambiguation page itself
            sentences = self.extract_sentences(disam_text, word)
            for sent in sentences[:3]:
                # Assign to first available sense
                sense_label = senses[0][0] if senses else "general"
                gloss = senses[0][1] if senses else ""
                event = self.create_event(word, sent, sense_label, gloss, f"{word.title()}_disambiguation")
                if event:
                    self.events.append(event)
                    count += 1

        # Search for articles using the word
        articles = self.search_for_word(word, limit=5)
        for title in articles:
            text = self.get_article_extract(title)
            if not text:
                continue

            sentences = self.extract_sentences(text, word)
            for i, sent in enumerate(sentences[:2]):
                # Infer sense from article title
                sense_label = title.split('(')[0].strip() if '(' in title else "context"
                event = self.create_event(word, sent, sense_label, "", title)
                if event:
                    self.events.append(event)
                    count += 1

        return count

    def scrape_all(self) -> int:
        """Scrape all disambiguation words."""
        self.log("=" * 60)
        self.log("WIKIPEDIA DISAMBIGUATION SCRAPER")
        self.log("=" * 60)
        self.log(f"Words to scrape: {len(self.DISAMBIGUATION_WORDS)}")

        total = 0
        for i, word in enumerate(self.DISAMBIGUATION_WORDS):
            if i % 50 == 0:
                self.log(f"Progress: {i}/{len(self.DISAMBIGUATION_WORDS)} words, {total} events collected")

            count = self.scrape_word(word)
            total += count
            self.stats['words_processed'] += 1

            if count > 0:
                self.stats['words_with_data'] += 1

        return total

    def save_events(self, filename: str = "wikipedia_disambiguation_v21.jsonl") -> Path:
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
        print(f"  Words processed: {self.stats['words_processed']}")
        print(f"  Words with data: {self.stats['words_with_data']}")
        print(f"  Senses found: {self.stats['senses_found']}")
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
    print("WIKIPEDIA DISAMBIGUATION SCRAPER")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    scraper = WikipediaDisambiguationScraper(rate_limit=0.5)

    try:
        total = scraper.scrape_all()
        print(f"\nTotal collected: {total}")
    except KeyboardInterrupt:
        print("\nInterrupted. Saving...")

    output = scraper.save_events()
    scraper.print_stats()

    print(f"\nOutput: {output}")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
