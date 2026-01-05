#!/usr/bin/env python3
"""
Bundle Factory - Swarm-Powered v2 Bundle Generation
====================================================

Generates polysemy bundles at scale using the Fleet swarm:
- 10k+ bundles across 200-300 words
- Balanced difficulty distribution (easy/med/hard)
- Automatic validation (rejects weak sense definitions)
- Outputs v2 format ready for training

Usage:
    python bundle_factory.py --words 50 --bundles-per-word 20 --output bundles_v2.jsonl
"""

import argparse
import json
import random
import requests
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.polysemy_bundle_v2 import (
    PolysemyBundle, SenseDefinition, BundleItem, ContrastiveTarget,
    Pairings, ContrastiveTargets, Provenance, WordInfo, MinimalPair,
    SENSE_CATALOG, compute_cue_difficulty, create_bundle, save_bundles
)

# Fleet swarm base
FLEET_BASE = "http://159.203.35.45"

# Extended word list for scaling (beyond the 8 in SENSE_CATALOG)
EXTENDED_WORDS = {
    "run": ["to move quickly", "to operate a machine", "a sequence/streak", "in stockings"],
    "bear": ["large animal", "to carry/support", "to endure/tolerate"],
    "fair": ["carnival/exhibition", "just/equitable", "light-colored", "moderately good"],
    "palm": ["tropical tree", "inner hand surface", "to conceal in hand"],
    "date": ["calendar day", "romantic outing", "fruit from palm tree"],
    "fan": ["cooling device", "enthusiastic supporter", "to spread out"],
    "jam": ["fruit preserve", "traffic congestion", "to force/wedge tightly"],
    "mine": ["excavation site", "belonging to me", "explosive device"],
    "port": ["harbor/dock", "type of wine", "computer connection", "left side of ship"],
    "watch": ["timepiece", "to observe carefully", "guard duty/vigil"],
    "clip": ["fastening device", "video segment", "to cut/trim"],
    "nail": ["finger/toe covering", "metal fastener", "to accomplish perfectly"],
    "pound": ["British currency", "unit of weight", "to strike repeatedly"],
    "train": ["railway vehicle", "to teach/instruct", "trailing part of dress"],
    "pitch": ["throw/toss", "musical tone", "sales presentation", "dark sticky substance"],
    "seal": ["marine mammal", "to close tightly", "official stamp/emblem"],
    "trunk": ["tree main stem", "elephant's nose", "car storage", "torso"],
    "crane": ["tall bird", "lifting machine", "to stretch one's neck"],
    "bow": ["weapon for arrows", "front of ship", "to bend forward", "decorative knot"],
    "lead": ["heavy metal", "to guide/direct", "electrical wire", "main role"],
    "row": ["line of things", "to propel boat", "noisy argument"],
    "tear": ["to rip apart", "drop from eye"],
    "wind": ["moving air", "to turn/coil"],
    "bass": ["type of fish", "low musical frequency"],
    "dove": ["bird", "past tense of dive"],
    "live": ["to be alive", "happening now/broadcast"],
    "read": ["to interpret text", "past tense of read"],
    "close": ["to shut", "nearby"],
    "object": ["physical thing", "to protest/oppose", "goal/purpose"],
    "project": ["planned undertaking", "to extend outward", "to display on screen"],
    "record": ["best achievement", "audio disc", "to document"],
    "content": ["satisfied/happy", "what is contained"],
    "contract": ["legal agreement", "to shrink/tighten"],
    "desert": ["arid region", "to abandon"],
    "present": ["gift", "current time", "to show/display", "in attendance"],
    "produce": ["fruits/vegetables", "to create/make"],
    "refuse": ["garbage/trash", "to decline"],
    "subject": ["topic", "to cause to undergo", "under authority"],
    "suspect": ["person under suspicion", "to believe guilty"],
    "invalid": ["not valid", "sick/disabled person"],
    "minute": ["60 seconds", "extremely small"],
    "moderate": ["not extreme", "to preside over"],
    "permit": ["official document", "to allow"],
    "rebel": ["person who resists", "to resist authority"],
    "console": ["control panel", "to comfort"],
    "conflict": ["disagreement", "to clash/oppose"],
    "decrease": ["reduction", "to become less"],
    "impact": ["collision/effect", "to press firmly"],
    "insult": ["offensive remark", "to offend"],
    "perfect": ["flawless", "to make perfect"],
    # Additional polysemous words to reach 200-300 total
    "address": ["location/street", "formal speech", "to speak to"],
    "back": ["rear part", "to support", "to move backward"],
    "band": ["musical group", "strip/ring", "range of frequencies"],
    "bark": ["tree covering", "dog sound", "sailing ship"],
    "bat": ["flying mammal", "sports equipment", "to hit"],
    "bill": ["invoice", "bird's beak", "proposed law", "paper money"],
    "block": ["solid piece", "city section", "to obstruct"],
    "board": ["flat piece of wood", "committee", "to get on vehicle"],
    "book": ["written work", "to reserve", "betting record"],
    "box": ["container", "to fight", "theater seating"],
    "break": ["to fracture", "pause/rest", "opportunity"],
    "bridge": ["structure over water", "card game", "dental work"],
    "brush": ["cleaning tool", "light touch", "shrubs/bushes"],
    "cabinet": ["storage furniture", "government advisors"],
    "case": ["container", "legal matter", "instance/example"],
    "cast": ["to throw", "actors in show", "plaster mold"],
    "cell": ["prison room", "biological unit", "battery unit"],
    "change": ["to alter", "coins", "different clothes"],
    "charge": ["to attack", "electrical property", "cost/fee"],
    "check": ["to verify", "payment document", "pattern", "chess move"],
    "chip": ["small piece", "computer component", "gambling token"],
    "clear": ["transparent", "obvious", "to remove obstacles"],
    "coat": ["outer garment", "layer of paint", "animal fur"],
    "cold": ["low temperature", "illness", "unfriendly"],
    "company": ["business", "companions", "military unit"],
    "compound": ["enclosed area", "chemical mixture", "to combine"],
    "cover": ["to place over", "book jacket", "shelter"],
    "craft": ["skill/trade", "boat", "to make by hand"],
    "crash": ["collision", "computer failure", "to sleep"],
    "credit": ["belief/trust", "financial loan", "recognition"],
    "current": ["flow of water", "electrical flow", "present time"],
    "cut": ["to slice", "reduction", "version of film"],
    "deal": ["agreement", "to distribute cards", "quantity"],
    "deck": ["ship floor", "card set", "to decorate"],
    "deposit": ["to put down", "bank payment", "sediment layer"],
    "draft": ["preliminary version", "air current", "military selection"],
    "draw": ["to sketch", "to pull", "tie in game"],
    "dress": ["clothing garment", "to clothe", "to prepare food"],
    "drill": ["tool for holes", "military exercise", "fabric type"],
    "drive": ["to operate vehicle", "motivation", "computer storage"],
    "drop": ["to fall", "small amount", "decrease"],
    "duck": ["waterfowl", "to lower quickly", "strong fabric"],
    "dump": ["garbage site", "to unload", "sad mood"],
    "dust": ["fine particles", "to clean", "to sprinkle powder"],
    "express": ["to communicate", "fast service", "squeeze out"],
    "face": ["front of head", "surface", "to confront"],
    "fall": ["to descend", "autumn season", "waterfall"],
    "fast": ["quick", "to not eat", "firmly fixed"],
    "figure": ["number", "body shape", "important person"],
    "file": ["document folder", "tool for smoothing", "line of people"],
    "fine": ["penalty payment", "high quality", "thin/delicate"],
    "fire": ["flames", "to dismiss", "to shoot weapon"],
    "firm": ["company", "solid/steady", "resolute"],
    "fit": ["suitable", "healthy condition", "sudden attack"],
    "flat": ["level surface", "apartment", "deflated tire"],
    "float": ["to stay on surface", "parade vehicle", "carbonated drink"],
    "floor": ["ground surface", "story of building", "to knock down"],
    "fly": ["insect", "to travel by air", "pants zipper flap"],
    "fold": ["to bend over", "group of sheep", "geological layer"],
    "foot": ["body part", "measurement unit", "bottom of something"],
    "force": ["power/strength", "military unit", "to compel"],
    "form": ["shape", "document", "to create"],
    "frame": ["border/structure", "single image", "to falsely accuse"],
    "game": ["recreational activity", "wild animals", "willing/ready"],
    "gear": ["equipment", "mechanical cog", "to prepare"],
    "grade": ["quality level", "school year", "slope"],
    "grain": ["cereal crop", "texture pattern", "small unit"],
    "grave": ["burial site", "serious/solemn"],
    "ground": ["earth surface", "reason/basis", "electrical connection"],
    "guard": ["protector", "protective device", "to protect"],
    "gum": ["chewing substance", "mouth tissue", "tree resin"],
    "hand": ["body part", "worker", "cards dealt", "to give"],
    "handle": ["grip part", "to manage", "name/alias"],
    "hang": ["to suspend", "to spend time", "knack for something"],
    "head": ["body part", "leader", "to go toward"],
    "hide": ["animal skin", "to conceal"],
    "hit": ["to strike", "successful song", "drug dose"],
    "hold": ["to grasp", "cargo space", "to delay"],
    "hook": ["curved device", "to catch", "catchy part"],
    "host": ["party giver", "large number", "organism carrying parasite"],
    "iron": ["metal element", "pressing device", "golf club"],
    "issue": ["topic/problem", "publication edition", "to distribute"],
    "jack": ["lifting device", "playing card", "electrical plug"],
    "joint": ["connection point", "body articulation", "shared"],
    "judge": ["court official", "to evaluate", "to form opinion"],
    "key": ["lock opener", "crucial", "musical tone", "keyboard button"],
    "kick": ["to strike with foot", "thrill", "recoil"],
    "kind": ["type/sort", "caring/gentle"],
    "lap": ["sitting area", "race circuit", "to drink with tongue"],
    "last": ["final", "to continue", "shoemaker's form"],
    "launch": ["to start", "boat", "to throw"],
    "lean": ["thin", "to incline", "meat without fat"],
    "leave": ["to depart", "permission", "foliage"],
    "left": ["opposite of right", "remaining", "departed"],
    "letter": ["alphabet character", "written message"],
    "lie": ["to recline", "false statement", "position"],
    "light": ["illumination", "not heavy", "to ignite"],
    "like": ["similar to", "to enjoy", "such as"],
    "line": ["mark/stroke", "queue", "rope", "text row"],
    "list": ["enumeration", "to tilt", "to register"],
    "lock": ["security device", "hair strand", "canal section"],
    "log": ["wood piece", "record/journal", "to record"],
    "lot": ["large amount", "plot of land", "fate"],
    "mail": ["postal items", "armor", "to send"],
    "major": ["significant", "military rank", "university focus"],
    "mark": ["visible sign", "target", "currency", "to note"],
    "mass": ["large quantity", "religious service", "physics measure"],
    "master": ["expert", "original copy", "to learn thoroughly"],
    "matter": ["substance", "issue/concern", "to be important"],
    "mean": ["to signify", "unkind", "average"],
    "measure": ["to determine size", "action step", "musical unit"],
    "miss": ["to fail to hit", "young woman", "to feel absence"],
    "model": ["representation", "fashion person", "type/version"],
    "mold": ["fungus", "shaping form", "to shape"],
    "monitor": ["display screen", "to observe", "type of lizard"],
    "mount": ["to climb", "mountain", "to attach"],
    "mouse": ["rodent", "computer device", "quiet person"],
    "move": ["to change position", "action step", "to affect emotionally"],
    "net": ["mesh material", "remaining after deductions", "internet"],
    "note": ["written message", "musical tone", "to observe"],
    "novel": ["book", "new/original"],
    "order": ["arrangement", "command", "to request purchase"],
    "organ": ["body part", "musical instrument", "organization"],
    "pack": ["bundle", "group of animals", "to fill container"],
    "paper": ["writing material", "newspaper", "academic essay"],
    "park": ["recreation area", "to stop vehicle"],
    "part": ["portion", "role", "to separate"],
    "party": ["celebration", "political group", "participant"],
    "pass": ["to go by", "mountain route", "ticket", "to throw"],
    "patient": ["medical recipient", "tolerant/calm"],
    "pen": ["writing instrument", "enclosure for animals"],
    "pick": ["to choose", "pointed tool", "guitar accessory"],
    "piece": ["portion", "artwork", "firearm"],
    "pipe": ["tube", "smoking device", "musical instrument"],
    "pit": ["hole", "fruit seed", "to set against"],
    "place": ["location", "to put", "position in race"],
    "plain": ["simple", "flat land", "obvious"],
    "plant": ["vegetation", "factory", "to place in ground"],
    "plate": ["dish", "metal sheet", "tectonic section"],
    "play": ["to engage in game", "theatrical work", "to perform music"],
    "plot": ["story outline", "land parcel", "to scheme"],
    "point": ["sharp end", "location", "purpose", "to indicate"],
    "pole": ["long stick", "geographic extreme", "electrical terminal"],
    "pool": ["water body", "shared resource", "billiard game"],
    "pop": ["sudden sound", "father", "popular music", "to burst"],
    "post": ["upright support", "mail", "job position", "to publish"],
    "pot": ["container", "marijuana", "poker pool"],
    "pound": ["weight unit", "currency", "to hit repeatedly"],
    "power": ["strength", "electricity", "authority"],
    "press": ["to push", "media", "printing machine"],
    "prime": ["best quality", "first", "to prepare"],
    "print": ["to reproduce text", "pattern", "fingerprint"],
    "process": ["procedure", "legal action", "to handle"],
    "program": ["plan", "software", "broadcast show"],
    "pupil": ["student", "eye part"],
    "quarter": ["one fourth", "coin", "district", "mercy"],
    "race": ["competition", "ethnicity", "to move fast"],
    "range": ["extent", "mountain chain", "cooking stove"],
    "rank": ["position level", "to arrange", "foul-smelling"],
    "rate": ["speed", "price", "to evaluate"],
    "rest": ["relaxation", "remainder", "support"],
    "rich": ["wealthy", "full-flavored", "abundant"],
    "right": ["correct", "direction", "entitlement"],
    "ring": ["circular band", "phone sound", "boxing area"],
    "rock": ["stone", "music genre", "to sway"],
    "roll": ["to rotate", "bread type", "list"],
    "room": ["enclosed space", "opportunity/scope"],
    "root": ["plant part", "origin", "to search"],
    "round": ["circular", "ammunition", "stage of competition"],
    "rule": ["regulation", "to govern", "measuring stick"],
    "safe": ["secure", "storage box"],
    "sale": ["selling", "reduced price event"],
    "sample": ["example", "to try", "music excerpt"],
    "save": ["to rescue", "to keep", "except for"],
    "scale": ["weighing device", "fish covering", "relative size"],
    "school": ["educational institution", "group of fish", "to train"],
    "score": ["points total", "twenty", "musical notation", "to achieve"],
    "screen": ["display", "barrier", "to filter"],
    "season": ["time of year", "to add flavor", "to mature"],
    "second": ["after first", "time unit", "to support"],
    "sense": ["feeling", "meaning", "good judgment"],
    "set": ["group", "to place", "ready", "stage scenery"],
    "shade": ["shadow area", "color variation", "window covering"],
    "shape": ["form", "condition", "to mold"],
    "share": ["portion", "stock unit", "to divide"],
    "sharp": ["having edge", "exactly", "musically higher"],
    "shed": ["small building", "to lose/drop"],
    "shell": ["hard covering", "ammunition", "to remove covering"],
    "shift": ["to move", "work period", "gear change"],
    "ship": ["vessel", "to transport"],
    "shock": ["surprise", "electrical jolt", "medical condition"],
    "shoot": ["to fire weapon", "plant growth", "photo session"],
    "shop": ["store", "to purchase", "workshop"],
    "shot": ["fired projectile", "attempt", "photograph", "injection"],
    "show": ["to display", "entertainment program", "exhibition"],
    "side": ["edge", "team", "aspect"],
    "sign": ["indication", "symbol", "to write name"],
    "sink": ["basin", "to descend", "to deteriorate"],
    "site": ["location", "website"],
    "size": ["dimensions", "to assess"],
    "skip": ["to jump", "to omit", "container"],
    "slip": ["to slide", "mistake", "undergarment", "paper strip"],
    "slot": ["narrow opening", "position", "to insert"],
    "snap": ["to break", "quick photo", "clothing fastener"],
    "soil": ["earth/dirt", "to make dirty"],
    "sole": ["only", "foot bottom", "type of fish"],
    "sound": ["noise", "healthy", "body of water"],
    "space": ["area", "outer space", "gap"],
    "spare": ["extra", "to save from harm", "thin"],
    "spell": ["to write letters", "magical incantation", "period of time"],
    "spot": ["location", "stain", "to notice"],
    "spread": ["to extend", "food paste", "range"],
    "spring": ["season", "water source", "coil", "to jump"],
    "square": ["shape", "plaza", "old-fashioned person"],
    "stable": ["steady", "horse building"],
    "staff": ["employees", "stick/pole", "musical lines"],
    "stage": ["platform", "phase", "to organize"],
    "stake": ["pointed post", "interest/share", "to risk"],
    "stamp": ["postal mark", "to step heavily", "to imprint"],
    "stand": ["to be upright", "booth", "position"],
    "star": ["celestial body", "celebrity", "shape"],
    "state": ["condition", "nation/region", "to declare"],
    "stay": ["to remain", "support rope", "postponement"],
    "step": ["foot movement", "stage in process", "stair"],
    "stick": ["wooden piece", "to adhere", "to poke"],
    "stock": ["inventory", "livestock", "shares", "broth"],
    "stone": ["rock", "fruit seed", "weight unit"],
    "stop": ["to cease", "bus location", "punctuation"],
    "store": ["shop", "to keep", "supply"],
    "storm": ["weather event", "to attack", "outburst"],
    "story": ["narrative", "floor of building", "news report"],
    "strain": ["to stretch", "variety", "stress"],
    "stream": ["small river", "continuous flow", "to broadcast"],
    "stress": ["pressure", "emphasis", "to emphasize"],
    "stretch": ["to extend", "period of time", "elastic"],
    "strike": ["to hit", "work stoppage", "to occur to"],
    "string": ["cord", "series", "to thread"],
    "strip": ["narrow piece", "to remove"],
    "stroke": ["hit", "medical event", "swimming style", "to caress"],
    "study": ["to learn", "room", "research"],
    "style": ["fashion", "manner", "to design"],
    "suit": ["clothing set", "lawsuit", "card type", "to fit"],
    "surface": ["outer layer", "to appear", "superficial"],
    "swallow": ["bird", "to ingest", "to accept"],
    "swing": ["to move back and forth", "playground equipment", "music style"],
    "switch": ["to change", "electrical device", "flexible rod"],
    "table": ["furniture", "to postpone", "data arrangement"],
    "tank": ["container", "military vehicle", "to fail"],
    "tap": ["to touch lightly", "faucet", "phone surveillance"],
    "tape": ["adhesive strip", "recording medium", "to record"],
    "target": ["goal", "shooting object", "to aim at"],
    "tax": ["government levy", "to strain"],
    "temple": ["religious building", "side of head"],
    "term": ["word", "period of time", "condition"],
    "tie": ["neckwear", "to fasten", "equal score"],
    "tip": ["end point", "gratuity", "advice", "to tilt"],
    "toast": ["grilled bread", "tribute drink", "to heat"],
    "toll": ["fee", "bell sound", "to ring", "damage total"],
    "top": ["highest point", "spinning toy", "lid"],
    "touch": ["to feel", "small amount", "contact"],
    "track": ["path", "to follow", "railroad rail"],
    "trade": ["commerce", "occupation", "to exchange"],
    "trail": ["path", "to follow", "to drag behind"],
    "transfer": ["to move", "ticket", "image copy"],
    "trap": ["snare", "to catch", "drum equipment"],
    "trip": ["journey", "to stumble", "drug experience"],
    "trust": ["belief", "organization", "to rely on"],
    "turn": ["to rotate", "opportunity", "change in direction"],
    "type": ["category", "to write", "printed text"],
    "vessel": ["container", "ship", "blood tube"],
    "volume": ["book", "loudness", "quantity"],
    "wake": ["to stop sleeping", "boat trail", "vigil for dead"],
    "wave": ["water movement", "hand gesture", "pattern"],
    "well": ["water source", "healthy", "satisfactorily"],
    "will": ["determination", "legal document", "future tense"],
    "wing": ["bird limb", "building section", "to travel"],
    "wire": ["metal strand", "to send telegram", "to connect"],
    "yard": ["ground area", "measurement unit", "ship spar"],
}


def call_swarm(endpoint: str, payload: Dict, timeout: int = 90) -> Optional[Dict]:
    """Call a Fleet swarm endpoint with retry."""
    url = f"{FLEET_BASE}:8007/swarm/{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  Swarm call failed after 3 attempts: {e}")
                return None
    return None


def generate_context_batch(word: str, sense: str, gloss: str, n: int = 5) -> List[str]:
    """Generate diverse contexts for a word in a specific sense."""
    prompt = f"""Generate {n} diverse, natural sentences using the word "{word}"
in the sense of "{sense}" ({gloss}).

Requirements:
1. Each sentence should clearly demonstrate this specific meaning
2. Vary the sentence structure and context
3. Include some clear sense-indicating words (cues)
4. Length: 10-25 words each
5. Natural, grammatically correct English

Return ONLY the sentences, one per line, no numbering or bullets."""

    result = call_swarm("explore", {
        "problem": prompt,
        "num_agents": 3
    })

    if not result:
        return []

    # Parse response
    contexts = []
    response_text = str(result.get("result", result.get("explorations", "")))

    for line in response_text.split("\n"):
        line = line.strip()
        # Skip empty, numbered, or meta lines
        if not line or line[0].isdigit() or line.startswith(("-", "*", "Agent", "Here")):
            continue
        line = line.strip('"\'')
        if word.lower() in line.lower() and 15 < len(line) < 200:
            contexts.append(line)

    return contexts[:n]


def generate_bundles_batch(
    word: str,
    senses: List[str],
    num_bundles: int = 20,
    difficulty_mix: str = "balanced"
) -> List[Dict[str, Any]]:
    """
    BATCHED bundle generation: request all bundles for a word in ONE swarm call.

    This is ~10-20x more efficient than individual calls per bundle/sense.
    Returns raw bundle dicts ready for validation and conversion to PolysemyBundle.
    """
    senses_json = json.dumps(senses)

    prompt = f"""Generate {num_bundles} polysemy training bundles for the word "{word}".

SENSES: {senses_json}

Each bundle should have:
- anchor: sentence using one sense
- positive: 1-2 sentences using the SAME sense as anchor
- negative: sentences using DIFFERENT senses
- hard_negative: sentences that look similar to anchor but use a different sense

Difficulty distribution ({difficulty_mix}):
- easy: clear context cues, obvious which sense (40%)
- medium: moderate cues, some ambiguity (40%)
- hard: minimal cues, structurally similar across senses (20%)

IMPORTANT: Return a valid JSON array with this exact structure:
[
  {{
    "anchor_sense": "sense name",
    "anchor": "sentence with word in anchor_sense",
    "positives": ["sentence1 same sense", "sentence2 same sense"],
    "negatives": [
      {{"sense": "other sense", "text": "sentence using other sense"}},
      ...
    ],
    "hard_negatives": [
      {{"sense": "other sense", "text": "similar-looking but different sense", "why_hard": "explanation"}}
    ],
    "difficulty": "easy|medium|hard"
  }},
  ...
]

Return ONLY the JSON array, no markdown code blocks or explanation."""

    result = call_swarm("explore", {
        "problem": prompt,
        "num_agents": 5,  # More agents for larger task
        "timeout": 180
    }, timeout=180)

    if not result:
        return []

    # Parse JSON response
    response_text = str(result.get("result", result.get("explorations", "")))

    # Try to extract JSON from response
    bundles = []
    try:
        # Try direct parse first
        bundles = json.loads(response_text)
    except json.JSONDecodeError:
        # Look for JSON array in the text
        import re
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
        if json_match:
            try:
                bundles = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    # Validate structure
    valid_bundles = []
    for b in bundles:
        if isinstance(b, dict) and "anchor" in b and "anchor_sense" in b:
            # Ensure required fields
            b.setdefault("positives", [])
            b.setdefault("negatives", [])
            b.setdefault("hard_negatives", [])
            b.setdefault("difficulty", "medium")
            valid_bundles.append(b)

    return valid_bundles


def convert_batch_to_polysemy_bundles(
    word: str,
    raw_bundles: List[Dict[str, Any]],
    sense_definitions: List[SenseDefinition],
    start_index: int = 0,
    seed: int = 42
) -> List[PolysemyBundle]:
    """Convert raw batch response to PolysemyBundle objects."""
    polysemy_bundles = []

    # Build sense lookup
    sense_to_def = {}
    for sdef in sense_definitions:
        sense_to_def[sdef.label.lower()] = sdef
        sense_to_def[sdef.gloss.lower()] = sdef

    for idx, raw in enumerate(raw_bundles):
        try:
            # Find anchor sense definition
            anchor_sense_name = raw.get("anchor_sense", "").lower()
            anchor_def = None
            for key in sense_to_def:
                if anchor_sense_name in key or key in anchor_sense_name:
                    anchor_def = sense_to_def[key]
                    break

            if not anchor_def and sense_definitions:
                anchor_def = sense_definitions[0]

            anchor_sense_id = anchor_def.sense_id if anchor_def else f"{word}#{anchor_sense_name}"

            # Build positive contexts
            positive_contexts = []
            for pos_text in raw.get("positives", []):
                if isinstance(pos_text, str) and word.lower() in pos_text.lower():
                    positive_contexts.append((anchor_sense_id, pos_text))

            # Build negative contexts
            negative_contexts = []
            for neg in raw.get("negatives", []):
                if isinstance(neg, dict):
                    neg_sense = neg.get("sense", "").lower()
                    neg_text = neg.get("text", "")
                    if neg_text and word.lower() in neg_text.lower():
                        neg_def = None
                        for key in sense_to_def:
                            if neg_sense in key or key in neg_sense:
                                neg_def = sense_to_def[key]
                                break
                        neg_sense_id = neg_def.sense_id if neg_def else f"{word}#{neg_sense}"
                        negative_contexts.append((neg_sense_id, neg_text))

            # Build hard negative contexts
            hard_negative_contexts = []
            for hn in raw.get("hard_negatives", []):
                if isinstance(hn, dict):
                    hn_sense = hn.get("sense", "").lower()
                    hn_text = hn.get("text", "")
                    hn_why = hn.get("why_hard", "Structurally similar")
                    if hn_text and word.lower() in hn_text.lower():
                        hn_def = None
                        for key in sense_to_def:
                            if hn_sense in key or key in hn_sense:
                                hn_def = sense_to_def[key]
                                break
                        hn_sense_id = hn_def.sense_id if hn_def else f"{word}#{hn_sense}"
                        hard_negative_contexts.append((hn_sense_id, hn_text, hn_why))

            # Skip if not enough data
            if not positive_contexts and not negative_contexts:
                continue

            # Create bundle
            bundle = create_bundle(
                word=word,
                anchor_sense_id=anchor_sense_id,
                anchor_context=raw.get("anchor", ""),
                positive_contexts=positive_contexts,
                negative_contexts=negative_contexts,
                hard_negative_contexts=hard_negative_contexts,
                bundle_index=start_index + idx,
                seed=seed + start_index + idx
            )

            # Validate
            valid, reason = validate_bundle(bundle)
            if valid:
                polysemy_bundles.append(bundle)

        except Exception as e:
            print(f"  Bundle conversion failed: {e}")
            continue

    return polysemy_bundles


def generate_hard_negative(word: str, anchor_sense: str, target_sense: str,
                           anchor_context: str) -> Optional[Tuple[str, str]]:
    """Generate a hard negative: similar surface form, different sense."""
    prompt = f"""Given this sentence using "{word}" in the sense of "{anchor_sense}":
"{anchor_context}"

Create a new sentence that:
1. Uses "{word}" in the sense of "{target_sense}" instead
2. Has similar sentence structure or topic
3. Could be easily confused with the original
4. Is natural and grammatically correct

Return ONLY the new sentence (no explanation)."""

    result = call_swarm("refine", {
        "task": prompt,
        "iterations": 1
    })

    if not result:
        return None

    text = str(result.get("refined_output", result.get("result", "")))
    for line in text.split("\n"):
        line = line.strip().strip('"\'')
        if word.lower() in line.lower() and 15 < len(line) < 200:
            return (line, f"Structurally similar to anchor but uses {target_sense} sense")

    return None


def build_sense_definitions(word: str, senses: List[str]) -> List[SenseDefinition]:
    """Build SenseDefinition objects with cues for a word."""
    # Check if already in catalog
    if word in SENSE_CATALOG:
        return SENSE_CATALOG[word]

    # Generate basic definitions
    definitions = []
    for sense in senses:
        sense_id = f"{word}#{sense.replace(' ', '_')[:20]}"

        # Use swarm to generate cues
        cue_prompt = f"""For the word "{word}" in the sense of "{sense}":

List 5-8 words or short phrases that strongly indicate this meaning.
Then list 3-5 words that indicate OTHER meanings of "{word}".

Format:
CUES: word1, word2, word3, ...
ANTI_CUES: word1, word2, word3, ..."""

        result = call_swarm("explore", {"problem": cue_prompt, "num_agents": 1})

        cues = []
        anti_cues = []
        if result:
            text = str(result.get("result", ""))
            for line in text.split("\n"):
                if line.upper().startswith("CUES:"):
                    cues = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
                elif line.upper().startswith("ANTI"):
                    anti_cues = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]

        # Fallback if swarm didn't return good cues
        if len(cues) < 3:
            cues = sense.lower().split()[:3]

        definitions.append(SenseDefinition(
            sense_id=sense_id,
            label=sense.replace(" ", "_")[:20],
            gloss=sense,
            cues=cues[:8],
            anti_cues=anti_cues[:5]
        ))

    return definitions


def validate_bundle(bundle: PolysemyBundle) -> Tuple[bool, str]:
    """Validate a bundle for quality."""
    # Check minimum items
    if len(bundle.items) < 3:
        return False, "Too few items"

    # Check for anchor
    has_anchor = any(i.role == "anchor" for i in bundle.items)
    if not has_anchor:
        return False, "No anchor"

    # Check for at least one positive and one negative
    has_positive = any(i.role == "positive" for i in bundle.items)
    has_negative = any(i.role in ("negative", "hard_negative") for i in bundle.items)
    if not has_positive or not has_negative:
        return False, "Missing positive or negative"

    # Check difficulty spread (not all same difficulty)
    difficulties = [i.difficulty for i in bundle.items]
    if max(difficulties) - min(difficulties) < 0.1:
        return False, "No difficulty spread"

    return True, "OK"


def generate_bundle_for_word(
    word: str,
    senses: List[str],
    sense_definitions: List[SenseDefinition],
    bundle_index: int,
    seed: int = 42,
    contexts_per_sense: int = 3
) -> Optional[PolysemyBundle]:
    """Generate a single bundle for a word."""
    if len(senses) < 2:
        return None

    # Pick anchor sense
    anchor_sense_idx = bundle_index % len(senses)
    anchor_sense = senses[anchor_sense_idx]
    anchor_def = sense_definitions[anchor_sense_idx] if anchor_sense_idx < len(sense_definitions) else None
    anchor_sense_id = anchor_def.sense_id if anchor_def else f"{word}#{anchor_sense}"

    # Generate anchor and positive contexts
    anchor_contexts = generate_context_batch(word, anchor_sense, anchor_sense, n=2)
    if len(anchor_contexts) < 2:
        return None

    anchor_context = anchor_contexts[0]
    positive_contexts = [(anchor_sense_id, ctx) for ctx in anchor_contexts[1:]]

    # Generate negative contexts (different senses)
    negative_contexts = []
    hard_negative_contexts = []

    for i, other_sense in enumerate(senses):
        if other_sense == anchor_sense:
            continue

        other_def = sense_definitions[i] if i < len(sense_definitions) else None
        other_sense_id = other_def.sense_id if other_def else f"{word}#{other_sense}"

        # Regular negative
        neg_contexts = generate_context_batch(word, other_sense, other_sense, n=1)
        if neg_contexts:
            negative_contexts.append((other_sense_id, neg_contexts[0]))

        # Try to generate hard negative
        hard_neg = generate_hard_negative(word, anchor_sense, other_sense, anchor_context)
        if hard_neg:
            hard_negative_contexts.append((other_sense_id, hard_neg[0], hard_neg[1]))

    if not negative_contexts:
        return None

    # Create bundle
    try:
        bundle = create_bundle(
            word=word,
            anchor_sense_id=anchor_sense_id,
            anchor_context=anchor_context,
            positive_contexts=positive_contexts,
            negative_contexts=negative_contexts,
            hard_negative_contexts=hard_negative_contexts,
            bundle_index=bundle_index,
            seed=seed
        )

        # Validate
        valid, reason = validate_bundle(bundle)
        if not valid:
            print(f"  Bundle validation failed: {reason}")
            return None

        return bundle

    except Exception as e:
        print(f"  Bundle creation failed: {e}")
        return None


def run_bundle_factory_batched(
    num_words: int = 50,
    bundles_per_word: int = 20,
    output_path: Path = None,
    seed: int = 42,
    difficulty_balance: bool = True,
    max_workers: int = 4
):
    """
    OPTIMIZED factory loop using batched swarm calls.

    Instead of ~100 calls per word, uses 1-2 calls per word:
    - 1 call for sense definitions/cues (if not in catalog)
    - 1 call for all bundles for that word

    With 300 words × 1-2 calls = ~400-600 total calls instead of ~30,000.
    """
    print("=" * 70)
    print("  BUNDLE FACTORY - BATCHED Swarm Generation")
    print("=" * 70)

    random.seed(seed)

    # Combine catalog words with extended words
    all_words = {}
    for word, defs in SENSE_CATALOG.items():
        all_words[word] = [d.gloss for d in defs]  # Use gloss for natural language
    for word, senses in EXTENDED_WORDS.items():
        if word not in all_words:
            all_words[word] = senses

    # Select words to process
    word_list = list(all_words.keys())
    random.shuffle(word_list)
    selected_words = word_list[:num_words]

    print(f"\nSelected {len(selected_words)} words for generation")
    print(f"Target bundles per word: {bundles_per_word}")
    print(f"Target total: {len(selected_words) * bundles_per_word} bundles")
    print(f"Using BATCHED generation (1-2 swarm calls per word)")

    all_bundles = []
    bundle_counter = 0
    stats = {"words_processed": 0, "swarm_calls": 0, "bundles_generated": 0}

    for word_idx, word in enumerate(selected_words):
        senses = all_words[word]
        print(f"\n[{word_idx+1}/{len(selected_words)}] {word} ({len(senses)} senses)")

        # Build or get sense definitions
        if word in SENSE_CATALOG:
            sense_definitions = SENSE_CATALOG[word]
        else:
            print(f"  Building sense definitions...")
            sense_definitions = build_sense_definitions(word, senses)
            stats["swarm_calls"] += 1

        # BATCHED: Get all bundles for this word in one call
        print(f"  Requesting {bundles_per_word} bundles in single batch...")
        raw_bundles = generate_bundles_batch(
            word=word,
            senses=senses,
            num_bundles=bundles_per_word,
            difficulty_mix="balanced"
        )
        stats["swarm_calls"] += 1

        print(f"  Got {len(raw_bundles)} raw bundles from swarm")

        # Convert to PolysemyBundle objects
        word_bundles = convert_batch_to_polysemy_bundles(
            word=word,
            raw_bundles=raw_bundles,
            sense_definitions=sense_definitions,
            start_index=bundle_counter,
            seed=seed
        )

        print(f"  Validated {len(word_bundles)} bundles")
        bundle_counter += len(word_bundles)
        stats["bundles_generated"] += len(word_bundles)
        stats["words_processed"] += 1

        all_bundles.extend(word_bundles)

        # Checkpoint every 25 words
        if (word_idx + 1) % 25 == 0 and output_path:
            checkpoint_path = output_path.with_suffix(f".checkpoint_{word_idx+1}.jsonl")
            save_bundles(all_bundles, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            print(f"  Stats: {stats['bundles_generated']} bundles, {stats['swarm_calls']} swarm calls")

        # Small delay between words to avoid rate limiting
        time.sleep(1.0)

    # Difficulty balance check
    if difficulty_balance:
        difficulties = [item.difficulty for b in all_bundles for item in b.items if item.role != "anchor"]
        if difficulties:
            easy = sum(1 for d in difficulties if d < 0.33)
            med = sum(1 for d in difficulties if 0.33 <= d < 0.66)
            hard = sum(1 for d in difficulties if d >= 0.66)
            total = len(difficulties)
            print(f"\nDifficulty distribution:")
            print(f"  Easy (<0.33):  {easy:5d} ({100*easy/total:.1f}%)")
            print(f"  Med (0.33-0.66): {med:5d} ({100*med/total:.1f}%)")
            print(f"  Hard (>=0.66): {hard:5d} ({100*hard/total:.1f}%)")

    # Save final output
    if output_path:
        save_bundles(all_bundles, output_path)
        print(f"\nSaved {len(all_bundles)} bundles to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  BATCHED GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total bundles: {len(all_bundles)}")
    print(f"Unique words: {len(set(b.word.surface for b in all_bundles))}")
    print(f"Total items: {sum(len(b.items) for b in all_bundles)}")
    print(f"Total swarm calls: {stats['swarm_calls']}")
    print(f"Efficiency: {len(all_bundles) / max(1, stats['swarm_calls']):.1f} bundles/call")

    return all_bundles


def run_bundle_factory(
    num_words: int = 50,
    bundles_per_word: int = 20,
    output_path: Path = None,
    seed: int = 42,
    difficulty_balance: bool = True
):
    """Main factory loop (LEGACY - individual calls per bundle)."""
    print("=" * 70)
    print("  BUNDLE FACTORY - Swarm-Powered v2 Generation (LEGACY)")
    print("=" * 70)

    random.seed(seed)

    # Combine catalog words with extended words
    all_words = {}
    for word, defs in SENSE_CATALOG.items():
        all_words[word] = [d.label for d in defs]
    for word, senses in EXTENDED_WORDS.items():
        if word not in all_words:
            all_words[word] = senses

    # Select words to process
    word_list = list(all_words.keys())
    random.shuffle(word_list)
    selected_words = word_list[:num_words]

    print(f"\nSelected {len(selected_words)} words for generation")
    print(f"Target bundles per word: {bundles_per_word}")
    print(f"Target total: {len(selected_words) * bundles_per_word} bundles")

    all_bundles = []
    bundle_counter = 0

    for word_idx, word in enumerate(selected_words):
        senses = all_words[word]
        print(f"\n[{word_idx+1}/{len(selected_words)}] {word} ({len(senses)} senses)")

        # Build or get sense definitions
        if word in SENSE_CATALOG:
            sense_definitions = SENSE_CATALOG[word]
        else:
            print(f"  Building sense definitions...")
            sense_definitions = build_sense_definitions(word, senses)

        # Generate bundles for this word
        word_bundles = []
        for b_idx in range(bundles_per_word):
            bundle = generate_bundle_for_word(
                word=word,
                senses=senses,
                sense_definitions=sense_definitions,
                bundle_index=bundle_counter,
                seed=seed + bundle_counter
            )

            if bundle:
                word_bundles.append(bundle)
                bundle_counter += 1

            # Rate limiting
            time.sleep(0.5)

        print(f"  Generated {len(word_bundles)} bundles")
        all_bundles.extend(word_bundles)

        # Checkpoint every 10 words
        if (word_idx + 1) % 10 == 0 and output_path:
            checkpoint_path = output_path.with_suffix(f".checkpoint_{word_idx+1}.jsonl")
            save_bundles(all_bundles, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Difficulty balance check
    if difficulty_balance:
        difficulties = [item.difficulty for b in all_bundles for item in b.items if item.role != "anchor"]
        if difficulties:
            easy = sum(1 for d in difficulties if d < 0.33)
            med = sum(1 for d in difficulties if 0.33 <= d < 0.66)
            hard = sum(1 for d in difficulties if d >= 0.66)
            total = len(difficulties)
            print(f"\nDifficulty distribution:")
            print(f"  Easy (<0.33):  {easy:5d} ({100*easy/total:.1f}%)")
            print(f"  Med (0.33-0.66): {med:5d} ({100*med/total:.1f}%)")
            print(f"  Hard (>=0.66): {hard:5d} ({100*hard/total:.1f}%)")

    # Save final output
    if output_path:
        save_bundles(all_bundles, output_path)
        print(f"\nSaved {len(all_bundles)} bundles to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total bundles: {len(all_bundles)}")
    print(f"Unique words: {len(set(b.word.surface for b in all_bundles))}")
    print(f"Total items: {sum(len(b.items) for b in all_bundles)}")

    return all_bundles


def main():
    parser = argparse.ArgumentParser(description="Bundle Factory - Generate v2 polysemy bundles")
    parser.add_argument('--words', type=int, default=50, help='Number of words to process')
    parser.add_argument('--bundles-per-word', type=int, default=20, help='Bundles per word')
    parser.add_argument('--output', type=str,
                        default=f'paradigm_factory/output/bundles_v2_{datetime.now().strftime("%Y%m%d")}.jsonl',
                        help='Output JSONL path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--legacy', action='store_true',
                        help='Use legacy mode (individual calls per bundle) - SLOW')
    parser.add_argument('--batched', action='store_true', default=True,
                        help='Use batched mode (all bundles per word in one call) - FAST (default)')

    args = parser.parse_args()

    if args.legacy:
        print("Using LEGACY mode (individual swarm calls per bundle)")
        run_bundle_factory(
            num_words=args.words,
            bundles_per_word=args.bundles_per_word,
            output_path=Path(args.output),
            seed=args.seed
        )
    else:
        print("Using BATCHED mode (single swarm call per word)")
        run_bundle_factory_batched(
            num_words=args.words,
            bundles_per_word=args.bundles_per_word,
            output_path=Path(args.output),
            seed=args.seed
        )


if __name__ == "__main__":
    main()
