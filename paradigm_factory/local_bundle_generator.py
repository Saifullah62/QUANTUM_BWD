#!/usr/bin/env python3
"""
Local Bundle Generator - No External Dependencies
==================================================

Generates polysemy bundles locally using template patterns and the
extended word catalog. No Fleet swarm or LLM API needed.

Target: 275 words × 60 bundles = 16,500 bundles for dress rehearsal.
"""

import json
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.polysemy_bundle_v2 import (
    PolysemyBundle, SenseDefinition, BundleItem, ContrastiveTarget,
    Pairings, ContrastiveTargets, Provenance, WordInfo,
    save_bundles
)

# Context templates for different syntactic patterns
CONTEXT_TEMPLATES = {
    "subject": [
        "The {word} was {adj} and {verb} {adv}.",
        "A {adj} {word} {verb} near the {noun}.",
        "That {word} {verb} every {time}.",
        "The {adj} {word} seemed to {verb} {adv}.",
        "{pronoun} noticed the {word} {verb} nearby.",
    ],
    "object": [
        "{pronoun} {verb} the {word} {adv}.",
        "The {noun} {verb} a {adj} {word}.",
        "{pronoun} found the {word} to be {adj}.",
        "Everyone {verb} the {adj} {word}.",
        "The {noun} required a {adj} {word}.",
    ],
    "possessive": [
        "The {noun}'s {word} was {adj}.",
        "{pronoun} {verb} {possessive} {word} {adv}.",
        "The {adj} {word} belonged to the {noun}.",
    ],
    "prepositional": [
        "Near the {word}, the {noun} {verb}.",
        "With the {adj} {word}, {pronoun} {verb}.",
        "After the {word}, everything {verb} {adv}.",
        "Because of the {word}, the {noun} {verb}.",
    ],
}

# Fillers for templates
ADJECTIVES = {
    "easy": ["large", "small", "bright", "dark", "quick", "slow", "old", "new", "heavy", "light"],
    "medium": ["remarkable", "peculiar", "subtle", "intricate", "familiar", "unexpected"],
    "hard": ["ostensible", "ephemeral", "inchoate", "liminal", "quotidian", "ineffable"],
}

VERBS = {
    "easy": ["was", "had", "moved", "appeared", "remained", "seemed"],
    "medium": ["evolved", "transformed", "manifested", "persisted", "fluctuated"],
    "hard": ["coalesced", "attenuated", "oscillated", "bifurcated"],
}

ADVERBS = ["quickly", "slowly", "carefully", "suddenly", "gradually", "thoroughly"]
NOUNS = ["situation", "process", "mechanism", "structure", "approach", "method"]
PRONOUNS = ["She", "He", "They", "We", "I"]
POSSESSIVES = ["her", "his", "their", "our", "my"]
TIMES = ["day", "week", "moment", "occasion", "instance", "time"]

# Extended word catalog (from bundle_factory.py)
EXTENDED_WORDS = {
    "run": ["to move quickly on foot", "to operate a machine", "a sequence or streak", "in stockings (a tear)"],
    "bear": ["large wild animal", "to carry or support weight", "to endure or tolerate"],
    "fair": ["carnival or exhibition", "just and equitable", "light-colored or pale", "moderately good"],
    "palm": ["tropical tree", "inner surface of hand", "to conceal in hand"],
    "date": ["calendar day", "romantic outing", "fruit from palm tree"],
    "fan": ["cooling device", "enthusiastic supporter", "to spread out"],
    "jam": ["fruit preserve", "traffic congestion", "to force tightly"],
    "mine": ["excavation for minerals", "belonging to me", "explosive device"],
    "port": ["harbor or dock", "type of wine", "computer connection", "left side of ship"],
    "watch": ["timepiece on wrist", "to observe carefully", "guard duty or vigil"],
    "clip": ["fastening device", "video segment", "to cut or trim"],
    "nail": ["finger covering", "metal fastener", "to accomplish perfectly"],
    "pound": ["British currency", "unit of weight", "to strike repeatedly"],
    "train": ["railway vehicle", "to teach or instruct", "trailing part of dress"],
    "pitch": ["to throw", "musical tone", "sales presentation", "sticky substance"],
    "seal": ["marine mammal", "to close tightly", "official stamp"],
    "trunk": ["tree main stem", "elephant nose", "car storage", "torso"],
    "crane": ["tall bird", "lifting machine", "to stretch neck"],
    "bow": ["weapon for arrows", "front of ship", "to bend forward", "decorative knot"],
    "lead": ["heavy metal", "to guide", "electrical wire", "main role"],
    "row": ["line of things", "to propel boat", "noisy argument"],
    "tear": ["to rip apart", "drop from eye"],
    "wind": ["moving air", "to turn or coil"],
    "bass": ["type of fish", "low musical frequency"],
    "dove": ["bird of peace", "past tense of dive"],
    "live": ["to be alive", "broadcast happening now"],
    "read": ["to interpret text (present)", "past tense of read"],
    "close": ["to shut", "nearby in distance"],
    "object": ["physical thing", "to protest", "goal or purpose"],
    "project": ["planned undertaking", "to extend outward", "to display on screen"],
    "record": ["best achievement", "audio disc", "to document"],
    "content": ["satisfied or happy", "what is contained"],
    "contract": ["legal agreement", "to shrink or tighten"],
    "desert": ["arid region", "to abandon"],
    "present": ["gift", "current time", "to show or display", "in attendance"],
    "produce": ["fruits and vegetables", "to create or make"],
    "refuse": ["garbage or trash", "to decline"],
    "subject": ["topic", "to cause to undergo", "under authority"],
    "suspect": ["person under suspicion", "to believe guilty"],
    "invalid": ["not valid", "sick or disabled person"],
    "minute": ["60 seconds", "extremely small"],
    "moderate": ["not extreme", "to preside over discussion"],
    "permit": ["official document", "to allow"],
    "rebel": ["person who resists", "to resist authority"],
    "console": ["control panel", "to comfort someone"],
    "conflict": ["disagreement", "to clash or oppose"],
    "decrease": ["reduction", "to become less"],
    "impact": ["collision or effect", "to press firmly"],
    "insult": ["offensive remark", "to offend"],
    "perfect": ["flawless", "to make perfect"],
    "address": ["location or street", "formal speech", "to speak to someone"],
    "back": ["rear part", "to support", "to move backward"],
    "band": ["musical group", "strip or ring", "range of frequencies"],
    "bark": ["tree covering", "dog sound", "sailing ship"],
    "bat": ["flying mammal", "sports equipment", "to hit"],
    "bill": ["invoice", "bird beak", "proposed law", "paper money"],
    "block": ["solid piece", "city section", "to obstruct"],
    "board": ["flat piece of wood", "committee", "to get on vehicle"],
    "book": ["written work", "to reserve", "betting record"],
    "box": ["container", "to fight", "theater seating"],
    "break": ["to fracture", "pause or rest", "opportunity"],
    "bridge": ["structure over water", "card game", "dental work"],
    "brush": ["cleaning tool", "light touch", "shrubs or bushes"],
    "cabinet": ["storage furniture", "government advisors"],
    "case": ["container", "legal matter", "instance or example"],
    "cast": ["to throw", "actors in show", "plaster mold"],
    "cell": ["prison room", "biological unit", "battery unit"],
    "change": ["to alter", "coins", "different clothes"],
    "charge": ["to attack", "electrical property", "cost or fee"],
    "check": ["to verify", "payment document", "pattern", "chess move"],
    "chip": ["small piece", "computer component", "gambling token"],
    "clear": ["transparent", "obvious", "to remove obstacles"],
    "coat": ["outer garment", "layer of paint", "animal fur"],
    "cold": ["low temperature", "illness", "unfriendly attitude"],
    "company": ["business", "companions", "military unit"],
    "compound": ["enclosed area", "chemical mixture", "to combine"],
    "cover": ["to place over", "book jacket", "shelter"],
    "craft": ["skill or trade", "boat", "to make by hand"],
    "crash": ["collision", "computer failure", "to sleep informally"],
    "credit": ["belief or trust", "financial loan", "recognition"],
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
    "fine": ["penalty payment", "high quality", "thin or delicate"],
    "fire": ["flames", "to dismiss", "to shoot weapon"],
    "firm": ["company", "solid or steady", "resolute"],
    "fit": ["suitable", "healthy condition", "sudden attack"],
    "flat": ["level surface", "apartment", "deflated tire"],
    "float": ["to stay on surface", "parade vehicle", "carbonated drink"],
    "floor": ["ground surface", "story of building", "to knock down"],
    "fly": ["insect", "to travel by air", "pants zipper flap"],
    "fold": ["to bend over", "group of sheep", "geological layer"],
    "foot": ["body part", "measurement unit", "bottom of something"],
    "force": ["power or strength", "military unit", "to compel"],
    "form": ["shape", "document", "to create"],
    "frame": ["border structure", "single image", "to falsely accuse"],
    "game": ["recreational activity", "wild animals hunted", "willing or ready"],
    "gear": ["equipment", "mechanical cog", "to prepare"],
    "grade": ["quality level", "school year", "slope"],
    "grain": ["cereal crop", "texture pattern", "small unit of weight"],
    "grave": ["burial site", "serious or solemn"],
    "ground": ["earth surface", "reason or basis", "electrical connection"],
    "guard": ["protector", "protective device", "to protect"],
    "gum": ["chewing substance", "mouth tissue", "tree resin"],
    "hand": ["body part", "worker", "cards dealt", "to give"],
    "handle": ["grip part", "to manage", "name or alias"],
    "hang": ["to suspend", "to spend time", "knack for something"],
    "head": ["body part", "leader", "to go toward"],
    "hide": ["animal skin", "to conceal"],
    "hit": ["to strike", "successful song", "drug dose"],
    "hold": ["to grasp", "cargo space", "to delay"],
    "hook": ["curved device", "to catch", "catchy part of song"],
    "host": ["party giver", "large number", "organism carrying parasite"],
    "iron": ["metal element", "pressing device", "golf club"],
    "issue": ["topic or problem", "publication edition", "to distribute"],
    "jack": ["lifting device", "playing card", "electrical plug"],
    "joint": ["connection point", "body articulation", "shared"],
    "judge": ["court official", "to evaluate", "to form opinion"],
    "key": ["lock opener", "crucial", "musical tone", "keyboard button"],
    "kick": ["to strike with foot", "thrill", "recoil"],
    "kind": ["type or sort", "caring or gentle"],
    "lap": ["sitting area", "race circuit", "to drink with tongue"],
    "last": ["final", "to continue", "shoemaker form"],
    "launch": ["to start", "boat", "to throw"],
    "lean": ["thin", "to incline", "meat without fat"],
    "leave": ["to depart", "permission", "foliage"],
    "left": ["opposite of right", "remaining", "departed"],
    "letter": ["alphabet character", "written message"],
    "lie": ["to recline", "false statement", "position"],
    "light": ["illumination", "not heavy", "to ignite"],
    "like": ["similar to", "to enjoy", "such as"],
    "line": ["mark or stroke", "queue", "rope", "text row"],
    "list": ["enumeration", "to tilt", "to register"],
    "lock": ["security device", "hair strand", "canal section"],
    "log": ["wood piece", "record or journal", "to record"],
    "lot": ["large amount", "plot of land", "fate"],
    "mail": ["postal items", "armor", "to send"],
    "major": ["significant", "military rank", "university focus"],
    "mark": ["visible sign", "target", "currency", "to note"],
    "mass": ["large quantity", "religious service", "physics measure"],
    "master": ["expert", "original copy", "to learn thoroughly"],
    "matter": ["substance", "issue or concern", "to be important"],
    "mean": ["to signify", "unkind", "average"],
    "measure": ["to determine size", "action step", "musical unit"],
    "miss": ["to fail to hit", "young woman", "to feel absence"],
    "model": ["representation", "fashion person", "type or version"],
    "mold": ["fungus", "shaping form", "to shape"],
    "monitor": ["display screen", "to observe", "type of lizard"],
    "mount": ["to climb", "mountain", "to attach"],
    "mouse": ["rodent", "computer device", "quiet person"],
    "move": ["to change position", "action step", "to affect emotionally"],
    "net": ["mesh material", "remaining after deductions", "internet"],
    "note": ["written message", "musical tone", "to observe"],
    "novel": ["book", "new or original"],
    "order": ["arrangement", "command", "to request purchase"],
    "organ": ["body part", "musical instrument", "organization"],
    "pack": ["bundle", "group of animals", "to fill container"],
    "paper": ["writing material", "newspaper", "academic essay"],
    "park": ["recreation area", "to stop vehicle"],
    "part": ["portion", "role", "to separate"],
    "party": ["celebration", "political group", "participant"],
    "pass": ["to go by", "mountain route", "ticket", "to throw"],
    "patient": ["medical recipient", "tolerant or calm"],
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
    "power": ["strength", "electricity", "authority"],
    "press": ["to push", "media", "printing machine"],
    "prime": ["best quality", "first", "to prepare"],
    "print": ["to reproduce text", "pattern", "fingerprint"],
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
    "shed": ["small building", "to lose or drop"],
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
    "spring": ["season", "water source", "coil", "to jump"],
    "square": ["shape", "plaza", "old-fashioned person"],
    "stable": ["steady", "horse building"],
    "staff": ["employees", "stick or pole", "musical lines"],
    "stage": ["platform", "phase", "to organize"],
    "stake": ["pointed post", "interest or share", "to risk"],
    "stamp": ["postal mark", "to step heavily", "to imprint"],
    "stand": ["to be upright", "booth", "position"],
    "star": ["celestial body", "celebrity", "shape"],
    "state": ["condition", "nation or region", "to declare"],
    "step": ["foot movement", "stage in process", "stair"],
    "stick": ["wooden piece", "to adhere", "to poke"],
    "stock": ["inventory", "livestock", "shares", "broth"],
    "stone": ["rock", "fruit seed", "weight unit"],
    "store": ["shop", "to keep", "supply"],
    "story": ["narrative", "floor of building", "news report"],
    "strain": ["to stretch", "variety", "stress"],
    "stream": ["small river", "continuous flow", "to broadcast"],
    "stress": ["pressure", "emphasis", "to emphasize"],
    "strike": ["to hit", "work stoppage", "to occur to"],
    "string": ["cord", "series", "to thread"],
    "suit": ["clothing set", "lawsuit", "card type", "to fit"],
    "swing": ["to move back and forth", "playground equipment", "music style"],
    "switch": ["to change", "electrical device", "flexible rod"],
    "table": ["furniture", "to postpone", "data arrangement"],
    "tank": ["container", "military vehicle", "to fail"],
    "tie": ["neckwear", "to fasten", "equal score"],
    "tip": ["end point", "gratuity", "advice", "to tilt"],
    "track": ["path", "to follow", "railroad rail"],
    "trade": ["commerce", "occupation", "to exchange"],
    "trip": ["journey", "to stumble", "drug experience"],
    "trust": ["belief", "organization", "to rely on"],
    "turn": ["to rotate", "opportunity", "change in direction"],
    "type": ["category", "to write", "printed text"],
    "wave": ["water movement", "hand gesture", "pattern"],
    "well": ["water source", "healthy", "satisfactorily"],
    "will": ["determination", "legal document", "future tense marker"],
    "yard": ["ground area", "measurement unit", "ship spar"],
}


def generate_context(word: str, sense: str, difficulty: str, template_type: str = None) -> str:
    """Generate a context sentence for a word in a specific sense."""
    if template_type is None:
        template_type = random.choice(list(CONTEXT_TEMPLATES.keys()))

    template = random.choice(CONTEXT_TEMPLATES[template_type])

    # Select fillers based on difficulty
    adj = random.choice(ADJECTIVES[difficulty])
    verb = random.choice(VERBS.get(difficulty, VERBS["easy"]))
    adv = random.choice(ADVERBS)
    noun = random.choice(NOUNS)
    pronoun = random.choice(PRONOUNS)
    possessive = random.choice(POSSESSIVES)
    time = random.choice(TIMES)

    # Add sense-specific cue words
    sense_cue = sense.split()[0].lower() if sense else ""

    context = template.format(
        word=word,
        adj=adj,
        verb=verb,
        adv=adv,
        noun=noun,
        pronoun=pronoun,
        possessive=possessive,
        time=time
    )

    # Sometimes add sense-clarifying prefix
    if random.random() < 0.3:
        clarifiers = [
            f"Speaking of {sense_cue}s, ",
            f"Regarding the {sense_cue}, ",
            f"In terms of {sense_cue}, ",
        ]
        if sense_cue and len(sense_cue) > 2:
            context = random.choice(clarifiers) + context[0].lower() + context[1:]

    return context


def compute_difficulty(sense_idx: int, total_senses: int, context_complexity: str) -> float:
    """Compute difficulty score based on sense rarity and context complexity."""
    # Rarer senses (higher index) are harder
    sense_rarity = sense_idx / max(1, total_senses - 1) if total_senses > 1 else 0.5

    # Context complexity
    complexity_map = {"easy": 0.2, "medium": 0.5, "hard": 0.8}
    complexity = complexity_map.get(context_complexity, 0.5)

    # Combine
    return 0.4 * sense_rarity + 0.6 * complexity


def create_bundle(
    word: str,
    senses: List[str],
    bundle_idx: int,
    seed: int
) -> Optional[PolysemyBundle]:
    """Create a single polysemy bundle."""
    random.seed(seed + bundle_idx)

    if len(senses) < 2:
        return None

    # Pick anchor sense (rotate through senses)
    anchor_sense_idx = bundle_idx % len(senses)
    anchor_sense = senses[anchor_sense_idx]
    anchor_sense_id = f"{word}#{anchor_sense.replace(' ', '_')[:20]}"

    # Pick difficulty for this bundle
    difficulty_weights = [0.4, 0.4, 0.2]  # easy, medium, hard
    difficulty = random.choices(["easy", "medium", "hard"], difficulty_weights)[0]

    # Generate anchor context
    anchor_context = generate_context(word, anchor_sense, difficulty)

    # Build sense definitions
    sense_defs = []
    for i, sense in enumerate(senses):
        sense_id = f"{word}#{sense.replace(' ', '_')[:20]}"
        # Simple cue extraction from sense gloss
        cues = [w for w in sense.lower().split() if len(w) > 3][:5]
        anti_cues = []
        for other_sense in senses:
            if other_sense != sense:
                anti_cues.extend([w for w in other_sense.lower().split() if len(w) > 3][:2])

        sense_defs.append(SenseDefinition(
            sense_id=sense_id,
            label=sense.replace(' ', '_')[:20],
            gloss=sense,
            cues=cues[:5],
            anti_cues=anti_cues[:5]
        ))

    # Create items
    items = []
    item_counter = 0

    # Anchor
    anchor_item_id = f"{word}_{bundle_idx}_anchor"
    items.append(BundleItem(
        item_id=anchor_item_id,
        role="anchor",
        sense_id=anchor_sense_id,
        context=anchor_context,
        target=ContrastiveTarget(
            disambiguation=anchor_sense,
            same_sense_as_anchor=True
        ),
        difficulty=compute_difficulty(anchor_sense_idx, len(senses), difficulty),
        hardness=difficulty
    ))
    item_counter += 1

    positive_ids = []
    negative_ids = []
    hard_negative_ids = []

    # Positives (1-2, same sense as anchor)
    num_positives = random.randint(1, 2)
    for p in range(num_positives):
        pos_context = generate_context(word, anchor_sense, difficulty)
        pos_id = f"{word}_{bundle_idx}_pos{p}"
        positive_ids.append(pos_id)
        items.append(BundleItem(
            item_id=pos_id,
            role="positive",
            sense_id=anchor_sense_id,
            context=pos_context,
            target=ContrastiveTarget(
                disambiguation=anchor_sense,
                same_sense_as_anchor=True
            ),
            difficulty=compute_difficulty(anchor_sense_idx, len(senses), difficulty),
            hardness=difficulty
        ))
        item_counter += 1

    # Negatives (different senses)
    for neg_idx, neg_sense in enumerate(senses):
        if neg_sense == anchor_sense:
            continue

        neg_sense_id = f"{word}#{neg_sense.replace(' ', '_')[:20]}"
        neg_difficulty = random.choice(["easy", "medium"])
        neg_context = generate_context(word, neg_sense, neg_difficulty)
        neg_id = f"{word}_{bundle_idx}_neg{neg_idx}"
        negative_ids.append(neg_id)

        items.append(BundleItem(
            item_id=neg_id,
            role="negative",
            sense_id=neg_sense_id,
            context=neg_context,
            target=ContrastiveTarget(
                disambiguation=neg_sense,
                same_sense_as_anchor=False
            ),
            difficulty=compute_difficulty(neg_idx, len(senses), neg_difficulty),
            hardness=neg_difficulty
        ))
        item_counter += 1

    # Hard negative (structurally similar but different sense)
    if len(senses) > 1:
        hn_sense_idx = (anchor_sense_idx + 1) % len(senses)
        hn_sense = senses[hn_sense_idx]
        hn_sense_id = f"{word}#{hn_sense.replace(' ', '_')[:20]}"

        # Use same template type as anchor for structural similarity
        hn_context = generate_context(word, hn_sense, "hard")
        hn_id = f"{word}_{bundle_idx}_hn0"
        hard_negative_ids.append(hn_id)

        items.append(BundleItem(
            item_id=hn_id,
            role="hard_negative",
            sense_id=hn_sense_id,
            context=hn_context,
            target=ContrastiveTarget(
                disambiguation=hn_sense,
                same_sense_as_anchor=False
            ),
            difficulty=compute_difficulty(hn_sense_idx, len(senses), "hard"),
            hardness="hard",
            rationale_hint=f"Similar structure to anchor but uses '{hn_sense}' sense instead of '{anchor_sense}'"
        ))

    # Build bundle
    bundle_id = hashlib.md5(f"{word}_{bundle_idx}_{seed}".encode()).hexdigest()[:12]

    pairings = Pairings(
        anchor_item_id=anchor_item_id,
        positives=positive_ids,
        negatives=negative_ids,
        hard_negatives=hard_negative_ids
    )

    contrastive_targets = ContrastiveTargets(
        expected_phase_relation={
            "anchor_positive": "aligned",
            "anchor_negative": "separated",
            "anchor_hard_negative": "separated"
        },
        margin={
            "positive": 0.05,
            "negative": 0.05,
            "hard_negative": 0.03
        }
    )

    provenance = Provenance(
        generator="local_bundle_generator_v1",
        source="template_synthesis",
        seed=seed + bundle_idx,
        timestamp=datetime.now().isoformat(),
        quality_flags=["synthetic", difficulty]
    )

    return PolysemyBundle(
        schema_version="2.0",
        record_type="polysemy_bundle",
        bundle_id=bundle_id,
        paradigm="polysemy",
        word=WordInfo(surface=word, lemma=word),
        sense_catalog=sense_defs,
        items=items,
        pairings=pairings,
        contrastive_targets=contrastive_targets,
        provenance=provenance
    )


def generate_dress_rehearsal_bundles(
    num_words: int = 275,
    bundles_per_word: int = 60,
    output_path: Path = None,
    seed: int = 42
) -> List[PolysemyBundle]:
    """Generate bundles for dress rehearsal."""
    print("=" * 70)
    print("  LOCAL BUNDLE GENERATOR - Dress Rehearsal")
    print("=" * 70)

    random.seed(seed)

    # Get words
    word_list = list(EXTENDED_WORDS.keys())
    random.shuffle(word_list)
    selected_words = word_list[:num_words]

    target_total = num_words * bundles_per_word
    print(f"\nSelected {len(selected_words)} words")
    print(f"Bundles per word: {bundles_per_word}")
    print(f"Target total: {target_total} bundles")

    all_bundles = []
    bundle_counter = 0

    for word_idx, word in enumerate(selected_words):
        senses = EXTENDED_WORDS[word]

        if (word_idx + 1) % 25 == 0:
            print(f"  [{word_idx+1}/{len(selected_words)}] Processing {word}... ({len(all_bundles)} bundles so far)")

        for b in range(bundles_per_word):
            bundle = create_bundle(word, senses, bundle_counter, seed)
            if bundle:
                all_bundles.append(bundle)
            bundle_counter += 1

        # Checkpoint every 50 words
        if (word_idx + 1) % 50 == 0 and output_path:
            checkpoint_path = output_path.with_suffix(f".checkpoint_{word_idx+1}.jsonl")
            save_bundles(all_bundles, checkpoint_path)
            print(f"  Checkpoint: {checkpoint_path} ({len(all_bundles)} bundles)")

    # Difficulty distribution
    difficulties = [item.difficulty for b in all_bundles for item in b.items if item.role != "anchor"]
    if difficulties:
        easy = sum(1 for d in difficulties if d < 0.33)
        med = sum(1 for d in difficulties if 0.33 <= d < 0.66)
        hard = sum(1 for d in difficulties if d >= 0.66)
        total = len(difficulties)
        print(f"\nDifficulty distribution:")
        print(f"  Easy (<0.33):    {easy:6d} ({100*easy/total:.1f}%)")
        print(f"  Medium (0.33-0.66): {med:6d} ({100*med/total:.1f}%)")
        print(f"  Hard (>=0.66):   {hard:6d} ({100*hard/total:.1f}%)")

    # Save final
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
    import argparse
    parser = argparse.ArgumentParser(description="Local Bundle Generator")
    parser.add_argument('--words', type=int, default=275, help='Number of words')
    parser.add_argument('--bundles-per-word', type=int, default=60, help='Bundles per word')
    parser.add_argument('--output', type=str,
                        default='paradigm_factory/output/dress_rehearsal_bundles.jsonl')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    generate_dress_rehearsal_bundles(
        num_words=args.words,
        bundles_per_word=args.bundles_per_word,
        output_path=Path(args.output),
        seed=args.seed
    )


if __name__ == "__main__":
    main()
