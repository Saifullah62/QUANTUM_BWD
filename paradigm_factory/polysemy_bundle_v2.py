#!/usr/bin/env python3
"""
Polysemy Bundle v2 Schema
=========================

Richer bundle format that explicitly declares contrastive geometry:
- Anchor + positives + negatives (easy/hard) as one coherent unit
- Sense catalog with glosses and cues
- Stable IDs for tracking
- Difficulty scoring based on cue ambiguity
- Explicit margin targets for contrastive loss
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class SenseDefinition:
    """A word sense with gloss and disambiguation cues."""
    sense_id: str           # e.g., "present#gift"
    label: str              # e.g., "gift"
    gloss: str              # e.g., "an item given to someone on an occasion"
    cues: List[str]         # words that suggest this sense
    anti_cues: List[str]    # words that suggest OTHER senses


@dataclass
class ContrastiveTarget:
    """What the model should predict for this item."""
    disambiguation: str
    same_sense_as_anchor: bool


@dataclass
class BundleItem:
    """A single context within a bundle."""
    item_id: str
    role: str               # "anchor", "positive", "negative", "hard_negative"
    sense_id: str
    context: str
    target: ContrastiveTarget
    difficulty: float = 0.0
    hardness: Optional[str] = None  # "easy", "medium", "hard"
    rationale_hint: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class MinimalPair:
    """A minimal edit that changes sense."""
    base_context: str
    edited_context: str
    edit_ops: List[Dict[str, str]]
    intent: str


@dataclass
class Pairings:
    """Explicit contrastive structure."""
    anchor_item_id: str
    positives: List[str]
    negatives: List[str]
    hard_negatives: List[str]


@dataclass
class ContrastiveTargets:
    """Expected phase relations and margins."""
    expected_phase_relation: Dict[str, str]
    margin: Dict[str, float]


@dataclass
class Provenance:
    """Generation metadata."""
    generator: str
    source: str
    seed: int
    timestamp: str
    quality_flags: List[str]


@dataclass
class WordInfo:
    """Word metadata."""
    surface: str
    lemma: str
    pos: str = "noun"
    language: str = "en"


@dataclass
class PolysemyBundle:
    """
    A complete contrastive bundle for polysemy training.

    This format explicitly declares the contrastive geometry:
    - anchor: the reference context
    - positives: same sense as anchor (should be phase-aligned)
    - negatives: different sense (should be phase-separated)
    - hard_negatives: different sense but lexically/structurally similar
    """
    schema_version: str
    record_type: str
    bundle_id: str
    paradigm: str
    word: WordInfo
    sense_catalog: List[SenseDefinition]
    items: List[BundleItem]
    pairings: Pairings
    contrastive_targets: ContrastiveTargets
    provenance: Provenance
    minimal_pair: Optional[MinimalPair] = None
    anchors: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary, handling nested dataclasses."""
        result = {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "bundle_id": self.bundle_id,
            "paradigm": self.paradigm,
            "word": asdict(self.word),
            "sense_catalog": [asdict(s) for s in self.sense_catalog],
            "items": [],
            "pairings": asdict(self.pairings),
            "contrastive_targets": asdict(self.contrastive_targets),
            "provenance": asdict(self.provenance)
        }

        for item in self.items:
            item_dict = {
                "item_id": item.item_id,
                "role": item.role,
                "sense_id": item.sense_id,
                "context": item.context,
                "target": asdict(item.target),
                "difficulty": item.difficulty
            }
            if item.hardness:
                item_dict["hardness"] = item.hardness
            if item.rationale_hint:
                item_dict["rationale_hint"] = item.rationale_hint
            if item.notes:
                item_dict["notes"] = item.notes
            result["items"].append(item_dict)

        if self.minimal_pair:
            result["minimal_pair"] = asdict(self.minimal_pair)
        if self.anchors:
            result["anchors"] = self.anchors

        return result

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'PolysemyBundle':
        """Deserialize from dictionary."""
        word = WordInfo(**data["word"])
        sense_catalog = [SenseDefinition(**s) for s in data["sense_catalog"]]

        items = []
        for item_data in data["items"]:
            target = ContrastiveTarget(**item_data["target"])
            items.append(BundleItem(
                item_id=item_data["item_id"],
                role=item_data["role"],
                sense_id=item_data["sense_id"],
                context=item_data["context"],
                target=target,
                difficulty=item_data.get("difficulty", 0.0),
                hardness=item_data.get("hardness"),
                rationale_hint=item_data.get("rationale_hint"),
                notes=item_data.get("notes")
            ))

        pairings = Pairings(**data["pairings"])
        contrastive_targets = ContrastiveTargets(**data["contrastive_targets"])
        provenance = Provenance(**data["provenance"])

        minimal_pair = None
        if "minimal_pair" in data and data["minimal_pair"]:
            minimal_pair = MinimalPair(**data["minimal_pair"])

        return cls(
            schema_version=data["schema_version"],
            record_type=data["record_type"],
            bundle_id=data["bundle_id"],
            paradigm=data["paradigm"],
            word=word,
            sense_catalog=sense_catalog,
            items=items,
            pairings=pairings,
            contrastive_targets=contrastive_targets,
            provenance=provenance,
            minimal_pair=minimal_pair,
            anchors=data.get("anchors")
        )


# =============================================================================
# SENSE CATALOG - Comprehensive definitions with cues
# =============================================================================

SENSE_CATALOG: Dict[str, List[SenseDefinition]] = {
    "present": [
        SenseDefinition(
            sense_id="present#gift",
            label="gift",
            gloss="an item given to someone on an occasion (birthday, holiday, etc.)",
            cues=["wrap", "gift", "birthday", "under the tree", "give", "recipient", "ribbon", "box", "unwrap", "surprise"],
            anti_cues=["currently", "at the moment", "attendance", "introduce", "nowadays", "time"]
        ),
        SenseDefinition(
            sense_id="present#now",
            label="current_time",
            gloss="the current time; 'now' as opposed to past/future",
            cues=["currently", "at present", "nowadays", "in the present", "moment", "today", "right now"],
            anti_cues=["wrap", "under the tree", "gift", "birthday", "box"]
        ),
        SenseDefinition(
            sense_id="present#verb",
            label="to_show",
            gloss="to show, introduce, or display something formally",
            cues=["present to", "presentation", "introduce", "show", "display", "demonstrate", "findings", "report"],
            anti_cues=["wrap", "gift", "currently", "nowadays"]
        ),
        SenseDefinition(
            sense_id="present#attendance",
            label="in_attendance",
            gloss="being in a place; not absent",
            cues=["roll call", "attendance", "here", "accounted for", "in the room", "present and"],
            anti_cues=["wrap", "gift", "currently", "show"]
        )
    ],
    "bank": [
        SenseDefinition(
            sense_id="bank#financial",
            label="financial_institution",
            gloss="a financial institution that accepts deposits and makes loans",
            cues=["deposit", "withdraw", "account", "loan", "ATM", "teller", "money", "savings", "interest", "branch"],
            anti_cues=["river", "stream", "shore", "water", "fish", "grassy", "steep", "erode"]
        ),
        SenseDefinition(
            sense_id="bank#river",
            label="river_edge",
            gloss="the sloping land beside a body of water",
            cues=["river", "stream", "shore", "water", "grassy", "steep", "erode", "flood", "muddy", "sat by"],
            anti_cues=["deposit", "withdraw", "account", "loan", "ATM", "teller", "money"]
        ),
        SenseDefinition(
            sense_id="bank#tilt",
            label="aircraft_tilt",
            gloss="to tilt or incline (especially aircraft turning)",
            cues=["plane", "aircraft", "turn", "banked left", "banked right", "pilot", "angle", "tilt"],
            anti_cues=["deposit", "river", "money", "water"]
        )
    ],
    "spring": [
        SenseDefinition(
            sense_id="spring#season",
            label="season",
            gloss="the season between winter and summer",
            cues=["flowers", "bloom", "April", "May", "warm", "birds", "season", "weather", "garden"],
            anti_cues=["water", "coil", "metal", "jump", "bounce", "mattress"]
        ),
        SenseDefinition(
            sense_id="spring#water",
            label="water_source",
            gloss="a natural source of water from the ground",
            cues=["water", "fresh", "bubbling", "natural", "drink", "mountain", "source", "mineral"],
            anti_cues=["flowers", "season", "coil", "jump", "mattress", "bounce"]
        ),
        SenseDefinition(
            sense_id="spring#coil",
            label="coiled_metal",
            gloss="a coiled piece of metal that returns to shape when compressed",
            cues=["coil", "metal", "mattress", "bounce", "tension", "mechanism", "clock", "compressed"],
            anti_cues=["flowers", "season", "water", "fresh", "natural"]
        ),
        SenseDefinition(
            sense_id="spring#jump",
            label="to_jump",
            gloss="to jump or leap suddenly",
            cues=["sprang", "leap", "jumped", "action", "feet", "suddenly", "forward"],
            anti_cues=["flowers", "season", "water", "coil", "mattress"]
        )
    ],
    "wave": [
        SenseDefinition(
            sense_id="wave#ocean",
            label="ocean_motion",
            gloss="a moving ridge of water on the surface of the sea",
            cues=["ocean", "sea", "surf", "crash", "beach", "tide", "water", "splash", "shore"],
            anti_cues=["hand", "hello", "goodbye", "gesture", "hair", "curl"]
        ),
        SenseDefinition(
            sense_id="wave#gesture",
            label="hand_gesture",
            gloss="a gesture made by moving one's hand",
            cues=["hand", "hello", "goodbye", "gesture", "greeting", "friendly", "acknowledge", "smiled"],
            anti_cues=["ocean", "sea", "surf", "crash", "beach", "water"]
        ),
        SenseDefinition(
            sense_id="wave#hair",
            label="hair_curl",
            gloss="a curl or undulation in hair",
            cues=["hair", "curl", "wavy", "style", "natural", "salon", "strand"],
            anti_cues=["ocean", "hand", "gesture", "greeting"]
        ),
        SenseDefinition(
            sense_id="wave#surge",
            label="surge",
            gloss="a sudden increase or occurrence of something",
            cues=["crime wave", "heat wave", "new wave", "wave of", "surge", "increase", "trend"],
            anti_cues=["ocean", "hand", "hair", "curl"]
        )
    ],
    "bark": [
        SenseDefinition(
            sense_id="bark#tree",
            label="tree_covering",
            gloss="the tough outer covering of a tree trunk",
            cues=["tree", "trunk", "rough", "peeled", "oak", "birch", "wood", "outer"],
            anti_cues=["dog", "loud", "growl", "puppy", "woof"]
        ),
        SenseDefinition(
            sense_id="bark#dog",
            label="dog_sound",
            gloss="the sharp explosive cry of a dog",
            cues=["dog", "puppy", "loud", "growl", "woof", "animal", "startled", "heard"],
            anti_cues=["tree", "trunk", "rough", "peeled", "oak", "wood"]
        )
    ],
    "light": [
        SenseDefinition(
            sense_id="light#illumination",
            label="illumination",
            gloss="electromagnetic radiation that makes things visible",
            cues=["bright", "lamp", "sun", "shine", "glow", "illuminate", "beam", "ray", "dark"],
            anti_cues=["heavy", "weight", "carry", "feather", "ignite", "match"]
        ),
        SenseDefinition(
            sense_id="light#weight",
            label="not_heavy",
            gloss="having little weight; not heavy",
            cues=["feather", "weight", "carry", "lift", "heavy", "pounds", "ounces", "portable"],
            anti_cues=["bright", "lamp", "sun", "shine", "glow", "ignite"]
        ),
        SenseDefinition(
            sense_id="light#ignite",
            label="to_ignite",
            gloss="to set fire to; ignite",
            cues=["match", "candle", "fire", "flame", "ignite", "lit", "burn"],
            anti_cues=["bright", "weight", "heavy", "feather", "shine"]
        )
    ],
    "match": [
        SenseDefinition(
            sense_id="match#fire",
            label="fire_starter",
            gloss="a short thin piece of wood tipped with combustible material",
            cues=["fire", "flame", "strike", "box of", "light", "ignite", "burn", "sulfur"],
            anti_cues=["game", "tournament", "correspond", "pair", "similar"]
        ),
        SenseDefinition(
            sense_id="match#competition",
            label="competition",
            gloss="a contest or game",
            cues=["game", "tournament", "won", "lost", "played", "opponent", "team", "final"],
            anti_cues=["fire", "flame", "strike", "ignite", "correspond"]
        ),
        SenseDefinition(
            sense_id="match#correspond",
            label="to_correspond",
            gloss="to be equal or similar to; correspond",
            cues=["match with", "matching", "pair", "similar", "correspond", "identical", "same"],
            anti_cues=["fire", "game", "tournament", "strike", "won"]
        )
    ],
    "bass": [
        SenseDefinition(
            sense_id="bass#fish",
            label="fish",
            gloss="a type of freshwater or sea fish",
            cues=["fish", "caught", "fishing", "lake", "sea", "striped", "largemouth"],
            anti_cues=["music", "guitar", "low", "frequency", "speaker", "drum"]
        ),
        SenseDefinition(
            sense_id="bass#music",
            label="low_frequency",
            gloss="the lowest part in musical harmony; low frequency sound",
            cues=["music", "guitar", "low", "frequency", "speaker", "drum", "deep", "tone", "boost"],
            anti_cues=["fish", "caught", "fishing", "lake", "sea"]
        )
    ]
}


def compute_cue_difficulty(context: str, target_sense: SenseDefinition,
                           other_senses: List[SenseDefinition]) -> float:
    """
    Compute difficulty based on cue ambiguity.

    Low difficulty: Many target cues present, few anti-cues
    High difficulty: Few target cues, many competing sense cues present
    """
    context_lower = context.lower()

    # Count target sense cues present
    target_cue_count = sum(1 for cue in target_sense.cues if cue.lower() in context_lower)
    target_anti_count = sum(1 for cue in target_sense.anti_cues if cue.lower() in context_lower)

    # Count competing sense cues present
    competing_cue_count = 0
    for other in other_senses:
        competing_cue_count += sum(1 for cue in other.cues if cue.lower() in context_lower)

    # Normalize
    max_cues = max(len(target_sense.cues), 1)
    target_ratio = target_cue_count / max_cues

    # Difficulty formula:
    # - High target cues → low difficulty
    # - High competing cues → high difficulty
    # - High anti-cues present → high difficulty (wrong sense indicators)

    base_difficulty = 1.0 - target_ratio
    ambiguity_penalty = min(competing_cue_count * 0.1, 0.3)
    anti_cue_penalty = min(target_anti_count * 0.15, 0.3)

    difficulty = base_difficulty + ambiguity_penalty + anti_cue_penalty
    return min(max(difficulty, 0.0), 1.0)


def generate_bundle_id(word: str, index: int) -> str:
    """Generate stable bundle ID."""
    return f"poly_{word}_{index:06d}"


def generate_item_id(bundle_id: str, role: str, index: int = 0) -> str:
    """Generate stable item ID."""
    role_code = {"anchor": "a", "positive": "p", "negative": "n", "hard_negative": "hn"}
    return f"{bundle_id}_{role_code.get(role, 'x')}{index}"


def create_bundle(
    word: str,
    anchor_sense_id: str,
    anchor_context: str,
    positive_contexts: List[tuple],  # [(sense_id, context), ...]
    negative_contexts: List[tuple],  # [(sense_id, context), ...]
    hard_negative_contexts: List[tuple],  # [(sense_id, context, notes), ...]
    bundle_index: int,
    seed: int = 42,
    minimal_pair: Optional[MinimalPair] = None
) -> PolysemyBundle:
    """
    Create a complete PolysemyBundle from components.
    """
    bundle_id = generate_bundle_id(word, bundle_index)
    sense_catalog = SENSE_CATALOG.get(word, [])

    # Find senses
    anchor_sense = next((s for s in sense_catalog if s.sense_id == anchor_sense_id), None)
    other_senses = [s for s in sense_catalog if s.sense_id != anchor_sense_id]

    items = []

    # Anchor item
    anchor_difficulty = compute_cue_difficulty(anchor_context, anchor_sense, other_senses) if anchor_sense else 0.3
    anchor_item = BundleItem(
        item_id=generate_item_id(bundle_id, "anchor"),
        role="anchor",
        sense_id=anchor_sense_id,
        context=anchor_context,
        target=ContrastiveTarget(
            disambiguation=anchor_sense.label if anchor_sense else anchor_sense_id.split("#")[1],
            same_sense_as_anchor=True
        ),
        difficulty=anchor_difficulty,
        rationale_hint=f"Reference context for {anchor_sense.label if anchor_sense else 'sense'}."
    )
    items.append(anchor_item)

    positive_ids = []
    negative_ids = []
    hard_negative_ids = []

    # Positive items
    for i, (sense_id, context) in enumerate(positive_contexts):
        pos_sense = next((s for s in sense_catalog if s.sense_id == sense_id), anchor_sense)
        difficulty = compute_cue_difficulty(context, pos_sense, other_senses) if pos_sense else 0.25
        item_id = generate_item_id(bundle_id, "positive", i + 1)
        items.append(BundleItem(
            item_id=item_id,
            role="positive",
            sense_id=sense_id,
            context=context,
            target=ContrastiveTarget(
                disambiguation=pos_sense.label if pos_sense else sense_id.split("#")[1],
                same_sense_as_anchor=True
            ),
            difficulty=difficulty
        ))
        positive_ids.append(item_id)

    # Negative items (easy)
    for i, (sense_id, context) in enumerate(negative_contexts):
        neg_sense = next((s for s in sense_catalog if s.sense_id == sense_id), None)
        difficulty = compute_cue_difficulty(context, neg_sense, [anchor_sense] if anchor_sense else []) if neg_sense else 0.35
        item_id = generate_item_id(bundle_id, "negative", i + 1)
        items.append(BundleItem(
            item_id=item_id,
            role="negative",
            sense_id=sense_id,
            context=context,
            target=ContrastiveTarget(
                disambiguation=neg_sense.label if neg_sense else sense_id.split("#")[1],
                same_sense_as_anchor=False
            ),
            difficulty=difficulty,
            hardness="easy"
        ))
        negative_ids.append(item_id)

    # Hard negative items
    for i, item_data in enumerate(hard_negative_contexts):
        sense_id, context = item_data[0], item_data[1]
        notes = item_data[2] if len(item_data) > 2 else None

        hn_sense = next((s for s in sense_catalog if s.sense_id == sense_id), None)
        # Hard negatives should have higher base difficulty
        difficulty = min(compute_cue_difficulty(context, hn_sense, [anchor_sense] if anchor_sense else []) + 0.3, 1.0) if hn_sense else 0.8
        item_id = generate_item_id(bundle_id, "hard_negative", i + 1)
        items.append(BundleItem(
            item_id=item_id,
            role="hard_negative",
            sense_id=sense_id,
            context=context,
            target=ContrastiveTarget(
                disambiguation=hn_sense.label if hn_sense else sense_id.split("#")[1],
                same_sense_as_anchor=False
            ),
            difficulty=difficulty,
            hardness="hard",
            notes=notes
        ))
        hard_negative_ids.append(item_id)

    return PolysemyBundle(
        schema_version="2.0",
        record_type="polysemy_bundle",
        bundle_id=bundle_id,
        paradigm="semantic_phase",
        word=WordInfo(surface=word, lemma=word, pos="noun"),
        sense_catalog=sense_catalog,
        items=items,
        pairings=Pairings(
            anchor_item_id=anchor_item.item_id,
            positives=positive_ids,
            negatives=negative_ids,
            hard_negatives=hard_negative_ids
        ),
        contrastive_targets=ContrastiveTargets(
            expected_phase_relation={
                "positive": "aligned",
                "negative": "separated",
                "hard_negative": "strongly_separated"
            },
            margin={
                "positive_vs_negative": 0.15,
                "positive_vs_hard_negative": 0.25
            }
        ),
        provenance=Provenance(
            generator="paradigm_factory.polysemy_bundle_v2",
            source="manual" if seed == 42 else "swarm",
            seed=seed,
            timestamp=datetime.now().isoformat(),
            quality_flags=["validated_gloss", "cue_difficulty_computed"]
        ),
        minimal_pair=minimal_pair
    )


def load_bundles(path: Path) -> List[PolysemyBundle]:
    """Load bundles from JSONL file."""
    bundles = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                bundles.append(PolysemyBundle.from_dict(data))
    return bundles


def save_bundles(bundles: List[PolysemyBundle], path: Path):
    """Save bundles to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for bundle in bundles:
            f.write(bundle.to_json() + "\n")


# =============================================================================
# Legacy conversion - generate input_text/output_text for backward compatibility
# =============================================================================

def bundle_to_legacy_pairs(bundle: PolysemyBundle) -> List[Dict]:
    """
    Convert a v2 bundle to legacy v1 format pairs.
    Useful for backward compatibility with existing trainer.
    """
    legacy_pairs = []

    anchor_item = next(i for i in bundle.items if i.role == "anchor")

    for item in bundle.items:
        if item.role == "anchor":
            continue

        # Generate legacy input_text
        input_text = f"""Context A: {anchor_item.context}
Context B: {item.context}

Do these use '{bundle.word.surface}' in the same sense?"""

        # Generate legacy output_text
        if item.target.same_sense_as_anchor:
            output_text = f"Yes, both use '{bundle.word.surface}' in the {anchor_item.target.disambiguation} sense. The meaning is consistent."
        else:
            output_text = f"No, '{bundle.word.surface}' has different meanings. Context A uses the {anchor_item.target.disambiguation} sense, while Context B uses the {item.target.disambiguation} sense."

        legacy_pairs.append({
            "input_text": input_text,
            "output_text": output_text,
            "paradigm": "semantic_phase",
            "subtype": f"polysemy_{item.role}",
            "metadata": {
                "bundle_id": bundle.bundle_id,
                "word": bundle.word.surface,
                "anchor_sense": anchor_item.sense_id,
                "item_sense": item.sense_id,
                "pair_type": "same_sense" if item.target.same_sense_as_anchor else "different_sense",
                "expected_phase": "aligned" if item.target.same_sense_as_anchor else "separated",
                "difficulty": item.difficulty,
                "hardness": item.hardness
            }
        })

    return legacy_pairs


if __name__ == "__main__":
    # Demo: Create a sample bundle
    bundle = create_bundle(
        word="present",
        anchor_sense_id="present#gift",
        anchor_context="She wrapped the birthday present carefully.",
        positive_contexts=[
            ("present#gift", "The present was hidden under the tree."),
            ("present#gift", "He bought her a present for their anniversary.")
        ],
        negative_contexts=[
            ("present#now", "At present, sales are rising faster than expected."),
            ("present#now", "The present situation requires immediate attention.")
        ],
        hard_negative_contexts=[
            ("present#verb", "The report will present the findings at noon.", "POS shift confuses shallow cues"),
            ("present#attendance", "All members were present at the meeting.", "Attendance sense with minimal cues")
        ],
        bundle_index=184,
        seed=42
    )

    print("=== Sample PolysemyBundle v2 ===")
    print(json.dumps(bundle.to_dict(), indent=2))

    print("\n=== Legacy Conversion ===")
    legacy = bundle_to_legacy_pairs(bundle)
    for pair in legacy[:2]:
        print(json.dumps(pair, indent=2))
