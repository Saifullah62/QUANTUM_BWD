#!/usr/bin/env python3
"""
External Eval Pack Builder
==========================

Creates evaluation packs for testing SemanticPhase beyond training data:

1. WSD PROBES: Word Sense Disambiguation test cases
   - Classic polysemy pairs (bank, crane, etc.)
   - Minimal pairs with subtle sense differences

2. AMBIGUITY PROBES: Edge cases and ambiguous contexts
   - Garden path sentences
   - Deliberately ambiguous contexts
   - Cross-sense interference tests

3. GENERALIZATION PROBES: Out-of-distribution tests
   - Novel words not in training
   - Rare senses of common words
   - Domain transfer tests

Usage:
    python eval_pack_builder.py --output eval_pack.json
    python eval_pack_builder.py --output eval_pack.json --include-wsd --include-ambiguity
"""

import argparse
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@dataclass
class EvalProbe:
    """A single evaluation probe."""
    probe_id: str
    category: str  # wsd, ambiguity, generalization
    subcategory: str  # specific type within category
    word: str
    context: str
    expected_sense: str
    alternative_senses: List[str]
    difficulty: str  # easy, medium, hard
    notes: str = ""


@dataclass
class MinimalPairProbe:
    """A minimal pair for contrastive evaluation."""
    probe_id: str
    word: str
    context_a: str
    sense_a: str
    context_b: str
    sense_b: str
    difference_type: str  # lexical, syntactic, pragmatic
    notes: str = ""


@dataclass
class EvalPack:
    """Complete evaluation pack."""
    version: str
    created_at: str
    description: str

    wsd_probes: List[EvalProbe] = field(default_factory=list)
    ambiguity_probes: List[EvalProbe] = field(default_factory=list)
    generalization_probes: List[EvalProbe] = field(default_factory=list)
    minimal_pairs: List[MinimalPairProbe] = field(default_factory=list)

    metadata: Dict = field(default_factory=dict)


# =============================================================================
# WSD PROBES - Classic word sense disambiguation cases
# =============================================================================

WSD_PROBES = [
    # BANK - financial vs. river
    EvalProbe(
        probe_id="wsd_bank_001",
        category="wsd",
        subcategory="noun_polysemy",
        word="bank",
        context="I need to deposit this check at the bank before it closes.",
        expected_sense="financial_institution",
        alternative_senses=["river_edge", "aircraft_turn"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="wsd_bank_002",
        category="wsd",
        subcategory="noun_polysemy",
        word="bank",
        context="We sat on the grassy bank watching the river flow past.",
        expected_sense="river_edge",
        alternative_senses=["financial_institution", "aircraft_turn"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="wsd_bank_003",
        category="wsd",
        subcategory="noun_polysemy",
        word="bank",
        context="The pilot executed a sharp bank to avoid the storm.",
        expected_sense="aircraft_turn",
        alternative_senses=["financial_institution", "river_edge"],
        difficulty="medium"
    ),

    # CRANE - bird vs. machine
    EvalProbe(
        probe_id="wsd_crane_001",
        category="wsd",
        subcategory="noun_polysemy",
        word="crane",
        context="A tall crane waded through the shallow marsh hunting for fish.",
        expected_sense="bird",
        alternative_senses=["lifting_machine", "neck_movement"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="wsd_crane_002",
        category="wsd",
        subcategory="noun_polysemy",
        word="crane",
        context="The construction crane lifted steel beams to the top floor.",
        expected_sense="lifting_machine",
        alternative_senses=["bird", "neck_movement"],
        difficulty="easy"
    ),

    # BAT - animal vs. sports equipment
    EvalProbe(
        probe_id="wsd_bat_001",
        category="wsd",
        subcategory="noun_polysemy",
        word="bat",
        context="The bat flew out of the cave at dusk to hunt insects.",
        expected_sense="flying_mammal",
        alternative_senses=["sports_equipment", "to_hit"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="wsd_bat_002",
        category="wsd",
        subcategory="noun_polysemy",
        word="bat",
        context="She picked up the wooden bat and stepped up to home plate.",
        expected_sense="sports_equipment",
        alternative_senses=["flying_mammal", "to_hit"],
        difficulty="easy"
    ),

    # SPRING - season vs. water source vs. coil
    EvalProbe(
        probe_id="wsd_spring_001",
        category="wsd",
        subcategory="noun_polysemy",
        word="spring",
        context="The flowers bloom every spring when the weather warms up.",
        expected_sense="season",
        alternative_senses=["water_source", "coil", "to_jump"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="wsd_spring_002",
        category="wsd",
        subcategory="noun_polysemy",
        word="spring",
        context="Fresh water bubbled up from the natural spring in the hills.",
        expected_sense="water_source",
        alternative_senses=["season", "coil", "to_jump"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="wsd_spring_003",
        category="wsd",
        subcategory="noun_polysemy",
        word="spring",
        context="The mattress spring was broken, making the bed uncomfortable.",
        expected_sense="coil",
        alternative_senses=["season", "water_source", "to_jump"],
        difficulty="medium"
    ),

    # LEAD - metal vs. guide (heteronym)
    EvalProbe(
        probe_id="wsd_lead_001",
        category="wsd",
        subcategory="heteronym",
        word="lead",
        context="The lead pipes in old buildings can contaminate drinking water.",
        expected_sense="metal_element",
        alternative_senses=["to_guide", "main_role", "advantage"],
        difficulty="medium"
    ),
    EvalProbe(
        probe_id="wsd_lead_002",
        category="wsd",
        subcategory="heteronym",
        word="lead",
        context="She will lead the team through the difficult negotiations.",
        expected_sense="to_guide",
        alternative_senses=["metal_element", "main_role", "advantage"],
        difficulty="easy"
    ),

    # TEAR - to rip vs. eye drop (heteronym)
    EvalProbe(
        probe_id="wsd_tear_001",
        category="wsd",
        subcategory="heteronym",
        word="tear",
        context="A single tear rolled down her cheek as she read the letter.",
        expected_sense="eye_drop",
        alternative_senses=["to_rip"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="wsd_tear_002",
        category="wsd",
        subcategory="heteronym",
        word="tear",
        context="Be careful not to tear the delicate fabric of the dress.",
        expected_sense="to_rip",
        alternative_senses=["eye_drop"],
        difficulty="easy"
    ),

    # PRESENT - gift vs. current time vs. to show (stress shift)
    EvalProbe(
        probe_id="wsd_present_001",
        category="wsd",
        subcategory="stress_polysemy",
        word="present",
        context="She wrapped the birthday present in colorful paper.",
        expected_sense="gift",
        alternative_senses=["current_time", "to_show", "in_attendance"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="wsd_present_002",
        category="wsd",
        subcategory="stress_polysemy",
        word="present",
        context="At the present time, we have no plans to expand.",
        expected_sense="current_time",
        alternative_senses=["gift", "to_show", "in_attendance"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="wsd_present_003",
        category="wsd",
        subcategory="stress_polysemy",
        word="present",
        context="I will present my findings at the conference next week.",
        expected_sense="to_show",
        alternative_senses=["gift", "current_time", "in_attendance"],
        difficulty="easy"
    ),

    # BASS - fish vs. musical (heteronym)
    EvalProbe(
        probe_id="wsd_bass_001",
        category="wsd",
        subcategory="heteronym",
        word="bass",
        context="We caught a large bass while fishing in the lake.",
        expected_sense="fish",
        alternative_senses=["low_frequency", "instrument"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="wsd_bass_002",
        category="wsd",
        subcategory="heteronym",
        word="bass",
        context="The bass guitar provides the rhythm section's foundation.",
        expected_sense="instrument",
        alternative_senses=["fish", "low_frequency"],
        difficulty="easy"
    ),
]


# =============================================================================
# AMBIGUITY PROBES - Edge cases and deliberately ambiguous contexts
# =============================================================================

AMBIGUITY_PROBES = [
    # Garden path sentences
    EvalProbe(
        probe_id="amb_garden_001",
        category="ambiguity",
        subcategory="garden_path",
        word="bank",
        context="The bank was steep and covered with flowers.",
        expected_sense="river_edge",  # Intended, but temporarily ambiguous
        alternative_senses=["financial_institution"],
        difficulty="hard",
        notes="Garden path: initially ambiguous, resolved by 'steep'"
    ),
    EvalProbe(
        probe_id="amb_garden_002",
        category="ambiguity",
        subcategory="garden_path",
        word="pen",
        context="The pen was full of animals.",
        expected_sense="enclosure",
        alternative_senses=["writing_instrument"],
        difficulty="hard",
        notes="Garden path: writing pen expected, enclosure revealed"
    ),

    # Truly ambiguous (no clear resolution)
    EvalProbe(
        probe_id="amb_unresolved_001",
        category="ambiguity",
        subcategory="unresolved",
        word="bank",
        context="I'm going to the bank later.",
        expected_sense="AMBIGUOUS",
        alternative_senses=["financial_institution", "river_edge"],
        difficulty="hard",
        notes="Genuinely ambiguous without more context"
    ),
    EvalProbe(
        probe_id="amb_unresolved_002",
        category="ambiguity",
        subcategory="unresolved",
        word="watch",
        context="Watch out!",
        expected_sense="to_observe",  # Primary, but could be timepiece-related
        alternative_senses=["timepiece"],
        difficulty="medium",
        notes="Idiomatic usage, verb sense primary"
    ),

    # Cross-sense interference
    EvalProbe(
        probe_id="amb_interference_001",
        category="ambiguity",
        subcategory="interference",
        word="bass",
        context="He played bass while fishing for bass.",
        expected_sense="BOTH",
        alternative_senses=["instrument", "fish"],
        difficulty="hard",
        notes="Both senses used in same context"
    ),
    EvalProbe(
        probe_id="amb_interference_002",
        category="ambiguity",
        subcategory="interference",
        word="crane",
        context="The crane operator watched a crane fly by.",
        expected_sense="BOTH",
        alternative_senses=["lifting_machine", "bird"],
        difficulty="hard",
        notes="Both senses in juxtaposition"
    ),

    # Minimal context (hard to disambiguate)
    EvalProbe(
        probe_id="amb_minimal_001",
        category="ambiguity",
        subcategory="minimal_context",
        word="rock",
        context="I love rock.",
        expected_sense="AMBIGUOUS",
        alternative_senses=["stone", "music_genre", "to_sway"],
        difficulty="hard",
        notes="Multiple plausible interpretations"
    ),
    EvalProbe(
        probe_id="amb_minimal_002",
        category="ambiguity",
        subcategory="minimal_context",
        word="light",
        context="The light was too bright.",
        expected_sense="illumination",  # Most likely
        alternative_senses=["not_heavy", "to_ignite"],
        difficulty="medium",
        notes="Adjective use suggests illumination but 'light' is polysemous"
    ),

    # Zeugma (one word, two senses)
    EvalProbe(
        probe_id="amb_zeugma_001",
        category="ambiguity",
        subcategory="zeugma",
        word="fire",
        context="She fired the gun and the employee.",
        expected_sense="ZEUGMA",
        alternative_senses=["to_shoot", "to_dismiss"],
        difficulty="hard",
        notes="Zeugma: same word used in two senses simultaneously"
    ),
]


# =============================================================================
# GENERALIZATION PROBES - Out-of-distribution tests
# =============================================================================

GENERALIZATION_PROBES = [
    # Rare senses
    EvalProbe(
        probe_id="gen_rare_001",
        category="generalization",
        subcategory="rare_sense",
        word="strike",
        context="The clock began to strike midnight.",
        expected_sense="to_sound_hour",
        alternative_senses=["to_hit", "work_stoppage", "bowling"],
        difficulty="medium",
        notes="Rare sense of strike (clock striking)"
    ),
    EvalProbe(
        probe_id="gen_rare_002",
        category="generalization",
        subcategory="rare_sense",
        word="vessel",
        context="The blood vessel was blocked.",
        expected_sense="tube_in_body",
        alternative_senses=["container", "ship"],
        difficulty="medium",
        notes="Medical sense"
    ),
    EvalProbe(
        probe_id="gen_rare_003",
        category="generalization",
        subcategory="rare_sense",
        word="table",
        context="Let's table this discussion for now.",
        expected_sense="to_postpone",
        alternative_senses=["furniture", "data_arrangement"],
        difficulty="hard",
        notes="Verbal sense of table (to postpone)"
    ),

    # Domain-specific usage
    EvalProbe(
        probe_id="gen_domain_001",
        category="generalization",
        subcategory="domain_specific",
        word="cell",
        context="The terrorist was held in a cell.",
        expected_sense="prison_room",
        alternative_senses=["biological_unit", "battery_unit", "group"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="gen_domain_002",
        category="generalization",
        subcategory="domain_specific",
        word="cell",
        context="The red blood cell carries oxygen.",
        expected_sense="biological_unit",
        alternative_senses=["prison_room", "battery_unit", "group"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="gen_domain_003",
        category="generalization",
        subcategory="domain_specific",
        word="bug",
        context="We found a bug in the software code.",
        expected_sense="software_defect",
        alternative_senses=["insect", "surveillance_device", "illness"],
        difficulty="easy"
    ),

    # Technical jargon
    EvalProbe(
        probe_id="gen_jargon_001",
        category="generalization",
        subcategory="jargon",
        word="trunk",
        context="The elephant's trunk can hold several liters of water.",
        expected_sense="elephant_nose",
        alternative_senses=["tree_stem", "car_storage", "torso"],
        difficulty="easy"
    ),
    EvalProbe(
        probe_id="gen_jargon_002",
        category="generalization",
        subcategory="jargon",
        word="mouse",
        context="Move the mouse to click on the icon.",
        expected_sense="computer_device",
        alternative_senses=["rodent", "quiet_person"],
        difficulty="easy"
    ),

    # Slang / informal
    EvalProbe(
        probe_id="gen_slang_001",
        category="generalization",
        subcategory="slang",
        word="crash",
        context="Can I crash at your place tonight?",
        expected_sense="to_sleep_informally",
        alternative_senses=["collision", "computer_failure"],
        difficulty="medium"
    ),
    EvalProbe(
        probe_id="gen_slang_002",
        category="generalization",
        subcategory="slang",
        word="sick",
        context="That skateboard trick was sick!",
        expected_sense="impressive_slang",
        alternative_senses=["ill", "disgusted"],
        difficulty="medium"
    ),
]


# =============================================================================
# MINIMAL PAIRS - Contrastive pairs for sense discrimination
# =============================================================================

MINIMAL_PAIRS = [
    MinimalPairProbe(
        probe_id="mp_bank_001",
        word="bank",
        context_a="I deposited my paycheck at the bank.",
        sense_a="financial_institution",
        context_b="I sat on the bank watching the ducks.",
        sense_b="river_edge",
        difference_type="lexical",
        notes="Classic polysemy pair"
    ),
    MinimalPairProbe(
        probe_id="mp_crane_001",
        word="crane",
        context_a="The crane lifted the heavy beam.",
        sense_a="lifting_machine",
        context_b="The crane stood in the shallow water.",
        sense_b="bird",
        difference_type="lexical",
        notes="Machine vs. animal"
    ),
    MinimalPairProbe(
        probe_id="mp_lead_001",
        word="lead",
        context_a="The lead pipe was heavy.",
        sense_a="metal_element",
        context_b="She will lead the meeting.",
        sense_b="to_guide",
        difference_type="syntactic",
        notes="Heteronym: noun vs. verb"
    ),
    MinimalPairProbe(
        probe_id="mp_tear_001",
        word="tear",
        context_a="A tear fell from her eye.",
        sense_a="eye_drop",
        context_b="Don't tear the paper.",
        sense_b="to_rip",
        difference_type="syntactic",
        notes="Heteronym: noun vs. verb"
    ),
    MinimalPairProbe(
        probe_id="mp_spring_001",
        word="spring",
        context_a="I love spring flowers.",
        sense_a="season",
        context_b="The spring was broken.",
        sense_b="coil",
        difference_type="lexical",
        notes="Season vs. mechanical part"
    ),
    MinimalPairProbe(
        probe_id="mp_present_001",
        word="present",
        context_a="I brought a present for you.",
        sense_a="gift",
        context_b="I will present my findings.",
        sense_b="to_show",
        difference_type="syntactic",
        notes="Noun vs. verb, stress shift"
    ),
    MinimalPairProbe(
        probe_id="mp_bass_001",
        word="bass",
        context_a="We caught a bass in the lake.",
        sense_a="fish",
        context_b="Turn up the bass on the speakers.",
        sense_b="low_frequency",
        difference_type="lexical",
        notes="Heteronym: fish vs. sound"
    ),
    MinimalPairProbe(
        probe_id="mp_bow_001",
        word="bow",
        context_a="She tied a bow in her hair.",
        sense_a="decorative_knot",
        context_b="The performer took a bow.",
        sense_b="bending_gesture",
        difference_type="lexical",
        notes="Knot vs. gesture"
    ),
    MinimalPairProbe(
        probe_id="mp_wind_001",
        word="wind",
        context_a="The wind blew fiercely.",
        sense_a="moving_air",
        context_b="Wind the clock before bed.",
        sense_b="to_turn",
        difference_type="syntactic",
        notes="Heteronym: noun vs. verb"
    ),
    MinimalPairProbe(
        probe_id="mp_close_001",
        word="close",
        context_a="Please close the door.",
        sense_a="to_shut",
        context_b="The store is close to my house.",
        sense_b="nearby",
        difference_type="syntactic",
        notes="Verb vs. adjective"
    ),
]


def build_eval_pack(
    include_wsd: bool = True,
    include_ambiguity: bool = True,
    include_generalization: bool = True,
    include_minimal_pairs: bool = True
) -> EvalPack:
    """Build complete evaluation pack."""
    pack = EvalPack(
        version="1.0",
        created_at=datetime.now().isoformat(),
        description="SemanticPhase Evaluation Pack: WSD, Ambiguity, and Generalization Probes"
    )

    if include_wsd:
        pack.wsd_probes = WSD_PROBES.copy()

    if include_ambiguity:
        pack.ambiguity_probes = AMBIGUITY_PROBES.copy()

    if include_generalization:
        pack.generalization_probes = GENERALIZATION_PROBES.copy()

    if include_minimal_pairs:
        pack.minimal_pairs = MINIMAL_PAIRS.copy()

    # Metadata
    pack.metadata = {
        "wsd_probe_count": len(pack.wsd_probes),
        "ambiguity_probe_count": len(pack.ambiguity_probes),
        "generalization_probe_count": len(pack.generalization_probes),
        "minimal_pair_count": len(pack.minimal_pairs),
        "total_probes": (
            len(pack.wsd_probes) +
            len(pack.ambiguity_probes) +
            len(pack.generalization_probes)
        ),
        "unique_words": len(set(
            [p.word for p in pack.wsd_probes] +
            [p.word for p in pack.ambiguity_probes] +
            [p.word for p in pack.generalization_probes] +
            [p.word for p in pack.minimal_pairs]
        ))
    }

    return pack


def save_eval_pack(pack: EvalPack, output_path: Path):
    """Save evaluation pack to JSON."""
    data = {
        "version": pack.version,
        "created_at": pack.created_at,
        "description": pack.description,
        "metadata": pack.metadata,
        "wsd_probes": [asdict(p) for p in pack.wsd_probes],
        "ambiguity_probes": [asdict(p) for p in pack.ambiguity_probes],
        "generalization_probes": [asdict(p) for p in pack.generalization_probes],
        "minimal_pairs": [asdict(p) for p in pack.minimal_pairs]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved eval pack to: {output_path}")


def print_pack_summary(pack: EvalPack):
    """Print summary of evaluation pack."""
    print("\n" + "=" * 60)
    print("  EVALUATION PACK SUMMARY")
    print("=" * 60)

    print(f"\n  Version: {pack.version}")
    print(f"  Created: {pack.created_at}")

    print(f"\n  Probe counts:")
    print(f"    WSD probes:          {len(pack.wsd_probes):3d}")
    print(f"    Ambiguity probes:    {len(pack.ambiguity_probes):3d}")
    print(f"    Generalization:      {len(pack.generalization_probes):3d}")
    print(f"    Minimal pairs:       {len(pack.minimal_pairs):3d}")
    print(f"    Total probes:        {pack.metadata['total_probes']:3d}")

    print(f"\n  Unique words covered:  {pack.metadata['unique_words']}")

    # WSD breakdown by subcategory
    if pack.wsd_probes:
        print(f"\n  WSD subcategories:")
        subcats = {}
        for p in pack.wsd_probes:
            subcats[p.subcategory] = subcats.get(p.subcategory, 0) + 1
        for subcat, count in sorted(subcats.items()):
            print(f"    {subcat}: {count}")

    # Ambiguity breakdown
    if pack.ambiguity_probes:
        print(f"\n  Ambiguity subcategories:")
        subcats = {}
        for p in pack.ambiguity_probes:
            subcats[p.subcategory] = subcats.get(p.subcategory, 0) + 1
        for subcat, count in sorted(subcats.items()):
            print(f"    {subcat}: {count}")

    # Difficulty breakdown
    all_probes = pack.wsd_probes + pack.ambiguity_probes + pack.generalization_probes
    if all_probes:
        print(f"\n  Difficulty distribution:")
        difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
        for p in all_probes:
            difficulty_counts[p.difficulty] = difficulty_counts.get(p.difficulty, 0) + 1
        for diff, count in sorted(difficulty_counts.items()):
            print(f"    {diff}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Build External Eval Pack")
    parser.add_argument("--output", type=str, default="paradigm_factory/output/eval_pack.json",
                        help="Output JSON path")
    parser.add_argument("--include-wsd", action="store_true", default=True)
    parser.add_argument("--include-ambiguity", action="store_true", default=True)
    parser.add_argument("--include-generalization", action="store_true", default=True)
    parser.add_argument("--include-minimal-pairs", action="store_true", default=True)
    parser.add_argument("--no-wsd", action="store_true", help="Exclude WSD probes")
    parser.add_argument("--no-ambiguity", action="store_true", help="Exclude ambiguity probes")

    args = parser.parse_args()

    # Build pack
    pack = build_eval_pack(
        include_wsd=not args.no_wsd,
        include_ambiguity=not args.no_ambiguity,
        include_generalization=args.include_generalization,
        include_minimal_pairs=args.include_minimal_pairs
    )

    # Print summary
    print_pack_summary(pack)

    # Save
    save_eval_pack(pack, Path(args.output))


if __name__ == "__main__":
    main()
