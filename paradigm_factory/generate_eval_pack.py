#!/usr/bin/env python3
"""
Paradigm Factory - Phase D: Eval Pack Generator
================================================

Generates evaluation packs that test paradigm-specific behaviors:
1. Polysemy disambiguation (choose correct sense)
2. Lindblad consistency (clean vs noisy agreement)
3. Difficulty-stratified samples

Two output sets:
- Core (frozen): Stable benchmark for tracking progress
- Fresh (rotating): New examples from each night's generation
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field

# Eval pack sizes
CORE_SIZE = 100  # Frozen benchmark
FRESH_SIZE = 50  # Nightly rotating


@dataclass
class PolysemyEvalItem:
    """A polysemy disambiguation evaluation item."""
    word: str
    context_with_blank: str  # Context with [BLANK] where word goes
    correct_sense: str
    distractor_senses: List[str]
    difficulty: float
    item_type: str = "polysemy_disambiguation"


@dataclass
class LindbladEvalItem:
    """A Lindblad consistency evaluation item."""
    clean_text: str
    noisy_text: str
    corruption_type: str
    expected_agreement: bool  # Should outputs match?
    similarity_score: float
    item_type: str = "lindblad_consistency"


@dataclass
class EvalPack:
    """A complete evaluation pack."""
    name: str
    created: str
    pack_type: str  # "core" or "fresh"
    polysemy_items: List[PolysemyEvalItem] = field(default_factory=list)
    lindblad_items: List[LindbladEvalItem] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


def create_polysemy_eval_item(
    word: str,
    positive_context: str,
    correct_sense: str,
    all_senses: List[str],
    difficulty: float
) -> Optional[PolysemyEvalItem]:
    """
    Create a polysemy disambiguation eval item.

    Format: Given a context with a blank, choose which sense the word has.
    """
    # Replace the word with [BLANK]
    import re
    pattern = re.compile(re.escape(word), re.IGNORECASE)
    context_with_blank = pattern.sub("[BLANK]", positive_context, count=1)

    if context_with_blank == positive_context:
        # Word not found in context
        return None

    # Get distractor senses
    distractors = [s for s in all_senses if s != correct_sense][:3]
    if not distractors:
        return None

    return PolysemyEvalItem(
        word=word,
        context_with_blank=context_with_blank,
        correct_sense=correct_sense,
        distractor_senses=distractors,
        difficulty=difficulty
    )


def create_lindblad_eval_item(twin_data: Dict) -> LindbladEvalItem:
    """Create a Lindblad consistency eval item from twin data."""
    return LindbladEvalItem(
        clean_text=twin_data["original_text"],
        noisy_text=twin_data["noisy_text"],
        corruption_type=twin_data["corruption_type"],
        expected_agreement=True,  # We expect semantic agreement
        similarity_score=twin_data.get("similarity_score", 0.0)
    )


def load_polysemy_data(path: Path) -> List[Dict]:
    """Load polysemy examples from JSONL."""
    examples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def load_lindblad_data(path: Path) -> List[Dict]:
    """Load Lindblad twins from JSONL."""
    twins = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                twins.append(json.loads(line))
    return twins


def build_polysemy_eval_items(
    examples: List[Dict],
    max_items: int = 100
) -> List[PolysemyEvalItem]:
    """Build polysemy eval items from examples."""
    # Group by word to get all senses
    from collections import defaultdict
    by_word = defaultdict(lambda: {"senses": set(), "examples": []})

    for ex in examples:
        if ex.get("subtype") == "positive":
            word = ex["word"]
            by_word[word]["senses"].add(ex["sense"])
            by_word[word]["examples"].append(ex)

    # Create eval items
    items = []
    for word, data in by_word.items():
        senses = list(data["senses"])
        if len(senses) < 2:
            continue

        for ex in data["examples"]:
            item = create_polysemy_eval_item(
                word=word,
                positive_context=ex["context"],
                correct_sense=ex["sense"],
                all_senses=senses,
                difficulty=ex.get("difficulty", 0.5)
            )
            if item:
                items.append(item)

    # Sort by difficulty and sample
    items.sort(key=lambda x: x.difficulty, reverse=True)
    return items[:max_items]


def build_lindblad_eval_items(
    twins: List[Dict],
    max_items: int = 100
) -> List[LindbladEvalItem]:
    """Build Lindblad eval items from twins."""
    items = [create_lindblad_eval_item(t) for t in twins]

    # Sort by similarity (prefer harder cases)
    items.sort(key=lambda x: x.similarity_score)

    # Stratified sampling: mix of similarity ranges
    easy = [i for i in items if i.similarity_score > 0.95]
    medium = [i for i in items if 0.85 <= i.similarity_score <= 0.95]
    hard = [i for i in items if i.similarity_score < 0.85]

    result = []
    per_tier = max_items // 3
    result.extend(hard[:per_tier])
    result.extend(medium[:per_tier])
    result.extend(easy[:per_tier])

    return result[:max_items]


def generate_eval_pack(
    polysemy_path: Optional[Path] = None,
    lindblad_path: Optional[Path] = None,
    pack_type: str = "fresh",
    polysemy_count: int = 50,
    lindblad_count: int = 50,
    output_path: Optional[Path] = None
) -> EvalPack:
    """Generate a complete eval pack."""

    pack = EvalPack(
        name=f"paradigm_eval_{pack_type}_{datetime.now().strftime('%Y%m%d')}",
        created=datetime.now().isoformat(),
        pack_type=pack_type,
        metadata={
            "polysemy_source": str(polysemy_path) if polysemy_path else None,
            "lindblad_source": str(lindblad_path) if lindblad_path else None,
        }
    )

    # Build polysemy items
    if polysemy_path and polysemy_path.exists():
        print(f"Loading polysemy data from {polysemy_path}...")
        polysemy_examples = load_polysemy_data(polysemy_path)
        pack.polysemy_items = build_polysemy_eval_items(
            polysemy_examples, max_items=polysemy_count
        )
        print(f"  Created {len(pack.polysemy_items)} polysemy eval items")

    # Build Lindblad items
    if lindblad_path and lindblad_path.exists():
        print(f"Loading Lindblad data from {lindblad_path}...")
        lindblad_twins = load_lindblad_data(lindblad_path)
        pack.lindblad_items = build_lindblad_eval_items(
            lindblad_twins, max_items=lindblad_count
        )
        print(f"  Created {len(pack.lindblad_items)} Lindblad eval items")

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        pack_dict = {
            "name": pack.name,
            "created": pack.created,
            "pack_type": pack.pack_type,
            "metadata": pack.metadata,
            "polysemy_items": [asdict(item) for item in pack.polysemy_items],
            "lindblad_items": [asdict(item) for item in pack.lindblad_items],
            "summary": {
                "polysemy_count": len(pack.polysemy_items),
                "lindblad_count": len(pack.lindblad_items),
                "total_items": len(pack.polysemy_items) + len(pack.lindblad_items)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(pack_dict, f, indent=2)
        print(f"\nSaved eval pack to {output_path}")

    return pack


def run_eval_pack(
    eval_pack_path: Path,
    model_endpoint: str = "http://159.203.35.45:8006/v1/chat/completions",
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run an eval pack against a model endpoint.
    Returns metrics for each paradigm.
    """
    import requests

    with open(eval_pack_path, 'r') as f:
        pack = json.load(f)

    results = {
        "pack_name": pack["name"],
        "model_endpoint": model_endpoint,
        "run_timestamp": datetime.now().isoformat(),
        "polysemy_results": [],
        "lindblad_results": [],
        "metrics": {}
    }

    # Polysemy evaluation
    print("Running polysemy evaluation...")
    polysemy_correct = 0
    for item in pack.get("polysemy_items", []):
        prompt = f"""Given the context: "{item['context_with_blank']}"

The word that fills [BLANK] can have these meanings:
A) {item['correct_sense']}
B) {item['distractor_senses'][0] if item['distractor_senses'] else 'N/A'}
C) {item['distractor_senses'][1] if len(item['distractor_senses']) > 1 else 'N/A'}

Which meaning does the word have in this context? Answer with just the letter."""

        try:
            resp = requests.post(
                model_endpoint,
                json={"messages": [{"role": "user", "content": prompt}]},
                timeout=30
            )
            resp.raise_for_status()
            answer = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            is_correct = answer.strip().upper().startswith("A")
            if is_correct:
                polysemy_correct += 1
            results["polysemy_results"].append({
                "item": item,
                "model_answer": answer,
                "correct": is_correct
            })
        except Exception as e:
            print(f"  Error: {e}")

    # Lindblad evaluation (checking output consistency)
    print("Running Lindblad evaluation...")
    lindblad_consistent = 0
    for item in pack.get("lindblad_items", [])[:20]:  # Limit for speed
        prompt_clean = f"Summarize in one sentence: {item['clean_text']}"
        prompt_noisy = f"Summarize in one sentence: {item['noisy_text']}"

        try:
            resp_clean = requests.post(
                model_endpoint,
                json={"messages": [{"role": "user", "content": prompt_clean}]},
                timeout=30
            )
            resp_noisy = requests.post(
                model_endpoint,
                json={"messages": [{"role": "user", "content": prompt_noisy}]},
                timeout=30
            )

            answer_clean = resp_clean.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            answer_noisy = resp_noisy.json().get("choices", [{}])[0].get("message", {}).get("content", "")

            # Simple consistency check: similar length and some overlap
            clean_words = set(answer_clean.lower().split())
            noisy_words = set(answer_noisy.lower().split())
            overlap = len(clean_words & noisy_words) / max(len(clean_words | noisy_words), 1)
            is_consistent = overlap > 0.5

            if is_consistent:
                lindblad_consistent += 1

            results["lindblad_results"].append({
                "corruption_type": item["corruption_type"],
                "clean_response": answer_clean,
                "noisy_response": answer_noisy,
                "overlap": overlap,
                "consistent": is_consistent
            })
        except Exception as e:
            print(f"  Error: {e}")

    # Compute metrics
    polysemy_total = len(pack.get("polysemy_items", []))
    lindblad_total = len(results["lindblad_results"])

    results["metrics"] = {
        "polysemy_accuracy": polysemy_correct / polysemy_total if polysemy_total else 0,
        "polysemy_correct": polysemy_correct,
        "polysemy_total": polysemy_total,
        "lindblad_consistency": lindblad_consistent / lindblad_total if lindblad_total else 0,
        "lindblad_consistent": lindblad_consistent,
        "lindblad_total": lindblad_total
    }

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved eval results to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate or run paradigm eval packs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate eval pack")
    gen_parser.add_argument('--polysemy', type=str, help='Polysemy JSONL path')
    gen_parser.add_argument('--lindblad', type=str, help='Lindblad twins JSONL path')
    gen_parser.add_argument('--type', type=str, default='fresh',
                            choices=['core', 'fresh'], help='Pack type')
    gen_parser.add_argument('--polysemy-count', type=int, default=50)
    gen_parser.add_argument('--lindblad-count', type=int, default=50)
    gen_parser.add_argument('--output', type=str,
                            default=f'paradigm_factory/output/eval_pack_{datetime.now().strftime("%Y%m%d")}.json')

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run eval pack")
    run_parser.add_argument('--pack', type=str, required=True, help='Eval pack JSON path')
    run_parser.add_argument('--endpoint', type=str,
                            default='http://159.203.35.45:8006/v1/chat/completions',
                            help='Model endpoint')
    run_parser.add_argument('--output', type=str,
                            default=f'paradigm_factory/output/eval_results_{datetime.now().strftime("%Y%m%d")}.json')

    args = parser.parse_args()

    print("=" * 60)
    print("  PARADIGM FACTORY - EVAL PACK GENERATOR")
    print("=" * 60)

    if args.command == "generate":
        pack = generate_eval_pack(
            polysemy_path=Path(args.polysemy) if args.polysemy else None,
            lindblad_path=Path(args.lindblad) if args.lindblad else None,
            pack_type=args.type,
            polysemy_count=args.polysemy_count,
            lindblad_count=args.lindblad_count,
            output_path=Path(args.output)
        )

        print("\n" + "=" * 60)
        print("  EVAL PACK SUMMARY")
        print("=" * 60)
        print(f"Pack name: {pack.name}")
        print(f"Type: {pack.pack_type}")
        print(f"Polysemy items: {len(pack.polysemy_items)}")
        print(f"Lindblad items: {len(pack.lindblad_items)}")

    elif args.command == "run":
        results = run_eval_pack(
            eval_pack_path=Path(args.pack),
            model_endpoint=args.endpoint,
            output_path=Path(args.output)
        )

        print("\n" + "=" * 60)
        print("  EVAL RESULTS")
        print("=" * 60)
        metrics = results["metrics"]
        print(f"Polysemy accuracy: {metrics['polysemy_accuracy']:.1%} ({metrics['polysemy_correct']}/{metrics['polysemy_total']})")
        print(f"Lindblad consistency: {metrics['lindblad_consistency']:.1%} ({metrics['lindblad_consistent']}/{metrics['lindblad_total']})")


if __name__ == "__main__":
    main()
