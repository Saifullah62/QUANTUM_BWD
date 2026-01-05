#!/usr/bin/env python3
"""
Evaluate SemanticPhase on External Probe Pack
==============================================

Runs trained model on WSD/ambiguity/generalization probes and
documents failure modes, not just scores.

Usage:
    python scripts/eval_on_probes.py --checkpoint model.pt --eval-pack eval_pack.json
    python scripts/eval_on_probes.py --checkpoint model.pt --eval-pack eval_pack.json --output-report results.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@dataclass
class ProbeResult:
    """Result from evaluating a single probe."""
    probe_id: str
    word: str
    category: str
    subcategory: str
    expected_sense: str
    difficulty: str

    # Model output
    predicted_sense: Optional[str] = None
    confidence: float = 0.0
    embedding_norm: float = 0.0

    # Evaluation
    correct: bool = False
    failure_mode: Optional[str] = None  # Why it failed
    notes: str = ""


@dataclass
class EvalReport:
    """Complete evaluation report."""
    timestamp: str
    checkpoint_path: str
    eval_pack_path: str

    # Overall metrics
    total_probes: int = 0
    correct: int = 0
    accuracy: float = 0.0

    # By category
    wsd_accuracy: float = 0.0
    ambiguity_accuracy: float = 0.0
    generalization_accuracy: float = 0.0

    # By difficulty
    easy_accuracy: float = 0.0
    medium_accuracy: float = 0.0
    hard_accuracy: float = 0.0

    # Failure modes (the important part!)
    failure_modes: Dict[str, int] = field(default_factory=dict)
    failure_examples: List[Dict] = field(default_factory=list)

    # Per-probe results
    probe_results: List[ProbeResult] = field(default_factory=list)


def load_eval_pack(path: Path) -> Dict:
    """Load evaluation pack from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_failure_mode(probe: Dict, result: ProbeResult) -> str:
    """
    Classify why a probe failed. This is the diagnostic heart of evaluation.

    Common failure modes:
    - rare_sense_confusion: Rare sense predicted as common sense
    - similar_surface_confusion: Confused by similar surface form
    - insufficient_context: Not enough disambiguating context
    - domain_transfer_failure: Failed on domain-specific usage
    - heteronym_failure: Failed on words with different pronunciations
    """
    expected = probe.get("expected_sense", "").lower()
    predicted = (result.predicted_sense or "").lower()

    # Check for rare sense confusion
    if "rare" in probe.get("subcategory", ""):
        return "rare_sense_confusion"

    # Check for domain-specific failure
    if "domain" in probe.get("subcategory", ""):
        return "domain_transfer_failure"

    # Check for heteronym failure
    if "heteronym" in probe.get("subcategory", ""):
        return "heteronym_failure"

    # Check for ambiguous context (expected AMBIGUOUS)
    if expected in ["ambiguous", "both", "zeugma"]:
        return "ambiguity_handling_failure"

    # Check for garden path
    if "garden_path" in probe.get("subcategory", ""):
        return "garden_path_failure"

    # Default: insufficient disambiguation
    return "insufficient_context"


def evaluate_probe(
    model,
    tokenizer,
    probe: Dict,
    device: str = "cuda"
) -> ProbeResult:
    """
    Evaluate a single probe with the model.

    For now, we use embedding similarity as a proxy for sense prediction.
    The model should produce different embeddings for different senses.
    """
    result = ProbeResult(
        probe_id=probe["probe_id"],
        word=probe["word"],
        category=probe["category"],
        subcategory=probe["subcategory"],
        expected_sense=probe["expected_sense"],
        difficulty=probe["difficulty"]
    )

    try:
        # Tokenize context
        context = probe["context"]
        tokens = tokenizer(
            context,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        # Get model output
        with torch.no_grad():
            outputs = model(**tokens)

            # Get phase states if available (SemanticPhase layer)
            if hasattr(outputs, 'phase_states') and outputs.phase_states is not None:
                embedding = outputs.phase_states[:, 0, :]  # First token
                if embedding.is_complex():
                    embedding = embedding.abs()
            else:
                # Fall back to hidden states
                embedding = outputs.last_hidden_state[:, 0, :]

            result.embedding_norm = embedding.norm().item()

            # For sense prediction, we'd compare to sense prototypes
            # For now, just mark as evaluated
            result.predicted_sense = "evaluated"
            result.confidence = 1.0
            result.correct = True  # Placeholder - real eval needs sense prototypes

    except Exception as e:
        result.failure_mode = f"evaluation_error: {str(e)}"
        result.correct = False

    return result


def evaluate_minimal_pair(
    model,
    tokenizer,
    pair: Dict,
    device: str = "cuda"
) -> Tuple[bool, str]:
    """
    Evaluate a minimal pair: do the two contexts produce different embeddings?

    A successful model should produce MORE similar embeddings for same-sense
    contexts than for different-sense contexts.
    """
    try:
        # Tokenize both contexts
        tokens_a = tokenizer(
            pair["context_a"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        tokens_b = tokenizer(
            pair["context_b"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs_a = model(**tokens_a)
            outputs_b = model(**tokens_b)

            # Get embeddings
            if hasattr(outputs_a, 'phase_states') and outputs_a.phase_states is not None:
                emb_a = outputs_a.phase_states[:, 0, :]
                emb_b = outputs_b.phase_states[:, 0, :]
                if emb_a.is_complex():
                    emb_a = emb_a.abs()
                    emb_b = emb_b.abs()
            else:
                emb_a = outputs_a.last_hidden_state[:, 0, :]
                emb_b = outputs_b.last_hidden_state[:, 0, :]

            # Compute similarity
            similarity = F.cosine_similarity(emb_a, emb_b).item()

            # For different senses, we expect LOW similarity
            # Success if similarity < 0.8 (embeddings are differentiated)
            success = similarity < 0.8

            return success, f"sim={similarity:.3f}"

    except Exception as e:
        return False, f"error: {str(e)}"


def run_evaluation(
    checkpoint_path: Path,
    eval_pack_path: Path,
    output_report: Optional[Path] = None
) -> EvalReport:
    """Run full evaluation on the probe pack."""

    print("\n" + "=" * 60)
    print("  EXTERNAL PROBE EVALUATION")
    print("=" * 60)

    # Load eval pack
    eval_pack = load_eval_pack(eval_pack_path)
    print(f"\nLoaded eval pack: {eval_pack_path}")
    print(f"  WSD probes: {len(eval_pack.get('wsd_probes', []))}")
    print(f"  Ambiguity probes: {len(eval_pack.get('ambiguity_probes', []))}")
    print(f"  Generalization probes: {len(eval_pack.get('generalization_probes', []))}")
    print(f"  Minimal pairs: {len(eval_pack.get('minimal_pairs', []))}")

    report = EvalReport(
        timestamp=datetime.now().isoformat(),
        checkpoint_path=str(checkpoint_path),
        eval_pack_path=str(eval_pack_path)
    )

    # Check if model checkpoint exists
    if not checkpoint_path.exists():
        print(f"\nWARNING: Checkpoint not found: {checkpoint_path}")
        print("Running in ANALYSIS-ONLY mode (no model predictions)")

        # Just analyze the probe pack structure
        all_probes = (
            eval_pack.get("wsd_probes", []) +
            eval_pack.get("ambiguity_probes", []) +
            eval_pack.get("generalization_probes", [])
        )

        report.total_probes = len(all_probes)

        # Analyze by category
        by_category = defaultdict(list)
        by_difficulty = defaultdict(list)
        by_subcategory = defaultdict(list)

        for probe in all_probes:
            by_category[probe["category"]].append(probe)
            by_difficulty[probe["difficulty"]].append(probe)
            by_subcategory[probe["subcategory"]].append(probe)

        print(f"\n  PROBE DISTRIBUTION ANALYSIS")
        print(f"  ----------------------------")
        print(f"\n  By category:")
        for cat, probes in sorted(by_category.items()):
            print(f"    {cat}: {len(probes)}")

        print(f"\n  By difficulty:")
        for diff, probes in sorted(by_difficulty.items()):
            print(f"    {diff}: {len(probes)}")

        print(f"\n  By subcategory:")
        for subcat, probes in sorted(by_subcategory.items()):
            print(f"    {subcat}: {len(probes)}")

        # Analyze minimal pairs
        minimal_pairs = eval_pack.get("minimal_pairs", [])
        print(f"\n  MINIMAL PAIRS ANALYSIS")
        print(f"  ----------------------")
        print(f"  Total pairs: {len(minimal_pairs)}")

        diff_types = defaultdict(int)
        for pair in minimal_pairs:
            diff_types[pair["difference_type"]] += 1

        print(f"  By difference type:")
        for dt, count in sorted(diff_types.items()):
            print(f"    {dt}: {count}")

        return report

    # TODO: Load actual model and run predictions
    print(f"\nLoading checkpoint: {checkpoint_path}")
    print("  (Full model evaluation not yet implemented)")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate on External Probe Pack")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--eval-pack", type=str, required=True,
                        help="Path to evaluation pack JSON")
    parser.add_argument("--output-report", type=str,
                        help="Output JSON report path")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Just analyze probe pack, don't run model")

    args = parser.parse_args()

    report = run_evaluation(
        checkpoint_path=Path(args.checkpoint),
        eval_pack_path=Path(args.eval_pack),
        output_report=Path(args.output_report) if args.output_report else None
    )

    if args.output_report:
        with open(args.output_report, "w") as f:
            json.dump({
                "timestamp": report.timestamp,
                "checkpoint_path": report.checkpoint_path,
                "eval_pack_path": report.eval_pack_path,
                "total_probes": report.total_probes,
                "accuracy": report.accuracy,
                "failure_modes": report.failure_modes
            }, f, indent=2)
        print(f"\nReport saved to: {args.output_report}")


if __name__ == "__main__":
    main()
