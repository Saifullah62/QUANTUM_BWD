#!/usr/bin/env python3
"""
SemanticPhase Certification Run
===============================

Executes a reproducible benchmark run according to certification_run_v1.json.
This is the official bar that any changes must beat.

Usage:
    python certification/run_certification.py --bundles data.jsonl
    python certification/run_certification.py --bundles data.jsonl --skip-generation
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_certification_spec(spec_path: Path) -> dict:
    """Load certification specification."""
    with open(spec_path, "r") as f:
        return json.load(f)


def run_certification(
    bundles_path: Path,
    spec_path: Path = None,
    output_dir: Path = None,
    skip_generation: bool = False
):
    """Execute full certification run."""

    print("\n" + "=" * 70)
    print("  SEMANTICPHASE CERTIFICATION RUN")
    print("=" * 70)

    # Load spec
    if spec_path is None:
        spec_path = Path(__file__).parent / "certification_run_v1.json"

    spec = load_certification_spec(spec_path)
    print(f"\nCertification version: {spec['certification_version']}")
    print(f"Created: {spec['created_at']}")

    if output_dir is None:
        output_dir = Path(f"certification/runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    results = {
        "certification_version": spec["certification_version"],
        "run_timestamp": datetime.now().isoformat(),
        "bundles_path": str(bundles_path),
        "output_dir": str(output_dir),
        "stages": {}
    }

    # Stage 1: QA Pipeline
    print(f"\n{'='*60}")
    print("  STAGE 1: QA PIPELINE")
    print(f"{'='*60}")

    from paradigm_factory.bundle_qa_pipeline import BundleQAPipeline, BundleVerifier

    qa_dir = output_dir / "qa"
    verifier = BundleVerifier(
        min_items=spec["dataset_spec"]["qa_pipeline"]["min_items_per_bundle"],
        check_leakage=spec["dataset_spec"]["qa_pipeline"]["leakage_check"],
        max_leakage_items=spec["dataset_spec"]["qa_pipeline"]["max_leakage_items"]
    )
    pipeline = BundleQAPipeline(qa_dir, verifier=verifier)

    qa_stats = pipeline.run_full_pipeline(bundles_path, batch_name="certification")
    results["stages"]["qa"] = {
        "total_input": qa_stats.total_input,
        "verified": qa_stats.verified,
        "rejected": qa_stats.rejected,
        "rejection_reasons": dict(qa_stats.rejection_reasons)
    }

    # Stage 2: By-Word Split
    print(f"\n{'='*60}")
    print("  STAGE 2: BY-WORD SPLIT")
    print(f"{'='*60}")

    from paradigm_factory.dataset_splits import create_splits

    verified_bundles = list((qa_dir / "train_ready").glob("*.jsonl"))[0]
    splits_dir = output_dir / "splits"

    split_manifest = create_splits(
        input_path=verified_bundles,
        output_dir=splits_dir,
        method="by_word",
        train_ratio=spec["dataset_spec"]["split_ratios"]["train"],
        val_ratio=spec["dataset_spec"]["split_ratios"]["val"],
        eval_ratio=spec["dataset_spec"]["split_ratios"]["eval"],
        seed=spec["dataset_spec"]["split_seed"]
    )

    results["stages"]["split"] = {
        "train_bundles": split_manifest.train_bundles,
        "train_words": split_manifest.train_words,
        "val_bundles": split_manifest.val_bundles,
        "val_words": split_manifest.val_words,
        "eval_bundles": split_manifest.eval_bundles,
        "eval_words": split_manifest.eval_words
    }

    # Stage 3: Characterization with Audits
    print(f"\n{'='*60}")
    print("  STAGE 3: DATASET CHARACTERIZATION")
    print(f"{'='*60}")

    from paradigm_factory.dataset_characterization import characterize_dataset, print_report
    from paradigm_factory.polysemy_bundle_v2 import load_bundles

    train_bundles = load_bundles(splits_dir / "train.jsonl")
    report = characterize_dataset(train_bundles, run_audits=True)
    print_report(report, verbose=False)

    results["stages"]["characterization"] = {
        "total_bundles": report.total_bundles,
        "unique_words": report.unique_words,
        "difficulty_balance": report.difficulty.balance_score,
        "hard_negative_audit_passed": report.hard_negative_audit.passed if report.hard_negative_audit else None,
        "leakage_audit_passed": report.leakage_audit.passed if report.leakage_audit else None,
        "quality_score": report.quality_score,
        "issues": report.issues
    }

    # Stage 4: 3-Seed Training (placeholder - full training takes time)
    print(f"\n{'='*60}")
    print("  STAGE 4: 3-SEED TRAINING")
    print(f"{'='*60}")

    print(f"  Seeds: {spec['training_spec']['seeds']}")
    print(f"  Epochs: {spec['training_spec']['epochs']}")
    print(f"  Success criterion: slack > 0 for {spec['success_criteria']['primary']['threshold']*100}% of late steps")
    print(f"\n  NOTE: Full training run required (~30-60 min per seed)")
    print(f"  Run: python scripts/run_3seed_validation.py --bundles {splits_dir}/train.jsonl")

    results["stages"]["training"] = {
        "status": "pending",
        "command": f"python scripts/run_3seed_validation.py --bundles {splits_dir}/train.jsonl --epochs {spec['training_spec']['epochs']}"
    }

    # Stage 5: External Eval (placeholder)
    print(f"\n{'='*60}")
    print("  STAGE 5: EXTERNAL PROBE EVALUATION")
    print(f"{'='*60}")

    print(f"  Eval pack: {spec['eval_spec']['eval_pack']}")
    print(f"  Probes: WSD={spec['eval_spec']['probe_counts']['wsd']}, "
          f"Ambiguity={spec['eval_spec']['probe_counts']['ambiguity']}, "
          f"Generalization={spec['eval_spec']['probe_counts']['generalization']}")
    print(f"\n  NOTE: Run after training completes with best checkpoint")

    results["stages"]["eval"] = {
        "status": "pending",
        "command": f"python scripts/eval_on_probes.py --checkpoint <best_checkpoint> --eval-pack {spec['eval_spec']['eval_pack']}"
    }

    # Save results
    results_path = output_dir / "certification_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("  CERTIFICATION RUN INITIALIZED")
    print(f"{'='*70}")
    print(f"\n  Results saved to: {results_path}")
    print(f"\n  Next steps:")
    print(f"    1. Run 3-seed training (Stage 4)")
    print(f"    2. Run external eval with best checkpoint (Stage 5)")
    print(f"    3. Compare against baseline to certify changes")

    return results


def main():
    parser = argparse.ArgumentParser(description="SemanticPhase Certification Run")
    parser.add_argument("--bundles", type=str, required=True,
                        help="Path to input bundles JSONL")
    parser.add_argument("--spec", type=str,
                        help="Path to certification spec JSON")
    parser.add_argument("--output-dir", type=str,
                        help="Output directory for certification run")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip bundle generation (use existing)")

    args = parser.parse_args()

    run_certification(
        bundles_path=Path(args.bundles),
        spec_path=Path(args.spec) if args.spec else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        skip_generation=args.skip_generation
    )


if __name__ == "__main__":
    main()
