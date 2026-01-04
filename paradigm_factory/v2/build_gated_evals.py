#!/usr/bin/env python3
"""
Gated Eval Builder - Produces Clean Eval Artifacts
===================================================

This script is the ONLY entry point for producing eval artifacts.
The scorer should NEVER consume raw eval files directly.

Pipeline:
  raw evals -> integrity gate -> gated artifacts

Output artifacts:
  evals_gated/
  ├── retrieval_valid.jsonl      # Ready to score
  ├── retrieval_quarantined.jsonl # Blocked (bad data)
  ├── retrieval_regen.jsonl      # Needs context regeneration
  ├── coherence_valid.jsonl      # Ready to score
  ├── coherence_quarantined.jsonl
  ├── coherence_regen.jsonl
  ├── BUILD_MANIFEST.json        # Build metadata + fingerprint
  └── EVAL_INTEGRITY_GOLD_v1.json # Frozen calibration (never modified)

Usage:
  python paradigm_factory/v2/build_gated_evals.py
  python paradigm_factory/v2/build_gated_evals.py --input evals/ --output evals_gated/
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paradigm_factory.v2.eval_integrity_gate import EvalIntegrityGate, ItemAction
from paradigm_factory.v2.eval_harness import (
    IntegrityPreflight,
    EXPECTED_GOLD_HASH,
    GOLD_FILE_VERSION,
    INTEGRITY_RULESET,
    INTEGRITY_RULESET_CHANGELOG
)


@dataclass
class BuildManifest:
    """Manifest for a gated eval build."""
    build_id: str
    timestamp: str

    # Input files
    input_retrieval: str
    input_coherence: str

    # Trust anchors
    gold_version: str
    gold_hash: str
    integrity_ruleset: str

    # Counts
    retrieval_valid: int
    retrieval_quarantined: int
    retrieval_regen: int
    coherence_valid: int
    coherence_quarantined: int
    coherence_regen: int

    # Rates
    retrieval_quarantine_rate: float
    coherence_quarantine_rate: float

    # Fingerprint for provenance
    fingerprint: str


class GatedEvalBuilder:
    """Builds gated eval artifacts from raw eval files."""

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.gate = EvalIntegrityGate()
        self.preflight = IntegrityPreflight()

    def verify_trust_anchors(self) -> bool:
        """Verify gold hash before building."""
        file_match, code_match, embedded, actual = self.preflight.verify_gold_hash()

        if not file_match:
            print(f"[ERROR] Gold file content hash mismatch")
            print(f"  Embedded: {embedded}")
            print(f"  Actual:   {actual}")
            return False

        if not code_match:
            print(f"[ERROR] Trust anchor mismatch")
            print(f"  EXPECTED_GOLD_HASH: {EXPECTED_GOLD_HASH}")
            print(f"  Actual:             {actual}")
            return False

        print(f"[OK] Trust anchors verified")
        print(f"  Gold version: {GOLD_FILE_VERSION}")
        print(f"  Hash: {actual[:16]}...")
        print(f"  Ruleset: {INTEGRITY_RULESET}")
        return True

    def gate_eval_file(self, input_file: Path, prefix: str, is_retrieval: bool = True) -> Tuple[int, int, int]:
        """
        Gate a single eval file and write output artifacts.

        Args:
            input_file: Path to input JSONL file
            prefix: Output file prefix (e.g., 'retrieval', 'coherence')
            is_retrieval: If True, apply full integrity gate. If False, pass through
                         (coherence evals have different structure)

        Returns:
            (valid_count, quarantined_count, regen_count)
        """
        # Load items
        items = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))

        if not items:
            print(f"  [WARN] No items in {input_file}")
            return 0, 0, 0

        # Gate each item
        valid_items = []
        quarantined_items = []
        regen_items = []

        for item in items:
            if is_retrieval:
                # Apply full integrity gate for retrieval evals
                result = self.gate.check_item(item)

                # Add gate metadata to item
                item['_gate'] = {
                    'action': result.action.value,
                    'issues': result.issues,
                    'suggestions': result.suggestions,
                    'query_realism': result.query_realism.value,
                    'gold_alignment': result.gold_alignment.value,
                    'context_richness': result.context_richness.value
                }

                if result.action == ItemAction.KEEP:
                    valid_items.append(item)
                elif result.action == ItemAction.QUARANTINE:
                    quarantined_items.append(item)
                elif result.action == ItemAction.REGEN:
                    regen_items.append(item)
            else:
                # For coherence evals, pass through as valid
                # (coherence has different structure - steps, not query/candidates)
                item['_gate'] = {
                    'action': 'keep',
                    'issues': [],
                    'suggestions': [],
                    'note': 'coherence_passthrough'
                }
                valid_items.append(item)

        # Write output files
        self.output_dir.mkdir(parents=True, exist_ok=True)

        valid_file = self.output_dir / f"{prefix}_valid.jsonl"
        quarantine_file = self.output_dir / f"{prefix}_quarantined.jsonl"
        regen_file = self.output_dir / f"{prefix}_regen.jsonl"

        for filepath, items_list in [
            (valid_file, valid_items),
            (quarantine_file, quarantined_items),
            (regen_file, regen_items)
        ]:
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in items_list:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        return len(valid_items), len(quarantined_items), len(regen_items)

    def build(self) -> BuildManifest:
        """Build all gated eval artifacts."""
        build_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("=" * 60)
        print("GATED EVAL BUILDER")
        print(f"Build ID: {build_id}")
        print("=" * 60)

        # Verify trust anchors first
        print("\n[1/4] Verifying trust anchors...")
        if not self.verify_trust_anchors():
            raise RuntimeError("Trust anchor verification failed")

        # Gate retrieval evals
        print("\n[2/4] Gating retrieval evals...")
        retrieval_input = self.input_dir / "eval_multi_sense_retrieval.jsonl"
        if retrieval_input.exists():
            r_valid, r_quarantine, r_regen = self.gate_eval_file(retrieval_input, "retrieval")
            print(f"  Valid: {r_valid}, Quarantined: {r_quarantine}, Regen: {r_regen}")
        else:
            print(f"  [WARN] Not found: {retrieval_input}")
            r_valid, r_quarantine, r_regen = 0, 0, 0

        # Gate coherence evals (passthrough - different structure)
        print("\n[3/4] Processing coherence evals (passthrough)...")
        coherence_input = self.input_dir / "eval_multi_step_coherence.jsonl"
        if coherence_input.exists():
            c_valid, c_quarantine, c_regen = self.gate_eval_file(coherence_input, "coherence", is_retrieval=False)
            print(f"  Valid: {c_valid}, Quarantined: {c_quarantine}, Regen: {c_regen}")
        else:
            print(f"  [WARN] Not found: {coherence_input}")
            c_valid, c_quarantine, c_regen = 0, 0, 0

        # Compute rates
        r_total = r_valid + r_quarantine + r_regen
        c_total = c_valid + c_quarantine + c_regen
        r_qrate = r_quarantine / r_total if r_total > 0 else 0.0
        c_qrate = c_quarantine / c_total if c_total > 0 else 0.0

        # Build fingerprint
        _, _, _, actual_hash = self.preflight.verify_gold_hash()
        fingerprint = f"gold={GOLD_FILE_VERSION}|rules={INTEGRITY_RULESET}|hash={actual_hash[:12]}|rq={r_qrate:.1%}|cq={c_qrate:.1%}"

        # Create manifest
        manifest = BuildManifest(
            build_id=build_id,
            timestamp=datetime.now().isoformat(),
            input_retrieval=str(retrieval_input),
            input_coherence=str(coherence_input),
            gold_version=GOLD_FILE_VERSION,
            gold_hash=actual_hash,
            integrity_ruleset=INTEGRITY_RULESET,
            retrieval_valid=r_valid,
            retrieval_quarantined=r_quarantine,
            retrieval_regen=r_regen,
            coherence_valid=c_valid,
            coherence_quarantined=c_quarantine,
            coherence_regen=c_regen,
            retrieval_quarantine_rate=r_qrate,
            coherence_quarantine_rate=c_qrate,
            fingerprint=fingerprint
        )

        # Write manifest
        print("\n[4/4] Writing manifest...")
        manifest_file = self.output_dir / "BUILD_MANIFEST.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(manifest), f, indent=2)
        print(f"  Manifest: {manifest_file}")

        # Summary
        print("\n" + "=" * 60)
        print("BUILD COMPLETE")
        print(f"Fingerprint: {fingerprint}")
        print("=" * 60)

        print("\nOutput artifacts:")
        for f in sorted(self.output_dir.glob("*.jsonl")):
            count = sum(1 for _ in open(f, 'r', encoding='utf-8'))
            print(f"  {f.name}: {count} items")

        return manifest


def main():
    parser = argparse.ArgumentParser(description="Build gated eval artifacts")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "paradigm_factory" / "v2" / "evals",
                       help="Input directory with raw evals")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paradigm_factory" / "v2" / "evals_gated",
                       help="Output directory for gated artifacts")
    args = parser.parse_args()

    builder = GatedEvalBuilder(args.input, args.output)
    manifest = builder.build()

    # Exit with error if quarantine rate is too high
    if manifest.retrieval_quarantine_rate > 0.20:
        print(f"\n[ERROR] Retrieval quarantine rate {manifest.retrieval_quarantine_rate:.1%} exceeds 20%")
        sys.exit(1)
    if manifest.coherence_quarantine_rate > 0.20:
        print(f"\n[ERROR] Coherence quarantine rate {manifest.coherence_quarantine_rate:.1%} exceeds 20%")
        sys.exit(1)


if __name__ == "__main__":
    main()
