#!/usr/bin/env python3
"""
CI Gate Script - Must pass before any merge to main
====================================================

Checks performed:
1. Gold hash verification (file + code constant)
2. Integrity gate preflight
3. Quarantine rate threshold
4. Canary score threshold
5. Smoke run (optional, for full validation)

Exit codes:
  0 = All gates passed
  1 = Gate failure (details printed)
  2 = Configuration/setup error

Usage:
  python scripts/ci_gate.py              # Quick check (no smoke)
  python scripts/ci_gate.py --smoke      # Full check with smoke run
  python scripts/ci_gate.py --strict     # Fail on warnings too
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paradigm_factory.v2.eval_harness import (
    IntegrityPreflight,
    EXPECTED_GOLD_HASH,
    GOLD_FILE_VERSION,
    INTEGRITY_RULESET,
    INTEGRITY_RULESET_CHANGELOG
)


@dataclass
class GateResult:
    """Result of a single gate check."""
    name: str
    passed: bool
    message: str
    is_warning: bool = False


class CIGate:
    """CI gate runner."""

    # Thresholds (can be overridden by config)
    QUARANTINE_THRESHOLD = 0.20  # Max 20% quarantine rate
    CANARY_THRESHOLD = 0.90      # Min 90% canary score
    SMOKE_STEPS = 500            # Steps for smoke run
    SMOKE_SLACK_ZERO_BY = 400    # Hard slack must cross zero by this step

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.results: List[GateResult] = []

    def check_gold_hash(self) -> GateResult:
        """Verify gold hash matches both file and code constant."""
        preflight = IntegrityPreflight()

        if not preflight.gold_file.exists():
            return GateResult(
                name="gold_hash",
                passed=False,
                message=f"Gold file not found: {preflight.gold_file}"
            )

        file_match, code_match, embedded_hash, actual_hash = preflight.verify_gold_hash()

        if not file_match:
            return GateResult(
                name="gold_hash",
                passed=False,
                message=f"Gold file content hash mismatch. Embedded: {embedded_hash[:16]}..., Actual: {actual_hash[:16]}..."
            )

        if not code_match:
            return GateResult(
                name="gold_hash",
                passed=False,
                message=f"TRUST ANCHOR MISMATCH: Actual hash {actual_hash[:16]}... != EXPECTED_GOLD_HASH {EXPECTED_GOLD_HASH[:16]}..."
            )

        return GateResult(
            name="gold_hash",
            passed=True,
            message=f"Hash verified: {actual_hash[:16]}... (gold={GOLD_FILE_VERSION}, rules={INTEGRITY_RULESET})"
        )

    def check_integrity_gate(self) -> GateResult:
        """Run integrity gate on eval items."""
        preflight = IntegrityPreflight()
        eval_file = PROJECT_ROOT / "paradigm_factory" / "v2" / "evals" / "eval_multi_sense_retrieval.jsonl"

        if not eval_file.exists():
            return GateResult(
                name="integrity_gate",
                passed=False,
                message=f"Eval file not found: {eval_file}"
            )

        # Load eval items
        items = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))

        if not items:
            return GateResult(
                name="integrity_gate",
                passed=False,
                message="No eval items found"
            )

        # Run preflight
        result = preflight.run_preflight(items)

        if not result.passed:
            return GateResult(
                name="integrity_gate",
                passed=False,
                message=f"Preflight failed: {result.error_message}"
            )

        return GateResult(
            name="integrity_gate",
            passed=True,
            message=f"Preflight passed: {result.items_valid} valid, {result.items_quarantined} quarantined ({result.quarantine_rate:.1%})"
        )

    def check_quarantine_rate(self) -> GateResult:
        """Check quarantine rate is below threshold."""
        preflight = IntegrityPreflight()
        eval_file = PROJECT_ROOT / "paradigm_factory" / "v2" / "evals" / "eval_multi_sense_retrieval.jsonl"

        if not eval_file.exists():
            return GateResult(
                name="quarantine_rate",
                passed=True,
                message="Skipped (no eval file)",
                is_warning=True
            )

        items = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))

        result = preflight.run_preflight(items)

        if result.quarantine_rate > self.QUARANTINE_THRESHOLD:
            return GateResult(
                name="quarantine_rate",
                passed=False,
                message=f"Quarantine rate {result.quarantine_rate:.1%} exceeds threshold {self.QUARANTINE_THRESHOLD:.0%}"
            )

        return GateResult(
            name="quarantine_rate",
            passed=True,
            message=f"Quarantine rate {result.quarantine_rate:.1%} <= {self.QUARANTINE_THRESHOLD:.0%}"
        )

    def check_canary_score(self) -> GateResult:
        """Check canary score meets threshold."""
        # In a real implementation, this would load and run the canary eval
        # For now, we check if canary results exist
        canary_file = PROJECT_ROOT / "paradigm_factory" / "v2" / "evals_gated" / "canary_results.json"

        if not canary_file.exists():
            return GateResult(
                name="canary_score",
                passed=True,
                message="Skipped (no canary results file - run full eval first)",
                is_warning=True
            )

        with open(canary_file, 'r', encoding='utf-8') as f:
            canary = json.load(f)

        score = canary.get('score', 0.0)
        if score < self.CANARY_THRESHOLD:
            return GateResult(
                name="canary_score",
                passed=False,
                message=f"Canary score {score:.3f} below threshold {self.CANARY_THRESHOLD:.2f}"
            )

        return GateResult(
            name="canary_score",
            passed=True,
            message=f"Canary score {score:.3f} >= {self.CANARY_THRESHOLD:.2f}"
        )

    def check_smoke_run(self) -> GateResult:
        """Run a short training smoke test."""
        # This would actually run training for SMOKE_STEPS
        # For now, check if a recent smoke result exists
        smoke_file = PROJECT_ROOT / "paradigm_factory" / "v2" / "evals_gated" / "smoke_results.json"

        if not smoke_file.exists():
            return GateResult(
                name="smoke_run",
                passed=True,
                message="Skipped (no smoke results - run: python scripts/run_smoke.py)",
                is_warning=True
            )

        with open(smoke_file, 'r', encoding='utf-8') as f:
            smoke = json.load(f)

        slack_zero_step = smoke.get('hard_slack_zero_step')
        if slack_zero_step is None:
            return GateResult(
                name="smoke_run",
                passed=False,
                message="Hard slack never crossed zero in smoke run"
            )

        if slack_zero_step > self.SMOKE_SLACK_ZERO_BY:
            return GateResult(
                name="smoke_run",
                passed=False,
                message=f"Hard slack crossed zero at step {slack_zero_step}, expected by step {self.SMOKE_SLACK_ZERO_BY}"
            )

        return GateResult(
            name="smoke_run",
            passed=True,
            message=f"Hard slack crossed zero at step {slack_zero_step} (threshold: {self.SMOKE_SLACK_ZERO_BY})"
        )

    def run_all(self, include_smoke: bool = False) -> bool:
        """Run all gate checks."""
        print("=" * 60)
        print("CI GATE CHECK")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Mode: {'strict' if self.strict else 'normal'}")
        print("=" * 60)

        # Core gates (always run)
        self.results.append(self.check_gold_hash())
        self.results.append(self.check_integrity_gate())
        self.results.append(self.check_quarantine_rate())
        self.results.append(self.check_canary_score())

        # Optional smoke run
        if include_smoke:
            self.results.append(self.check_smoke_run())

        # Print results
        print("\nGate Results:")
        print("-" * 60)

        all_passed = True
        warnings = 0

        for result in self.results:
            if result.passed:
                if result.is_warning:
                    status = "WARN"
                    warnings += 1
                else:
                    status = "PASS"
            else:
                status = "FAIL"
                all_passed = False

            print(f"  [{status}] {result.name}: {result.message}")

        print("-" * 60)

        # Summary
        if not all_passed:
            print("\n[X] CI GATE FAILED")
            return False

        if warnings > 0 and self.strict:
            print(f"\n[!] CI GATE FAILED (strict mode, {warnings} warnings)")
            return False

        if warnings > 0:
            print(f"\n[OK] CI GATE PASSED ({warnings} warnings)")
        else:
            print("\n[OK] CI GATE PASSED")

        return True


def main():
    parser = argparse.ArgumentParser(description="CI Gate Check")
    parser.add_argument("--smoke", action="store_true", help="Include smoke run check")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings")
    args = parser.parse_args()

    gate = CIGate(strict=args.strict)
    passed = gate.run_all(include_smoke=args.smoke)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
