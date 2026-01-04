#!/usr/bin/env python3
"""
Test script for ship-blocker gates.

Verifies that:
1. v3.1-certified dashboard PASSES the gates
2. A deliberately broken dashboard FAILS the gates
"""

import sys
import json
import tempfile
import importlib.util
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from run_full_pipeline using spec loader
spec = importlib.util.spec_from_file_location(
    "run_full_pipeline",
    SCRIPT_DIR / "run_full_pipeline.py"
)
pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_module)

stage_check_ship_blockers = pipeline_module.stage_check_ship_blockers
SHIP_BLOCKER_THRESHOLDS = pipeline_module.SHIP_BLOCKER_THRESHOLDS
DASHBOARD_DIR = pipeline_module.DASHBOARD_DIR


def test_passing_dashboard():
    """Test that v3.1-certified dashboard passes."""
    print("=" * 60)
    print("TEST 1: v3.1-certified dashboard should PASS")
    print("=" * 60)

    # Find the latest real dashboard
    dashboards = list(DASHBOARD_DIR.glob("dashboard_*.json"))
    if not dashboards:
        print("[SKIP] No dashboard files found")
        return None

    latest = max(dashboards, key=lambda x: x.stat().st_mtime)
    print(f"Using: {latest.name}")

    # Load and check metrics
    with open(latest, 'r') as f:
        dashboard = json.load(f)

    print(f"  retrieval_real_top1: {dashboard.get('retrieval_real_top1', 0):.1%}")
    print(f"  retrieval_real_top3: {dashboard.get('retrieval_real_top3', 0):.1%}")
    print(f"  coherence_accuracy:  {dashboard.get('coherence_per_step_accuracy', 0):.1%}")
    print(f"Thresholds: top1>={SHIP_BLOCKER_THRESHOLDS['retrieval_real_top1']:.0%}, "
          f"top3>={SHIP_BLOCKER_THRESHOLDS['retrieval_real_top3']:.0%}, "
          f"coh>={SHIP_BLOCKER_THRESHOLDS['coherence_accuracy']:.0%}")

    # Run the gate check
    passed, result = stage_check_ship_blockers(str(latest))

    if passed:
        print("\n[PASS] v3.1-certified dashboard passes ship-blocker gates")
        return True
    else:
        print(f"\n[FAIL] v3.1-certified dashboard should have passed!")
        print(f"  Failures: {result.failures}")
        return False


def test_failing_dashboard():
    """Test that a deliberately broken dashboard fails."""
    print("\n" + "=" * 60)
    print("TEST 2: Broken dashboard should FAIL")
    print("=" * 60)

    # Create a deliberately broken dashboard
    broken_dashboard = {
        "retrieval_real_top1": 0.10,  # 10% - below 30% threshold
        "retrieval_real_top3": 0.30,  # 30% - below 50% threshold
        "coherence_per_step_accuracy": 0.50,  # 50% - below 80% threshold
    }

    print(f"  retrieval_real_top1: {broken_dashboard['retrieval_real_top1']:.1%}")
    print(f"  retrieval_real_top3: {broken_dashboard['retrieval_real_top3']:.1%}")
    print(f"  coherence_accuracy:  {broken_dashboard['coherence_per_step_accuracy']:.1%}")
    print(f"Thresholds: top1>={SHIP_BLOCKER_THRESHOLDS['retrieval_real_top1']:.0%}, "
          f"top3>={SHIP_BLOCKER_THRESHOLDS['retrieval_real_top3']:.0%}, "
          f"coh>={SHIP_BLOCKER_THRESHOLDS['coherence_accuracy']:.0%}")

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(broken_dashboard, f)
        temp_path = f.name

    # Run the gate check
    passed, result = stage_check_ship_blockers(temp_path)

    # Clean up
    Path(temp_path).unlink()

    if not passed:
        print(f"\n[PASS] Broken dashboard correctly failed ship-blocker gates")
        print(f"  Expected failures: 3, Got: {len(result.failures)}")
        return True
    else:
        print(f"\n[FAIL] Broken dashboard should have failed!")
        return False


def main():
    print("Ship-Blocker Gate Tests")
    print("=" * 60)
    print(f"Thresholds configured:")
    for key, value in SHIP_BLOCKER_THRESHOLDS.items():
        print(f"  {key}: {value:.0%}")
    print()

    results = []

    # Test 1: Passing dashboard
    result1 = test_passing_dashboard()
    if result1 is not None:
        results.append(result1)

    # Test 2: Failing dashboard
    result2 = test_failing_dashboard()
    results.append(result2)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if all(results):
        print("\n[OK] All ship-blocker gate tests PASSED")
        sys.exit(0)
    else:
        print("\n[X] Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
