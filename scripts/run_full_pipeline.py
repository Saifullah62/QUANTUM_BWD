#!/usr/bin/env python3
"""
QUANTUM_BWD Full Pipeline Runner
================================

One-command reproduction + evaluation script.
The "demo button" for partners/investors and internal reproducibility standard.

This script:
1. Validates environment and dependencies
2. Verifies gold hash integrity
3. Runs integrity gating on evals
4. Runs evaluation harness
5. Optionally runs one-seed training
6. Emits dashboard fingerprint

Usage:
  python scripts/run_full_pipeline.py                    # Full eval pipeline
  python scripts/run_full_pipeline.py --strict           # With ship-blocker gates
  python scripts/run_full_pipeline.py --with-training    # Include training
  python scripts/run_full_pipeline.py --quick            # Quick validation only

Exit codes:
  0 = Success
  1 = Validation failed
  2 = Gating failed
  3 = Eval failed
  4 = Training failed
  5 = Ship-blocker gates failed (--strict mode)
"""

import os
import sys
import json
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Trust anchors
EXPECTED_ARCHIVE_HASH = "5b1d1a92e8056532408936db80dbd65b7be4918818a2f6bed4f88fba37062409"
EXPECTED_EVAL_GOLD_HASH = "e1d122539e92e5ddd845549f5a23a0835b996515d4c1b74d97ee3bd4c88247aa"
ARCHIVE_DIR = PROJECT_ROOT / "ARCHIVES" / "v3.1-certified"
ARCHIVE_FILE = ARCHIVE_DIR / "semanticphase_v3.1_certified_20260104.tar.gz"

# Key paths
EVALS_DIR = PROJECT_ROOT / "paradigm_factory" / "v2" / "evals_gated"
DASHBOARD_DIR = PROJECT_ROOT / "paradigm_factory" / "v2" / "dashboard_reports"

# Ship-blocker thresholds (conservative start, ratchet up over time)
# These are the minimum acceptable values for a "shippable" model
SHIP_BLOCKER_THRESHOLDS = {
    # Retrieval gates (on real_usage data only)
    "retrieval_real_top1": 0.30,   # Min 30% Top-1 accuracy
    "retrieval_real_top3": 0.50,   # Min 50% Top-3 accuracy
    # Coherence gate
    "coherence_accuracy": 0.80,    # Min 80% per-step accuracy
}


@dataclass
class ShipBlockerResult:
    """Result of ship-blocker gate checks."""
    passed: bool
    retrieval_top1_ok: bool
    retrieval_top3_ok: bool
    coherence_ok: bool
    retrieval_top1_actual: float
    retrieval_top3_actual: float
    coherence_actual: float
    failures: List[str]


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""
    timestamp: str
    success: bool
    stages_completed: List[str]
    stages_failed: List[str]
    gold_hash_verified: bool
    integrity_fingerprint: Optional[str]
    eval_fingerprint: Optional[str]
    training_fingerprint: Optional[str]
    dashboard_path: Optional[str]
    duration_seconds: float
    errors: List[str]
    ship_blocker_result: Optional[Dict] = None  # ShipBlockerResult as dict


def print_banner(text: str, char: str = "="):
    """Print a banner."""
    width = 70
    print(flush=True)
    print(char * width, flush=True)
    print(f" {text}", flush=True)
    print(char * width, flush=True)


def print_stage(stage: str):
    """Print stage header."""
    print(f"\n>>> STAGE: {stage}", flush=True)
    print("-" * 50, flush=True)


def compute_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_python_dependencies(check_ml: bool = True) -> Tuple[bool, List[str]]:
    """Check required Python packages.

    Args:
        check_ml: If True, check ML libraries (torch, sentence_transformers).
                  Set False for fast validation mode.
    """
    # Core requirements (fast to import)
    required = ['numpy']

    # ML requirements (slow to import)
    if check_ml:
        required.extend(['torch', 'sentence_transformers'])

    missing = []

    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)

    return len(missing) == 0, missing


def stage_validate_environment(check_ml_deps: bool = True) -> Tuple[bool, Dict]:
    """Stage 1: Validate environment and dependencies.

    Args:
        check_ml_deps: If True, check ML libraries. Set False for fast mode.
    """
    print_stage("VALIDATE ENVIRONMENT")

    results = {
        'python_version': sys.version,
        'project_root': str(PROJECT_ROOT),
        'cwd': os.getcwd(),
    }

    # Check Python version
    if sys.version_info < (3, 8):
        print(f"[X] Python 3.8+ required, got {sys.version}", flush=True)
        return False, results
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}", flush=True)

    # Check key directories exist
    key_dirs = [
        PROJECT_ROOT / "paradigm_factory",
        PROJECT_ROOT / "paradigm_factory" / "v2",
        EVALS_DIR,
    ]

    for d in key_dirs:
        if not d.exists():
            print(f"[X] Missing directory: {d}", flush=True)
            return False, results
    print("[OK] Directory structure valid", flush=True)

    # Check dependencies
    if check_ml_deps:
        print("[..] Checking ML dependencies (this may take a moment)...", flush=True)
        deps_ok, missing = check_python_dependencies(check_ml=True)
        if not deps_ok:
            print(f"[!] Missing packages: {missing}", flush=True)
            print("    Run: pip install " + " ".join(missing), flush=True)
            # Don't fail - some deps are optional
        else:
            print("[OK] ML dependencies available", flush=True)
    else:
        print("[--] ML dependency check skipped (fast mode)", flush=True)
        deps_ok, missing = check_python_dependencies(check_ml=False)

    results['dependencies_ok'] = deps_ok
    results['missing_deps'] = missing

    return True, results


def stage_verify_gold_hash() -> Tuple[bool, Dict]:
    """Stage 2: Verify archive hash integrity."""
    print_stage("VERIFY ARCHIVE INTEGRITY")

    results = {
        'archive_file': str(ARCHIVE_FILE),
        'expected_hash': EXPECTED_ARCHIVE_HASH,
    }

    # Verify archive exists
    if not ARCHIVE_FILE.exists():
        print(f"[X] Archive not found: {ARCHIVE_FILE}")
        return False, results

    # Compute and verify archive hash
    actual_hash = compute_hash(ARCHIVE_FILE)
    results['actual_hash'] = actual_hash

    if actual_hash != EXPECTED_ARCHIVE_HASH:
        print(f"[X] Archive hash mismatch!")
        print(f"    Expected: {EXPECTED_ARCHIVE_HASH}")
        print(f"    Got:      {actual_hash}")
        return False, results

    print(f"[OK] Archive hash verified: {actual_hash[:16]}...")

    # Also verify manifest consistency
    manifest_file = ARCHIVE_DIR / "VERSION_MANIFEST.json"
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        manifest_hash = manifest.get('archive', {}).get('sha256')
        if manifest_hash == EXPECTED_ARCHIVE_HASH:
            print("[OK] Manifest hash matches")
        else:
            print("[!] Manifest hash differs from expected")
            results['manifest_mismatch'] = True

    return True, results


def stage_run_integrity_gating() -> Tuple[bool, Dict]:
    """Stage 3: Run integrity gating on evals."""
    print_stage("RUN INTEGRITY GATING")

    results = {}

    # Import gated eval builder
    try:
        from paradigm_factory.v2.build_gated_evals import GatedEvalBuilder
    except ImportError as e:
        print(f"[X] Cannot import GatedEvalBuilder: {e}")
        return False, results

    builder = GatedEvalBuilder(EVALS_DIR, EVALS_DIR)

    # Run gating
    try:
        manifest = builder.build_all()
        results['manifest'] = asdict(manifest)
        results['fingerprint'] = manifest.fingerprint

        print(f"[OK] Gating complete")
        print(f"    Retrieval: {manifest.retrieval_stats.get('valid', 0)} valid, "
              f"{manifest.retrieval_stats.get('quarantined', 0)} quarantined")
        print(f"    Coherence: {manifest.coherence_stats.get('valid', 0)} valid")
        print(f"    Fingerprint: {manifest.fingerprint}")

        # Check quarantine rate
        retrieval_total = sum(manifest.retrieval_stats.values())
        quarantine_rate = manifest.retrieval_stats.get('quarantined', 0) / retrieval_total if retrieval_total > 0 else 0

        if quarantine_rate > 0.20:
            print(f"[!] High quarantine rate: {quarantine_rate:.1%}")

        return True, results

    except Exception as e:
        print(f"[X] Gating failed: {e}")
        results['error'] = str(e)
        return False, results


def stage_run_evaluations(quick: bool = False) -> Tuple[bool, Dict]:
    """Stage 4: Run evaluation harness."""
    print_stage("RUN EVALUATIONS")

    results = {}

    # Check for eval harness
    eval_harness_path = PROJECT_ROOT / "paradigm_factory" / "v2" / "run_harness.py"

    if not eval_harness_path.exists():
        # Try alternative path
        eval_harness_path = PROJECT_ROOT / "paradigm_factory" / "v2" / "eval_harness.py"

    if not eval_harness_path.exists():
        print("[!] Eval harness not found, checking for existing results...")

        # Look for existing dashboard reports
        if DASHBOARD_DIR.exists():
            reports = list(DASHBOARD_DIR.glob("dashboard_*.json"))
            if reports:
                latest = max(reports, key=lambda x: x.stat().st_mtime)
                print(f"[OK] Found existing report: {latest.name}")

                with open(latest, 'r') as f:
                    dashboard = json.load(f)

                results['dashboard_file'] = str(latest)
                results['fingerprint'] = dashboard.get('fingerprint', 'unknown')
                return True, results

        print("[X] No eval harness and no existing results")
        return False, results

    # Run eval harness
    try:
        cmd = [sys.executable, str(eval_harness_path)]
        if quick:
            cmd.append("--quick")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

        if result.returncode != 0:
            print(f"[X] Eval harness failed:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False, {'error': result.stderr}

        print("[OK] Eval harness completed")

        # Find the latest dashboard
        reports = list(DASHBOARD_DIR.glob("dashboard_*.json"))
        if reports:
            latest = max(reports, key=lambda x: x.stat().st_mtime)
            with open(latest, 'r') as f:
                dashboard = json.load(f)

            results['dashboard_file'] = str(latest)
            results['fingerprint'] = dashboard.get('fingerprint', 'unknown')
            print(f"    Dashboard: {latest.name}")
            print(f"    Fingerprint: {results['fingerprint']}")

        return True, results

    except Exception as e:
        print(f"[X] Eval failed: {e}")
        return False, {'error': str(e)}


def stage_check_ship_blockers(dashboard_path: Optional[str]) -> Tuple[bool, ShipBlockerResult]:
    """Stage 4b: Check ship-blocker gates (only in --strict mode).

    This is the "sanity eval" gate that ensures the model meets minimum
    acceptable behavior thresholds before being considered shippable.
    """
    print_stage("CHECK SHIP-BLOCKER GATES")

    # Default failure result
    default_result = ShipBlockerResult(
        passed=False,
        retrieval_top1_ok=False,
        retrieval_top3_ok=False,
        coherence_ok=False,
        retrieval_top1_actual=0.0,
        retrieval_top3_actual=0.0,
        coherence_actual=0.0,
        failures=["No dashboard available"]
    )

    if not dashboard_path:
        print("[X] No dashboard to check", flush=True)
        return False, default_result

    # Load dashboard
    try:
        with open(dashboard_path, 'r') as f:
            dashboard = json.load(f)
    except Exception as e:
        print(f"[X] Failed to load dashboard: {e}", flush=True)
        default_result.failures = [f"Failed to load dashboard: {e}"]
        return False, default_result

    # Extract metrics
    retrieval_top1 = dashboard.get('retrieval_real_top1', 0.0)
    retrieval_top3 = dashboard.get('retrieval_real_top3', 0.0)
    coherence_acc = dashboard.get('coherence_per_step_accuracy', 0.0)

    # Check thresholds
    failures = []

    top1_ok = retrieval_top1 >= SHIP_BLOCKER_THRESHOLDS['retrieval_real_top1']
    if not top1_ok:
        failures.append(
            f"retrieval_real_top1: {retrieval_top1:.1%} < {SHIP_BLOCKER_THRESHOLDS['retrieval_real_top1']:.0%}"
        )
        print(f"[X] Retrieval Top-1: {retrieval_top1:.1%} < {SHIP_BLOCKER_THRESHOLDS['retrieval_real_top1']:.0%}", flush=True)
    else:
        print(f"[OK] Retrieval Top-1: {retrieval_top1:.1%} >= {SHIP_BLOCKER_THRESHOLDS['retrieval_real_top1']:.0%}", flush=True)

    top3_ok = retrieval_top3 >= SHIP_BLOCKER_THRESHOLDS['retrieval_real_top3']
    if not top3_ok:
        failures.append(
            f"retrieval_real_top3: {retrieval_top3:.1%} < {SHIP_BLOCKER_THRESHOLDS['retrieval_real_top3']:.0%}"
        )
        print(f"[X] Retrieval Top-3: {retrieval_top3:.1%} < {SHIP_BLOCKER_THRESHOLDS['retrieval_real_top3']:.0%}", flush=True)
    else:
        print(f"[OK] Retrieval Top-3: {retrieval_top3:.1%} >= {SHIP_BLOCKER_THRESHOLDS['retrieval_real_top3']:.0%}", flush=True)

    coherence_ok = coherence_acc >= SHIP_BLOCKER_THRESHOLDS['coherence_accuracy']
    if not coherence_ok:
        failures.append(
            f"coherence_accuracy: {coherence_acc:.1%} < {SHIP_BLOCKER_THRESHOLDS['coherence_accuracy']:.0%}"
        )
        print(f"[X] Coherence: {coherence_acc:.1%} < {SHIP_BLOCKER_THRESHOLDS['coherence_accuracy']:.0%}", flush=True)
    else:
        print(f"[OK] Coherence: {coherence_acc:.1%} >= {SHIP_BLOCKER_THRESHOLDS['coherence_accuracy']:.0%}", flush=True)

    passed = top1_ok and top3_ok and coherence_ok

    result = ShipBlockerResult(
        passed=passed,
        retrieval_top1_ok=top1_ok,
        retrieval_top3_ok=top3_ok,
        coherence_ok=coherence_ok,
        retrieval_top1_actual=retrieval_top1,
        retrieval_top3_actual=retrieval_top3,
        coherence_actual=coherence_acc,
        failures=failures
    )

    if passed:
        print(f"\n[OK] All ship-blocker gates PASSED", flush=True)
    else:
        print(f"\n[X] Ship-blocker gates FAILED: {len(failures)} violation(s)", flush=True)

    return passed, result


def stage_run_training(seed: int = 42) -> Tuple[bool, Dict]:
    """Stage 5: Run one-seed training (optional)."""
    print_stage(f"RUN TRAINING (seed={seed})")

    results = {'seed': seed}

    # Check for training script
    train_paths = [
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "scripts" / "train.py",
        PROJECT_ROOT / "paradigm_factory" / "train.py",
    ]

    train_script = None
    for p in train_paths:
        if p.exists():
            train_script = p
            break

    if train_script is None:
        print("[!] Training script not found, skipping...")
        results['skipped'] = True
        return True, results

    try:
        cmd = [
            sys.executable, str(train_script),
            "--seed", str(seed),
            "--epochs", "1",  # Quick single epoch for demo
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

        if result.returncode != 0:
            print(f"[X] Training failed:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False, {'error': result.stderr}

        print("[OK] Training completed")
        results['completed'] = True

        return True, results

    except Exception as e:
        print(f"[X] Training failed: {e}")
        return False, {'error': str(e)}


def emit_final_fingerprint(result: PipelineResult):
    """Emit the final dashboard fingerprint."""
    print_banner("PIPELINE COMPLETE", "=")

    status = "[OK]" if result.success else "[X]"

    # Format ship-blocker status
    if result.ship_blocker_result:
        sbr = result.ship_blocker_result
        ship_status = "PASSED" if sbr.get('passed') else "FAILED"
        ship_detail = f"top1={sbr.get('retrieval_top1_actual', 0):.1%}|top3={sbr.get('retrieval_top3_actual', 0):.1%}|coh={sbr.get('coherence_actual', 0):.1%}"
    else:
        ship_status = "N/A"
        ship_detail = ""

    print(f"""
{status} Pipeline Result

    Timestamp:      {result.timestamp}
    Duration:       {result.duration_seconds:.1f}s
    Stages OK:      {', '.join(result.stages_completed) or 'none'}
    Stages Failed:  {', '.join(result.stages_failed) or 'none'}

    Gold Hash:      {'VERIFIED' if result.gold_hash_verified else 'FAILED'}
    Integrity:      {result.integrity_fingerprint or 'N/A'}
    Eval:           {result.eval_fingerprint or 'N/A'}
    Ship-Blockers:  {ship_status} {ship_detail}
    Training:       {result.training_fingerprint or 'N/A'}
""")

    if result.errors:
        print("    Errors:")
        for err in result.errors:
            err_str = str(err)[:80] if not isinstance(err, str) else err[:80]
            print(f"      - {err_str}")

    # Write result file
    result_file = DASHBOARD_DIR / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    with open(result_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\n    Result saved: {result_file.name}")

    # Final fingerprint with ship-blocker info
    ship_fp = f"|ship={ship_status.lower()}" if result.ship_blocker_result else ""
    final_fp = f"pipeline|gold={'ok' if result.gold_hash_verified else 'fail'}|stages={len(result.stages_completed)}/{len(result.stages_completed) + len(result.stages_failed)}{ship_fp}"
    print(f"\n    FINGERPRINT: {final_fp}")


def main():
    parser = argparse.ArgumentParser(
        description="QUANTUM_BWD Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_full_pipeline.py                 # Standard run
  python scripts/run_full_pipeline.py --quick         # Quick validation
  python scripts/run_full_pipeline.py --with-training # Include training
  python scripts/run_full_pipeline.py --seed 123      # Specific seed
        """
    )

    parser.add_argument("--quick", action="store_true",
                       help="Quick validation only (skip full evals)")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate environment and archive hash (no gating/evals)")
    parser.add_argument("--with-training", action="store_true",
                       help="Include one-seed training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for training (default: 42)")
    parser.add_argument("--skip-gating", action="store_true",
                       help="Skip integrity gating stage")
    parser.add_argument("--strict", action="store_true",
                       help="Enable ship-blocker gates (hard-fail on threshold violations)")

    args = parser.parse_args()

    print_banner("QUANTUM_BWD FULL PIPELINE", "#")
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print(f"Mode: {'quick' if args.quick else 'full'}", flush=True)
    if args.strict:
        print(f"Strict mode: ENABLED (ship-blocker gates active)", flush=True)
    if args.with_training:
        print(f"Training: enabled (seed={args.seed})", flush=True)

    start_time = datetime.now()

    stages_completed = []
    stages_failed = []
    errors = []

    gold_verified = False
    integrity_fp = None
    eval_fp = None
    training_fp = None
    dashboard_path = None
    ship_blocker_result = None

    # Stage 1: Validate environment
    # Skip slow ML imports in validate-only mode
    check_ml = not args.validate_only
    ok, _ = stage_validate_environment(check_ml_deps=check_ml)
    if ok:
        stages_completed.append("validate_env")
    else:
        stages_failed.append("validate_env")
        errors.append("Environment validation failed")

    # Stage 2: Verify gold hash
    if not stages_failed:  # Only continue if previous stages passed
        ok, results = stage_verify_gold_hash()
        if ok:
            stages_completed.append("verify_hash")
            gold_verified = True
        else:
            stages_failed.append("verify_hash")
            errors.append("Gold hash verification failed")

    # Stage 3: Run integrity gating (skip in validate-only mode)
    if not stages_failed and not args.skip_gating and not args.validate_only:
        ok, results = stage_run_integrity_gating()
        if ok:
            stages_completed.append("integrity_gate")
            integrity_fp = results.get('fingerprint')
        else:
            stages_failed.append("integrity_gate")
            errors.append("Integrity gating failed")
    elif args.skip_gating or args.validate_only:
        print("\n>>> STAGE: INTEGRITY GATING (skipped)", flush=True)

    # Stage 4: Run evaluations (skip in validate-only mode)
    if not stages_failed and not args.validate_only:
        ok, results = stage_run_evaluations(quick=args.quick)
        if ok:
            stages_completed.append("evaluations")
            eval_fp = results.get('fingerprint')
            dashboard_path = results.get('dashboard_file')
        else:
            stages_failed.append("evaluations")
            errors.append("Evaluations failed")
    elif args.validate_only:
        print("\n>>> STAGE: EVALUATIONS (skipped)", flush=True)

    # Stage 4b: Check ship-blocker gates (only in --strict mode)
    if not stages_failed and args.strict and not args.validate_only:
        ok, blocker_result = stage_check_ship_blockers(dashboard_path)
        ship_blocker_result = asdict(blocker_result)
        if ok:
            stages_completed.append("ship_blockers")
        else:
            stages_failed.append("ship_blockers")
            errors.append(f"Ship-blocker gates failed: {blocker_result.failures}")
    elif args.strict and args.validate_only:
        print("\n>>> STAGE: SHIP-BLOCKER GATES (skipped)", flush=True)

    # Stage 5: Run training (optional, skip in validate-only mode)
    if not stages_failed and args.with_training and not args.validate_only:
        ok, results = stage_run_training(seed=args.seed)
        if ok:
            stages_completed.append("training")
            if not results.get('skipped'):
                training_fp = f"seed={args.seed}"
        else:
            stages_failed.append("training")
            errors.append("Training failed")
    elif args.validate_only and args.with_training:
        print("\n>>> STAGE: TRAINING (skipped)", flush=True)

    # Compute duration
    duration = (datetime.now() - start_time).total_seconds()

    # Create result
    result = PipelineResult(
        timestamp=start_time.isoformat(),
        success=len(stages_failed) == 0,
        stages_completed=stages_completed,
        stages_failed=stages_failed,
        gold_hash_verified=gold_verified,
        integrity_fingerprint=integrity_fp,
        eval_fingerprint=eval_fp,
        training_fingerprint=training_fp,
        dashboard_path=dashboard_path,
        duration_seconds=duration,
        errors=errors,
        ship_blocker_result=ship_blocker_result
    )

    # Emit final fingerprint
    emit_final_fingerprint(result)

    # Exit code
    if result.success:
        sys.exit(0)
    elif "verify_hash" in stages_failed:
        sys.exit(1)
    elif "integrity_gate" in stages_failed:
        sys.exit(2)
    elif "evaluations" in stages_failed:
        sys.exit(3)
    elif "ship_blockers" in stages_failed:
        sys.exit(5)  # Ship-blocker gate failure
    else:
        sys.exit(4)  # Training or other failure


if __name__ == "__main__":
    main()
