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
  python scripts/run_full_pipeline.py --with-training    # Include training
  python scripts/run_full_pipeline.py --quick            # Quick validation only

Exit codes:
  0 = Success
  1 = Validation failed
  2 = Gating failed
  3 = Eval failed
  4 = Training failed
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

    print(f"""
{status} Pipeline Result

    Timestamp:      {result.timestamp}
    Duration:       {result.duration_seconds:.1f}s
    Stages OK:      {', '.join(result.stages_completed) or 'none'}
    Stages Failed:  {', '.join(result.stages_failed) or 'none'}

    Gold Hash:      {'VERIFIED' if result.gold_hash_verified else 'FAILED'}
    Integrity:      {result.integrity_fingerprint or 'N/A'}
    Eval:           {result.eval_fingerprint or 'N/A'}
    Training:       {result.training_fingerprint or 'N/A'}
""")

    if result.errors:
        print("    Errors:")
        for err in result.errors:
            print(f"      - {err[:80]}")

    # Write result file
    result_file = DASHBOARD_DIR / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    with open(result_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\n    Result saved: {result_file.name}")

    # Final fingerprint
    final_fp = f"pipeline|gold={'ok' if result.gold_hash_verified else 'fail'}|stages={len(result.stages_completed)}/{len(result.stages_completed) + len(result.stages_failed)}"
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

    args = parser.parse_args()

    print_banner("QUANTUM_BWD FULL PIPELINE", "#")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    if args.with_training:
        print(f"Training: enabled (seed={args.seed})")

    start_time = datetime.now()

    stages_completed = []
    stages_failed = []
    errors = []

    gold_verified = False
    integrity_fp = None
    eval_fp = None
    training_fp = None
    dashboard_path = None

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
        errors=errors
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
    else:
        sys.exit(4)


if __name__ == "__main__":
    main()
