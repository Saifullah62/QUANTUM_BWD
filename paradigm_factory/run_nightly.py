#!/usr/bin/env python3
"""
Paradigm Factory - Phase E: Nightly Orchestrator
=================================================

Runs the complete nightly pipeline:
1. Generate polysemy candidates (Phase A)
2. Embed, dedupe, cluster (Phase B)
3. Generate Lindblad twins (Phase C)
4. Generate eval pack (Phase D)
5. Run evals on latest checkpoint (optional)
6. Store summary in Memory Box

Designed to run on gpu-swarm as coordinator, pushing heavy work
to gpu-ramp when needed.

Usage:
    python run_nightly.py --date 20260102 --words bank spring bat
    python run_nightly.py --full  # All default words
"""

import argparse
import json
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fleet services
FLEET_BASE = "http://159.203.35.45"
MEMORY_BOX = f"{FLEET_BASE}:8004"
ORCHESTRATOR = f"{FLEET_BASE}:8011"


def log(msg: str):
    """Timestamped log."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def run_phase(script_name: str, args: List[str], phase_name: str) -> bool:
    """Run a phase script and return success status."""
    log(f"Starting {phase_name}...")

    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max
        )
        if result.returncode == 0:
            log(f"  {phase_name} completed successfully")
            return True
        else:
            log(f"  {phase_name} failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        log(f"  {phase_name} timed out")
        return False
    except Exception as e:
        log(f"  {phase_name} error: {e}")
        return False


def store_to_memory_box(session_id: str, data: Dict) -> bool:
    """Store summary to Memory Box for agent reference."""
    try:
        resp = requests.post(
            f"{MEMORY_BOX}/memory/add",
            json={
                "session_id": session_id,
                "role": "system",
                "text": json.dumps(data)
            },
            timeout=30
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        log(f"  Failed to store to Memory Box: {e}")
        return False


def submit_async_task(task_type: str, payload: Dict, priority: int = 5) -> Optional[str]:
    """Submit async task to orchestrator."""
    try:
        resp = requests.post(
            f"{ORCHESTRATOR}/task/submit",
            json={
                "type": task_type,
                "payload": payload,
                "priority": priority
            },
            timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("task_id")
    except Exception as e:
        log(f"  Failed to submit task: {e}")
        return None


def run_nightly_pipeline(
    date_str: str,
    words: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    skip_eval: bool = False,
    contexts_per_sense: int = 3,
    twins_per_text: int = 3
) -> Dict[str, Any]:
    """
    Run the complete nightly pipeline.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "date": date_str,
        "start_time": datetime.now().isoformat(),
        "phases": {},
        "artifacts": {},
        "metrics": {}
    }

    # Phase A: Generate polysemy candidates
    polysemy_output = output_dir / f"polysemy_{date_str}.jsonl"
    phase_a_args = [
        "--output", str(polysemy_output),
        "--contexts-per-sense", str(contexts_per_sense)
    ]
    if words:
        phase_a_args.extend(["--words"] + words)

    phase_a_success = run_phase(
        "generate_polysemy.py",
        phase_a_args,
        "Phase A: Polysemy Generation"
    )
    summary["phases"]["polysemy_generation"] = {
        "success": phase_a_success,
        "output": str(polysemy_output)
    }

    # Phase B: Embed, dedupe, cluster (only if Phase A succeeded)
    polysemy_processed = output_dir / f"polysemy_{date_str}.processed.jsonl"
    if phase_a_success and polysemy_output.exists():
        phase_b_success = run_phase(
            "embed_and_cluster.py",
            ["--input", str(polysemy_output), "--output", str(polysemy_processed)],
            "Phase B: Embedding & Clustering"
        )
        summary["phases"]["embedding_clustering"] = {
            "success": phase_b_success,
            "output": str(polysemy_processed)
        }
    else:
        summary["phases"]["embedding_clustering"] = {"success": False, "skipped": True}

    # Phase C: Generate Lindblad twins
    lindblad_output = output_dir / f"lindblad_twins_{date_str}.jsonl"
    source_for_lindblad = polysemy_processed if polysemy_processed.exists() else polysemy_output

    if source_for_lindblad.exists():
        phase_c_success = run_phase(
            "generate_lindblad_twins.py",
            [
                "--input", str(source_for_lindblad),
                "--output", str(lindblad_output),
                "--twins-per-text", str(twins_per_text)
            ],
            "Phase C: Lindblad Twin Generation"
        )
        summary["phases"]["lindblad_generation"] = {
            "success": phase_c_success,
            "output": str(lindblad_output)
        }
    else:
        summary["phases"]["lindblad_generation"] = {"success": False, "skipped": True}

    # Phase D: Generate eval pack
    eval_pack_output = output_dir / f"eval_pack_{date_str}.json"
    phase_d_args = ["generate", "--type", "fresh", "--output", str(eval_pack_output)]

    if polysemy_processed.exists():
        phase_d_args.extend(["--polysemy", str(polysemy_processed)])
    elif polysemy_output.exists():
        phase_d_args.extend(["--polysemy", str(polysemy_output)])

    if lindblad_output.exists():
        phase_d_args.extend(["--lindblad", str(lindblad_output)])

    phase_d_success = run_phase(
        "generate_eval_pack.py",
        phase_d_args,
        "Phase D: Eval Pack Generation"
    )
    summary["phases"]["eval_pack_generation"] = {
        "success": phase_d_success,
        "output": str(eval_pack_output)
    }

    # Optional: Run evals
    if not skip_eval and eval_pack_output.exists():
        eval_results_output = output_dir / f"eval_results_{date_str}.json"
        phase_run_success = run_phase(
            "generate_eval_pack.py",
            ["run", "--pack", str(eval_pack_output), "--output", str(eval_results_output)],
            "Phase E: Running Evals"
        )
        summary["phases"]["eval_run"] = {
            "success": phase_run_success,
            "output": str(eval_results_output)
        }

        # Load metrics if successful
        if phase_run_success and eval_results_output.exists():
            with open(eval_results_output, 'r') as f:
                eval_data = json.load(f)
            summary["metrics"] = eval_data.get("metrics", {})

    # Record artifacts
    summary["artifacts"] = {
        "polysemy_raw": str(polysemy_output) if polysemy_output.exists() else None,
        "polysemy_processed": str(polysemy_processed) if polysemy_processed.exists() else None,
        "lindblad_twins": str(lindblad_output) if lindblad_output.exists() else None,
        "eval_pack": str(eval_pack_output) if eval_pack_output.exists() else None,
    }

    summary["end_time"] = datetime.now().isoformat()

    # Save summary
    summary_path = output_dir / f"nightly_summary_{date_str}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"Summary saved to {summary_path}")

    # Store to Memory Box for agent reference
    store_to_memory_box(
        session_id="paradigm_factory",
        data={
            "type": "nightly_run",
            "date": date_str,
            "summary": summary
        }
    )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Paradigm Factory Nightly Orchestrator"
    )
    parser.add_argument('--date', type=str,
                        default=datetime.now().strftime("%Y%m%d"),
                        help='Date string for output naming')
    parser.add_argument('--words', type=str, nargs='+', default=None,
                        help='Specific words to generate')
    parser.add_argument('--full', action='store_true',
                        help='Use all default words')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip running evals')
    parser.add_argument('--contexts-per-sense', type=int, default=3,
                        help='Contexts per sense for polysemy')
    parser.add_argument('--twins-per-text', type=int, default=3,
                        help='Twins per text for Lindblad')

    args = parser.parse_args()

    print("=" * 60)
    print("  PARADIGM FACTORY - NIGHTLY ORCHESTRATOR")
    print("=" * 60)
    print(f"Date: {args.date}")
    print(f"Words: {args.words if args.words else 'all defaults' if args.full else 'subset'}")

    # Select words
    words = args.words
    if args.full:
        words = None  # Use all defaults in generate_polysemy.py
    elif words is None:
        # Default subset for quick runs
        words = ["bank", "spring", "bat", "crane"]

    summary = run_nightly_pipeline(
        date_str=args.date,
        words=words,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        skip_eval=args.skip_eval,
        contexts_per_sense=args.contexts_per_sense,
        twins_per_text=args.twins_per_text
    )

    # Final report
    print("\n" + "=" * 60)
    print("  NIGHTLY RUN COMPLETE")
    print("=" * 60)

    all_success = all(p.get("success", False) for p in summary["phases"].values())
    status = "SUCCESS" if all_success else "PARTIAL"

    print(f"Status: {status}")
    print(f"Duration: {summary['start_time']} -> {summary['end_time']}")
    print("\nPhases:")
    for phase, info in summary["phases"].items():
        status_icon = "OK" if info.get("success") else ("SKIP" if info.get("skipped") else "FAIL")
        print(f"  [{status_icon}] {phase}")

    if summary.get("metrics"):
        print("\nMetrics:")
        for k, v in summary["metrics"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
