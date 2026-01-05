"""
Crucible Evaluation Module
==========================

Thin wrapper around experiments/eval_capsules.py.

Usage:
    python -m crucible.eval --eval evals/frozen.jsonl --results-root results/

For available flags, run:
    python -m crucible.eval --help
"""

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_SCRIPT = PROJECT_ROOT / "experiments" / "eval_capsules.py"


def main():
    """Entry point that delegates to the canonical evaluation script."""
    sys.argv[0] = str(EVAL_SCRIPT)
    runpy.run_path(str(EVAL_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
