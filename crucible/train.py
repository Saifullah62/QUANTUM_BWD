"""
Crucible Training Module
========================

Thin wrapper around scripts/train_curriculum_v3.py.

Usage:
    python -m crucible.train --bundles data/bundles.jsonl --steps 10000

For available flags, run:
    python -m crucible.train --help
"""

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRAINER_SCRIPT = PROJECT_ROOT / "scripts" / "train_curriculum_v3.py"


def main():
    """Entry point that delegates to the canonical training script."""
    # Use runpy to execute the script as __main__
    sys.argv[0] = str(TRAINER_SCRIPT)
    runpy.run_path(str(TRAINER_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
