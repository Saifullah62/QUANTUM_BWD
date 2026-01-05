"""
Crucible Capsule Module
=======================

Thin wrapper around experiments/build_production_capsule.py.

Usage:
    python -m crucible.capsule --bundles ... --frozen-eval ... --organic-eval ...

For available flags, run:
    python -m crucible.capsule --help
"""

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CAPSULE_SCRIPT = PROJECT_ROOT / "experiments" / "build_production_capsule.py"


def main():
    """Entry point that delegates to the canonical capsule builder."""
    sys.argv[0] = str(CAPSULE_SCRIPT)
    runpy.run_path(str(CAPSULE_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
