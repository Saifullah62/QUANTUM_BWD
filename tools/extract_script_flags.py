#!/usr/bin/env python3
"""
Script Flag Extractor
=====================

Extracts --help output from actual training/eval scripts and generates
CURRENT_IMPLEMENTATION.md with authoritative flag documentation.

This eliminates drift between docs and reality: the scripts are the
single source of truth for what flags exist today.

Usage:
    python tools/extract_script_flags.py

Output:
    docs/crucible/CURRENT_IMPLEMENTATION.md
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs" / "crucible"

# Scripts to document (path relative to project root, description)
SCRIPTS = [
    ("scripts/train_curriculum_v3.py", "Training (v3, three-pool ratios)"),
    ("scripts/train_curriculum_v2.py", "Training (v2, single tier3-mix)"),
    ("experiments/eval_capsules.py", "Evaluation"),
]


def get_help_output(script_path: Path) -> str:
    """Run script with --help and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_ROOT,
        )
        return result.stdout or result.stderr
    except subprocess.TimeoutExpired:
        return "[ERROR: Script timed out]"
    except Exception as e:
        return f"[ERROR: {e}]"


def generate_implementation_doc() -> str:
    """Generate CURRENT_IMPLEMENTATION.md content."""
    lines = [
        "# Current Implementation Reference",
        "",
        "> **Auto-generated** by `tools/extract_script_flags.py`",
        f"> Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This document shows the actual flags available in the current scripts.",
        "For the target `crucible` CLI interface, see [CLI.md](CLI.md).",
        "",
    ]

    for rel_path, description in SCRIPTS:
        script_path = PROJECT_ROOT / rel_path

        lines.append(f"## {description}")
        lines.append("")
        lines.append(f"**Script**: `{rel_path}`")
        lines.append("")

        if not script_path.exists():
            lines.append(f"> Script not found: {rel_path}")
            lines.append("")
            continue

        help_output = get_help_output(script_path)

        lines.append("```")
        lines.append(help_output.strip())
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Add comparison note
    lines.extend([
        "## CLI vs Scripts",
        "",
        "| Feature | Target CLI (`crucible`) | Current Scripts |",
        "|---------|------------------------|-----------------|",
        "| Three-pool ratios | `--tier3-legacy/organic/expanded` | v3 only |",
        "| Single tier3 mix | - | v2 `--tier3-mix` |",
        "| Eval flags | `--checkpoint`, `--eval-set` | `--eval`, `--results-root` |",
        "",
        "The `crucible` CLI documented in [CLI.md](CLI.md) is the target interface.",
        "Use the scripts above for current implementation.",
    ])

    return "\n".join(lines)


def main():
    """Generate and write the implementation doc."""
    print(f"Extracting --help from {len(SCRIPTS)} scripts...")

    content = generate_implementation_doc()
    output_path = DOCS_DIR / "CURRENT_IMPLEMENTATION.md"

    output_path.write_text(content, encoding="utf-8")
    print(f"Wrote: {output_path}")

    # Also update the linter to know about this file
    print("\nNote: Add CURRENT_IMPLEMENTATION.md to INDEX.md if you want it visible.")
    print("      It's auto-generated, so you may prefer to keep it as internal reference.")


if __name__ == "__main__":
    main()
