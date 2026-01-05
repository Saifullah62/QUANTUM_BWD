#!/usr/bin/env python3
"""
Crucible Documentation Linter
==============================

Validates consistency across docs/crucible/*.md files:
1. Internal markdown links resolve within docs/crucible/
2. Code fences are properly closed
3. Core formulas appear exactly once in METRICS.md and are referenced elsewhere
4. Deliverable count matches actual file count

Usage:
    python tools/docs_lint_crucible.py

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

DOCS_DIR = Path(__file__).parent.parent / "docs" / "crucible"

# Files excluded from doc count (auto-generated, internal tooling)
EXCLUDED_FROM_COUNT = {"CURRENT_IMPLEMENTATION.md"}

# Core formulas that must appear exactly once in METRICS.md
CORE_FORMULAS = {
    "danger_score": r"danger\s*=\s*max\(sim\(anchor,\s*neg_i\)\)\s*-\s*sim\(anchor,\s*positive\)",
    "margin": r"margin\s*=\s*-danger",
    "pass_rate": r"pass_rate\s*=\s*count\(margin\s*>\s*0\)\s*/\s*total_items",
    "exposure": r"E\[exposures[^\]]*\]\s*=\s*\(S\s*×\s*B\s*×\s*p_tier3\s*×\s*r_legacy\)\s*/\s*\|L\|",
}


def check_internal_links(files: Dict[str, str]) -> List[str]:
    """Check that all internal markdown links resolve."""
    errors = []
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    valid_targets = set(files.keys())
    # Add anchor variants
    for fname in files.keys():
        valid_targets.add(fname.lower())

    for fname, content in files.items():
        for match in link_pattern.finditer(content):
            link_text, target = match.groups()

            # Skip external links
            if target.startswith(('http://', 'https://', 'mailto:')):
                continue

            # Extract file part (before #anchor)
            file_part = target.split('#')[0] if '#' in target else target

            # Skip empty file part (same-file anchor)
            if not file_part:
                continue

            # Check if target exists
            if file_part not in valid_targets and file_part.upper() not in [f.upper() for f in valid_targets]:
                errors.append(f"{fname}: broken link to '{target}'")

    return errors


def check_code_fences(files: Dict[str, str]) -> List[str]:
    """Check that all code fences are properly closed."""
    errors = []

    for fname, content in files.items():
        lines = content.split('\n')
        in_fence = False
        fence_start = 0

        for i, line in enumerate(lines, 1):
            if line.strip().startswith('```'):
                if not in_fence:
                    in_fence = True
                    fence_start = i
                else:
                    in_fence = False

        if in_fence:
            errors.append(f"{fname}:{fence_start}: unclosed code fence")

    return errors


def check_formula_single_source(files: Dict[str, str]) -> List[str]:
    """Check that core formulas appear exactly once in METRICS.md."""
    errors = []

    metrics_content = files.get("METRICS.md", "")

    for formula_name, pattern in CORE_FORMULAS.items():
        # Check METRICS.md has exactly one occurrence
        matches_in_metrics = len(re.findall(pattern, metrics_content, re.IGNORECASE))

        if matches_in_metrics == 0:
            errors.append(f"METRICS.md: missing formula '{formula_name}'")
        elif matches_in_metrics > 1:
            errors.append(f"METRICS.md: formula '{formula_name}' appears {matches_in_metrics} times (expected 1)")

        # Check other files reference rather than duplicate
        for fname, content in files.items():
            if fname == "METRICS.md":
                continue

            matches = re.findall(pattern, content, re.IGNORECASE)
            if len(matches) > 0:
                # Check if it's in a reference context (mentions METRICS.md nearby)
                # Allow if the file contains a link to METRICS.md
                has_reference = "METRICS.md" in content
                if not has_reference:
                    errors.append(f"{fname}: contains formula '{formula_name}' without referencing METRICS.md")

    return errors


def check_file_count(files: Dict[str, str]) -> List[str]:
    """Check that README.md documentation count matches actual files."""
    errors = []

    readme_content = files.get("README.md", "")

    # Count documentation links in README (links to .md files)
    doc_links = re.findall(r'\[([^\]]+)\]\(([A-Z_]+\.md)\)', readme_content)
    linked_count = len(doc_links)

    # Count actual .md files (excluding README itself and auto-generated files)
    actual_count = len([
        f for f in files.keys()
        if f != "README.md" and f not in EXCLUDED_FROM_COUNT
    ])

    if linked_count != actual_count:
        errors.append(
            f"README.md: lists {linked_count} doc links but there are {actual_count} other .md files "
            f"(excluding {EXCLUDED_FROM_COUNT})"
        )

    return errors


def main():
    """Run all linting checks."""
    if not DOCS_DIR.exists():
        print(f"ERROR: Documentation directory not found: {DOCS_DIR}")
        return 1

    # Load all markdown files
    files = {}
    for md_file in DOCS_DIR.glob("*.md"):
        files[md_file.name] = md_file.read_text(encoding="utf-8")

    if not files:
        print(f"ERROR: No .md files found in {DOCS_DIR}")
        return 1

    print(f"Linting {len(files)} documentation files in {DOCS_DIR}")
    print("=" * 60)

    all_errors = []

    # Run checks
    checks = [
        ("Internal Links", check_internal_links),
        ("Code Fences", check_code_fences),
        ("Formula Single-Source", check_formula_single_source),
        ("File Count", check_file_count),
    ]

    for check_name, check_fn in checks:
        errors = check_fn(files)
        if errors:
            print(f"\n[FAIL] {check_name}:")
            for err in errors:
                print(f"  - {err}")
            all_errors.extend(errors)
        else:
            print(f"[PASS] {check_name}")

    print("=" * 60)

    if all_errors:
        print(f"\nTotal errors: {len(all_errors)}")
        return 1
    else:
        print("\nAll checks passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
