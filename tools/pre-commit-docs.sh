#!/bin/bash
# Pre-commit hook for Crucible docs
# Install: cp tools/pre-commit-docs.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

set -e

echo "=== Crucible Docs Pre-commit Check ==="

# Check if any crucible docs or scripts are staged
STAGED=$(git diff --cached --name-only --diff-filter=ACM | grep -E '(docs/crucible/|scripts/.*\.py|experiments/.*\.py)' || true)

if [ -z "$STAGED" ]; then
    echo "No crucible docs or scripts staged, skipping checks"
    exit 0
fi

echo "Regenerating CURRENT_IMPLEMENTATION.md..."
python tools/extract_script_flags.py

# Check if regeneration changed anything
if ! git diff --quiet docs/crucible/CURRENT_IMPLEMENTATION.md; then
    echo ""
    echo "WARNING: CURRENT_IMPLEMENTATION.md was updated by extract_script_flags.py"
    echo "Please stage the updated file:"
    echo "  git add docs/crucible/CURRENT_IMPLEMENTATION.md"
    echo ""
    exit 1
fi

echo "Running docs linter..."
python tools/docs_lint_crucible.py

echo "=== Docs check passed ==="
