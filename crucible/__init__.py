"""
Crucible: Contrastive Curriculum and Adversarial Evaluation Harness
====================================================================

A production-grade experimentation harness for teams training embedding models,
retrieval systems, and contrastive learners.

This module provides thin wrappers around the canonical scripts:
- crucible.train: Curriculum training with three-pool tier3 mixing
- crucible.eval: Tier-stratified evaluation with margin distributions
- crucible.capsule: Production capsule building

Usage:
    python -m crucible.train --bundles data/bundles.jsonl --steps 10000
    python -m crucible.eval --eval evals/frozen.jsonl --results-root results/
    python -m crucible.capsule --bundles ... --frozen-eval ... --organic-eval ...

For the full CLI specification (target interface), see docs/crucible/CLI.md.
For current script flags (authoritative), see docs/crucible/CURRENT_IMPLEMENTATION.md.
"""

__version__ = "0.1.0"
__all__ = ["train", "eval", "capsule"]
