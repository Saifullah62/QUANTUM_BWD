#!/usr/bin/env python3
"""
Attention Audit Module
======================

Tracks SenseHead attention patterns to verify the model is using cues correctly.

Metrics tracked per epoch:
- Top attended tokens
- Attention entropy (higher = more spread)
- Cue mass: attention on cue_tokens
- Lemma mass: attention on target lemma span
- Cue hit score: cue_mass / (cue_mass + lemma_mass)

Produces:
- artifacts/attention_audit_epoch_{k}.jsonl
- Dashboard summary: attn|entropy=...|cue_mass=...|lemma_mass=...
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class AttentionAuditItem:
    """A single attention audit result."""
    item_id: str
    lemma: str
    sense_id: str
    cue_tier: str

    # Attention metrics
    entropy: float
    cue_mass: float
    lemma_mass: float
    context_mass: float
    cue_hit_score: float

    # Top attended tokens
    top_5_tokens: List[Dict]  # [{token, position, weight}]

    # Span info
    cue_tokens: List[str]
    lemma_span: Tuple[int, int]


@dataclass
class AttentionAuditSummary:
    """Epoch-level attention audit summary."""
    epoch: int
    timestamp: str
    n_items: int

    # Aggregate metrics
    entropy_mean: float
    entropy_std: float
    cue_mass_mean: float
    cue_mass_std: float
    lemma_mass_mean: float
    context_mass_mean: float
    cue_hit_score_mean: float

    # By cue tier
    by_tier: Dict[str, Dict]

    # Fingerprint for dashboard
    fingerprint: str


class AttentionAuditor:
    """
    Audits SenseHead attention patterns on a frozen probe set.

    Usage:
        auditor = AttentionAuditor(audit_set_path, output_dir)

        # During training loop:
        for epoch in range(epochs):
            for batch in loader:
                ...
            auditor.audit_epoch(model, tokenizer, epoch)

        # Get dashboard summary:
        summary = auditor.get_summary()
    """

    def __init__(
        self,
        audit_set_path: Path,
        output_dir: Path,
        device: str = "cuda"
    ):
        """
        Args:
            audit_set_path: Path to frozen audit set JSONL
            output_dir: Directory for attention_audit_epoch_{k}.jsonl files
            device: torch device
        """
        self.audit_set_path = Path(audit_set_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Load frozen audit set
        self.audit_items = self._load_audit_set()
        self.summaries: List[AttentionAuditSummary] = []

    def _load_audit_set(self) -> List[Dict]:
        """Load the frozen attention audit set."""
        items = []
        if self.audit_set_path.exists():
            with open(self.audit_set_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        return items

    def _find_token_positions(
        self,
        tokens: List[str],
        target_tokens: List[str],
        case_insensitive: bool = True
    ) -> List[int]:
        """Find positions of target tokens in token list."""
        positions = []
        target_set = set(t.lower() for t in target_tokens) if case_insensitive else set(target_tokens)

        for i, tok in enumerate(tokens):
            check_tok = tok.lower() if case_insensitive else tok
            # Remove subword markers for comparison
            clean_tok = check_tok.replace('##', '').replace('Ġ', '').strip()
            if clean_tok in target_set or check_tok in target_set:
                positions.append(i)

        return positions

    def _find_lemma_span(
        self,
        tokens: List[str],
        lemma: str,
        case_insensitive: bool = True
    ) -> Tuple[int, int]:
        """Find the span of the target lemma in token list."""
        lemma_lower = lemma.lower() if case_insensitive else lemma

        for i, tok in enumerate(tokens):
            check_tok = tok.lower() if case_insensitive else tok
            clean_tok = check_tok.replace('##', '').replace('Ġ', '').strip()

            if lemma_lower in clean_tok or clean_tok == lemma_lower:
                # Found start, look for end of word
                end = i + 1
                while end < len(tokens) and tokens[end].startswith('##'):
                    end += 1
                return (i, end)

        return (-1, -1)

    def _compute_mass(
        self,
        weights: np.ndarray,
        positions: List[int]
    ) -> float:
        """Compute total attention mass at given positions."""
        if not positions:
            return 0.0
        return float(sum(weights[p] for p in positions if p < len(weights)))

    def audit_item(
        self,
        item: Dict,
        attention_weights: np.ndarray,
        tokens: List[str]
    ) -> AttentionAuditItem:
        """Audit a single item's attention pattern."""
        # Get item metadata
        item_id = item.get('eval_id', item.get('id', 'unknown'))
        lemma = item.get('lemma', '')
        sense_id = item.get('target_sense_id', item.get('sense_id', ''))
        cue_tier = item.get('cue_tier', 'unknown')
        cue_tokens = item.get('cue_tokens', [])

        # Compute entropy
        eps = 1e-8
        entropy = float(-np.sum(attention_weights * np.log(attention_weights + eps)))

        # Find positions
        cue_positions = self._find_token_positions(tokens, cue_tokens)
        lemma_span = self._find_lemma_span(tokens, lemma)
        lemma_positions = list(range(lemma_span[0], lemma_span[1])) if lemma_span[0] >= 0 else []

        # Compute mass on different token groups
        cue_mass = self._compute_mass(attention_weights, cue_positions)
        lemma_mass = self._compute_mass(attention_weights, lemma_positions)

        # Context mass = everything except lemma
        all_positions = set(range(len(attention_weights)))
        context_positions = all_positions - set(lemma_positions)
        context_mass = self._compute_mass(attention_weights, list(context_positions))

        # Cue hit score
        total_relevant = cue_mass + lemma_mass
        cue_hit_score = cue_mass / total_relevant if total_relevant > 0 else 0.0

        # Top 5 attended tokens
        top_indices = np.argsort(attention_weights)[-5:][::-1]
        top_5_tokens = [
            {
                "token": tokens[idx] if idx < len(tokens) else "<pad>",
                "position": int(idx),
                "weight": float(attention_weights[idx])
            }
            for idx in top_indices
        ]

        return AttentionAuditItem(
            item_id=item_id,
            lemma=lemma,
            sense_id=sense_id,
            cue_tier=cue_tier,
            entropy=entropy,
            cue_mass=cue_mass,
            lemma_mass=lemma_mass,
            context_mass=context_mass,
            cue_hit_score=cue_hit_score,
            top_5_tokens=top_5_tokens,
            cue_tokens=cue_tokens,
            lemma_span=lemma_span
        )

    def audit_epoch(
        self,
        model,
        tokenizer,
        epoch: int,
        max_items: int = 50
    ) -> AttentionAuditSummary:
        """
        Run attention audit on the frozen probe set for this epoch.

        Args:
            model: Model with SenseHead that returns attention weights
            tokenizer: Tokenizer for decoding
            epoch: Current epoch number
            max_items: Maximum items to audit

        Returns:
            AttentionAuditSummary for this epoch
        """
        model.eval()
        results: List[AttentionAuditItem] = []

        items_to_audit = self.audit_items[:max_items]

        with torch.no_grad():
            for item in items_to_audit:
                # Tokenize query
                query = item.get('query', item.get('context', ''))
                if not query:
                    continue

                inputs = tokenizer(
                    query,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get attention weights from model
                # Assumes model has a method to get sense head attention
                try:
                    outputs = model(**inputs, return_attention=True)
                    attention_weights = outputs.get('attention_weights')
                    if attention_weights is None:
                        continue

                    # Convert to numpy
                    weights = attention_weights[0].cpu().numpy()  # First item in batch

                    # Get tokens
                    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

                    # Audit this item
                    audit_result = self.audit_item(item, weights, tokens)
                    results.append(audit_result)

                except Exception as e:
                    # Skip items that fail
                    continue

        # Compute aggregate metrics
        if results:
            entropies = [r.entropy for r in results]
            cue_masses = [r.cue_mass for r in results]
            lemma_masses = [r.lemma_mass for r in results]
            context_masses = [r.context_mass for r in results]
            cue_hits = [r.cue_hit_score for r in results]

            # By tier
            by_tier = {}
            for tier in set(r.cue_tier for r in results):
                tier_results = [r for r in results if r.cue_tier == tier]
                by_tier[tier] = {
                    "count": len(tier_results),
                    "entropy_mean": float(np.mean([r.entropy for r in tier_results])),
                    "cue_mass_mean": float(np.mean([r.cue_mass for r in tier_results])),
                    "cue_hit_mean": float(np.mean([r.cue_hit_score for r in tier_results]))
                }

            summary = AttentionAuditSummary(
                epoch=epoch,
                timestamp=datetime.now().isoformat(),
                n_items=len(results),
                entropy_mean=float(np.mean(entropies)),
                entropy_std=float(np.std(entropies)),
                cue_mass_mean=float(np.mean(cue_masses)),
                cue_mass_std=float(np.std(cue_masses)),
                lemma_mass_mean=float(np.mean(lemma_masses)),
                context_mass_mean=float(np.mean(context_masses)),
                cue_hit_score_mean=float(np.mean(cue_hits)),
                by_tier=by_tier,
                fingerprint=f"attn|entropy={np.mean(entropies):.2f}|cue_mass={np.mean(cue_masses):.2f}|lemma_mass={np.mean(lemma_masses):.2f}"
            )
        else:
            summary = AttentionAuditSummary(
                epoch=epoch,
                timestamp=datetime.now().isoformat(),
                n_items=0,
                entropy_mean=0.0,
                entropy_std=0.0,
                cue_mass_mean=0.0,
                cue_mass_std=0.0,
                lemma_mass_mean=0.0,
                context_mass_mean=0.0,
                cue_hit_score_mean=0.0,
                by_tier={},
                fingerprint="attn|no_data"
            )

        # Save results
        output_file = self.output_dir / f"attention_audit_epoch_{epoch}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')

        # Save summary
        summary_file = self.output_dir / f"attention_audit_epoch_{epoch}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, indent=2)

        self.summaries.append(summary)
        return summary

    def get_latest_summary(self) -> Optional[AttentionAuditSummary]:
        """Get the most recent epoch summary."""
        return self.summaries[-1] if self.summaries else None

    def get_trend(self) -> Dict:
        """Get trend data across all audited epochs."""
        if not self.summaries:
            return {}

        return {
            "epochs": [s.epoch for s in self.summaries],
            "entropy": [s.entropy_mean for s in self.summaries],
            "cue_mass": [s.cue_mass_mean for s in self.summaries],
            "cue_hit_score": [s.cue_hit_score_mean for s in self.summaries],
        }


def create_frozen_audit_set(
    evals_path: Path,
    output_path: Path,
    n_items: int = 50,
    seed: int = 42
) -> int:
    """
    Create a frozen attention audit set from gated evals.

    Selects items with:
    - Mix of cue tiers (high, medium, low)
    - Mix of domains/lemmas
    - Items that have cue_tokens defined

    Args:
        evals_path: Path to gated retrieval evals (retrieval_valid.jsonl)
        output_path: Output path for frozen audit set
        n_items: Number of items to include
        seed: Random seed for reproducibility

    Returns:
        Number of items in the audit set
    """
    import random
    random.seed(seed)

    # Load evals
    items = []
    with open(evals_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Only include items with cue_tokens
                if item.get('cue_tokens'):
                    items.append(item)

    if not items:
        print(f"Warning: No items with cue_tokens found in {evals_path}")
        return 0

    # Group by cue tier
    by_tier = {}
    for item in items:
        tier = item.get('cue_tier', 'unknown')
        by_tier.setdefault(tier, []).append(item)

    # Sample proportionally from each tier
    selected = []
    tier_counts = {tier: len(items) for tier, items in by_tier.items()}
    total = sum(tier_counts.values())

    for tier, tier_items in by_tier.items():
        # Proportional allocation
        tier_quota = max(1, int(n_items * len(tier_items) / total))
        tier_sample = random.sample(tier_items, min(tier_quota, len(tier_items)))
        selected.extend(tier_sample)

    # Trim to exact size
    if len(selected) > n_items:
        selected = random.sample(selected, n_items)

    # Shuffle
    random.shuffle(selected)

    # Write audit set
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Created frozen audit set: {output_path}")
    print(f"  Total items: {len(selected)}")
    print(f"  By tier: {dict((t, len([i for i in selected if i.get('cue_tier') == t])) for t in by_tier.keys())}")

    return len(selected)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create frozen attention audit set")
    parser.add_argument("--evals", type=Path, required=True,
                       help="Path to retrieval_valid.jsonl")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output path for audit set")
    parser.add_argument("--n-items", type=int, default=50,
                       help="Number of items (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    args = parser.parse_args()
    create_frozen_audit_set(args.evals, args.output, args.n_items, args.seed)
