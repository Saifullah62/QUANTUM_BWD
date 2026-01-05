#!/usr/bin/env python3
"""
Curriculum Training Script with Phased Tier Weighting
======================================================

Implements phased curriculum for v2.3 bundles:
- Early phase (0-35%): gentle tier3 presence
- Mid phase (35-70%): increased tier3
- Late phase (70-100%): tier3 held steady at cap

Key features:
- Per-batch tier quotas with hard cap on tier3 (max 20-25%)
- Tier3 positive rotation (reuse anchors with different positives)
- Margin distribution tracking

Usage:
    python scripts/train_curriculum_v2.py \
        --bundles path/to/bundles.jsonl \
        --steps 5000 \
        --curriculum phased  # or 'uniform' for baseline
"""

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qllm.layers.semantic_phase import SemanticPhaseEmbedding


@dataclass
class CurriculumConfig:
    """Configuration for phased curriculum."""
    # Phase boundaries (as fraction of total steps)
    early_end: float = 0.35
    mid_end: float = 0.70

    # Per-batch quotas for batch_size=32
    # Format: (tier1, tier2, tier3)
    early_quota: Tuple[int, int, int] = (24, 7, 1)
    mid_quota: Tuple[int, int, int] = (20, 9, 3)
    late_quota: Tuple[int, int, int] = (18, 9, 5)

    # Hard cap on tier3 fraction
    tier3_max_fraction: float = 0.20

    # Tier3 positive rotation
    tier3_positive_rotation: bool = True

    # Tier3 replay: replay negative-margin tier3 items within N batches
    tier3_replay_enabled: bool = False
    tier3_replay_window: int = 20  # batches


# C2b: "Phase with teeth" - more aggressive tier3 in mid/late phases
C2B_CONFIG = CurriculumConfig(
    early_end=0.35,
    mid_end=0.70,
    early_quota=(22, 8, 2),   # ~6% tier3
    mid_quota=(18, 9, 5),     # ~16% tier3
    late_quota=(14, 10, 8),   # ~25% tier3
    tier3_max_fraction=0.25,  # Raised from 0.20 to let late phase express
    tier3_positive_rotation=True,
    tier3_replay_enabled=True,
    tier3_replay_window=20
)


class TieredBundleDataset:
    """Dataset that supports tier-aware sampling with positive rotation."""

    def __init__(self, bundle_path: str, max_samples: int = None):
        self.bundles_by_tier: Dict[str, List[dict]] = defaultdict(list)
        self.tier3_positive_index: Dict[str, int] = {}  # For rotation

        # Tier3 split: core (original) vs expanded
        self.tier3_core: List[dict] = []
        self.tier3_expanded: List[dict] = []

        with open(bundle_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    bundle = json.loads(line)

                    # Extract tier
                    metadata = bundle.get('metadata', {})
                    tier = metadata.get('difficulty_tier', 'tier1_easy')

                    # Validate bundle has required fields
                    anchor = bundle.get('anchor')
                    positive = bundle.get('positive')
                    if not anchor or not positive:
                        continue
                    if not anchor.get('text') or not positive.get('text'):
                        continue

                    self.bundles_by_tier[tier].append(bundle)

                    # Split tier3 into core vs expanded
                    if tier == 'tier3_adversarial':
                        is_expanded = (
                            metadata.get('expansion') == 'tier3x' or
                            metadata.get('source') == 'tier3_expansion'
                        )
                        if is_expanded:
                            self.tier3_expanded.append(bundle)
                        else:
                            self.tier3_core.append(bundle)

                    if max_samples:
                        total = sum(len(v) for v in self.bundles_by_tier.values())
                        if total >= max_samples:
                            break

        self.tiers = list(self.bundles_by_tier.keys())
        total = sum(len(v) for v in self.bundles_by_tier.values())
        print(f"Loaded {total} bundles from {bundle_path}")
        for tier, bundles in sorted(self.bundles_by_tier.items()):
            print(f"  {tier}: {len(bundles)}")

        # Report tier3 split
        if self.tier3_core or self.tier3_expanded:
            print(f"  tier3 split: {len(self.tier3_core)} core + {len(self.tier3_expanded)} expanded")

    def get_tier_counts(self) -> Dict[str, int]:
        return {t: len(b) for t, b in self.bundles_by_tier.items()}

    def sample_bundle(self, tier: str, rotate_positives: bool = False,
                      tier3_core_ratio: float = None) -> dict:
        """
        Sample a bundle from the specified tier.

        For tier3_adversarial with tier3_core_ratio set, samples from
        core vs expanded pools according to the ratio.
        """
        # Special handling for tier3 with mix ratio
        if tier == 'tier3_adversarial' and tier3_core_ratio is not None:
            if self.tier3_core and self.tier3_expanded:
                if random.random() < tier3_core_ratio:
                    return random.choice(self.tier3_core)
                else:
                    return random.choice(self.tier3_expanded)
            elif self.tier3_core:
                return random.choice(self.tier3_core)
            elif self.tier3_expanded:
                return random.choice(self.tier3_expanded)

        bundles = self.bundles_by_tier.get(tier, [])
        if not bundles:
            # Fallback to any tier
            all_bundles = [b for bs in self.bundles_by_tier.values() for b in bs]
            return random.choice(all_bundles)

        bundle = random.choice(bundles)
        return bundle

    def extract_triplet(self, bundle: dict) -> Tuple[str, str, str]:
        """Extract (anchor, positive, negative) texts from bundle."""
        anchor_text = bundle['anchor']['text']
        positive_text = bundle['positive']['text']

        # Get negatives
        negatives = bundle.get('negatives', {})
        if isinstance(negatives, dict):
            all_negs = negatives.get('within_lemma', []) + negatives.get('cross_lemma', [])
        else:
            all_negs = negatives if isinstance(negatives, list) else []

        if all_negs:
            neg = random.choice(all_negs)
            negative_text = neg['text']
        else:
            # Fallback: use different bundle's positive
            other_tier = random.choice(self.tiers)
            other_bundle = random.choice(self.bundles_by_tier[other_tier])
            negative_text = other_bundle['positive']['text']

        return anchor_text, positive_text, negative_text


class CurriculumSampler:
    """Samples batches according to phased curriculum."""

    def __init__(self, dataset: TieredBundleDataset, config: CurriculumConfig,
                 batch_size: int = 32, total_steps: int = 5000,
                 tier3_core_ratio: float = None):
        self.dataset = dataset
        self.config = config
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.current_step = 0
        self.tier3_core_ratio = tier3_core_ratio  # None = no mixing, 0.7 = 70% core

        # Tier3 replay queue: stores (bundle, expire_step) tuples
        self.tier3_replay_queue: List[Tuple[dict, int]] = []

        if tier3_core_ratio is not None:
            print(f"Tier3 mix enabled: {tier3_core_ratio:.0%} core, {1-tier3_core_ratio:.0%} expanded")

    def get_phase(self) -> str:
        """Determine current training phase."""
        progress = self.current_step / self.total_steps
        if progress < self.config.early_end:
            return 'early'
        elif progress < self.config.mid_end:
            return 'mid'
        else:
            return 'late'

    def get_quota(self) -> Tuple[int, int, int]:
        """Get tier quotas for current phase."""
        phase = self.get_phase()
        if phase == 'early':
            quota = self.config.early_quota
        elif phase == 'mid':
            quota = self.config.mid_quota
        else:
            quota = self.config.late_quota

        # Scale quota to actual batch size
        total_quota = sum(quota)
        scale = self.batch_size / total_quota
        scaled = [int(q * scale) for q in quota]

        # Ensure we hit exact batch size (adjust tier1)
        diff = self.batch_size - sum(scaled)
        scaled[0] += diff

        # Apply tier3 cap
        max_tier3 = int(self.batch_size * self.config.tier3_max_fraction)
        if scaled[2] > max_tier3:
            excess = scaled[2] - max_tier3
            scaled[2] = max_tier3
            scaled[0] += excess  # Give excess to tier1

        return tuple(scaled)

    def sample_batch(self) -> List[Tuple[str, str, str, str, Optional[dict]]]:
        """
        Sample a batch according to current curriculum phase.

        Returns list of (anchor, positive, negative, tier, bundle) tuples.
        The bundle is included for tier3 items to support replay on failure.
        """
        quota = self.get_quota()
        tier_names = ['tier1_easy', 'tier2_robust', 'tier3_adversarial']

        batch = []

        # First, inject any replay items (reduce tier3 fresh quota accordingly)
        replay_count = 0
        replay_limit = min(2, quota[2])  # Max 2 replays per batch
        while replay_count < replay_limit and self.tier3_replay_queue:
            replay_bundle = self.pop_replay()
            if replay_bundle:
                anchor, positive, negative = self.dataset.extract_triplet(replay_bundle)
                batch.append((anchor, positive, negative, 'tier3_adversarial', replay_bundle))
                replay_count += 1

        # Adjust tier3 quota for replays already added
        adjusted_quota = list(quota)
        adjusted_quota[2] = max(0, adjusted_quota[2] - replay_count)

        for tier_idx, count in enumerate(adjusted_quota):
            tier = tier_names[tier_idx]

            # Check if tier exists in dataset
            if tier not in self.dataset.bundles_by_tier or not self.dataset.bundles_by_tier[tier]:
                # Fallback to tier1
                tier = 'tier1_easy'

            for _ in range(count):
                # Pass tier3_core_ratio for tier3 sampling
                tier3_ratio = self.tier3_core_ratio if tier == 'tier3_adversarial' else None
                bundle = self.dataset.sample_bundle(
                    tier,
                    rotate_positives=(tier == 'tier3_adversarial' and
                                     self.config.tier3_positive_rotation),
                    tier3_core_ratio=tier3_ratio
                )
                anchor, positive, negative = self.dataset.extract_triplet(bundle)
                # Include bundle for tier3 items (for potential replay)
                include_bundle = bundle if tier == 'tier3_adversarial' else None
                batch.append((anchor, positive, negative, tier, include_bundle))

        random.shuffle(batch)
        return batch

    def step(self):
        """Advance curriculum step."""
        self.current_step += 1
        # Prune expired replay items
        self.tier3_replay_queue = [
            (b, exp) for b, exp in self.tier3_replay_queue
            if exp > self.current_step
        ]

    def add_to_replay(self, bundle: dict):
        """Add a tier3 bundle to replay queue (for negative-margin items)."""
        if not self.config.tier3_replay_enabled:
            return
        expire_step = self.current_step + self.config.tier3_replay_window
        self.tier3_replay_queue.append((bundle, expire_step))

    def pop_replay(self) -> Optional[dict]:
        """Pop one item from replay queue if available."""
        if not self.tier3_replay_queue:
            return None
        bundle, _ = self.tier3_replay_queue.pop(0)
        return bundle


class SemanticPhaseModel(nn.Module):
    """Model combining text encoder with SemanticPhase embedding."""

    def __init__(self, vocab_size: int = 30000, embed_dim: int = 256, phase_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.phase_dim = phase_dim

        self.semantic_phase = SemanticPhaseEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            phase_dim=phase_dim,
            padding_idx=0
        )

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.vocab = {chr(i): i for i in range(256)}

    def tokenize(self, text: str, max_len: int = 128) -> torch.Tensor:
        tokens = [self.vocab.get(c, 1) for c in text[:max_len]]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        return torch.tensor(tokens[:max_len])

    def forward(self, texts: list) -> torch.Tensor:
        tokens = torch.stack([self.tokenize(t) for t in texts])
        device = next(self.parameters()).device
        tokens = tokens.to(device)

        real_emb, imag_emb = self.semantic_phase(tokens)
        phase_emb = torch.sqrt(real_emb**2 + imag_emb**2 + 1e-8)

        mask = (tokens != 0).float().unsqueeze(-1).to(device)
        pooled = (phase_emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        return self.proj(pooled)


@dataclass
class MarginStats:
    """Statistics for margin distribution."""
    mean: float
    median: float
    std: float
    pass_rate: float  # % with margin > 0
    q10: float  # 10th percentile
    q90: float  # 90th percentile


def compute_margin_stats(margins: List[float]) -> MarginStats:
    """Compute margin distribution statistics."""
    if not margins:
        return MarginStats(0, 0, 0, 0, 0, 0)

    import numpy as np
    arr = np.array(margins)
    return MarginStats(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        pass_rate=float((arr > 0).mean()),
        q10=float(np.percentile(arr, 10)),
        q90=float(np.percentile(arr, 90))
    )


def contrastive_loss(anchor: torch.Tensor, positive: torch.Tensor,
                     negative: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def compute_margins(anchor: torch.Tensor, positive: torch.Tensor,
                   negative: torch.Tensor) -> torch.Tensor:
    """Compute margin = neg_sim - pos_sim (positive margin = correct)."""
    anchor_norm = F.normalize(anchor, dim=-1)
    positive_norm = F.normalize(positive, dim=-1)
    negative_norm = F.normalize(negative, dim=-1)

    pos_sim = (anchor_norm * positive_norm).sum(dim=-1)
    neg_sim = (anchor_norm * negative_norm).sum(dim=-1)

    return pos_sim - neg_sim  # Positive = correct


def train(args):
    """Main training loop with curriculum."""
    print(f"Training with curriculum: {args.curriculum}")
    print(f"Bundle path: {args.bundles}")
    print(f"Device: {args.device}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Load dataset
    dataset = TieredBundleDataset(args.bundles, max_samples=args.max_samples)

    # Create curriculum config
    if args.curriculum == 'phased':
        config = CurriculumConfig(
            tier3_positive_rotation=True
        )
    elif args.curriculum == 'phased_v2':
        # C2b: "Phase with teeth" config
        config = C2B_CONFIG
        print("Using C2b config: phased with teeth (22/8/2 → 18/9/5 → 14/10/8)")
    else:  # uniform
        # Equal weights, no phasing
        config = CurriculumConfig(
            early_quota=(11, 11, 10),
            mid_quota=(11, 11, 10),
            late_quota=(11, 11, 10),
            tier3_max_fraction=0.35,
            tier3_positive_rotation=False
        )

    sampler = CurriculumSampler(
        dataset, config,
        batch_size=args.batch_size,
        total_steps=args.steps,
        tier3_core_ratio=args.tier3_mix
    )

    # Create model
    model = SemanticPhaseModel(embed_dim=args.embed_dim, phase_dim=args.phase_dim)
    model = model.to(args.device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Training tracking
    model.train()
    total_loss = 0
    total_correct = 0
    margins_by_tier: Dict[str, List[float]] = defaultdict(list)

    pbar = tqdm(total=args.steps, desc="Training")

    for step in range(args.steps):
        # Sample batch according to curriculum
        batch = sampler.sample_batch()

        anchors = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        negatives = [b[2] for b in batch]
        tiers = [b[3] for b in batch]
        bundles = [b[4] for b in batch]  # For tier3 replay

        # Forward pass
        anchor_emb = model(anchors)
        positive_emb = model(positives)
        negative_emb = model(negatives)

        # Compute loss
        loss = contrastive_loss(anchor_emb, positive_emb, negative_emb, margin=args.margin)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track margins per tier and add failed tier3 to replay
        with torch.no_grad():
            batch_margins = compute_margins(anchor_emb, positive_emb, negative_emb)
            for i, tier in enumerate(tiers):
                margin_val = batch_margins[i].item()
                margins_by_tier[tier].append(margin_val)

                # Tier3 replay: if margin is negative, queue for replay
                if tier == 'tier3_adversarial' and margin_val < 0 and bundles[i] is not None:
                    sampler.add_to_replay(bundles[i])

        # Track accuracy
        correct = (batch_margins > 0).float().mean().item()
        total_loss += loss.item()
        total_correct += correct

        sampler.step()

        if (step + 1) % 50 == 0:
            avg_loss = total_loss / 50
            avg_acc = total_correct / 50
            phase = sampler.get_phase()
            quota = sampler.get_quota()

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.1%}',
                'phase': phase,
                'quota': f'{quota[0]}/{quota[1]}/{quota[2]}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            total_loss = 0
            total_correct = 0

        pbar.update(1)

    pbar.close()

    # Compute final margin stats
    print("\n" + "=" * 60)
    print("MARGIN DISTRIBUTION BY TIER")
    print("=" * 60)

    margin_stats = {}
    for tier in sorted(margins_by_tier.keys()):
        stats = compute_margin_stats(margins_by_tier[tier])
        margin_stats[tier] = asdict(stats)
        print(f"\n{tier}:")
        print(f"  Mean margin:   {stats.mean:.4f}")
        print(f"  Median margin: {stats.median:.4f}")
        print(f"  Std:           {stats.std:.4f}")
        print(f"  Pass rate:     {stats.pass_rate:.1%}")
        print(f"  Q10/Q90:       {stats.q10:.4f} / {stats.q90:.4f}")

    # Final evaluation
    model.eval()
    eval_margins_by_tier: Dict[str, List[float]] = defaultdict(list)
    eval_correct = 0
    eval_count = 0

    print("\nRunning final evaluation...")
    with torch.no_grad():
        for _ in tqdm(range(200), desc="Eval"):
            batch = sampler.sample_batch()

            anchors = [b[0] for b in batch]
            positives = [b[1] for b in batch]
            negatives = [b[2] for b in batch]
            tiers = [b[3] for b in batch]
            # b[4] = bundles, ignored in eval

            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)

            batch_margins = compute_margins(anchor_emb, positive_emb, negative_emb)

            for i, tier in enumerate(tiers):
                eval_margins_by_tier[tier].append(batch_margins[i].item())

            eval_correct += (batch_margins > 0).sum().item()
            eval_count += len(batch)

    final_acc = eval_correct / eval_count

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    print(f"Overall accuracy: {final_acc:.1%}")

    eval_margin_stats = {}
    for tier in sorted(eval_margins_by_tier.keys()):
        stats = compute_margin_stats(eval_margins_by_tier[tier])
        eval_margin_stats[tier] = asdict(stats)
        tier_acc = stats.pass_rate
        print(f"\n{tier}:")
        print(f"  Accuracy:      {tier_acc:.1%}")
        print(f"  Mean margin:   {stats.mean:.4f}")
        print(f"  Median margin: {stats.median:.4f}")
        print(f"  Pass rate:     {stats.pass_rate:.1%}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "model.pt")

    metrics = {
        'final_accuracy': final_acc,
        'curriculum': args.curriculum,
        'steps': args.steps,
        'seed': args.seed,
        'bundle_path': args.bundles,
        'timestamp': datetime.now().isoformat(),
        'training_margin_stats': margin_stats,
        'eval_margin_stats': eval_margin_stats,
        'curriculum_config': asdict(config) if args.curriculum == 'phased' else None
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Curriculum training for v2.3 bundles")
    parser.add_argument('--bundles', required=True, help='Path to v2.3 bundles JSONL')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.5, help='Contrastive margin')
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--phase-dim', type=int, default=64, help='Phase dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='runs/curriculum', help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None, help='Max training samples')
    parser.add_argument('--curriculum', choices=['phased', 'phased_v2', 'uniform'], default='phased',
                       help='Curriculum type: phased (C2), phased_v2 (C2b with teeth), or uniform (C1 baseline)')
    parser.add_argument('--tier3-mix', type=float, default=None, metavar='RATIO',
                       help='Tier3 core ratio (0.0-1.0). E.g., 0.7 = 70%% core, 30%% expanded. None = no mixing.')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
