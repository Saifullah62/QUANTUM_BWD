#!/usr/bin/env python3
"""
Curriculum Training Script with Three-Pool Tier3 Mixing
========================================================

Extends train_curriculum_v2 with three-pool tier3 mixing:
- tier3_legacy: original killer bundles (highest rehearsal)
- tier3_organic: newly mined adversarial cases
- tier3_expanded: vetted tier3x expansion (lowest priority)

Usage:
    python scripts/train_curriculum_v3.py \
        --bundles path/to/bundles.jsonl \
        --steps 5000 \
        --tier3-legacy 0.80 --tier3-organic 0.10 --tier3-expanded 0.10
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from qllm.layers.semantic_phase import SemanticPhaseEmbedding


@dataclass
class CurriculumConfig:
    early_end: float = 0.35
    mid_end: float = 0.70
    early_quota: Tuple[int, int, int] = (24, 7, 1)
    mid_quota: Tuple[int, int, int] = (20, 9, 3)
    late_quota: Tuple[int, int, int] = (18, 9, 5)
    tier3_max_fraction: float = 0.20
    tier3_positive_rotation: bool = True
    tier3_replay_enabled: bool = False
    tier3_replay_window: int = 20


C2B_CONFIG = CurriculumConfig(
    early_end=0.35,
    mid_end=0.70,
    early_quota=(22, 8, 2),
    mid_quota=(18, 9, 5),
    late_quota=(14, 10, 8),
    tier3_max_fraction=0.25,
    tier3_positive_rotation=True,
    tier3_replay_enabled=True,
    tier3_replay_window=20
)


@dataclass
class Tier3MixRatios:
    """Three-pool tier3 mixing ratios."""
    legacy: float = 0.80
    organic: float = 0.10
    expanded: float = 0.10

    def sample_pool(self) -> str:
        """Return which pool to sample from."""
        r = random.random()
        if r < self.legacy:
            return "legacy"
        elif r < self.legacy + self.organic:
            return "organic"
        else:
            return "expanded"


class TieredBundleDataset:
    """Dataset with three-pool tier3 support."""

    def __init__(self, bundle_path: str, max_samples: int = None):
        self.bundles_by_tier: Dict[str, List[dict]] = defaultdict(list)
        self.tier3_positive_index: Dict[str, int] = {}

        # Three-pool tier3 split
        self.tier3_legacy: List[dict] = []
        self.tier3_organic: List[dict] = []
        self.tier3_expanded: List[dict] = []

        with open(bundle_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    bundle = json.loads(line)
                    metadata = bundle.get('metadata', {})
                    tier = metadata.get('difficulty_tier', 'tier1_easy')

                    anchor = bundle.get('anchor')
                    positive = bundle.get('positive')
                    if not anchor or not positive:
                        continue
                    if not anchor.get('text') or not positive.get('text'):
                        continue

                    self.bundles_by_tier[tier].append(bundle)

                    # Three-pool tier3 classification
                    if tier == 'tier3_adversarial':
                        source = metadata.get('source', '')
                        expansion = metadata.get('expansion', '')

                        if source == 'tier3_organic_miner':
                            self.tier3_organic.append(bundle)
                        elif expansion == 'tier3x' or source == 'tier3_expansion':
                            self.tier3_expanded.append(bundle)
                        else:
                            self.tier3_legacy.append(bundle)

                    if max_samples:
                        total = sum(len(v) for v in self.bundles_by_tier.values())
                        if total >= max_samples:
                            break

        self.tiers = list(self.bundles_by_tier.keys())
        total = sum(len(v) for v in self.bundles_by_tier.values())
        print(f"Loaded {total} bundles from {bundle_path}")
        for tier, bundles in sorted(self.bundles_by_tier.items()):
            print(f"  {tier}: {len(bundles)}")

        print(f"  tier3 pools: {len(self.tier3_legacy)} legacy + "
              f"{len(self.tier3_organic)} organic + {len(self.tier3_expanded)} expanded")

    def get_tier_counts(self) -> Dict[str, int]:
        return {t: len(b) for t, b in self.bundles_by_tier.items()}

    def sample_bundle(self, tier: str, rotate_positives: bool = False,
                      tier3_mix: Tier3MixRatios = None) -> dict:
        """Sample with three-pool tier3 mixing."""

        if tier == 'tier3_adversarial' and tier3_mix is not None:
            pool = tier3_mix.sample_pool()

            if pool == "legacy" and self.tier3_legacy:
                return random.choice(self.tier3_legacy)
            elif pool == "organic" and self.tier3_organic:
                return random.choice(self.tier3_organic)
            elif pool == "expanded" and self.tier3_expanded:
                return random.choice(self.tier3_expanded)

            # Fallback: try any non-empty pool
            for fallback_pool in [self.tier3_legacy, self.tier3_organic, self.tier3_expanded]:
                if fallback_pool:
                    return random.choice(fallback_pool)

        bundles = self.bundles_by_tier.get(tier, [])
        if not bundles:
            all_bundles = [b for bs in self.bundles_by_tier.values() for b in bs]
            return random.choice(all_bundles)

        return random.choice(bundles)

    def extract_triplet(self, bundle: dict) -> Tuple[str, str, str]:
        """Extract (anchor, positive, negative) texts from bundle."""
        anchor_text = bundle['anchor']['text']
        positive_text = bundle['positive']['text']

        negatives = bundle.get('negatives', {})
        if isinstance(negatives, dict):
            all_negs = negatives.get('within_lemma', []) + negatives.get('cross_lemma', [])
        else:
            all_negs = negatives if isinstance(negatives, list) else []

        if all_negs:
            neg = random.choice(all_negs)
            negative_text = neg['text'] if isinstance(neg, dict) else neg
        else:
            other_tier = random.choice(self.tiers)
            other_bundle = random.choice(self.bundles_by_tier[other_tier])
            negative_text = other_bundle['positive']['text']

        return anchor_text, positive_text, negative_text


class CurriculumSampler:
    """Samples batches with three-pool tier3 mixing."""

    def __init__(self, dataset: TieredBundleDataset, config: CurriculumConfig,
                 batch_size: int = 32, total_steps: int = 5000,
                 tier3_mix: Tier3MixRatios = None):
        self.dataset = dataset
        self.config = config
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.current_step = 0
        self.tier3_mix = tier3_mix
        self.tier3_replay_queue: List[Tuple[dict, int]] = []

        if tier3_mix is not None:
            print(f"Tier3 three-pool mix: {tier3_mix.legacy:.0%} legacy, "
                  f"{tier3_mix.organic:.0%} organic, {tier3_mix.expanded:.0%} expanded")

    def get_phase(self) -> str:
        progress = self.current_step / self.total_steps
        if progress < self.config.early_end:
            return 'early'
        elif progress < self.config.mid_end:
            return 'mid'
        else:
            return 'late'

    def get_quota(self) -> Tuple[int, int, int]:
        phase = self.get_phase()
        if phase == 'early':
            quota = self.config.early_quota
        elif phase == 'mid':
            quota = self.config.mid_quota
        else:
            quota = self.config.late_quota

        total_quota = sum(quota)
        scale = self.batch_size / total_quota
        scaled = [int(q * scale) for q in quota]
        diff = self.batch_size - sum(scaled)
        scaled[0] += diff

        max_tier3 = int(self.batch_size * self.config.tier3_max_fraction)
        if scaled[2] > max_tier3:
            excess = scaled[2] - max_tier3
            scaled[2] = max_tier3
            scaled[0] += excess

        return tuple(scaled)

    def sample_batch(self) -> List[Tuple[str, str, str, str, Optional[dict]]]:
        quota = self.get_quota()
        tier_names = ['tier1_easy', 'tier2_robust', 'tier3_adversarial']
        batch = []

        replay_count = 0
        replay_limit = min(2, quota[2])
        while replay_count < replay_limit and self.tier3_replay_queue:
            replay_bundle = self.pop_replay()
            if replay_bundle:
                anchor, positive, negative = self.dataset.extract_triplet(replay_bundle)
                batch.append((anchor, positive, negative, 'tier3_adversarial', replay_bundle))
                replay_count += 1

        adjusted_quota = list(quota)
        adjusted_quota[2] = max(0, adjusted_quota[2] - replay_count)

        for tier_idx, count in enumerate(adjusted_quota):
            tier = tier_names[tier_idx]
            if tier not in self.dataset.bundles_by_tier or not self.dataset.bundles_by_tier[tier]:
                tier = 'tier1_easy'

            for _ in range(count):
                tier3_mix_arg = self.tier3_mix if tier == 'tier3_adversarial' else None
                bundle = self.dataset.sample_bundle(
                    tier,
                    rotate_positives=(tier == 'tier3_adversarial' and self.config.tier3_positive_rotation),
                    tier3_mix=tier3_mix_arg
                )
                anchor, positive, negative = self.dataset.extract_triplet(bundle)
                include_bundle = bundle if tier == 'tier3_adversarial' else None
                batch.append((anchor, positive, negative, tier, include_bundle))

        random.shuffle(batch)
        return batch

    def step(self):
        self.current_step += 1
        self.tier3_replay_queue = [
            (b, exp) for b, exp in self.tier3_replay_queue
            if exp > self.current_step
        ]

    def add_to_replay(self, bundle: dict):
        if not self.config.tier3_replay_enabled:
            return
        expire_step = self.current_step + self.config.tier3_replay_window
        self.tier3_replay_queue.append((bundle, expire_step))

    def pop_replay(self) -> Optional[dict]:
        if not self.tier3_replay_queue:
            return None
        bundle, _ = self.tier3_replay_queue.pop(0)
        return bundle


class SemanticPhaseModel(nn.Module):
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
    mean: float
    median: float
    std: float
    pass_rate: float
    q10: float
    q90: float


def compute_margin_stats(margins: List[float]) -> MarginStats:
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
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    return neg_dist - pos_dist


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Training with seed {args.seed}")
    print(f"Curriculum: {args.curriculum}")

    dataset = TieredBundleDataset(args.bundles, max_samples=args.max_samples)

    if args.curriculum == 'phased':
        config = CurriculumConfig()
    elif args.curriculum == 'phased_v2':
        config = C2B_CONFIG
    else:
        config = CurriculumConfig(
            early_quota=(11, 11, 10),
            mid_quota=(11, 11, 10),
            late_quota=(11, 11, 10),
            tier3_max_fraction=0.35,
            tier3_positive_rotation=False
        )

    # Build tier3 mix ratios
    tier3_mix = None
    if args.tier3_legacy is not None or args.tier3_organic is not None or args.tier3_expanded is not None:
        legacy = args.tier3_legacy if args.tier3_legacy is not None else 0.80
        organic = args.tier3_organic if args.tier3_organic is not None else 0.10
        expanded = args.tier3_expanded if args.tier3_expanded is not None else 0.10

        # Normalize if they don't sum to 1
        total = legacy + organic + expanded
        if total > 0:
            legacy /= total
            organic /= total
            expanded /= total

        tier3_mix = Tier3MixRatios(legacy=legacy, organic=organic, expanded=expanded)

    sampler = CurriculumSampler(
        dataset, config,
        batch_size=args.batch_size,
        total_steps=args.steps,
        tier3_mix=tier3_mix
    )

    model = SemanticPhaseModel(embed_dim=args.embed_dim, phase_dim=args.phase_dim)
    model = model.to(args.device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    model.train()
    total_loss = 0
    total_correct = 0
    margins_by_tier: Dict[str, List[float]] = defaultdict(list)
    pbar = tqdm(total=args.steps, desc="Training")

    for step in range(args.steps):
        batch = sampler.sample_batch()
        anchors = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        negatives = [b[2] for b in batch]
        tiers = [b[3] for b in batch]
        bundles = [b[4] for b in batch]

        anchor_emb = model(anchors)
        positive_emb = model(positives)
        negative_emb = model(negatives)

        loss = contrastive_loss(anchor_emb, positive_emb, negative_emb, margin=args.margin)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            batch_margins = compute_margins(anchor_emb, positive_emb, negative_emb)
            for i, tier in enumerate(tiers):
                margin_val = batch_margins[i].item()
                margins_by_tier[tier].append(margin_val)
                if tier == 'tier3_adversarial' and margin_val < 0 and bundles[i] is not None:
                    sampler.add_to_replay(bundles[i])

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

    # Margin stats
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
        print(f"  Pass rate:     {stats.pass_rate:.1%}")

    # Final eval
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
        print(f"\n{tier}:")
        print(f"  Accuracy:      {stats.pass_rate:.1%}")
        print(f"  Mean margin:   {stats.mean:.4f}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")

    tier3_mix_dict = None
    if tier3_mix:
        tier3_mix_dict = {"legacy": tier3_mix.legacy, "organic": tier3_mix.organic, "expanded": tier3_mix.expanded}

    metrics = {
        'final_accuracy': final_acc,
        'curriculum': args.curriculum,
        'steps': args.steps,
        'seed': args.seed,
        'bundle_path': args.bundles,
        'tier3_mix': tier3_mix_dict,
        'timestamp': datetime.now().isoformat(),
        'training_margin_stats': margin_stats,
        'eval_margin_stats': eval_margin_stats,
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save checkpoint for eval_capsules.py compatibility
    torch.save(model.state_dict(), output_dir / "checkpoint_final.pt")

    print(f"\nResults saved to {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Curriculum training with three-pool tier3 mixing")
    parser.add_argument('--bundles', required=True, help='Path to bundles JSONL')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--phase-dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='runs/curriculum')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--curriculum', choices=['phased', 'phased_v2', 'uniform'], default='phased')

    # Three-pool tier3 mixing
    parser.add_argument('--tier3-legacy', type=float, default=None,
                       help='Tier3 legacy (original killer) ratio. Default 0.80')
    parser.add_argument('--tier3-organic', type=float, default=None,
                       help='Tier3 organic (newly mined) ratio. Default 0.10')
    parser.add_argument('--tier3-expanded', type=float, default=None,
                       help='Tier3 expanded (vetted tier3x) ratio. Default 0.10')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
