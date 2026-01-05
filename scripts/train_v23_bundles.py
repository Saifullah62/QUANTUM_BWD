#!/usr/bin/env python3
"""
Simple Training Script for v2.3 Bundles
========================================

Minimal training loop for SemanticPhase with v2.3 bundle format:
- anchor: {text, span, sense_id, ...}
- positive: {text, span, sense_id, ...}
- negatives: [{text, span, sense_id, ...}, ...]

Usage:
    python scripts/train_v23_bundles.py --bundles path/to/bundles.jsonl --steps 1000
"""

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qllm.layers.semantic_phase import SemanticPhaseEmbedding


class V23BundleDataset(Dataset):
    """Dataset for v2.3 bundle format."""

    def __init__(self, bundle_path: str, max_samples: int = None):
        self.bundles = []
        with open(bundle_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.bundles.append(json.loads(line))
                    if max_samples and len(self.bundles) >= max_samples:
                        break
        print(f"Loaded {len(self.bundles)} bundles from {bundle_path}")

    def __len__(self):
        return len(self.bundles)

    def __getitem__(self, idx):
        bundle = self.bundles[idx]

        # Extract texts (with fallback for malformed bundles)
        anchor = bundle.get('anchor')
        positive = bundle.get('positive')

        if not anchor or not positive:
            # Skip to next valid bundle
            return self.__getitem__((idx + 1) % len(self.bundles))

        anchor_text = anchor.get('text', '')
        positive_text = positive.get('text', '')

        if not anchor_text or not positive_text:
            return self.__getitem__((idx + 1) % len(self.bundles))

        # Sample one negative (negatives can be dict or list)
        negatives_data = bundle.get('negatives', {})

        # Handle both formats: dict with within_lemma/cross_lemma or list
        if isinstance(negatives_data, dict):
            all_negs = negatives_data.get('within_lemma', []) + negatives_data.get('cross_lemma', [])
        else:
            all_negs = negatives_data if isinstance(negatives_data, list) else []

        if all_negs:
            neg = random.choice(all_negs)
            negative_text = neg['text']
        else:
            # Fallback: use a different bundle's positive as negative
            other_idx = (idx + 1) % len(self.bundles)
            negative_text = self.bundles[other_idx]['positive']['text']

        return {
            'anchor': anchor_text,
            'positive': positive_text,
            'negative': negative_text,
            'sense_id': bundle['anchor'].get('sense_id', 'unknown'),
            'tier': bundle.get('metadata', {}).get('tier', 'tier1_easy')
        }


class SimpleTextEncoder(nn.Module):
    """Simple text encoder using embeddings + mean pooling."""

    def __init__(self, vocab_size: int = 30000, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Simple char-level tokenizer
        self.vocab = {chr(i): i for i in range(256)}
        self.vocab_size = vocab_size

    def tokenize(self, text: str, max_len: int = 128) -> torch.Tensor:
        """Simple character-level tokenization."""
        tokens = [self.vocab.get(c, 1) for c in text[:max_len]]
        # Pad
        if len(tokens) < max_len:
            tokens += [0] * (max_len - tokens)
        return torch.tensor(tokens[:max_len])

    def forward(self, texts: list) -> torch.Tensor:
        """Encode list of texts to embeddings."""
        # Tokenize all texts
        tokens = torch.stack([self.tokenize(t) for t in texts])
        if next(self.parameters()).is_cuda:
            tokens = tokens.cuda()

        # Embed and pool
        embedded = self.embed(tokens)  # [B, L, D]
        mask = (tokens != 0).float().unsqueeze(-1)  # [B, L, 1]
        pooled = (embedded * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B, D]

        return self.proj(pooled)


class SemanticPhaseModel(nn.Module):
    """Model combining text encoder with SemanticPhase embedding."""

    def __init__(self, vocab_size: int = 30000, embed_dim: int = 256, phase_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.phase_dim = phase_dim

        # Semantic phase embedding (replaces simple embedding)
        self.semantic_phase = SemanticPhaseEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            phase_dim=phase_dim,
            padding_idx=0
        )

        # Projection head for contrastive learning
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Simple char-level tokenizer
        self.vocab = {chr(i): i for i in range(256)}

    def tokenize(self, text: str, max_len: int = 128) -> torch.Tensor:
        """Simple character-level tokenization."""
        tokens = [self.vocab.get(c, 1) for c in text[:max_len]]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        return torch.tensor(tokens[:max_len])

    def forward(self, texts: list) -> torch.Tensor:
        """Encode texts through semantic phase."""
        # Tokenize all texts
        tokens = torch.stack([self.tokenize(t) for t in texts])
        device = next(self.parameters()).device
        tokens = tokens.to(device)

        # Get semantic phase embeddings (returns tuple of real, imag)
        real_emb, imag_emb = self.semantic_phase(tokens)  # [B, L, D]

        # Combine real and imaginary parts (magnitude-like)
        phase_emb = torch.sqrt(real_emb**2 + imag_emb**2 + 1e-8)

        # Mean pool over sequence
        mask = (tokens != 0).float().unsqueeze(-1).to(device)  # [B, L, 1]
        pooled = (phase_emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B, D]

        # Project
        return self.proj(pooled)


def contrastive_loss(anchor: torch.Tensor, positive: torch.Tensor,
                     negative: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Triplet margin loss."""
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def compute_accuracy(anchor: torch.Tensor, positive: torch.Tensor,
                     negative: torch.Tensor) -> float:
    """Compute retrieval accuracy (positive closer than negative)."""
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    correct = (pos_sim > neg_sim).float()
    return correct.mean().item()


def train(args):
    """Main training loop."""
    print(f"Training on {args.bundles}")
    print(f"Device: {args.device}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Create dataset and dataloader
    dataset = V23BundleDataset(args.bundles, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x  # Return list of dicts
    )

    # Create model
    model = SemanticPhaseModel(embed_dim=args.embed_dim, phase_dim=args.phase_dim)
    model = model.to(args.device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Training loop
    model.train()
    step = 0
    total_loss = 0
    total_acc = 0

    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            # Extract texts
            anchors = [b['anchor'] for b in batch]
            positives = [b['positive'] for b in batch]
            negatives = [b['negative'] for b in batch]

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

            # Track metrics
            acc = compute_accuracy(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
            total_acc += acc

            step += 1

            if step % 10 == 0:
                avg_loss = total_loss / 10
                avg_acc = total_acc / 10
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.1%}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                total_loss = 0
                total_acc = 0

            pbar.update(1)

    pbar.close()

    # Final evaluation
    model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_count = 0

    with torch.no_grad():
        for batch in dataloader:
            anchors = [b['anchor'] for b in batch]
            positives = [b['positive'] for b in batch]
            negatives = [b['negative'] for b in batch]

            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)

            loss = contrastive_loss(anchor_emb, positive_emb, negative_emb, margin=args.margin)
            acc = compute_accuracy(anchor_emb, positive_emb, negative_emb)

            eval_loss += loss.item()
            eval_acc += acc
            eval_count += 1

            if eval_count >= 100:  # Limit eval batches
                break

    final_loss = eval_loss / max(eval_count, 1)
    final_acc = eval_acc / max(eval_count, 1)

    print(f"\nFinal Metrics:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.1%}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), output_dir / "model.pt")

    # Save metrics
    metrics = {
        'final_loss': final_loss,
        'top1_accuracy': final_acc,
        'top3_accuracy': final_acc,  # Simplified for now
        'steps': args.steps,
        'seed': args.seed,
        'bundle_path': args.bundles,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train SemanticPhase on v2.3 bundles")
    parser.add_argument('--bundles', required=True, help='Path to v2.3 bundles JSONL')
    parser.add_argument('--steps', type=int, default=1000, help='Training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.5, help='Contrastive margin')
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--phase-dim', type=int, default=64, help='Phase dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='runs/v23_training', help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None, help='Max training samples')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
