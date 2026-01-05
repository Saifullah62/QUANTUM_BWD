#!/usr/bin/env python3
"""
Post-Hoc Capsule Evaluator
==========================

Evaluates all trained capsules against a frozen eval set, producing a
tier-stratified scoreboard for comparing conditions A/B/C.

This script loads SemanticPhaseModel checkpoints (model.pt) and evaluates
them against the same frozen eval items.

Usage:
    python experiments/eval_capsules.py \
        --eval evals/eval_v23_contrastive_5k.jsonl \
        --results-root results/scaling \
        --device cuda \
        --out results/scaling_eval_scoreboard.json

Output:
    - One line per capsule: "A seed 42: eval_acc=0.723 mean_margin=0.156"
    - JSON scoreboard with overall + tier-stratified results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from qllm.layers.semantic_phase import SemanticPhaseEmbedding
except ImportError:
    print("Warning: Could not import SemanticPhaseEmbedding from qllm.layers")
    print("Using inline definition...")
    SemanticPhaseEmbedding = None


class SemanticPhaseEmbeddingInline(nn.Module):
    """
    Inline copy of SemanticPhaseEmbedding for standalone evaluation.
    This matches the architecture from qllm/layers/semantic_phase.py
    """

    def __init__(self, vocab_size: int = 30000, embedding_dim: int = 256,
                 phase_dim: int = 64, max_seq_len: int = 2048, padding_idx: int = 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.phase_dim = phase_dim

        # Rotation frequencies
        self.register_buffer('rotation_freqs', torch.randn(phase_dim * 2))
        self.register_buffer('positional_phases', torch.randn(max_seq_len, phase_dim * 2))

        # Embeddings
        self.real_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.imag_embedding = nn.Embedding(vocab_size, phase_dim, padding_idx=padding_idx)

        # Context to rotation
        self.context_to_rotation = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, phase_dim * 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (real_embedding, imag_embedding)"""
        real = self.real_embedding(x)
        imag = self.imag_embedding(x)
        return real, imag


class SemanticPhaseModel(nn.Module):
    """
    Model combining text encoder with SemanticPhase embedding.
    Matches the architecture from scripts/train_v23_bundles.py
    """

    def __init__(self, vocab_size: int = 30000, embed_dim: int = 256, phase_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.phase_dim = phase_dim

        # Use imported or inline SemanticPhaseEmbedding
        EmbeddingClass = SemanticPhaseEmbedding if SemanticPhaseEmbedding else SemanticPhaseEmbeddingInline
        self.semantic_phase = EmbeddingClass(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            phase_dim=phase_dim,
            padding_idx=0
        )

        # Projection head
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

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts through semantic phase."""
        # Tokenize all texts
        tokens = torch.stack([self.tokenize(t) for t in texts])
        device = next(self.parameters()).device
        tokens = tokens.to(device)

        # Get semantic phase embeddings (returns tuple of real, imag)
        real_emb, imag_emb = self.semantic_phase(tokens)  # [B, L, D]

        # Combine real and imaginary parts (magnitude)
        phase_emb = torch.sqrt(real_emb**2 + imag_emb**2 + 1e-8)

        # Mean pool over sequence
        mask = (tokens != 0).float().unsqueeze(-1).to(device)  # [B, L, 1]
        pooled = (phase_emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B, D]

        # Project
        return self.proj(pooled)


def load_eval_set(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load eval set from JSONL file."""
    header = {}
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("_header"):
                header = d
            else:
                items.append(d)
    return header, items


def load_model(capsule_dir: Path, device: str) -> SemanticPhaseModel:
    """Load trained model from capsule directory."""
    model_path = capsule_dir / "model.pt"

    # Load state dict
    state_dict = torch.load(model_path, map_location="cpu")

    # Infer model dimensions from state dict
    embed_dim = state_dict["proj.0.weight"].shape[1]  # Input dim of first proj layer
    phase_dim = state_dict["semantic_phase.imag_embedding.weight"].shape[1]
    vocab_size = state_dict["semantic_phase.real_embedding.weight"].shape[0]

    # Create model with matching dimensions
    model = SemanticPhaseModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        phase_dim=phase_dim
    )

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


@torch.no_grad()
def score_item(model: SemanticPhaseModel, anchor_text: str, positive_text: str,
               negative_texts: List[str], device: str) -> Tuple[bool, float, int]:
    """
    Score a single eval item.

    Returns:
        (correct_top1, margin, best_correct_rank)
    """
    # Encode all texts
    all_texts = [anchor_text, positive_text] + negative_texts
    embeddings = model(all_texts)

    # Normalize for cosine similarity
    embeddings = F.normalize(embeddings, dim=-1)

    anchor_emb = embeddings[0:1]  # [1, D]
    positive_emb = embeddings[1:2]  # [1, D]
    negative_embs = embeddings[2:]  # [K, D]

    # Compute similarities
    pos_sim = (anchor_emb * positive_emb).sum(dim=-1)  # [1]
    neg_sims = (anchor_emb * negative_embs).sum(dim=-1)  # [K]
    best_neg_sim = neg_sims.max() if len(neg_sims) > 0 else torch.tensor(-1.0, device=device)

    # Top-1 correct if positive is closer than all negatives
    correct = (pos_sim > best_neg_sim).item()
    margin = (pos_sim - best_neg_sim).item()

    # Compute rank (1 = positive is top, higher = worse)
    all_sims = torch.cat([pos_sim, neg_sims])
    rank = (all_sims > pos_sim).sum().item() + 1  # 1-indexed

    return correct, margin, rank


def evaluate_capsule(model: SemanticPhaseModel, items: List[Dict],
                     device: str, batch_desc: str = "") -> Dict[str, Any]:
    """
    Evaluate a model against eval items.

    Returns dict with overall and tier-stratified metrics including margin distribution.
    """
    totals = {"n": 0, "correct": 0, "margin_sum": 0.0, "rank_sum": 0}
    by_tier = {}
    margins_by_tier: Dict[str, List[float]] = {}
    all_margins: List[float] = []

    for item in tqdm(items, desc=batch_desc, leave=False):
        tier = str(item.get("tier", "NA"))
        if tier not in by_tier:
            by_tier[tier] = {"n": 0, "correct": 0, "margin_sum": 0.0, "rank_sum": 0}
            margins_by_tier[tier] = []

        correct, margin, rank = score_item(
            model,
            item["anchor_text"],
            item["positive_text"],
            item["negative_texts"],
            device
        )

        # Update totals
        totals["n"] += 1
        totals["correct"] += int(correct)
        totals["margin_sum"] += margin
        totals["rank_sum"] += rank
        all_margins.append(margin)

        # Update tier stats
        by_tier[tier]["n"] += 1
        by_tier[tier]["correct"] += int(correct)
        by_tier[tier]["margin_sum"] += margin
        by_tier[tier]["rank_sum"] += rank
        margins_by_tier[tier].append(margin)

    # Compute margin distribution stats
    import numpy as np

    def margin_stats(margins: List[float]) -> Dict[str, float]:
        if not margins:
            return {"median_margin": 0.0, "pass_rate": 0.0, "q10": 0.0, "q90": 0.0}
        arr = np.array(margins)
        return {
            "median_margin": float(np.median(arr)),
            "pass_rate": float((arr > 0).mean()),
            "q10_margin": float(np.percentile(arr, 10)),
            "q90_margin": float(np.percentile(arr, 90))
        }

    # Finalize totals
    n = max(1, totals["n"])
    totals["accuracy"] = totals["correct"] / n
    totals["mean_margin"] = totals["margin_sum"] / n
    totals["mean_rank"] = totals["rank_sum"] / n
    totals.update(margin_stats(all_margins))
    del totals["margin_sum"], totals["rank_sum"]

    # Finalize tier stats
    for tier, tier_stats in by_tier.items():
        tn = max(1, tier_stats["n"])
        tier_stats["accuracy"] = tier_stats["correct"] / tn
        tier_stats["mean_margin"] = tier_stats["margin_sum"] / tn
        tier_stats["mean_rank"] = tier_stats["rank_sum"] / tn
        tier_stats.update(margin_stats(margins_by_tier[tier]))
        del tier_stats["margin_sum"], tier_stats["rank_sum"]

    return {"overall": totals, "by_tier": by_tier}


def main():
    parser = argparse.ArgumentParser(description="Evaluate capsules against frozen eval set")
    parser.add_argument("--eval", required=True, help="Eval JSONL file")
    parser.add_argument("--results-root", default="results/scaling",
                       help="Root folder containing condition/seed capsules")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/scaling_eval_scoreboard.json",
                       help="Output scoreboard JSON")
    args = parser.parse_args()

    # Load eval set
    eval_path = Path(args.eval)
    header, items = load_eval_set(eval_path)

    # Log eval fingerprint for reproducibility
    fingerprint = f"{header.get('source_sha256', 'unknown')[:16]}|{header.get('content_hash', 'unknown')}|n={header.get('eval_count', len(items))}"
    print(f"Loaded eval set: {eval_path}")
    print(f"EVAL_FINGERPRINT: {fingerprint}")
    print(f"  Items: {len(items)}")
    print(f"  Tiers: {header.get('tier_counts', {})}")

    # Find all capsules
    results_root = Path(args.results_root)
    capsules = sorted([
        p for p in results_root.glob("*/*")
        if p.is_dir() and (p / "model.pt").exists()
    ])
    print(f"\nFound {len(capsules)} capsules to evaluate")

    # Build scoreboard
    scoreboard = {
        "eval_header": header,
        "eval_path": str(eval_path),
        "device": args.device,
        "capsules": [],
    }

    # Evaluate each capsule
    for capsule_dir in capsules:
        condition = capsule_dir.parent.name
        seed_str = capsule_dir.name.split("_")[-1]
        seed = int(seed_str) if seed_str.isdigit() else None

        print(f"\nEvaluating {condition}/seed_{seed}...")

        try:
            model = load_model(capsule_dir, args.device)
            results = evaluate_capsule(model, items, args.device, f"{condition} s{seed}")

            record = {
                "condition": condition,
                "seed": seed,
                "capsule_dir": str(capsule_dir),
                **results
            }
            scoreboard["capsules"].append(record)

            # Print summary
            overall = results["overall"]
            print(f"  {condition} seed {seed}: "
                  f"eval_acc={overall['accuracy']:.3f} "
                  f"mean_margin={overall['mean_margin']:.3f} "
                  f"mean_rank={overall['mean_rank']:.1f}")

            # Print tier breakdown
            for tier, stats in sorted(results["by_tier"].items()):
                print(f"    {tier}: acc={stats['accuracy']:.3f} "
                      f"margin={stats['mean_margin']:.3f} n={stats['n']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            scoreboard["capsules"].append({
                "condition": condition,
                "seed": seed,
                "capsule_dir": str(capsule_dir),
                "error": str(e)
            })

    # Aggregate by condition
    print("\n" + "=" * 60)
    print("CONDITION SUMMARY")
    print("=" * 60)

    by_condition = {}
    for rec in scoreboard["capsules"]:
        if "error" in rec:
            continue
        cond = rec["condition"]
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(rec)

    summary = {}
    for cond in sorted(by_condition.keys()):
        recs = by_condition[cond]
        accs = [r["overall"]["accuracy"] for r in recs]
        margins = [r["overall"]["mean_margin"] for r in recs]

        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
        mean_margin = sum(margins) / len(margins)

        summary[cond] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_margin": mean_margin,
            "n_seeds": len(recs)
        }

        print(f"{cond}: {mean_acc:.1%} +/- {std_acc:.1%} (margin={mean_margin:.3f}, n={len(recs)})")

    # Tier-stratified summary by condition
    print("\nTIER-STRATIFIED BREAKDOWN:")
    all_tiers = set()
    for rec in scoreboard["capsules"]:
        if "error" not in rec:
            all_tiers.update(rec.get("by_tier", {}).keys())

    for tier in sorted(all_tiers):
        print(f"\n  {tier}:")
        for cond in sorted(by_condition.keys()):
            tier_accs = [
                r["by_tier"].get(tier, {}).get("accuracy", 0)
                for r in by_condition[cond]
                if tier in r.get("by_tier", {})
            ]
            if tier_accs:
                mean_tier = sum(tier_accs) / len(tier_accs)
                print(f"    {cond}: {mean_tier:.1%}")

    scoreboard["summary"] = summary

    # Save scoreboard
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(scoreboard, f, indent=2)
    print(f"\nWrote scoreboard: {out_path}")


if __name__ == "__main__":
    main()
