#!/usr/bin/env python3
"""
Bundle Dataset for v2 Polysemy Bundles
======================================

Loads v2 bundles and provides them to the trainer with explicit
contrastive structure preserved - no mining needed.
"""

import json
import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Ensure paradigm_factory is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.polysemy_bundle_v2 import (
    PolysemyBundle, load_bundles, BundleItem
)


@dataclass
class BundleBatch:
    """A batch that preserves bundle structure for direct contrastive loss."""
    # Core tensors
    input_ids: torch.Tensor          # [B, seq_len]
    attention_mask: torch.Tensor     # [B, seq_len]

    # Bundle structure (for contrastive loss)
    bundle_ids: List[str]            # Which bundle each item belongs to
    item_ids: List[str]              # Unique item ID within bundle (for killer neg tracking)
    item_roles: List[str]            # "anchor", "positive", "negative", "hard_negative"
    sense_ids: List[str]             # Sense ID for each item
    same_sense_as_anchor: List[bool] # Target label

    # Contrastive pairs (precomputed from bundle structure)
    anchor_indices: List[int]        # Index of anchor for each non-anchor item
    pair_margins: List[float]        # Expected margin for each pair

    # Metadata
    difficulties: List[float]


class BundleDataset(Dataset):
    """
    Dataset that loads v2 polysemy bundles and preserves structure.

    Each __getitem__ returns a single BundleItem with its bundle context,
    but the collator reconstructs bundle structure for contrastive loss.
    """

    def __init__(
        self,
        bundle_path: Path,
        max_length: int = 128,
        vocab_size: int = 1000,
        include_hard_negatives: bool = True
    ):
        self.bundle_path = Path(bundle_path)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.include_hard_negatives = include_hard_negatives

        # Load bundles
        self.bundles = load_bundles(self.bundle_path)
        print(f"Loaded {len(self.bundles)} v2 bundles")

        # Flatten to items for iteration, but keep bundle reference
        self.items = []
        for bundle in self.bundles:
            for item in bundle.items:
                if item.role == "hard_negative" and not include_hard_negatives:
                    continue
                self.items.append({
                    "bundle": bundle,
                    "item": item,
                    "bundle_id": bundle.bundle_id
                })

        print(f"Total items: {len(self.items)}")

        # Build vocab from all contexts
        self._build_vocab()

        # Build bundle index for efficient batch construction
        self._build_bundle_index()

    def _build_vocab(self):
        """Build vocabulary from bundle contexts."""
        word_counts = {}
        for entry in self.items:
            text = entry["item"].context
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

        self.word_to_id = {'<pad>': 0, '<unk>': 1}
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - 2]):
            self.word_to_id[word] = i + 2

    def _build_bundle_index(self):
        """Build index for efficient bundle-based sampling."""
        self.bundle_to_indices = {}
        for idx, entry in enumerate(self.items):
            bid = entry["bundle_id"]
            if bid not in self.bundle_to_indices:
                self.bundle_to_indices[bid] = []
            self.bundle_to_indices[bid].append(idx)

        # Separate anchors from non-anchors for pairing
        self.anchor_indices = []
        self.non_anchor_indices = []
        for idx, entry in enumerate(self.items):
            if entry["item"].role == "anchor":
                self.anchor_indices.append(idx)
            else:
                self.non_anchor_indices.append(idx)

        print(f"Bundle index: {len(self.bundle_to_indices)} bundles, "
              f"{len(self.anchor_indices)} anchors, {len(self.non_anchor_indices)} non-anchors")

    def tokenize(self, text: str) -> List[int]:
        """Simple word-level tokenization."""
        tokens = []
        for word in text.lower().split()[:self.max_length]:
            tokens.append(self.word_to_id.get(word, 1))  # 1 = <unk>

        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(0)  # 0 = <pad>

        return tokens[:self.max_length]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.items[idx]
        item = entry["item"]
        bundle = entry["bundle"]

        input_ids = self.tokenize(item.context)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1 if t > 0 else 0 for t in input_ids], dtype=torch.long),
            "bundle_id": bundle.bundle_id,
            "item_id": item.item_id,
            "role": item.role,
            "sense_id": item.sense_id,
            "same_sense_as_anchor": item.target.same_sense_as_anchor,
            "difficulty": item.difficulty,
            "word": bundle.word.surface,
            "paradigm": "semantic_phase",
            "margin": bundle.contrastive_targets.margin.get(
                "positive_vs_hard_negative" if item.role == "hard_negative" else "positive_vs_negative",
                0.15
            )
        }


class BundleAwareSampler(Sampler):
    """
    Sampler that ensures each batch contains complete bundle structure.

    Strategy: Sample anchors, then include their positives/negatives
    to guarantee contrastive pairs in every batch.
    """

    def __init__(
        self,
        dataset: BundleDataset,
        batch_size: int = 8,
        bundles_per_batch: int = 2,  # Number of complete bundles per batch
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bundles_per_batch = bundles_per_batch
        self.seed = seed

        self.bundle_ids = list(dataset.bundle_to_indices.keys())
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __iter__(self):
        # Shuffle bundle order
        bundle_order = torch.randperm(len(self.bundle_ids), generator=self.rng).tolist()

        batch = []
        current_bundles = 0

        for bundle_idx in bundle_order:
            bundle_id = self.bundle_ids[bundle_idx]
            item_indices = self.dataset.bundle_to_indices[bundle_id]

            # Check if adding this bundle would exceed batch size significantly
            # If so, yield current batch first (keeping bundles intact)
            if len(batch) > 0 and len(batch) + len(item_indices) > self.batch_size * 1.5:
                yield batch
                batch = []
                current_bundles = 0

            # Add ALL items from this bundle (never split a bundle)
            batch.extend(item_indices)
            current_bundles += 1

            # Yield when we have enough complete bundles
            if current_bundles >= self.bundles_per_batch and len(batch) >= self.batch_size:
                yield batch
                batch = []
                current_bundles = 0

        # Final partial batch (if has at least one complete bundle)
        if batch:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def bundle_collator(batch: List[Dict[str, Any]]) -> BundleBatch:
    """
    Collate function that preserves bundle structure.

    Key output: precomputed contrastive pairs with their expected margins.
    """
    # Stack tensors
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])

    # Extract metadata lists
    bundle_ids = [b["bundle_id"] for b in batch]
    item_ids = [b["item_id"] for b in batch]
    item_roles = [b["role"] for b in batch]
    sense_ids = [b["sense_id"] for b in batch]
    same_sense = [b["same_sense_as_anchor"] for b in batch]
    difficulties = [b["difficulty"] for b in batch]
    margins = [b["margin"] for b in batch]

    # Precompute anchor indices for each non-anchor item
    # (For contrastive loss: pair each positive/negative with its anchor)
    anchor_map = {}  # bundle_id -> index of anchor in this batch
    for idx, (bid, role) in enumerate(zip(bundle_ids, item_roles)):
        if role == "anchor":
            anchor_map[bid] = idx

    anchor_indices = []
    pair_margins = []
    for idx, (bid, role, margin) in enumerate(zip(bundle_ids, item_roles, margins)):
        if role != "anchor":
            anchor_idx = anchor_map.get(bid, -1)  # -1 if anchor not in batch
            anchor_indices.append(anchor_idx)
            pair_margins.append(margin)
        else:
            anchor_indices.append(-1)  # Anchors don't have an anchor
            pair_margins.append(0.0)

    return BundleBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        bundle_ids=bundle_ids,
        item_ids=item_ids,
        item_roles=item_roles,
        sense_ids=sense_ids,
        same_sense_as_anchor=same_sense,
        anchor_indices=anchor_indices,
        pair_margins=pair_margins,
        difficulties=difficulties
    )


def _bucket3(d: float) -> str:
    """Map difficulty to bucket: easy/med/hard."""
    if d < 0.33:
        return "easy"
    if d < 0.66:
        return "med"
    return "hard"


def compute_bundle_contrastive_loss(
    embeddings: torch.Tensor,  # [B, hidden_dim] - pooled embeddings
    batch: BundleBatch,
    temperature: float = 0.1,
    margin_easy: float = 0.05,  # Certification margin for easy negatives
    margin_hard: float = 0.15,  # Certification margin for hard negatives
    device: str = 'cuda',
    margin_boost: float = 1.0,  # Late-stage margin multiplier (1.0 = no boost)
    hard_neg_penalty_mult: float = 2.0,  # Focus on hard negatives in slack loss (2-3x recommended)
    track_killers: int = 0,  # Track K worst hard-neg violations per batch (0 = disabled)
    hard_neg_top_k: int = 1,  # Top-k hard negatives for soft aggregation (1 = max only, 3 = recommended)
    hard_neg_temperature: float = 0.1  # Temperature for softmax weighting of top-k
) -> Dict[str, Any]:
    """
    Compute contrastive loss using explicit bundle structure with gap-based margins.

    Key insight: margins are relative to positive similarity, not absolute thresholds.
    Constraint: s_pos >= s_neg + margin (preference gap)

    This measures "meaning preference gap" - how much better the positive is than
    each negative for the same anchor. That's what actually represents sense separation.

    NEW ARCHITECTURE (v3):
    - contrastive_loss: Base hinge loss only (not blended with slack)
    - slack_penalty_loss: TENSOR for standalone use in total_loss (not diluted)
    - Uses softplus(margin - gap) for smooth gradients near boundary
    - Focuses on "most dangerous negative" per anchor (max similarity)

    Args:
        margin_easy: Target margin for easy negatives (default 0.05 = certification margin)
        margin_hard: Target margin for hard negatives (default 0.15 = certification margin)
        margin_boost: Multiplier for margins (1.0 = base margins, 1.3 = 30% harder margins).
        hard_neg_penalty_mult: Multiplier for hard-neg slack penalty (default 2.0).
            Hard negatives are the real boss fight - weight their slack loss more heavily.
        track_killers: Number of worst hard-neg violations to track per batch (default 0 = disabled).
            When > 0, returns 'killer_negatives' list with metadata for curriculum learning.

    Returns dict with:
    - 'contrastive_loss': Base hinge loss only (for backprop)
    - 'slack_penalty_loss': Standalone slack loss TENSOR (for direct use in total_loss)
    - 'slack_penalty_loss_value': Float version for logging only
    - 'positive_similarity': Avg similarity for positive pairs
    - 'negative_similarity': Avg similarity for negative pairs
    - 'hard_negative_similarity': Avg similarity for hard negatives
    - 'gap_easy': Mean (s_pos - s_neg) for easy negatives
    - 'gap_hard': Mean (s_pos - s_hard) for hard negatives
    - 'msr_easy': Margin satisfaction rate for easy negatives
    - 'msr_hard': Margin satisfaction rate for hard negatives
    - 'msr_total': Overall margin satisfaction rate
    - 'avg_slack_easy': Average slack (how far inside safe zone) for easy
    - 'avg_slack_hard': Average slack for hard negatives
    - 'msr_by_difficulty': Dict with MSR/slack/n for easy/med/hard difficulty buckets
    - 'killer_negatives': (if track_killers > 0) List of K worst hard-neg violations:
        [{'anchor_id': str, 'anchor_idx': int, 'hard_neg_id': str, 'hard_neg_idx': int,
          'bundle_id': str, 'sim_pos': float, 'sim_neg': float, 'gap': float, 'slack': float}, ...]
    """
    B = embeddings.size(0)

    # CRITICAL: Normalize embeddings so similarity scale is interpretable
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    # Compute all pairwise similarities (cosine, since normalized)
    sim_matrix = torch.mm(embeddings, embeddings.t())  # [B, B]

    # Step 1: Build per-anchor positive similarity (O(B))
    # Key insight: use anchor_idx directly, not bundle_id
    pos_sim_by_anchor = {}  # anchor_idx (int) -> s_pos (tensor)
    for i in range(B):
        if batch.item_roles[i] == "positive":
            a = batch.anchor_indices[i]
            if a >= 0:
                pos_sim_by_anchor[a] = sim_matrix[i, a]  # sim(pos, anchor)

    # Apply margin boost (for margin schedule late in training)
    margin_easy_eff = margin_easy * margin_boost
    margin_hard_eff = margin_hard * margin_boost

    # Step 2: Build per-anchor "most dangerous negative" (highest similarity)
    # This is the negative that actually threatens the margin - the one certification cares about
    easy_neg_max_by_anchor = {}   # anchor_idx -> max sim tensor
    hard_neg_max_by_anchor = {}   # anchor_idx -> max sim tensor (or top-k aggregated)
    hard_neg_max_idx_by_anchor = {}  # anchor_idx -> item batch index of most dangerous hard neg
    hard_neg_all_by_anchor = {}  # anchor_idx -> list of (sim, idx) for top-k aggregation

    # Also collect ALL hard neg violations for killer tracking (if enabled)
    all_hard_neg_violations = []  # [(slack, anchor_idx, hard_neg_idx, gap, sim_pos, sim_neg), ...]

    for i in range(B):
        role = batch.item_roles[i]
        a = batch.anchor_indices[i]
        if role in ("anchor", "positive") or a < 0:
            continue
        if a not in pos_sim_by_anchor:
            continue

        s = sim_matrix[i, a]  # sim(item, anchor)

        if role == "negative":
            prev = easy_neg_max_by_anchor.get(a, None)
            easy_neg_max_by_anchor[a] = s if prev is None else torch.maximum(prev, s)
        elif role == "hard_negative":
            # Collect all hard negs for this anchor (for top-k aggregation)
            if a not in hard_neg_all_by_anchor:
                hard_neg_all_by_anchor[a] = []
            hard_neg_all_by_anchor[a].append((s, i))
            
            # Still track max for backward compatibility and killer logging
            prev = hard_neg_max_by_anchor.get(a, None)
            if prev is None or s.item() > prev.item():
                hard_neg_max_by_anchor[a] = s
                hard_neg_max_idx_by_anchor[a] = i
            else:
                hard_neg_max_by_anchor[a] = torch.maximum(prev, s)

            # Track all hard neg violations for killer logging
            if track_killers > 0:
                s_pos = pos_sim_by_anchor[a]
                gap = s_pos - s
                slack = gap - margin_hard * margin_boost  # Use effective margin
                all_hard_neg_violations.append((
                    slack.item(), a, i, gap.item(), s_pos.item(), s.item()
                ))

    # Top-k soft aggregation for hard negatives
    # Instead of max-only, use softmax-weighted average of top-k to reduce noise
    if hard_neg_top_k > 1:
        for a, hn_list in hard_neg_all_by_anchor.items():
            if len(hn_list) <= 1:
                continue  # Keep single value as-is
            
            # Sort by similarity (descending) and take top-k
            hn_list.sort(key=lambda x: x[0].item(), reverse=True)
            top_k_items = hn_list[:hard_neg_top_k]
            
            # Softmax-weighted aggregation
            sims = torch.stack([s for s, _ in top_k_items])
            weights = torch.softmax(sims / hard_neg_temperature, dim=0)
            aggregated_sim = (weights * sims).sum()
            
            # Replace max with aggregated value
            hard_neg_max_by_anchor[a] = aggregated_sim
            # Keep the max idx for killer logging
            hard_neg_max_idx_by_anchor[a] = top_k_items[0][1]

    # Step 3: Accumulators for metrics
    positive_sims = []
    negative_sims = []
    hard_negative_sims = []
    losses = []

    num_easy = num_hard = 0
    vio_easy = vio_hard = 0
    slack_easy_sum = torch.tensor(0.0, device=device)
    slack_hard_sum = torch.tensor(0.0, device=device)
    gap_easy_sum = torch.tensor(0.0, device=device)
    gap_hard_sum = torch.tensor(0.0, device=device)

    # Difficulty buckets
    buckets = {
        "easy": {"n": 0, "v": 0, "slack": 0.0},
        "med":  {"n": 0, "v": 0, "slack": 0.0},
        "hard": {"n": 0, "v": 0, "slack": 0.0},
    }

    # NON-NEGATIVE HINGE-STYLE CONTRASTIVE LOSS
    # Both positive and negative terms use ReLU so loss bottoms at 0 when margins satisfied.
    # This makes "lower is better" always true and weights behave sanely in multi-loss training.
    pos_floor = 0.90  # Positives should have sim >= this (tune 0.85-0.95)

    # Step 3: Process each item (O(B))
    for i in range(B):
        role = batch.item_roles[i]
        a = batch.anchor_indices[i]

        if role == "anchor" or a < 0:
            continue

        sim = sim_matrix[i, a]  # sim(item, anchor)

        if role == "positive":
            positive_sims.append(sim)
            # Penalize if sim < pos_floor (want positives close to anchor)
            loss = torch.relu(pos_floor - sim)
            losses.append(loss)
            continue

        # Negative or hard_negative: need positive for gap computation
        if a not in pos_sim_by_anchor:
            continue  # Skip if no positive for this anchor in batch

        s_pos = pos_sim_by_anchor[a]
        s_neg = sim
        gap = s_pos - s_neg  # How much better positive is

        # Use effective margins (with optional late-stage boost)
        margin = margin_hard_eff if role == "hard_negative" else margin_easy_eff
        slack = gap - margin  # >0 means satisfies margin
        violated = slack.item() < 0

        # Collect raw similarities for logging
        if role == "hard_negative":
            hard_negative_sims.append(sim)
            num_hard += 1
            gap_hard_sum = gap_hard_sum + gap
            slack_hard_sum = slack_hard_sum + slack
            if violated:
                vio_hard += 1
        else:
            negative_sims.append(sim)
            num_easy += 1
            gap_easy_sum = gap_easy_sum + gap
            slack_easy_sum = slack_easy_sum + slack
            if violated:
                vio_easy += 1

        # Difficulty bucket tracking
        bname = _bucket3(float(batch.difficulties[i]))
        buckets[bname]["n"] += 1
        buckets[bname]["v"] += int(violated)
        buckets[bname]["slack"] += float(slack.item())

        # Penalize if sim > neg_ceiling (want negatives far from anchor)
        # neg_ceiling = 1.0 - margin ensures we push negatives to at least margin below positives
        neg_ceiling = 1.0 - margin
        loss = torch.relu(sim - neg_ceiling)
        losses.append(loss)

    # Step 4: Compile results
    def rate(v, n):
        return 0.0 if n == 0 else (1.0 - (v / n))

    result = {}

    # Main contrastive loss (base hinge only - NOT blended with slack)
    if losses:
        base_loss = torch.stack(losses).mean()
    else:
        base_loss = embeddings.sum() * 0.0  # Maintain graph

    result['contrastive_loss'] = base_loss  # Pure contrastive, no slack blended in

    # ==========================================
    # NEW v3: Standalone slack loss using softplus + most-dangerous-negative
    # ==========================================
    # softplus(margin - gap) gives smooth gradients even near boundary
    # Focus on most dangerous negative per anchor (the one that threatens margin)

    slack_losses_easy = []
    slack_losses_hard = []

    for a, s_pos in pos_sim_by_anchor.items():
        # Easy negatives: penalize if most dangerous is too close
        if a in easy_neg_max_by_anchor:
            s_neg_max = easy_neg_max_by_anchor[a]
            gap = s_pos - s_neg_max
            # softplus(margin - gap) = soft barrier pushing gap above margin
            slack_losses_easy.append(F.softplus(margin_easy_eff - gap))

        # Hard negatives: penalize more heavily (the real boss fight)
        if a in hard_neg_max_by_anchor:
            s_neg_max = hard_neg_max_by_anchor[a]
            gap = s_pos - s_neg_max
            slack_losses_hard.append(F.softplus(margin_hard_eff - gap))

    # Aggregate slack losses (keep as tensors for backprop)
    if slack_losses_easy:
        slack_loss_easy = torch.stack(slack_losses_easy).mean()
    else:
        slack_loss_easy = embeddings.sum() * 0.0

    if slack_losses_hard:
        slack_loss_hard = torch.stack(slack_losses_hard).mean()
    else:
        slack_loss_hard = embeddings.sum() * 0.0

    # Combined slack loss with hard-neg focus (hard negatives weighted more heavily)
    # NORMALIZE by (1 + hard_neg_mult) to keep slack magnitude comparable to CE
    # This preserves the "hard matters 2x" intent while preventing slack from dwarfing CE
    slack_penalty_loss = (slack_loss_easy + hard_neg_penalty_mult * slack_loss_hard) / (1.0 + hard_neg_penalty_mult)

    # Return TENSOR for standalone use in total_loss (not diluted by contrastive_weight!)
    result['slack_penalty_loss'] = slack_penalty_loss  # TENSOR for training
    result['slack_penalty_loss_value'] = slack_penalty_loss.item()  # Float for logging
    result['slack_loss_easy'] = slack_loss_easy.item()
    result['slack_loss_hard'] = slack_loss_hard.item()

    # Similarity stats
    if positive_sims:
        result['positive_similarity'] = torch.stack(positive_sims).mean().item()
    if negative_sims:
        result['negative_similarity'] = torch.stack(negative_sims).mean().item()
    if hard_negative_sims:
        result['hard_negative_similarity'] = torch.stack(hard_negative_sims).mean().item()

    # Gap metrics (the key diagnostic - log both rate AND gap)
    result['gap_easy'] = (gap_easy_sum / max(1, num_easy)).item()
    result['gap_hard'] = (gap_hard_sum / max(1, num_hard)).item()

    # Margin Satisfaction Rate (MSR)
    result['msr_easy'] = rate(vio_easy, num_easy)
    result['msr_hard'] = rate(vio_hard, num_hard)
    result['msr_total'] = rate(vio_easy + vio_hard, num_easy + num_hard)

    # Average slack (how far inside safe zone - key robustness indicator)
    result['avg_slack_easy'] = (slack_easy_sum / max(1, num_easy)).item()
    result['avg_slack_hard'] = (slack_hard_sum / max(1, num_hard)).item()

    # MSR by difficulty bucket (for curriculum monitoring)
    result['msr_by_difficulty'] = {}
    for k, v in buckets.items():
        n = v["n"]
        result['msr_by_difficulty'][k] = {
            "msr": 0.0 if n == 0 else (1.0 - v["v"] / n),
            "avg_slack": 0.0 if n == 0 else (v["slack"] / n),
            "n": n
        }

    result['num_pairs'] = len(losses)
    result['num_easy'] = num_easy
    result['num_hard'] = num_hard

    # Killer negative tracking (if enabled)
    # Sort by slack ascending (worst = most negative slack first) and return K worst
    if track_killers > 0 and all_hard_neg_violations:
        # Sort by slack (ascending = worst violations first)
        sorted_violations = sorted(all_hard_neg_violations, key=lambda x: x[0])
        killers = []
        for slack, anchor_idx, hard_neg_idx, gap, sim_pos, sim_neg in sorted_violations[:track_killers]:
            killers.append({
                'anchor_id': batch.item_ids[anchor_idx] if anchor_idx < len(batch.item_ids) else f"idx_{anchor_idx}",
                'anchor_idx': anchor_idx,
                'hard_neg_id': batch.item_ids[hard_neg_idx] if hard_neg_idx < len(batch.item_ids) else f"idx_{hard_neg_idx}",
                'hard_neg_idx': hard_neg_idx,
                'bundle_id': batch.bundle_ids[anchor_idx] if anchor_idx < len(batch.bundle_ids) else "unknown",
                'anchor_sense': batch.sense_ids[anchor_idx] if anchor_idx < len(batch.sense_ids) else "unknown",
                'hard_neg_sense': batch.sense_ids[hard_neg_idx] if hard_neg_idx < len(batch.sense_ids) else "unknown",
                'sim_pos': sim_pos,
                'sim_neg': sim_neg,
                'gap': gap,
                'slack': slack,
            })
        result['killer_negatives'] = killers

    return result


def create_bundle_dataloader(
    bundle_path: Path,
    batch_size: int = 8,
    bundles_per_batch: int = 2,
    max_length: int = 128,
    vocab_size: int = 1000,
    seed: int = 42
) -> DataLoader:
    """Create a DataLoader for v2 bundles."""
    dataset = BundleDataset(
        bundle_path=bundle_path,
        max_length=max_length,
        vocab_size=vocab_size
    )

    sampler = BundleAwareSampler(
        dataset=dataset,
        batch_size=batch_size,
        bundles_per_batch=bundles_per_batch,
        seed=seed
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,  # Use batch_sampler since sampler yields batches
        collate_fn=bundle_collator
    )


if __name__ == "__main__":
    # Demo usage
    from paradigm_factory.polysemy_bundle_v2 import create_bundle, save_bundles

    # Create sample bundles
    bundles = [
        create_bundle(
            word="present",
            anchor_sense_id="present#gift",
            anchor_context="She wrapped the birthday present carefully.",
            positive_contexts=[
                ("present#gift", "The present was hidden under the tree."),
            ],
            negative_contexts=[
                ("present#now", "At present, sales are rising faster than expected."),
            ],
            hard_negative_contexts=[
                ("present#verb", "The report will present the findings at noon.", "POS shift"),
            ],
            bundle_index=0,
            seed=42
        ),
        create_bundle(
            word="bank",
            anchor_sense_id="bank#financial",
            anchor_context="I deposited my paycheck at the bank this morning.",
            positive_contexts=[
                ("bank#financial", "The bank approved my loan application yesterday."),
            ],
            negative_contexts=[
                ("bank#river", "We sat on the grassy bank watching the river flow."),
            ],
            hard_negative_contexts=[
                ("bank#river", "The bank was located right by the waterfront downtown.", "Location ambiguity"),
            ],
            bundle_index=1,
            seed=42
        )
    ]

    # Save test bundles
    test_path = Path("paradigm_factory/output/test_bundles_v2.jsonl")
    save_bundles(bundles, test_path)

    # Load and test dataloader
    dataloader = create_bundle_dataloader(test_path, batch_size=4)

    print("\n=== Sample Batch Structure ===")
    for batch in dataloader:
        print(f"Batch size: {batch.input_ids.size(0)}")
        print(f"\nPer-item structure:")
        for i in range(len(batch.bundle_ids)):
            print(f"  [{i}] bundle={batch.bundle_ids[i][-12:]}, role={batch.item_roles[i]:15s}, "
                  f"anchor_idx={batch.anchor_indices[i]:2d}, same_sense={batch.same_sense_as_anchor[i]}, "
                  f"difficulty={batch.difficulties[i]:.2f}")

        # Simulate embeddings (with some structure to test metrics)
        fake_embeddings = torch.randn(batch.input_ids.size(0), 128)
        loss_dict = compute_bundle_contrastive_loss(fake_embeddings, batch, device='cpu')

        print(f"\n=== Gap-Based Metrics (random embeddings) ===")
        print(f"Contrastive loss: {loss_dict['contrastive_loss']:.4f}")
        print(f"Positive similarity: {loss_dict.get('positive_similarity', 0):.4f}")
        print(f"Negative similarity: {loss_dict.get('negative_similarity', 0):.4f}")
        print(f"Hard neg similarity: {loss_dict.get('hard_negative_similarity', 0):.4f}")
        print(f"\nGap metrics:")
        print(f"  gap_easy: {loss_dict.get('gap_easy', 0):.4f}")
        print(f"  gap_hard: {loss_dict.get('gap_hard', 0):.4f}")
        print(f"\nMargin Satisfaction Rate (MSR):")
        print(f"  MSR_easy: {loss_dict.get('msr_easy', 0):.2%}")
        print(f"  MSR_hard: {loss_dict.get('msr_hard', 0):.2%}")
        print(f"  MSR_total: {loss_dict.get('msr_total', 0):.2%}")
        print(f"\nAverage slack:")
        print(f"  avg_slack_easy: {loss_dict.get('avg_slack_easy', 0):.4f}")
        print(f"  avg_slack_hard: {loss_dict.get('avg_slack_hard', 0):.4f}")
        print(f"\nMSR by difficulty bucket: {loss_dict.get('msr_by_difficulty', {})}")
        print(f"\nPair counts: {loss_dict['num_easy']} easy, {loss_dict['num_hard']} hard")
        break
