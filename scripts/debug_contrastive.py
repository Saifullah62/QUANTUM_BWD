#!/usr/bin/env python3
"""Debug contrastive loss computation."""

import sys
import os
import torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.bundle_dataset import create_bundle_dataloader, compute_bundle_contrastive_loss
from qllm.core.config import QLLMConfig
from qllm.core.model import QLLM
from pathlib import Path

def debug():
    # Load bundles
    bundle_path = Path("paradigm_factory/output/splits/train_bundles.jsonl")
    dataloader = create_bundle_dataloader(bundle_path, batch_size=16, max_length=128, vocab_size=1000)

    # Create model
    model_config = QLLMConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        intermediate_dim=256,
        max_seq_length=128,
        use_semantic_phase=True,
        semantic_phase_dim=128,
        use_emergent_init=True
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = QLLM(model_config).to(device)

    # Get a batch
    batch = next(iter(dataloader))
    print(f"Batch size: {batch.input_ids.size(0)}")
    print(f"Bundle IDs: {set(batch.bundle_ids)}")
    print(f"Roles: {batch.item_roles}")
    print(f"Same sense as anchor: {batch.same_sense_as_anchor}")
    print(f"Anchor indices: {batch.anchor_indices}")

    # Forward pass
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    print(f"\nModel outputs keys: {outputs.keys()}")

    # Get embeddings the same way as trainer (FIRST TOKEN pooling)
    if 'phase_states' in outputs and outputs['phase_states'] is not None:
        phase = outputs['phase_states']
        print(f"Using phase_states, shape: {phase.shape}")
        if phase.is_complex():
            embeddings = phase[:, 0, :].abs()  # First token
        else:
            embeddings = phase[:, 0, :]  # First token
        print(f"Using FIRST TOKEN pooling (CLS-style)")
    elif 'real_embedding' in outputs and outputs['real_embedding'] is not None:
        real = outputs['real_embedding'][:, 0, :]
        imag = outputs.get('imag_embedding', torch.zeros_like(outputs['real_embedding']))[:, 0, :]
        print(f"Real embedding shape: {real.shape}")
        embeddings = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        print(f"Using semantic phase embeddings (complex magnitude, first token)")
    elif 'hidden_states' in outputs and outputs['hidden_states'] is not None:
        hidden = outputs['hidden_states']
        embeddings = hidden[:, 0, :]  # First token
        print(f"Using hidden states (first token), shape: {hidden.shape}")
    else:
        print("WARNING: No usable embeddings!")
        return

    print(f"Pooled embeddings shape: {embeddings.shape}")
    print(f"Embedding norms (first 5): {embeddings.norm(dim=-1)[:5]}")

    # Normalize
    embeddings_norm = torch.nn.functional.normalize(embeddings, dim=-1)

    # Compute similarity matrix
    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    print(f"\nSimilarity matrix shape: {sim_matrix.shape}")
    print(f"Similarity range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")
    print(f"Diagonal (self-sim): {sim_matrix.diag()[:5]}")

    # Find anchor and its positive/negative
    anchor_idx = None
    positive_idx = None
    negative_idx = None

    for i, role in enumerate(batch.item_roles):
        if role == "anchor":
            anchor_idx = i
        elif role == "positive" and batch.anchor_indices[i] == anchor_idx:
            positive_idx = i
        elif role == "negative" and batch.anchor_indices[i] == anchor_idx:
            negative_idx = i

        if anchor_idx is not None and positive_idx is not None and negative_idx is not None:
            break

    if anchor_idx is not None:
        print(f"\nFound anchor at idx {anchor_idx}")
        if positive_idx is not None:
            s_pos = sim_matrix[positive_idx, anchor_idx].item()
            print(f"  Positive (idx {positive_idx}) similarity to anchor: {s_pos:.4f}")
        if negative_idx is not None:
            s_neg = sim_matrix[negative_idx, anchor_idx].item()
            print(f"  Negative (idx {negative_idx}) similarity to anchor: {s_neg:.4f}")
        if positive_idx and negative_idx:
            gap = s_pos - s_neg
            print(f"  Gap (s_pos - s_neg): {gap:.4f}")
            print(f"  Required margin (easy): 0.15")
            print(f"  Slack: {gap - 0.15:.4f}")

    # Compute full contrastive loss
    print("\n" + "="*50)
    print("Full contrastive loss computation:")
    metrics = compute_bundle_contrastive_loss(embeddings, batch, device=device)
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    debug()
