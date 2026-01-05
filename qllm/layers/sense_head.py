"""
SenseHead - Attentive Pooling for Sense Disambiguation
=======================================================

Addresses the representation collapse problem where all embeddings
converge to high similarity (~0.9), destroying discriminative gaps.

Key insight: Standard pooling (CLS/mean) collapses toward a "general 
meaning" manifold. Attentive pooling lets the model learn to attend
to disambiguating tokens (prepositions, domain keywords, syntactic frame).

CE continues in main embedding space, while slack/contrastive operates
in a dedicated "sense space" that can focus on context-bearing tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SenseHead(nn.Module):
    """
    Attentive pooling head for sense embeddings.
    
    Instead of fixed pooling (CLS/mean), learns which tokens matter for
    each sense distinction. Returns normalized sense embeddings for
    contrastive/slack loss.
    
    Architecture:
    1. Attention scorer: token -> scalar score
    2. Softmax over tokens (masked for padding)
    3. Weighted sum of token states
    4. MLP + LayerNorm projection
    5. L2 normalization
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        proj_dim: int = 256, 
        dropout: float = 0.1,
        target_token_penalty: float = 0.0
    ):
        """
        Args:
            hidden_dim: Input dimension from encoder
            proj_dim: Output sense embedding dimension
            dropout: Dropout rate in MLP
            target_token_penalty: If > 0, discourages attending only to target token
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.target_token_penalty = target_token_penalty
        
        # Token attention scorer
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Projection MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )
        
        self.ln = nn.LayerNorm(proj_dim)
        
    def forward(
        self, 
        token_states: torch.Tensor, 
        attention_mask: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sense embeddings via attentive pooling.
        
        Args:
            token_states: [B, T, H] token-level hidden states
            attention_mask: [B, T] with 1 for real tokens, 0 for padding
            target_positions: [B] position of target word (optional, for penalty)
            
        Returns:
            sense_embeddings: [B, D] normalized sense vectors
            attention_weights: [B, T] for debugging/visualization
        """
        B, T, H = token_states.shape
        
        # Compute attention scores
        scores = self.attn(token_states).squeeze(-1)  # [B, T]
        
        # Mask padding tokens
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Optional: penalize attending too much to target token
        if self.target_token_penalty > 0 and target_positions is not None:
            # Create penalty mask for target positions
            target_mask = torch.zeros_like(scores)
            for i, pos in enumerate(target_positions):
                if pos >= 0 and pos < T:
                    target_mask[i, pos] = self.target_token_penalty
            scores = scores - target_mask
        
        # Softmax attention
        weights = F.softmax(scores, dim=-1)  # [B, T]
        
        # Weighted sum pooling
        pooled = (weights.unsqueeze(-1) * token_states).sum(dim=1)  # [B, H]
        
        # Project and normalize
        z = self.mlp(pooled)  # [B, proj_dim]
        z = self.ln(z)
        z = F.normalize(z, dim=-1)
        
        return z, weights


class SenseHeadWithEntropy(SenseHead):
    """
    SenseHead variant that encourages attention entropy.
    
    Prevents degenerate attention patterns (attending to single token).
    Adds regularization term to encourage spread across context.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        proj_dim: int = 256, 
        dropout: float = 0.1,
        target_token_penalty: float = 0.0,
        entropy_weight: float = 0.1,
        min_entropy: float = 1.0
    ):
        super().__init__(hidden_dim, proj_dim, dropout, target_token_penalty)
        self.entropy_weight = entropy_weight
        self.min_entropy = min_entropy
        
    def forward(
        self, 
        token_states: torch.Tensor, 
        attention_mask: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            sense_embeddings: [B, D]
            attention_weights: [B, T]
            entropy_loss: scalar regularization term
        """
        z, weights = super().forward(token_states, attention_mask, target_positions)
        
        # Compute attention entropy
        # Higher entropy = more spread out attention
        eps = 1e-8
        entropy = -(weights * (weights + eps).log()).sum(dim=-1)  # [B]
        
        # Regularize to maintain minimum entropy
        entropy_loss = F.relu(self.min_entropy - entropy).mean() * self.entropy_weight
        
        return z, weights, entropy_loss


def log_attention_patterns(
    weights: torch.Tensor,
    tokens: list,
    item_ids: list,
    step: int,
    log_path: str,
    top_k: int = 5
):
    """
    Log attention patterns for debugging.
    
    Shows which tokens the SenseHead attends to for each item.
    Useful for verifying that attention focuses on disambiguating context.
    """
    import json
    
    B = weights.shape[0]
    entries = []
    
    for i in range(min(B, 8)):  # Log first 8 items per batch
        w = weights[i].detach().cpu().numpy()
        top_indices = w.argsort()[-top_k:][::-1]
        
        pattern = {
            "step": step,
            "item_id": item_ids[i] if i < len(item_ids) else f"item_{i}",
            "top_tokens": [
                {"idx": int(idx), "token": tokens[i][idx] if idx < len(tokens[i]) else "<unk>", "weight": float(w[idx])}
                for idx in top_indices
            ],
            "entropy": float(-sum(w[j] * (w[j] + 1e-8) for j in range(len(w)) if w[j] > 0))
        }
        entries.append(pattern)
    
    with open(log_path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
