"""
Semantic Phase Embedding Layer (v2 - Refined)
==============================================

From Daugherty's Semantic Quantum Computing paradigm:
"Quantum phase encodes not just probabilistic amplitude, but meaning and context"

REFINED IMPLEMENTATION:
- Two coupled real tensors (real/imag channels) - no complex dtypes
- Phase as learnable rotation conditioned on context (generalized RoPE for meaning)
- WIN CONDITION: Polysemy improves when magnitude is constrained

Key insight: Ambiguity becomes representable as equal-magnitude states
whose interference changes with context. When magnitude is normalized,
only phase (context) determines meaning differentiation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SemanticPhaseEmbedding(nn.Module):
    """
    Complex-valued embedding using coupled real/imaginary channels.

    Each token is represented as two coupled real tensors:
    - real_embed: semantic content (magnitude-like)
    - imag_embed: phase-like component

    Context-conditioned rotation allows same word to have different
    "phases" in different contexts - this is key for polysemy handling.

    This is like generalized RoPE, but for semantic state rather than position.
    The rotation is conditioned on LOCAL CONTEXT, not just position.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        phase_dim: Optional[int] = None,
        padding_idx: Optional[int] = None,
        phase_modulation_strength: float = 0.3,
        context_window: int = 128
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.phase_dim = phase_dim or embedding_dim
        self.phase_modulation_strength = phase_modulation_strength
        self.context_window = context_window

        # Coupled real/imaginary embeddings (two real tensors)
        self.real_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.imag_embedding = nn.Embedding(
            vocab_size, self.phase_dim, padding_idx=padding_idx
        )

        # Context network: produces per-token rotation angle
        # This is the "semantic gauge field"
        self.context_to_rotation = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim // 2)  # One angle per pair
        )

        # Learnable base rotation frequencies (like RoPE but semantic)
        self.rotation_freqs = nn.Parameter(torch.randn(embedding_dim // 2) * 0.01)

        # Positional rotation phases (sinusoidal baseline)
        self.register_buffer(
            'positional_phases',
            self._create_positional_phases(2048, embedding_dim)
        )

        # Initialize imag embeddings smaller
        nn.init.normal_(self.imag_embedding.weight, mean=0, std=0.02)

    def _create_positional_phases(
        self,
        max_seq_length: int,
        embed_dim: int
    ) -> torch.Tensor:
        """Create positional phase offsets (RoPE-style frequencies)"""
        position = torch.arange(max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim // 2).float() * (-math.log(10000.0) / (embed_dim // 2))
        )
        phases = position * div_term  # [seq, dim//2]
        return phases

    def compute_context_vector(
        self,
        real_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute context vector that determines rotation angles.

        Uses causal aggregation (only previous tokens).
        This is the "semantic gauge field" from Daugherty's paper.
        """
        batch_size, seq_len, dim = real_embeddings.shape

        # Exponential decay weights for temporal locality
        positions = torch.arange(seq_len, device=real_embeddings.device)
        decay = torch.exp(-0.1 * positions.float())
        weights = decay.view(1, -1, 1)

        if attention_mask is not None:
            weights = weights * attention_mask.unsqueeze(-1)

        # Cumulative weighted average (causal)
        cumsum = torch.cumsum(real_embeddings * weights, dim=1)
        count = torch.cumsum(weights.expand(batch_size, -1, 1), dim=1).clamp(min=1e-6)
        context = cumsum / count

        return context

    def apply_rotation(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D rotation to (real, imag) pairs.

        This is the core operation: rotation in semantic space.
        real' = real * cos(θ) - imag * sin(θ)
        imag' = real * sin(θ) + imag * cos(θ)
        """
        # Reshape for paired rotation: [batch, seq, dim] -> [batch, seq, dim//2, 2]
        d = real.shape[-1]
        real_pairs = real.view(*real.shape[:-1], d // 2, 2)
        imag_pairs = imag.view(*imag.shape[:-1], d // 2, 2)

        cos_theta = torch.cos(theta).unsqueeze(-1)  # [batch, seq, dim//2, 1]
        sin_theta = torch.sin(theta).unsqueeze(-1)

        # Apply rotation
        real_rot = real_pairs * cos_theta - imag_pairs * sin_theta
        imag_rot = real_pairs * sin_theta + imag_pairs * cos_theta

        # Reshape back
        return real_rot.view(*real.shape), imag_rot.view(*imag.shape)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_complex: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing rotated embeddings.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask
            return_complex: If True, return as complex tensor

        Returns:
            real_part: [batch, seq_len, dim]
            imag_part: [batch, seq_len, dim]
        """
        batch_size, seq_len = input_ids.shape

        # Get base embeddings
        real = self.real_embedding(input_ids)
        imag = self.imag_embedding(input_ids)

        # Ensure same dimension
        if self.phase_dim != self.embedding_dim:
            imag = F.pad(imag, (0, self.embedding_dim - self.phase_dim))

        # 1. Positional rotation (like RoPE baseline)
        pos_theta = self.positional_phases[:seq_len]  # [seq, dim//2]
        pos_theta = pos_theta.unsqueeze(0).expand(batch_size, -1, -1)
        real, imag = self.apply_rotation(real, imag, pos_theta)

        # 2. Context-conditioned rotation (the key innovation)
        context = self.compute_context_vector(real, attention_mask)
        context_theta = self.context_to_rotation(context)

        # Modulate by learnable frequencies
        context_theta = context_theta * torch.sigmoid(self.rotation_freqs) * self.phase_modulation_strength
        real, imag = self.apply_rotation(real, imag, context_theta)

        if return_complex:
            return torch.complex(real, imag)

        return real, imag

    def get_magnitude(
        self,
        real: torch.Tensor,
        imag: torch.Tensor
    ) -> torch.Tensor:
        """Get magnitude (semantic content strength)"""
        return torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

    def get_phase(
        self,
        real: torch.Tensor,
        imag: torch.Tensor
    ) -> torch.Tensor:
        """Get phase angle (contextual meaning)"""
        return torch.atan2(imag, real + 1e-8)

    def magnitude_constrained_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with unit magnitude normalization.

        WIN CONDITION TEST: When magnitude is constrained to 1,
        only phase (context) determines meaning differentiation.
        Polysemy tasks should improve in this mode.
        """
        real, imag = self.forward(input_ids, attention_mask)

        # Normalize to unit magnitude
        magnitude = self.get_magnitude(real, imag)
        real = real / magnitude
        imag = imag / magnitude

        return real, imag

    def get_phase_coherence(
        self,
        embeddings: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute phase coherence across sequence.

        High coherence = consistent phase = resolved meaning
        Low coherence = phase variance = ambiguous meaning
        """
        real, imag = embeddings

        # Mean complex value
        mean_real = real.mean(dim=1)
        mean_imag = imag.mean(dim=1)

        # Coherence = |mean| / mean(|.|)
        mean_magnitude = torch.sqrt(mean_real ** 2 + mean_imag ** 2 + 1e-8)
        magnitudes = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        mean_of_magnitudes = magnitudes.mean(dim=1)

        coherence = mean_magnitude / (mean_of_magnitudes + 1e-8)
        return coherence.mean(dim=-1)  # Average across dimensions


class PhaseModulator(nn.Module):
    """
    Dynamic phase modulation layer.

    Implements attention-based phase adjustment where tokens
    influence each other's semantic phase based on relevance.
    This is the "measurement-like" resolution of meaning.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Input normalization - critical for non-uniform attention
        # Without this, small embedding values lead to near-zero attention scores
        # which causes uniform attention and zero gradients for Q/K
        self.real_norm = nn.LayerNorm(dim)
        self.imag_norm = nn.LayerNorm(dim)

        # Phase attention projections
        self.phase_query = nn.Linear(dim, dim)
        self.phase_key = nn.Linear(dim, dim)
        self.phase_value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        # Learnable collapse strength
        self.collapse_strength = nn.Parameter(torch.tensor(0.5))

        # Initialize to produce good attention score variance
        self._init_weights()

    def _init_weights(self):
        """Initialize attention weights to produce non-uniform attention.

        The problem: small embeddings (~0.08 range) + default init = near-zero scores
                   = uniform attention = zero gradient for Q/K = no learning.

        Solution: Scale Q/K with large gain to compensate for small inputs.
        Target: attention scores should have std ~1 for proper softmax differentiation.
        """
        # With LayerNorm, inputs have std ~1. Target score std ~1 for good softmax.
        # Observed: gain=4 -> score_std=15 (too high, one-hot attention)
        # Score std scales as gain^2, so gain=1 -> score_std = 15/16 ~1
        gain = 1.0

        nn.init.xavier_uniform_(self.phase_query.weight, gain=gain)
        nn.init.xavier_uniform_(self.phase_key.weight, gain=gain)
        nn.init.zeros_(self.phase_query.bias)
        nn.init.zeros_(self.phase_key.bias)

        # V and out_proj use standard init
        nn.init.xavier_uniform_(self.phase_value.weight)
        nn.init.zeros_(self.phase_value.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Mark all linear layers to skip parent class initialization
        for module in [self.phase_query, self.phase_key, self.phase_value, self.out_proj]:
            module._skip_init = True

    def forward(
        self,
        real_embed: torch.Tensor,
        imag_embed: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modulate phases through attention-based interaction.
        """
        batch_size, seq_len, _ = real_embed.shape

        # Normalize inputs to unit variance - critical for non-trivial attention
        real_normed = self.real_norm(real_embed)
        imag_normed = self.imag_norm(imag_embed)

        # Use normalized real embeddings for Q/K, normalized imag for V
        Q = self.phase_query(real_normed).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.phase_key(real_normed).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.phase_value(imag_normed).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~padding_mask.bool(), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Phase update through attention
        phase_update = torch.matmul(attn_weights, V)
        phase_update = phase_update.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        phase_update = self.out_proj(phase_update)

        # Interpolate between original and updated
        collapse = torch.sigmoid(self.collapse_strength)
        modulated_imag = (1 - collapse) * imag_embed + collapse * phase_update

        # Real is slightly modulated by phase magnitude
        phase_mag = torch.norm(modulated_imag, dim=-1, keepdim=True)
        modulated_real = real_embed * (1 + 0.05 * torch.tanh(phase_mag))

        return modulated_real, modulated_imag


class ContrastivePhaseObjective(nn.Module):
    """
    Contrastive objective for semantic phase that prevents trivial collapse.

    The polysemy thesis: "be consistent within a basin, but separable across basins"

    This objective:
    - Rewards phase ALIGNMENT for same-sense contexts (positive pairs)
    - Rewards phase SEPARATION for different-sense contexts (negative pairs)
    - Prevents the trivial solution of collapsing all phases to zero

    Without this, coherence can be "won" by making all phases identical,
    which defeats the purpose of having phase encode meaning.
    """

    def __init__(
        self,
        embedding_dim: int,
        temperature: float = 0.1,
        margin: float = 1.0,
        separation_weight: float = 1.0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.margin = margin  # Minimum phase distance for different senses
        self.separation_weight = separation_weight

    def compute_phase_distance(
        self,
        phase1: torch.Tensor,
        phase2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute angular distance between phases.

        Phase distance is circular: dist(0, 2π) = 0
        Returns value in [0, π]
        """
        # Normalize phases to [-π, π]
        diff = phase1 - phase2
        # Wrap to [-π, π]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        return diff.abs()

    def compute_phase_similarity(
        self,
        real1: torch.Tensor,
        imag1: torch.Tensor,
        real2: torch.Tensor,
        imag2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phase-based similarity using complex inner product.

        Two states with aligned phases have high similarity.
        Two states with orthogonal phases have zero similarity.
        """
        # Treat as complex: z1 = real1 + i*imag1, z2 = real2 + i*imag2
        # Similarity = Re(z1 · conj(z2)) / (|z1| |z2|)
        # = (real1*real2 + imag1*imag2) / (|z1| |z2|)

        dot_real = (real1 * real2 + imag1 * imag2).sum(dim=-1)

        mag1 = torch.sqrt((real1**2 + imag1**2).sum(dim=-1) + 1e-8)
        mag2 = torch.sqrt((real2**2 + imag2**2).sum(dim=-1) + 1e-8)

        similarity = dot_real / (mag1 * mag2 + 1e-8)
        return similarity  # Range [-1, 1]

    def contrastive_loss(
        self,
        anchor_real: torch.Tensor,
        anchor_imag: torch.Tensor,
        positive_real: torch.Tensor,
        positive_imag: torch.Tensor,
        negative_real: torch.Tensor,
        negative_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss for phase differentiation.

        Args:
            anchor_*: The reference embedding (e.g., "bank" in context A)
            positive_*: Same-sense embedding (e.g., "bank" in similar financial context)
            negative_*: Different-sense embedding (e.g., "bank" in river context)

        Returns:
            loss: Scalar loss value
        """
        # Positive similarity (should be HIGH)
        pos_sim = self.compute_phase_similarity(
            anchor_real, anchor_imag, positive_real, positive_imag
        )

        # Negative similarity (should be LOW)
        neg_sim = self.compute_phase_similarity(
            anchor_real, anchor_imag, negative_real, negative_imag
        )

        # InfoNCE-style contrastive loss
        # We want: pos_sim >> neg_sim
        logits = torch.stack([pos_sim, neg_sim], dim=-1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        contrastive_loss = F.cross_entropy(logits, labels)

        # Additional margin loss to ensure separation
        # Penalize if negative similarity is too high
        separation_loss = F.relu(neg_sim - (-self.margin)).mean()

        return contrastive_loss + self.separation_weight * separation_loss

    def in_batch_contrastive_loss(
        self,
        real_embeds: torch.Tensor,
        imag_embeds: torch.Tensor,
        token_ids: torch.Tensor,
        context_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute in-batch contrastive loss using token co-occurrence.

        For each token that appears multiple times in the batch:
        - Occurrences with similar context = positive pairs
        - Occurrences with different context = negative pairs

        This works even without explicit sense labels by using
        local context as a proxy for sense.

        Args:
            real_embeds: [batch, seq, dim] real part of embeddings
            imag_embeds: [batch, seq, dim] imaginary part
            token_ids: [batch, seq] token IDs
            context_ids: [batch, seq] optional context cluster IDs

        Returns:
            loss: Contrastive loss encouraging phase differentiation
        """
        batch_size, seq_len, dim = real_embeds.shape
        device = real_embeds.device

        # Flatten for easier processing
        flat_real = real_embeds.view(-1, dim)  # [batch*seq, dim]
        flat_imag = imag_embeds.view(-1, dim)
        flat_tokens = token_ids.view(-1)  # [batch*seq]

        # Find tokens that appear multiple times
        unique_tokens, inverse, counts = torch.unique(
            flat_tokens, return_inverse=True, return_counts=True
        )

        # Only consider tokens appearing 2+ times
        repeated_mask = counts[inverse] >= 2
        if not repeated_mask.any():
            return torch.tensor(0.0, device=device)

        # For efficiency, sample a subset of repeated tokens
        repeated_indices = torch.where(repeated_mask)[0]
        if len(repeated_indices) > 256:
            perm = torch.randperm(len(repeated_indices), device=device)[:256]
            repeated_indices = repeated_indices[perm]

        total_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        # For each repeated token occurrence, create pairs
        for idx in repeated_indices:
            token = flat_tokens[idx]

            # Find all occurrences of this token
            same_token_mask = flat_tokens == token
            same_token_indices = torch.where(same_token_mask)[0]

            if len(same_token_indices) < 2:
                continue

            # Anchor is current occurrence
            anchor_real = flat_real[idx:idx+1]
            anchor_imag = flat_imag[idx:idx+1]

            # Other occurrences
            other_indices = same_token_indices[same_token_indices != idx]

            # Without context labels, use embedding distance to infer same/different sense
            # Closer embeddings (before phase) = likely same sense
            # This is a bootstrap heuristic
            other_real = flat_real[other_indices]
            other_imag = flat_imag[other_indices]

            # Compute similarities to anchor
            sims = self.compute_phase_similarity(
                anchor_real.expand(len(other_indices), -1),
                anchor_imag.expand(len(other_indices), -1),
                other_real,
                other_imag
            )

            if len(sims) >= 2:
                # Most similar = positive, least similar = negative
                sorted_indices = torch.argsort(sims, descending=True)

                pos_idx = other_indices[sorted_indices[0]]
                neg_idx = other_indices[sorted_indices[-1]]

                # Always create pairs - let the loss learn to separate
                # (Removed 0.1 threshold that blocked pairs at initialization)
                pos_real = flat_real[pos_idx:pos_idx+1]
                pos_imag = flat_imag[pos_idx:pos_idx+1]
                neg_real = flat_real[neg_idx:neg_idx+1]
                neg_imag = flat_imag[neg_idx:neg_idx+1]

                loss = self.contrastive_loss(
                    anchor_real, anchor_imag,
                    pos_real, pos_imag,
                    neg_real, neg_imag
                )
                total_loss = total_loss + loss
                num_pairs += 1

        if num_pairs == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / num_pairs

    def anti_collapse_regularizer(
        self,
        real_embeds: torch.Tensor,
        imag_embeds: torch.Tensor,
        min_phase_variance: float = 0.1
    ) -> torch.Tensor:
        """
        Regularizer to prevent trivial phase collapse.

        Penalizes if phase variance across the batch is too low,
        which would indicate all phases converging to the same value.

        Args:
            real_embeds: [batch, seq, dim]
            imag_embeds: [batch, seq, dim]
            min_phase_variance: Minimum variance to encourage

        Returns:
            regularization_loss: Penalty for low variance
        """
        # Compute phase angles
        phase = torch.atan2(imag_embeds, real_embeds + 1e-8)  # [batch, seq, dim]

        # Compute circular variance per dimension
        # For circular data: var = 1 - |mean(e^{i*phase})|
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Mean across batch and sequence
        mean_cos = cos_phase.mean(dim=(0, 1))  # [dim]
        mean_sin = sin_phase.mean(dim=(0, 1))  # [dim]

        # Resultant length (inverse of variance)
        resultant_length = torch.sqrt(mean_cos**2 + mean_sin**2 + 1e-8)
        circular_variance = 1 - resultant_length  # [dim]

        # Penalize if variance is below threshold
        variance_penalty = F.relu(min_phase_variance - circular_variance).mean()

        return variance_penalty


class SemanticInterference(nn.Module):
    """
    Semantic interference for meaning resolution.

    Multiple meaning channels interfere constructively/destructively
    based on phase alignment, resolving toward coherent interpretation.
    """

    def __init__(self, dim: int, num_meanings: int = 4):
        super().__init__()

        self.dim = dim
        self.num_meanings = num_meanings

        # Project to meaning channels
        self.meaning_projections = nn.ModuleList([
            nn.Linear(dim * 2, dim) for _ in range(num_meanings)
        ])

        # Interference weights
        self.interference_weights = nn.Parameter(torch.ones(num_meanings) / num_meanings)

    def forward(
        self,
        real_embed: torch.Tensor,
        imag_embed: torch.Tensor
    ) -> torch.Tensor:
        """Compute interference and return resolved meaning."""
        # Combine real and imag
        combined = torch.cat([real_embed, imag_embed], dim=-1)

        # Project to meaning channels with phase shifts
        meanings = []
        for i, proj in enumerate(self.meaning_projections):
            phase_shift = 2 * math.pi * i / self.num_meanings
            # Phase-shifted combination
            shifted = combined * math.cos(phase_shift) + combined.roll(1, dims=-1) * math.sin(phase_shift)
            meanings.append(proj(shifted))

        # Stack and weight
        meanings_stack = torch.stack(meanings, dim=2)  # [batch, seq, num_meanings, dim]
        weights = F.softmax(self.interference_weights, dim=0).view(1, 1, -1, 1)

        # Interference sum
        return (meanings_stack * weights).sum(dim=2)
