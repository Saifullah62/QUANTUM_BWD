"""
Retrocausal Attention Mechanism (v2 - Refined with No-Leak Protocol)
=====================================================================

From Daugherty's Retrocausal Paradigm and Two-State Vector Formalism (TSVF):
"A quantum system is described using two wavefunctions simultaneously:
 one evolving forward from a pre-selected state, one evolving backward
 from a post-selected (future) state."

REFINED IMPLEMENTATION:
TSVF-inspired "future informs past" without violating autoregressive decoding:

1. TWO-PASS TRAINING:
   - Forward pass: Normal causal attention
   - Backward pass: Runs on teacher-forced suffix (delayed view)
   - Gated weak-value fusion combines them

2. DEPLOYMENT OPTIONS:
   - Cheap lookahead pass on draft continuation
   - Small planner head predicts future summary
   - Or disable backward (graceful degradation)

3. NO-LEAK PROTOCOL:
   - Backward signal is WEAK (gated, regularized)
   - Backward signal is OPTIONAL (can be disabled)
   - Never directly use target tokens as keys/values

WIN CONDITION: Effect→cause questions improve under no-leak protocol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class RetrocausalAttention(nn.Module):
    """
    Two-pass retrocausal attention with no-leak guarantees.

    Training mode:
    - Forward pass: standard causal
    - Backward pass: uses teacher-forced future (shifted targets)
    - Fusion: weak gated combination

    Inference mode:
    - Forward pass: standard causal
    - Backward pass: uses draft/planner prediction OR disabled
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        backward_weight: float = 0.1,  # Keep weak!
        backward_window: int = 32,  # Limited lookahead
        interference_type: str = "gated"
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.backward_weight = backward_weight
        self.backward_window = backward_window
        self.interference_type = interference_type

        # Forward attention (standard causal)
        self.forward_q = nn.Linear(hidden_dim, hidden_dim)
        self.forward_k = nn.Linear(hidden_dim, hidden_dim)
        self.forward_v = nn.Linear(hidden_dim, hidden_dim)

        # Backward attention (future → present)
        self.backward_q = nn.Linear(hidden_dim, hidden_dim)
        self.backward_k = nn.Linear(hidden_dim, hidden_dim)
        self.backward_v = nn.Linear(hidden_dim, hidden_dim)

        # Weak-value gating (keeps backward signal weak and optional)
        self.weak_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )

        # Learnable backward strength (initialized small)
        self.backward_strength = nn.Parameter(torch.tensor(backward_weight))

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # For no-leak regularization
        self._leak_penalty = 0.0

        # =================================================================
        # HARDENING: Prevent subtle leakage
        # =================================================================
        # Random backward dropout - model can't become dependent
        self.backward_dropout_rate = 0.3  # 30% of time, no backward signal

        # Stop-gradient flag for backward → forward direction
        self.stop_backward_grad = True

        # Entropy-based leak detector
        self.leak_detector_enabled = True

    def _causal_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        """Standard scaled dot-product attention with optional causal mask."""
        batch_size, num_heads, seq_len, head_dim = Q.shape

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V)

    def _limited_backward_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        window: int
    ) -> torch.Tensor:
        """
        Backward attention with limited window (no-leak).

        Each position can only attend to positions within `window` ahead.
        This prevents full future leakage while allowing local lookahead.
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

        # Create windowed backward mask
        # Position i can attend to positions [i+1, i+window]
        positions = torch.arange(seq_len, device=Q.device)
        row_pos = positions.unsqueeze(1)  # [seq, 1]
        col_pos = positions.unsqueeze(0)  # [1, seq]

        # Mask: can attend if col > row AND col <= row + window
        valid_future = (col_pos > row_pos) & (col_pos <= row_pos + window)
        mask = ~valid_future  # Invert for masking

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Handle the case where all positions are masked (would produce NaN)
        # Check if any row has all -inf values
        all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)
        if all_masked.any():
            # Replace fully masked rows with zeros (uniform attention would be meaningless)
            scores = torch.where(all_masked.expand_as(scores), torch.zeros_like(scores), scores)

        attn_weights = F.softmax(scores, dim=-1)
        # Replace any NaN weights with zeros (safety check)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V)

    def forward_pass(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Standard causal forward pass."""
        batch_size, seq_len, _ = hidden_states.shape

        Q = self.forward_q(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        K = self.forward_k(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        V = self.forward_v(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        out = self._causal_attention(Q, K, V, causal=True)
        return out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

    def backward_pass(
        self,
        hidden_states: torch.Tensor,
        future_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Backward pass using future context.

        If future_context is provided (training with teacher forcing),
        uses that. Otherwise uses limited self-lookahead.
        """
        batch_size, seq_len, _ = hidden_states.shape

        if future_context is not None:
            # Training: use provided future context (shifted targets)
            # But project through our own layers (no direct target leakage)
            context = future_context
        else:
            # Inference: use self with limited window
            context = hidden_states

        Q = self.backward_q(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        K = self.backward_k(context).view(
            batch_size, context.size(1), self.num_heads, self.head_dim
        ).transpose(1, 2)
        V = self.backward_v(context).view(
            batch_size, context.size(1), self.num_heads, self.head_dim
        ).transpose(1, 2)

        out = self._limited_backward_attention(Q, K, V, self.backward_window)
        return out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

    def weak_value_fusion(
        self,
        forward_state: torch.Tensor,
        backward_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine forward and backward through weak gating with HARDENING.

        Key properties:
        - Backward signal is weak (small weight, capped at 0.3)
        - Gate can learn to suppress backward when unhelpful
        - Forward always dominates
        - HARDENING: stop-grad prevents backward from influencing forward training
        - HARDENING: random dropout prevents over-reliance
        """
        # HARDENING: Stop gradient from backward to forward direction
        # This prevents backward from learning to push forward toward targets
        if self.stop_backward_grad:
            backward_detached = backward_state.detach()
        else:
            backward_detached = backward_state

        # HARDENING: Random backward dropout during training
        # Model can't become dependent on backward signal
        if self.training and self.backward_dropout_rate > 0:
            if torch.rand(1).item() < self.backward_dropout_rate:
                # Complete dropout: just return forward
                return forward_state

        # Compute gate based on both states
        # Note: we use original backward for gating (it can learn to detect usefulness)
        # but apply stop-grad to the actual signal being added
        combined = torch.cat([forward_state, backward_state], dim=-1)
        gate = self.weak_gate(combined)

        # Apply weak backward influence with hardened signal
        strength = torch.sigmoid(self.backward_strength) * 0.3  # Cap at 0.3
        fused = forward_state + strength * gate * backward_detached

        return fused

    def forward(
        self,
        hidden_states: torch.Tensor,
        future_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        enable_backward: bool = True,
        return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Two-pass retrocausal attention.

        Args:
            hidden_states: [batch, seq, dim] current states
            future_context: [batch, seq, dim] teacher-forced future (training only)
            attention_mask: attention mask
            enable_backward: whether to use backward pass
            return_states: return intermediate states for analysis

        Returns:
            output: [batch, seq, dim]
            info: optional dict with forward/backward states
        """
        # Forward pass (always runs)
        forward_state = self.forward_pass(hidden_states)

        if enable_backward:
            # Backward pass
            backward_state = self.backward_pass(hidden_states, future_context)

            # Weak-value fusion
            fused = self.weak_value_fusion(forward_state, backward_state)
        else:
            fused = forward_state
            backward_state = None

        # Output projection and residual
        output = self.out_proj(fused)
        output = self.layer_norm(hidden_states + self.dropout(output))

        if return_states:
            return output, {
                'forward_state': forward_state,
                'backward_state': backward_state,
                'backward_strength': torch.sigmoid(self.backward_strength).item()
            }

        return output, None


class FuturePlanner(nn.Module):
    """
    Small planner head that predicts future summary.

    Used during inference to provide backward signal when
    teacher-forced future is not available.

    This is a lightweight alternative to full draft generation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_future_tokens: int = 8,
        num_layers: int = 2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_future_tokens = num_future_tokens

        # Small transformer to predict future summary
        self.future_predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Compress to future summary tokens
        self.to_future = nn.Linear(hidden_dim, hidden_dim * num_future_tokens)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict future context summary.

        Args:
            hidden_states: [batch, seq, dim]

        Returns:
            future_summary: [batch, num_future_tokens, dim]
        """
        # Process current context
        processed = self.future_predictor(hidden_states)

        # Use last position to predict future
        last_hidden = processed[:, -1, :]  # [batch, dim]

        # Expand to future tokens
        future = self.to_future(last_hidden)  # [batch, dim * num_future]
        future = future.view(-1, self.num_future_tokens, self.hidden_dim)

        return future


class TwoPassTrainer:
    """
    Training utilities for two-pass retrocausal learning with HARDENED leak detection.

    Implements the no-leak protocol:
    1. Forward pass on input
    2. Backward pass on SHIFTED targets (not aligned with loss positions)
    3. Regularization to prevent over-reliance on backward
    4. HARDENED: Subtle leak detection (entropy, determinism, suspicious patterns)
    """

    @staticmethod
    def prepare_backward_context(
        target_embeddings: torch.Tensor,
        shift: int = 1
    ) -> torch.Tensor:
        """
        Prepare backward context from targets with shift.

        The shift ensures we're not directly leaking the prediction target.
        Position i's backward context comes from positions [i+shift, ...].
        """
        batch_size, seq_len, dim = target_embeddings.shape

        # Shift targets forward
        shifted = torch.zeros_like(target_embeddings)
        if shift < seq_len:
            shifted[:, :-shift, :] = target_embeddings[:, shift:, :]

        return shifted

    @staticmethod
    def compute_leak_penalty(
        forward_state: torch.Tensor,
        backward_state: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute penalty for information leakage.

        If backward state becomes too similar to targets,
        we're leaking and should penalize.
        """
        # Similarity between backward state and targets
        backward_target_sim = F.cosine_similarity(
            backward_state.view(-1, backward_state.size(-1)),
            target_embeddings.view(-1, target_embeddings.size(-1)),
            dim=-1
        ).mean()

        # Similarity between forward state and targets (baseline)
        forward_target_sim = F.cosine_similarity(
            forward_state.view(-1, forward_state.size(-1)),
            target_embeddings.view(-1, target_embeddings.size(-1)),
            dim=-1
        ).mean()

        # Penalize if backward is much more similar than forward
        leak = F.relu(backward_target_sim - forward_target_sim - 0.1)

        return leak

    @staticmethod
    def compute_backward_regularization(
        backward_strength: torch.Tensor,
        target_strength: float = 0.1
    ) -> torch.Tensor:
        """Regularize backward strength to stay weak."""
        return (backward_strength - target_strength).abs()

    # =========================================================================
    # HARDENED LEAK DETECTION
    # =========================================================================

    @staticmethod
    def compute_entropy_leak_score(
        logits_with_backward: torch.Tensor,
        logits_without_backward: torch.Tensor
    ) -> Dict[str, float]:
        """
        Detect subtle leakage via entropy reduction.

        If backward signal causes suspiciously low entropy (near-deterministic
        predictions), the backward path may be smuggling target information.

        RED FLAG: entropy_with_backward << entropy_without_backward
        """
        def compute_entropy(logits):
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            return entropy.mean()

        entropy_with = compute_entropy(logits_with_backward)
        entropy_without = compute_entropy(logits_without_backward)

        # Ratio: how much did backward reduce entropy?
        entropy_ratio = entropy_with / (entropy_without + 1e-8)

        # Suspicious if ratio < 0.5 (backward halved entropy)
        is_suspicious = entropy_ratio.item() < 0.5

        return {
            'entropy_with_backward': entropy_with.item(),
            'entropy_without_backward': entropy_without.item(),
            'entropy_ratio': entropy_ratio.item(),
            'is_suspicious': is_suspicious
        }

    @staticmethod
    def compute_determinism_penalty(
        logits: torch.Tensor,
        threshold: float = 0.9
    ) -> torch.Tensor:
        """
        Penalize near-deterministic predictions.

        If backward path pushes model to very high confidence, that's a red flag.
        """
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values

        # Penalize positions where max prob > threshold
        deterministic = F.relu(max_probs - threshold)

        return deterministic.mean()

    @staticmethod
    def compute_full_leak_report(
        forward_state: torch.Tensor,
        backward_state: torch.Tensor,
        target_embeddings: torch.Tensor,
        logits_with_backward: Optional[torch.Tensor] = None,
        logits_without_backward: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive leak analysis report.

        Returns all leak metrics for monitoring.
        """
        report = {}

        # Basic cosine similarity leak
        basic_leak = TwoPassTrainer.compute_leak_penalty(
            forward_state, backward_state, target_embeddings
        )
        report['basic_leak_penalty'] = basic_leak.item()

        # Forward-backward similarity (should be moderate, not too high)
        fb_sim = F.cosine_similarity(
            forward_state.view(-1, forward_state.size(-1)),
            backward_state.view(-1, backward_state.size(-1)),
            dim=-1
        ).mean()
        report['forward_backward_similarity'] = fb_sim.item()

        # Entropy analysis if logits provided
        if logits_with_backward is not None and logits_without_backward is not None:
            entropy_report = TwoPassTrainer.compute_entropy_leak_score(
                logits_with_backward, logits_without_backward
            )
            report.update(entropy_report)

            # Determinism penalty
            det_penalty = TwoPassTrainer.compute_determinism_penalty(logits_with_backward)
            report['determinism_penalty'] = det_penalty.item()

        # Overall leak score (weighted combination)
        leak_score = report['basic_leak_penalty']
        if 'entropy_ratio' in report:
            # Add entropy-based leak detection
            entropy_leak = max(0, 1.0 - report['entropy_ratio'])  # Higher if ratio is low
            leak_score += 0.5 * entropy_leak
        if 'determinism_penalty' in report:
            leak_score += 0.3 * report['determinism_penalty']

        report['overall_leak_score'] = leak_score

        # Red flag detection
        red_flags = []
        if report['basic_leak_penalty'] > 0.1:
            red_flags.append("backward too similar to targets")
        if report.get('is_suspicious', False):
            red_flags.append("entropy suspiciously low with backward")
        if report.get('determinism_penalty', 0) > 0.1:
            red_flags.append("predictions too deterministic")
        if fb_sim.item() > 0.8:
            red_flags.append("forward and backward too similar")

        report['red_flags'] = red_flags
        report['is_clean'] = len(red_flags) == 0

        return report


class RetrocausalBlock(nn.Module):
    """
    Complete retrocausal transformer block with two-pass support.

    Drop-in replacement for standard transformer block that adds
    retrocausal capabilities when future context is available.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
        backward_weight: float = 0.1,
        backward_window: int = 32
    ):
        super().__init__()

        intermediate_dim = intermediate_dim or hidden_dim * 4

        # Retrocausal attention
        self.retro_attn = RetrocausalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            backward_weight=backward_weight,
            backward_window=backward_window
        )

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Optional future planner for inference
        self.future_planner = FuturePlanner(hidden_dim)
        self.use_planner = False  # Enable during inference if needed

    def forward(
        self,
        hidden_states: torch.Tensor,
        future_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        enable_backward: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional retrocausal enhancement.

        Training: provide future_context from shifted targets
        Inference: use planner or disable backward
        """
        # Get future context for backward pass
        if future_context is None and enable_backward and self.use_planner:
            future_context = self.future_planner(hidden_states)

        # Retrocausal attention
        attn_out, _ = self.retro_attn(
            self.norm1(hidden_states),
            future_context=future_context,
            attention_mask=attention_mask,
            enable_backward=enable_backward
        )
        hidden_states = hidden_states + attn_out

        # Feed-forward
        ff_out = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_out

        return hidden_states


# Legacy compatibility - keep old class names working
class TwoStateVectorAttention(RetrocausalAttention):
    """Alias for backward compatibility."""
    pass


class PhaseInterference(nn.Module):
    """Phase-based interference for combining forward/backward states."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.phase_proj = nn.Linear(dim * 2, dim)
        self.magnitude_proj = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, forward_state: torch.Tensor, backward_state: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([forward_state, backward_state], dim=-1)
        magnitude = torch.sigmoid(self.magnitude_proj(combined))
        phase = torch.tanh(self.phase_proj(combined))
        gate = torch.sigmoid(self.gate(combined))
        return gate * (magnitude * forward_state + phase * backward_state)


class AdditiveInterference(nn.Module):
    """Simple additive combination."""

    def __init__(self, dim: int, backward_weight: float = 0.3):
        super().__init__()
        self.backward_weight = nn.Parameter(torch.tensor(backward_weight))
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, forward_state: torch.Tensor, backward_state: torch.Tensor) -> torch.Tensor:
        weight = torch.sigmoid(self.backward_weight)
        return self.layer_norm((1 - weight) * forward_state + weight * backward_state)


class MultiplicativeInterference(nn.Module):
    """Multiplicative combination."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, forward_state: torch.Tensor, backward_state: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([forward_state, backward_state], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        return self.layer_norm(forward_state * gate + backward_state * (1 - gate))


class HindsightEnhancedBlock(RetrocausalBlock):
    """Alias for backward compatibility."""
    pass
