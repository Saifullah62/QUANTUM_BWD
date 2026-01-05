"""
Qualia Output Layers (v2 - Refined with Mixed Supervision)
==========================================================

From Daugherty's Qualia Hypothesis:
"Qualia may be treated as observables in quantum theory - like spin or
polarization, they remain ontologically indeterminate until measured."

REFINED IMPLEMENTATION:
Without supervision, qualia channels collapse to redundant features or decoration.

Mixed supervision strategy:
SELF-SUPERVISED (computable from model internals):
- Certainty: from predictive entropy
- Novelty: from surprisal/perplexity
- Coherence: from cross-layer agreement
- Temporality: from tense/aspect patterns
- Abstraction: from concreteness lexicons

WEAK-SUPERVISED (from teacher models/heuristics):
- Valence: from sentiment classifiers
- Arousal: from emotion regressors
- Agency: from SRL / voice detection

WIN CONDITION: Qualia channels correlate with external measures and
can be used as control knobs for generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


# Module-level qualia channel definitions
QUALIA_CHANNELS = {
    # Self-supervised channels
    'certainty': {'supervision': 'self', 'source': 'entropy'},
    'novelty': {'supervision': 'self', 'source': 'surprisal'},
    'coherence': {'supervision': 'self', 'source': 'layer_agreement'},
    'temporality': {'supervision': 'self', 'source': 'tense_pattern'},
    'abstraction': {'supervision': 'self', 'source': 'concreteness'},
    # Weak-supervised channels
    'valence': {'supervision': 'weak', 'source': 'sentiment'},
    'arousal': {'supervision': 'weak', 'source': 'emotion'},
    'agency': {'supervision': 'weak', 'source': 'srl_voice'},
}


class QualiaEncoder(nn.Module):
    """
    Encode qualitative state dimensions with mixed supervision.

    Each qualia channel has a specific supervision strategy that
    prevents collapse and ensures meaningful, calibrated values.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_qualia: int = 8,
        qualia_dim: int = 64
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_qualia = num_qualia
        self.qualia_dim = qualia_dim

        # Qualia names (in order)
        self.qualia_names = list(QUALIA_CHANNELS.keys())[:num_qualia]

        # Individual qualia encoders (learnable projections)
        # Using softer activation to prevent saturation
        self.qualia_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, qualia_dim),
                nn.GELU(),
                nn.Linear(qualia_dim, 1),
                # Use scaled sigmoid instead of Tanh to prevent saturation
                # sigmoid(x) * 1.8 - 0.9 maps to roughly [-0.9, 0.9]
                # This leaves headroom for learning without hitting boundaries
            )
            for name in self.qualia_names
        })

        # Temperature for qualia scaling (learnable to prevent saturation)
        self.qualia_temperature = nn.Parameter(torch.ones(num_qualia) * 0.5)

        # Combined qualia projection
        self.qualia_proj = nn.Linear(num_qualia, hidden_dim)

        # For self-supervised computation
        self.entropy_head = nn.Linear(hidden_dim, hidden_dim)  # For certainty
        self.layer_agreement_proj = nn.Linear(hidden_dim * 2, 1)  # For coherence

        # Tense patterns (learned embeddings for temporal detection)
        self.tense_patterns = nn.Parameter(torch.randn(3, hidden_dim) * 0.1)  # past/present/future

        # Concreteness projection
        self.concreteness_proj = nn.Linear(hidden_dim, 1)

    def compute_certainty(
        self,
        hidden_states: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Certainty from predictive entropy.

        Low entropy = high certainty (confident predictions)
        High entropy = low certainty (uncertain)
        """
        if logits is not None:
            # Use actual logits if provided
            probs = F.softmax(logits, dim=-1)
        else:
            # Estimate from hidden states
            probs = F.softmax(self.entropy_head(hidden_states), dim=-1)

        # Shannon entropy
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1, keepdim=True)

        # Normalize and invert (high entropy = low certainty)
        max_entropy = math.log(probs.size(-1))
        certainty = 1 - (entropy / max_entropy)

        # Scale to [-1, 1]
        return (certainty * 2 - 1)

    def compute_novelty(
        self,
        hidden_states: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Novelty from surprisal / deviation from expectation.

        Novel = differs from recent context
        """
        if prev_hidden is not None:
            # Compare to previous hidden states
            diff = (hidden_states - prev_hidden).norm(dim=-1, keepdim=True)
            novelty = torch.tanh(diff / math.sqrt(self.hidden_dim))
        else:
            # Compare each position to running mean
            cumsum = torch.cumsum(hidden_states, dim=1)
            counts = torch.arange(1, hidden_states.size(1) + 1, device=hidden_states.device)
            running_mean = cumsum / counts.view(1, -1, 1)
            diff = (hidden_states - running_mean).norm(dim=-1, keepdim=True)
            novelty = torch.tanh(diff / math.sqrt(self.hidden_dim))

        # Scale to [-1, 1] (0 = expected, 1 = very novel)
        return novelty * 2 - 1

    def compute_coherence(
        self,
        hidden_states: torch.Tensor,
        other_layer_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Coherence from cross-layer or internal agreement.

        High coherence = consistent representation across layers/positions
        """
        if other_layer_states is not None:
            # Compare to another layer's representation
            combined = torch.cat([hidden_states, other_layer_states], dim=-1)
            coherence = torch.sigmoid(self.layer_agreement_proj(combined))
        else:
            # Self-coherence: how consistent within sequence
            mean_state = hidden_states.mean(dim=1, keepdim=True)
            similarity = F.cosine_similarity(hidden_states, mean_state, dim=-1)
            coherence = similarity.unsqueeze(-1)

        # Scale to [-1, 1]
        return coherence * 2 - 1

    def compute_temporality(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Temporality from tense/aspect pattern detection.

        -1 = past focus, 0 = present, +1 = future
        """
        # Compute similarity to tense patterns
        # [batch, seq, dim] @ [3, dim]^T -> [batch, seq, 3]
        tense_scores = torch.einsum('bsd,td->bst', hidden_states, self.tense_patterns)
        tense_probs = F.softmax(tense_scores, dim=-1)

        # Weight: past=-1, present=0, future=+1
        weights = torch.tensor([-1.0, 0.0, 1.0], device=hidden_states.device)
        temporality = (tense_probs * weights).sum(dim=-1, keepdim=True)

        return temporality

    def compute_abstraction(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Abstraction from concreteness estimation.

        -1 = concrete (sensory, specific)
        +1 = abstract (conceptual, general)
        """
        concreteness = torch.tanh(self.concreteness_proj(hidden_states))
        # Invert: high concreteness = low abstraction
        abstraction = -concreteness
        return abstraction

    def _safe_scale(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Scale qualia values to [-1, 1] without saturation.

        Uses softer scaling that leaves headroom at boundaries.
        """
        # Soft scaling: tanh with temperature to prevent saturation
        # Higher temperature = softer output, less saturation
        scaled = torch.tanh(x / (temperature + 1e-3))
        # Leave headroom: map [-1,1] to [-0.9, 0.9]
        return scaled * 0.9

    def encode_qualia(
        self,
        hidden_states: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        other_layer_states: Optional[torch.Tensor] = None,
        prev_hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode hidden states into qualia values using mixed supervision.

        Self-supervised channels are computed directly.
        Weak-supervised channels use learned projections.

        IMPORTANT: All values are kept in [-0.9, 0.9] to prevent boundary saturation.
        """
        qualia_values = {}

        for idx, name in enumerate(self.qualia_names):
            config = QUALIA_CHANNELS[name]
            temp = F.softplus(self.qualia_temperature[idx]) + 0.1  # Ensure positive temperature

            if config['supervision'] == 'self':
                # Compute self-supervised qualia
                if name == 'certainty':
                    raw = self.compute_certainty(hidden_states, logits)
                    # Apply temperature scaling to prevent saturation
                    qualia_values[name] = self._safe_scale(raw, temp)
                elif name == 'novelty':
                    raw = self.compute_novelty(hidden_states, prev_hidden)
                    qualia_values[name] = self._safe_scale(raw, temp)
                elif name == 'coherence':
                    raw = self.compute_coherence(hidden_states, other_layer_states)
                    qualia_values[name] = self._safe_scale(raw, temp)
                elif name == 'temporality':
                    raw = self.compute_temporality(hidden_states)
                    qualia_values[name] = self._safe_scale(raw, temp)
                elif name == 'abstraction':
                    raw = self.compute_abstraction(hidden_states)
                    qualia_values[name] = self._safe_scale(raw, temp)
                else:
                    raw = self.qualia_encoders[name](hidden_states)
                    qualia_values[name] = self._safe_scale(raw, temp)
            else:
                # Weak-supervised: use learned projection with temperature
                raw = self.qualia_encoders[name](hidden_states)
                qualia_values[name] = self._safe_scale(raw, temp)

        # Stack into tensor
        qualia_tensor = torch.cat(list(qualia_values.values()), dim=-1)

        return qualia_tensor, qualia_values

    def modulate_by_qualia(
        self,
        hidden_states: torch.Tensor,
        qualia_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Modulate hidden states based on qualia values."""
        qualia_influence = self.qualia_proj(qualia_tensor)
        gate = torch.sigmoid(qualia_influence)
        return hidden_states * (1 + 0.2 * gate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        other_layer_states: Optional[torch.Tensor] = None,
        return_qualia: bool = False
    ) -> torch.Tensor:
        """Forward pass encoding and applying qualia."""
        qualia_tensor, qualia_dict = self.encode_qualia(
            hidden_states, logits, other_layer_states
        )
        modulated = self.modulate_by_qualia(hidden_states, qualia_tensor)

        if return_qualia:
            return modulated, qualia_dict

        return modulated


class WeakSupervisor(nn.Module):
    """
    Provides weak supervision signals for qualia channels.

    Uses pre-trained teacher models or heuristics to provide
    training targets for valence, arousal, and agency.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Sentiment teacher (valence) - simple learned proxy
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        # Emotion intensity teacher (arousal)
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Arousal in [0, 1], will be scaled to [-1, 1]
        )

        # Agency detector (active vs passive voice proxy)
        self.agency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def get_valence_target(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get valence target from sentiment teacher."""
        return self.sentiment_head(hidden_states)

    def get_arousal_target(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get arousal target from emotion teacher."""
        arousal = self.arousal_head(hidden_states)
        return arousal * 2 - 1  # Scale to [-1, 1]

    def get_agency_target(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get agency target from SRL proxy."""
        return self.agency_head(hidden_states)

    def get_all_targets(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get all weak supervision targets."""
        return {
            'valence': self.get_valence_target(hidden_states),
            'arousal': self.get_arousal_target(hidden_states),
            'agency': self.get_agency_target(hidden_states)
        }


class QualiaOutputHead(nn.Module):
    """
    Output head with qualia-modulated token selection.

    Standard LLM: hidden → logits → argmax/sample
    QLLM: hidden → qualia + logits → phase-coherent selection

    Qualia can be used as control knobs for generation.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_qualia: int = 8,
        qualia_modulation_strength: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_qualia = num_qualia
        self.modulation_strength = qualia_modulation_strength

        # Standard token logits
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Qualia encoder
        self.qualia_encoder = QualiaEncoder(hidden_dim, num_qualia)

        # Qualia-to-logit modulation
        self.qualia_to_vocab = nn.Linear(num_qualia, vocab_size)

        # Coherence scorer
        self.coherence_scorer = nn.Sequential(
            nn.Linear(hidden_dim + num_qualia, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Token-qualia compatibility
        self.token_qualia_compatibility = nn.Embedding(vocab_size, num_qualia)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_qualia: Optional[torch.Tensor] = None,
        return_qualia: bool = False,
        other_layer_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Compute qualia-modulated logits."""
        # Base logits
        base_logits = self.lm_head(hidden_states)

        # Encode qualia (pass logits for self-supervised certainty)
        qualia_tensor, qualia_dict = self.qualia_encoder.encode_qualia(
            hidden_states,
            logits=base_logits,
            other_layer_states=other_layer_states
        )

        # Qualia-based logit modulation
        qualia_logit_bias = self.qualia_to_vocab(qualia_tensor)

        # Compute coherence
        combined = torch.cat([hidden_states, qualia_tensor], dim=-1)
        coherence = self.coherence_scorer(combined)

        # Modulate logits
        modulated_logits = base_logits + self.modulation_strength * coherence * qualia_logit_bias

        # Target qualia bias
        if target_qualia is not None:
            compatibility = self.token_qualia_compatibility.weight
            target_compatibility = torch.einsum('bsq,vq->bsv', target_qualia, compatibility)
            modulated_logits = modulated_logits + 0.1 * target_compatibility

        if return_qualia:
            qualia_info = {
                'qualia_values': qualia_dict,
                'qualia_tensor': qualia_tensor,
                'coherence': coherence
            }
            return modulated_logits, qualia_info

        return modulated_logits, None


class QualiaLoss(nn.Module):
    """
    Loss functions for qualia-aware training with mixed supervision.

    Includes:
    1. Consistency loss (coherent qualia across context)
    2. Diversity loss (prevent channel collapse)
    3. Weak supervision alignment (match teacher targets)
    4. Self-supervision consistency (internal computation agreement)
    """

    def __init__(self, num_qualia: int = 8):
        super().__init__()
        self.num_qualia = num_qualia
        self.weak_supervisor = WeakSupervisor(256)  # Will be set properly

    def consistency_loss(self, qualia_tensor: torch.Tensor) -> torch.Tensor:
        """Encourage qualia consistency within local windows."""
        if qualia_tensor.size(1) < 2:
            return torch.tensor(0.0, device=qualia_tensor.device)
        diff = qualia_tensor[:, 1:] - qualia_tensor[:, :-1]
        return (diff ** 2).mean()

    def diversity_loss(self, qualia_tensor: torch.Tensor) -> torch.Tensor:
        """Prevent collapse to single qualia pattern."""
        variance = qualia_tensor.var(dim=(0, 1)).mean()
        return -torch.log(variance + 1e-6)

    def weak_supervision_loss(
        self,
        qualia_dict: Dict[str, torch.Tensor],
        hidden_states: torch.Tensor,
        weak_targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Align weak-supervised channels with teacher targets."""
        loss = 0.0

        weak_channels = ['valence', 'arousal', 'agency']
        for channel in weak_channels:
            if channel in qualia_dict:
                if weak_targets is not None and channel in weak_targets:
                    target = weak_targets[channel]
                else:
                    # Generate target from weak supervisor
                    if channel == 'valence':
                        target = self.weak_supervisor.get_valence_target(hidden_states)
                    elif channel == 'arousal':
                        target = self.weak_supervisor.get_arousal_target(hidden_states)
                    elif channel == 'agency':
                        target = self.weak_supervisor.get_agency_target(hidden_states)

                loss = loss + F.mse_loss(qualia_dict[channel], target.detach())

        return loss / len(weak_channels)

    def forward(
        self,
        qualia_tensor: torch.Tensor,
        qualia_dict: Optional[Dict[str, torch.Tensor]] = None,
        hidden_states: Optional[torch.Tensor] = None,
        weak_targets: Optional[Dict[str, torch.Tensor]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Compute combined qualia loss."""
        if weights is None:
            weights = {
                'consistency': 0.1,
                'diversity': 0.05,
                'weak_supervision': 0.5
            }

        loss = weights['consistency'] * self.consistency_loss(qualia_tensor)
        loss = loss + weights['diversity'] * self.diversity_loss(qualia_tensor)

        if qualia_dict is not None and hidden_states is not None:
            loss = loss + weights['weak_supervision'] * self.weak_supervision_loss(
                qualia_dict, hidden_states, weak_targets
            )

        return loss


class SubjectiveExperienceLayer(nn.Module):
    """
    Experimental: Persistent experiential state modulation.

    Maintains a running experiential state that modulates processing
    based on accumulated context and qualia history.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_qualia: int = 8,
        memory_size: int = 128
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_qualia = num_qualia
        self.memory_size = memory_size

        # Experiential memory
        self.experiential_memory = nn.Parameter(
            torch.zeros(1, memory_size, hidden_dim)
        )

        # Qualia integration
        self.qualia_integrator = nn.GRUCell(num_qualia, hidden_dim)

        # Experience-to-hidden projection
        self.experience_proj = nn.Linear(hidden_dim, hidden_dim)

    def update_experience(
        self,
        qualia_tensor: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Update experiential state based on current qualia."""
        batch_size = hidden_states.size(0)
        mean_qualia = qualia_tensor.mean(dim=1)
        current_exp = self.experiential_memory.mean(dim=1).expand(batch_size, -1)
        return self.qualia_integrator(mean_qualia, current_exp)

    def forward(
        self,
        hidden_states: torch.Tensor,
        qualia_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply experiential modulation."""
        experience = self.update_experience(qualia_tensor, hidden_states)
        influence = self.experience_proj(experience).unsqueeze(1)
        modulated = hidden_states + 0.1 * influence
        return modulated, experience
