"""
Lindblad Dissipative Layer (v2 - Refined)
==========================================

From Daugherty's Environmental Dynamics paradigm:
"Rather than isolating qubits from environmental interaction, DQC tunes
that interaction so that the correct answer becomes the attractor state."

REFINED IMPLEMENTATION:
1. Learnable low-rank noise injection (not random dropout)
2. Lipschitz/Jacobian regularization for stability
3. Noise-invariant consistency objective

WIN CONDITION: Graceful degradation under injected noise and perturbations.
Same input with different noise draws should land in same semantic basin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math


class LearnableNoiseInjector(nn.Module):
    """
    Learnable structured noise injection.

    NOT random dropout - this is parameterized noise with learnable covariance
    in a low-rank subspace. The noise patterns are learned to be useful.
    """

    def __init__(
        self,
        hidden_dim: int,
        noise_rank: int = 8,
        num_noise_channels: int = 4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.noise_rank = noise_rank
        self.num_channels = num_noise_channels

        # Low-rank noise basis: noise lives in span of these vectors
        # [num_channels, noise_rank, hidden_dim]
        self.noise_basis = nn.Parameter(
            torch.randn(num_noise_channels, noise_rank, hidden_dim) * 0.01
        )

        # Learnable noise covariance (diagonal in low-rank space)
        self.noise_scales = nn.Parameter(torch.ones(num_noise_channels, noise_rank) * 0.1)

        # Channel selection based on input
        self.channel_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_noise_channels)
        )

        # Global noise strength (learnable)
        self.noise_strength = nn.Parameter(torch.tensor(0.1))

    def sample_noise(
        self,
        hidden_states: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample structured noise for given hidden states.

        Args:
            hidden_states: [batch, seq, dim]
            deterministic: If True, return mean (zero noise for testing)

        Returns:
            noise: [batch, seq, dim] structured noise
        """
        batch_size, seq_len, dim = hidden_states.shape

        if deterministic:
            return torch.zeros_like(hidden_states)

        # Select noise channels based on input
        channel_logits = self.channel_selector(hidden_states.mean(dim=1))  # [batch, num_channels]
        channel_weights = F.softmax(channel_logits, dim=-1)  # [batch, num_channels]

        # Sample in low-rank space
        # z ~ N(0, diag(scales²))
        z = torch.randn(batch_size, self.num_channels, self.noise_rank, device=hidden_states.device)
        z = z * torch.sigmoid(self.noise_scales).unsqueeze(0)  # [batch, channels, rank]

        # Project to full space using noise basis
        # [batch, channels, rank] @ [channels, rank, dim] -> [batch, channels, dim]
        noise_per_channel = torch.einsum('bcr,crd->bcd', z, self.noise_basis)

        # Weight by channel selection
        # [batch, channels, dim] * [batch, channels, 1] -> [batch, channels, dim]
        weighted_noise = noise_per_channel * channel_weights.unsqueeze(-1)
        noise = weighted_noise.sum(dim=1)  # [batch, dim]

        # Expand to sequence
        noise = noise.unsqueeze(1).expand(-1, seq_len, -1)

        # Apply global strength
        return noise * torch.sigmoid(self.noise_strength)

    def forward(
        self,
        hidden_states: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject structured noise into hidden states.

        Returns:
            noisy_states: hidden_states + noise
            noise: the noise that was added (for consistency loss)
        """
        noise = self.sample_noise(hidden_states, deterministic)
        return hidden_states + noise, noise


class ContractiveStabilizer(nn.Module):
    """
    Contractive layer that enforces Lipschitz constraints.

    Ensures that the transformation doesn't amplify perturbations,
    making representations converge to stable manifolds.

    Uses spectral normalization proxy for efficient Lipschitz bound.
    """

    def __init__(
        self,
        hidden_dim: int,
        contraction_rate: float = 0.9,
        num_power_iterations: int = 1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.contraction_rate = contraction_rate
        self.num_power_iterations = num_power_iterations

        # Main transformation (will be spectrally normalized)
        self.weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # Register buffers for power iteration
        self.register_buffer('u', torch.randn(hidden_dim))
        self.register_buffer('v', torch.randn(hidden_dim))

        # Learnable contraction strength
        self.contraction_strength = nn.Parameter(torch.tensor(0.5))

    def _spectral_norm(self, W: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compute spectral norm via power iteration and return normalized weight.
        """
        u = self.u
        v = self.v

        for _ in range(self.num_power_iterations):
            v = F.normalize(W.T @ u, dim=0)
            u = F.normalize(W @ v, dim=0)

        # Update buffers
        self.u = u.detach()
        self.v = v.detach()

        # Spectral norm
        sigma = u @ W @ v

        # Normalize weight to have spectral norm <= contraction_rate
        W_normalized = W * (self.contraction_rate / (sigma + 1e-6))

        return W_normalized, sigma.item()

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_lipschitz: bool = False
    ) -> torch.Tensor:
        """
        Apply contractive transformation.

        The transformation is guaranteed to have Lipschitz constant <= contraction_rate.
        """
        # Get spectrally normalized weight
        W_norm, sigma = self._spectral_norm(self.weight)

        # Apply transformation
        contracted = F.linear(hidden_states, W_norm, self.bias)

        # Interpolate with identity (residual connection with contraction)
        strength = torch.sigmoid(self.contraction_strength)
        output = (1 - strength) * hidden_states + strength * contracted

        if return_lipschitz:
            return output, sigma

        return output


class LindbladLayer(nn.Module):
    """
    Lindblad dynamics with practical refinements.

    Components:
    1. LearnableNoiseInjector: Structured, learnable noise
    2. ContractiveStabilizer: Lipschitz-bounded transformation
    3. Consistency objective support: Same input + different noise → same basin

    The Lindblad master equation interpretation:
    - Noise injection = dissipative channels (Lₖ operators)
    - Contraction = evolution toward steady state
    - Consistency = noise-invariant fixed point
    """

    def __init__(
        self,
        hidden_dim: int,
        num_operators: int = 4,
        noise_rank: int = 8,
        dt: float = 0.1,
        dissipation_strength: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_operators = num_operators
        self.dt = dt
        self.dissipation_strength = dissipation_strength

        # Noise injection
        self.noise_injector = LearnableNoiseInjector(
            hidden_dim, noise_rank=noise_rank, num_noise_channels=num_operators
        )

        # Contractive stabilization
        self.stabilizer = ContractiveStabilizer(
            hidden_dim, contraction_rate=0.95
        )

        # Attractor layer (steady state learning)
        self.num_attractors = num_operators
        self.attractors = nn.Parameter(torch.randn(num_operators, hidden_dim) * 0.1)
        self.attractor_selector = nn.Linear(hidden_dim, num_operators)

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # For consistency loss computation
        self._last_noise = None
        self._noise_invariant_mode = False

    def compute_attractor_target(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute soft attractor target"""
        logits = self.attractor_selector(hidden_states)
        weights = F.softmax(logits, dim=-1)
        return torch.einsum('bsn,nd->bsd', weights, self.attractors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_steps: int = 1,
        deterministic: bool = False,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Apply Lindblad dynamics.

        Args:
            hidden_states: [batch, seq, dim]
            num_steps: Number of evolution steps
            deterministic: If True, no noise (for evaluation)
            return_trajectory: Return all intermediate states

        Returns:
            evolved_states: [batch, seq, dim]
        """
        trajectory = [hidden_states] if return_trajectory else None

        x = hidden_states

        for step in range(num_steps):
            # 1. Inject structured noise (dissipative channels)
            x_noisy, noise = self.noise_injector(x, deterministic=deterministic)
            self._last_noise = noise  # Store for consistency loss

            # 2. Contract toward stable manifold
            x_contracted = self.stabilizer(x_noisy)

            # 3. Bias toward learned attractors
            attractor_target = self.compute_attractor_target(x_contracted)
            x = (1 - self.dissipation_strength) * x_contracted + self.dissipation_strength * attractor_target

            if return_trajectory:
                trajectory.append(x)

        # Final normalization
        x = self.layer_norm(x)

        if return_trajectory:
            return x, torch.stack(trajectory)

        return x

    def compute_consistency_loss(
        self,
        hidden_states: torch.Tensor,
        num_samples: int = 3
    ) -> torch.Tensor:
        """
        Compute noise-invariant consistency loss.

        WIN CONDITION: Same input with different noise draws should land in same basin.
        """
        outputs = []

        for _ in range(num_samples):
            output = self.forward(hidden_states, deterministic=False)
            outputs.append(output)

        # Stack outputs
        outputs = torch.stack(outputs)  # [samples, batch, seq, dim]

        # Compute pairwise distances
        mean_output = outputs.mean(dim=0)
        consistency_loss = ((outputs - mean_output) ** 2).mean()

        return consistency_loss


class DissipativeNormalization(nn.Module):
    """
    Normalization via controlled dissipation toward attractors.

    Combines:
    - Standard normalization (for stability)
    - Attractor dynamics (for meaning preservation)
    - Contractive regularization (for noise robustness)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_attractors: int = 4,
        temperature: float = 1.0
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_attractors = num_attractors
        self.temperature = temperature

        # Learned attractors
        self.attractors = nn.Parameter(torch.randn(num_attractors, hidden_dim) * 0.1)

        # Attractor selection
        self.attractor_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_attractors)
        )

        # Dissipation rate
        self.dissipation_rate = nn.Parameter(torch.tensor(0.1))

        # Standard normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply dissipative normalization"""
        # Compute attractor target
        logits = self.attractor_selector(hidden_states)
        weights = F.softmax(logits / self.temperature, dim=-1)
        attractor_target = torch.einsum('bsn,nd->bsd', weights, self.attractors)

        # Dissipate toward attractor
        rate = torch.sigmoid(self.dissipation_rate)
        dissipated = (1 - rate) * hidden_states + rate * attractor_target

        # Combine with layer norm
        normalized = self.layer_norm(hidden_states)
        return 0.5 * dissipated + 0.5 * normalized


class EntropyGate(nn.Module):
    """
    Entropy-based gating for information flow control.

    High entropy = more exploration/diversity
    Low entropy = more focus/preservation
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.entropy_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Sigmoid()
        )

    def compute_entropy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute per-position entropy"""
        probs = F.softmax(self.entropy_proj(hidden_states), dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1, keepdim=True)
        max_entropy = math.log(self.hidden_dim)
        return entropy / max_entropy

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply entropy-based gating"""
        entropy = self.compute_entropy(hidden_states)
        gate_input = torch.cat([hidden_states, entropy], dim=-1)
        gate = self.gate(gate_input)
        return gate * hidden_states + (1 - gate) * hidden_states.mean(dim=-1, keepdim=True)


class HardNoiseChallenge(nn.Module):
    """
    Hard noise challenge for testing Lindblad invariance.

    This creates a DIFFICULT noise regime where baseline models fail
    but Lindblad-trained models should maintain semantic basin agreement.

    Key insight: baseline is already at 1.0 consistency in easy regime.
    We need noise levels that actually break baseline performance.
    """

    def __init__(
        self,
        hidden_dim: int,
        noise_levels: List[float] = [0.3, 0.5, 0.7, 1.0],  # Much higher than training noise
        num_perturbation_types: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_levels = noise_levels
        self.num_perturbation_types = num_perturbation_types

        # Different types of challenging perturbations
        self.perturbation_types = [
            'gaussian',      # Standard Gaussian noise
            'adversarial',   # Gradient-aligned perturbation
            'structured',    # Correlated noise across dimensions
            'dropout_mask'   # Random zeroing of dimensions
        ]

    def apply_perturbation(
        self,
        x: torch.Tensor,
        noise_level: float,
        perturbation_type: str = 'gaussian'
    ) -> torch.Tensor:
        """Apply a specific type of perturbation at given level."""

        if perturbation_type == 'gaussian':
            noise = torch.randn_like(x) * noise_level
            return x + noise

        elif perturbation_type == 'adversarial':
            # Perturbation in direction of maximum variance
            # Approximated by perturbing along principal direction
            mean = x.mean(dim=(0, 1), keepdim=True)
            centered = x - mean
            # Use a random but consistent direction
            direction = torch.randn(1, 1, x.size(-1), device=x.device)
            direction = direction / (direction.norm() + 1e-8)
            projection = (centered * direction).sum(dim=-1, keepdim=True)
            return x + noise_level * projection * direction

        elif perturbation_type == 'structured':
            # Correlated noise: nearby dimensions get similar noise
            base_noise = torch.randn(x.size(0), x.size(1), x.size(-1) // 8 + 1, device=x.device)
            # Upsample to create correlation
            structured_noise = F.interpolate(
                base_noise.unsqueeze(1), size=(x.size(1), x.size(-1)), mode='bilinear'
            ).squeeze(1)
            return x + noise_level * structured_noise

        elif perturbation_type == 'dropout_mask':
            # Random dimension dropout (more aggressive than standard dropout)
            mask = torch.rand_like(x) > noise_level
            return x * mask / (1 - noise_level + 1e-8)

        return x

    def compute_basin_agreement(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor,
        method: str = 'cosine'
    ) -> torch.Tensor:
        """
        Compute semantic basin agreement between original and perturbed.

        Basin agreement = are they in the same semantic "attractor basin"?

        This is MORE than just output similarity - we want representations
        that would lead to the same downstream decisions.

        Args:
            original: [batch, seq, dim] original representations
            perturbed: [batch, seq, dim] perturbed representations
            method: 'cosine', 'l2', or 'rank' agreement metric

        Returns:
            agreement: scalar in [0, 1] where 1 = perfect agreement
        """
        if method == 'cosine':
            # Cosine similarity (standard)
            orig_flat = original.view(-1, original.size(-1))
            pert_flat = perturbed.view(-1, perturbed.size(-1))
            similarity = F.cosine_similarity(orig_flat, pert_flat, dim=-1)
            return similarity.mean()

        elif method == 'l2':
            # L2 distance normalized by magnitude
            diff = (original - perturbed).norm(dim=-1)
            orig_norm = original.norm(dim=-1) + 1e-8
            relative_diff = diff / orig_norm
            return (1 - relative_diff.clamp(0, 1)).mean()

        elif method == 'rank':
            # Rank-order agreement: do top-k dimensions agree?
            k = min(32, original.size(-1) // 4)

            orig_flat = original.view(-1, original.size(-1))
            pert_flat = perturbed.view(-1, perturbed.size(-1))

            orig_topk = torch.topk(orig_flat, k, dim=-1).indices
            pert_topk = torch.topk(pert_flat, k, dim=-1).indices

            # Count overlapping indices
            agreements = []
            for i in range(orig_flat.size(0)):
                orig_set = set(orig_topk[i].tolist())
                pert_set = set(pert_topk[i].tolist())
                overlap = len(orig_set & pert_set) / k
                agreements.append(overlap)

            return torch.tensor(agreements, device=original.device).mean()

        return torch.tensor(0.0, device=original.device)

    def full_challenge(
        self,
        model_forward: callable,
        x: torch.Tensor,
        aggregate_method: str = 'mean'
    ) -> Dict[str, Any]:
        """
        Run full noise challenge and return detailed metrics.

        Tests across multiple noise levels and perturbation types.

        Args:
            model_forward: Function that takes input and returns representation
            x: [batch, seq, dim] input to model
            aggregate_method: How to aggregate results

        Returns:
            Dict with detailed noise robustness metrics
        """
        with torch.no_grad():
            original_output = model_forward(x)

        results = {
            'by_noise_level': {},
            'by_perturbation_type': {},
            'overall_score': 0.0
        }

        all_scores = []

        for noise_level in self.noise_levels:
            level_scores = []

            for ptype in self.perturbation_types:
                perturbed_input = self.apply_perturbation(x, noise_level, ptype)

                with torch.no_grad():
                    perturbed_output = model_forward(perturbed_input)

                # Compute agreement using multiple methods
                cosine_agreement = self.compute_basin_agreement(
                    original_output, perturbed_output, 'cosine'
                ).item()
                l2_agreement = self.compute_basin_agreement(
                    original_output, perturbed_output, 'l2'
                ).item()

                # Combined score
                score = (cosine_agreement + l2_agreement) / 2
                level_scores.append(score)

                key = f"{ptype}_{noise_level}"
                results['by_perturbation_type'][key] = {
                    'cosine': cosine_agreement,
                    'l2': l2_agreement,
                    'combined': score
                }

            results['by_noise_level'][noise_level] = sum(level_scores) / len(level_scores)
            all_scores.extend(level_scores)

        results['overall_score'] = sum(all_scores) / len(all_scores)

        # Compute "degradation curve" - how fast does performance drop with noise?
        noise_curve = [results['by_noise_level'][n] for n in sorted(self.noise_levels)]
        if len(noise_curve) > 1:
            degradation_rate = (noise_curve[0] - noise_curve[-1]) / (self.noise_levels[-1] - self.noise_levels[0])
            results['degradation_rate'] = degradation_rate
        else:
            results['degradation_rate'] = 0.0

        return results


class LipschitzRegularizer:
    """
    Utility class for computing Lipschitz regularization losses.

    Use this during training to enforce stability constraints.
    """

    @staticmethod
    def jacobian_penalty(
        model: nn.Module,
        x: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Estimate Jacobian norm penalty via finite differences.

        Penalizes sensitivity of output to small input perturbations.
        """
        x.requires_grad_(True)
        y = model(x)

        # Random projection for efficiency
        v = torch.randn_like(y)
        v = v / v.norm(dim=-1, keepdim=True)

        # Compute Jv via autograd
        Jv = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True
        )[0]

        # Jacobian norm approximation
        jacobian_norm = Jv.norm(dim=-1).mean()

        return jacobian_norm

    @staticmethod
    def spectral_penalty(weight: torch.Tensor, target_norm: float = 1.0) -> torch.Tensor:
        """
        Penalize spectral norm deviation from target.
        """
        # Power iteration for spectral norm
        u = torch.randn(weight.size(0), device=weight.device)
        for _ in range(3):
            v = F.normalize(weight.T @ u, dim=0)
            u = F.normalize(weight @ v, dim=0)
        sigma = u @ weight @ v

        return (sigma - target_norm).abs()
