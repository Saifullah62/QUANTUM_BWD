"""
Emergent Computation Layers
===========================

From Daugherty's Emergent Computation paradigm:
"Constants are not fixed values but stable attractors - informational
motifs that persist across differentiation because they are resilient,
self-similar, and recursively useful."

This module implements:
1. Emergent weight initialization (finding "frozen flows")
2. Complexity-based metrics (time as derivative of complexity)
3. Attractor discovery mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import math


class ComplexityMeasure(nn.Module):
    """
    Measure computational complexity of representations.

    From the Complexity-Time Correspondence:
    dS/dt ∝ dC/dt

    Complexity C can be measured via:
    - Entropy (informational)
    - Fisher information (curvature)
    - Effective dimension (geometric)
    """

    def __init__(
        self,
        measure_type: str = "entropy"  # "entropy", "fisher", "effective_dim"
    ):
        super().__init__()
        self.measure_type = measure_type

    def entropy_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shannon entropy as complexity measure.

        Higher entropy = more complex = more "time" has passed.
        """
        # Normalize to probability distribution
        probs = F.softmax(x, dim=-1)

        # Shannon entropy
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

        return entropy

    def fisher_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fisher information as complexity measure.

        Measures curvature of the probability landscape.
        High Fisher info = sharp features = high complexity.
        """
        probs = F.softmax(x, dim=-1)

        # Approximate Fisher information via gradient magnitude
        # F = E[(∂log p / ∂θ)²]
        log_probs = (probs + 1e-10).log()

        # Compute "gradient" via finite difference
        diff = log_probs[..., 1:] - log_probs[..., :-1]
        fisher = (diff ** 2).mean(dim=-1)

        return fisher

    def effective_dimension(self, x: torch.Tensor) -> torch.Tensor:
        """
        Effective dimension as complexity measure.

        Based on participation ratio of singular values.
        """
        # Reshape for SVD: [batch, seq*dim] or similar
        flat = x.view(x.size(0), -1)

        # SVD
        try:
            _, s, _ = torch.svd(flat)
            # Participation ratio: (Σs²)² / Σs⁴
            s2 = s ** 2
            s4 = s ** 4
            eff_dim = (s2.sum(dim=-1) ** 2) / (s4.sum(dim=-1) + 1e-10)
        except:
            eff_dim = torch.ones(x.size(0), device=x.device) * x.size(-1)

        return eff_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute complexity measure"""
        if self.measure_type == "entropy":
            return self.entropy_complexity(x)
        elif self.measure_type == "fisher":
            return self.fisher_complexity(x)
        elif self.measure_type == "effective_dim":
            return self.effective_dimension(x)
        else:
            return self.entropy_complexity(x)


class EmergentInitializer:
    """
    Initialize weights by finding "frozen flows" - stable attractors.

    Instead of random initialization, this finds weight configurations
    that are stable under a complexity-preserving flow.

    The idea: good weights are those that have "crystallized" from
    the informational dynamics of the model architecture.
    """

    def __init__(
        self,
        flow_iterations: int = 100,
        learning_rate: float = 0.01,
        complexity_measure: str = "entropy",
        stability_threshold: float = 1e-4
    ):
        self.flow_iterations = flow_iterations
        self.learning_rate = learning_rate
        self.complexity_measure = ComplexityMeasure(complexity_measure)
        self.stability_threshold = stability_threshold

    def compute_flow_gradient(
        self,
        weights: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Compute gradient of the "flow" - direction toward stability.

        The flow is defined as moving toward configurations that
        maximize some measure of "useful complexity" while maintaining
        stability.
        """
        weights.requires_grad_(True)

        # Measure complexity of weight matrix
        complexity = self.complexity_measure(weights)

        # We want weights that are:
        # 1. Complex enough to be expressive
        # 2. Stable (low gradient magnitude)
        # 3. Structured (low entropy of singular values)

        # Compute SVD structure
        try:
            _, s, _ = torch.svd(weights.view(-1, weights.size(-1)))
            # Entropy of singular value distribution
            s_norm = s / (s.sum() + 1e-10)
            sv_entropy = -(s_norm * (s_norm + 1e-10).log()).sum()
        except:
            sv_entropy = torch.tensor(0.0, device=weights.device)

        # Loss: balance complexity and structure
        # We want moderate complexity, low singular value entropy
        target_complexity = 0.7 * math.log(weights.numel())
        complexity_loss = (complexity.mean() - target_complexity) ** 2
        structure_loss = sv_entropy

        total_loss = complexity_loss + 0.1 * structure_loss

        # Compute gradient
        if weights.grad is not None:
            weights.grad.zero_()
        total_loss.backward()

        grad = weights.grad.clone() if weights.grad is not None else torch.zeros_like(weights)
        weights.requires_grad_(False)

        return grad

    def find_frozen_flow(
        self,
        initial_weights: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Find stable attractor from initial weights.

        Returns:
            final_weights: Stable weight configuration
            info: Dict with convergence information
        """
        weights = initial_weights.clone()
        history = []

        for i in range(self.flow_iterations):
            # Compute flow gradient
            grad = self.compute_flow_gradient(weights, model)

            # Update weights (flow toward attractor)
            weights = weights - self.learning_rate * grad

            # Track stability
            grad_norm = grad.norm().item()
            history.append(grad_norm)

            # Check for convergence
            if grad_norm < self.stability_threshold:
                break

        info = {
            'iterations': i + 1,
            'final_grad_norm': history[-1] if history else 0,
            'converged': history[-1] < self.stability_threshold if history else False,
            'history': history
        }

        return weights, info

    def initialize_layer(
        self,
        layer: nn.Module,
        model: Optional[nn.Module] = None
    ) -> dict:
        """
        Apply emergent initialization to a layer.

        Returns info about the initialization process.
        """
        info = {}

        for name, param in layer.named_parameters():
            if param.dim() >= 2:  # Only for weight matrices
                with torch.no_grad():
                    new_weights, init_info = self.find_frozen_flow(param.data, model)
                    param.data.copy_(new_weights)
                    info[name] = init_info

        return info


class AttractorLayer(nn.Module):
    """
    Layer that maintains and uses learned attractors.

    Attractors are stable fixed points in the representation space.
    Processing involves:
    1. Finding nearest attractor(s)
    2. Evolving toward attractor(s)
    3. Combining with original representation
    """

    def __init__(
        self,
        hidden_dim: int,
        num_attractors: int = 16,
        attraction_strength: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_attractors = num_attractors
        self.attraction_strength = attraction_strength

        # Learned attractor points
        self.attractors = nn.Parameter(
            torch.randn(num_attractors, hidden_dim) * 0.1
        )

        # Attractor basin sizes (how strongly each attracts)
        self.basin_strengths = nn.Parameter(
            torch.ones(num_attractors)
        )

        # Projection to attractor space
        self.to_attractor_space = nn.Linear(hidden_dim, hidden_dim)

    def find_nearest_attractors(
        self,
        x: torch.Tensor,
        top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find nearest attractors for each position.

        Returns:
            attractor_mixture: [batch, seq, dim] weighted attractor
            weights: [batch, seq, num_attractors] attractor weights
        """
        # Project to attractor space
        x_proj = self.to_attractor_space(x)

        # Compute distances to attractors
        # x_proj: [batch, seq, dim]
        # attractors: [num_attractors, dim]
        distances = torch.cdist(x_proj, self.attractors.unsqueeze(0))  # [batch, seq, num_attractors]

        # Convert to affinities (closer = higher affinity)
        # Use softplus with min value to prevent division by very small numbers
        basin = F.softplus(self.basin_strengths) + 0.1  # Min basin size of 0.1
        # Clamp the exponent to prevent overflow
        exp_arg = -distances / basin.unsqueeze(0).unsqueeze(0)
        exp_arg = torch.clamp(exp_arg, min=-50, max=50)  # Prevent overflow
        affinities = torch.exp(exp_arg)

        # Top-k selection
        top_weights, top_indices = torch.topk(affinities, k=min(top_k, self.num_attractors), dim=-1)
        top_weights = F.softmax(top_weights, dim=-1)

        # Gather top attractors
        # [num_attractors, dim] -> [batch, seq, k, dim]
        batch_size, seq_len = x.shape[:2]
        top_attractors = self.attractors[top_indices.view(-1)].view(
            batch_size, seq_len, -1, self.hidden_dim
        )

        # Weighted sum
        attractor_mixture = (top_attractors * top_weights.unsqueeze(-1)).sum(dim=2)

        return attractor_mixture, affinities

    def forward(
        self,
        x: torch.Tensor,
        return_attractor_info: bool = False
    ) -> torch.Tensor:
        """
        Apply attractor dynamics.

        Pulls representation toward nearest attractor(s).
        """
        attractor_mixture, weights = self.find_nearest_attractors(x)

        # Evolve toward attractor
        strength = torch.sigmoid(torch.tensor(self.attraction_strength))
        output = (1 - strength) * x + strength * attractor_mixture

        if return_attractor_info:
            return output, {'attractor_weights': weights}

        return output


class ComplexityTimeLayer(nn.Module):
    """
    Layer that tracks and uses complexity-time correspondence.

    From the paper:
    "The passage of time is proportional to the rate at which
    a quantum system differentiates itself."

    This layer:
    1. Measures complexity growth
    2. Uses complexity as a "clock" signal
    3. Modulates processing based on complexity trajectory
    """

    def __init__(
        self,
        hidden_dim: int,
        complexity_measure: str = "entropy"
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.complexity = ComplexityMeasure(complexity_measure)

        # Complexity-to-modulation projection
        self.complexity_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )

        # Track complexity history (for gradient)
        self.register_buffer('prev_complexity', torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply complexity-time modulation.

        High complexity growth rate = more transformation
        Low complexity growth rate = more preservation
        """
        # Measure current complexity
        current_complexity = self.complexity(x).mean()

        # Compute complexity "velocity" (rate of change)
        complexity_delta = current_complexity - self.prev_complexity

        # Update history
        self.prev_complexity = current_complexity.detach()

        # Convert to modulation
        complexity_signal = complexity_delta.view(1, 1).expand(x.size(0), 1)
        modulation = torch.sigmoid(self.complexity_proj(complexity_signal))

        # Apply modulation
        # High complexity growth = apply more transformation
        # Low complexity growth = preserve more
        output = x * (1 + 0.1 * modulation.unsqueeze(1))

        return output


class InformationalFlowLayer(nn.Module):
    """
    Layer implementing the "thermodynamic imperative": information must flow.

    Models the universe as a dynamic informational network where
    computation happens through the flow of relational information.
    """

    def __init__(
        self,
        hidden_dim: int,
        flow_heads: int = 4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.flow_heads = flow_heads
        self.head_dim = hidden_dim // flow_heads

        # Flow direction projections
        self.flow_query = nn.Linear(hidden_dim, hidden_dim)
        self.flow_key = nn.Linear(hidden_dim, hidden_dim)
        self.flow_value = nn.Linear(hidden_dim, hidden_dim)

        # Flow gate (controls information flow rate)
        self.flow_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Conservation layer (ensures information is preserved)
        self.conservation = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply informational flow dynamics.

        Information flows from high-entropy to low-entropy regions,
        creating structure through constrained dissipation.
        """
        batch_size, seq_len, _ = x.shape

        # Compute flow directions
        Q = self.flow_query(x).view(batch_size, seq_len, self.flow_heads, self.head_dim)
        K = self.flow_key(x).view(batch_size, seq_len, self.flow_heads, self.head_dim)
        V = self.flow_value(x).view(batch_size, seq_len, self.flow_heads, self.head_dim)

        # Transpose for attention
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # Flow attention (asymmetric - information flows directionally)
        flow_matrix = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Make flow asymmetric (break symmetry for directionality)
        flow_matrix = flow_matrix - flow_matrix.transpose(-2, -1)
        flow_weights = F.softmax(flow_matrix, dim=-1)

        # Apply flow
        flowed = torch.matmul(flow_weights, V)
        flowed = flowed.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Gate the flow
        gate_input = torch.cat([x, flowed], dim=-1)
        gate = self.flow_gate(gate_input)

        # Combine with conservation
        output = self.conservation(x + gate * flowed)

        return output
