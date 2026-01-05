"""
Paradigm Metrics
================

Metrics for measuring how well the model embodies each quantum paradigm.
These go beyond standard LLM metrics to measure paradigm-specific behaviors.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class MetricResult:
    """Result of a metric computation"""
    name: str
    value: float
    paradigm: str
    details: Dict[str, Any] = field(default_factory=dict)


class ParadigmMetrics:
    """
    Compute paradigm-specific metrics for QLLM evaluation.

    Metrics by paradigm:

    Semantic Phase:
    - Ambiguity resolution accuracy
    - Context sensitivity
    - Phase coherence

    Retrocausal:
    - Backward reasoning consistency
    - Causal chain accuracy
    - Future-conditioned prediction improvement

    Lindblad:
    - Noise-to-signal ratio improvement
    - Attractor stability
    - Entropy reduction

    Qualia:
    - Subjective description richness
    - Qualia consistency across similar inputs
    - Multi-dimensional balance

    Emergent:
    - Complexity-to-structure ratio
    - Pattern emergence detection
    - Attractor basin coverage
    """

    def __init__(self):
        self.history: List[MetricResult] = []

    def compute_phase_coherence(
        self,
        hidden_states: torch.Tensor,
        phase_states: torch.Tensor
    ) -> MetricResult:
        """
        Measure phase coherence of representations.

        High coherence = phases are consistent and meaningful
        Low coherence = phases are random/noisy
        """
        # Compute phase alignment across sequence
        # phases should be consistent for related tokens

        # Normalize phases to unit circle
        phase_unit = torch.exp(1j * phase_states)

        # Compute coherence as mean resultant length
        mean_phase = phase_unit.mean(dim=1)  # Average across sequence
        coherence = torch.abs(mean_phase).mean()  # Magnitude of mean

        result = MetricResult(
            name='phase_coherence',
            value=coherence.item(),
            paradigm='semantic_phase',
            details={
                'mean_phase_angle': torch.angle(mean_phase.mean()).item(),
                'phase_variance': phase_states.var().item()
            }
        )
        self.history.append(result)
        return result

    def compute_context_sensitivity(
        self,
        outputs_context1: torch.Tensor,
        outputs_context2: torch.Tensor,
        expected_different: bool = True
    ) -> MetricResult:
        """
        Measure how much context changes the output.

        For ambiguous inputs, different contexts should produce different outputs.
        """
        # Compute similarity between outputs
        similarity = F.cosine_similarity(
            outputs_context1.flatten(),
            outputs_context2.flatten(),
            dim=0
        ).item()

        # If expected to be different, low similarity is good
        if expected_different:
            score = 1 - similarity
        else:
            score = similarity

        result = MetricResult(
            name='context_sensitivity',
            value=score,
            paradigm='semantic_phase',
            details={
                'raw_similarity': similarity,
                'expected_different': expected_different
            }
        )
        self.history.append(result)
        return result

    def compute_retrocausal_consistency(
        self,
        forward_predictions: torch.Tensor,
        backward_predictions: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None
    ) -> MetricResult:
        """
        Measure consistency between forward and backward reasoning.

        In retrocausal processing, forward and backward should agree
        on intermediate steps.
        """
        # Alignment between forward and backward
        alignment = F.cosine_similarity(
            forward_predictions.view(-1),
            backward_predictions.view(-1),
            dim=0
        ).item()

        details = {'forward_backward_alignment': alignment}

        # If ground truth available, measure improvement from backward
        if ground_truth is not None:
            fwd_acc = F.cosine_similarity(
                forward_predictions.view(-1),
                ground_truth.view(-1),
                dim=0
            ).item()
            bwd_acc = F.cosine_similarity(
                backward_predictions.view(-1),
                ground_truth.view(-1),
                dim=0
            ).item()
            details['forward_accuracy'] = fwd_acc
            details['backward_accuracy'] = bwd_acc
            details['backward_improvement'] = bwd_acc - fwd_acc

        result = MetricResult(
            name='retrocausal_consistency',
            value=alignment,
            paradigm='retrocausal',
            details=details
        )
        self.history.append(result)
        return result

    def compute_noise_reduction(
        self,
        noisy_input: torch.Tensor,
        clean_output: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None
    ) -> MetricResult:
        """
        Measure noise-to-signal transformation quality.

        Lindblad dynamics should reduce noise and find stable patterns.
        """
        # Compute entropy reduction
        input_entropy = self._compute_entropy(noisy_input)
        output_entropy = self._compute_entropy(clean_output)
        entropy_reduction = input_entropy - output_entropy

        # Compute stability (low variance in output)
        output_stability = 1.0 / (1.0 + clean_output.var().item())

        score = (entropy_reduction + output_stability) / 2

        details = {
            'input_entropy': input_entropy,
            'output_entropy': output_entropy,
            'entropy_reduction': entropy_reduction,
            'output_stability': output_stability
        }

        if ground_truth is not None:
            accuracy = F.cosine_similarity(
                clean_output.view(-1),
                ground_truth.view(-1),
                dim=0
            ).item()
            details['accuracy'] = accuracy

        result = MetricResult(
            name='noise_reduction',
            value=score,
            paradigm='lindblad',
            details=details
        )
        self.history.append(result)
        return result

    def compute_attractor_stability(
        self,
        representations: List[torch.Tensor],
        attractor_points: torch.Tensor
    ) -> MetricResult:
        """
        Measure how well representations converge to attractors.

        Stable attractors = representations cluster around fixed points.
        """
        # Stack representations
        reps = torch.stack(representations)

        # Find nearest attractor for each
        distances = torch.cdist(
            reps.view(len(representations), -1),
            attractor_points.view(attractor_points.size(0), -1)
        )
        min_distances, nearest = distances.min(dim=1)

        # Stability = inverse of mean distance to nearest attractor
        stability = 1.0 / (1.0 + min_distances.mean().item())

        # Coverage = how many attractors are used
        unique_attractors = len(torch.unique(nearest))
        coverage = unique_attractors / attractor_points.size(0)

        result = MetricResult(
            name='attractor_stability',
            value=stability,
            paradigm='lindblad',
            details={
                'mean_distance': min_distances.mean().item(),
                'attractor_coverage': coverage,
                'unique_attractors_used': unique_attractors
            }
        )
        self.history.append(result)
        return result

    def compute_qualia_richness(
        self,
        qualia_tensor: torch.Tensor
    ) -> MetricResult:
        """
        Measure richness of qualia representation.

        Rich qualia = diverse, balanced activation across channels.
        """
        # Should have 8 qualia channels
        # Good qualia = all channels active, balanced distribution

        # Channel activation (mean absolute value)
        channel_activations = qualia_tensor.abs().mean(dim=(0, 1))

        # Diversity = entropy of channel distribution
        probs = F.softmax(channel_activations, dim=0)
        diversity = -(probs * (probs + 1e-10).log()).sum().item()
        max_entropy = math.log(len(channel_activations))
        normalized_diversity = diversity / max_entropy

        # Balance = how evenly distributed
        balance = 1.0 - channel_activations.std().item() / (channel_activations.mean().item() + 1e-10)

        score = (normalized_diversity + max(0, balance)) / 2

        result = MetricResult(
            name='qualia_richness',
            value=score,
            paradigm='qualia',
            details={
                'channel_activations': channel_activations.tolist(),
                'diversity': normalized_diversity,
                'balance': balance
            }
        )
        self.history.append(result)
        return result

    def compute_qualia_consistency(
        self,
        qualia_similar1: torch.Tensor,
        qualia_similar2: torch.Tensor,
        qualia_different: torch.Tensor
    ) -> MetricResult:
        """
        Measure qualia consistency - similar inputs should have similar qualia.
        """
        # Similar inputs should have similar qualia
        similar_match = F.cosine_similarity(
            qualia_similar1.view(-1),
            qualia_similar2.view(-1),
            dim=0
        ).item()

        # Different input should have different qualia
        diff_from_1 = F.cosine_similarity(
            qualia_similar1.view(-1),
            qualia_different.view(-1),
            dim=0
        ).item()
        diff_from_2 = F.cosine_similarity(
            qualia_similar2.view(-1),
            qualia_different.view(-1),
            dim=0
        ).item()
        different_mismatch = 1 - (diff_from_1 + diff_from_2) / 2

        score = (similar_match + different_mismatch) / 2

        result = MetricResult(
            name='qualia_consistency',
            value=score,
            paradigm='qualia',
            details={
                'similar_match': similar_match,
                'different_mismatch': different_mismatch
            }
        )
        self.history.append(result)
        return result

    def compute_emergence_score(
        self,
        part_representations: List[torch.Tensor],
        whole_representation: torch.Tensor
    ) -> MetricResult:
        """
        Measure emergence - the whole should be more than sum of parts.

        Emergence = whole contains information not predictable from parts.
        """
        # Simple sum of parts
        parts_sum = sum(part_representations)

        # Difference between whole and sum
        emergence_vector = whole_representation - parts_sum

        # Emergence magnitude (how much the whole differs)
        emergence_magnitude = emergence_vector.norm().item()

        # Relative emergence (normalized by whole magnitude)
        relative_emergence = emergence_magnitude / (whole_representation.norm().item() + 1e-10)

        # The emergence should be structured, not random
        # Measure by entropy of the emergence vector
        emergence_entropy = self._compute_entropy(emergence_vector)
        emergence_structure = 1.0 / (1.0 + emergence_entropy)

        score = relative_emergence * emergence_structure

        result = MetricResult(
            name='emergence_score',
            value=score,
            paradigm='emergent',
            details={
                'emergence_magnitude': emergence_magnitude,
                'relative_emergence': relative_emergence,
                'emergence_structure': emergence_structure
            }
        )
        self.history.append(result)
        return result

    def compute_complexity_time(
        self,
        complexity_trajectory: List[float]
    ) -> MetricResult:
        """
        Measure complexity-time correspondence.

        Good processing = steady complexity growth (time passes meaningfully).
        """
        if len(complexity_trajectory) < 2:
            result = MetricResult(
                name='complexity_time',
                value=0.0,
                paradigm='emergent',
                details={'error': 'Need at least 2 points'}
            )
            self.history.append(result)
            return result

        # Compute complexity derivatives
        derivatives = [
            complexity_trajectory[i+1] - complexity_trajectory[i]
            for i in range(len(complexity_trajectory) - 1)
        ]

        # Mean growth rate
        mean_growth = sum(derivatives) / len(derivatives)

        # Consistency (low variance = steady time flow)
        variance = sum((d - mean_growth)**2 for d in derivatives) / len(derivatives)
        consistency = 1.0 / (1.0 + variance)

        # Positive growth is preferred (forward time)
        positive_growth = max(0, mean_growth)

        score = positive_growth * consistency

        result = MetricResult(
            name='complexity_time',
            value=score,
            paradigm='emergent',
            details={
                'mean_growth': mean_growth,
                'consistency': consistency,
                'trajectory_length': len(complexity_trajectory)
            }
        )
        self.history.append(result)
        return result

    def _compute_entropy(self, x: torch.Tensor) -> float:
        """Compute entropy of a tensor"""
        probs = F.softmax(x.view(-1), dim=0)
        entropy = -(probs * (probs + 1e-10).log()).sum()
        return entropy.item()

    def get_paradigm_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of metrics by paradigm"""
        summary = {}
        for result in self.history:
            if result.paradigm not in summary:
                summary[result.paradigm] = {}
            if result.name not in summary[result.paradigm]:
                summary[result.paradigm][result.name] = []
            summary[result.paradigm][result.name].append(result.value)

        # Compute means
        for paradigm in summary:
            for metric in summary[paradigm]:
                values = summary[paradigm][metric]
                summary[paradigm][metric] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

        return summary

    def clear_history(self):
        """Clear metric history"""
        self.history = []


class ParadigmBenchmark:
    """
    Benchmark suite for evaluating QLLM paradigm effectiveness.

    Runs structured tests for each paradigm and reports scores.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = ParadigmMetrics()

    def run_semantic_phase_benchmark(self) -> Dict[str, float]:
        """Test semantic phase disambiguation"""
        # Ambiguous sentences with different contexts
        test_cases = [
            {
                'ambiguous': 'I saw her duck',
                'context1': 'We were at the pond',
                'context2': 'A ball was flying toward her head'
            },
            {
                'ambiguous': 'The bank was steep',
                'context1': 'We were fishing by the river',
                'context2': 'Interest rates were rising'
            }
        ]

        scores = []
        for case in test_cases:
            # Get outputs for different contexts
            input1 = f"{case['context1']}. {case['ambiguous']}"
            input2 = f"{case['context2']}. {case['ambiguous']}"

            with torch.no_grad():
                out1 = self.model(
                    self.tokenizer.encode(input1, return_tensors='pt')['input_ids']
                )
                out2 = self.model(
                    self.tokenizer.encode(input2, return_tensors='pt')['input_ids']
                )

            result = self.metrics.compute_context_sensitivity(
                out1['hidden_states'],
                out2['hidden_states'],
                expected_different=True
            )
            scores.append(result.value)

        return {'semantic_phase_score': sum(scores) / len(scores)}

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete paradigm benchmark"""
        results = {}

        # Run each paradigm benchmark
        results['semantic_phase'] = self.run_semantic_phase_benchmark()
        # Add other benchmarks...

        # Overall summary
        results['summary'] = self.metrics.get_paradigm_summary()

        return results
