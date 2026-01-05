#!/usr/bin/env python3
"""
Qualia Calibration Benchmark
============================

Tests whether qualia channels are being used effectively.

Win condition:
1. No single channel dominates (saturation check)
2. All channels show activity (utilization check)
3. Complex/interesting sentences activate more channels (correlation check)

The qualia paradigm says: "Different aspects of experience should map to
different channels, creating a rich multidimensional representation."

Metrics:
1. Channel Utilization: Fraction of channels with significant activity
2. Channel Entropy: Evenness of channel activation distribution
3. Complexity Correlation: Do complex sentences activate more channels?

Usage:
    python scripts/benchmark_qualia.py --model-path proof_ablation_runs/proof_v2_seed42_*/all_paradigms/model.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy.stats import entropy, spearmanr

from qllm.core.config import QLLMConfig
from qllm.core.model import QLLM


class QualiaBenchmark:
    """
    Benchmark for qualia channel calibration.

    Tests whether the model uses qualia channels effectively
    to represent different aspects of experience.
    """

    # Test sentences with estimated complexity scores
    # (Simple = 1, Medium = 2, Complex = 3)
    TEST_SENTENCES = [
        # Simple (short, common words, basic concepts)
        ("The cat sat.", 1),
        ("I am here.", 1),
        ("It is hot.", 1),
        ("She runs fast.", 1),
        ("Birds fly high.", 1),

        # Medium (longer, more abstract)
        ("The quick brown fox jumps over the lazy dog.", 2),
        ("Knowledge brings power and responsibility.", 2),
        ("Music fills the air with beautiful sounds.", 2),
        ("The river flows gently through the valley.", 2),
        ("Time passes quickly when you are busy.", 2),

        # Complex (abstract, philosophical, nested)
        ("The concept of infinity challenges our finite understanding.", 3),
        ("Consciousness emerges from the complex interplay of neurons.", 3),
        ("The quantum nature of reality defies classical intuition.", 3),
        ("Love transcends the boundaries of time and space.", 3),
        ("Truth exists independently of our ability to perceive it.", 3),

        # Very complex (paradoxes, deep philosophy)
        ("If nothing exists, then something exists to observe nothing.", 4),
        ("The present moment is the only reality we can truly know.", 4),
        ("Free will may be an illusion created by deterministic processes.", 4),
        ("The universe contemplates itself through conscious beings.", 4),
        ("Meaning is constructed not discovered in the fabric of existence.", 4),
    ]

    def __init__(
        self,
        model: QLLM,
        tokenizer_vocab: Dict[str, int],
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = tokenizer_vocab
        self.device = device
        self.max_length = model.config.max_seq_length
        self.num_channels = model.config.num_qualia_channels

    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple whitespace tokenization."""
        tokens = []
        for word in text.lower().split():
            tokens.append(self.vocab.get(word, 1))

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)

    def get_qualia_values(self, text: str) -> np.ndarray:
        """
        Get qualia channel activations for a sentence.

        Returns: [num_channels] array of activation values
        """
        input_ids = self._tokenize(text).unsqueeze(0).to(self.device)
        attention_mask = (input_ids != 0).float()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Extract qualia values
        qualia_info = outputs.get('qualia_info', {})
        if qualia_info is None:
            qualia_info = {}

        # Try qualia_tensor first (combined [batch, seq, channels])
        qualia_tensor = qualia_info.get('qualia_tensor')
        if qualia_tensor is not None and isinstance(qualia_tensor, torch.Tensor):
            # Pool over sequence dimension
            seq_len = (input_ids != 0).sum().item()
            qualia_values = qualia_tensor[0, :seq_len].mean(dim=0)
            return qualia_values.cpu().numpy()

        # Fallback: try qualia_values dict
        qualia_values = qualia_info.get('qualia_values')
        if qualia_values is not None:
            if isinstance(qualia_values, dict):
                # Dict of channel_name -> tensor [batch, seq, 1]
                channel_values = []
                seq_len = (input_ids != 0).sum().item()
                for name in sorted(qualia_values.keys()):
                    val = qualia_values[name]
                    if isinstance(val, torch.Tensor):
                        # Pool over sequence, squeeze last dim
                        pooled = val[0, :seq_len, 0].mean().item()
                        channel_values.append(pooled)
                if channel_values:
                    return np.array(channel_values)
            elif isinstance(qualia_values, torch.Tensor):
                # Tensor directly
                if len(qualia_values.shape) == 3:
                    seq_len = (input_ids != 0).sum().item()
                    qualia_values = qualia_values[0, :seq_len].mean(dim=0)
                elif len(qualia_values.shape) == 2:
                    qualia_values = qualia_values[0]
                return qualia_values.cpu().numpy()

        # Fallback: return zeros
        return np.zeros(self.num_channels)

    def compute_channel_metrics(
        self,
        qualia_values: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute metrics for a single qualia activation vector.
        """
        # Normalize to [0, 1] for analysis
        q_min, q_max = qualia_values.min(), qualia_values.max()
        if q_max - q_min > 0:
            q_norm = (qualia_values - q_min) / (q_max - q_min)
        else:
            q_norm = np.zeros_like(qualia_values)

        # 1. Channel utilization: fraction of channels with significant activity
        threshold = 0.1  # Consider active if > 10% of max
        active_channels = (q_norm > threshold).sum()
        utilization = active_channels / len(qualia_values)

        # 2. Channel entropy: evenness of distribution
        # Add small epsilon to avoid log(0)
        q_positive = np.abs(qualia_values) + 1e-8
        q_dist = q_positive / q_positive.sum()
        channel_entropy = entropy(q_dist)
        max_entropy = np.log(len(qualia_values))  # Max possible entropy
        normalized_entropy = channel_entropy / max_entropy if max_entropy > 0 else 0

        # 3. Dominance: how much does the top channel dominate?
        sorted_q = np.sort(q_positive)[::-1]
        top_dominance = sorted_q[0] / sorted_q.sum() if sorted_q.sum() > 0 else 0

        # 4. Activation magnitude: overall strength
        magnitude = np.abs(qualia_values).mean()

        return {
            'utilization': utilization,
            'entropy': normalized_entropy,
            'top_dominance': top_dominance,
            'magnitude': magnitude,
            'active_channels': int(active_channels)
        }

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run full qualia calibration benchmark.
        """
        results = {
            'per_sentence': [],
            'per_complexity': {},
            'overall': {}
        }

        # Collect results for each sentence
        all_complexities = []
        all_metrics = []
        all_qualia = []

        for sentence, complexity in self.TEST_SENTENCES:
            qualia = self.get_qualia_values(sentence)
            metrics = self.compute_channel_metrics(qualia)
            metrics['complexity'] = complexity
            metrics['sentence'] = sentence

            results['per_sentence'].append({
                'sentence': sentence,
                'complexity': complexity,
                'qualia_values': qualia.tolist(),
                'metrics': metrics
            })

            all_complexities.append(complexity)
            all_metrics.append(metrics)
            all_qualia.append(qualia)

        # Aggregate by complexity level
        for level in [1, 2, 3, 4]:
            level_metrics = [m for m in all_metrics if m['complexity'] == level]
            if level_metrics:
                results['per_complexity'][f'level_{level}'] = {
                    'mean_utilization': np.mean([m['utilization'] for m in level_metrics]),
                    'mean_entropy': np.mean([m['entropy'] for m in level_metrics]),
                    'mean_dominance': np.mean([m['top_dominance'] for m in level_metrics]),
                    'mean_magnitude': np.mean([m['magnitude'] for m in level_metrics]),
                    'mean_active_channels': np.mean([m['active_channels'] for m in level_metrics]),
                    'count': len(level_metrics)
                }

        # Overall metrics
        all_utils = [m['utilization'] for m in all_metrics]
        all_entropies = [m['entropy'] for m in all_metrics]
        all_dominances = [m['top_dominance'] for m in all_metrics]
        all_magnitudes = [m['magnitude'] for m in all_metrics]
        all_active = [m['active_channels'] for m in all_metrics]

        # Compute complexity correlation
        # Does magnitude increase with complexity?
        magnitude_corr, magnitude_p = spearmanr(all_complexities, all_magnitudes)

        # Do more channels activate with complexity?
        active_corr, active_p = spearmanr(all_complexities, all_active)

        # Does entropy increase with complexity?
        entropy_corr, entropy_p = spearmanr(all_complexities, all_entropies)

        results['overall'] = {
            'mean_utilization': np.mean(all_utils),
            'std_utilization': np.std(all_utils),
            'mean_entropy': np.mean(all_entropies),
            'std_entropy': np.std(all_entropies),
            'mean_dominance': np.mean(all_dominances),
            'std_dominance': np.std(all_dominances),
            'mean_magnitude': np.mean(all_magnitudes),
            'std_magnitude': np.std(all_magnitudes),
            'mean_active_channels': np.mean(all_active),
            'num_channels': self.num_channels,
            'complexity_magnitude_correlation': float(magnitude_corr),
            'complexity_magnitude_pvalue': float(magnitude_p),
            'complexity_active_correlation': float(active_corr),
            'complexity_active_pvalue': float(active_p),
            'complexity_entropy_correlation': float(entropy_corr),
            'complexity_entropy_pvalue': float(entropy_p),
            'num_sentences': len(self.TEST_SENTENCES)
        }

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("  QUALIA CALIBRATION BENCHMARK SUMMARY")
        print("=" * 60)

        overall = results['overall']
        print(f"\nOverall Metrics ({overall['num_sentences']} sentences, {overall['num_channels']} channels):")
        print(f"  Mean Utilization: {overall['mean_utilization']:.1%}")
        print(f"  Mean Entropy: {overall['mean_entropy']:.4f}")
        print(f"  Mean Dominance: {overall['mean_dominance']:.1%}")
        print(f"  Mean Active Channels: {overall['mean_active_channels']:.1f}/{overall['num_channels']}")

        # Per-complexity breakdown
        print("\n" + "-" * 60)
        print("BY COMPLEXITY LEVEL")
        print("-" * 60)

        for level in [1, 2, 3, 4]:
            key = f'level_{level}'
            if key in results['per_complexity']:
                data = results['per_complexity'][key]
                labels = {1: 'Simple', 2: 'Medium', 3: 'Complex', 4: 'V.Complex'}
                print(f"\n{labels[level]} ({data['count']} sentences):")
                print(f"  Active channels: {data['mean_active_channels']:.1f}")
                print(f"  Magnitude: {data['mean_magnitude']:.4f}")
                print(f"  Entropy: {data['mean_entropy']:.4f}")

        # Correlations
        print("\n" + "-" * 60)
        print("COMPLEXITY CORRELATIONS")
        print("-" * 60)
        print(f"  Complexity vs Magnitude: r={overall['complexity_magnitude_correlation']:.3f} "
              f"(p={overall['complexity_magnitude_pvalue']:.3f})")
        print(f"  Complexity vs Active Channels: r={overall['complexity_active_correlation']:.3f} "
              f"(p={overall['complexity_active_pvalue']:.3f})")
        print(f"  Complexity vs Entropy: r={overall['complexity_entropy_correlation']:.3f} "
              f"(p={overall['complexity_entropy_pvalue']:.3f})")

        # Channel visualization
        print("\n" + "-" * 60)
        print("CHANNEL ACTIVITY (sample sentences)")
        print("-" * 60)

        # Show a few examples
        for i, item in enumerate(results['per_sentence'][:3] + results['per_sentence'][-2:]):
            complexity_label = {1: 'Simple', 2: 'Medium', 3: 'Complex', 4: 'V.Complex'}[item['complexity']]
            print(f"\n{complexity_label}: \"{item['sentence'][:40]}...\"")
            qualia = np.array(item['qualia_values'])
            for j, val in enumerate(qualia):
                bar = "#" * int(abs(val) * 20) if abs(val) > 0.01 else "."
                print(f"  Ch{j}: {bar:20s} {val:.3f}")

        # Win condition assessment
        print("\n" + "-" * 60)
        print("WIN CONDITION ASSESSMENT")
        print("-" * 60)

        # Thresholds for "passing"
        util_pass = overall['mean_utilization'] > 0.3  # At least 30% of channels active
        entropy_pass = overall['mean_entropy'] > 0.5  # Reasonably even distribution
        dom_pass = overall['mean_dominance'] < 0.7  # No single channel dominates > 70%
        corr_pass = overall['complexity_magnitude_correlation'] > 0  # Positive correlation

        print(f"  Utilization > 30%: {'PASS' if util_pass else 'FAIL'} ({overall['mean_utilization']:.1%})")
        print(f"  Entropy > 0.5: {'PASS' if entropy_pass else 'FAIL'} ({overall['mean_entropy']:.3f})")
        print(f"  Dominance < 70%: {'PASS' if dom_pass else 'FAIL'} ({overall['mean_dominance']:.1%})")
        print(f"  Complexity corr > 0: {'PASS' if corr_pass else 'FAIL'} "
              f"(r={overall['complexity_magnitude_correlation']:.3f})")

        if util_pass and entropy_pass and dom_pass and corr_pass:
            print("\n  >>> WIN CONDITION MET <<<")
        else:
            print("\n  >>> WIN CONDITION NOT MET <<<")


def build_vocab_from_data(data_path: str, vocab_size: int = 1000) -> Dict[str, int]:
    """Build vocabulary from training data."""
    word_counts = {}

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            text = ex['input_text'] + ' ' + ex['output_text']
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

    vocab = {'<pad>': 0, '<unk>': 1}
    for i, (word, _) in enumerate(sorted_words[:vocab_size - 2]):
        vocab[word] = i + 2

    return vocab


def load_model_from_checkpoint(
    checkpoint_path: str,
    data_path: str = './data/pilot_paradigm_data.jsonl'
) -> Tuple[QLLM, Dict[str, int]]:
    """Load model from checkpoint."""

    vocab = build_vocab_from_data(data_path)
    print(f"  Built vocab with {len(vocab)} words")

    config = QLLMConfig(
        vocab_size=len(vocab),
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        intermediate_dim=256,
        max_seq_length=128,
        use_semantic_phase=True,
        semantic_phase_dim=128,
        use_retrocausal_attention=True,
        retrocausal_layers=[0, 2],
        use_lindblad_layers=True,
        lindblad_every_n_layers=2,
        use_qualia_output=True,
        num_qualia_channels=8,
        use_emergent_init=True
    )

    model = QLLM(config)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    model_state = model.state_dict()
    filtered_state = {}
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value

    model.load_state_dict(filtered_state, strict=False)

    return model, vocab


def main():
    parser = argparse.ArgumentParser(description="Run qualia calibration benchmark")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='./data/pilot_paradigm_data.jsonl',
                        help='Path to training data (for vocab)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results JSON')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')

    args = parser.parse_args()

    print("=" * 60)
    print("  QUALIA CALIBRATION BENCHMARK")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model, vocab = load_model_from_checkpoint(args.model_path, args.data)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Run benchmark
    benchmark = QualiaBenchmark(model, vocab, device)
    results = benchmark.run_benchmark()

    # Print summary
    benchmark.print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_dir = Path(args.model_path).parent
        output_path = model_dir / "qualia_benchmark.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
