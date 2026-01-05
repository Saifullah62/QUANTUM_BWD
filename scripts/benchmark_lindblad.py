#!/usr/bin/env python3
"""
Lindblad Degradation Curve Benchmark
=====================================

Tests whether the model maintains coherent representations under noise injection.

Win condition: At noise level X, the model's internal coherence metric
must remain above threshold Y. The degradation curve should be smooth,
not cliff-like.

The Lindblad paradigm says: "Noise is friend, not foe. The system should
learn representations that are INVARIANT to reasonable noise levels."

Metrics:
1. Coherence Score: Cosine similarity between clean and noisy embeddings
2. Degradation Rate: How fast coherence drops with increasing noise
3. Critical Noise Level: Noise level where coherence drops below 0.9

Usage:
    python scripts/benchmark_lindblad.py --model-path proof_ablation_runs/proof_v2_seed42_*/all_paradigms/model.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from qllm.core.config import QLLMConfig
from qllm.core.model import QLLM


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""
    noise_type: str  # 'token_drop', 'token_swap', 'gaussian', 'mask'
    noise_level: float  # Probability or magnitude


class LindbladBenchmark:
    """
    Benchmark for noise invariance (Lindblad paradigm).

    Tests model's ability to maintain coherent representations
    despite input noise.
    """

    # Noise levels to test (probability/magnitude)
    NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    # Test sentences (semantically diverse)
    TEST_SENTENCES = [
        "The cat sat on the mat.",
        "Science advances through careful observation.",
        "Music fills the room with harmony.",
        "The river flows to the sea.",
        "Knowledge is power in the modern age.",
        "Birds fly south for the winter.",
        "The sun rises in the east.",
        "Books open doors to new worlds.",
        "Time waits for no one.",
        "Nature heals all wounds eventually.",
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
        self.id_to_word = {v: k for k, v in tokenizer_vocab.items()}
        self.device = device
        self.max_length = model.config.max_seq_length

    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple whitespace tokenization."""
        tokens = []
        for word in text.lower().split():
            tokens.append(self.vocab.get(word, 1))  # 1 = <unk>

        # Pad/truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)

    def inject_noise(
        self,
        input_ids: torch.Tensor,
        noise_type: str,
        noise_level: float
    ) -> torch.Tensor:
        """
        Inject noise into token sequence.

        Args:
            input_ids: [seq_len] tensor of token IDs
            noise_type: Type of noise to apply
            noise_level: Probability/magnitude of noise

        Returns:
            Noisy token sequence
        """
        ids = input_ids.clone().tolist()
        non_pad_length = sum(1 for x in ids if x != 0)

        if noise_type == 'token_drop':
            # Drop tokens with probability noise_level
            result = []
            for i, tok in enumerate(ids[:non_pad_length]):
                if random.random() > noise_level:
                    result.append(tok)
            # Pad back
            result = result + [0] * (self.max_length - len(result))
            return torch.tensor(result[:self.max_length], dtype=torch.long)

        elif noise_type == 'token_swap':
            # Swap adjacent tokens with probability noise_level
            for i in range(non_pad_length - 1):
                if random.random() < noise_level:
                    ids[i], ids[i+1] = ids[i+1], ids[i]
            return torch.tensor(ids, dtype=torch.long)

        elif noise_type == 'mask':
            # Replace tokens with <unk> (id=1) with probability noise_level
            for i in range(non_pad_length):
                if random.random() < noise_level:
                    ids[i] = 1
            return torch.tensor(ids, dtype=torch.long)

        elif noise_type == 'random_replace':
            # Replace tokens with random vocab tokens
            vocab_size = len(self.vocab)
            for i in range(non_pad_length):
                if random.random() < noise_level:
                    ids[i] = random.randint(2, vocab_size - 1)
            return torch.tensor(ids, dtype=torch.long)

        else:
            return input_ids

    def get_embedding(self, input_ids: torch.Tensor) -> np.ndarray:
        """
        Get embedding representation for token sequence.

        Returns combined real + imaginary embedding.
        """
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = (input_ids != 0).float()

        with torch.no_grad():
            # Get phase embeddings
            if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'real_embedding'):
                real_embed = self.model.embeddings.real_embedding(input_ids)
                imag_embed = self.model.embeddings.imag_embedding(input_ids)

                # Pool over sequence (mean of non-padding)
                seq_len = (input_ids != 0).sum().item()
                real_pooled = real_embed[0, :seq_len].mean(dim=0).cpu().numpy()
                imag_pooled = imag_embed[0, :seq_len].mean(dim=0).cpu().numpy()

                return np.concatenate([real_pooled, imag_pooled])
            else:
                # Fallback: use output embeddings
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.get('hidden_states', outputs['logits'])
                seq_len = (input_ids != 0).sum().item()
                return hidden[0, :seq_len].mean(dim=0).cpu().numpy()

    def compute_coherence(self, clean_emb: np.ndarray, noisy_emb: np.ndarray) -> float:
        """
        Compute coherence score between clean and noisy embeddings.

        Uses cosine similarity.
        """
        dot = np.dot(clean_emb, noisy_emb)
        norm_clean = np.linalg.norm(clean_emb)
        norm_noisy = np.linalg.norm(noisy_emb)

        if norm_clean == 0 or norm_noisy == 0:
            return 0.0

        return dot / (norm_clean * norm_noisy)

    def run_degradation_curve(
        self,
        sentence: str,
        noise_type: str = 'mask',
        num_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Run degradation curve for a single sentence.

        Returns coherence scores at each noise level.
        """
        input_ids = self._tokenize(sentence)
        clean_emb = self.get_embedding(input_ids)

        results = {
            'sentence': sentence,
            'noise_type': noise_type,
            'noise_levels': self.NOISE_LEVELS,
            'coherence_scores': [],
            'coherence_stds': []
        }

        for noise_level in self.NOISE_LEVELS:
            if noise_level == 0.0:
                results['coherence_scores'].append(1.0)
                results['coherence_stds'].append(0.0)
                continue

            trial_scores = []
            for _ in range(num_trials):
                noisy_ids = self.inject_noise(input_ids, noise_type, noise_level)
                noisy_emb = self.get_embedding(noisy_ids)
                score = self.compute_coherence(clean_emb, noisy_emb)
                trial_scores.append(score)

            results['coherence_scores'].append(np.mean(trial_scores))
            results['coherence_stds'].append(np.std(trial_scores))

        return results

    def compute_degradation_metrics(
        self,
        coherence_scores: List[float]
    ) -> Dict[str, float]:
        """
        Compute summary metrics from degradation curve.
        """
        noise_levels = self.NOISE_LEVELS

        # Area under curve (higher = more robust)
        auc = np.trapz(coherence_scores, noise_levels)

        # Degradation rate (slope of coherence vs noise)
        if len(noise_levels) > 1:
            slope, _ = np.polyfit(noise_levels, coherence_scores, 1)
            degradation_rate = -slope  # Positive = faster degradation
        else:
            degradation_rate = 0

        # Critical noise level (where coherence drops below 0.9)
        critical_level = None
        for i, score in enumerate(coherence_scores):
            if score < 0.9:
                critical_level = noise_levels[i]
                break
        if critical_level is None:
            critical_level = noise_levels[-1]  # Never dropped below 0.9

        # Coherence at 20% noise (key metric)
        idx_20 = noise_levels.index(0.2) if 0.2 in noise_levels else len(noise_levels) // 2
        coherence_at_20 = coherence_scores[idx_20]

        return {
            'auc': auc,
            'degradation_rate': degradation_rate,
            'critical_noise_level': critical_level,
            'coherence_at_20pct': coherence_at_20
        }

    def run_benchmark(
        self,
        noise_types: List[str] = ['mask', 'token_swap', 'random_replace'],
        num_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Run full Lindblad degradation benchmark.
        """
        results = {
            'per_sentence': {},
            'per_noise_type': {},
            'overall': {}
        }

        all_curves = []
        all_metrics = []

        for noise_type in noise_types:
            print(f"\nTesting noise type: {noise_type}")
            noise_curves = []

            for i, sentence in enumerate(self.TEST_SENTENCES):
                curve = self.run_degradation_curve(
                    sentence,
                    noise_type=noise_type,
                    num_trials=num_trials
                )
                noise_curves.append(curve)

                # Store per-sentence results
                key = f"sentence_{i}_{noise_type}"
                results['per_sentence'][key] = curve

            # Aggregate across sentences for this noise type
            avg_coherence = np.mean(
                [c['coherence_scores'] for c in noise_curves],
                axis=0
            )
            std_coherence = np.std(
                [c['coherence_scores'] for c in noise_curves],
                axis=0
            )

            metrics = self.compute_degradation_metrics(avg_coherence.tolist())

            results['per_noise_type'][noise_type] = {
                'avg_coherence': avg_coherence.tolist(),
                'std_coherence': std_coherence.tolist(),
                'noise_levels': self.NOISE_LEVELS,
                'metrics': metrics
            }

            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  Degradation rate: {metrics['degradation_rate']:.4f}")
            print(f"  Critical noise level: {metrics['critical_noise_level']:.2f}")
            print(f"  Coherence @ 20%: {metrics['coherence_at_20pct']:.4f}")

            all_curves.append(avg_coherence)
            all_metrics.append(metrics)

        # Overall metrics (average across noise types)
        results['overall'] = {
            'mean_auc': np.mean([m['auc'] for m in all_metrics]),
            'mean_degradation_rate': np.mean([m['degradation_rate'] for m in all_metrics]),
            'mean_critical_level': np.mean([m['critical_noise_level'] for m in all_metrics]),
            'mean_coherence_at_20pct': np.mean([m['coherence_at_20pct'] for m in all_metrics]),
            'noise_types_tested': noise_types,
            'num_sentences': len(self.TEST_SENTENCES),
            'num_trials': num_trials
        }

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("  LINDBLAD DEGRADATION BENCHMARK SUMMARY")
        print("=" * 60)

        overall = results['overall']
        print(f"\nOverall Metrics ({overall['num_sentences']} sentences, {len(overall['noise_types_tested'])} noise types):")
        print(f"  Mean AUC: {overall['mean_auc']:.4f}")
        print(f"  Mean Degradation Rate: {overall['mean_degradation_rate']:.4f}")
        print(f"  Mean Critical Noise Level: {overall['mean_critical_level']:.2f}")
        print(f"  Mean Coherence @ 20%: {overall['mean_coherence_at_20pct']:.4f}")

        # Degradation curve visualization (ASCII)
        print("\n" + "-" * 60)
        print("DEGRADATION CURVES (by noise type)")
        print("-" * 60)

        for noise_type in overall['noise_types_tested']:
            data = results['per_noise_type'][noise_type]
            print(f"\n{noise_type}:")
            for i, (level, score) in enumerate(zip(data['noise_levels'], data['avg_coherence'])):
                bar = "#" * int(score * 40)
                print(f"  {level:.2f}: {bar} {score:.3f}")

        # Win condition assessment
        print("\n" + "-" * 60)
        print("WIN CONDITION ASSESSMENT")
        print("-" * 60)

        # Thresholds for "passing"
        auc_pass = overall['mean_auc'] > 0.35  # ~70% average coherence
        crit_pass = overall['mean_critical_level'] > 0.15  # Stays above 0.9 until 15%+ noise
        c20_pass = overall['mean_coherence_at_20pct'] > 0.85  # 85% coherent at 20% noise

        print(f"  AUC > 0.35: {'PASS' if auc_pass else 'FAIL'} ({overall['mean_auc']:.3f})")
        print(f"  Critical > 15%: {'PASS' if crit_pass else 'FAIL'} ({overall['mean_critical_level']:.1%})")
        print(f"  Coherence@20% > 85%: {'PASS' if c20_pass else 'FAIL'} ({overall['mean_coherence_at_20pct']:.1%})")

        if auc_pass and crit_pass and c20_pass:
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
    parser = argparse.ArgumentParser(description="Run Lindblad degradation benchmark")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='./data/pilot_paradigm_data.jsonl',
                        help='Path to training data (for vocab)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results JSON')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--num-trials', type=int, default=10,
                        help='Number of trials per noise level')

    args = parser.parse_args()

    print("=" * 60)
    print("  LINDBLAD DEGRADATION CURVE BENCHMARK")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model, vocab = load_model_from_checkpoint(args.model_path, args.data)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Run benchmark
    benchmark = LindbladBenchmark(model, vocab, device)
    results = benchmark.run_benchmark(num_trials=args.num_trials)

    # Print summary
    benchmark.print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_dir = Path(args.model_path).parent
        output_path = model_dir / "lindblad_benchmark.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
