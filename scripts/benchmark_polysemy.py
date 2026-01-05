#!/usr/bin/env python3
"""
Polysemy Win-Condition Benchmark
================================

Tests whether the model correctly differentiates word senses via phase embeddings.

Win condition: Given "bank (money)" vs "bank (river)" in context, the model
should produce measurably different phase vectors that cluster by sense.

Metrics:
1. Sense Cluster Separation (SCS): Are same-sense examples closer than cross-sense?
2. Sense Retrieval Accuracy (SRA): Given a query, can we retrieve same-sense examples?
3. Phase Angle Divergence (PAD): Do different senses occupy different phase regions?

Usage:
    python scripts/benchmark_polysemy.py --model-path proof_ablation_runs/proof_v2_seed42_*/all_paradigms/model.pt
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
from collections import defaultdict
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

from qllm.core.config import QLLMConfig
from qllm.core.model import QLLM


class PolysemyBenchmark:
    """
    Benchmark for polysemy sense discrimination via phase embeddings.
    """

    # Test cases: word -> {sense_label: [example sentences]}
    # The target word appears in each sentence
    POLYSEMY_TEST_CASES = {
        'bank': {
            'financial': [
                "I went to the bank to deposit my paycheck.",
                "The bank approved my loan application yesterday.",
                "She works at a major investment bank downtown.",
                "The bank charges high fees for international transfers.",
                "My bank account was frozen due to suspicious activity.",
            ],
            'river': [
                "We had a picnic on the bank of the river.",
                "The fisherman sat on the muddy bank all morning.",
                "Erosion along the bank threatened the nearby houses.",
                "Wild flowers grew along the bank in spring.",
                "The canoe was pulled up onto the sandy bank.",
            ]
        },
        'wave': {
            'ocean': [
                "A huge wave crashed against the shore.",
                "Surfers waited for the perfect wave to ride.",
                "The wave knocked him off his feet.",
                "Sound of the wave was calming and peaceful.",
                "A rogue wave capsized the small boat.",
            ],
            'gesture': [
                "She gave me a friendly wave from across the street.",
                "He answered with a dismissive wave of his hand.",
                "The queen's wave delighted the crowd.",
                "A wave goodbye was all she could manage.",
                "He acknowledged them with a casual wave.",
            ]
        },
        'bark': {
            'tree': [
                "The bark of the old oak was rough and cracked.",
                "She peeled the bark from the birch tree.",
                "Insects had damaged the bark of the apple tree.",
                "The bark protects the tree from disease.",
                "Cinnamon comes from the bark of a tropical tree.",
            ],
            'dog': [
                "The dog's bark echoed through the empty house.",
                "A sharp bark warned us someone was approaching.",
                "His bark was worse than his bite.",
                "The bark of the guard dog scared away intruders.",
                "She recognized her puppy's distinctive bark.",
            ]
        },
        'spring': {
            'season': [
                "Flowers bloom beautifully in spring.",
                "Spring is my favorite time of year.",
                "The spring weather was warm and pleasant.",
                "Birds return from migration every spring.",
                "Spring cleaning is an annual tradition.",
            ],
            'water': [
                "Fresh water bubbled up from the spring.",
                "The spring provided water for the village.",
                "A natural hot spring attracted tourists.",
                "The spring was the river's source.",
                "Crystal clear water flowed from the mountain spring.",
            ],
            'coil': [
                "The spring in the mattress was broken.",
                "He compressed the spring and released it.",
                "A metal spring held the mechanism together.",
                "The spring bounced back to its original shape.",
                "Watch springs are incredibly precise.",
            ]
        },
        'light': {
            'illumination': [
                "The light from the lamp was too dim.",
                "Sunlight is the best natural light for photography.",
                "Turn on the light so I can read.",
                "The light in the room flickered and went out.",
                "Morning light streamed through the window.",
            ],
            'weight': [
                "This suitcase is surprisingly light.",
                "She preferred light meals in summer.",
                "The light fabric was perfect for hot weather.",
                "Feathers are light as air.",
                "A light touch was all it needed.",
            ]
        },
        'match': {
            'fire': [
                "He struck a match to light the candle.",
                "The match flickered and died in the wind.",
                "Keep matches away from children.",
                "A single match started the forest fire.",
                "She lit the gas stove with a match.",
            ],
            'competition': [
                "The tennis match lasted three hours.",
                "It was an exciting match between rivals.",
                "The match ended in a draw.",
                "Chess match between grandmasters.",
                "The soccer match attracted thousands of fans.",
            ],
            'pair': [
                "These socks don't match at all.",
                "Find a match for this puzzle piece.",
                "Her shoes were a perfect match for her dress.",
                "The colors match beautifully together.",
                "He was looking for his match in life.",
            ]
        },
    }

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

    def _find_word_position(self, input_ids: torch.Tensor, target_word: str) -> int:
        """Find position of target word in token sequence."""
        target_id = self.vocab.get(target_word.lower(), -1)

        if target_id == -1:
            # Try to find word containing target
            for i, tok_id in enumerate(input_ids.tolist()):
                if tok_id in self.id_to_word:
                    word = self.id_to_word[tok_id]
                    if target_word.lower() in word:
                        return i
            return 0  # Default to first position

        positions = (input_ids == target_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            return positions[0].item()
        return 0

    def extract_phase_embedding(
        self,
        text: str,
        target_word: str
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract phase embedding for target word in context.

        Returns:
            real_embed: Real part of embedding at target position
            imag_embed: Imaginary part of embedding at target position
            phase_angle: Mean phase angle at target position
        """
        input_ids = self._tokenize(text).unsqueeze(0).to(self.device)
        attention_mask = (input_ids != 0).float()

        with torch.no_grad():
            # Get embeddings from model
            if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'real_embedding'):
                real_embed = self.model.embeddings.real_embedding(input_ids)
                imag_embed = self.model.embeddings.imag_embedding(input_ids)
            else:
                # Fallback: just use regular embeddings
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                real_embed = outputs.get('hidden_states', outputs['logits'])
                imag_embed = torch.zeros_like(real_embed)

        # Find target word position
        pos = self._find_word_position(input_ids[0], target_word)

        # Extract embeddings at target position
        real_vec = real_embed[0, pos].cpu().numpy()
        imag_vec = imag_embed[0, pos].cpu().numpy()

        # Compute phase angle
        phase_angles = np.arctan2(imag_vec, real_vec)
        mean_phase = np.mean(phase_angles)

        return real_vec, imag_vec, mean_phase

    def compute_sense_embeddings(
        self,
        word: str,
        senses: Dict[str, List[str]]
    ) -> Dict[str, List[Dict]]:
        """
        Compute embeddings for all senses of a word.

        Returns dict: sense_label -> list of {real, imag, phase, text}
        """
        sense_embeddings = {}

        for sense_label, sentences in senses.items():
            sense_embeddings[sense_label] = []

            for sentence in sentences:
                real_vec, imag_vec, phase = self.extract_phase_embedding(sentence, word)
                sense_embeddings[sense_label].append({
                    'real': real_vec,
                    'imag': imag_vec,
                    'phase': phase,
                    'text': sentence
                })

        return sense_embeddings

    def compute_cluster_separation(
        self,
        sense_embeddings: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """
        Compute sense cluster separation metrics.

        Returns:
            intra_sense_dist: Average distance within same sense
            inter_sense_dist: Average distance between different senses
            separation_ratio: inter/intra (higher = better separation)
        """
        # Collect all embeddings with labels
        all_embeddings = []
        all_labels = []
        label_map = {}

        for i, (sense, embeddings) in enumerate(sense_embeddings.items()):
            label_map[i] = sense
            for emb in embeddings:
                # Use combined real+imag as feature vector
                combined = np.concatenate([emb['real'], emb['imag']])
                all_embeddings.append(combined)
                all_labels.append(i)

        all_embeddings = np.array(all_embeddings)
        all_labels = np.array(all_labels)

        # Compute intra-sense distances
        intra_distances = []
        for label in np.unique(all_labels):
            mask = all_labels == label
            cluster = all_embeddings[mask]
            if len(cluster) > 1:
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        dist = np.linalg.norm(cluster[i] - cluster[j])
                        intra_distances.append(dist)

        # Compute inter-sense distances
        inter_distances = []
        unique_labels = np.unique(all_labels)
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                cluster1 = all_embeddings[all_labels == label1]
                cluster2 = all_embeddings[all_labels == label2]
                for e1 in cluster1:
                    for e2 in cluster2:
                        dist = np.linalg.norm(e1 - e2)
                        inter_distances.append(dist)

        intra_mean = np.mean(intra_distances) if intra_distances else 0
        inter_mean = np.mean(inter_distances) if inter_distances else 0

        # Compute silhouette score if we have enough samples
        silhouette = 0.0
        if len(all_embeddings) >= 4 and len(np.unique(all_labels)) >= 2:
            try:
                silhouette = silhouette_score(all_embeddings, all_labels)
            except:
                pass

        return {
            'intra_sense_distance': intra_mean,
            'inter_sense_distance': inter_mean,
            'separation_ratio': inter_mean / (intra_mean + 1e-8),
            'silhouette_score': silhouette
        }

    def compute_phase_divergence(
        self,
        sense_embeddings: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """
        Compute phase angle divergence between senses.

        Returns metrics on whether different senses occupy different phase regions.
        """
        sense_phases = {}
        for sense, embeddings in sense_embeddings.items():
            phases = [emb['phase'] for emb in embeddings]
            sense_phases[sense] = {
                'mean': np.mean(phases),
                'std': np.std(phases),
                'phases': phases
            }

        # Compute pairwise phase differences
        sense_labels = list(sense_phases.keys())
        phase_diffs = []

        for i, s1 in enumerate(sense_labels):
            for s2 in sense_labels[i+1:]:
                diff = abs(sense_phases[s1]['mean'] - sense_phases[s2]['mean'])
                # Normalize to [0, pi]
                diff = min(diff, 2*np.pi - diff)
                phase_diffs.append({
                    'senses': (s1, s2),
                    'phase_diff': diff
                })

        mean_diff = np.mean([d['phase_diff'] for d in phase_diffs]) if phase_diffs else 0

        return {
            'sense_phases': {s: {'mean': v['mean'], 'std': v['std']}
                           for s, v in sense_phases.items()},
            'mean_phase_divergence': mean_diff,
            'phase_differences': phase_diffs
        }

    def compute_retrieval_accuracy(
        self,
        sense_embeddings: Dict[str, List[Dict]]
    ) -> float:
        """
        Compute sense retrieval accuracy.

        For each example, find its nearest neighbor and check if same sense.
        """
        # Collect all embeddings with labels
        all_embeddings = []
        all_labels = []

        for sense, embeddings in sense_embeddings.items():
            for emb in embeddings:
                combined = np.concatenate([emb['real'], emb['imag']])
                all_embeddings.append(combined)
                all_labels.append(sense)

        all_embeddings = np.array(all_embeddings)

        # For each embedding, find nearest neighbor (excluding self)
        correct = 0
        total = 0

        for i in range(len(all_embeddings)):
            query = all_embeddings[i]
            query_label = all_labels[i]

            # Compute distances to all others
            distances = []
            for j in range(len(all_embeddings)):
                if i != j:
                    dist = np.linalg.norm(query - all_embeddings[j])
                    distances.append((dist, all_labels[j]))

            # Find nearest neighbor
            distances.sort(key=lambda x: x[0])
            nearest_label = distances[0][1]

            if nearest_label == query_label:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run full polysemy benchmark on all test words.

        Returns comprehensive metrics for each word and overall.
        """
        results = {
            'per_word': {},
            'overall': {}
        }

        all_separation_ratios = []
        all_silhouettes = []
        all_retrieval_accs = []
        all_phase_divs = []

        for word, senses in self.POLYSEMY_TEST_CASES.items():
            print(f"\nBenchmarking word: '{word}' ({len(senses)} senses)")

            # Extract embeddings
            sense_embeddings = self.compute_sense_embeddings(word, senses)

            # Compute metrics
            cluster_metrics = self.compute_cluster_separation(sense_embeddings)
            phase_metrics = self.compute_phase_divergence(sense_embeddings)
            retrieval_acc = self.compute_retrieval_accuracy(sense_embeddings)

            results['per_word'][word] = {
                'senses': list(senses.keys()),
                'num_examples': sum(len(v) for v in senses.values()),
                'cluster_separation': cluster_metrics,
                'phase_divergence': phase_metrics,
                'retrieval_accuracy': retrieval_acc
            }

            # Collect for overall metrics
            all_separation_ratios.append(cluster_metrics['separation_ratio'])
            all_silhouettes.append(cluster_metrics['silhouette_score'])
            all_retrieval_accs.append(retrieval_acc)
            all_phase_divs.append(phase_metrics['mean_phase_divergence'])

            print(f"  Separation ratio: {cluster_metrics['separation_ratio']:.3f}")
            print(f"  Silhouette score: {cluster_metrics['silhouette_score']:.3f}")
            print(f"  Retrieval accuracy: {retrieval_acc:.3f}")
            print(f"  Phase divergence: {phase_metrics['mean_phase_divergence']:.3f}")

        # Compute overall metrics
        results['overall'] = {
            'mean_separation_ratio': np.mean(all_separation_ratios),
            'mean_silhouette_score': np.mean(all_silhouettes),
            'mean_retrieval_accuracy': np.mean(all_retrieval_accs),
            'mean_phase_divergence': np.mean(all_phase_divs),
            'num_words_tested': len(self.POLYSEMY_TEST_CASES)
        }

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("  POLYSEMY BENCHMARK SUMMARY")
        print("=" * 60)

        overall = results['overall']
        print(f"\nOverall Metrics ({overall['num_words_tested']} words):")
        print(f"  Mean Separation Ratio: {overall['mean_separation_ratio']:.4f}")
        print(f"  Mean Silhouette Score: {overall['mean_silhouette_score']:.4f}")
        print(f"  Mean Retrieval Accuracy: {overall['mean_retrieval_accuracy']:.4f}")
        print(f"  Mean Phase Divergence: {overall['mean_phase_divergence']:.4f} rad")

        # Win condition assessment
        print("\n" + "-" * 60)
        print("WIN CONDITION ASSESSMENT")
        print("-" * 60)

        # Thresholds for "passing"
        sep_pass = overall['mean_separation_ratio'] > 1.0
        sil_pass = overall['mean_silhouette_score'] > 0.0
        ret_pass = overall['mean_retrieval_accuracy'] > 0.5

        print(f"  Separation > 1.0: {'PASS' if sep_pass else 'FAIL'} ({overall['mean_separation_ratio']:.3f})")
        print(f"  Silhouette > 0.0: {'PASS' if sil_pass else 'FAIL'} ({overall['mean_silhouette_score']:.3f})")
        print(f"  Retrieval > 50%:  {'PASS' if ret_pass else 'FAIL'} ({overall['mean_retrieval_accuracy']:.1%})")

        if sep_pass and sil_pass and ret_pass:
            print("\n  >>> WIN CONDITION MET <<<")
        else:
            print("\n  >>> WIN CONDITION NOT MET <<<")


def build_vocab_from_data(data_path: str, vocab_size: int = 1000) -> Dict[str, int]:
    """Build vocabulary from training data (matching training process)."""
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
    data_path: str = './data/pilot_paradigm_data.jsonl',
    config_overrides: Dict = None
) -> Tuple[QLLM, Dict[str, int]]:
    """Load model from checkpoint with vocab from training data."""

    # Build vocab from training data (matching training process)
    vocab = build_vocab_from_data(data_path)
    print(f"  Built vocab with {len(vocab)} words from {data_path}")

    # Default config (should match training)
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

    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    model = QLLM(config)

    # Load weights - filter out mismatched shapes
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # Get model's state dict to compare shapes
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                filtered_state[key] = value
            else:
                skipped.append(f"{key}: checkpoint {value.shape} vs model {model_state[key].shape}")
        else:
            skipped.append(f"{key}: not in model")

    if skipped:
        print(f"  Skipped {len(skipped)} mismatched keys")

    model.load_state_dict(filtered_state, strict=False)

    return model, vocab


def main():
    parser = argparse.ArgumentParser(description="Run polysemy sense discrimination benchmark")
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
    print("  POLYSEMY WIN-CONDITION BENCHMARK")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model, vocab = load_model_from_checkpoint(args.model_path, args.data)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Run benchmark
    benchmark = PolysemyBenchmark(model, vocab, device)
    results = benchmark.run_benchmark()

    # Print summary
    benchmark.print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: save next to model
        model_dir = Path(args.model_path).parent
        output_path = model_dir / "polysemy_benchmark.json"

    # Convert numpy to python types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
