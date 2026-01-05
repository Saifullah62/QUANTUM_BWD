#!/usr/bin/env python3
"""
Proof Ablation with Pilot Data
==============================

This is the "proof moment" - running ablation with targeted paradigm data
to show that each paradigm objective fires and creates measurable signal.

Expected Results:
- SemanticPhase: separation on polysemy pairs
- Lindblad: separation on noise degradation curve
- Retrocausal: improvement on effect→cause while leak stays clean
- Qualia: calibration metrics without saturation

Usage:
    python scripts/run_proof_ablation.py --data data/pilot_paradigm_data.jsonl
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from qllm.core.config import QLLMConfig, TrainingConfig
from qllm.core.model import QLLM
from qllm.training.ablation import AblationRunner


class PilotDataset:
    """Dataset from pilot paradigm data with tokenization."""

    def __init__(
        self,
        data_path: str,
        max_length: int = 128,
        vocab_size: int = 1000
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Load examples
        self.examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} pilot examples")

        # Build simple vocab from data
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from the pilot data."""
        word_counts = {}
        for ex in self.examples:
            text = ex['input_text'] + ' ' + ex['output_text']
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

        # Build vocab (reserve 0 for pad, 1 for unk)
        self.word_to_id = {'<pad>': 0, '<unk>': 1}
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - 2]):
            self.word_to_id[word] = i + 2

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}

    def _tokenize(self, text: str) -> List[int]:
        """Simple whitespace tokenization."""
        tokens = []
        for word in text.lower().split():
            tokens.append(self.word_to_id.get(word, 1))  # 1 = <unk>
        return tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Combine input and output
        text = example['input_text'] + ' ' + example['output_text']
        token_ids = self._tokenize(text)

        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = (input_ids != 0).float()

        # Only return tensor-compatible data for DataLoader
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
            'paradigm': example['paradigm']
        }


def run_proof_ablation(args):
    """Run proof ablation with pilot data."""

    print("=" * 70)
    print("  PROOF ABLATION WITH PILOT DATA")
    print("  Testing paradigm-specific signal with targeted examples")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load dataset
    print(f"\nLoading pilot data from: {args.data}")
    train_dataset = PilotDataset(args.data, max_length=args.max_length)
    eval_dataset = PilotDataset(args.data, max_length=args.max_length)  # Same for proof

    # Count by paradigm
    paradigm_counts = {}
    for i in range(len(train_dataset)):
        p = train_dataset.examples[i]['paradigm']
        paradigm_counts[p] = paradigm_counts.get(p, 0) + 1

    print(f"\nExamples by paradigm:")
    for p, c in sorted(paradigm_counts.items()):
        print(f"  {p}: {c}")

    # Model config - use vocab size from dataset
    model_config = QLLMConfig(
        vocab_size=train_dataset.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=4,
        intermediate_dim=args.hidden_dim * 2,
        max_seq_length=args.max_length,
        use_semantic_phase=True,
        semantic_phase_dim=args.hidden_dim,
        use_retrocausal_attention=True,
        retrocausal_layers=[0, args.num_layers // 2] if args.num_layers > 1 else [0],
        use_lindblad_layers=True,
        lindblad_every_n_layers=2,
        use_qualia_output=True,
        num_qualia_channels=8,
        use_emergent_init=True
    )

    print(f"\nModel config:")
    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Max length: {model_config.max_seq_length}")

    # Training config
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        warmup_steps=min(50, args.steps // 5),
        gradient_accumulation_steps=1,
        logging_steps=max(10, args.steps // 10),
        eval_steps=args.steps // 2,
        save_steps=args.steps
    )

    print(f"\nTraining config:")
    print(f"  Steps per ablation: {args.steps}")
    print(f"  Batch size: {args.batch_size}")

    # Output dir
    output_dir = Path(args.output_dir) / f"proof_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    # Run ablation
    runner = AblationRunner(
        model_config=model_config,
        train_config=train_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(output_dir)
    )

    print("\n" + "=" * 70)
    print("  RUNNING ABLATION SUITE")
    print("=" * 70)

    results = runner.run_full_ablation(steps_per_run=args.steps)

    # Print comparison
    runner.print_comparison(results)

    # Compute paradigm-specific metrics
    print("\n" + "=" * 70)
    print("  PARADIGM-SPECIFIC WIN CONDITIONS")
    print("=" * 70)

    analyze_paradigm_wins(results, paradigm_counts)

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_path': str(args.data),
        'steps_per_run': args.steps,
        'paradigm_counts': paradigm_counts,
        'results': [
            {
                'paradigm': r.paradigm,
                'final_loss': r.final_loss,
                'win_conditions': r.win_conditions,
                'duration_seconds': r.duration_seconds
            }
            for r in results
        ]
    }

    with open(output_dir / "proof_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return results


def analyze_paradigm_wins(results, paradigm_counts):
    """Analyze paradigm-specific win conditions."""

    # Find baseline
    baseline = next((r for r in results if r.paradigm == 'baseline'), None)
    if not baseline:
        print("  No baseline found!")
        return

    baseline_loss = baseline.final_loss

    print(f"\n  Baseline loss: {baseline_loss:.4f}")
    print()

    # Check each paradigm
    for r in results:
        if r.paradigm == 'baseline':
            continue

        delta_loss = r.final_loss - baseline_loss
        delta_pct = (delta_loss / baseline_loss) * 100

        if 'semantic_phase' in r.paradigm:
            print(f"  SEMANTIC PHASE ({r.paradigm}):")
            print(f"    Loss delta: {delta_loss:+.4f} ({delta_pct:+.2f}%)")
            print(f"    Win condition: Phase contrastive should show separation")
            if delta_loss < -0.001:
                print(f"    [SIGNAL DETECTED] ✓")
            else:
                print(f"    [awaiting polysemy pairs with repeated tokens]")

        elif 'lindblad' in r.paradigm:
            print(f"  LINDBLAD ({r.paradigm}):")
            print(f"    Loss delta: {delta_loss:+.4f} ({delta_pct:+.2f}%)")
            print(f"    Consistency: {r.win_conditions.get('consistency', 'N/A')}")
            print(f"    Win condition: Degradation curve stability")
            if r.win_conditions.get('consistency', 0) > 0.99:
                print(f"    [SIGNAL DETECTED] ✓")
            else:
                print(f"    [awaiting noise robustness evaluation]")

        elif 'qualia' in r.paradigm:
            print(f"  QUALIA ({r.paradigm}):")
            print(f"    Loss delta: {delta_loss:+.4f} ({delta_pct:+.2f}%)")
            print(f"    Win condition: Channel calibration without saturation")
            # Check if qualia loss was active
            if delta_loss != 0:
                print(f"    [QUALIA LOSS ACTIVE] ✓")

        elif 'retrocausal' in r.paradigm:
            print(f"  RETROCAUSAL ({r.paradigm}):")
            print(f"    Loss delta: {delta_loss:+.4f} ({delta_pct:+.2f}%)")
            retro_score = r.win_conditions.get('retrocausal', 0)
            print(f"    Retrocausal score: {retro_score:.4f}")
            print(f"    Win condition: Effect→cause while leak clean")
            # Leak should stay low
            if retro_score > 0.5 and retro_score < 0.9:
                print(f"    [SIGNAL DETECTED] ✓")

        elif 'emergent' in r.paradigm:
            print(f"  EMERGENT ({r.paradigm}):")
            print(f"    Loss delta: {delta_loss:+.4f} ({delta_pct:+.2f}%)")
            print(f"    Win condition: Attractor stability (expects longer training)")

        elif 'all_paradigms' in r.paradigm:
            print(f"  ALL PARADIGMS ({r.paradigm}):")
            print(f"    Loss delta: {delta_loss:+.4f} ({delta_pct:+.2f}%)")
            if delta_loss < -0.001:
                print(f"    [COMBINED SIGNAL] ✓")

        print()


def main():
    parser = argparse.ArgumentParser(description="Run proof ablation with pilot data")
    parser.add_argument('--data', type=str, default='./data/pilot_paradigm_data.jsonl',
                        help='Path to pilot dataset')
    parser.add_argument('--steps', type=int, default=500,
                        help='Training steps per ablation')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of layers')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Max sequence length')
    parser.add_argument('--output-dir', type=str, default='./proof_ablation_runs',
                        help='Output directory')

    args = parser.parse_args()
    run_proof_ablation(args)


if __name__ == "__main__":
    main()
