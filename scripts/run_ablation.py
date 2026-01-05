#!/usr/bin/env python3
"""
QLLM Ablation Study Runner
===========================

Run ablation study to validate each paradigm independently.
Each paradigm is toggled on/off to measure its isolated contribution.

Usage:
    python scripts/run_ablation.py --steps 500
    python scripts/run_ablation.py --steps 200 --quick  # Quick sanity check
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import json
from pathlib import Path
from datetime import datetime

from qllm.core.config import QLLMConfig, TrainingConfig
from qllm.core.model import QLLM
from qllm.training.ablation import AblationRunner, ParadigmLossConfig


class DummyDataset:
    """Dummy dataset for ablation testing"""

    def __init__(self, size: int, vocab_size: int, seq_length: int):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_length = seq_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, self.vocab_size, (self.seq_length,)),
            'attention_mask': torch.ones(self.seq_length),
            'labels': torch.randint(0, self.vocab_size, (self.seq_length,)),
            'paradigm': 'ablation_test'
        }


def run_ablation(args):
    """Run the ablation study"""

    print("=" * 70)
    print("  QLLM ABLATION STUDY")
    print("  Validating paradigm contributions independently")
    print("=" * 70)

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nDevice: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    else:
        print("\nWarning: Running on CPU - will be slow!")

    # Create minimal model config for ablation
    model_config = QLLMConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        intermediate_dim=256,
        max_seq_length=128,
        # Semantic phase
        use_semantic_phase=True,
        semantic_phase_dim=128,
        # Retrocausal
        use_retrocausal_attention=True,
        retrocausal_layers=[0, 2],
        # Lindblad
        use_lindblad_layers=True,
        lindblad_every_n_layers=2,
        # Qualia
        use_qualia_output=True,
        num_qualia_channels=8,
        # Emergent
        use_emergent_init=True
    )

    print(f"\nModel config:")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Vocab: {model_config.vocab_size}")

    # Create training config
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
    print(f"  Learning rate: {args.learning_rate}")

    # Create dummy dataset
    dataset_size = max(100, args.steps * args.batch_size)
    train_dataset = DummyDataset(
        size=dataset_size,
        vocab_size=model_config.vocab_size,
        seq_length=64
    )
    eval_dataset = DummyDataset(
        size=20,
        vocab_size=model_config.vocab_size,
        seq_length=64
    )

    print(f"\nDataset:")
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Eval examples: {len(eval_dataset)}")

    # Create output directory
    output_dir = Path(args.output_dir) / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    # Create ablation runner
    runner = AblationRunner(
        model_config=model_config,
        train_config=train_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(output_dir)
    )

    if args.quick:
        # Quick mode: just baseline and one paradigm
        print("\n[Quick mode: baseline + semantic_phase only]")
        results = []

        results.append(runner.run_single_ablation(
            "baseline",
            ParadigmLossConfig.baseline(),
            steps=args.steps
        ))

        results.append(runner.run_single_ablation(
            "semantic_phase_only",
            ParadigmLossConfig.semantic_phase_only(),
            steps=args.steps
        ))

        runner.save_results(results)
        runner.print_comparison(results)
    else:
        # Full ablation
        print("\n[Full ablation: all paradigms]")
        results = runner.run_full_ablation(steps_per_run=args.steps)
        runner.print_comparison(results)

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'steps_per_run': args.steps,
        'model_config': model_config.to_dict(),
        'num_paradigms_tested': len(results),
        'results_summary': [
            {
                'paradigm': r.paradigm,
                'final_loss': r.final_loss,
                'win_conditions': r.win_conditions,
                'duration_seconds': r.duration_seconds
            }
            for r in results
        ]
    }

    with open(output_dir / "ablation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  ABLATION COMPLETE")
    print(f"  Results saved to: {output_dir}")
    print(f"{'=' * 70}")

    # Return results for analysis
    return results


def main():
    parser = argparse.ArgumentParser(description="QLLM Ablation Study")
    parser.add_argument('--steps', type=int, default=500,
                        help='Training steps per ablation run')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./ablation_runs',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: only baseline + semantic_phase')

    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
