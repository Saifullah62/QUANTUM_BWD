#!/usr/bin/env python3
"""
QLLM Training Script
====================

Main training script for running QLLM on the GPU cluster.

Usage:
    # Local training (small model)
    python scripts/train.py --config configs/minimal.json

    # Full training on gpu-ramp
    python scripts/train.py --config configs/full.json --distributed

    # LoRA fine-tuning
    python scripts/train.py --config configs/lora.json --use-lora
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from qllm.core.config import QLLMConfig, TrainingConfig
from qllm.core.model import QLLM
from qllm.training.trainer import QLLMTrainer, LoRATrainer
from qllm.training.dataset import QuantumParadigmDataset
from qllm.utils.cluster import ClusterManager


def parse_args():
    parser = argparse.ArgumentParser(description="Train QLLM")

    # Data
    parser.add_argument('--data', type=str, default='./data/quantum_paradigm_data.jsonl',
                        help='Path to training data')
    parser.add_argument('--eval-data', type=str, default=None,
                        help='Path to evaluation data')

    # Model
    parser.add_argument('--config', type=str, default=None,
                        help='Path to model config JSON')
    parser.add_argument('--base-model', type=str, default=None,
                        help='Base model for fine-tuning (e.g., llama3.1:8b)')
    parser.add_argument('--model-size', type=str, default='minimal',
                        choices=['minimal', 'small', 'medium', 'large'],
                        help='Model size preset')

    # Training
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--gradient-accumulation', type=int, default=4)

    # LoRA
    parser.add_argument('--use-lora', action='store_true',
                        help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora-r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16,
                        help='LoRA alpha')

    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this training run')

    # Distributed
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    parser.add_argument('--node', type=str, default='local',
                        choices=['local', 'gpu-swarm', 'gpu-ramp'],
                        help='Which cluster node to use')

    # Paradigm weights
    parser.add_argument('--phase-loss-weight', type=float, default=0.1)
    parser.add_argument('--retrocausal-weight', type=float, default=0.1)
    parser.add_argument('--qualia-weight', type=float, default=0.1)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    return parser.parse_args()


def create_model_config(args) -> QLLMConfig:
    """Create model configuration from args"""
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        return QLLMConfig(**config_dict)

    if args.base_model:
        return QLLMConfig.from_base_model(args.base_model)

    # Use size preset
    size_configs = {
        'minimal': QLLMConfig.minimal,
        'small': QLLMConfig.small,
        'medium': QLLMConfig.medium,
        'large': QLLMConfig.large
    }

    return size_configs[args.model_size]()


def create_training_config(args) -> TrainingConfig:
    """Create training configuration from args"""
    return TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        phase_consistency_loss_weight=args.phase_loss_weight,
        retrocausal_coherence_weight=args.retrocausal_weight,
        qualia_diversity_loss_weight=args.qualia_weight
    )


def setup_tokenizer(base_model: str = None):
    """Setup tokenizer"""
    try:
        from transformers import AutoTokenizer

        if base_model:
            # Map Ollama model names to HuggingFace equivalents
            hf_mapping = {
                'llama3.1:8b': 'meta-llama/Llama-3.1-8B',
                'llama3.2:3b': 'meta-llama/Llama-3.2-3B',
                'phi3:mini': 'microsoft/phi-3-mini-4k-instruct',
                'mixtral:8x7b': 'mistralai/Mixtral-8x7B-v0.1'
            }
            model_name = hf_mapping.get(base_model, base_model)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            # Default to GPT2 tokenizer for testing
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Using simple tokenizer fallback")
        return None


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup output directory
    run_name = args.run_name or f"qllm_{args.model_size}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QLLM Training")
    print("=" * 60)

    # Check cluster status if distributed
    if args.distributed:
        import asyncio
        cluster = ClusterManager()
        status = asyncio.run(cluster.get_cluster_status())
        print(f"\nCluster status:")
        for node, info in status['nodes'].items():
            print(f"  {node}: {info.get('status', 'unknown')}")

    # Create configs
    print("\nCreating configurations...")
    model_config = create_model_config(args)
    train_config = create_training_config(args)

    # Save configs
    with open(output_dir / "model_config.json", 'w') as f:
        json.dump(model_config.to_dict(), f, indent=2)
    with open(output_dir / "train_config.json", 'w') as f:
        json.dump(train_config.__dict__, f, indent=2, default=str)

    # Setup tokenizer
    print("Setting up tokenizer...")
    tokenizer = setup_tokenizer(args.base_model)

    # Create model
    print("Creating model...")
    model = QLLM(model_config)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Paradigms: {model.get_paradigm_summary()}")

    # Load base model weights if specified
    if args.base_model and not args.use_lora:
        print(f"Loading base model: {args.base_model}")
        model = QLLM.from_pretrained(args.base_model, model_config)

    # Setup dataset
    print("Loading dataset...")
    if tokenizer and Path(args.data).exists():
        train_dataset = QuantumParadigmDataset(
            args.data,
            tokenizer,
            max_length=model_config.max_seq_length
        )
        eval_dataset = None
        if args.eval_data and Path(args.eval_data).exists():
            eval_dataset = QuantumParadigmDataset(
                args.eval_data,
                tokenizer,
                max_length=model_config.max_seq_length
            )
        print(f"  Training examples: {len(train_dataset)}")
    else:
        print("  Warning: No tokenizer or data file. Using dummy dataset.")
        # Create dummy dataset for testing
        class DummyDataset:
            def __init__(self, size=100):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randint(0, model_config.vocab_size, (256,)),
                    'attention_mask': torch.ones(256),
                    'labels': torch.randint(0, model_config.vocab_size, (256,)),
                    'paradigm': 'semantic_phase'
                }

        train_dataset = DummyDataset()
        eval_dataset = None

    # Create trainer
    print("Creating trainer...")
    if args.use_lora:
        trainer = LoRATrainer(
            model=model,
            train_config=train_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(output_dir)
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  LoRA trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    else:
        trainer = QLLMTrainer(
            model=model,
            train_config=train_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(output_dir)
        )

    # Resume if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    print("-" * 60)
    trainer.train()

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
