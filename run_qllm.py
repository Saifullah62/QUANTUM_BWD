#!/usr/bin/env python3
"""
QLLM - Quantum-Inspired Language Model
=======================================

Main entry point for QLLM operations.

Usage:
    python run_qllm.py train --config configs/minimal.json
    python run_qllm.py evaluate --model outputs/model.pt
    python run_qllm.py generate-data --examples 100
    python run_qllm.py inference --model outputs/model.pt --prompt "Hello"
    python run_qllm.py cluster-status
"""

import argparse
import asyncio
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="QLLM - Quantum-Inspired Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train a minimal model
    python run_qllm.py train --model-size minimal --max-steps 1000

    # Generate training data using swarm
    python run_qllm.py generate-data --examples 100

    # Evaluate trained model
    python run_qllm.py evaluate --model outputs/qllm_minimal/model.pt

    # Run inference
    python run_qllm.py inference --model outputs/model.pt --prompt "Explain quantum semantics"

    # Check cluster status
    python run_qllm.py cluster-status
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train QLLM model')
    train_parser.add_argument('--data', type=str, default='./data/quantum_paradigm_data.jsonl')
    train_parser.add_argument('--model-size', type=str, default='minimal',
                              choices=['minimal', 'small', 'medium', 'large'])
    train_parser.add_argument('--use-lora', action='store_true')
    train_parser.add_argument('--max-steps', type=int, default=10000)
    train_parser.add_argument('--batch-size', type=int, default=4)
    train_parser.add_argument('--learning-rate', type=float, default=2e-4)
    train_parser.add_argument('--output-dir', type=str, default='./outputs')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', type=str, required=True)
    eval_parser.add_argument('--paradigm', type=str, default=None)
    eval_parser.add_argument('--run-ablation', action='store_true')
    eval_parser.add_argument('--output-dir', type=str, default='./evaluation_results')

    # Generate data command
    gen_parser = subparsers.add_parser('generate-data', help='Generate training data')
    gen_parser.add_argument('--output', type=str, default='./data/quantum_paradigm_data.jsonl')
    gen_parser.add_argument('--paradigm', type=str, default=None)
    gen_parser.add_argument('--examples', type=int, default=100)
    gen_parser.add_argument('--swarm-pattern', type=str, default='think')

    # Inference command
    inf_parser = subparsers.add_parser('inference', help='Run inference')
    inf_parser.add_argument('--model', type=str, required=True)
    inf_parser.add_argument('--prompt', type=str, required=True)
    inf_parser.add_argument('--max-tokens', type=int, default=100)
    inf_parser.add_argument('--paradigm-mode', type=str, default=None,
                            choices=['semantic_phase', 'retrocausal', 'qualia'])

    # Cluster status command
    cluster_parser = subparsers.add_parser('cluster-status', help='Check cluster status')

    args = parser.parse_args()

    if args.command == 'train':
        # Import and run train script
        sys.argv = ['train.py',
                    '--data', args.data,
                    '--model-size', args.model_size,
                    '--max-steps', str(args.max_steps),
                    '--batch-size', str(args.batch_size),
                    '--learning-rate', str(args.learning_rate),
                    '--output-dir', args.output_dir]
        if args.use_lora:
            sys.argv.append('--use-lora')
        from scripts.train import main as train_main
        train_main()

    elif args.command == 'evaluate':
        sys.argv = ['evaluate.py',
                    '--model', args.model,
                    '--output-dir', args.output_dir]
        if args.paradigm:
            sys.argv.extend(['--paradigm', args.paradigm])
        if args.run_ablation:
            sys.argv.append('--run-ablation')
        from scripts.evaluate import main as eval_main
        eval_main()

    elif args.command == 'generate-data':
        sys.argv = ['generate_data.py',
                    '--output', args.output,
                    '--examples', str(args.examples),
                    '--swarm-pattern', args.swarm_pattern]
        if args.paradigm:
            sys.argv.extend(['--paradigm', args.paradigm])
        from scripts.generate_data import main as gen_main
        asyncio.run(gen_main())

    elif args.command == 'inference':
        run_inference(args)

    elif args.command == 'cluster-status':
        asyncio.run(check_cluster())

    else:
        parser.print_help()


def run_inference(args):
    """Run inference with trained model"""
    import torch
    import json

    from qllm.core.config import QLLMConfig
    from qllm.core.model import QLLM

    print("=" * 60)
    print("QLLM Inference")
    print("=" * 60)

    model_path = Path(args.model)
    config_path = model_path.parent / "model_config.json"

    # Load config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = QLLMConfig(**json.load(f))
    else:
        config = QLLMConfig.minimal()

    # Load model
    print(f"Loading model from: {model_path}")
    model = QLLM(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Setup tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    except:
        print("Warning: Could not load tokenizer")
        return

    # Prepare input
    prompt = args.prompt
    if args.paradigm_mode:
        from qllm.utils.tokenizer import QuantumTokenizer
        qt = QuantumTokenizer(tokenizer)
        prompt = qt.create_paradigm_prompt(args.paradigm_mode, prompt)

    print(f"\nPrompt: {prompt}")
    print("-" * 60)

    # Generate
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")


async def check_cluster():
    """Check cluster status"""
    from qllm.utils.cluster import ClusterManager

    print("=" * 60)
    print("GPU Cluster Status")
    print("=" * 60)

    cluster = ClusterManager()
    status = await cluster.get_cluster_status()

    print(f"\nTotal VRAM: {status['total_vram_gb']} GB")

    print("\nNodes:")
    for name, info in status['nodes'].items():
        node = cluster.NODES[name]
        print(f"  {name} ({node.gpu_model}, {node.vram_gb}GB)")
        print(f"    Status: {info.get('status', 'unknown')}")
        if 'gpu_info' in info:
            print(f"    GPU: {info['gpu_info']}")
        if 'error' in info:
            print(f"    Error: {info['error']}")

    print("\nFleet Services (gpu-swarm:8000-8011):")
    for service, info in status['fleet_services'].items():
        status_str = info.get('status', 'unknown')
        port = info.get('port', '?')
        status_icon = "" if status_str == 'healthy' else ""
        print(f"  {status_icon} {service} (:{port}): {status_str}")


if __name__ == "__main__":
    main()
