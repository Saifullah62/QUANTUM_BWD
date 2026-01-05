#!/usr/bin/env python3
"""
QLLM Evaluation Script
======================

Run paradigm benchmarks and analysis on trained QLLM models.

Usage:
    # Run full benchmark suite
    python scripts/evaluate.py --model outputs/qllm_final/model.pt

    # Run specific paradigm
    python scripts/evaluate.py --model outputs/qllm_final/model.pt --paradigm semantic_phase

    # Compare with baseline
    python scripts/evaluate.py --model outputs/qllm_final/model.pt --baseline gpt2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from qllm.core.config import QLLMConfig
from qllm.core.model import QLLM
from qllm.evaluation.benchmarks import ParadigmBenchmarkSuite
from qllm.evaluation.analysis import ParadigmAnalyzer, ComparativeAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QLLM")

    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to model config (auto-detected if not provided)')

    parser.add_argument('--paradigm', type=str, default=None,
                        choices=['semantic_phase', 'retrocausal', 'lindblad', 'qualia', 'emergent'],
                        help='Specific paradigm to evaluate')

    parser.add_argument('--baseline', type=str, default=None,
                        help='Baseline model for comparison')

    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Output directory for results')

    parser.add_argument('--run-ablation', action='store_true',
                        help='Run ablation studies')

    parser.add_argument('--generate-report', action='store_true',
                        help='Generate full analysis report with visualizations')

    return parser.parse_args()


def load_model(model_path: str, config_path: str = None):
    """Load trained QLLM model"""
    model_path = Path(model_path)

    # Find config
    if config_path:
        config_file = Path(config_path)
    else:
        # Try to find config next to model
        config_file = model_path.parent / "model_config.json"
        if not config_file.exists():
            config_file = model_path.parent / "config.json"

    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        config = QLLMConfig(**config_dict)
    else:
        print("Warning: No config found, using minimal config")
        config = QLLMConfig.minimal()

    # Create and load model
    model = QLLM(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    return model, config


def setup_tokenizer(config: QLLMConfig):
    """Setup tokenizer"""
    try:
        from transformers import AutoTokenizer

        # Try to load appropriate tokenizer
        if hasattr(config, 'base_model') and config.base_model:
            tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        else:
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        return None


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QLLM Evaluation")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model}")
    model, config = load_model(args.model, args.config)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Paradigms: {model.get_paradigm_summary()}")

    # Setup tokenizer
    tokenizer = setup_tokenizer(config)
    if tokenizer is None:
        print("Error: Tokenizer required for evaluation")
        return

    # Create benchmark suite
    benchmark_suite = ParadigmBenchmarkSuite(model, tokenizer)

    # Run benchmarks
    if args.paradigm:
        print(f"\nRunning {args.paradigm} benchmarks...")
        results = benchmark_suite.run_paradigm(args.paradigm)
        results_dict = {
            args.paradigm: {
                'benchmarks': [
                    {
                        'name': r.benchmark_name,
                        'score': r.score,
                        'max_score': r.max_score,
                        'normalized': r.normalized_score
                    }
                    for r in results
                ]
            }
        }
    else:
        print("\nRunning full benchmark suite...")
        results_dict = benchmark_suite.run_all()

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for paradigm, data in results_dict.items():
        if paradigm == 'overall':
            continue
        print(f"\n{paradigm.upper()}")
        print("-" * 40)
        if 'benchmarks' in data:
            for bench in data['benchmarks']:
                score = bench.get('normalized', bench.get('score', 0))
                print(f"  {bench['name']}: {score:.3f}")
        if 'overall' in data:
            print(f"  OVERALL: {data['overall']:.3f}")

    if 'overall' in results_dict:
        print(f"\n{'='*40}")
        print(f"TOTAL SCORE: {results_dict['overall']['normalized']:.3f}")

    # Save results
    results_file = output_dir / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Run ablation studies if requested
    if args.run_ablation:
        print("\n" + "=" * 60)
        print("ABLATION STUDIES")
        print("=" * 60)

        analyzer = ParadigmAnalyzer(model)

        # Create test inputs
        test_prompts = [
            "The bank was steep and covered with grass.",
            "Given the outcome of success, what steps led here?",
            "noisy signal extract meaning clarity",
            "Describe what seeing red feels like.",
            "Individual neurons create consciousness how?"
        ]
        test_inputs = [
            tokenizer.encode(p, return_tensors='pt')['input_ids']
            for p in test_prompts
        ]

        ablation_results = {}
        for paradigm in ['semantic_phase', 'retrocausal', 'lindblad', 'qualia', 'emergent']:
            print(f"\nAblating {paradigm}...")
            result = analyzer.run_ablation_study(test_inputs, paradigm)
            ablation_results[paradigm] = result['impact']
            print(f"  Loss increase: {result['impact']['loss_increase']:.4f}")
            print(f"  Relative impact: {result['impact']['relative_impact']:.2%}")

        ablation_file = output_dir / "ablation_results.json"
        with open(ablation_file, 'w') as f:
            json.dump(ablation_results, f, indent=2)

    # Generate full report if requested
    if args.generate_report:
        print("\n" + "=" * 60)
        print("GENERATING ANALYSIS REPORT")
        print("=" * 60)

        analyzer = ParadigmAnalyzer(model)

        test_prompts = [
            "The bank was steep and covered with grass.",
            "Given the outcome of success, what steps led here?",
            "noisy signal extract meaning clarity",
            "Describe what seeing red feels like.",
            "Individual neurons create consciousness how?"
        ] * 3  # More samples

        test_inputs = [
            tokenizer.encode(p, return_tensors='pt')['input_ids']
            for p in test_prompts
        ]

        report = analyzer.generate_paradigm_report(
            test_inputs,
            output_dir=str(output_dir / "analysis")
        )

        print("\nReport generated with visualizations")

    # Compare with baseline if specified
    if args.baseline:
        print("\n" + "=" * 60)
        print(f"COMPARING WITH BASELINE: {args.baseline}")
        print("=" * 60)

        try:
            from transformers import AutoModelForCausalLM

            baseline = AutoModelForCausalLM.from_pretrained(args.baseline)
            baseline.eval()

            comparator = ComparativeAnalyzer(model, baseline, tokenizer)

            test_cases = [
                {
                    'input': 'The bank by the river was steep.',
                    'expected_output': 'riverbank geography terrain slope'
                },
                {
                    'input': 'She deposited money at the bank.',
                    'expected_output': 'financial institution money deposit'
                }
            ]

            comparison = comparator.compare_on_task('ambiguity_resolution', test_cases)

            print(f"\nQLL mean score: {comparison['summary']['qllm_mean']:.3f}")
            print(f"Baseline mean score: {comparison['summary']['baseline_mean']:.3f}")
            print(f"Improvement: {comparison['summary']['improvement']:.3f}")

            comparison_file = output_dir / "comparison_results.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)

        except Exception as e:
            print(f"Warning: Could not load baseline: {e}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"All results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
