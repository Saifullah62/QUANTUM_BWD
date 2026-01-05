#!/usr/bin/env python3
"""
QLLM Training Data Generator
=============================

Generate quantum paradigm training data using the GPU swarm.

Uses the fleet's swarm controller on gpu-swarm (159.203.35.45:8007)
to generate high-quality training examples for each paradigm.

Usage:
    # Generate data for all paradigms
    python scripts/generate_data.py --output data/training_data.jsonl

    # Generate for specific paradigm
    python scripts/generate_data.py --paradigm semantic_phase --examples 200

    # Use specific swarm pattern
    python scripts/generate_data.py --swarm-pattern debate --examples 100
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qllm.training.dataset import DatasetGenerator
from qllm.utils.cluster import ClusterManager


def parse_args():
    parser = argparse.ArgumentParser(description="Generate QLLM Training Data")

    parser.add_argument('--output', type=str, default='./data/quantum_paradigm_data.jsonl',
                        help='Output file path')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory')

    parser.add_argument('--paradigm', type=str, default=None,
                        choices=['semantic_phase', 'retrocausal', 'lindblad', 'qualia', 'emergent'],
                        help='Specific paradigm to generate data for')

    parser.add_argument('--examples', type=int, default=100,
                        help='Number of examples per paradigm')

    parser.add_argument('--swarm-pattern', type=str, default='think',
                        choices=['think', 'explore', 'debate', 'consensus', 'tree'],
                        help='Swarm pattern to use for generation')

    parser.add_argument('--check-swarm', action='store_true',
                        help='Just check swarm connectivity')

    return parser.parse_args()


async def check_swarm_status():
    """Check if swarm controller is accessible"""
    cluster = ClusterManager()

    print("Checking swarm controller status...")

    try:
        result = await cluster.swarm_think("Hello, are you working?")
        if result.get('status') == 'SUCCESS':
            print("  Swarm controller: ONLINE")
            return True
        else:
            print(f"  Swarm controller: ERROR - {result}")
            return False
    except Exception as e:
        print(f"  Swarm controller: OFFLINE - {e}")
        return False


async def generate_with_custom_pattern(
    paradigm: str,
    num_examples: int,
    pattern: str,
    cluster: ClusterManager
) -> list:
    """Generate examples using a specific swarm pattern"""

    paradigm_prompts = {
        'semantic_phase': [
            "Generate an example sentence with an ambiguous word, then show how different contexts change its meaning.",
            "Create a polysemous word example with at least 3 different meanings in different contexts.",
            "Write a sentence that can be interpreted multiple ways, then provide the interpretations."
        ],
        'retrocausal': [
            "Given an outcome, trace backward to identify the chain of causes that led to it.",
            "Describe a goal and work backward to identify what must be true now to achieve it.",
            "Take a historical event and analyze what factors in the past made it inevitable."
        ],
        'lindblad': [
            "Take this noisy, unstructured text and find the stable, coherent meaning within it.",
            "Extract the key signal from this chaotic information stream.",
            "Transform this messy input into organized, stable output."
        ],
        'qualia': [
            "Describe not just what something is, but what it feels like to experience it.",
            "Capture the subjective, qualitative dimension of an experience.",
            "Express the ineffable quality of a sensation or emotion."
        ],
        'emergent': [
            "Identify what emergent properties arise when simple components combine into a complex system.",
            "Describe how the whole becomes more than the sum of its parts.",
            "Find the 'frozen flows' - the stable patterns that persist across differentiation."
        ]
    }

    examples = []
    prompts = paradigm_prompts.get(paradigm, paradigm_prompts['semantic_phase'])

    for i in range(num_examples):
        prompt = prompts[i % len(prompts)]

        try:
            if pattern == 'think':
                result = await cluster.swarm_think(prompt)
            elif pattern == 'explore':
                result = await cluster.swarm_explore(prompt, num_explorers=3)
            elif pattern == 'debate':
                result = await cluster.swarm_debate(
                    prompt,
                    positions=["Pro", "Con", "Synthesis"]
                )
            else:
                result = await cluster.swarm_think(prompt)

            if result.get('status') == 'SUCCESS':
                answer = result.get('answer') or result.get('synthesis') or result.get('full_response', '')
                examples.append({
                    'input': prompt,
                    'output': answer,
                    'paradigm': paradigm,
                    'metadata': {
                        'pattern': pattern,
                        'index': i
                    }
                })
                print(f"  Generated {paradigm} example {i+1}/{num_examples}")
            else:
                print(f"  Warning: Failed to generate example {i+1}")

        except Exception as e:
            print(f"  Error generating example {i+1}: {e}")

        # Small delay to avoid overwhelming the swarm
        await asyncio.sleep(0.5)

    return examples


async def main():
    args = parse_args()

    print("=" * 60)
    print("QLLM Training Data Generator")
    print("=" * 60)

    # Check swarm status
    swarm_online = await check_swarm_status()

    if args.check_swarm:
        return

    if not swarm_online:
        print("\nWarning: Swarm controller is offline.")
        print("Will generate data using local templates only.")
        use_swarm = False
    else:
        use_swarm = True
        print("\nSwarm controller connected!")

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output)

    cluster = ClusterManager() if use_swarm else None

    if args.paradigm:
        # Generate for specific paradigm
        paradigms = [args.paradigm]
    else:
        # Generate for all paradigms
        paradigms = ['semantic_phase', 'retrocausal', 'lindblad', 'qualia', 'emergent']

    all_examples = []

    for paradigm in paradigms:
        print(f"\nGenerating {args.examples} examples for {paradigm}...")

        if use_swarm:
            examples = await generate_with_custom_pattern(
                paradigm,
                args.examples,
                args.swarm_pattern,
                cluster
            )
        else:
            # Use local generator without swarm
            generator = DatasetGenerator(output_dir=str(output_dir))

            # Get examples from templates
            if paradigm == 'semantic_phase':
                examples = await generator.generate_semantic_phase_examples(args.examples)
            elif paradigm == 'retrocausal':
                examples = await generator.generate_retrocausal_examples(args.examples)
            elif paradigm == 'lindblad':
                examples = await generator.generate_lindblad_examples(args.examples)
            elif paradigm == 'qualia':
                examples = await generator.generate_qualia_examples(args.examples)
            elif paradigm == 'emergent':
                examples = await generator.generate_emergent_examples(args.examples)

            examples = [e.to_dict() for e in examples]

        all_examples.extend(examples)
        print(f"  Generated {len(examples)} examples for {paradigm}")

    # Save to file
    print(f"\nSaving {len(all_examples)} total examples to {output_file}...")
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    # Also save paradigm-specific files
    for paradigm in paradigms:
        paradigm_examples = [e for e in all_examples if e['paradigm'] == paradigm]
        paradigm_file = output_dir / f"{paradigm}_data.jsonl"
        with open(paradigm_file, 'w') as f:
            for example in paradigm_examples:
                f.write(json.dumps(example) + '\n')
        print(f"  Saved {len(paradigm_examples)} {paradigm} examples to {paradigm_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal examples: {len(all_examples)}")
    for paradigm in paradigms:
        count = len([e for e in all_examples if e['paradigm'] == paradigm])
        print(f"  {paradigm}: {count}")
    print(f"\nOutput file: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
