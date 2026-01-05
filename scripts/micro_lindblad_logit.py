#!/usr/bin/env python3
"""
Micro-experiment: Lindblad + Logit-Consistency
==============================================

Runs a focused 600-step experiment comparing:
- Baseline (no paradigm losses)
- Lindblad with logit-consistency (new filtered approach)
- Lindblad + SemanticPhase (combined)

Evaluates JS divergence every 200 steps to track robustness.
Uses StratifiedParadigmSampler to guarantee paradigm presence from step 1.
"""

import argparse
import json
import torch
import sys
import os
from pathlib import Path
from datetime import datetime

# Import the main training infrastructure
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

from run_proof_ablation_v2 import (
    PilotDatasetV2, StratifiedParadigmSampler, ParadigmAwareTrainer, ActivationLogger
)
from qllm.core.config import TrainingConfig
from qllm.core.model import QLLM
from qllm.core.config import QLLMConfig


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_js_divergence(model, eval_items, tokenizer_encode, device='cuda'):
    """Compute average JS divergence on Lindblad eval items."""
    model.eval()
    js_scores = []

    for item in eval_items[:20]:  # Limit for speed
        clean_text = item.get('clean_text', '')[:80]
        noisy_text = item.get('noisy_text', '')[:80]

        try:
            with torch.no_grad():
                clean_ids = tokenizer_encode(clean_text)[:60]
                noisy_ids = tokenizer_encode(noisy_text)[:60]

                # Pad to same length
                max_len = max(len(clean_ids), len(noisy_ids))
                clean_ids = clean_ids + [0] * (max_len - len(clean_ids))
                noisy_ids = noisy_ids + [0] * (max_len - len(noisy_ids))

                clean_tensor = torch.tensor([clean_ids], device=device)
                noisy_tensor = torch.tensor([noisy_ids], device=device)

                clean_out = model(input_ids=clean_tensor)
                noisy_out = model(input_ids=noisy_tensor)

                clean_logits = clean_out.get('logits')
                noisy_logits = noisy_out.get('logits')

                if clean_logits is not None and noisy_logits is not None:
                    n_pos = min(8, clean_logits.size(1))
                    clean_final = clean_logits[0, -n_pos:, :]
                    noisy_final = noisy_logits[0, -n_pos:, :]

                    # JS divergence
                    p = torch.nn.functional.softmax(clean_final, dim=-1)
                    q = torch.nn.functional.softmax(noisy_final, dim=-1)
                    m = 0.5 * (p + q)

                    kl_pm = (p * (p / m).log()).sum(dim=-1)
                    kl_qm = (q * (q / m).log()).sum(dim=-1)
                    js = 0.5 * (kl_pm + kl_qm)

                    js_scores.append(js.mean().item())
        except:
            pass

    model.train()
    return sum(js_scores) / len(js_scores) if js_scores else 1.0


def simple_tokenizer_encode(text):
    """Simple character-level tokenizer."""
    char_to_id = {chr(i): (i - 31) for i in range(32, 127)}
    char_to_id['\n'] = 96
    char_to_id['\t'] = 97

    ids = []
    for c in text:
        if c in char_to_id:
            ids.append(char_to_id[c])
        else:
            ids.append(1)
    return ids


def run_micro_experiment(
    seed: int = 42,
    steps: int = 600,
    eval_every: int = 200,
    device: str = 'cuda'
):
    """Run the 3-way micro-experiment."""
    print("=" * 60)
    print("  MICRO-EXPERIMENT: 3-Way Comparison")
    print("  Baseline vs Lindblad vs Lindblad+SemanticPhase")
    print("=" * 60)

    # Load eval pack
    eval_pack_path = Path("paradigm_factory/output/20260102/eval_pack_20260102.json")
    if eval_pack_path.exists():
        with open(eval_pack_path) as f:
            eval_pack = json.load(f)
        lindblad_items = eval_pack.get('lindblad_items', [])
        print(f"Loaded {len(lindblad_items)} Lindblad eval items")
    else:
        print("Warning: No eval pack found, will skip JS eval")
        lindblad_items = []

    # Create dataset
    data_path = Path("data/paradigm_training_enriched.jsonl")
    if not data_path.exists():
        data_path = Path("data/pilot_paradigm_data.jsonl")

    dataset = PilotDatasetV2(
        data_path=str(data_path),
        max_length=128,
        vocab_size=1000
    )

    # Model config
    model_config = QLLMConfig(
        vocab_size=dataset.vocab_size,
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

    # Training config
    train_config = TrainingConfig(
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_steps=steps
    )

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"proof_ablation_runs/micro_3way_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3-way comparison: baseline, EMA distillation, EMA + semantic
    # Testing new confidence-gated EMA distillation approach
    ablations = {
        'baseline': {'losses': [], 'lindblad_schedule': 'constant', 'lindblad_weight': 0.05},
        'ema_distill': {'losses': ['lindblad_invariance'], 'lindblad_schedule': 'constant', 'lindblad_weight': 0.10},
        'ema_semantic': {'losses': ['lindblad_invariance', 'phase_contrastive'], 'lindblad_schedule': 'constant', 'lindblad_weight': 0.10}
    }

    for ablation_name, config in ablations.items():
        enabled_losses = config['losses']
        lindblad_schedule = config['lindblad_schedule']
        lindblad_weight = config.get('lindblad_weight', 0.05)

        print(f"\n{'='*60}")
        print(f"Running: {ablation_name}")
        print(f"Enabled losses: {enabled_losses if enabled_losses else 'none'}")
        print(f"Lindblad schedule: {lindblad_schedule}, weight: {lindblad_weight}")
        print('='*60)

        # Set seed for reproducibility
        set_seed(seed)

        # Create fresh model
        model = QLLM(model_config)
        logger = ActivationLogger()

        # Create trainer with stratified sampler
        trainer = ParadigmAwareTrainer(
            model=model,
            train_dataset=dataset,
            train_config=train_config,
            device=device,
            activation_logger=logger,
            enabled_losses=enabled_losses,
            semantic_phase_schedule='three_stage' if 'phase_contrastive' in enabled_losses else 'none',
            lindblad_schedule=lindblad_schedule,
            lindblad_base_weight=lindblad_weight,
            max_steps=steps,
            seed=seed
        )

        # Training with periodic JS eval
        js_history = []
        dataloader_iter = iter(trainer.dataloader)

        for step in range(1, steps + 1):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(trainer.dataloader)
                batch = next(dataloader_iter)

            # Train step
            loss_dict = trainer.train_step(batch, step)
            loss = loss_dict.get('loss', 0.0)

            if step % eval_every == 0:
                # Evaluate JS divergence
                if lindblad_items:
                    js_div = compute_js_divergence(
                        model, lindblad_items, simple_tokenizer_encode, device
                    )
                else:
                    js_div = 0.0

                js_history.append({
                    'step': step,
                    'js_divergence': js_div,
                    'loss': loss
                })

                # Get first_step_with from logger
                first_step = logger.first_step_with
                print(f"Step {step} | Loss: {loss:.4f} | JS: {js_div:.4f} | first_step: {first_step}")

                # Save checkpoint
                checkpoint_path = output_dir / ablation_name / f"step_{step}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'js_divergence': js_div,
                    'first_step_with': first_step
                }, checkpoint_path)

        # Save activation logs
        logger.save(output_dir / ablation_name / "activation_logs.json")

        results[ablation_name] = {
            'js_history': js_history,
            'final_js': js_history[-1]['js_divergence'] if js_history else 0.0,
            'first_step_with': logger.first_step_with,
            'cumulative_loss': logger.cumulative_loss
        }

    # Summary table
    print("\n" + "=" * 90)
    print("MICRO-EXPERIMENT RESULTS (JS Divergence - lower is better)")
    print("=" * 90)
    # Print header with actual ablation names
    header = f"{'Step':<8}"
    for name in ablations.keys():
        header += f" {name[:12]:<12}"
    header += f" {'Best':<12}"
    print(header)
    print("-" * 90)

    # Get ablation names dynamically
    ablation_names = list(results.keys())

    for i in range(len(results[ablation_names[0]]['js_history'])):
        step = results[ablation_names[0]]['js_history'][i]['step']
        js_values = {}
        for name in ablation_names:
            js_values[name] = results[name]['js_history'][i]['js_divergence']

        best_name = min(js_values, key=js_values.get)

        row = f"{step:<8}"
        for name in ablation_names:
            row += f" {js_values[name]:<12.4f}"
        row += f" {best_name:<12}"
        print(row)

    # First step analysis
    print("\n" + "-" * 70)
    print("FIRST STEP WITH ACTIVATION (eliminates warmup gap):")
    for ablation_name, data in results.items():
        fsw = data['first_step_with']
        print(f"  {ablation_name}: phase={fsw.get('phase_contrastive')}, lindblad={fsw.get('lindblad_invariance')}, qualia={fsw.get('qualia_diversity')}")

    # Cumulative loss contribution
    print("\nCUMULATIVE LOSS CONTRIBUTION:")
    for ablation_name, data in results.items():
        cum = data['cumulative_loss']
        print(f"  {ablation_name}: phase={cum.get('phase_contrastive', 0):.4f}, lindblad={cum.get('lindblad_invariance', 0):.4f}, qualia={cum.get('qualia_diversity', 0):.4f}")

    # Save results
    results_path = output_dir / "micro_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Micro-experiment: 3-Way Comparison")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    run_micro_experiment(
        seed=args.seed,
        steps=args.steps,
        eval_every=args.eval_every,
        device=args.device
    )


if __name__ == "__main__":
    main()
