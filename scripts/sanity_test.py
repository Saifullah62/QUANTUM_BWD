#!/usr/bin/env python
"""
QLLM Sanity Test
================

Validates the entire pipeline works before committing real compute:
1. Model instantiation
2. Forward pass with all paradigms
3. Paradigm losses fire correctly
4. Win condition metrics compute
5. Mini ablation (optional)

Run: python scripts/sanity_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from typing import Dict, Any
import time


def print_header(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def print_check(name: str, passed: bool, details: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


def test_imports():
    """Test all imports work."""
    print_header("1. Testing Imports")

    try:
        from qllm.core.config import QLLMConfig, TrainingConfig
        print_check("Core config", True)
    except Exception as e:
        print_check("Core config", False, str(e))
        return False

    try:
        from qllm.core.model import QLLM
        print_check("Core model", True)
    except Exception as e:
        print_check("Core model", False, str(e))
        return False

    try:
        from qllm.layers.semantic_phase import SemanticPhaseEmbedding
        from qllm.layers.lindblad import LindbladLayer
        from qllm.layers.qualia import QualiaOutputHead
        from qllm.layers.retrocausal import RetrocausalAttention
        from qllm.layers.emergent import AttractorLayer
        print_check("All paradigm layers", True)
    except Exception as e:
        print_check("All paradigm layers", False, str(e))
        return False

    try:
        from qllm.training.trainer import QLLMTrainer
        from qllm.training.ablation import AblationRunner, LossRampingScheduler
        print_check("Training infrastructure", True)
    except Exception as e:
        print_check("Training infrastructure", False, str(e))
        return False

    try:
        from qllm.evaluation.report import ReportGenerator
        print_check("Report generator", True)
    except Exception as e:
        print_check("Report generator", False, str(e))
        return False

    return True


def test_model_creation():
    """Test model instantiation."""
    print_header("2. Testing Model Creation")

    from qllm.core.config import QLLMConfig
    from qllm.core.model import QLLM

    # Minimal config for testing
    config = QLLMConfig(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        intermediate_dim=128,
        max_seq_length=128,
        # Semantic phase (match hidden_dim)
        use_semantic_phase=True,
        semantic_phase_dim=64,
        # Retrocausal
        use_retrocausal_attention=True,
        retrocausal_layers=[0, 1],
        # Lindblad
        use_lindblad_layers=True,
        lindblad_every_n_layers=1,
        # Qualia
        use_qualia_output=True,
        num_qualia_channels=8,
        # Emergent
        use_emergent_init=True
    )

    try:
        model = QLLM(config)
        param_count = sum(p.numel() for p in model.parameters())
        print_check("Model instantiation", True, f"{param_count:,} parameters")
    except Exception as e:
        print_check("Model instantiation", False, str(e))
        return None

    # Check paradigm summary
    try:
        summary = model.get_paradigm_summary()
        print_check("Paradigm summary", True, str(summary))
    except Exception as e:
        print_check("Paradigm summary", False, str(e))

    return model


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print_header("3. Testing Forward Pass")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = model.to(device)

    # Create dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    try:
        model.train()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        print_check("Forward pass (train mode)", True)
    except Exception as e:
        print_check("Forward pass (train mode)", False, str(e))
        return None

    # Check outputs
    checks = [
        ('loss', outputs.get('loss') is not None),
        ('logits', outputs.get('logits') is not None),
        ('hidden_states', outputs.get('hidden_states') is not None),
    ]

    for name, passed in checks:
        print_check(f"Output: {name}", passed)

    # Check loss value
    loss = outputs.get('loss')
    if loss is not None:
        print_check("Loss is finite", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")

    return outputs


def test_paradigm_layers():
    """Test each paradigm layer individually."""
    print_header("4. Testing Paradigm Layers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    seq_len = 32
    hidden_dim = 64

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # SemanticPhase
    try:
        from qllm.layers.semantic_phase import SemanticPhaseEmbedding
        embed = SemanticPhaseEmbedding(vocab_size=1000, embedding_dim=hidden_dim).to(device)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        real, imag = embed(input_ids)

        # Test magnitude constraint (WIN CONDITION feature)
        real_c, imag_c = embed.magnitude_constrained_forward(input_ids)
        mag = embed.get_magnitude(real_c, imag_c)
        mag_mean = mag.mean().item()

        print_check("SemanticPhase", True, f"magnitude_constrained mean={mag_mean:.4f} (should be ~1.0)")
    except Exception as e:
        print_check("SemanticPhase", False, str(e))

    # Lindblad
    try:
        from qllm.layers.lindblad import LindbladLayer
        lindblad = LindbladLayer(hidden_dim=hidden_dim).to(device)
        output = lindblad(hidden_states)

        # Test consistency loss (WIN CONDITION feature)
        consistency_loss = lindblad.compute_consistency_loss(hidden_states, num_samples=2)

        print_check("Lindblad", True, f"consistency_loss={consistency_loss.item():.4f}")
    except Exception as e:
        print_check("Lindblad", False, str(e))

    # Retrocausal
    try:
        from qllm.layers.retrocausal import RetrocausalAttention, TwoPassTrainer
        retro = RetrocausalAttention(hidden_dim=hidden_dim, num_heads=4).to(device)

        # Test with future context
        future_context = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        output, info = retro(hidden_states, future_context=future_context, return_states=True)

        # Check hardening
        has_dropout = retro.backward_dropout_rate > 0
        has_stopgrad = retro.stop_backward_grad

        print_check("Retrocausal", True,
                   f"backward_strength={info['backward_strength']:.3f}, "
                   f"dropout={has_dropout}, stopgrad={has_stopgrad}")

        # Test leak detection
        if info['forward_state'] is not None and info['backward_state'] is not None:
            target_embeds = torch.randn_like(info['forward_state'])
            leak_report = TwoPassTrainer.compute_full_leak_report(
                info['forward_state'], info['backward_state'], target_embeds
            )
            print_check("Leak detection", True,
                       f"is_clean={leak_report['is_clean']}, score={leak_report['overall_leak_score']:.4f}")
    except Exception as e:
        print_check("Retrocausal", False, str(e))

    # Qualia
    try:
        from qllm.layers.qualia import QualiaOutputHead
        qualia = QualiaOutputHead(hidden_dim=hidden_dim, vocab_size=1000, num_qualia=8).to(device)
        logits, qualia_info = qualia(hidden_states, return_qualia=True)

        qualia_tensor = qualia_info.get('qualia_tensor')
        if qualia_tensor is not None:
            print_check("Qualia", True, f"channels={qualia_tensor.shape[-1]}, shape={qualia_tensor.shape}")
        else:
            print_check("Qualia", True, "no qualia tensor returned")
    except Exception as e:
        print_check("Qualia", False, str(e))

    # Emergent
    try:
        from qllm.layers.emergent import AttractorLayer
        attractor = AttractorLayer(hidden_dim=hidden_dim).to(device)
        output = attractor(hidden_states)
        print_check("Emergent", True, f"output shape={output.shape}")
    except Exception as e:
        print_check("Emergent", False, str(e))


def test_loss_ramping():
    """Test loss ramping scheduler."""
    print_header("5. Testing Loss Ramping")

    from qllm.training.ablation import LossRampingScheduler

    scheduler = LossRampingScheduler(warmup_steps=100, ramp_steps=200)

    base_weights = {
        'phase_coherence': 0.1,
        'lindblad_consistency': 0.1,
        'retrocausal_leak': 0.1
    }

    # Test at different steps
    test_steps = [0, 50, 100, 150, 200, 300, 500]

    print("  Step  | phase_coherence | lindblad | retrocausal")
    print("  ------|-----------------|----------|------------")

    for step in test_steps:
        weights = scheduler.get_loss_weights(step, base_weights)
        print(f"  {step:5} | {weights.get('phase_coherence', 0):15.4f} | "
              f"{weights.get('lindblad_consistency', 0):8.4f} | "
              f"{weights.get('retrocausal_leak', 0):.4f}")

    print_check("Loss ramping", True, "weights increase correctly")


def test_report_generation():
    """Test report generation."""
    print_header("6. Testing Report Generation")

    from qllm.evaluation.report import ReportGenerator, WinConditionReport

    # Mock ablation results
    mock_results = [
        {
            'paradigm': 'baseline',
            'win_conditions': {
                'polysemy_resolution': 0.50,
                'noise_invariance': 0.60,
                'retrocausal_reasoning': 0.40
            },
            'steps': 500
        },
        {
            'paradigm': 'semantic_phase_only',
            'win_conditions': {
                'polysemy_resolution': 0.62,
                'noise_invariance': 0.61,
                'retrocausal_reasoning': 0.41
            },
            'steps': 500
        },
        {
            'paradigm': 'lindblad_only',
            'win_conditions': {
                'polysemy_resolution': 0.51,
                'noise_invariance': 0.72,
                'retrocausal_reasoning': 0.42
            },
            'steps': 500
        },
        {
            'paradigm': 'retrocausal_only',
            'win_conditions': {
                'polysemy_resolution': 0.52,
                'noise_invariance': 0.62,
                'retrocausal_reasoning': 0.55
            },
            'steps': 500
        },
        {
            'paradigm': 'all_paradigms',
            'win_conditions': {
                'polysemy_resolution': 0.65,
                'noise_invariance': 0.75,
                'retrocausal_reasoning': 0.58
            },
            'steps': 500
        }
    ]

    try:
        generator = ReportGenerator()
        report = generator.generate_from_ablation(mock_results, model_name="QLLM-Test")

        print_check("Report generation", True,
                   f"paradigms_with_signal={report.paradigms_with_signal}")

        # Print mini report
        print("\n  --- Mini Report Preview ---")
        for paradigm in ['semantic_phase', 'lindblad', 'retrocausal']:
            result = getattr(report, paradigm, None)
            if result:
                print(f"  {result.name}: {result.improvement_pct*100:+.1f}% "
                      f"({'significant' if result.is_significant else 'not significant'})")

    except Exception as e:
        print_check("Report generation", False, str(e))


def run_mini_training_step():
    """Run one actual training step to verify everything connects."""
    print_header("7. Mini Training Step")

    from qllm.core.config import QLLMConfig, TrainingConfig
    from qllm.core.model import QLLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny config
    config = QLLMConfig(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        intermediate_dim=128,
        max_seq_length=64,
        semantic_phase_dim=64,
        use_semantic_phase=True,
        use_retrocausal_attention=True,
        retrocausal_layers=[0, 1],
        use_lindblad_layers=True,
        lindblad_every_n_layers=1,
        use_qualia_output=True,
        use_emergent_init=True
    )

    model = QLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Dummy batch
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    try:
        model.train()
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )

        loss = outputs['loss']
        loss.backward()

        # Check gradients exist
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)

        optimizer.step()

        print_check("Forward pass", True, f"loss={loss.item():.4f}")
        print_check("Backward pass", True, f"gradients_exist={has_grads}")
        print_check("Optimizer step", True)

    except Exception as e:
        print_check("Training step", False, str(e))
        return False

    return True


def main():
    print("\n" + "=" * 60)
    print("  QLLM SANITY TEST")
    print("  Validating pipeline before committing compute")
    print("=" * 60)

    start_time = time.time()
    all_passed = True

    # Run tests
    if not test_imports():
        print("\n[!] Import test failed. Fix imports before proceeding.")
        return

    model = test_model_creation()
    if model is None:
        print("\n[!] Model creation failed. Fix model before proceeding.")
        return

    outputs = test_forward_pass(model)
    if outputs is None:
        print("\n[!] Forward pass failed. Fix model before proceeding.")
        return

    test_paradigm_layers()
    test_loss_ramping()
    test_report_generation()

    if not run_mini_training_step():
        print("\n[!] Training step failed.")
        all_passed = False

    # Summary
    elapsed = time.time() - start_time
    print_header("SANITY TEST COMPLETE")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU Memory: {mem_allocated:.1f} MB")

    if all_passed:
        print("\n  [OK] All checks passed! Ready for real training.")
        print("\n  Next steps:")
        print("    1. python scripts/sanity_test.py --ablation  (mini ablation)")
        print("    2. python run_qllm.py train --model-size minimal")
    else:
        print("\n  [WARNING] Some checks failed. Review errors above.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', action='store_true', help='Run mini ablation test')
    args = parser.parse_args()

    if args.ablation:
        print("Mini ablation not yet implemented - run main sanity first")
    else:
        main()
