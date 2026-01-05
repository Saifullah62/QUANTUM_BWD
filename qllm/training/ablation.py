"""
Ablation Runner - One Paradigm at a Time
=========================================

Validates each paradigm's contribution independently:
- Run baseline (no paradigm losses)
- Toggle each paradigm ON individually
- Compare win conditions

If "polysemy" improves ONLY when SemanticPhase is on → real signal.
If "consistency" improves ONLY when Lindblad is on → real signal.
If "retrocausal" improves while leak stays clean → credibility proof.
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import copy

from ..core.config import QLLMConfig, TrainingConfig
from ..core.model import QLLM
from .trainer import QLLMTrainer


@dataclass
class AblationResult:
    """Result from a single ablation run"""
    paradigm: str  # "baseline", "semantic_phase", "lindblad", etc.
    enabled: bool
    steps: int
    final_loss: float
    win_conditions: Dict[str, float]
    paradigm_metrics: Dict[str, float]
    duration_seconds: float


@dataclass
class ParadigmLossConfig:
    """Controls which paradigm losses are active"""
    phase_coherence: bool = False
    magnitude_reg: bool = False
    lindblad_consistency: bool = False
    lipschitz: bool = False
    qualia: bool = False
    qualia_self_supervised: bool = False
    retrocausal_leak: bool = False
    attractor_energy: bool = False

    @classmethod
    def baseline(cls) -> 'ParadigmLossConfig':
        """All paradigm losses OFF"""
        return cls()

    @classmethod
    def all_enabled(cls) -> 'ParadigmLossConfig':
        """All paradigm losses ON"""
        return cls(
            phase_coherence=True,
            magnitude_reg=True,
            lindblad_consistency=True,
            lipschitz=True,
            qualia=True,
            qualia_self_supervised=True,
            retrocausal_leak=True,
            attractor_energy=True
        )

    @classmethod
    def semantic_phase_only(cls) -> 'ParadigmLossConfig':
        """Only SemanticPhase losses"""
        return cls(phase_coherence=True, magnitude_reg=True)

    @classmethod
    def lindblad_only(cls) -> 'ParadigmLossConfig':
        """Only Lindblad losses"""
        return cls(lindblad_consistency=True, lipschitz=True)

    @classmethod
    def qualia_only(cls) -> 'ParadigmLossConfig':
        """Only Qualia losses"""
        return cls(qualia=True, qualia_self_supervised=True)

    @classmethod
    def retrocausal_only(cls) -> 'ParadigmLossConfig':
        """Only Retrocausal losses"""
        return cls(retrocausal_leak=True)

    @classmethod
    def emergent_only(cls) -> 'ParadigmLossConfig':
        """Only Emergent losses"""
        return cls(attractor_energy=True)


class AblationRunner:
    """
    Runs ablation studies to validate each paradigm independently.

    Usage:
        runner = AblationRunner(model_config, train_config, dataset)
        results = runner.run_full_ablation(steps_per_run=500)
        runner.print_comparison(results)
    """

    def __init__(
        self,
        model_config: QLLMConfig,
        train_config: TrainingConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        output_dir: str = "./ablation_runs"
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single_ablation(
        self,
        paradigm_name: str,
        loss_config: ParadigmLossConfig,
        steps: int = 500,
        seed: int = 42
    ) -> AblationResult:
        """Run a single ablation with specific paradigm config."""
        import time
        start_time = time.time()

        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Fresh model for each ablation
        model = QLLM(self.model_config)

        # Modified training config
        config = copy.deepcopy(self.train_config)
        config.max_steps = steps

        # Create trainer with masked losses
        trainer = MaskedLossTrainer(
            model=model,
            train_config=config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            output_dir=str(self.output_dir / paradigm_name),
            loss_config=loss_config
        )

        print(f"\n{'='*60}")
        print(f"ABLATION: {paradigm_name}")
        print(f"Losses enabled: {[k for k, v in asdict(loss_config).items() if v]}")
        print(f"{'='*60}")

        # Train
        trainer.train()

        # Collect results
        win_conditions = trainer.validate_win_conditions()
        paradigm_metrics = trainer.compute_paradigm_metrics()

        duration = time.time() - start_time

        return AblationResult(
            paradigm=paradigm_name,
            enabled=any(asdict(loss_config).values()),
            steps=steps,
            final_loss=trainer.state.best_loss,
            win_conditions=win_conditions,
            paradigm_metrics=paradigm_metrics,
            duration_seconds=duration
        )

    def run_full_ablation(self, steps_per_run: int = 500) -> List[AblationResult]:
        """
        Run complete ablation study: baseline + each paradigm individually.

        Returns list of AblationResult for comparison.
        """
        results = []

        # 1. Baseline (no paradigm losses)
        results.append(self.run_single_ablation(
            "baseline",
            ParadigmLossConfig.baseline(),
            steps=steps_per_run
        ))

        # 2. SemanticPhase only
        results.append(self.run_single_ablation(
            "semantic_phase_only",
            ParadigmLossConfig.semantic_phase_only(),
            steps=steps_per_run
        ))

        # 3. Lindblad only
        results.append(self.run_single_ablation(
            "lindblad_only",
            ParadigmLossConfig.lindblad_only(),
            steps=steps_per_run
        ))

        # 4. Qualia only
        results.append(self.run_single_ablation(
            "qualia_only",
            ParadigmLossConfig.qualia_only(),
            steps=steps_per_run
        ))

        # 5. Retrocausal only
        results.append(self.run_single_ablation(
            "retrocausal_only",
            ParadigmLossConfig.retrocausal_only(),
            steps=steps_per_run
        ))

        # 6. Emergent only
        results.append(self.run_single_ablation(
            "emergent_only",
            ParadigmLossConfig.emergent_only(),
            steps=steps_per_run
        ))

        # 7. All paradigms
        results.append(self.run_single_ablation(
            "all_paradigms",
            ParadigmLossConfig.all_enabled(),
            steps=steps_per_run
        ))

        # Save results
        self.save_results(results)

        return results

    def save_results(self, results: List[AblationResult]):
        """Save ablation results to JSON."""
        results_dict = [asdict(r) for r in results]
        output_path = self.output_dir / f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {output_path}")

    def print_comparison(self, results: List[AblationResult]):
        """Print comparison table of ablation results."""
        print("\n" + "="*80)
        print("ABLATION COMPARISON")
        print("="*80)

        # Header
        print(f"{'Paradigm':<25} {'Loss':>8} {'Polysemy':>10} {'Consistency':>12} {'Retrocausal':>12}")
        print("-"*80)

        baseline = results[0] if results else None

        for r in results:
            polysemy = r.win_conditions.get('polysemy_resolution', 0)
            consistency = r.win_conditions.get('noise_invariance', 0)
            retrocausal = r.win_conditions.get('retrocausal_reasoning', 0)

            # Compute deltas from baseline
            if baseline and r.paradigm != "baseline":
                poly_delta = polysemy - baseline.win_conditions.get('polysemy_resolution', 0)
                cons_delta = consistency - baseline.win_conditions.get('noise_invariance', 0)
                retro_delta = retrocausal - baseline.win_conditions.get('retrocausal_reasoning', 0)
                delta_str = f" (Δ: {poly_delta:+.3f}, {cons_delta:+.3f}, {retro_delta:+.3f})"
            else:
                delta_str = ""

            print(f"{r.paradigm:<25} {r.final_loss:>8.4f} {polysemy:>10.4f} {consistency:>12.4f} {retrocausal:>12.4f}")

        print("="*80)

        # Signal detection
        print("\nSIGNAL DETECTION:")
        if baseline:
            for r in results[1:]:
                improvements = []
                if r.win_conditions.get('polysemy_resolution', 0) > baseline.win_conditions.get('polysemy_resolution', 0) + 0.01:
                    improvements.append("polysemy↑")
                if r.win_conditions.get('noise_invariance', 0) > baseline.win_conditions.get('noise_invariance', 0) + 0.01:
                    improvements.append("consistency↑")
                if r.win_conditions.get('retrocausal_reasoning', 0) > baseline.win_conditions.get('retrocausal_reasoning', 0) + 0.01:
                    improvements.append("retrocausal↑")

                if improvements:
                    print(f"  {r.paradigm}: {', '.join(improvements)} ✓")
                else:
                    print(f"  {r.paradigm}: no significant improvement")


class MaskedLossTrainer(QLLMTrainer):
    """
    Trainer with selective paradigm loss masking.

    Allows toggling individual paradigm losses on/off for ablation.
    """

    def __init__(
        self,
        model: QLLM,
        train_config: TrainingConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        output_dir: str = "./outputs",
        loss_config: Optional[ParadigmLossConfig] = None
    ):
        super().__init__(model, train_config, train_dataset, eval_dataset, output_dir)
        self.loss_config = loss_config or ParadigmLossConfig.all_enabled()

    def compute_paradigm_losses(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        hidden_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute paradigm losses with masking based on loss_config."""
        # Get all losses from parent
        all_losses = super().compute_paradigm_losses(outputs, batch, hidden_states)

        # Mask based on config
        masked_losses = {}
        config_dict = asdict(self.loss_config)

        for loss_name, loss_value in all_losses.items():
            # Map loss names to config keys
            config_key = loss_name.replace('loss_', '')
            if config_dict.get(config_key, False):
                masked_losses[loss_name] = loss_value
            # If not in config, skip this loss

        return masked_losses


class LossRampingScheduler:
    """
    Gradually ramps in paradigm losses to prevent early instability.

    The base language objective settles first, then paradigm losses
    are ramped in over a specified number of steps.
    """

    def __init__(
        self,
        warmup_steps: int = 500,
        ramp_steps: int = 1000,
        paradigm_order: Optional[List[str]] = None
    ):
        self.warmup_steps = warmup_steps
        self.ramp_steps = ramp_steps
        self.paradigm_order = paradigm_order or [
            'phase_coherence',
            'lindblad_consistency',
            'qualia',
            'retrocausal_leak',
            'attractor_energy'
        ]

    def get_loss_weights(self, step: int, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Get loss weights for current step.

        Before warmup: all paradigm weights = 0
        During ramp: linear increase
        After ramp: full weights
        """
        if step < self.warmup_steps:
            # Pure language modeling warmup
            return {k: 0.0 for k in base_weights}

        ramp_progress = min(1.0, (step - self.warmup_steps) / self.ramp_steps)

        # Optional: stagger paradigms
        weights = {}
        num_paradigms = len(self.paradigm_order)
        # Each paradigm gets an equal fraction of the ramp period
        paradigm_ramp_fraction = 1.0 / max(1, num_paradigms)

        for i, paradigm in enumerate(self.paradigm_order):
            if paradigm in base_weights:
                # Each paradigm starts ramping after its predecessors
                paradigm_start = i * paradigm_ramp_fraction * 0.5  # Stagger by 50% of their slot
                paradigm_progress = max(0, (ramp_progress - paradigm_start) / (1 - paradigm_start))
                paradigm_progress = min(1.0, paradigm_progress)
                weights[paradigm] = base_weights[paradigm] * paradigm_progress

        # Add any weights not in the order
        for k, v in base_weights.items():
            if k not in weights:
                weights[k] = v * ramp_progress

        return weights

    def get_schedule_summary(self, total_steps: int) -> str:
        """Print schedule summary."""
        lines = [
            f"Loss Ramping Schedule (total {total_steps} steps):",
            f"  Steps 0-{self.warmup_steps}: Pure language modeling",
            f"  Steps {self.warmup_steps}-{self.warmup_steps + self.ramp_steps}: Paradigm loss ramp-in",
        ]
        for i, p in enumerate(self.paradigm_order):
            start = self.warmup_steps + int(i * 0.1 * self.ramp_steps)
            lines.append(f"    - {p}: ramps from step {start}")
        lines.append(f"  Steps {self.warmup_steps + self.ramp_steps}+: Full paradigm losses")
        return "\n".join(lines)


# CLI for ablation
def main():
    """Run ablation study from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="QLLM Ablation Study")
    parser.add_argument('--steps', type=int, default=500, help='Steps per ablation run')
    parser.add_argument('--output-dir', default='./ablation_runs', help='Output directory')
    parser.add_argument('--paradigm', default=None, help='Run single paradigm (or "all")')

    args = parser.parse_args()

    # Would need actual dataset and config here
    print("Ablation runner ready.")
    print("Use: runner = AblationRunner(config, train_config, dataset)")
    print("     results = runner.run_full_ablation(steps_per_run=500)")


if __name__ == "__main__":
    main()
