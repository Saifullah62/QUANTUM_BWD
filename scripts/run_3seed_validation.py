#!/usr/bin/env python3
"""
3-Seed Validation with Slack-Positive Criterion
================================================

Runs SemanticPhase training across 3 seeds and validates:
- "Slack crosses zero and stays positive" as success criterion
- Slack > 0 for 30-40%+ of late steps indicates robust margin

Usage:
    python scripts/run_3seed_validation.py --bundles data.jsonl --epochs 3
    python scripts/run_3seed_validation.py --bundles data.jsonl --seeds 42 123 456
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import training components
from qllm.core.config import TrainingConfig, QLLMConfig
from qllm.core.model import QLLM

import importlib.util
spec = importlib.util.spec_from_file_location("train_semantic_phase_v2",
    os.path.join(os.path.dirname(__file__), "train_semantic_phase_v2.py"))
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
SemanticPhaseTrainer = train_module.SemanticPhaseTrainerV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from paradigm_factory.bundle_dataset import BundleDataset, BundleBatch


@dataclass
class DifficultyBucketMetrics:
    """Metrics tracked per difficulty bucket."""
    msr_history: List[float] = field(default_factory=list)
    gap_history: List[float] = field(default_factory=list)
    final_msr: float = 0.0
    final_gap: float = 0.0
    first_positive_gap_step: Optional[int] = None


@dataclass
class SeedResult:
    """Results from a single seed run."""
    seed: int
    best_msr: float = 0.0
    best_msr_step: int = 0
    final_msr: float = 0.0
    final_gap: float = 0.0
    final_slack: float = 0.0

    # Slack tracking
    steps_with_positive_slack: int = 0
    late_steps_total: int = 0  # Steps in last 50% of training
    late_steps_positive_slack: int = 0
    slack_positive_rate: float = 0.0  # % of late steps with positive slack

    # First positive slack
    first_positive_slack_step: Optional[int] = None

    # Loss trajectory
    best_contrastive_loss: float = float('inf')
    final_contrastive_loss: float = 0.0

    # Difficulty bucket tracking (the "learning curve story")
    easy_metrics: DifficultyBucketMetrics = field(default_factory=DifficultyBucketMetrics)
    medium_metrics: DifficultyBucketMetrics = field(default_factory=DifficultyBucketMetrics)
    hard_metrics: DifficultyBucketMetrics = field(default_factory=DifficultyBucketMetrics)

    passed: bool = False  # Did this seed pass the criterion?


@dataclass
class ValidationReport:
    """Aggregate validation report across all seeds."""
    timestamp: str
    seeds: List[int]
    bundles_file: str
    epochs: int
    margin: float

    seed_results: List[SeedResult] = field(default_factory=list)

    # Aggregate metrics
    avg_best_msr: float = 0.0
    avg_final_slack: float = 0.0
    avg_slack_positive_rate: float = 0.0
    seeds_passed: int = 0
    seeds_failed: int = 0

    # Overall verdict
    passed: bool = False
    verdict: str = ""


class SlackTracker:
    """Tracks slack over training for validation, with separate easy/hard tracking."""

    def __init__(self, margin_easy: float = 0.05, margin_hard: float = 0.15):
        self.margin_easy = margin_easy
        self.margin_hard = margin_hard
        # Separate easy/hard tracking with correct margins
        self.easy_history: List[Dict] = []
        self.hard_history: List[Dict] = []
        # Combined history (for backwards compatibility)
        self.history: List[Dict] = []

    def record(self, step: int, gap_easy: float, gap_hard: float, msr_easy: float = 0, msr_hard: float = 0):
        """Record metrics with separate easy/hard gaps and correct margins."""
        slack_easy = gap_easy - self.margin_easy
        slack_hard = gap_hard - self.margin_hard

        self.easy_history.append({
            "step": step,
            "gap": gap_easy,
            "slack": slack_easy,
            "msr": msr_easy,
            "slack_positive": slack_easy > 0
        })

        self.hard_history.append({
            "step": step,
            "gap": gap_hard,
            "slack": slack_hard,
            "msr": msr_hard,
            "slack_positive": slack_hard > 0
        })

        # Combined entry for backwards compat (use hard slack as primary criterion)
        self.history.append({
            "step": step,
            "gap": gap_hard,  # Hard gap is the boss fight
            "slack": slack_hard,  # Hard slack is the certification target
            "msr": (msr_easy + msr_hard) / 2 if msr_easy and msr_hard else msr_easy or msr_hard,
            "slack_positive": slack_hard > 0
        })

    def record_by_difficulty(self, step: int, difficulty: str, gap: float, msr: float):
        """Record metrics for a specific difficulty bucket (deprecated, use record())."""
        margin = self.margin_easy if difficulty == "easy" else self.margin_hard
        slack = gap - margin
        entry = {"step": step, "gap": gap, "slack": slack, "msr": msr, "slack_positive": slack > 0}
        if difficulty == "easy":
            self.easy_history.append(entry)
        else:  # hard
            self.hard_history.append(entry)

    def get_difficulty_metrics(self) -> Dict[str, DifficultyBucketMetrics]:
        """Get final metrics per difficulty bucket."""
        metrics = {}
        for name, history in [("easy", self.easy_history),
                              ("medium", self.medium_history),
                              ("hard", self.hard_history)]:
            dm = DifficultyBucketMetrics()
            if history:
                dm.msr_history = [h["msr"] for h in history]
                dm.gap_history = [h["gap"] for h in history]
                dm.final_msr = history[-1]["msr"] if history else 0.0
                dm.final_gap = history[-1]["gap"] if history else 0.0
                # Find first step where gap > 0
                for h in history:
                    if h["gap"] > 0:
                        dm.first_positive_gap_step = h["step"]
                        break
            metrics[name] = dm
        return metrics

    def _analyze_late_steps(self, history: List[Dict], late_fraction: float = 0.5) -> Dict:
        """Analyze slack in the late portion of a history list."""
        if not history:
            return {"late_steps": 0, "positive_slack_steps": 0, "rate": 0.0, "avg_slack": 0.0}

        total_steps = len(history)
        late_start = int(total_steps * (1 - late_fraction))
        late_steps = history[late_start:]

        positive_slack_steps = sum(1 for s in late_steps if s["slack_positive"])
        avg_slack = sum(s["slack"] for s in late_steps) / len(late_steps) if late_steps else 0.0

        return {
            "late_steps": len(late_steps),
            "positive_slack_steps": positive_slack_steps,
            "rate": positive_slack_steps / len(late_steps) if late_steps else 0.0,
            "avg_slack": avg_slack
        }

    def get_late_steps_analysis(self, late_fraction: float = 0.5) -> Dict:
        """Analyze hard slack in the late portion (hard is the certification target)."""
        return self._analyze_late_steps(self.history, late_fraction)

    def get_late_steps_analysis_easy(self, late_fraction: float = 0.5) -> Dict:
        """Analyze easy slack in the late portion."""
        return self._analyze_late_steps(self.easy_history, late_fraction)

    def get_late_steps_analysis_hard(self, late_fraction: float = 0.5) -> Dict:
        """Analyze hard slack in the late portion (this is the boss fight!)."""
        return self._analyze_late_steps(self.hard_history, late_fraction)

    def get_first_positive_slack_step(self) -> Optional[int]:
        """Find first step where hard slack became positive."""
        for entry in self.history:
            if entry["slack_positive"]:
                return entry["step"]
        return None

    def get_first_positive_slack_step_easy(self) -> Optional[int]:
        """Find first step where easy slack became positive."""
        for entry in self.easy_history:
            if entry["slack_positive"]:
                return entry["step"]
        return None

    def get_first_positive_slack_step_hard(self) -> Optional[int]:
        """Find first step where hard slack became positive."""
        for entry in self.hard_history:
            if entry["slack_positive"]:
                return entry["step"]
        return None

    def get_best_msr(self) -> Tuple[float, int]:
        """Get best MSR and the step it occurred."""
        if not self.history:
            return 0.0, 0
        best = max(self.history, key=lambda x: x["msr"])
        return best["msr"], best["step"]


def run_single_seed(
    bundles_path: Path,
    seed: int,
    epochs: int = 3,
    margin: float = 0.05,
    batch_size: int = 8,
    lr: float = 1e-4,
    ce_weight: float = 0.3,
    late_stage_boost: float = 1.0,  # 1.0 = no boost (baseline)
    late_stage_start: float = 0.5,  # When slack ramp starts
    late_stage_ramp: float = 0.4,   # Ramp duration
    slack_weight: float = 3.0,      # v3: Standalone slack weight (competes with CE!)
    hard_neg_penalty_mult: float = 2.0,  # Hard-neg focus multiplier
    ce_late_taper: float = 1.0,  # Reduce CE late to prevent margin erosion
    track_killers: int = 0,  # Track K worst hard-neg violations per batch
    killer_log_every: int = 100,  # Log killer negatives every N steps
    use_phi_balance: bool = False,  # φ-based adaptive weight balancing
    phi_ema_decay: float = 0.99,  # EMA decay for φ-balance
    use_sense_head: bool = False,  # SenseHead attentive pooling
    sense_head_dim: int = 256,
    sense_head_dropout: float = 0.1,
    sense_head_entropy_weight: float = 0.1,
    sense_head_always_on_slack: float = 0.1,
    hard_neg_top_k: int = 1,
    hard_neg_temperature: float = 0.1
) -> SeedResult:
    """Run training for a single seed and collect metrics."""
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")

    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Count bundles to compute max_steps
    dataset = BundleDataset(bundles_path)
    num_bundles = len(dataset.bundles)
    steps_per_epoch = num_bundles // batch_size
    max_steps = epochs * steps_per_epoch
    print(f"  Bundles: {num_bundles}, Steps/epoch: {steps_per_epoch}, Max steps: {max_steps}")

    # Create training config (step-driven)
    train_config = TrainingConfig(
        learning_rate=lr,
        batch_size=batch_size,
        max_steps=max_steps,
        warmup_steps=min(100, max_steps // 10),
        logging_steps=50,
        eval_steps=steps_per_epoch,
        save_steps=steps_per_epoch,
        gradient_accumulation_steps=1,
        use_lora=True,
        lora_r=32,
        lora_alpha=64,
    )

    # Create model
    print(f"  Initializing QLLM model...")
    model_config = QLLMConfig(
        hidden_dim=768,
        num_layers=6,
        num_heads=12,
        use_semantic_phase=True,
        semantic_phase_dim=768,
    )
    model = QLLM(model_config).to(DEVICE)

    # Create trainer with v3 architecture (standalone slack loss)
    # Use different output dir for φ-balance mode to separate experiments
    mode_suffix = "_phi" if use_phi_balance else ""
    output_dir = Path(f"checkpoints/validation_seed_{seed}{mode_suffix}")
    trainer = SemanticPhaseTrainer(
        model=model,
        bundle_path=bundles_path,
        train_config=train_config,
        device=DEVICE,
        margin_easy=margin,
        margin_hard=margin + 0.1,
        ce_weight=ce_weight,
        contrastive_weight=0.5,
        output_dir=output_dir,
        seed=seed,
        late_stage_boost=late_stage_boost,
        late_stage_start=late_stage_start,
        late_stage_ramp=late_stage_ramp,
        slack_weight=slack_weight,  # v3: standalone slack (first-class citizen)
        hard_neg_penalty_mult=hard_neg_penalty_mult,
        ce_late_taper=ce_late_taper,
        track_killers=track_killers,
        killer_log_every=killer_log_every,
        use_phi_balance=use_phi_balance,
        phi_ema_decay=phi_ema_decay,
        use_sense_head=use_sense_head,
        sense_head_dim=sense_head_dim,
        sense_head_dropout=sense_head_dropout,
        sense_head_entropy_weight=sense_head_entropy_weight,
        sense_head_always_on_slack=sense_head_always_on_slack,
        hard_neg_top_k=hard_neg_top_k,
        hard_neg_temperature=hard_neg_temperature
    )

    # Train (trainer has built-in dashboard logging)
    try:
        print(f"  Starting training...")
        trainer.train(max_steps=max_steps, log_every=50, save_every=steps_per_epoch)
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return SeedResult(seed=seed, passed=False)

    # Read dashboard log from trainer
    # Use correct certification margins: easy=0.05, hard=0.15
    slack_tracker = SlackTracker(margin_easy=0.05, margin_hard=0.15)
    dashboard_path = output_dir / "dashboard_log.json"
    if dashboard_path.exists():
        import json
        with open(dashboard_path) as f:
            dashboard = json.load(f)
        for entry in dashboard.get("steps", []):
            step = entry.get("step", 0)
            # Get separate easy/hard gaps
            gap_e = entry.get("gap_easy", 0) or 0
            gap_h = entry.get("gap_hard", 0) or 0
            msr_e = entry.get("msr_easy", 0) or 0
            msr_h = entry.get("msr_hard", 0) or 0
            slack_tracker.record(step, gap_e, gap_h, msr_e, msr_h)

    # Collect results
    result = SeedResult(seed=seed)

    # Default late analysis values
    late_analysis_easy = {"rate": 0.0, "avg_slack": 0.0, "late_steps": 0, "positive_slack_steps": 0}
    late_analysis_hard = {"rate": 0.0, "avg_slack": 0.0, "late_steps": 0, "positive_slack_steps": 0}

    if slack_tracker.history:
        result.best_msr, result.best_msr_step = slack_tracker.get_best_msr()

        # Use hard slack as the primary metric (the boss fight)
        last_entry = slack_tracker.history[-1]
        result.final_gap = last_entry["gap"]
        result.final_slack = last_entry["slack"]
        result.final_msr = last_entry["msr"]

        result.first_positive_slack_step = slack_tracker.get_first_positive_slack_step_hard()
        result.steps_with_positive_slack = sum(1 for s in slack_tracker.hard_history if s["slack_positive"])

        # Analyze late steps separately for easy and hard
        late_analysis_easy = slack_tracker.get_late_steps_analysis_easy()
        late_analysis_hard = slack_tracker.get_late_steps_analysis_hard()

        # Use hard slack for certification (pass criterion)
        result.late_steps_total = late_analysis_hard["late_steps"]
        result.late_steps_positive_slack = late_analysis_hard["positive_slack_steps"]
        result.slack_positive_rate = late_analysis_hard["rate"]

        # Pass criterion: HARD slack positive for 30%+ of late steps
        result.passed = result.slack_positive_rate >= 0.30

    print(f"\n  Seed {seed} Results:")
    print(f"    Best MSR: {result.best_msr*100:.1f}% (step {result.best_msr_step})")
    print(f"    Final slack (hard): {result.final_slack:+.4f}")
    print(f"    Late positive rate: Easy={late_analysis_easy['rate']*100:.1f}% | Hard={late_analysis_hard['rate']*100:.1f}%")
    print(f"    Avg late slack: Easy={late_analysis_easy['avg_slack']:+.4f} | Hard={late_analysis_hard['avg_slack']:+.4f}")
    print(f"    Passed (hard 30%+): {'YES' if result.passed else 'NO'}")

    return result


def run_validation(
    bundles_path: Path,
    seeds: List[int] = [42, 123, 456],
    epochs: int = 3,
    margin: float = 0.05,
    output_report: Optional[Path] = None,
    late_stage_boost: float = 1.0,  # 1.0 = baseline
    late_stage_start: float = 0.5,
    late_stage_ramp: float = 0.4,
    slack_weight: float = 3.0,      # v3: standalone slack weight
    hard_neg_penalty_mult: float = 2.0,
    ce_late_taper: float = 1.0,
    track_killers: int = 0,
    killer_log_every: int = 100,
    use_phi_balance: bool = False,
    phi_ema_decay: float = 0.99,
    use_sense_head: bool = False,
    sense_head_dim: int = 256,
    sense_head_dropout: float = 0.1,
    sense_head_entropy_weight: float = 0.1,
    sense_head_always_on_slack: float = 0.1,
    hard_neg_top_k: int = 1,
    hard_neg_temperature: float = 0.1
) -> ValidationReport:
    """Run full 3-seed validation with v3 architecture."""
    mode_name = "φ-BALANCED" if use_phi_balance else "v3"
    print("\n" + "=" * 70)
    print(f"  3-SEED VALIDATION WITH SLACK-POSITIVE CRITERION ({mode_name})")
    print("=" * 70)
    print(f"\nSeeds: {seeds}")
    print(f"Epochs: {epochs}")
    print(f"Margin: {margin}")
    if late_stage_boost > 1.0:
        print(f"Late-stage guardrail: {late_stage_boost:.1f}x boost starting at {late_stage_start*100:.0f}%")
    print(f"Slack (v3 standalone): weight={slack_weight:.1f}, late ramp at {late_stage_start*100:.0f}%")
    print(f"  Hard-neg focus: {hard_neg_penalty_mult:.1f}x, softplus(margin-gap)")
    if ce_late_taper < 1.0:
        print(f"CE late taper: {ce_late_taper:.1f}x")
    if use_phi_balance:
        print(f"φ-BALANCE MODE: Adaptive slack weight targeting slack ≈ CE/φ")
        print(f"  Nested φ schedule: pre-geometry=0.618T, ramp=0.236T, hold=0.146T")
    print(f"Success criterion: Slack > 0 for 30%+ of late steps")

    report = ValidationReport(
        timestamp=datetime.now().isoformat(),
        seeds=seeds,
        bundles_file=str(bundles_path),
        epochs=epochs,
        margin=margin
    )

    # Run each seed
    for seed in seeds:
        result = run_single_seed(
            bundles_path=bundles_path,
            seed=seed,
            epochs=epochs,
            margin=margin,
            late_stage_boost=late_stage_boost,
            late_stage_start=late_stage_start,
            late_stage_ramp=late_stage_ramp,
            slack_weight=slack_weight,
            hard_neg_penalty_mult=hard_neg_penalty_mult,
            ce_late_taper=ce_late_taper,
            track_killers=track_killers,
            killer_log_every=killer_log_every,
            use_phi_balance=use_phi_balance,
            phi_ema_decay=phi_ema_decay,
            use_sense_head=use_sense_head,
            sense_head_dim=sense_head_dim,
            sense_head_dropout=sense_head_dropout,
            sense_head_entropy_weight=sense_head_entropy_weight,
            sense_head_always_on_slack=sense_head_always_on_slack,
            hard_neg_top_k=hard_neg_top_k,
            hard_neg_temperature=hard_neg_temperature
        )
        report.seed_results.append(result)

    # Aggregate metrics
    passed_seeds = [r for r in report.seed_results if r.passed]
    report.seeds_passed = len(passed_seeds)
    report.seeds_failed = len(report.seed_results) - report.seeds_passed

    if report.seed_results:
        report.avg_best_msr = sum(r.best_msr for r in report.seed_results) / len(report.seed_results)
        report.avg_final_slack = sum(r.final_slack for r in report.seed_results) / len(report.seed_results)
        report.avg_slack_positive_rate = sum(r.slack_positive_rate for r in report.seed_results) / len(report.seed_results)

    # Overall verdict
    # Pass if at least 2/3 seeds pass, OR if avg slack positive rate >= 30%
    report.passed = report.seeds_passed >= 2 or report.avg_slack_positive_rate >= 0.30

    if report.passed:
        report.verdict = f"PASSED: {report.seeds_passed}/{len(seeds)} seeds achieved positive slack"
    else:
        report.verdict = f"FAILED: Only {report.seeds_passed}/{len(seeds)} seeds achieved positive slack"

    # Print summary
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n  Seeds passed: {report.seeds_passed}/{len(seeds)}")
    print(f"  Avg best MSR: {report.avg_best_msr*100:.1f}%")
    print(f"  Avg final slack: {report.avg_final_slack:+.4f}")
    print(f"  Avg slack positive rate: {report.avg_slack_positive_rate*100:.1f}%")
    print(f"\n  Verdict: {report.verdict}")

    # Per-seed breakdown
    print("\n  Per-seed breakdown:")
    for r in report.seed_results:
        status = "[PASS]" if r.passed else "[FAIL]"
        print(f"    Seed {r.seed}: MSR={r.best_msr*100:.0f}%, slack={r.final_slack:+.3f}, "
              f"positive_rate={r.slack_positive_rate*100:.0f}% {status}")

    # Save report
    if output_report:
        report_data = {
            "timestamp": report.timestamp,
            "seeds": report.seeds,
            "bundles_file": report.bundles_file,
            "epochs": report.epochs,
            "margin": report.margin,
            "seed_results": [
                {
                    "seed": r.seed,
                    "best_msr": r.best_msr,
                    "best_msr_step": r.best_msr_step,
                    "final_slack": r.final_slack,
                    "slack_positive_rate": r.slack_positive_rate,
                    "passed": r.passed
                }
                for r in report.seed_results
            ],
            "avg_best_msr": report.avg_best_msr,
            "avg_final_slack": report.avg_final_slack,
            "avg_slack_positive_rate": report.avg_slack_positive_rate,
            "seeds_passed": report.seeds_passed,
            "passed": report.passed,
            "verdict": report.verdict
        }

        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n  Report saved to: {output_report}")

    return report


def main():
    parser = argparse.ArgumentParser(description="3-Seed Validation with Slack-Positive Criterion (v3)")
    parser.add_argument("--bundles", type=str, required=True, help="Path to bundles JSONL file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Seeds to use for validation")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs per seed")
    parser.add_argument("--margin", type=float, default=0.05, help="Margin for slack calculation")
    parser.add_argument("--output", type=str, help="Output report JSON path")
    # Late-stage schedule
    parser.add_argument("--late-boost", type=float, default=1.0,
                        help="Late-stage contrastive boost (1.0=off)")
    parser.add_argument("--late-start", type=float, default=0.5,
                        help="When slack ramp starts (fraction of training)")
    parser.add_argument("--late-ramp", type=float, default=0.4,
                        help="Slack ramp duration (fraction of training)")
    # v3: Standalone slack weight
    parser.add_argument("--slack-weight", type=float, default=3.0,
                        help="Standalone slack weight (competes directly with CE!)")
    parser.add_argument("--hard-neg-mult", type=float, default=2.0,
                        help="Hard-neg focus multiplier (2-3x recommended)")
    parser.add_argument("--ce-taper", type=float, default=1.0,
                        help="CE late taper (0.5=halve CE late, 1.0=no taper)")
    # Killer negative logging
    parser.add_argument("--track-killers", type=int, default=0,
                        help="Track K worst hard-neg violations per batch (0=disabled)")
    parser.add_argument("--killer-log-every", type=int, default=100,
                        help="Log killer negatives every N steps")
    # Golden ratio (φ) balance mode
    parser.add_argument("--phi-balance", action="store_true",
                        help="Enable φ-based adaptive weight balancing (slack ≈ CE/φ)")
    parser.add_argument("--phi-ema-decay", type=float, default=0.99,
                        help="EMA decay for φ-balance loss magnitude tracking")
    
    # SenseHead arguments
    parser.add_argument("--sense-head", action="store_true",
                        help="Enable SenseHead attentive pooling")
    parser.add_argument("--sense-head-dim", type=int, default=256,
                        help="SenseHead output dimension")
    parser.add_argument("--sense-head-dropout", type=float, default=0.1,
                        help="SenseHead dropout rate")
    parser.add_argument("--sense-head-entropy", type=float, default=0.1,
                        help="SenseHead attention entropy regularization")
    parser.add_argument("--sense-head-early-slack", type=float, default=0.1,
                        help="Always-on slack weight early to prevent collapse")
    
    # Top-k soft aggregation
    parser.add_argument("--hard-neg-top-k", type=int, default=1,
                        help="Top-k hard negatives for soft aggregation (1=max, 3=recommended)")
    parser.add_argument("--hard-neg-temp", type=float, default=0.1,
                        help="Temperature for softmax weighting of top-k hard negs")

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    report = run_validation(
        bundles_path=Path(args.bundles),
        seeds=args.seeds,
        epochs=args.epochs,
        margin=args.margin,
        output_report=output_path,
        late_stage_boost=args.late_boost,
        late_stage_start=args.late_start,
        late_stage_ramp=args.late_ramp,
        slack_weight=args.slack_weight,
        hard_neg_penalty_mult=args.hard_neg_mult,
        ce_late_taper=args.ce_taper,
        track_killers=args.track_killers,
        killer_log_every=args.killer_log_every,
        use_phi_balance=args.phi_balance,
        phi_ema_decay=args.phi_ema_decay,
        use_sense_head=args.sense_head,
        sense_head_dim=args.sense_head_dim,
        sense_head_dropout=args.sense_head_dropout,
        sense_head_entropy_weight=args.sense_head_entropy,
        sense_head_always_on_slack=args.sense_head_early_slack,
        hard_neg_top_k=args.hard_neg_top_k,
        hard_neg_temperature=args.hard_neg_temp
    )

    # Exit code based on pass/fail
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
