#!/usr/bin/env python3
"""
SemanticPhase v2 Trainer
========================

Trains QLLM using v2 polysemy bundles with gap-based contrastive loss.
Logs the full MSR/gap dashboard for monitoring meaning separation.

Key features:
- Direct consumption of v2 bundles (no mining)
- Gap-based margins relative to positive similarity
- MSR/slack/difficulty tracking per step
- 3-seed validation support

Usage:
    python train_semantic_phase_v2.py --bundles path/to/bundles.jsonl --steps 3000 --seed 42
"""

import argparse
import json
import torch
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Setup imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.bundle_dataset import (
    BundleDataset, BundleAwareSampler, bundle_collator,
    compute_bundle_contrastive_loss, create_bundle_dataloader, BundleBatch
)
from qllm.core.config import QLLMConfig, TrainingConfig
from qllm.core.model import QLLM

# SenseHead for attentive pooling (addresses representation collapse)
try:
    from qllm.layers.sense_head import SenseHead, SenseHeadWithEntropy
    SENSEHEAD_AVAILABLE = True
except ImportError:
    SENSEHEAD_AVAILABLE = False
    SenseHead = None

# ============================================================================
# Golden Ratio (φ) Constants and Helpers
# ============================================================================
# φ provides principled partitions that avoid extreme regimes where one
# objective bulldozes the others. The key insight: φ ≈ 1.618 creates
# proportions where "big part" and "small part" maintain consistent
# relationship without collapsing into each other.

PHI = 1.6180339887  # Golden ratio φ
PHI_INV = 0.6180339887  # 1/φ = φ - 1 ≈ 0.618
PHI_SQ_INV = 0.3819660113  # 1/φ² ≈ 0.382

# Nested φ schedule partitions:
# - Pre-geometry phase: 1/φ ≈ 0.618 of training (representation learns to breathe)
# - Slack phase: 1/φ² ≈ 0.382 of training, further split:
#   - Ramp: 0.382/φ ≈ 0.236 (smooth transition)
#   - Hold: 0.382 - 0.236 ≈ 0.146 (full pressure)
PHI_LATE_START = PHI_INV  # 0.618
PHI_RAMP_FRAC = PHI_SQ_INV / PHI  # 0.236 of total
PHI_HOLD_FRAC = PHI_SQ_INV - PHI_RAMP_FRAC  # 0.146 of total


class PhiWeightController:
    """
    Adaptive weight controller using golden ratio relationships.

    Instead of fixed weights, maintains target proportions:
    - slack_contribution ≈ CE_contribution / φ
    - contrastive stays smaller but nonzero

    Uses EMA of loss magnitudes to compute adaptive weights each step.
    If model still can't push hard slack above zero with φ-balanced forces,
    that's a loud signal the limit is representational, not tuning.
    """

    def __init__(
        self,
        target_ce_weight: float = 0.3,
        target_contrastive_weight: float = 0.5,
        ema_decay: float = 0.99,
        min_weight: float = 0.1,
        max_weight: float = 10.0
    ):
        self.target_ce_weight = target_ce_weight
        self.target_contrastive_weight = target_contrastive_weight
        self.ema_decay = ema_decay
        self.min_weight = min_weight
        self.max_weight = max_weight

        # EMA trackers for loss magnitudes
        self.ema_ce = None
        self.ema_slack = None
        self.ema_contrastive = None

        # Computed adaptive weights
        self.adaptive_slack_weight = 1.0

    def update(self, ce_loss: float, slack_loss: float, contrastive_loss: float):
        """Update EMAs and compute adaptive slack weight."""
        # Initialize EMAs on first call
        if self.ema_ce is None:
            self.ema_ce = ce_loss
            self.ema_slack = slack_loss
            self.ema_contrastive = contrastive_loss
        else:
            self.ema_ce = self.ema_decay * self.ema_ce + (1 - self.ema_decay) * ce_loss
            self.ema_slack = self.ema_decay * self.ema_slack + (1 - self.ema_decay) * slack_loss
            self.ema_contrastive = self.ema_decay * self.ema_contrastive + (1 - self.ema_decay) * contrastive_loss

        # Target: slack_contribution ≈ CE_contribution / φ
        # CE_contribution = ema_ce * ce_weight
        # slack_contribution = ema_slack * slack_weight
        # Want: ema_slack * slack_weight ≈ (ema_ce * ce_weight) / φ
        # So: slack_weight ≈ (ema_ce * ce_weight) / (φ * ema_slack)

        if self.ema_slack > 1e-8:  # Avoid division by zero
            target_slack_contribution = (self.ema_ce * self.target_ce_weight) / PHI
            self.adaptive_slack_weight = target_slack_contribution / self.ema_slack
            self.adaptive_slack_weight = max(self.min_weight, min(self.max_weight, self.adaptive_slack_weight))

        return self.adaptive_slack_weight

    def get_slack_weight(self) -> float:
        """Get current adaptive slack weight."""
        return self.adaptive_slack_weight

    def get_balance_stats(self) -> Dict[str, float]:
        """Get current balance statistics for logging."""
        ce_contrib = (self.ema_ce or 0) * self.target_ce_weight
        slack_contrib = (self.ema_slack or 0) * self.adaptive_slack_weight
        ratio = ce_contrib / slack_contrib if slack_contrib > 1e-8 else float('inf')

        return {
            "phi_ema_ce": self.ema_ce or 0,
            "phi_ema_slack": self.ema_slack or 0,
            "phi_ce_contrib": ce_contrib,
            "phi_slack_contrib": slack_contrib,
            "phi_ce_slack_ratio": ratio,  # Target is φ ≈ 1.618
            "phi_adaptive_slack_weight": self.adaptive_slack_weight
        }


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DashboardLogger:
    """Logs MSR/gap metrics for the training dashboard."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_logs = []
        self.best_msr_total = 0.0
        self.slack_crossed_zero = None  # Step where slack first crosses zero

    def log_step(self, step: int, metrics: Dict[str, Any]):
        """Log metrics for a single step."""
        entry = {"step": step, **metrics}
        self.step_logs.append(entry)

        # Track milestones
        if metrics.get("msr_total", 0) > self.best_msr_total:
            self.best_msr_total = metrics["msr_total"]

        # Track slack zero crossing (key milestone)
        if self.slack_crossed_zero is None:
            avg_slack = (metrics.get("avg_slack_easy", -1) + metrics.get("avg_slack_hard", -1)) / 2
            if avg_slack > 0:
                self.slack_crossed_zero = step
                print(f"  *** MILESTONE: Slack crossed zero at step {step} ***")

    def print_dashboard(self, step: int, metrics: Dict[str, Any], loss: float):
        """Print a compact dashboard line."""
        msr_e = metrics.get("msr_easy", 0)
        msr_h = metrics.get("msr_hard", 0)
        gap_e = metrics.get("gap_easy", 0)
        gap_h = metrics.get("gap_hard", 0)
        slack_e = metrics.get("avg_slack_easy", 0)
        slack_h = metrics.get("avg_slack_hard", 0)

        # Difficulty bucket summary
        by_diff = metrics.get("msr_by_difficulty", {})
        diff_str = " | ".join([f"{k}:{v.get('msr', 0):.0%}" for k, v in by_diff.items() if v.get('n', 0) > 0])

        print(f"Step {step:5d} | Loss: {loss:.4f} | "
              f"MSR: {msr_e:.0%}/{msr_h:.0%} | "
              f"Gap: {gap_e:+.3f}/{gap_h:+.3f} | "
              f"Slack: {slack_e:+.3f}/{slack_h:+.3f} | "
              f"Diff: [{diff_str}]")

    def save(self, filename: str = "dashboard_log.json"):
        """Save all logs to JSON."""
        path = self.output_dir / filename
        summary = {
            "total_steps": len(self.step_logs),
            "best_msr_total": self.best_msr_total,
            "slack_crossed_zero_at": self.slack_crossed_zero,
            "steps": self.step_logs
        }
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Dashboard saved to {path}")


class SemanticPhaseTrainerV2:
    """
    Trainer for SemanticPhase using v2 bundles.

    Key differences from v1:
    - No mining: pairings come directly from bundle structure
    - Gap-based loss: margin relative to positive, not absolute threshold
    - Full dashboard logging: MSR, gap, slack by role and difficulty
    """

    def __init__(
        self,
        model: QLLM,
        bundle_path: Path,
        train_config: TrainingConfig,
        device: str = 'cuda',
        margin_easy: float = 0.05,   # Certification margin for easy negatives
        margin_hard: float = 0.15,   # Certification margin for hard negatives
        temperature: float = 0.1,
        ce_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        ce_weight_ramp: bool = False,
        output_dir: Path = None,
        seed: int = 42,
        # Late-stage contrastive guardrail parameters
        late_stage_boost: float = 1.0,  # Multiplier for contrastive weight (1.0 = no boost)
        late_stage_start: float = 0.5,  # When to start late-stage effects (fraction of max_steps)
        late_stage_ramp: float = 0.4,   # How quickly to ramp up (fraction over which to ramp)
        # Standalone slack loss parameters (v3 architecture)
        slack_weight: float = 3.0,      # Standalone slack weight (competes directly with CE!)
        hard_neg_penalty_mult: float = 2.0,  # Hard-neg focus multiplier (2-3x recommended)
        # CE late taper parameters (prevent CE from bulldozing margins)
        ce_late_taper: float = 1.0,  # Final CE multiplier late (0.5 = halve CE late, 1.0 = no taper)
        # Killer negative logging (for curriculum learning)
        track_killers: int = 0,  # Number of worst hard-neg violations to track per batch (0 = disabled)
        killer_log_every: int = 100,  # How often to write killer logs (steps)
        # Golden ratio (φ) balance mode - adaptive weight control
        use_phi_balance: bool = False,  # Enable φ-based adaptive weight balancing
        phi_ema_decay: float = 0.99,  # EMA decay for loss magnitude tracking
        # SenseHead parameters
        use_sense_head: bool = False,
        sense_head_dim: int = 256,
        sense_head_dropout: float = 0.1,
        sense_head_entropy_weight: float = 0.1,
        sense_head_always_on_slack: float = 0.1,
        hard_neg_top_k: int = 1,
        hard_neg_temperature: float = 0.1
    ):
        self.model = model.to(device)
        self.ce_weight_ramp = ce_weight_ramp
        self.config = train_config
        self.device = device
        self.margin_easy = margin_easy
        self.margin_hard = margin_hard
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.contrastive_weight = contrastive_weight
        self.seed = seed

        # Late-stage contrastive guardrail
        self.late_stage_boost = late_stage_boost
        self.late_stage_start = late_stage_start
        self.late_stage_ramp = late_stage_ramp

        # Standalone slack loss (v3 architecture - first-class citizen in total_loss)
        self.slack_weight = slack_weight
        self.hard_neg_penalty_mult = hard_neg_penalty_mult

        # CE late taper (prevent CE from bulldozing margins)
        self.ce_late_taper = ce_late_taper

        # Killer negative tracking (for curriculum learning)
        self.track_killers = track_killers
        self.killer_log_every = killer_log_every
        self.killer_log_path = None  # Set after output_dir is created

        # Golden ratio (φ) balance mode
        self.use_phi_balance = use_phi_balance
        self.phi_controller = None
        if use_phi_balance:
            self.phi_controller = PhiWeightController(
                target_ce_weight=ce_weight,
                target_contrastive_weight=contrastive_weight,
                ema_decay=phi_ema_decay
            )
            # Override schedule to use nested φ partitions
            self.late_stage_start = PHI_LATE_START  # 0.618
            self.late_stage_ramp = PHI_RAMP_FRAC    # 0.236

        # SenseHead initialization
        self.use_sense_head = use_sense_head and SENSEHEAD_AVAILABLE
        self.sense_head = None
        self.sense_head_entropy_weight = sense_head_entropy_weight
        self.sense_head_always_on_slack = sense_head_always_on_slack
        self.hard_neg_top_k = hard_neg_top_k
        self.hard_neg_temperature = hard_neg_temperature
        
        if self.use_sense_head:
            if sense_head_entropy_weight > 0:
                self.sense_head = SenseHeadWithEntropy(
                    hidden_dim=model.config.hidden_dim,
                    proj_dim=sense_head_dim,
                    dropout=sense_head_dropout,
                    entropy_weight=sense_head_entropy_weight
                ).to(device)
            else:
                self.sense_head = SenseHead(
                    hidden_dim=model.config.hidden_dim,
                    proj_dim=sense_head_dim,
                    dropout=sense_head_dropout
                ).to(device)
            print(f"  SenseHead enabled: dim={sense_head_dim}")
        
        # Output directory
        self.output_dir = output_dir or Path(f"runs/semantic_phase_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize killer log file
        if self.track_killers > 0:
            self.killer_log_path = self.output_dir / "killer_negatives.jsonl"
            # Clear any existing file
            with open(self.killer_log_path, 'w') as f:
                pass  # Empty file

        # Dashboard logger
        self.dashboard = DashboardLogger(self.output_dir)

        # Create dataloader
        self.dataloader = create_bundle_dataloader(
            bundle_path=bundle_path,
            batch_size=train_config.batch_size,
            bundles_per_batch=2,
            max_length=128,
            vocab_size=model.config.vocab_size,
            seed=seed
        )

        # Optimizer (include SenseHead if enabled)
        params = list(model.parameters())
        if self.use_sense_head and self.sense_head is not None:
            params.extend(self.sense_head.parameters())
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )

        # LR scheduler (cosine annealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.max_steps,
            eta_min=train_config.learning_rate * 0.1
        )

    def get_pooled_embeddings(self, batch: BundleBatch) -> torch.Tensor:
        """Get pooled embeddings from model for contrastive loss.

        Uses FIRST TOKEN pooling (CLS-style) instead of mean pooling.
        Analysis showed first token gives -0.16 to 1.0 similarity range,
        vs 0.99-1.0 for mean pooling which collapses all embeddings.
        """
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Priority: phase_states > real_embedding > hidden_states
        # phase_states provides the best semantic differentiation

        if 'phase_states' in outputs and outputs['phase_states'] is not None:
            # Use phase states - these encode semantic phase information
            phase = outputs['phase_states']  # [B, seq, dim]
            if phase.is_complex():
                # First token pooling (CLS-style) - much better differentiation
                embeddings = phase[:, 0, :].abs()
            else:
                # First token pooling
                embeddings = phase[:, 0, :]

        elif 'real_embedding' in outputs and outputs['real_embedding'] is not None:
            # Use complex magnitude as embedding (first token)
            real = outputs['real_embedding'][:, 0, :]  # [B, hidden]
            imag = outputs.get('imag_embedding', torch.zeros_like(outputs['real_embedding']))[:, 0, :]
            embeddings = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        elif 'hidden_states' in outputs and outputs['hidden_states'] is not None:
            # Fall back to hidden states (first token)
            hidden = outputs['hidden_states']
            embeddings = hidden[:, 0, :]

        else:
            raise ValueError("No usable embeddings from model")

        return embeddings

    def get_sense_embeddings(self, batch):
        if self.sense_head is None:
            raise ValueError("SenseHead not initialized")
        
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        if "hidden_states" in outputs and outputs["hidden_states"] is not None:
            token_states = outputs["hidden_states"]
        else:
            raise ValueError("No token-level hidden states available")
        
        if hasattr(self.sense_head, "entropy_weight") and self.sense_head.entropy_weight > 0:
            embeddings, weights, entropy_loss = self.sense_head(token_states, attention_mask)
            return embeddings, weights, entropy_loss
        else:
            embeddings, weights = self.sense_head(token_states, attention_mask)
            return embeddings, weights, None

    def compute_early_warning(self, killers, step):
        if not killers:
            return "unknown"
        inversions = sum(1 for k in killers if k.get("gap", 0) < 0)
        ngr = inversions / len(killers)
        if ngr < 0.15:
            branch = "A"
        elif ngr < 0.40:
            branch = "B"
        else:
            branch = "C"
        print(f"=== EARLY WARNING step {step}: NGR={ngr*100:.1f}% -> Branch {branch} ===")
        return branch

    def get_ce_weight_for_step(self, step: int, max_steps: int) -> float:
        """Get CE weight with optional ramping and late taper.

        If ce_weight_ramp is set, ramps CE from 0 to ce_weight over the ramp period.
        This allows contrastive to carve sense geometry first, then CE learns inside it.

        If ce_late_taper < 1.0, tapers CE weight late in training to prevent
        CE from bulldozing the margins established by contrastive loss.
        """
        # Base CE weight (with optional early ramp)
        if not self.ce_weight_ramp:
            base_weight = self.ce_weight
        else:
            # First 300 steps: contrastive-only
            # Then ramp CE from 0 to ce_weight over next 700 steps
            warmup_steps = 300
            ramp_steps = 700

            if step < warmup_steps:
                base_weight = 0.0
            elif step < warmup_steps + ramp_steps:
                progress = (step - warmup_steps) / ramp_steps
                base_weight = self.ce_weight * progress
            else:
                base_weight = self.ce_weight

        # Apply late taper (uses same schedule as late_stage contrastive boost)
        if self.ce_late_taper < 1.0:
            late_start_step = int(max_steps * self.late_stage_start)
            ramp_steps_late = int(max_steps * self.late_stage_ramp)
            late_end_step = late_start_step + ramp_steps_late

            if step < late_start_step:
                # Early/mid training: full CE weight
                taper_mult = 1.0
            elif step < late_end_step:
                # Ramp period: linear interpolation from 1.0 to ce_late_taper
                progress = (step - late_start_step) / max(ramp_steps_late, 1)
                taper_mult = 1.0 + (self.ce_late_taper - 1.0) * progress
            else:
                # Late stage: tapered CE
                taper_mult = self.ce_late_taper

            base_weight *= taper_mult

        return base_weight

    def get_contrastive_weight_for_step(self, step: int, max_steps: int) -> float:
        """Get contrastive weight with late-stage boost (guardrail).

        Late-stage contrastive guardrail prevents slack collapse by boosting
        contrastive loss weight during the final portion of training.

        The boost ramps smoothly from 1.0x to late_stage_boost over the ramp period:
        - Before late_stage_start: weight = contrastive_weight
        - During ramp: linear interpolation from 1.0x to boost
        - After ramp: weight = contrastive_weight * late_stage_boost
        """
        late_start_step = int(max_steps * self.late_stage_start)
        ramp_steps = int(max_steps * self.late_stage_ramp)
        late_end_step = late_start_step + ramp_steps

        if step < late_start_step:
            # Early/mid training: base weight
            return self.contrastive_weight
        elif step < late_end_step:
            # Ramp period: linear interpolation from 1.0x to boost
            progress = (step - late_start_step) / max(ramp_steps, 1)
            multiplier = 1.0 + (self.late_stage_boost - 1.0) * progress
            return self.contrastive_weight * multiplier
        else:
            # Late stage: full boost
            return self.contrastive_weight * self.late_stage_boost

    def get_slack_weight_for_step(self, step: int, max_steps: int) -> float:
        """Get standalone slack weight with late ramping.

        v3 Architecture: Slack is a first-class citizen in total_loss, not diluted
        through contrastive_weight. Late ramp avoids early instability while
        providing force where collapse actually happens.

        Schedule: 0 early -> ramp to slack_weight late -> hold at slack_weight
        """
        # Late ramp: 0 early, ramp to full weight late
        late_start_step = int(max_steps * self.late_stage_start)
        ramp_steps = int(max_steps * self.late_stage_ramp)
        late_end_step = late_start_step + ramp_steps

        if step < late_start_step:
            return 0.0
        elif step < late_end_step:
            progress = (step - late_start_step) / max(ramp_steps, 1)
            return self.slack_weight * progress
        else:
            return self.slack_weight

    def train_step(self, batch: BundleBatch, step: int, max_steps: int = 2000) -> Dict[str, Any]:
        """Execute one training step."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)

        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get current CE weight (may be ramped)
        current_ce_weight = self.get_ce_weight_for_step(step, max_steps)

        # Cross-entropy loss (language modeling)
        ce_loss = torch.tensor(0.0, device=self.device)
        if current_ce_weight > 0 and 'logits' in outputs and outputs['logits'] is not None:
            logits = outputs['logits']
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            ce_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0  # Ignore padding
            )

        # Contrastive loss (gap-based) - base hinge only, slack is separate
        # Get embeddings (SenseHead or standard pooling)
        entropy_loss = None
        if self.use_sense_head and self.sense_head is not None:
            embeddings, attn_weights, entropy_loss = self.get_sense_embeddings(batch)
        else:
            embeddings = self.get_pooled_embeddings(batch)
        contrastive_metrics = compute_bundle_contrastive_loss(
            embeddings=embeddings,
            batch=batch,
            temperature=self.temperature,
            margin_easy=self.margin_easy,
            margin_hard=self.margin_hard,
            device=self.device,
            hard_neg_penalty_mult=self.hard_neg_penalty_mult,  # Hard-neg focus in slack
            track_killers=self.track_killers,  # Track K worst hard-neg violations
            hard_neg_top_k=self.hard_neg_top_k,
            hard_neg_temperature=self.hard_neg_temperature
        )
        contrastive_loss = contrastive_metrics['contrastive_loss']  # Pure base contrastive
        slack_loss = contrastive_metrics['slack_penalty_loss']  # TENSOR - standalone slack

        # Get current weights (with late-stage schedules)
        current_contrastive_weight = self.get_contrastive_weight_for_step(step, max_steps)

        # φ-balance mode: Use adaptive slack weight based on loss magnitude proportions
        # Target: slack_contribution ≈ CE_contribution / φ
        if self.use_phi_balance and self.phi_controller is not None:
            # Get schedule multiplier (0 early, ramp to 1 late)
            schedule_mult = self.get_slack_weight_for_step(step, max_steps) / max(self.slack_weight, 1e-8)

            # Update φ controller with current loss magnitudes
            self.phi_controller.update(
                ce_loss.item(),
                slack_loss.item(),
                contrastive_loss.item()
            )

            # Apply schedule multiplier to adaptive weight (0 early, full adaptive late)
            current_slack_weight = schedule_mult * self.phi_controller.get_slack_weight()
        else:
            current_slack_weight = self.get_slack_weight_for_step(step, max_steps)

        # v3 Architecture: Three-term loss with slack as first-class citizen
        # SenseHead: add entropy regularization and early slack
        if self.use_sense_head:
            early_slack = self.sense_head_always_on_slack if current_slack_weight == 0 else 0
            entropy_term = entropy_loss if entropy_loss is not None else 0
            total_loss = (current_ce_weight * ce_loss +
                          current_contrastive_weight * contrastive_loss +
                          (current_slack_weight + early_slack) * slack_loss +
                          entropy_term)
        else:
            total_loss = (current_ce_weight * ce_loss +
                          current_contrastive_weight * contrastive_loss +
                          current_slack_weight * slack_loss)

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Build metrics dict (exclude tensors and killer_negatives from logging)
        exclude_keys = {'contrastive_loss', 'slack_penalty_loss', 'killer_negatives'}
        metrics = {
            "loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "slack_loss": slack_loss.item(),  # Standalone slack loss value
            "slack_loss_easy": contrastive_metrics.get('slack_loss_easy', 0.0),
            "slack_loss_hard": contrastive_metrics.get('slack_loss_hard', 0.0),
            "slack_weight": current_slack_weight,  # Track dynamic slack weight
            "contrastive_weight": current_contrastive_weight,  # Track the dynamic weight
            "ce_weight": current_ce_weight,  # Track CE weight (for taper)
            "lr": self.scheduler.get_last_lr()[0],
            **{k: v for k, v in contrastive_metrics.items() if k not in exclude_keys}
        }

        # Add φ-balance stats if enabled
        if self.use_phi_balance and self.phi_controller is not None:
            metrics.update(self.phi_controller.get_balance_stats())

        # Add killer negatives separately (not for dashboard, for JSONL log)
        if 'killer_negatives' in contrastive_metrics:
            metrics['_killer_negatives'] = contrastive_metrics['killer_negatives']

        return metrics

    def log_killers(self, step: int, killers: List[Dict]):
        """Append killer negatives to JSONL log file."""
        if not self.killer_log_path or not killers:
            return

        with open(self.killer_log_path, 'a') as f:
            for killer in killers:
                entry = {"step": step, **killer}
                f.write(json.dumps(entry) + "\n")

    def train(self, max_steps: int, log_every: int = 100, save_every: int = 500):
        """Main training loop."""
        print("=" * 80)
        print("  SemanticPhase v2 Training")
        print("=" * 80)
        print(f"Output dir: {self.output_dir}")
        print(f"Max steps: {max_steps}")
        print(f"Margins: easy={self.margin_easy}, hard={self.margin_hard}")
        print(f"Weights: CE={self.ce_weight}, contrastive={self.contrastive_weight}")
        if self.ce_weight_ramp:
            print(f"CE Ramp: 0->1.0 over steps 300-1000 (contrastive carves geometry first)")
        if self.late_stage_boost > 1.0:
            late_start = int(max_steps * self.late_stage_start)
            late_end = int(max_steps * (self.late_stage_start + self.late_stage_ramp))
            print(f"Late-stage guardrail: {self.late_stage_boost:.1f}x boost from step {late_start} to {late_end}")
        # v3 Architecture: Standalone slack loss
        slack_start = int(max_steps * self.late_stage_start)
        slack_end = int(max_steps * (self.late_stage_start + self.late_stage_ramp))
        print(f"Slack (v3 standalone): weight={self.slack_weight:.1f}, late ramp {slack_start}->{slack_end}")
        print(f"  Hard-neg focus multiplier: {self.hard_neg_penalty_mult:.1f}x")
        print(f"  Using softplus(margin-gap) for smooth gradients")
        if self.ce_late_taper < 1.0:
            print(f"CE late taper: {self.ce_late_taper:.1f}x CE weight late (reduces CE bulldozing)")
        if self.track_killers > 0:
            print(f"Killer tracking: top {self.track_killers} worst hard-neg violations every {self.killer_log_every} steps")
            print(f"  Log file: {self.killer_log_path}")
        if self.use_phi_balance:
            print(f"φ-BALANCE MODE: Adaptive slack weight targeting slack ≈ CE/φ")
            print(f"  Nested φ schedule: pre-geometry={PHI_LATE_START:.3f}T, ramp={PHI_RAMP_FRAC:.3f}T, hold={PHI_HOLD_FRAC:.3f}T")
            print(f"  Target CE:slack ratio ≈ {PHI:.3f}")
        print()

        dataloader_iter = iter(self.dataloader)
        step = 0

        while step < max_steps:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                batch = next(dataloader_iter)

            step += 1
            metrics = self.train_step(batch, step, max_steps)

            # Log to dashboard (exclude internal keys like _killer_negatives)
            dashboard_metrics = {k: v for k, v in metrics.items() if not k.startswith('_')}
            self.dashboard.log_step(step, dashboard_metrics)

            # Log killer negatives to JSONL (if enabled and at the right interval)
            if self.track_killers > 0 and step % self.killer_log_every == 0:
                killers = metrics.get('_killer_negatives', [])
                if killers:
                    self.log_killers(step, killers)

            # Print progress
            if step % log_every == 0:
                self.dashboard.print_dashboard(step, metrics, metrics['loss'])

            # Save checkpoint
            if step % save_every == 0:
                self.save_checkpoint(step, metrics)

        # Final save
        self.save_checkpoint(step, metrics, final=True)
        self.dashboard.save()

        print("\n" + "=" * 80)
        print("  Training Complete")
        print("=" * 80)
        print(f"Best MSR total: {self.dashboard.best_msr_total:.2%}")
        if self.dashboard.slack_crossed_zero:
            print(f"Slack crossed zero at step: {self.dashboard.slack_crossed_zero}")
        else:
            print("Slack never crossed zero (model may need more training)")

    def save_checkpoint(self, step: int, metrics: Dict, final: bool = False):
        """Save model checkpoint."""
        suffix = "final" if final else f"step_{step}"
        path = self.output_dir / f"checkpoint_{suffix}.pt"

        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": {
                "margin_easy": self.margin_easy,
                "margin_hard": self.margin_hard,
                "temperature": self.temperature,
                "seed": self.seed
            }
        }, path)

        if not final:
            print(f"  Checkpoint saved: {path}")


def run_training(
    bundle_path: Path,
    steps: int = 3000,
    seed: int = 42,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = 'cuda',
    output_dir: Path = None,
    ce_weight: float = 1.0,
    contrastive_weight: float = 0.5,
    ce_weight_ramp: bool = False
):
    """Run a single training run."""
    set_seed(seed)

    # Model config
    model_config = QLLMConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        intermediate_dim=256,
        max_seq_length=128,
        use_semantic_phase=True,
        semantic_phase_dim=128,
        use_retrocausal_attention=False,  # Focus on SemanticPhase only
        use_lindblad_layers=False,        # Focus on SemanticPhase only
        use_qualia_output=False,          # Focus on SemanticPhase only
        use_emergent_init=True
    )

    # Training config
    train_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        max_steps=steps
    )

    # Create model
    model = QLLM(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Output directory
    if output_dir is None:
        output_dir = Path(f"runs/semantic_phase_v2_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Create trainer
    trainer = SemanticPhaseTrainerV2(
        model=model,
        bundle_path=bundle_path,
        train_config=train_config,
        device=device,
        output_dir=output_dir,
        seed=seed,
        ce_weight=ce_weight,
        contrastive_weight=contrastive_weight,
        ce_weight_ramp=ce_weight_ramp
    )

    # Train
    trainer.train(max_steps=steps)

    return trainer


def run_multi_seed(
    bundle_path: Path,
    seeds: List[int] = [42, 123, 456],
    steps: int = 3000,
    device: str = 'cuda'
):
    """Run training across multiple seeds for validation."""
    print("=" * 80)
    print("  Multi-Seed Validation")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Steps per run: {steps}")
    print()

    results = {}
    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"  SEED {seed}")
        print(f"{'='*40}")

        trainer = run_training(
            bundle_path=bundle_path,
            steps=steps,
            seed=seed,
            device=device
        )

        results[seed] = {
            "best_msr_total": trainer.dashboard.best_msr_total,
            "slack_crossed_zero_at": trainer.dashboard.slack_crossed_zero,
            "output_dir": str(trainer.output_dir)
        }

    # Summary
    print("\n" + "=" * 80)
    print("  Multi-Seed Summary")
    print("=" * 80)
    print(f"{'Seed':<10} {'Best MSR':<12} {'Slack Zero':<15}")
    print("-" * 40)
    for seed, r in results.items():
        slack_str = str(r['slack_crossed_zero_at']) if r['slack_crossed_zero_at'] else "Never"
        print(f"{seed:<10} {r['best_msr_total']:<12.2%} {slack_str:<15}")

    # Save summary
    summary_path = Path("runs/multi_seed_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SemanticPhase v2 Trainer")
    parser.add_argument('--bundles', type=str, required=True, help='Path to v2 bundles JSONL')
    parser.add_argument('--steps', type=int, default=3000, help='Training steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--multi-seed', action='store_true', help='Run 3-seed validation')
    parser.add_argument('--ce-weight', type=float, default=1.0, help='CE loss weight')
    parser.add_argument('--contrastive-weight', type=float, default=0.5, help='Contrastive loss weight')
    parser.add_argument('--ce-weight-ramp', action='store_true', help='Ramp CE from 0->1 over steps 300-1000')

    args = parser.parse_args()

    bundle_path = Path(args.bundles)
    if not bundle_path.exists():
        print(f"Error: Bundle file not found: {bundle_path}")
        return

    if args.multi_seed:
        run_multi_seed(
            bundle_path=bundle_path,
            seeds=[42, 123, 456],
            steps=args.steps,
            device=args.device
        )
    else:
        run_training(
            bundle_path=bundle_path,
            steps=args.steps,
            seed=args.seed,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            ce_weight=args.ce_weight,
            contrastive_weight=args.contrastive_weight,
            ce_weight_ramp=args.ce_weight_ramp
        )


if __name__ == "__main__":
    main()
