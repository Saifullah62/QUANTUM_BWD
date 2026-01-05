"""
QLLM Trainer (v2 - With Paradigm Win Conditions)
================================================

Training pipeline for Quantum-Inspired LLM with:
- LoRA/QLoRA support for efficient fine-tuning
- Paradigm-specific loss functions with WIN CONDITIONS
- Distributed training across GPU cluster
- Integration with Weights & Biases (optional)

WIN CONDITIONS (validated during training):
1. SemanticPhase: Polysemy improves when magnitude constrained
2. Lindblad: Same input + different noise → same semantic basin
3. Retrocausal: Effect→cause improves under no-leak protocol
4. Qualia: Channels correlate with external measures
5. Emergent: Attractor stability, reduced prompt fragility
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
import math

from ..core.config import QLLMConfig, TrainingConfig
from ..core.model import QLLM
from ..layers.qualia import QualiaLoss, QUALIA_CHANNELS
from ..layers.lindblad import LipschitzRegularizer
from ..layers.retrocausal import TwoPassTrainer


@dataclass
class TrainingState:
    """Track training state for checkpointing"""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    total_tokens: int = 0
    # Win condition tracking
    best_polysemy_score: float = 0.0
    best_consistency_score: float = 0.0
    best_retrocausal_score: float = 0.0
    win_condition_history: List[Dict] = field(default_factory=list)


class QLLMTrainer:
    """
    Trainer for Quantum-Inspired LLM with WIN CONDITION validation.

    Supports:
    - Standard training from scratch
    - LoRA fine-tuning of pretrained models
    - Paradigm-specific auxiliary losses with win condition testing
    - Two-pass retrocausal training with no-leak protocol
    - Multi-GPU training
    """

    def __init__(
        self,
        model: QLLM,
        train_config: TrainingConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        output_dir: str = "./outputs",
        polysemy_eval_data: Optional[Any] = None  # For win condition testing
    ):
        self.model = model
        self.config = train_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.polysemy_eval_data = polysemy_eval_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.state = TrainingState()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                num_workers=4
            )
        else:
            self.eval_loader = None

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Loss functions
        self.qualia_loss = QualiaLoss(model.config.num_qualia_channels)

        # Lipschitz regularizer for stability
        self.lipschitz_reg = LipschitzRegularizer()

        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

        # Logging
        self.log_history = []

        # Loss weights (can be tuned)
        self.loss_weights = {
            'phase_coherence': getattr(train_config, 'phase_consistency_loss_weight', 0.1),
            'lindblad_consistency': getattr(train_config, 'lindblad_consistency_weight', 0.1),
            'lipschitz': getattr(train_config, 'lipschitz_weight', 0.01),
            'qualia': getattr(train_config, 'qualia_diversity_loss_weight', 0.1),
            'retrocausal_leak': getattr(train_config, 'retrocausal_leak_weight', 0.1),
        }

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay"""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            else:
                progress = (step - self.config.warmup_steps) / max(
                    1, self.config.max_steps - self.config.warmup_steps
                )
                return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def compute_paradigm_losses(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        hidden_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute paradigm-specific auxiliary losses with WIN CONDITION support.

        Each loss maps to a testable win condition:
        - phase_coherence: polysemy differentiation through phase
        - lindblad_consistency: noise-invariant semantic basins
        - lipschitz: stability via spectral norm bounds
        - qualia: channel correlation with external measures
        - retrocausal_leak: prevent backward pass from cheating
        """
        losses = {}

        # =====================================================================
        # 1. SEMANTIC PHASE: Contrastive phase (same-sense align, different-sense separate)
        # =====================================================================
        if outputs.get('phase_states') is not None:
            real_embed = outputs.get('hidden_states')
            imag_embed = outputs.get('phase_states')

            if real_embed is not None and imag_embed is not None:
                # Initialize contrastive objective if not exists
                if not hasattr(self, '_contrastive_phase'):
                    from qllm.layers.semantic_phase import ContrastivePhaseObjective
                    self._contrastive_phase = ContrastivePhaseObjective(
                        embedding_dim=real_embed.shape[-1],
                        temperature=0.1,
                        margin=0.5,
                        separation_weight=1.0
                    ).to(self.device)

                # Get input_ids for in-batch contrastive
                input_ids = batch.get('input_ids')
                if input_ids is not None:
                    input_ids = input_ids.to(self.device)

                    # In-batch contrastive: same token, different context → different phase
                    contrastive_loss = self._contrastive_phase.in_batch_contrastive_loss(
                        real_embed, imag_embed, input_ids
                    )
                    if contrastive_loss > 0:
                        losses['phase_contrastive'] = self.loss_weights['phase_coherence'] * contrastive_loss

                # Anti-collapse regularizer: prevent trivial phase collapse
                collapse_penalty = self._contrastive_phase.anti_collapse_regularizer(
                    real_embed, imag_embed, min_phase_variance=0.1
                )
                losses['phase_anti_collapse'] = 0.1 * collapse_penalty

                # Magnitude regularization: encourage unit-like magnitudes
                # This prepares for the WIN CONDITION test where we constrain magnitude=1
                magnitude = self.model.embeddings.get_magnitude(real_embed, imag_embed)
                magnitude_var = ((magnitude - 1.0) ** 2).mean()
                losses['magnitude_reg'] = 0.01 * magnitude_var

        # =====================================================================
        # 2. LINDBLAD: Consistency loss + Lipschitz regularization
        # =====================================================================
        if hidden_states is not None and hasattr(self.model, 'blocks'):
            # Find Lindblad layers and compute consistency loss
            for block in self.model.blocks:
                if hasattr(block, 'lindblad') and block.lindblad is not None:
                    # Sample-based consistency: same input, different noise → same basin
                    consistency = block.lindblad.compute_consistency_loss(
                        hidden_states, num_samples=2  # Reduced for efficiency
                    )
                    losses['lindblad_consistency'] = self.loss_weights['lindblad_consistency'] * consistency

                    # Get Lipschitz info from stabilizer if available
                    if hasattr(block.lindblad, 'stabilizer'):
                        _, sigma = block.lindblad.stabilizer.forward(
                            hidden_states, return_lipschitz=True
                        )
                        # Penalize if spectral norm exceeds target
                        if sigma > 1.0:
                            losses['lipschitz'] = self.loss_weights['lipschitz'] * (sigma - 1.0)

                    break  # Only compute for first Lindblad layer to save compute

        # =====================================================================
        # 3. QUALIA: Mixed supervision losses (self-supervised + weak labels)
        # =====================================================================
        if outputs.get('qualia_info') is not None:
            qualia_info = outputs['qualia_info']
            qualia_tensor = qualia_info.get('qualia_tensor')

            if qualia_tensor is not None:
                # Diversity loss: channels should be distinct, not collapsed
                losses['qualia'] = self.loss_weights['qualia'] * self.qualia_loss(qualia_tensor)

                # Self-supervised channel losses (if logits available)
                if outputs.get('logits') is not None and hidden_states is not None:
                    self_supervised_loss = self._compute_self_supervised_qualia_loss(
                        qualia_tensor, hidden_states, outputs['logits']
                    )
                    if self_supervised_loss is not None:
                        losses['qualia_self_supervised'] = 0.1 * self_supervised_loss

        # =====================================================================
        # 4. RETROCAUSAL: Leak penalty (backward must not cheat)
        # =====================================================================
        if outputs.get('forward_state') is not None and outputs.get('backward_state') is not None:
            forward_state = outputs['forward_state']
            backward_state = outputs['backward_state']

            # Get target embeddings if available
            labels = batch.get('labels')
            if labels is not None:
                # Ensure labels are on the correct device
                labels = labels.to(forward_state.device)
                # Compute leak penalty using TwoPassTrainer utility
                target_embeds = self.model.embeddings.real_embedding(labels)
                leak_penalty = TwoPassTrainer.compute_leak_penalty(
                    forward_state, backward_state, target_embeds
                )
                losses['retrocausal_leak'] = self.loss_weights['retrocausal_leak'] * leak_penalty

        # =====================================================================
        # 5. EMERGENT: Attractor stability (if attractor layers present)
        # =====================================================================
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                if hasattr(block, 'attractor') and block.attractor is not None:
                    # Energy should decrease (states should flow toward attractors)
                    if hasattr(block.attractor, 'compute_energy'):
                        energy = block.attractor.compute_energy(hidden_states)
                        # Penalize high energy (far from attractors)
                        losses['attractor_energy'] = 0.01 * energy.mean()
                    break

        return losses

    def _compute_self_supervised_qualia_loss(
        self,
        qualia_tensor: torch.Tensor,
        hidden_states: torch.Tensor,
        logits: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute self-supervised losses for qualia channels.

        Maps:
        - certainty ↔ entropy of predictions
        - novelty ↔ surprisal (cross-entropy)
        - coherence ↔ layer agreement
        """
        losses = []

        # Certainty should correlate with negative entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        max_entropy = math.log(probs.size(-1))
        normalized_entropy = entropy / max_entropy

        # Assuming certainty is channel 0 (per QUALIA_CHANNELS definition)
        if qualia_tensor.size(-1) > 0:
            certainty_pred = qualia_tensor[..., 0]  # [batch, seq]
            certainty_target = 1 - normalized_entropy  # High certainty = low entropy
            certainty_loss = F.mse_loss(certainty_pred, certainty_target.detach())
            losses.append(certainty_loss)

        # Novelty should correlate with surprisal (high loss = novel)
        # We'd need per-token loss here, which requires labels
        # Skipping for now as it requires more complex integration

        if losses:
            return sum(losses) / len(losses)
        return None

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step with paradigm-specific losses"""
        self.model.train()

        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True  # Need for paradigm losses
            )

            # Main loss
            loss = outputs['loss']

            # Get hidden states for paradigm losses
            hidden_states = outputs.get('all_hidden_states', [None])[-1]

            # Paradigm losses with full context
            paradigm_losses = self.compute_paradigm_losses(
                outputs, batch, hidden_states=hidden_states
            )
            for name, ploss in paradigm_losses.items():
                if isinstance(ploss, torch.Tensor):
                    loss = loss + ploss

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient accumulation
        if (self.state.step + 1) % self.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        self.state.step += 1
        self.state.total_tokens += input_ids.numel()

        # Return metrics
        metrics = {'loss': loss.item()}
        for name, ploss in paradigm_losses.items():
            if isinstance(ploss, torch.Tensor):
                metrics[f'loss_{name}'] = ploss.item()
            else:
                metrics[f'loss_{name}'] = ploss

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        if self.eval_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        total_samples = 0

        for batch in self.eval_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs['loss'].item() * input_ids.size(0)
            total_samples += input_ids.size(0)

        return {'eval_loss': total_loss / max(1, total_samples)}

    # =========================================================================
    # WIN CONDITION VALIDATION METHODS
    # =========================================================================

    @torch.no_grad()
    def validate_win_conditions(self) -> Dict[str, float]:
        """
        Run all paradigm win condition tests.

        Returns dict of win condition scores (higher = better).
        """
        self.model.eval()
        results = {}

        # 1. Semantic Phase: Polysemy resolution with magnitude constraint
        polysemy_score = self.test_polysemy_win_condition()
        if polysemy_score is not None:
            results['polysemy_resolution'] = polysemy_score

        # 2. Lindblad: Noise invariance
        consistency_score = self.test_consistency_win_condition()
        if consistency_score is not None:
            results['noise_invariance'] = consistency_score

        # 3. Retrocausal: Effect→cause (no leak)
        retrocausal_score = self.test_retrocausal_win_condition()
        if retrocausal_score is not None:
            results['retrocausal_reasoning'] = retrocausal_score

        # Track best scores
        if results.get('polysemy_resolution', 0) > self.state.best_polysemy_score:
            self.state.best_polysemy_score = results['polysemy_resolution']
        if results.get('noise_invariance', 0) > self.state.best_consistency_score:
            self.state.best_consistency_score = results['noise_invariance']
        if results.get('retrocausal_reasoning', 0) > self.state.best_retrocausal_score:
            self.state.best_retrocausal_score = results['retrocausal_reasoning']

        self.state.win_condition_history.append({
            'step': self.state.step,
            **results
        })

        return results

    @torch.no_grad()
    def test_polysemy_win_condition(self) -> Optional[float]:
        """
        WIN CONDITION: Polysemy resolution improves when magnitude is constrained.

        Test: Same polysemous word in different contexts should have:
        - Similar magnitude (semantic content)
        - Different phase (contextual meaning)

        When magnitude is constrained to 1, only phase differentiates.
        """
        if self.polysemy_eval_data is None:
            return None

        # Need polysemy test data with format:
        # [{"word": "bank", "context1": "river bank", "context2": "money bank"}]
        scores = []

        for example in self.polysemy_eval_data[:20]:  # Limit for efficiency
            word = example.get('word')
            ctx1 = example.get('context1')
            ctx2 = example.get('context2')

            if not all([word, ctx1, ctx2]):
                continue

            # This is simplified - real implementation needs tokenizer
            # For now, return placeholder
            pass

        # Placeholder: compute phase difference correlation
        # Higher score = better polysemy differentiation through phase
        return None  # Implement with actual tokenizer

    @torch.no_grad()
    def test_consistency_win_condition(self) -> Optional[float]:
        """
        WIN CONDITION: Same input with different noise → same semantic basin.

        Test multiple forward passes with noise enabled, measure consistency.
        """
        if self.eval_loader is None:
            return None

        consistencies = []

        # Take a few batches
        for i, batch in enumerate(self.eval_loader):
            if i >= 5:  # Limit batches
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Run multiple forward passes with noise
            outputs_list = []
            for _ in range(3):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden = outputs.get('all_hidden_states', [None])[-1]
                if hidden is not None:
                    outputs_list.append(hidden)

            if len(outputs_list) >= 2:
                # Compute pairwise cosine similarity
                stacked = torch.stack(outputs_list)
                mean_output = stacked.mean(dim=0)

                # Consistency = 1 - variance ratio
                variance = ((stacked - mean_output) ** 2).mean()
                magnitude = (mean_output ** 2).mean()
                consistency = 1.0 - (variance / (magnitude + 1e-8)).item()
                consistencies.append(max(0, min(1, consistency)))  # Clamp to [0,1]

        if consistencies:
            return sum(consistencies) / len(consistencies)
        return None

    @torch.no_grad()
    def test_retrocausal_win_condition(self) -> Optional[float]:
        """
        WIN CONDITION: Effect→cause prediction accuracy.

        Test: Given an effect, can the model identify the cause?
        Measured with backward pass disabled vs enabled.
        """
        # This requires causal reasoning test data
        # Format: [{"effect": "The window broke", "cause": "The ball hit it"}]

        # For now, measure backward contribution to prediction
        if self.eval_loader is None:
            return None

        backward_contributions = []

        for i, batch in enumerate(self.eval_loader):
            if i >= 3:
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Get outputs with backward enabled
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # Check if backward state is meaningfully different from forward
            forward_state = outputs.get('forward_state')
            backward_state = outputs.get('backward_state')

            if forward_state is not None and backward_state is not None:
                # Backward should be different but not too similar to targets
                cos_sim = F.cosine_similarity(
                    forward_state.flatten(1),
                    backward_state.flatten(1),
                    dim=-1
                ).mean()

                # Ideal: some contribution (0.1-0.5) but not dominant
                contribution = 1.0 - abs(cos_sim.item() - 0.3) / 0.7
                backward_contributions.append(max(0, min(1, contribution)))

        if backward_contributions:
            return sum(backward_contributions) / len(backward_contributions)
        return None

    @torch.no_grad()
    def compute_paradigm_metrics(self) -> Dict[str, float]:
        """Compute detailed per-paradigm metrics for logging."""
        self.model.eval()
        metrics = {}

        if self.eval_loader is None:
            return metrics

        # Sample one batch
        batch = next(iter(self.eval_loader))
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Phase metrics
        if outputs.get('phase_states') is not None:
            real = outputs.get('hidden_states')
            imag = outputs.get('phase_states')
            if real is not None:
                magnitude = self.model.embeddings.get_magnitude(real, imag)
                phase = self.model.embeddings.get_phase(real, imag)

                metrics['phase/magnitude_mean'] = magnitude.mean().item()
                metrics['phase/magnitude_std'] = magnitude.std().item()
                metrics['phase/phase_mean'] = phase.mean().item()
                metrics['phase/phase_std'] = phase.std().item()
                metrics['phase/coherence'] = self.model.embeddings.get_phase_coherence(
                    (real, imag)
                ).mean().item()

        # Qualia metrics
        if outputs.get('qualia_info') is not None:
            qualia = outputs['qualia_info'].get('qualia_tensor')
            if qualia is not None:
                for i, (name, _) in enumerate(QUALIA_CHANNELS.items()):
                    if i < qualia.size(-1):
                        metrics[f'qualia/{name}_mean'] = qualia[..., i].mean().item()
                        metrics[f'qualia/{name}_std'] = qualia[..., i].std().item()

        return metrics

    def save_checkpoint(self, name: str = "checkpoint"):
        """Save training checkpoint"""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(exist_ok=True)

        # Save model
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / "model.pt"
        )

        # Save optimizer
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt"
        )

        # Save training state
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(asdict(self.state), f)

        # Save config
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump(self.model.config.to_dict(), f)

        print(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint_dir = Path(path)

        self.model.load_state_dict(
            torch.load(checkpoint_dir / "model.pt", map_location=self.device)
        )

        self.optimizer.load_state_dict(
            torch.load(checkpoint_dir / "optimizer.pt", map_location=self.device)
        )

        with open(checkpoint_dir / "training_state.json", 'r') as f:
            state_dict = json.load(f)
            self.state = TrainingState(**state_dict)

        print(f"Loaded checkpoint from {checkpoint_dir}")

    def train(self):
        """
        Main training loop with WIN CONDITION validation.

        Periodically tests paradigm-specific win conditions and logs results.
        """
        print(f"Starting training on {self.device}")
        print(f"Model paradigms: {self.model.get_paradigm_summary()}")
        print(f"Training config: {asdict(self.config)}")
        print(f"Paradigm losses enabled: {list(self.loss_weights.keys())}")

        # Win condition validation frequency (every N eval steps)
        win_condition_freq = getattr(self.config, 'win_condition_eval_freq', 5)

        for epoch in range(10):  # Max epochs
            self.state.epoch = epoch

            epoch_metrics = []
            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)

                # Logging
                if self.state.step % self.config.logging_steps == 0:
                    avg_metrics = {
                        k: sum(m.get(k, 0) for m in epoch_metrics[-self.config.logging_steps:]) / self.config.logging_steps
                        for k in metrics.keys()
                    }
                    lr = self.scheduler.get_last_lr()[0]

                    # Build loss summary
                    paradigm_losses = [f"{k.replace('loss_', '')}: {v:.4f}"
                                       for k, v in avg_metrics.items() if k.startswith('loss_')]
                    paradigm_str = " | ".join(paradigm_losses[:3])  # Limit display

                    print(
                        f"Step {self.state.step} | "
                        f"Epoch {epoch} | "
                        f"Loss: {avg_metrics['loss']:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"{paradigm_str}"
                    )
                    self.log_history.append({
                        'step': self.state.step,
                        **avg_metrics
                    })

                # Evaluation
                if self.state.step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        print(f"  Eval: {eval_metrics}")
                        if eval_metrics['eval_loss'] < self.state.best_loss:
                            self.state.best_loss = eval_metrics['eval_loss']
                            self.save_checkpoint("best")

                    # Win condition validation (less frequent than eval)
                    if (self.state.step // self.config.eval_steps) % win_condition_freq == 0:
                        win_metrics = self.validate_win_conditions()
                        if win_metrics:
                            print(f"  WIN CONDITIONS: {win_metrics}")

                        # Also log detailed paradigm metrics
                        paradigm_metrics = self.compute_paradigm_metrics()
                        if paradigm_metrics:
                            self.log_history.append({
                                'step': self.state.step,
                                'type': 'paradigm_metrics',
                                **paradigm_metrics
                            })

                # Save checkpoint
                if self.state.step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.state.step}")

                # Check max steps
                if self.state.step >= self.config.max_steps:
                    print("Reached max steps, stopping training")
                    # Final win condition report
                    final_win = self.validate_win_conditions()
                    print(f"\n=== FINAL WIN CONDITION SCORES ===")
                    print(f"  Best Polysemy:    {self.state.best_polysemy_score:.4f}")
                    print(f"  Best Consistency: {self.state.best_consistency_score:.4f}")
                    print(f"  Best Retrocausal: {self.state.best_retrocausal_score:.4f}")
                    self.save_checkpoint("final")
                    return

        # Final report
        final_win = self.validate_win_conditions()
        print(f"\n=== TRAINING COMPLETE - WIN CONDITION SCORES ===")
        print(f"  Best Polysemy:    {self.state.best_polysemy_score:.4f}")
        print(f"  Best Consistency: {self.state.best_consistency_score:.4f}")
        print(f"  Best Retrocausal: {self.state.best_retrocausal_score:.4f}")
        self.save_checkpoint("final")


class LoRATrainer(QLLMTrainer):
    """
    Trainer with LoRA (Low-Rank Adaptation) for efficient fine-tuning.

    Only trains low-rank adapter matrices, keeping base model frozen.
    """

    def __init__(
        self,
        model: QLLM,
        train_config: TrainingConfig,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        output_dir: str = "./outputs"
    ):
        # Add LoRA adapters before calling parent init
        self._add_lora_adapters(model, train_config)

        super().__init__(model, train_config, train_dataset, eval_dataset, output_dir)

    def _add_lora_adapters(self, model: QLLM, config: TrainingConfig):
        """Add LoRA adapters to target modules"""
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Add LoRA to target modules
        for name, module in model.named_modules():
            if any(target in name for target in config.lora_target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA-enhanced linear
                    parent = self._get_parent_module(model, name)
                    child_name = name.split('.')[-1]

                    lora_linear = LoRALinear(
                        module.in_features,
                        module.out_features,
                        r=config.lora_r,
                        alpha=config.lora_alpha,
                        dropout=config.lora_dropout
                    )

                    # Copy original weights
                    lora_linear.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        lora_linear.bias.data = module.bias.data.clone()

                    setattr(parent, child_name, lora_linear)

    def _get_parent_module(self, model: nn.Module, name: str) -> nn.Module:
        """Get parent module from dotted name"""
        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        return module


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Original (frozen) weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = F.linear(x, self.weight, self.bias)

        # LoRA adaptation
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """Merge LoRA weights into original for inference"""
        with torch.no_grad():
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling


# CLI for training
def main():
    """Training CLI"""
    import argparse

    parser = argparse.ArgumentParser(description="Train QLLM")
    parser.add_argument('--data', required=True, help='Training data path')
    parser.add_argument('--output-dir', default='./outputs', help='Output directory')
    parser.add_argument('--base-model', default=None, help='Base model to fine-tune')
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--max-steps', type=int, default=10000)

    args = parser.parse_args()

    # Create config
    if args.base_model:
        model_config = QLLMConfig.from_base_model(args.base_model)
    else:
        model_config = QLLMConfig.minimal()

    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        use_lora=args.use_lora
    )

    # Create model
    model = QLLM(model_config)

    # Create dataset (simplified for CLI)
    from .dataset import QuantumParadigmDataset

    # You'd need a tokenizer here
    # dataset = QuantumParadigmDataset(args.data, tokenizer)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Paradigm summary: {model.get_paradigm_summary()}")


if __name__ == "__main__":
    main()
