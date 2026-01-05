#!/usr/bin/env python3
"""
Proof Ablation v2 - With Paradigm-Aware Batching
=================================================

Key improvements over v1:
1. Pair-aware sampler for polysemy (batches contain anchor/pos/neg for same word)
2. On-the-fly noise injection collator for Lindblad twins
3. Activation counts logging per step

This ensures paradigm losses actually fire by providing the right data structure.

Usage:
    python scripts/run_proof_ablation_v2.py --data data/pilot_paradigm_data.jsonl
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
from torch.utils.data import Dataset, Sampler, DataLoader

from qllm.core.config import QLLMConfig, TrainingConfig


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")
from qllm.core.model import QLLM
from qllm.training.ablation import AblationRunner


def get_lindblad_schedule(step: int, max_steps: int, schedule_type: str = 'late_ramp') -> float:
    """
    Get Lindblad weight based on training progress.

    Late-ramp schedule: Lindblad works best AFTER the base mapping is formed.
    - Stage 1 (0-33%): 0% - let model learn basic patterns first
    - Stage 2 (33-66%): ramp up from 0% to 100%
    - Stage 3 (66-100%): 100% - full invariance training

    Args:
        step: Current training step
        max_steps: Total training steps
        schedule_type: 'constant' (always 1.0), 'late_ramp', or 'late_only'

    Returns:
        Weight multiplier for Lindblad loss (0.0-1.0)
    """
    if schedule_type == 'constant':
        return 1.0

    progress = step / max_steps

    if schedule_type == 'late_ramp':
        # Lindblad hurts early training (shown in micro-experiment)
        # Ramp in late to preserve robustness without hurting basin finding
        early_end = 0.33     # No Lindblad for first third
        ramp_end = 0.66      # Ramp up during middle third

        if progress < early_end:
            return 0.0
        elif progress < ramp_end:
            # Linear ramp from 0 to 1
            ramp_progress = (progress - early_end) / (ramp_end - early_end)
            return ramp_progress
        else:
            return 1.0

    elif schedule_type == 'late_only':
        # Even more conservative: Lindblad only in final 33%
        if progress < 0.66:
            return 0.0
        else:
            return 1.0

    return 1.0


def get_semantic_phase_schedule(step: int, max_steps: int, schedule_type: str = 'three_stage') -> float:
    """
    Get SemanticPhase weight based on training progress.

    Three-stage schedule based on diagnostic findings:
    - Stage 1 (0-10%): 100% strength - early basin finding
    - Stage 2 (10-80%): ramp down to 30% - avoid mid-phase drag
    - Stage 3 (80-100%): ramp up to 60% - late convergence advantage

    Args:
        step: Current training step
        max_steps: Total training steps
        schedule_type: 'constant' (1.0), 'three_stage', or 'cosine'

    Returns:
        Weight multiplier for SemanticPhase loss (0.0-1.0)
    """
    if schedule_type == 'constant':
        return 1.0

    progress = step / max_steps

    if schedule_type == 'three_stage':
        # Piecewise schedule matching observed training dynamics:
        # - Strong early (0-10%): SemanticPhase wins at step 300
        # - Low mid (10-80%): Baseline leads steps 600-1800
        # - Medium late (80-100%): SemanticPhase catches up steps 2400+

        early_end = 0.10      # First 10% of training
        mid_end = 0.80        # Middle 70% of training

        early_weight = 1.0    # Full strength early
        mid_weight = 0.30     # Reduced during mid-phase
        late_weight = 0.60    # Moderate for late convergence

        if progress < early_end:
            # Stage 1: Full strength
            return early_weight
        elif progress < mid_end:
            # Stage 2: Linear ramp down to mid_weight, then hold
            ramp_end = 0.25  # Ramp down until 25%
            if progress < ramp_end:
                ramp_progress = (progress - early_end) / (ramp_end - early_end)
                return early_weight + (mid_weight - early_weight) * ramp_progress
            else:
                return mid_weight
        else:
            # Stage 3: Linear ramp up to late_weight
            ramp_progress = (progress - mid_end) / (1.0 - mid_end)
            return mid_weight + (late_weight - mid_weight) * ramp_progress

    elif schedule_type == 'cosine':
        # Alternative: smooth cosine curve (high-low-medium)
        import math
        # Two-cycle cosine: high -> low -> medium
        if progress < 0.5:
            # First half: high to low
            return 0.5 + 0.5 * math.cos(progress * 2 * math.pi)
        else:
            # Second half: low to medium
            return 0.3 + 0.3 * math.cos((progress - 0.5) * 2 * math.pi)

    return 1.0  # Default


def lindblad_distill_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    teacher_temp: float = 0.7,     # sharpen teacher
    student_temp: float = 1.0,
    conf_thresh: float = 0.65,     # only enforce on confident teacher tokens
    max_loss: float = 5.0
) -> torch.Tensor:
    """
    Confidence-gated EMA distillation under noise.

    Key improvements over vanilla KL:
    1. Teacher is stable (EMA) - can't wiggle the target
    2. Only enforce on confident teacher tokens - can't cheat via uniform uncertainty
    3. KL(teacher || student) direction - pushes student toward teacher's confident mass

    L = CE(softmax(teacher/tt), log_softmax(student/st)) over confident tokens.
    """
    # [B, T, V]
    t_probs = torch.nn.functional.softmax(teacher_logits / teacher_temp, dim=-1)    # teacher targets
    s_logp = torch.nn.functional.log_softmax(student_logits / student_temp, dim=-1) # student log-probs

    # Teacher confidence per token: max probability
    t_conf = t_probs.max(dim=-1).values  # [B, T]

    # Valid token mask: non-pad tokens
    # (confidence gating disabled for now - early models have very low confidence)
    valid = (attention_mask > 0.5)  # [B, T]
    # Optionally add confidence gating when teacher is trained enough
    # if conf_thresh > 0:
    #     valid = valid & (t_conf >= conf_thresh)

    # Tokenwise CE with soft targets: -sum(p_t * log p_s)
    token_ce = -(t_probs * s_logp).sum(dim=-1)  # [B, T]

    # Masked mean
    denom = valid.float().sum().clamp_min(1.0)
    loss = (token_ce * valid.float()).sum() / denom

    return loss.clamp(max=max_loss)


class PilotDatasetV2(Dataset):
    """
    Enhanced dataset with paradigm-aware indexing.

    Maintains indices by:
    - word (for polysemy grouping)
    - paradigm (for paradigm-specific batching)
    - pair_type (positive/negative)
    """

    def __init__(
        self,
        data_path: str,
        max_length: int = 128,
        vocab_size: int = 1000
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Load examples
        self.examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} pilot examples")

        # Build vocab and indices
        self._build_vocab()
        self._build_indices()

    def _build_vocab(self):
        """Build vocabulary from the pilot data."""
        word_counts = {}
        for ex in self.examples:
            text = ex['input_text'] + ' ' + ex['output_text']
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

        self.word_to_id = {'<pad>': 0, '<unk>': 1}
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - 2]):
            self.word_to_id[word] = i + 2

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}

    def _build_indices(self):
        """Build paradigm-aware indices for strategic sampling."""
        # Index by paradigm
        self.by_paradigm = defaultdict(list)

        # Index polysemy by word for contrastive batching
        self.polysemy_by_word = defaultdict(lambda: {'positive': [], 'negative': [], 'hard': []})

        # Index lindblad by canonical for paired views
        self.lindblad_by_canonical = defaultdict(list)

        for idx, ex in enumerate(self.examples):
            paradigm = ex['paradigm']
            self.by_paradigm[paradigm].append(idx)

            if paradigm == 'semantic_phase':
                word = ex['metadata'].get('word', 'unknown')
                subtype = ex['subtype']
                if 'positive' in subtype:
                    self.polysemy_by_word[word]['positive'].append(idx)
                elif 'hard' in subtype:
                    self.polysemy_by_word[word]['hard'].append(idx)
                else:
                    self.polysemy_by_word[word]['negative'].append(idx)

            elif paradigm == 'lindblad':
                canonical = ex['metadata'].get('canonical', ex['input_text'][:50])
                self.lindblad_by_canonical[canonical].append(idx)

        # Log index stats
        print(f"\nIndex stats:")
        print(f"  Paradigms: {list(self.by_paradigm.keys())}")
        print(f"  Polysemy words: {list(self.polysemy_by_word.keys())}")
        print(f"  Lindblad canonicals: {len(self.lindblad_by_canonical)}")

    def _tokenize(self, text: str) -> List[int]:
        """Simple whitespace tokenization."""
        tokens = []
        for word in text.lower().split():
            tokens.append(self.word_to_id.get(word, 1))  # 1 = <unk>
        return tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Combine input and output
        text = example['input_text'] + ' ' + example['output_text']
        token_ids = self._tokenize(text)

        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = (input_ids != 0).float()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
            'paradigm': example['paradigm'],
            'subtype': example.get('subtype', ''),
            'idx': idx  # Track original index for debugging
        }


class StratifiedParadigmSampler(Sampler):
    """
    Stratified sampler that GUARANTEES paradigm presence from step 1.

    Each batch contains:
    - K semantic_phase examples (grouped by word for contrastive)
    - M lindblad examples (when enabled)
    - Q qualia examples (when enabled)

    This eliminates the "basin lottery" where early batches have 0/0 activations.
    """

    def __init__(
        self,
        dataset: 'ParadigmDataset',
        batch_size: int = 8,
        enabled_paradigms: set = None,
        semantic_per_batch: int = 4,  # Grouped by word
        lindblad_per_batch: int = 2,
        qualia_per_batch: int = 1,
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.enabled_paradigms = enabled_paradigms or set()
        self.rng = random.Random(seed)

        # Quotas per enabled paradigm
        self.semantic_quota = semantic_per_batch if 'phase_contrastive' in self.enabled_paradigms else 0
        self.lindblad_quota = lindblad_per_batch if 'lindblad_invariance' in self.enabled_paradigms else 0
        self.qualia_quota = qualia_per_batch if 'qualia_diversity' in self.enabled_paradigms else 0

        # Pre-index by paradigm
        self.polysemy_words = list(dataset.polysemy_by_word.keys()) if hasattr(dataset, 'polysemy_by_word') else []
        self.lindblad_indices = dataset.by_paradigm.get('lindblad', [])
        self.qualia_indices = dataset.by_paradigm.get('qualia', [])
        self.other_indices = []
        for paradigm, indices in dataset.by_paradigm.items():
            if paradigm not in ('semantic_phase', 'lindblad', 'qualia'):
                self.other_indices.extend(indices)

        print(f"\nStratifiedParadigmSampler initialized:")
        print(f"  Enabled: {self.enabled_paradigms}")
        print(f"  Quotas per batch: semantic={self.semantic_quota}, lindblad={self.lindblad_quota}, qualia={self.qualia_quota}")
        print(f"  Available: polysemy_words={len(self.polysemy_words)}, lindblad={len(self.lindblad_indices)}, qualia={len(self.qualia_indices)}")

    def __iter__(self):
        """Generate batches with guaranteed paradigm presence."""
        # Shuffle all pools
        words = self.polysemy_words.copy()
        self.rng.shuffle(words)
        lindblad = self.lindblad_indices.copy()
        self.rng.shuffle(lindblad)
        qualia = self.qualia_indices.copy()
        self.rng.shuffle(qualia)
        other = self.other_indices.copy()
        self.rng.shuffle(other)

        # Circular iterators for guaranteed presence
        word_idx = 0
        lindblad_idx = 0
        qualia_idx = 0
        other_idx = 0

        total_examples = len(self.dataset)
        yielded = 0

        while yielded < total_examples:
            batch = []

            # 1. Add semantic_phase examples (grouped by word)
            if self.semantic_quota > 0 and words:
                word = words[word_idx % len(words)]
                word_idx += 1
                if hasattr(self.dataset, 'polysemy_by_word'):
                    word_data = self.dataset.polysemy_by_word.get(word, {})
                    available = (
                        word_data.get('positive', []) +
                        word_data.get('negative', []) +
                        word_data.get('hard', [])
                    )
                    self.rng.shuffle(available)
                    batch.extend(available[:self.semantic_quota])

            # 2. Add lindblad examples (guaranteed when enabled)
            if self.lindblad_quota > 0 and lindblad:
                for _ in range(self.lindblad_quota):
                    batch.append(lindblad[lindblad_idx % len(lindblad)])
                    lindblad_idx += 1

            # 3. Add qualia examples (guaranteed when enabled)
            if self.qualia_quota > 0 and qualia:
                for _ in range(self.qualia_quota):
                    batch.append(qualia[qualia_idx % len(qualia)])
                    qualia_idx += 1

            # 4. Fill remaining with other examples
            remaining = self.batch_size - len(batch)
            if remaining > 0 and other:
                for _ in range(remaining):
                    batch.append(other[other_idx % len(other)])
                    other_idx += 1

            # Yield batch indices
            if batch:
                self.rng.shuffle(batch)  # Shuffle within batch
                for idx in batch:
                    if yielded < total_examples:
                        yield idx
                        yielded += 1

    def __len__(self):
        return len(self.dataset)


class PolysemyAwareSampler(Sampler):
    """
    Original sampler - kept for backward compatibility.
    Use StratifiedParadigmSampler for combined runs.
    """

    def __init__(
        self,
        dataset: 'ParadigmDataset',
        batch_size: int = 8,
        polysemy_ratio: float = 0.5,
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.polysemy_ratio = polysemy_ratio
        self.rng = random.Random(seed)

        # Pre-compute polysemy groups
        self.polysemy_words = list(dataset.polysemy_by_word.keys()) if hasattr(dataset, 'polysemy_by_word') else []
        self.other_indices = []
        for paradigm, indices in dataset.by_paradigm.items():
            if paradigm != 'semantic_phase':
                self.other_indices.extend(indices)

        print(f"\nPolysemySampler initialized:")
        print(f"  Polysemy words: {len(self.polysemy_words)}")
        print(f"  Other examples: {len(self.other_indices)}")

    def __iter__(self):
        """Generate batches with polysemy grouping."""
        # Shuffle word order
        words = self.polysemy_words.copy()
        self.rng.shuffle(words)

        # Shuffle other indices
        other = self.other_indices.copy()
        self.rng.shuffle(other)

        all_batches = []
        word_idx = 0
        other_idx = 0

        while word_idx < len(words) or other_idx < len(other):
            batch = []

            # Add polysemy examples (grouped by word)
            polysemy_count = int(self.batch_size * self.polysemy_ratio)
            if word_idx < len(words) and polysemy_count > 0:
                word = words[word_idx]
                word_data = self.dataset.polysemy_by_word[word]

                # Get positive, negative, and hard examples for this word
                available = (
                    word_data['positive'] +
                    word_data['negative'] +
                    word_data['hard']
                )
                self.rng.shuffle(available)

                batch.extend(available[:polysemy_count])
                word_idx += 1

            # Fill rest with other paradigms
            remaining = self.batch_size - len(batch)
            if other_idx < len(other) and remaining > 0:
                batch.extend(other[other_idx:other_idx + remaining])
                other_idx += remaining

            if batch:
                all_batches.append(batch)

        # Shuffle batch order
        self.rng.shuffle(all_batches)

        # Yield indices
        for batch in all_batches:
            yield from batch

    def __len__(self):
        return len(self.dataset)


class LindbladTwinCollator:
    """
    Collator that produces clean/noisy twins for Lindblad invariance learning.

    For each Lindblad example, generates a paired noisy version on-the-fly.
    This is more efficient than pre-generating all pairs.
    """

    NOISE_FUNCTIONS = [
        ('token_drop', lambda ids, p=0.1: [t for t in ids if random.random() > p]),
        ('token_swap', lambda ids, p=0.1: LindbladTwinCollator._swap_tokens(ids, p)),
        ('token_repeat', lambda ids, p=0.05: LindbladTwinCollator._repeat_tokens(ids, p)),
        ('mask_noise', lambda ids, p=0.15: [1 if random.random() < p else t for t in ids]),  # 1=unk
    ]

    def __init__(self, max_length: int = 128, noise_level: float = 0.3):
        self.max_length = max_length
        self.noise_level = noise_level

    @staticmethod
    def _swap_tokens(ids: List[int], p: float) -> List[int]:
        ids = ids.copy()
        for i in range(len(ids) - 1):
            if random.random() < p:
                ids[i], ids[i+1] = ids[i+1], ids[i]
        return ids

    @staticmethod
    def _repeat_tokens(ids: List[int], p: float) -> List[int]:
        result = []
        for t in ids:
            result.append(t)
            if random.random() < p:
                result.append(t)
        return result

    def add_noise(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply noise to create a noisy twin."""
        ids = input_ids.tolist()

        # Apply random noise function
        noise_name, noise_fn = random.choice(self.NOISE_FUNCTIONS)
        noisy_ids = noise_fn(ids, self.noise_level)

        # Truncate or pad back to max_length
        if len(noisy_ids) > self.max_length:
            noisy_ids = noisy_ids[:self.max_length]
        else:
            noisy_ids = noisy_ids + [0] * (self.max_length - len(noisy_ids))

        return torch.tensor(noisy_ids, dtype=torch.long)

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate batch with Lindblad twins."""
        input_ids = torch.stack([b['input_ids'] for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
        labels = torch.stack([b['labels'] for b in batch])
        paradigms = [b['paradigm'] for b in batch]
        subtypes = [b.get('subtype', '') for b in batch]

        # Generate noisy twins for Lindblad examples
        noisy_ids = []
        noisy_mask = []
        has_lindblad = False

        for i, paradigm in enumerate(paradigms):
            if paradigm == 'lindblad':
                has_lindblad = True
                noisy = self.add_noise(input_ids[i])
                noisy_ids.append(noisy)
                noisy_mask.append((noisy != 0).float())
            else:
                # Placeholder for non-Lindblad (same as original)
                noisy_ids.append(input_ids[i])
                noisy_mask.append(attention_mask[i])

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'paradigm': paradigms,
            'subtype': subtypes,
        }

        if has_lindblad:
            result['noisy_input_ids'] = torch.stack(noisy_ids)
            result['noisy_attention_mask'] = torch.stack(noisy_mask)

        return result


class ActivationLogger:
    """Logs activation counts and loss contributions per step for debugging paradigm signal."""

    def __init__(self):
        self.step_logs = []
        # Track first step with non-zero activation (eliminates warmup gap diagnosis)
        self.first_step_with = {
            'phase_contrastive': None,
            'lindblad_invariance': None,
            'qualia_diversity': None
        }
        # Cumulative loss contributions
        self.cumulative_loss = {
            'phase_contrastive': 0.0,
            'lindblad_invariance': 0.0,
            'qualia_diversity': 0.0
        }

    def log_step(
        self,
        step: int,
        contrastive_pairs: int,
        lindblad_twins: int,
        phase_collapse_score: float,
        losses: Dict[str, float]
    ):
        """Log activation counts and loss contributions for this step."""
        self.step_logs.append({
            'step': step,
            'contrastive_pairs_found': contrastive_pairs,
            'lindblad_twins_generated': lindblad_twins,
            'phase_collapse_score': phase_collapse_score,
            **losses
        })

        # Track first step with non-zero activation
        if contrastive_pairs > 0 and self.first_step_with['phase_contrastive'] is None:
            self.first_step_with['phase_contrastive'] = step
        if lindblad_twins > 0 and self.first_step_with['lindblad_invariance'] is None:
            self.first_step_with['lindblad_invariance'] = step
        if losses.get('qualia_diversity', 0) > 0 and self.first_step_with['qualia_diversity'] is None:
            self.first_step_with['qualia_diversity'] = step

        # Accumulate loss contributions
        self.cumulative_loss['phase_contrastive'] += losses.get('phase_contrastive', 0)
        self.cumulative_loss['lindblad_invariance'] += losses.get('lindblad_invariance', 0)
        self.cumulative_loss['qualia_diversity'] += losses.get('qualia_diversity', 0)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.step_logs:
            return {}

        total_contrastive = sum(log['contrastive_pairs_found'] for log in self.step_logs)
        total_lindblad = sum(log['lindblad_twins_generated'] for log in self.step_logs)
        avg_collapse = sum(log['phase_collapse_score'] for log in self.step_logs) / len(self.step_logs)

        return {
            'total_contrastive_pairs': total_contrastive,
            'total_lindblad_twins': total_lindblad,
            'avg_phase_collapse_score': avg_collapse,
            'steps_logged': len(self.step_logs),
            'first_step_with': self.first_step_with,
            'cumulative_loss_contribution': self.cumulative_loss
        }

    def save(self, path: Path):
        """Save detailed logs to file."""
        with open(path, 'w') as f:
            json.dump({
                'step_logs': self.step_logs,
                'summary': self.summary()
            }, f, indent=2)


class ParadigmAwareTrainer:
    """
    Trainer that properly handles paradigm-aware batching.

    Hooks into the loss computation to:
    1. Extract contrastive pairs from polysemy batches
    2. Use Lindblad twins for invariance loss
    3. Log activation counts
    """

    def __init__(
        self,
        model: QLLM,
        train_dataset: PilotDatasetV2,
        train_config: TrainingConfig,
        device: str = 'cuda',
        activation_logger: Optional[ActivationLogger] = None,
        enabled_losses: Optional[List[str]] = None,
        semantic_phase_schedule: str = 'three_stage',
        lindblad_schedule: str = 'late_ramp',
        phase_base_weight: float = 0.10,
        lindblad_base_weight: float = 0.05,
        qualia_base_weight: float = 0.05,
        max_steps: int = 3000,
        seed: int = 42
    ):
        self.model = model.to(device)
        self.dataset = train_dataset
        self.config = train_config
        self.device = device
        self.logger = activation_logger or ActivationLogger()
        self.seed = seed

        # Loss configuration
        self.enabled_losses = set(enabled_losses or [])
        self.semantic_phase_schedule = semantic_phase_schedule
        self.lindblad_schedule = lindblad_schedule
        self.phase_base_weight = phase_base_weight
        self.lindblad_base_weight = lindblad_base_weight
        self.qualia_base_weight = qualia_base_weight
        self.max_steps_for_schedule = max_steps

        # Create paradigm-aware dataloader
        # Use StratifiedParadigmSampler when paradigm losses are enabled
        # This guarantees paradigm presence from step 1, eliminating "basin lottery"
        if self.enabled_losses:
            sampler = StratifiedParadigmSampler(
                train_dataset,
                batch_size=train_config.batch_size,
                enabled_paradigms=self.enabled_losses,
                semantic_per_batch=4,
                lindblad_per_batch=2,
                qualia_per_batch=1,
                seed=self.seed
            )
        else:
            # Baseline: no paradigm losses, use simple sampler
            sampler = PolysemyAwareSampler(
                train_dataset,
                batch_size=train_config.batch_size
            )
        collator = LindbladTwinCollator(max_length=train_dataset.max_length)

        self.dataloader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            sampler=sampler,
            collate_fn=collator,
            drop_last=True
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )

        # EMA teacher model for stable Lindblad distillation targets
        # This prevents the model from "cheating" by making the clean target blurrier
        import copy
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.to(device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_decay = 0.999

        # Initialize contrastive objective
        self._init_contrastive()

    def _init_contrastive(self):
        """Initialize contrastive phase objective."""
        try:
            from qllm.layers.semantic_phase import ContrastivePhaseObjective
            self.contrastive = ContrastivePhaseObjective(
                embedding_dim=self.model.config.hidden_dim,
                temperature=0.1,
                margin=0.5
            ).to(self.device)
        except ImportError:
            self.contrastive = None
            print("Warning: ContrastivePhaseObjective not available")

    @torch.no_grad()
    def _ema_update(self):
        """Update EMA teacher model after each optimizer step."""
        d = self.ema_decay
        msd = self.model.state_dict()
        esd = self.ema_model.state_dict()
        for k in esd.keys():
            esd[k].mul_(d).add_(msd[k], alpha=1.0 - d)

    def compute_paradigm_losses(
        self,
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Compute paradigm-specific losses with proper pairing and schedule."""
        losses = {}

        paradigms = batch['paradigm']
        input_ids = batch['input_ids'].to(self.device)

        # Get embeddings
        real_embed = outputs.get('real_embedding')
        imag_embed = outputs.get('imag_embedding')

        # --- CONTRASTIVE PHASE LOSS (with gating + schedule) ---
        contrastive_pairs = 0
        if 'phase_contrastive' in self.enabled_losses and self.contrastive and real_embed is not None and imag_embed is not None:
            # Find polysemy examples in batch
            polysemy_mask = [p == 'semantic_phase' for p in paradigms]

            if sum(polysemy_mask) >= 2:
                # Mine contrastive pairs within batch
                polysemy_indices = [i for i, m in enumerate(polysemy_mask) if m]

                # Use input_ids to find repeated tokens (our anchor points)
                batch_tokens = input_ids[polysemy_indices]  # [num_polysemy, seq_len]

                # Count token occurrences across batch
                flat_tokens = batch_tokens.flatten()
                unique, counts = torch.unique(flat_tokens, return_counts=True)
                repeated_tokens = unique[counts > 1]
                repeated_tokens = repeated_tokens[repeated_tokens > 1]  # Exclude pad/unk

                contrastive_pairs = len(repeated_tokens)

                if contrastive_pairs > 0:
                    # Compute in-batch contrastive loss
                    phase_loss = self.contrastive.in_batch_contrastive_loss(
                        real_embed[polysemy_indices],
                        imag_embed[polysemy_indices],
                        batch_tokens
                    )

                    # Always log raw for visibility
                    phase_loss_val = phase_loss.item()
                    losses['phase_contrastive_raw'] = phase_loss_val

                    if phase_loss_val > 0:
                        # Apply scheduled weighting (THIS is the annealing fix)
                        sched = get_semantic_phase_schedule(
                            step=step,
                            max_steps=self.max_steps_for_schedule,
                            schedule_type=self.semantic_phase_schedule
                        )
                        losses['semantic_phase_weight'] = float(sched)  # Log for QA
                        losses['phase_contrastive'] = phase_loss * (self.phase_base_weight * sched)

        # --- LINDBLAD INVARIANCE LOSS (EMA Distillation) ---
        # Key improvements over vanilla KL:
        # 1. EMA teacher provides stable target (can't wiggle)
        # 2. Confidence gating: only enforce on confident teacher tokens
        # 3. KL(teacher||student) direction: pushes student toward teacher's confident mass
        lindblad_twins = 0
        if 'lindblad_invariance' in self.enabled_losses and 'noisy_input_ids' in batch:
            lindblad_mask = [p == 'lindblad' for p in paradigms]

            if sum(lindblad_mask) > 0:
                lindblad_indices = [i for i, m in enumerate(lindblad_mask) if m]
                lindblad_twins = len(lindblad_indices)

                # Get clean and noisy inputs for lindblad examples
                clean_ids = batch['input_ids'][lindblad_indices].to(self.device)
                clean_mask = batch['attention_mask'][lindblad_indices].to(self.device)
                noisy_ids = batch['noisy_input_ids'][lindblad_indices].to(self.device)
                noisy_mask = batch['noisy_attention_mask'][lindblad_indices].to(self.device)

                # Student forward on noisy (WITH gradients)
                student_noisy = self.model(input_ids=noisy_ids, attention_mask=noisy_mask)
                noisy_logits = student_noisy.get('logits')

                # Teacher forward on clean (EMA model, NO gradients)
                with torch.no_grad():
                    teacher_clean = self.ema_model(input_ids=clean_ids, attention_mask=clean_mask)
                    clean_logits = teacher_clean.get('logits')

                if noisy_logits is not None and clean_logits is not None:
                    # Confidence-gated distillation loss
                    # Note: conf_thresh lowered to 0.05 because early in training,
                    # model confidence is low (max prob ≈ 1/vocab_size for random init)
                    distill_loss = lindblad_distill_loss(
                        teacher_logits=clean_logits,
                        student_logits=noisy_logits,
                        attention_mask=clean_mask,
                        teacher_temp=1.0,      # no sharpening initially
                        student_temp=1.0,
                        conf_thresh=0.05,      # lower threshold for early training
                        max_loss=5.0
                    )

                    losses['lindblad_distill_raw'] = distill_loss.item()

                    if distill_loss.item() > 0.01:
                        # Apply late-ramp schedule
                        lindblad_sched = get_lindblad_schedule(
                            step, self.max_steps_for_schedule,
                            schedule_type=self.lindblad_schedule
                        )
                        losses['lindblad_weight'] = float(lindblad_sched)
                        losses['lindblad_invariance'] = distill_loss * (self.lindblad_base_weight * lindblad_sched)

        # --- QUALIA SELF-SUPERVISED LOSS (with gating) ---
        qualia = outputs.get('qualia')
        if 'qualia_diversity' in self.enabled_losses and qualia is not None and isinstance(qualia, torch.Tensor):
            qualia_mask = [p == 'qualia' for p in paradigms]

            if sum(qualia_mask) > 0:
                qualia_indices = torch.tensor([i for i, m in enumerate(qualia_mask) if m], device=qualia.device)
                qualia_values = qualia[qualia_indices]

                # Diversity loss: encourage spread across channels
                channel_std = qualia_values.std(dim=0).mean()
                if channel_std < 0.3:  # Too collapsed
                    diversity_loss = 0.3 - channel_std
                    losses['qualia_diversity'] = diversity_loss * self.qualia_base_weight

        return losses, {
            'contrastive_pairs': contrastive_pairs,
            'lindblad_twins': lindblad_twins
        }

    def train_step(self, batch: Dict[str, Any], step: int) -> Dict[str, float]:
        """Single training step with paradigm-aware losses."""
        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass (model returns phase/qualia based on config)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get phase embeddings directly from embedding layer
        if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'real_embedding'):
            real_embed = self.model.embeddings.real_embedding(input_ids)
            imag_embed = self.model.embeddings.imag_embedding(input_ids)
            outputs['real_embedding'] = real_embed
            outputs['imag_embedding'] = imag_embed

        # Get qualia from qualia_info
        if outputs.get('qualia_info') is not None:
            outputs['qualia'] = outputs['qualia_info'].get('qualia_values')

        # Language modeling loss
        logits = outputs['logits']
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Clamp for stability
        shift_logits = torch.clamp(shift_logits, min=-100, max=100)

        lm_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0  # Ignore padding
        )

        # Paradigm-specific losses (with step for schedule)
        paradigm_losses, counts = self.compute_paradigm_losses(batch, outputs, step=step)

        # Total loss
        total_loss = lm_loss
        for name, loss in paradigm_losses.items():
            # Skip diagnostic entries (floats) and only add tensor losses
            if isinstance(loss, torch.Tensor) and torch.isfinite(loss):
                total_loss = total_loss + loss

        # Check for NaN
        if not torch.isfinite(total_loss):
            print(f"Warning: NaN loss at step {step}, skipping")
            return {'loss': float('nan')}

        # Backward and optimize
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update EMA teacher for stable Lindblad targets
        self._ema_update()

        # Compute phase collapse score
        real_embed = outputs.get('real_embedding')
        imag_embed = outputs.get('imag_embedding')
        collapse_score = 0.0
        if real_embed is not None and imag_embed is not None:
            # Circular variance (low = collapsed)
            phases = torch.atan2(imag_embed, real_embed)
            phase_mean = phases.mean(dim=(0, 1))
            phase_var = 1.0 - torch.sqrt(
                torch.cos(phases - phase_mean).mean() ** 2 +
                torch.sin(phases - phase_mean).mean() ** 2
            )
            collapse_score = phase_var.mean().item()

        # Log activations
        loss_dict = {'lm_loss': lm_loss.item()}
        for name, loss in paradigm_losses.items():
            loss_dict[name] = loss.item() if isinstance(loss, torch.Tensor) else loss

        self.logger.log_step(
            step=step,
            contrastive_pairs=counts['contrastive_pairs'],
            lindblad_twins=counts['lindblad_twins'],
            phase_collapse_score=collapse_score,
            losses=loss_dict
        )

        return {'loss': total_loss.item(), **loss_dict}

    def train(self, max_steps: int = 500, log_every: int = 50, avg_last_n: int = 100) -> Dict[str, Any]:
        """Train with paradigm-aware batching."""
        print(f"\nStarting paradigm-aware training for {max_steps} steps...")

        step = 0
        epoch = 0
        best_loss = float('inf')
        best_loss_late = float('inf')  # Best loss after 66% of training
        late_start_step = int(max_steps * 0.66)  # ~step 2000 for 3000 steps
        loss_history = []  # Track all losses for robust metrics

        while step < max_steps:
            epoch += 1
            for batch in self.dataloader:
                if step >= max_steps:
                    break

                metrics = self.train_step(batch, step)
                step += 1

                # Track loss for averaging
                if not np.isnan(metrics['loss']):
                    loss_history.append(metrics['loss'])

                if step % log_every == 0:
                    loss_str = f"Loss: {metrics['loss']:.4f}"
                    if 'phase_contrastive' in metrics:
                        loss_str += f" | phase: {metrics['phase_contrastive']:.4f}"
                    if 'lindblad_invariance' in metrics:
                        loss_str += f" | lindblad: {metrics['lindblad_invariance']:.4f}"
                        if 'lindblad_kl' in metrics:
                            loss_str += f" (kl:{metrics['lindblad_kl']:.3f})"

                    print(f"Step {step} | Epoch {epoch} | {loss_str}")

                    # Log activation summary
                    recent_logs = self.logger.step_logs[-log_every:]
                    avg_pairs = sum(l['contrastive_pairs_found'] for l in recent_logs) / len(recent_logs)
                    avg_twins = sum(l['lindblad_twins_generated'] for l in recent_logs) / len(recent_logs)
                    print(f"  Activations: {avg_pairs:.1f} contrastive pairs, {avg_twins:.1f} lindblad twins/step")

                # Track best loss (overall)
                if metrics['loss'] < best_loss:
                    best_loss = metrics['loss']

                # Track best loss (late stage only) - isolates late-convergence advantage
                if step >= late_start_step and metrics['loss'] < best_loss_late:
                    best_loss_late = metrics['loss']

        # Compute robust metrics
        avg_last_n_loss = np.mean(loss_history[-avg_last_n:]) if len(loss_history) >= avg_last_n else np.mean(loss_history)
        std_last_n_loss = np.std(loss_history[-avg_last_n:]) if len(loss_history) >= avg_last_n else np.std(loss_history)
        final_loss = loss_history[-1] if loss_history else float('nan')

        print(f"\nTraining complete.")
        print(f"  Best loss: {best_loss:.4f}")
        print(f"  Best loss (late, >{late_start_step}): {best_loss_late:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Avg last {avg_last_n}: {avg_last_n_loss:.4f} (±{std_last_n_loss:.4f})")
        print(f"Activation summary: {self.logger.summary()}")

        return {
            'best_loss': best_loss,
            'best_loss_late': best_loss_late,  # Best after 66% of training
            'final_loss': final_loss,
            'avg_last_n_loss': avg_last_n_loss,
            'std_last_n_loss': std_last_n_loss,
            'avg_last_n': avg_last_n,
            'late_start_step': late_start_step,
            'final_step': step,
            'loss_history': loss_history,  # Full history for analysis
            'activation_summary': self.logger.summary()
        }


def run_proof_ablation_v2(args):
    """Run proof ablation with paradigm-aware batching."""

    print("=" * 70)
    print("  PROOF ABLATION V2 - PARADIGM-AWARE BATCHING")
    print("  Each batch guarantees contrastive pairs + noise twins")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load dataset with enhanced indexing
    print(f"\nLoading pilot data from: {args.data}")
    dataset = PilotDatasetV2(args.data, max_length=args.max_length)

    # Count by paradigm
    paradigm_counts = {}
    for i in range(len(dataset)):
        p = dataset.examples[i]['paradigm']
        paradigm_counts[p] = paradigm_counts.get(p, 0) + 1

    print(f"\nExamples by paradigm:")
    for p, c in sorted(paradigm_counts.items()):
        print(f"  {p}: {c}")

    # Model config
    model_config = QLLMConfig(
        vocab_size=dataset.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=4,
        intermediate_dim=args.hidden_dim * 2,
        max_seq_length=args.max_length,
        use_semantic_phase=True,
        semantic_phase_dim=args.hidden_dim,
        use_retrocausal_attention=True,
        retrocausal_layers=[0, args.num_layers // 2] if args.num_layers > 1 else [0],
        use_lindblad_layers=True,
        lindblad_every_n_layers=2,
        use_qualia_output=True,
        num_qualia_channels=8,
        use_emergent_init=True
    )

    print(f"\nModel config:")
    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Max length: {model_config.max_seq_length}")

    # Training config
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        warmup_steps=min(50, args.steps // 5),
        gradient_accumulation_steps=1,
        logging_steps=max(10, args.steps // 10),
        eval_steps=args.steps // 2,
        save_steps=args.steps
    )

    print(f"\nTraining config:")
    print(f"  Steps per ablation: {args.steps}")
    print(f"  Batch size: {args.batch_size}")

    # Output dir (includes seed for reproducibility tracking)
    output_dir = Path(args.output_dir) / f"proof_v2_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSeed: {args.seed}")
    print(f"Output: {output_dir}")

    # Run ablation configs
    ablation_configs = [
        ('baseline', []),
        ('semantic_phase', ['phase_contrastive']),
        ('lindblad', ['lindblad_invariance']),
        ('qualia', ['qualia_diversity']),
        ('all_paradigms', ['phase_contrastive', 'lindblad_invariance', 'qualia_diversity']),
    ]

    results = {}

    for ablation_name, enabled_losses in ablation_configs:
        print("\n" + "=" * 60)
        print(f"ABLATION: {ablation_name}")
        print(f"Losses enabled: {enabled_losses}")
        print("=" * 60)

        # Fresh model for each ablation
        model = QLLM(model_config)
        logger = ActivationLogger()

        trainer = ParadigmAwareTrainer(
            model=model,
            train_dataset=dataset,
            train_config=train_config,
            device=device,
            activation_logger=logger,
            enabled_losses=enabled_losses,
            semantic_phase_schedule=args.semantic_phase_schedule,
            max_steps=args.steps,
            seed=args.seed
        )

        # Train
        result = trainer.train(
            max_steps=args.steps,
            log_every=args.steps // 10,
            avg_last_n=args.avg_last_n
        )

        results[ablation_name] = result

        # Save activation logs
        logger.save(output_dir / f"{ablation_name}_activations.json")

        # Save model checkpoint
        ablation_dir = output_dir / ablation_name
        ablation_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ablation_dir / "model.pt")

    # Print comparison
    print("\n" + "=" * 70)
    print("  ABLATION COMPARISON")
    print("=" * 70)

    baseline_best = results.get('baseline', {}).get('best_loss', float('inf'))
    baseline_best_late = results.get('baseline', {}).get('best_loss_late', float('inf'))
    baseline_avg = results.get('baseline', {}).get('avg_last_n_loss', float('inf'))

    for name, result in results.items():
        best_loss = result.get('best_loss', float('inf'))
        best_loss_late = result.get('best_loss_late', float('inf'))
        final_loss = result.get('final_loss', float('nan'))
        avg_loss = result.get('avg_last_n_loss', float('inf'))
        std_loss = result.get('std_last_n_loss', 0)
        avg_n = result.get('avg_last_n', 100)
        late_start = result.get('late_start_step', 0)

        # Compute deltas vs baseline
        delta_best = best_loss - baseline_best
        delta_best_pct = (delta_best / baseline_best) * 100 if baseline_best > 0 else 0
        delta_best_late = best_loss_late - baseline_best_late
        delta_best_late_pct = (delta_best_late / baseline_best_late) * 100 if baseline_best_late > 0 else 0
        delta_avg = avg_loss - baseline_avg
        delta_avg_pct = (delta_avg / baseline_avg) * 100 if baseline_avg > 0 else 0

        summary = result.get('activation_summary', {})
        pairs = summary.get('total_contrastive_pairs', 0)
        twins = summary.get('total_lindblad_twins', 0)

        print(f"\n{name}:")
        print(f"  Best loss: {best_loss:.4f} ({delta_best:+.4f}, {delta_best_pct:+.2f}%)")
        print(f"  Best loss (late >{late_start}): {best_loss_late:.4f} ({delta_best_late:+.4f}, {delta_best_late_pct:+.2f}%)")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Avg last {avg_n}: {avg_loss:.4f} ±{std_loss:.4f} ({delta_avg:+.4f}, {delta_avg_pct:+.2f}%)")
        print(f"  Contrastive pairs: {pairs}")
        print(f"  Lindblad twins: {twins}")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
        'avg_last_n': args.avg_last_n,
        'semantic_phase_schedule': args.semantic_phase_schedule,
        'data_path': str(args.data),
        'steps_per_run': args.steps,
        'paradigm_counts': paradigm_counts,
        'results': results
    }

    with open(output_dir / "proof_v2_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run proof ablation v2 with paradigm-aware batching")
    parser.add_argument('--data', type=str, default='./data/pilot_paradigm_data.jsonl',
                        help='Path to pilot dataset')
    parser.add_argument('--steps', type=int, default=500,
                        help='Training steps per ablation')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of layers')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Max sequence length')
    parser.add_argument('--output-dir', type=str, default='./proof_ablation_runs',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--avg-last-n', type=int, default=300,
                        help='Number of final steps to average for robust loss metric')
    parser.add_argument('--semantic-phase-schedule', type=str, default='three_stage',
                        choices=['constant', 'three_stage', 'cosine'],
                        help='Schedule for SemanticPhase weighting (three_stage recommended)')

    args = parser.parse_args()

    # Set seed before anything else
    set_seed(args.seed)

    run_proof_ablation_v2(args)


if __name__ == "__main__":
    main()
