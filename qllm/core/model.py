"""
QLLM - Quantum-Inspired Large Language Model
=============================================

The main model that integrates all paradigm layers:
1. Semantic Phase Embeddings
2. Retrocausal Attention
3. Lindblad Dissipative Layers
4. Qualia Output Head
5. Emergent Computation

This model can be initialized from a pre-trained LLM (like LLaMA)
and enhanced with quantum-paradigm layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math

from .config import QLLMConfig, TrainingConfig
from ..layers.semantic_phase import SemanticPhaseEmbedding, PhaseModulator
from ..layers.retrocausal import RetrocausalAttention, HindsightEnhancedBlock
from ..layers.lindblad import LindbladLayer, DissipativeNormalization
from ..layers.qualia import QualiaEncoder, QualiaOutputHead
from ..layers.emergent import AttractorLayer, ComplexityTimeLayer


class QLLMBlock(nn.Module):
    """
    Single transformer block with quantum paradigm enhancements.

    Can operate in different modes:
    - standard: Regular transformer attention + FFN
    - retrocausal: Bidirectional attention (TSVF)
    - lindblad: With dissipative dynamics
    - full: All enhancements active
    """

    def __init__(
        self,
        config: QLLMConfig,
        layer_idx: int
    ):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim

        # Determine which enhancements to apply at this layer
        self.use_retrocausal = (
            config.use_retrocausal_attention and
            layer_idx in config.retrocausal_layers
        )
        self.use_lindblad = (
            config.use_lindblad_layers and
            layer_idx % config.lindblad_every_n_layers == 0
        )

        # Attention layer
        if self.use_retrocausal:
            self.attention = RetrocausalAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.attention_dropout,
                backward_weight=config.backward_attention_weight,
                interference_type=config.interference_type
            )
        else:
            self.attention = StandardAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.attention_dropout
            )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

        # Normalization layers
        if self.use_lindblad:
            self.norm1 = DissipativeNormalization(config.hidden_dim)
            self.norm2 = DissipativeNormalization(config.hidden_dim)
            self.lindblad = LindbladLayer(
                hidden_dim=config.hidden_dim,
                num_operators=config.num_lindblad_operators,
                dt=config.lindblad_dt,
                dissipation_strength=config.dissipation_strength
            )
        else:
            self.norm1 = nn.LayerNorm(config.hidden_dim)
            self.norm2 = nn.LayerNorm(config.hidden_dim)
            self.lindblad = None

        # Phase modulator (if using semantic phase)
        if config.use_semantic_phase:
            self.phase_modulator = PhaseModulator(
                dim=config.hidden_dim,
                num_heads=config.num_heads // 4,
                dropout=config.dropout
            )
        else:
            self.phase_modulator = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        phase_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_hint: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the block.

        Args:
            hidden_states: [batch, seq, dim]
            phase_states: [batch, seq, phase_dim] (optional)
            attention_mask: [batch, seq]
            target_hint: [batch, seq, dim] for retrocausal enhancement

        Returns:
            hidden_states: [batch, seq, dim]
            phase_states: [batch, seq, phase_dim] (if applicable)
        """
        residual = hidden_states

        # Attention
        normed = self.norm1(hidden_states)

        # Track retrocausal states for leak detection
        retrocausal_info = None
        if self.use_retrocausal:
            attn_out, retrocausal_info = self.attention(
                normed, attention_mask=attention_mask, return_states=True
            )
        else:
            attn_out = self.attention(normed, attention_mask)

        hidden_states = residual + attn_out

        # FFN
        residual = hidden_states
        normed = self.norm2(hidden_states)
        hidden_states = residual + self.ffn(normed)

        # Lindblad dynamics (if enabled)
        if self.lindblad is not None:
            hidden_states = self.lindblad(hidden_states)

        # Phase modulation (if enabled)
        if self.phase_modulator is not None and phase_states is not None:
            hidden_states, phase_states = self.phase_modulator(
                hidden_states, phase_states, attention_mask
            )

        return hidden_states, phase_states, retrocausal_info


class StandardAttention(nn.Module):
    """Standard causal self-attention for non-retrocausal layers"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        return output


class QLLM(nn.Module):
    """
    Quantum-Inspired Large Language Model

    Integrates all paradigm layers into a complete model:
    1. Semantic Phase Embeddings (input)
    2. Transformer blocks with Retrocausal Attention and Lindblad dynamics
    3. Qualia-modulated output head

    Can be:
    - Trained from scratch (small scale)
    - Initialized from pre-trained LLM weights
    - Fine-tuned with LoRA adapters
    """

    def __init__(self, config: QLLMConfig):
        super().__init__()

        self.config = config

        # === EMBEDDING LAYER ===
        if config.use_semantic_phase:
            self.embeddings = SemanticPhaseEmbedding(
                vocab_size=config.vocab_size,
                embedding_dim=config.hidden_dim,
                phase_dim=config.semantic_phase_dim,
                phase_modulation_strength=config.phase_modulation_strength,
                context_window=config.context_window_for_phase
            )
        else:
            self.embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Positional embeddings (RoPE-style or learned)
        self.pos_embeddings = nn.Embedding(config.max_seq_length, config.hidden_dim)

        # === TRANSFORMER BLOCKS ===
        self.blocks = nn.ModuleList([
            QLLMBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # === COMPLEXITY-TIME LAYER ===
        if config.use_emergent_init:
            self.complexity_time = ComplexityTimeLayer(
                hidden_dim=config.hidden_dim,
                complexity_measure=config.complexity_measure
            )
        else:
            self.complexity_time = None

        # === ATTRACTOR LAYER ===
        self.attractor_layer = AttractorLayer(
            hidden_dim=config.hidden_dim,
            num_attractors=16
        )

        # === FINAL NORM ===
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # === OUTPUT HEAD ===
        if config.use_qualia_output:
            self.output_head = QualiaOutputHead(
                hidden_dim=config.hidden_dim,
                vocab_size=config.vocab_size,
                num_qualia=config.num_qualia_channels,
                qualia_modulation_strength=config.qualia_modulation_strength
            )
        else:
            self.output_head = nn.Linear(config.hidden_dim, config.vocab_size)

        # === QUALIA ENCODER (for intermediate tracking) ===
        self.qualia_encoder = QualiaEncoder(
            hidden_dim=config.hidden_dim,
            num_qualia=config.num_qualia_channels
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights.

        Skip modules marked with _skip_init (set by PhaseModulator)
        to preserve their custom initialization.
        """
        # Skip modules with custom init (PhaseModulator marks its linears)
        if getattr(module, '_skip_init', False):
            return

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        target_hint: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] input token IDs
            attention_mask: [batch, seq_len] attention mask
            labels: [batch, seq_len] target labels (for loss)
            target_hint: [batch, seq_len, dim] future hint for retrocausal
            return_dict: Whether to return dict (True) or tuple
            output_hidden_states: Whether to return all hidden states

        Returns:
            Dict with: logits, loss (if labels), qualia_info, hidden_states,
                       forward_state, backward_state (for leak detection)
        """
        batch_size, seq_len = input_ids.shape

        # === EMBEDDINGS ===
        if self.config.use_semantic_phase:
            hidden_states, phase_states = self.embeddings(
                input_ids, attention_mask
            )
        else:
            hidden_states = self.embeddings(input_ids)
            phase_states = None

        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden_states = hidden_states + self.pos_embeddings(positions)

        # === TRANSFORMER BLOCKS ===
        all_hidden_states = []
        all_qualia = []
        forward_state = None
        backward_state = None

        for block in self.blocks:
            hidden_states, phase_states, retro_info = block(
                hidden_states,
                phase_states,
                attention_mask,
                target_hint
            )

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            # Capture retrocausal states from first retrocausal layer
            if retro_info is not None and forward_state is None:
                forward_state = retro_info.get('forward_state')
                backward_state = retro_info.get('backward_state')

            # Track qualia through layers
            if self.config.use_qualia_output:
                qualia = self.qualia_encoder(hidden_states)
                all_qualia.append(qualia)

        # === COMPLEXITY-TIME MODULATION ===
        if self.complexity_time is not None:
            hidden_states = self.complexity_time(hidden_states)

        # === ATTRACTOR DYNAMICS ===
        hidden_states = self.attractor_layer(hidden_states)

        # === FINAL NORM ===
        hidden_states = self.final_norm(hidden_states)

        # === OUTPUT HEAD ===
        if self.config.use_qualia_output:
            logits, qualia_info = self.output_head(
                hidden_states, return_qualia=True
            )
        else:
            logits = self.output_head(hidden_states)
            qualia_info = None

        # === LOSS COMPUTATION ===
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Clamp logits for numerical stability
            shift_logits = torch.clamp(shift_logits, min=-100, max=100)

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

            # Add paradigm-specific losses (only if loss is finite)
            if torch.isfinite(loss) and qualia_info is not None:
                # Phase coherence loss
                if phase_states is not None and self.config.use_semantic_phase:
                    # Get embeddings from the original embedding layer output
                    real_embed = self.embeddings.real_embedding(input_ids)
                    imag_embed = self.embeddings.imag_embedding(input_ids)
                    if self.config.semantic_phase_dim != self.config.hidden_dim:
                        imag_embed = F.pad(imag_embed, (0, self.config.hidden_dim - self.config.semantic_phase_dim))
                    coherence = self.embeddings.get_phase_coherence((real_embed, imag_embed))
                    coherence_loss = 0.1 * (1 - coherence.mean())
                    if torch.isfinite(coherence_loss):
                        loss = loss + coherence_loss

        if return_dict:
            return {
                'logits': logits,
                'loss': loss,
                'hidden_states': hidden_states,
                'phase_states': phase_states,
                'qualia_info': qualia_info,
                'all_hidden_states': all_hidden_states if all_hidden_states else None,
                'all_qualia': all_qualia if all_qualia else None,
                # For retrocausal leak detection
                'forward_state': forward_state,
                'backward_state': backward_state
            }

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        target_qualia: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate tokens with optional qualia control.

        Args:
            input_ids: [batch, seq] initial tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            target_qualia: Dict of target qualia values to bias generation

        Returns:
            generated_ids: [batch, seq + max_new_tokens]
            generation_info: Dict with qualia trajectory, etc.
        """
        self.eval()

        generated = input_ids.clone()
        qualia_trajectory = []

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(generated)
            logits = outputs['logits'][:, -1, :]  # Last position

            # Apply temperature
            logits = logits / temperature

            # Apply top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=-1)

            # Track qualia
            if outputs['qualia_info'] is not None:
                qualia_trajectory.append({
                    k: v[:, -1].cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in outputs['qualia_info'].items()
                })

        generation_info = {
            'qualia_trajectory': qualia_trajectory,
            'final_qualia': qualia_trajectory[-1] if qualia_trajectory else None
        }

        return generated, generation_info

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        config: Optional[QLLMConfig] = None
    ) -> "QLLM":
        """
        Load from a pretrained model and add QLLM layers.

        This is for fine-tuning an existing model with quantum paradigm enhancements.
        """
        if config is None:
            config = QLLMConfig.from_base_model(model_name)

        model = cls(config)

        # TODO: Load pretrained weights and merge
        # This would involve loading LLaMA/etc weights and mapping them
        # to our architecture

        return model

    def get_paradigm_summary(self) -> Dict[str, Any]:
        """Get summary of active paradigms and their configurations"""
        return {
            'semantic_phase': {
                'enabled': self.config.use_semantic_phase,
                'phase_dim': self.config.semantic_phase_dim,
                'modulation_strength': self.config.phase_modulation_strength
            },
            'retrocausal': {
                'enabled': self.config.use_retrocausal_attention,
                'layers': self.config.retrocausal_layers,
                'backward_weight': self.config.backward_attention_weight
            },
            'lindblad': {
                'enabled': self.config.use_lindblad_layers,
                'num_operators': self.config.num_lindblad_operators,
                'dissipation_strength': self.config.dissipation_strength
            },
            'qualia': {
                'enabled': self.config.use_qualia_output,
                'num_channels': self.config.num_qualia_channels,
                'names': self.config.qualia_names
            },
            'emergent': {
                'enabled': self.config.use_emergent_init,
                'complexity_measure': self.config.complexity_measure
            }
        }
