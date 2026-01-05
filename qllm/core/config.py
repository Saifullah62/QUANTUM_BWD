"""
QLLM Configuration
==================

Configuration classes for the Quantum-Inspired LLM architecture.
Maps Daugherty's paradigms to model hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
from enum import Enum


class ParadigmMode(Enum):
    """Which quantum paradigm to emphasize"""
    SEMANTIC_PHASE = "semantic_phase"       # Focus on phase-encoded meaning
    RETROCAUSAL = "retrocausal"             # Future-influenced attention
    LINDBLAD = "lindblad"                   # Dissipative dynamics
    EMERGENT = "emergent"                   # Emergent computation
    FULL = "full"                           # All paradigms active


@dataclass
class QLLMConfig:
    """
    Configuration for Quantum-Inspired LLM

    Paradigm Mappings:
    - semantic_phase_dim: Dimensionality of imaginary (phase) component
    - retrocausal_depth: How many layers use bidirectional attention
    - lindblad_operators: Number of Lindblad jump operators per layer
    - qualia_channels: Number of qualic state dimensions
    """

    # Base model configuration
    vocab_size: int = 32000
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    intermediate_dim: int = 11008
    max_seq_length: int = 4096

    # === SEMANTIC PHASE PARADIGM ===
    # Complex-valued embeddings where imaginary part encodes context/phase
    use_semantic_phase: bool = True
    semantic_phase_dim: int = 4096  # Same as hidden_dim for full complex
    phase_modulation_strength: float = 0.3  # How much context affects phase
    context_window_for_phase: int = 128  # Tokens used to compute context vector

    # === RETROCAUSAL PARADIGM ===
    # Bidirectional state vectors inspired by Two-State Vector Formalism
    use_retrocausal_attention: bool = True
    retrocausal_layers: List[int] = field(default_factory=lambda: [0, 8, 16, 24, 31])
    backward_attention_weight: float = 0.3  # Weight of backward state
    interference_type: Literal["additive", "multiplicative", "phase"] = "phase"

    # === LINDBLAD PARADIGM ===
    # Dissipative layers that use "noise" as computational resource
    use_lindblad_layers: bool = True
    lindblad_every_n_layers: int = 4  # Insert Lindblad layer every N transformer layers
    num_lindblad_operators: int = 4  # Number of jump operators L_k
    lindblad_dt: float = 0.1  # Time step for evolution
    dissipation_strength: float = 0.1  # Strength of dissipative term

    # === QUALIA PARADIGM ===
    # Qualitative state channels for emotional/contextual valence
    use_qualia_output: bool = True
    num_qualia_channels: int = 8  # Number of qualic observables
    qualia_names: List[str] = field(default_factory=lambda: [
        "valence",      # Positive/negative
        "arousal",      # Intensity
        "certainty",    # Epistemic confidence
        "novelty",      # Information surprise
        "coherence",    # Semantic consistency
        "agency",       # Active/passive
        "temporality",  # Past/present/future orientation
        "abstraction"   # Concrete/abstract
    ])
    qualia_modulation_strength: float = 0.1

    # === EMERGENT COMPUTATION PARADIGM ===
    # Weights as "frozen flows" - stable attractors
    use_emergent_init: bool = True
    emergent_flow_iterations: int = 100
    complexity_measure: Literal["entropy", "fisher", "curvature"] = "entropy"
    attractor_threshold: float = 1e-4

    # === TOPOLOGICAL PARADIGM ===
    # Future: Braiding operations, spin networks
    use_topological_encoding: bool = False  # Not yet implemented

    # Training configuration
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    # Device configuration
    dtype: str = "bfloat16"
    device: str = "cuda"

    def get_paradigm_mode(self) -> ParadigmMode:
        """Determine which paradigm mode is active based on config"""
        active = []
        if self.use_semantic_phase:
            active.append("semantic")
        if self.use_retrocausal_attention:
            active.append("retrocausal")
        if self.use_lindblad_layers:
            active.append("lindblad")
        if self.use_qualia_output:
            active.append("qualia")

        if len(active) >= 3:
            return ParadigmMode.FULL
        elif "semantic" in active:
            return ParadigmMode.SEMANTIC_PHASE
        elif "retrocausal" in active:
            return ParadigmMode.RETROCAUSAL
        elif "lindblad" in active:
            return ParadigmMode.LINDBLAD
        else:
            return ParadigmMode.EMERGENT

    @classmethod
    def from_base_model(cls, model_name: str) -> "QLLMConfig":
        """Create config matching a base model's dimensions"""

        configs = {
            "llama3.1:8b": cls(
                vocab_size=128256,
                hidden_dim=4096,
                num_layers=32,
                num_heads=32,
                intermediate_dim=14336,
                max_seq_length=8192
            ),
            "llama3.2:3b": cls(
                vocab_size=128256,
                hidden_dim=3072,
                num_layers=28,
                num_heads=24,
                intermediate_dim=8192,
                max_seq_length=8192
            ),
            "phi3:mini": cls(
                vocab_size=32064,
                hidden_dim=3072,
                num_layers=32,
                num_heads=32,
                intermediate_dim=8192,
                max_seq_length=4096
            ),
            "mixtral:8x7b": cls(
                vocab_size=32000,
                hidden_dim=4096,
                num_layers=32,
                num_heads=32,
                intermediate_dim=14336,
                max_seq_length=32768
            )
        }

        return configs.get(model_name, cls())

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            k: v if not isinstance(v, Enum) else v.value
            for k, v in self.__dict__.items()
        }

    @classmethod
    def minimal(cls) -> "QLLMConfig":
        """Minimal config for testing"""
        return cls(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            intermediate_dim=512,
            max_seq_length=512,
            num_lindblad_operators=2,
            num_qualia_channels=4
        )


@dataclass
class TrainingConfig:
    """Training configuration for QLLM"""

    # Basic training
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantum paradigm specific training
    phase_consistency_loss_weight: float = 0.1  # Encourage phase coherence
    lindblad_stability_loss_weight: float = 0.05  # Encourage stable attractors
    qualia_diversity_loss_weight: float = 0.05  # Encourage varied qualia
    retrocausal_coherence_weight: float = 0.1  # Forward-backward consistency

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100

    # Cluster configuration
    nodes: List[str] = field(default_factory=lambda: ["gpu-ramp"])
    distributed: bool = False
