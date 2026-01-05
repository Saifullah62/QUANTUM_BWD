"""
QLLM Novel Layers
=================

Quantum-paradigm-inspired neural network layers.
"""

from .semantic_phase import SemanticPhaseEmbedding, PhaseModulator, SemanticInterference
from .retrocausal import (
    RetrocausalAttention,
    TwoStateVectorAttention,
    RetrocausalBlock,
    HindsightEnhancedBlock,
    FuturePlanner,
    TwoPassTrainer
)
from .lindblad import (
    LindbladLayer,
    LearnableNoiseInjector,
    ContractiveStabilizer,
    DissipativeNormalization,
    EntropyGate,
    LipschitzRegularizer
)
from .qualia import QualiaOutputHead, QualiaEncoder, QUALIA_CHANNELS
from .emergent import (
    EmergentInitializer,
    ComplexityMeasure,
    AttractorLayer,
    ComplexityTimeLayer,
    InformationalFlowLayer
)

__all__ = [
    # Semantic Phase
    "SemanticPhaseEmbedding",
    "PhaseModulator",
    "SemanticInterference",
    # Retrocausal
    "RetrocausalAttention",
    "TwoStateVectorAttention",
    "RetrocausalBlock",
    "HindsightEnhancedBlock",
    "FuturePlanner",
    "TwoPassTrainer",
    # Lindblad
    "LindbladLayer",
    "LearnableNoiseInjector",
    "ContractiveStabilizer",
    "DissipativeNormalization",
    "EntropyGate",
    "LipschitzRegularizer",
    # Qualia
    "QualiaOutputHead",
    "QualiaEncoder",
    "QUALIA_CHANNELS",
    # Emergent
    "EmergentInitializer",
    "ComplexityMeasure",
    "AttractorLayer",
    "ComplexityTimeLayer",
    "InformationalFlowLayer"
]
