"""
QLLM - Quantum-Inspired Large Language Model
=============================================

A novel LLM architecture based on Bryan W. Daugherty's
"Unconventional Quantum Paradigms" paper.

Core innovations:
1. Semantic Phase Embeddings - Complex-valued embeddings encoding meaning + context
2. Retrocausal Attention - Bidirectional state vectors (TSVF-inspired)
3. Lindblad Dissipative Layers - Noise as computational resource
4. Qualia-Modulated Output - Phase-coherent token selection
5. Emergent Weight Discovery - Weights as "frozen flows"

Author: Quantum Paradigms Research Team
Based on: "Unconventional Quantum Paradigms" by Bryan W. Daugherty
"""

__version__ = "0.1.0"
__author__ = "Quantum Paradigms Research Team"

from .core.config import QLLMConfig
from .core.model import QLLM
from .layers import (
    SemanticPhaseEmbedding,
    RetrocausalAttention,
    LindbladLayer,
    QualiaOutputHead
)

__all__ = [
    "QLLM",
    "QLLMConfig",
    "SemanticPhaseEmbedding",
    "RetrocausalAttention",
    "LindbladLayer",
    "QualiaOutputHead"
]
