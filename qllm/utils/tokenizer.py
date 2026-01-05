"""
Quantum-Aware Tokenizer
=======================

Tokenizer with paradigm-specific enhancements:
- Phase markers for semantic context
- Retrocausal delimiters for temporal reasoning
- Qualia tokens for subjective experience encoding
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QuantumToken:
    """Token with quantum paradigm metadata"""
    token_id: int
    text: str
    paradigm_hints: Dict[str, float]  # Paradigm -> relevance score
    phase_marker: Optional[str] = None
    qualia_channel: Optional[int] = None


class QuantumTokenizer:
    """
    Tokenizer wrapper that adds quantum paradigm awareness.

    Wraps any base tokenizer (e.g., from transformers) and adds:
    1. Special tokens for paradigm markers
    2. Phase context encoding
    3. Qualia dimension hints
    """

    # Special tokens for paradigm markers
    PARADIGM_TOKENS = {
        # Semantic phase markers
        '<PHASE_START>': 'Begin semantic superposition',
        '<PHASE_COLLAPSE>': 'Context resolves ambiguity',
        '<PHASE_END>': 'End semantic superposition',

        # Retrocausal markers
        '<RETRO_FUTURE>': 'Future outcome known',
        '<RETRO_TRACE>': 'Tracing backward',
        '<RETRO_CAUSE>': 'Root cause identified',

        # Lindblad/dissipation markers
        '<NOISE_START>': 'Begin noisy/chaotic input',
        '<STABLE_ATTRACTOR>': 'Stable pattern found',
        '<NOISE_END>': 'End noisy section',

        # Qualia markers
        '<QUALIA>': 'Subjective experience follows',
        '<VALENCE_POS>': 'Positive valence',
        '<VALENCE_NEG>': 'Negative valence',
        '<AROUSAL_HIGH>': 'High arousal/intensity',
        '<AROUSAL_LOW>': 'Low arousal/calm',

        # Emergent markers
        '<EMERGE_PARTS>': 'Component elements',
        '<EMERGE_WHOLE>': 'Emergent whole'
    }

    # Patterns that suggest paradigm relevance
    PARADIGM_PATTERNS = {
        'semantic_phase': [
            r'\b(ambiguous|context|meaning|interpret|polysem|homonym)\b',
            r'\b(depends on|could mean|in this case|alternatively)\b',
            r'\b(sense|definition|usage|connotation)\b'
        ],
        'retrocausal': [
            r'\b(because|therefore|led to|resulted in|caused by)\b',
            r'\b(if we want|to achieve|working backward|hindsight)\b',
            r'\b(outcome|result|consequence|preceded by)\b'
        ],
        'lindblad': [
            r'\b(noisy|chaotic|conflicting|unclear|messy)\b',
            r'\b(stable|pattern|signal|coherent|organized)\b',
            r'\b(filter|extract|synthesize|distill)\b'
        ],
        'qualia': [
            r'\b(feels like|experience|subjective|qualitative)\b',
            r'\b(sensation|emotion|perception|aware)\b',
            r'\b(vivid|intense|subtle|overwhelming)\b'
        ],
        'emergent': [
            r'\b(emerge|arise|develop|evolve|complex)\b',
            r'\b(system|network|interact|collective)\b',
            r'\b(more than sum|whole|pattern|structure)\b'
        ]
    }

    def __init__(self, base_tokenizer: Any):
        """
        Initialize with a base tokenizer.

        Args:
            base_tokenizer: Any tokenizer with encode/decode methods
                           (e.g., transformers AutoTokenizer)
        """
        self.base = base_tokenizer
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Add paradigm-specific special tokens to base tokenizer"""
        if hasattr(self.base, 'add_special_tokens'):
            self.base.add_special_tokens({
                'additional_special_tokens': list(self.PARADIGM_TOKENS.keys())
            })

    def analyze_paradigm_relevance(self, text: str) -> Dict[str, float]:
        """
        Analyze text for paradigm relevance scores.

        Returns dict mapping paradigm -> relevance score (0-1)
        """
        scores = {}
        text_lower = text.lower()

        for paradigm, patterns in self.PARADIGM_PATTERNS.items():
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(pattern, text_lower, re.IGNORECASE))
            # Normalize by text length
            scores[paradigm] = min(1.0, matches / max(1, len(text.split()) / 10))

        return scores

    def encode_with_paradigm_markers(
        self,
        text: str,
        add_markers: bool = True
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Encode text with paradigm analysis.

        Returns:
            token_ids: List of token IDs
            metadata: Dict with paradigm relevance and marker positions
        """
        # Analyze paradigm relevance
        relevance = self.analyze_paradigm_relevance(text)

        # Find dominant paradigm
        dominant = max(relevance, key=relevance.get) if relevance else None

        # Optionally add paradigm markers
        if add_markers and dominant and relevance[dominant] > 0.3:
            marker_map = {
                'semantic_phase': ('<PHASE_START>', '<PHASE_END>'),
                'retrocausal': ('<RETRO_FUTURE>', '<RETRO_CAUSE>'),
                'lindblad': ('<NOISE_START>', '<STABLE_ATTRACTOR>'),
                'qualia': ('<QUALIA>', ''),
                'emergent': ('<EMERGE_PARTS>', '<EMERGE_WHOLE>')
            }
            start, end = marker_map.get(dominant, ('', ''))
            text = f"{start} {text} {end}".strip()

        # Encode with base tokenizer
        token_ids = self.base.encode(text)

        metadata = {
            'paradigm_relevance': relevance,
            'dominant_paradigm': dominant,
            'markers_added': add_markers and dominant is not None
        }

        return token_ids, metadata

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        padding: str = 'max_length',
        truncation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Standard encode interface compatible with transformers.

        Passes through to base tokenizer.
        """
        return self.base(
            text,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs
        )

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text"""
        return self.base.decode(token_ids, **kwargs)

    def create_paradigm_prompt(
        self,
        paradigm: str,
        content: str
    ) -> str:
        """
        Create a prompt formatted for a specific paradigm.

        Args:
            paradigm: One of semantic_phase, retrocausal, lindblad, qualia, emergent
            content: The main content/question

        Returns:
            Formatted prompt string
        """
        templates = {
            'semantic_phase': (
                "<PHASE_START>\n"
                "Consider the multiple possible meanings:\n"
                "{content}\n"
                "<PHASE_COLLAPSE>\n"
                "Given the context, the most coherent interpretation is:\n"
            ),
            'retrocausal': (
                "<RETRO_FUTURE>\n"
                "Known outcome: {content}\n"
                "<RETRO_TRACE>\n"
                "Tracing backward to identify causes:\n"
            ),
            'lindblad': (
                "<NOISE_START>\n"
                "Chaotic/noisy input: {content}\n"
                "<STABLE_ATTRACTOR>\n"
                "The stable, coherent interpretation:\n"
            ),
            'qualia': (
                "<QUALIA>\n"
                "Experience: {content}\n"
                "Subjective, qualitative description:\n"
            ),
            'emergent': (
                "<EMERGE_PARTS>\n"
                "Components: {content}\n"
                "<EMERGE_WHOLE>\n"
                "Emergent properties that arise:\n"
            )
        }

        template = templates.get(paradigm, "{content}")
        return template.format(content=content)

    def extract_qualia_hints(self, text: str) -> Dict[str, float]:
        """
        Extract qualia dimension hints from text.

        Returns scores for 8 qualia channels:
        - valence: positive/negative
        - arousal: intensity
        - certainty: confidence
        - novelty: new/familiar
        - coherence: organized/chaotic
        - agency: active/passive
        - temporality: past/present/future
        - abstraction: concrete/abstract
        """
        hints = {
            'valence': 0.5,
            'arousal': 0.5,
            'certainty': 0.5,
            'novelty': 0.5,
            'coherence': 0.5,
            'agency': 0.5,
            'temporality': 0.5,
            'abstraction': 0.5
        }

        text_lower = text.lower()

        # Valence detection
        pos_words = len(re.findall(r'\b(good|happy|positive|success|joy|love|great)\b', text_lower))
        neg_words = len(re.findall(r'\b(bad|sad|negative|fail|anger|hate|terrible)\b', text_lower))
        if pos_words + neg_words > 0:
            hints['valence'] = pos_words / (pos_words + neg_words)

        # Arousal detection
        high_arousal = len(re.findall(r'\b(exciting|intense|urgent|extreme|overwhelming)\b', text_lower))
        low_arousal = len(re.findall(r'\b(calm|peaceful|quiet|relaxed|gentle)\b', text_lower))
        if high_arousal + low_arousal > 0:
            hints['arousal'] = high_arousal / (high_arousal + low_arousal)

        # Certainty detection
        certain = len(re.findall(r'\b(definitely|certainly|sure|know|proven)\b', text_lower))
        uncertain = len(re.findall(r'\b(maybe|perhaps|might|unclear|uncertain)\b', text_lower))
        if certain + uncertain > 0:
            hints['certainty'] = certain / (certain + uncertain)

        # Abstraction detection
        concrete = len(re.findall(r'\b(specific|example|instance|particular|tangible)\b', text_lower))
        abstract = len(re.findall(r'\b(concept|theory|general|abstract|principle)\b', text_lower))
        if concrete + abstract > 0:
            hints['abstraction'] = abstract / (concrete + abstract)

        return hints

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size including special tokens"""
        return len(self.base)

    @property
    def pad_token_id(self) -> int:
        """Get pad token ID"""
        return self.base.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID"""
        return self.base.eos_token_id


def create_tokenizer(model_name: str = "llama3.1:8b") -> QuantumTokenizer:
    """
    Create a QuantumTokenizer for a given model.

    Attempts to load the appropriate base tokenizer.
    """
    try:
        from transformers import AutoTokenizer
        base = AutoTokenizer.from_pretrained(model_name)
    except:
        # Fallback to a simple tokenizer for testing
        from transformers import GPT2Tokenizer
        base = GPT2Tokenizer.from_pretrained('gpt2')

    return QuantumTokenizer(base)
