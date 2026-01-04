"""
Embedding Scorer for Multi-Sense Retrieval Evaluation
======================================================

Replaces the placeholder word-overlap scorer with real embeddings.
Uses either:
1. SenseHead model (if checkpoint available) - context-aware attentive pooling
2. Base encoder only (fallback) - mean pooling

The scorer computes cosine similarity between query and candidate embeddings.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import json

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available. Using fallback word-overlap scorer.")

# Try to import SenseHead
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from qllm.layers.sense_head import SenseHead
    HAS_SENSEHEAD = True
except ImportError:
    HAS_SENSEHEAD = False


@dataclass
class ScorerConfig:
    """Configuration for embedding scorer."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sensehead_checkpoint: Optional[str] = None
    proj_dim: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_length: int = 128
    use_sensehead: bool = True  # If False, use mean pooling


class EmbeddingScorer:
    """
    Real embedding-based scorer for retrieval evaluation.

    Can use either:
    - SenseHead (attentive pooling) for context-aware embeddings
    - Mean pooling as fallback
    """

    def __init__(self, config: ScorerConfig = None):
        self.config = config or ScorerConfig()
        self.device = self.config.device
        self.model = None
        self.tokenizer = None
        self.sensehead = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of models."""
        if self._initialized:
            return

        if not HAS_TRANSFORMERS:
            print("Transformers not available - using word overlap fallback")
            self._initialized = True
            return

        print(f"Loading encoder: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load SenseHead if checkpoint exists
        if self.config.use_sensehead and self.config.sensehead_checkpoint and HAS_SENSEHEAD:
            checkpoint_path = Path(self.config.sensehead_checkpoint)
            if checkpoint_path.exists():
                print(f"Loading SenseHead from: {checkpoint_path}")
                hidden_dim = self.model.config.hidden_size
                self.sensehead = SenseHead(
                    hidden_dim=hidden_dim,
                    proj_dim=self.config.proj_dim
                )
                state = torch.load(checkpoint_path, map_location=self.device)
                if 'sensehead' in state:
                    self.sensehead.load_state_dict(state['sensehead'])
                else:
                    self.sensehead.load_state_dict(state)
                self.sensehead = self.sensehead.to(self.device)
                self.sensehead.eval()
                print("SenseHead loaded successfully")
            else:
                print(f"SenseHead checkpoint not found: {checkpoint_path}")

        self._initialized = True

    def _word_overlap_similarity(self, query: str, passage: str) -> float:
        """Fallback word overlap similarity."""
        q_words = set(query.lower().split())
        p_words = set(passage.lower().split())
        if not q_words or not p_words:
            return 0.0
        intersection = len(q_words & p_words)
        union = len(q_words | p_words)
        return intersection / union if union > 0 else 0.0

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts to embeddings."""
        if not HAS_TRANSFORMERS or self.model is None:
            # Fallback: return None, will use word overlap
            return None

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            token_states = outputs.last_hidden_state  # [B, T, H]
            attention_mask = inputs['attention_mask']  # [B, T]

            if self.sensehead is not None:
                # Use attentive pooling
                embeddings, _ = self.sensehead(token_states, attention_mask)
            else:
                # Mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = (token_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                embeddings = sum_embeddings / sum_mask
                embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def score_item(self, item: Dict) -> Tuple[List[str], Dict[str, float]]:
        """
        Score a retrieval eval item.

        Args:
            item: Eval item with 'query' and 'candidates'

        Returns:
            ranked_ids: List of passage_ids ordered by score (descending)
            score_map: Dict mapping passage_id to score
        """
        self.initialize()

        query_text = item['query']['text']
        candidates = item['candidates']

        if not HAS_TRANSFORMERS or self.model is None:
            # Fallback to word overlap
            scores = []
            for cand in candidates:
                sim = self._word_overlap_similarity(query_text, cand['text'])
                scores.append((cand['passage_id'], sim))
            scores.sort(key=lambda x: -x[1])
            ranked_ids = [s[0] for s in scores]
            score_map = {s[0]: s[1] for s in scores}
            return ranked_ids, score_map

        # Encode query
        query_emb = self._encode_batch([query_text])

        # Encode candidates in batches
        candidate_texts = [c['text'] for c in candidates]
        candidate_ids = [c['passage_id'] for c in candidates]

        all_scores = []
        for i in range(0, len(candidate_texts), self.config.batch_size):
            batch_texts = candidate_texts[i:i + self.config.batch_size]
            batch_embs = self._encode_batch(batch_texts)

            # Cosine similarity (embeddings are already normalized)
            similarities = (query_emb @ batch_embs.T).squeeze(0)  # [batch_size]
            all_scores.extend(similarities.cpu().tolist())

        # Build score map and ranking
        score_map = {cid: score for cid, score in zip(candidate_ids, all_scores)}
        ranked = sorted(score_map.items(), key=lambda x: -x[1])
        ranked_ids = [r[0] for r in ranked]

        return ranked_ids, score_map

    def similarity(self, query: str, passage: str) -> float:
        """Compute similarity between query and passage."""
        self.initialize()

        if not HAS_TRANSFORMERS or self.model is None:
            return self._word_overlap_similarity(query, passage)

        query_emb = self._encode_batch([query])
        passage_emb = self._encode_batch([passage])

        similarity = (query_emb @ passage_emb.T).squeeze().item()
        return similarity


def create_similarity_fn(config: ScorerConfig = None) -> Callable[[str, str], float]:
    """
    Factory function to create a similarity function for the eval harness.

    Usage:
        from paradigm_factory.v2.embedding_scorer import create_similarity_fn, ScorerConfig

        config = ScorerConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            sensehead_checkpoint="checkpoints/sensehead_best.pt"
        )
        sim_fn = create_similarity_fn(config)

        # In eval harness:
        scorer = RetrievalScorer(similarity_fn=sim_fn)
    """
    scorer = EmbeddingScorer(config)
    return scorer.similarity


# Convenience function for quick testing
def test_scorer():
    """Quick test of the embedding scorer."""
    config = ScorerConfig(use_sensehead=False)  # Use mean pooling for test
    scorer = EmbeddingScorer(config)

    # Test similarity
    q = "The bank approved the loan application."
    p1 = "Financial institutions provide various lending services."
    p2 = "The river bank was covered with wildflowers."

    sim1 = scorer.similarity(q, p1)
    sim2 = scorer.similarity(q, p2)

    print(f"Query: {q}")
    print(f"Financial passage similarity: {sim1:.4f}")
    print(f"River passage similarity: {sim2:.4f}")
    print(f"Margin: {sim1 - sim2:.4f}")

    return sim1 > sim2  # Should be True


if __name__ == "__main__":
    success = test_scorer()
    print(f"\nTest {'passed' if success else 'FAILED'}")
