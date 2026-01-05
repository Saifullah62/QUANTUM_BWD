"""
Sentinel Attention Audits
=========================

Fixed set of killer pairs for monitoring attention patterns across training.
Logs top attended tokens at checkpoints to verify SenseHead is using context.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class SentinelPair:
    """A killer pair to monitor throughout training."""
    bundle_id: str
    anchor_sense: str
    hard_neg_sense: str
    anchor_context: str
    hard_neg_context: str
    baseline_gap: float


class SentinelSet:
    """Fixed set of killer pairs for attention monitoring."""

    def __init__(self, pairs: List[SentinelPair], log_dir: Path = None):
        self.pairs = pairs
        self.log_dir = log_dir or Path("checkpoints/sentinel_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.attention_log = []

    @classmethod
    def from_killer_dossier(
        cls,
        killer_path: str,
        bundle_path: str,
        n: int = 12,
        prefer_inversions: bool = True
    ) -> "SentinelSet":
        """Load sentinel pairs from killer negative log."""
        killers = []
        with open(killer_path) as f:
            for line in f:
                if line.strip():
                    killers.append(json.loads(line))

        if prefer_inversions:
            killers.sort(key=lambda k: k.get("gap", 0))

        bundles = {}
        with open(bundle_path) as f:
            for line in f:
                b = json.loads(line)
                bundles[b["bundle_id"]] = b

        pairs = []
        seen_senses = set()

        for k in killers:
            sense_key = (k["anchor_sense"], k["hard_neg_sense"])
            if sense_key in seen_senses:
                continue
            seen_senses.add(sense_key)

            bundle = bundles.get(k["bundle_id"])
            if not bundle:
                continue

            anchor_ctx = ""
            hn_ctx = ""
            for item in bundle["items"]:
                if item["item_id"] == k["anchor_id"]:
                    anchor_ctx = item.get("context", "")
                elif item["item_id"] == k["hard_neg_id"]:
                    hn_ctx = item.get("context", "")

            pairs.append(SentinelPair(
                bundle_id=k["bundle_id"],
                anchor_sense=k["anchor_sense"],
                hard_neg_sense=k["hard_neg_sense"],
                anchor_context=anchor_ctx,
                hard_neg_context=hn_ctx,
                baseline_gap=k["gap"]
            ))

            if len(pairs) >= n:
                break

        return cls(pairs)

    def get_top_attended(self, weights: torch.Tensor, tokens: List[str], k: int) -> List[Dict]:
        """Get top-k attended tokens with weights."""
        w = weights.detach().cpu()
        top_idx = w.argsort(descending=True)[:k].tolist()
        return [
            {"token": tokens[i] if i < len(tokens) else "<unk>", "weight": float(w[i]), "pos": i}
            for i in top_idx
        ]

    def compute_entropy(self, weights: torch.Tensor) -> float:
        """Compute attention entropy."""
        w = weights.detach().cpu()
        w = w + 1e-10
        entropy = -(w * w.log()).sum()
        return float(entropy)

    def compute_overlap(self, w1: torch.Tensor, w2: torch.Tensor) -> float:
        """Compute attention overlap (0-1, lower = more different)."""
        w1 = w1.detach().cpu()
        w2 = w2.detach().cpu()
        min_len = min(len(w1), len(w2))
        overlap = torch.minimum(w1[:min_len], w2[:min_len]).sum()
        return float(overlap)

    def log_attention(
        self,
        sense_head,
        model,
        tokenizer,
        step: int,
        device: str = "cuda"
    ):
        """Log attention patterns for all sentinel pairs."""
        if not self.pairs:
            return

        entries = []

        for pair in self.pairs:
            try:
                anchor_tokens = tokenizer(
                    pair.anchor_context,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True
                ).to(device)

                hn_tokens = tokenizer(
                    pair.hard_neg_context,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True
                ).to(device)

                with torch.no_grad():
                    anchor_out = model(
                        input_ids=anchor_tokens["input_ids"],
                        attention_mask=anchor_tokens["attention_mask"]
                    )
                    hn_out = model(
                        input_ids=hn_tokens["input_ids"],
                        attention_mask=hn_tokens["attention_mask"]
                    )

                    anchor_states = anchor_out.get("hidden_states")
                    hn_states = hn_out.get("hidden_states")

                    if anchor_states is None or hn_states is None:
                        continue

                    _, anchor_weights = sense_head(anchor_states, anchor_tokens["attention_mask"])
                    _, hn_weights = sense_head(hn_states, hn_tokens["attention_mask"])

                anchor_strs = tokenizer.convert_ids_to_tokens(anchor_tokens["input_ids"][0].tolist())
                hn_strs = tokenizer.convert_ids_to_tokens(hn_tokens["input_ids"][0].tolist())

                anchor_top = self.get_top_attended(anchor_weights[0], anchor_strs, 5)
                hn_top = self.get_top_attended(hn_weights[0], hn_strs, 5)
                overlap = self.compute_overlap(anchor_weights[0], hn_weights[0])

                entry = {
                    "step": step,
                    "anchor_sense": pair.anchor_sense,
                    "hard_neg_sense": pair.hard_neg_sense,
                    "baseline_gap": pair.baseline_gap,
                    "anchor_top_tokens": anchor_top,
                    "hard_neg_top_tokens": hn_top,
                    "attention_overlap": overlap,
                    "anchor_entropy": self.compute_entropy(anchor_weights[0]),
                    "hard_neg_entropy": self.compute_entropy(hn_weights[0])
                }
                entries.append(entry)
            except Exception as e:
                print(f"Error processing sentinel pair: {e}")
                continue

        self.attention_log.extend(entries)

        log_path = self.log_dir / f"sentinel_step_{step}.json"
        with open(log_path, "w") as f:
            json.dump(entries, f, indent=2)

        print(f"\n=== Sentinel Attention @ Step {step} ===")
        for e in entries[:3]:
            print(f"  {e['anchor_sense']} vs {e['hard_neg_sense']}")
            anchor_tokens = [t['token'] for t in e['anchor_top_tokens'][:3]]
            hn_tokens = [t['token'] for t in e['hard_neg_top_tokens'][:3]]
            print(f"    Anchor top: {anchor_tokens}")
            print(f"    HardNeg top: {hn_tokens}")
            print(f"    Overlap: {e['attention_overlap']:.3f}")


def create_sentinel_from_run(
    killer_path: str = "checkpoints/validation_seed_42/killer_negatives.jsonl",
    bundle_path: str = "data/dress_rehearsal_bundles.jsonl",
    n: int = 12
) -> SentinelSet:
    """Create sentinel set from latest run."""
    return SentinelSet.from_killer_dossier(killer_path, bundle_path, n=n)
