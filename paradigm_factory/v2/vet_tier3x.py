#!/usr/bin/env python3
"""
vet_tier3x.py
=============
Quality-filter tier3x expanded bundles by scoring them with a reference capsule
(e.g., D1 baseline) using the SAME margin logic as eval: margin = pos_sim - max(neg_sim).

Goal:
- Expanded tier3x passed a danger threshold, but many are "pseudo-tier3" that do not generalize.
- Keep only expanded items that *behave like real killers* under a trusted model.

Inputs:
- --bundles: JSONL of contrastive bundles (e.g., contrastive_bundles_v23_tier3x.jsonl)
- --capsule: path to capsule dir containing model.pt + config.json (reference scorer model)
- --out: output JSONL (recommended: contrastive_bundles_v23_tier3x_vetted.jsonl)

Filtering:
- Keep all non-expanded bundles as-is.
- For expanded tier3x bundles:
    - Score margin (pos_sim - max_neg_sim)
    - Keep top-K by margin (keep_frac) AND optionally require margin > min_margin

Notes:
- This assumes your model produces embeddings for text and is stored as a torch state_dict.
- It instantiates qllm.layers.semantic_phase.SemanticPhaseEmbedding using embed_dim from config.json.
- If your repo uses a different model wrapper, adapt `load_model()` and `encode_texts()` only.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from qllm.layers.semantic_phase import SemanticPhaseEmbedding
except ImportError:
    SemanticPhaseEmbedding = None


class SemanticPhaseEmbeddingInline(nn.Module):
    """Inline fallback for SemanticPhaseEmbedding."""

    def __init__(self, vocab_size: int = 30000, embedding_dim: int = 256,
                 phase_dim: int = 64, max_seq_len: int = 2048, padding_idx: int = 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.phase_dim = phase_dim

        self.register_buffer('rotation_freqs', torch.randn(phase_dim * 2))
        self.register_buffer('positional_phases', torch.randn(max_seq_len, phase_dim * 2))

        self.real_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.imag_embedding = nn.Embedding(vocab_size, phase_dim, padding_idx=padding_idx)

        self.context_to_rotation = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, phase_dim * 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real = self.real_embedding(x)
        imag = self.imag_embedding(x)
        return real, imag


class SemanticPhaseModel(nn.Module):
    """Model combining text encoder with SemanticPhase embedding."""

    def __init__(self, vocab_size: int = 30000, embed_dim: int = 256, phase_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.phase_dim = phase_dim

        EmbeddingClass = SemanticPhaseEmbedding if SemanticPhaseEmbedding else SemanticPhaseEmbeddingInline
        self.semantic_phase = EmbeddingClass(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            phase_dim=phase_dim,
            padding_idx=0
        )

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.vocab = {chr(i): i for i in range(256)}

    def tokenize(self, text: str, max_len: int = 128) -> torch.Tensor:
        tokens = [self.vocab.get(c, 1) for c in text[:max_len]]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        return torch.tensor(tokens[:max_len])

    def forward(self, texts: List[str]) -> torch.Tensor:
        tokens = torch.stack([self.tokenize(t) for t in texts])
        device = next(self.parameters()).device
        tokens = tokens.to(device)

        real_emb, imag_emb = self.semantic_phase(tokens)
        phase_emb = torch.sqrt(real_emb**2 + imag_emb**2 + 1e-8)

        mask = (tokens != 0).float().unsqueeze(-1).to(device)
        pooled = (phase_emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        return self.proj(pooled)


def _get_text(x: Any) -> str:
    """Handle either a dict with {'text': ...} or a raw string."""
    if isinstance(x, dict):
        return x.get("text", "") or ""
    if isinstance(x, str):
        return x
    return ""


def _is_expanded_tier3(bundle: Dict[str, Any]) -> bool:
    md = (bundle.get("metadata") or {})
    tier = md.get("difficulty_tier")
    if tier != "tier3_adversarial":
        return False
    return (md.get("expansion") == "tier3x") or (md.get("source") == "tier3_expansion")


def load_capsule_config(capsule_dir: Path) -> Dict[str, Any]:
    cfg_path = capsule_dir / "config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def load_model(capsule_dir: Path, device: str) -> SemanticPhaseModel:
    """Load trained model from capsule directory."""
    model_path = capsule_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pt not found in capsule: {capsule_dir}")

    state_dict = torch.load(model_path, map_location="cpu")

    # Infer model dimensions from state dict
    embed_dim = state_dict.get("proj.0.weight", state_dict.get("proj.0.bias", torch.zeros(256))).shape[0]
    phase_dim = 64  # Default

    # Create model with matching dimensions
    model = SemanticPhaseModel(
        vocab_size=30000,
        embed_dim=embed_dim,
        phase_dim=phase_dim
    )

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


@torch.no_grad()
def encode_texts(model: SemanticPhaseModel, texts: List[str], device: str, batch_size: int = 64) -> torch.Tensor:
    """Encodes texts into embeddings using SemanticPhaseModel."""
    embs: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out = model(batch)
        embs.append(out.detach().to("cpu"))
    return torch.cat(embs, dim=0)


def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.T


def score_bundle_margin(model: torch.nn.Module, bundle: Dict[str, Any], device: str) -> Tuple[float, int]:
    """
    Returns (margin, rank) where:
      margin = pos_sim - max_neg_sim
      rank = 1 + number of negatives with sim > pos_sim (1 is best)
    """
    anchor = _get_text(bundle.get("anchor"))
    pos = _get_text(bundle.get("positive"))

    # Handle negatives structure (can be dict with within_lemma/cross_lemma or list)
    negs_raw = bundle.get("negatives") or []
    if isinstance(negs_raw, dict):
        neg_texts = []
        for neg in negs_raw.get("within_lemma", []):
            neg_texts.append(_get_text(neg))
        for neg in negs_raw.get("cross_lemma", []):
            neg_texts.append(_get_text(neg))
    else:
        neg_texts = [_get_text(n) for n in negs_raw]

    neg_texts = [t for t in neg_texts if t]

    if not anchor or not pos or not neg_texts:
        return -1e9, 999  # treat malformed as worst

    texts = [anchor, pos] + neg_texts
    E = encode_texts(model, texts, device=device, batch_size=64)

    a = E[0:1]
    p = E[1:2]
    n = E[2:]

    pos_sim = float(cosine_sim_matrix(a, p)[0, 0].item())
    neg_sims = cosine_sim_matrix(a, n).squeeze(0).numpy()
    max_neg = float(np.max(neg_sims))
    margin = pos_sim - max_neg
    rank = 1 + int(np.sum(neg_sims > pos_sim))

    return margin, rank


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True, help="Input bundles JSONL (tier3x merged file).")
    ap.add_argument("--capsule", required=True, help="Reference capsule dir (D1 baseline recommended).")
    ap.add_argument("--out", required=True, help="Output bundles JSONL (vetted).")

    ap.add_argument("--keep-frac", type=float, default=0.35,
                    help="Keep top fraction of expanded tier3x by margin (0-1). Default 0.35.")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="Additionally require margin >= this value for expanded items. Default 0.0.")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--max-expanded", type=int, default=None,
                    help="Optional cap for debugging (score only first N expanded).")

    args = ap.parse_args()

    bundles_path = Path(args.bundles)
    capsule_dir = Path(args.capsule)
    out_path = Path(args.out)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA not available, falling back to CPU")

    print(f"[info] Loading bundles: {bundles_path}")
    bundles = read_jsonl(bundles_path)

    expanded = [b for b in bundles if _is_expanded_tier3(b)]
    kept_static = [b for b in bundles if not _is_expanded_tier3(b)]

    print(f"[info] Total bundles: {len(bundles)}")
    print(f"[info] Expanded tier3x candidates: {len(expanded)}")
    print(f"[info] Static (kept as-is): {len(kept_static)}")

    if not expanded:
        print("[warn] No expanded tier3x found. Writing original bundles.")
        write_jsonl(out_path, bundles)
        return

    if args.max_expanded is not None:
        expanded = expanded[:args.max_expanded]
        print(f"[debug] max-expanded applied: scoring {len(expanded)} expanded bundles")

    print(f"[info] Loading reference model from capsule: {capsule_dir}")
    model = load_model(capsule_dir, device=device)

    margins: List[float] = []
    ranks: List[int] = []
    scored: List[Tuple[float, int, Dict[str, Any]]] = []

    for i, b in enumerate(expanded, 1):
        m, r = score_bundle_margin(model, b, device=device)
        margins.append(m)
        ranks.append(r)

        md = b.get("metadata") or {}
        md["vet_margin"] = float(m)
        md["vet_rank"] = int(r)
        md["vet_ref_capsule"] = str(capsule_dir)
        b["metadata"] = md

        scored.append((m, r, b))

        if i % 100 == 0:
            print(f"[score] {i}/{len(expanded)}  median_margin={np.median(margins):.6f}")

    margins_np = np.array(margins, dtype=np.float64)
    print("\n[summary] Expanded tier3x margin stats:")
    print(f"  n: {len(margins_np)}")
    print(f"  min/median/max: {margins_np.min():.6f} / {np.median(margins_np):.6f} / {margins_np.max():.6f}")
    print(f"  % margin >= 0: {float(np.mean(margins_np >= 0.0)) * 100:.1f}%")
    print(f"  % margin >= {args.min_margin}: {float(np.mean(margins_np >= args.min_margin)) * 100:.1f}%")

    # Filter: require margin >= min_margin then take top keep-frac by margin
    eligible = [t for t in scored if t[0] >= args.min_margin]
    eligible.sort(key=lambda x: x[0], reverse=True)

    k = max(1, int(len(scored) * args.keep_frac))
    kept_expanded = eligible[:k]

    kept_expanded_items = [t[2] for t in kept_expanded]
    print(f"\n[filter] keep-frac={args.keep_frac:.2f} => target_keep={k}")
    print(f"[filter] eligible (margin>=min_margin): {len(eligible)}")
    print(f"[filter] kept expanded: {len(kept_expanded_items)}")

    # Write vetted file: static + vetted expanded
    out_items = kept_static + kept_expanded_items

    print(f"[write] Output bundles: {len(out_items)} -> {out_path}")
    write_jsonl(out_path, out_items)

    # Write a small stats JSON next to output for convenience
    stats = {
        "input_bundles": str(bundles_path),
        "reference_capsule": str(capsule_dir),
        "expanded_total": len(scored),
        "expanded_eligible": len(eligible),
        "expanded_kept": len(kept_expanded_items),
        "keep_frac": args.keep_frac,
        "min_margin": args.min_margin,
        "margin_stats": {
            "min": float(margins_np.min()),
            "median": float(np.median(margins_np)),
            "max": float(margins_np.max()),
            "pct_ge_0": float(np.mean(margins_np >= 0.0)),
            "pct_ge_min_margin": float(np.mean(margins_np >= args.min_margin)),
        }
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[write] Stats: {stats_path}")


if __name__ == "__main__":
    main()
