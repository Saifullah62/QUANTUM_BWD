#!/usr/bin/env python3
"""
vet_tier3x_sbert.py
===================
Quality-filter tier3x expanded bundles using sentence-transformers (SBERT)
as the reference scorer instead of requiring a specific capsule.

Uses the SAME margin logic as eval: margin = pos_sim - max(neg_sim).

Goal:
- Keep only expanded items that *behave like real killers* under a trusted encoder.

Inputs:
- --bundles: JSONL of contrastive bundles (e.g., contrastive_bundles_v23_tier3x.jsonl)
- --out: output JSONL (e.g., contrastive_bundles_v23_tier3x_vetted.jsonl)

Filtering:
- Keep all non-expanded bundles as-is.
- For expanded tier3x bundles:
    - Score margin (pos_sim - max_neg_sim)
    - Keep top-K by margin (keep_frac) AND optionally require margin > min_margin
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


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
    if tier not in ("tier3_adversarial", "tier3_expanded"):
        return False
    return (md.get("expansion") == "tier3x") or (md.get("source") == "tier3_expansion")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def score_bundle_margin(encoder: Any, bundle: Dict[str, Any]) -> Tuple[float, int]:
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

    # Encode all texts
    texts = [anchor, pos] + neg_texts
    embeddings = encoder.encode(texts, convert_to_numpy=True)

    anchor_emb = embeddings[0]
    pos_emb = embeddings[1]
    neg_embs = embeddings[2:]

    pos_sim = cosine_sim(anchor_emb, pos_emb)
    neg_sims = np.array([cosine_sim(anchor_emb, n) for n in neg_embs])
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
    ap.add_argument("--out", required=True, help="Output bundles JSONL (vetted).")

    ap.add_argument("--keep-frac", type=float, default=0.35,
                    help="Keep top fraction of expanded tier3x by margin (0-1). Default 0.35.")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="Additionally require margin >= this value for expanded items. Default 0.0.")
    ap.add_argument("--model", default="all-MiniLM-L6-v2",
                    help="Sentence-transformers model to use. Default: all-MiniLM-L6-v2")
    ap.add_argument("--max-expanded", type=int, default=None,
                    help="Optional cap for debugging (score only first N expanded).")

    args = ap.parse_args()

    if not SBERT_AVAILABLE:
        print("[ERROR] sentence-transformers not available. Install with: pip install sentence-transformers")
        sys.exit(1)

    bundles_path = Path(args.bundles)
    out_path = Path(args.out)

    print(f"[info] Loading sentence-transformers model: {args.model}")
    encoder = SentenceTransformer(args.model)
    print(f"[OK] Model loaded")

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

    margins: List[float] = []
    ranks: List[int] = []
    scored: List[Tuple[float, int, Dict[str, Any]]] = []

    for i, b in enumerate(expanded, 1):
        m, r = score_bundle_margin(encoder, b)
        margins.append(m)
        ranks.append(r)

        md = b.get("metadata") or {}
        md["vet_margin"] = float(m)
        md["vet_rank"] = int(r)
        md["vet_model"] = args.model
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
        "reference_model": args.model,
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
