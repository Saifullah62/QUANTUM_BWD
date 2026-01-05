#!/usr/bin/env python3
"""
Organic Tier3 Miner
===================
Mines *new* tier3_adversarial bundles from canonicalized events by searching for
naturally confusable negatives, then validating hardness across multiple
negative lineups to avoid "pseudo-killers".

Core rule (strict tier3):
  danger = max_neg_sim - pos_sim
  pass if danger >= tier3_threshold in >= min_lineups

Outputs:
  - tier3_organic.jsonl
  - tier3_organic_stats.json
  - optional checkpoints

Assumptions about events JSONL:
  Each line is a JSON object containing at least:
    - lemma (or "headword")
    - sense_id (or "sense")
    - a text field ("text", "event_text", "sentence", "anchor", etc.)

This script is intentionally defensive and will try multiple key names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------
# Utilities
# ---------------------------

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    # assumes a,b are 1D normalized
    return float(np.dot(a, b))

def _safe_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default

def _normalize_text(s: str) -> str:
    s = (s or "").strip()
    # keep it light; we want stable dedupe signatures
    s = " ".join(s.split())
    return s

def _event_to_record(obj: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    lemma = _safe_get(obj, ["lemma", "headword", "word", "term"])
    sense_id = _safe_get(obj, ["sense_id", "sense", "senseId", "sense_key", "senseKey"])
    text = _safe_get(obj, ["text", "event_text", "sentence", "anchor", "example", "content", "utterance"])
    if not lemma or not sense_id or not text:
        return None
    rec = {
        "event_id": _safe_get(obj, ["event_id", "id", "uuid"], default=f"evt_{idx}"),
        "lemma": str(lemma),
        "sense_id": str(sense_id),
        "text": _normalize_text(str(text)),
    }
    return rec


# ---------------------------
# Embedding backend
# ---------------------------

class Embedder:
    def __init__(self, model_name: str, batch_size: int = 256, device: str = "cuda"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers is required for this miner.\n"
                "Install on RAMP: pip install -U sentence-transformers"
            ) from e

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.astype(np.float32)


# ---------------------------
# Approximate NN (FAISS optional)
# ---------------------------

class NearestIndex:
    def __init__(self, embs: np.ndarray):
        self.embs = embs.astype(np.float32)
        self.use_faiss = False
        self.index = None

        try:
            import faiss  # type: ignore
            self.use_faiss = True
            dim = self.embs.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # cosine if vectors normalized
            self.index.add(self.embs)
        except Exception:
            self.use_faiss = False
            self.index = None

    def search(self, query: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.use_faiss and self.index is not None:
            import faiss  # type: ignore
            sims, idxs = self.index.search(query, topk)
            return idxs[0], sims[0]

        # fallback: brute force cosine via dot product (vectors assumed normalized)
        sims = np.dot(self.embs, query[0])
        idxs = np.argsort(-sims)[:topk]
        return idxs, sims[idxs]


# ---------------------------
# Mining
# ---------------------------

@dataclass
class MinerConfig:
    tier3_threshold: float = 0.593
    lemma_topk: int = 40                 # within-lemma neighbor pool
    global_topk: int = 80                # global neighbor pool for cross-domain
    negatives_per_lineup: int = 5
    lineups_per_candidate: int = 6       # different negative sets to test
    min_lineups_to_pass: int = 3         # must pass in >= this many lineups
    candidates_per_sense: int = 6        # how many (anchor,pos) tries per sense
    max_senses: Optional[int] = None     # for debugging / partial runs
    seed: int = 42
    checkpoint_every: int = 200          # senses
    max_output: int = 20000              # safety cap

@dataclass
class MinerStats:
    timestamp: str
    events_loaded: int = 0
    senses: int = 0
    senses_with_pos: int = 0
    candidates_tested: int = 0
    candidates_passed: int = 0
    bundles_written: int = 0
    dedup_hits: int = 0
    skipped_no_pos: int = 0
    skipped_no_neg_pool: int = 0


class OrganicTier3Miner:
    def __init__(
        self,
        events_path: str,
        existing_bundles_path: Optional[str],
        out_path: str,
        stats_path: str,
        checkpoint_path: Optional[str],
        embed_model: str,
        device: str,
        cfg: MinerConfig,
    ):
        self.events_path = events_path
        self.existing_bundles_path = existing_bundles_path
        self.out_path = out_path
        self.stats_path = stats_path
        self.checkpoint_path = checkpoint_path
        self.embed_model = embed_model
        self.device = device
        self.cfg = cfg

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.stats = MinerStats(timestamp=_now_iso())
        self.events: List[Dict[str, Any]] = []
        self.embs: Optional[np.ndarray] = None
        self.index: Optional[NearestIndex] = None

        self.by_lemma: Dict[str, List[int]] = {}
        self.by_sense: Dict[str, List[int]] = {}

        self.dedupe_signatures: set[str] = set()

    def load_events(self):
        events: List[Dict[str, Any]] = []
        with open(self.events_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rec = _event_to_record(obj, i)
                if rec is None:
                    continue
                events.append(rec)

        self.events = events
        self.stats.events_loaded = len(events)

        for i, e in enumerate(events):
            self.by_lemma.setdefault(e["lemma"], []).append(i)
            self.by_sense.setdefault(e["sense_id"], []).append(i)

        self.stats.senses = len(self.by_sense)

    def load_dedupe(self):
        if not self.existing_bundles_path:
            return

        # Dedup against existing tier3 bundles by (sense_id, pos_sense_id, neg_sense_ids set)
        # This is conservative but effective.
        seen = 0
        with open(self.existing_bundles_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    b = json.loads(line)
                except Exception:
                    continue
                meta = b.get("metadata", {}) or {}
                tier = meta.get("difficulty_tier", "")
                # If you want to dedupe against all, remove this filter.
                if tier and tier != "tier3_adversarial":
                    continue
                anchor = b.get("anchor", {}) or {}
                pos = b.get("positive", {}) or {}
                negs = b.get("negatives", []) or []
                sid = anchor.get("sense_id") or anchor.get("senseId") or ""
                psid = pos.get("sense_id") or pos.get("senseId") or ""
                nsids = sorted([(n.get("sense_id") or n.get("senseId") or "") for n in negs if isinstance(n, dict)])
                sig = _sha1("|".join([str(sid), str(psid)] + nsids))
                self.dedupe_signatures.add(sig)
                seen += 1

        print(f"[dedupe] loaded {seen} tier3 signatures from {self.existing_bundles_path}")

    def embed_all(self):
        emb = Embedder(self.embed_model, device=self.device)
        texts = [e["text"] for e in self.events]
        embs = []

        bs = 2048  # big batches are fine on GPU; adjust if needed
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            embs.append(emb.encode(chunk))

        self.embs = np.vstack(embs).astype(np.float32)
        # already normalized by SentenceTransformer, but keep safe
        self.embs = _l2_normalize(self.embs)
        self.index = NearestIndex(self.embs)

    def _pick_anchor_and_pos(self, sense_id: str) -> Optional[Tuple[int, int]]:
        idxs = self.by_sense.get(sense_id, [])
        if len(idxs) < 2:
            return None
        anchor_i = random.choice(idxs)
        # pick a different example as positive
        pos_i = random.choice([j for j in idxs if j != anchor_i])
        return anchor_i, pos_i

    def _within_lemma_confusers(self, anchor_i: int, topk: int) -> List[int]:
        assert self.embs is not None and self.index is not None
        e = self.events[anchor_i]
        lemma = e["lemma"]
        lemma_idxs = self.by_lemma.get(lemma, [])
        if len(lemma_idxs) < 2:
            return []

        # search global neighbors, then filter to lemma pool
        idxs, sims = self.index.search(self.embs[anchor_i], topk=topk*4)
        out = []
        anchor_sid = e["sense_id"]
        for j in idxs:
            if j == anchor_i:
                continue
            if j not in lemma_idxs:
                continue
            if self.events[j]["sense_id"] == anchor_sid:
                continue
            out.append(int(j))
            if len(out) >= topk:
                break
        return out

    def _global_confusers(self, anchor_i: int, topk: int) -> List[int]:
        assert self.embs is not None and self.index is not None
        e = self.events[anchor_i]
        lemma = e["lemma"]
        anchor_sid = e["sense_id"]

        idxs, sims = self.index.search(self.embs[anchor_i], topk=topk)
        out = []
        for j in idxs:
            if j == anchor_i:
                continue
            ej = self.events[int(j)]
            # exclude same sense and (optionally) same lemma to force cross-domain
            if ej["sense_id"] == anchor_sid:
                continue
            if ej["lemma"] == lemma:
                continue
            out.append(int(j))
        return out

    def _score_lineup(self, anchor_i: int, pos_i: int, neg_is: List[int]) -> Tuple[float, float, float]:
        """
        Returns:
          pos_sim, max_neg_sim, danger
        """
        assert self.embs is not None
        a = self.embs[anchor_i]
        p = self.embs[pos_i]
        pos_sim = _cos_sim(a, p)

        neg_sims = [_cos_sim(a, self.embs[n]) for n in neg_is]
        max_neg = float(max(neg_sims)) if neg_sims else -1.0
        danger = max_neg - pos_sim
        return pos_sim, max_neg, danger

    def _bundle_signature(self, anchor_sid: str, pos_sid: str, neg_sids: List[str]) -> str:
        neg_sids = sorted(neg_sids)
        return _sha1("|".join([anchor_sid, pos_sid] + neg_sids))

    def _write_bundle(self, bundle: Dict[str, Any], fout):
        fout.write(json.dumps(bundle, ensure_ascii=False) + "\n")
        self.stats.bundles_written += 1

    def run(self):
        print("=" * 70)
        print("ORGANIC TIER3 MINER")
        print("=" * 70)
        print(f"events:      {self.events_path}")
        print(f"dedupe:      {self.existing_bundles_path or 'none'}")
        print(f"out:         {self.out_path}")
        print(f"threshold:   {self.cfg.tier3_threshold:.4f}")
        print(f"device:      {self.device}")
        print(f"embed_model: {self.embed_model}")
        print("")

        self.load_events()
        self.load_dedupe()

        print(f"[load] events_loaded={self.stats.events_loaded:,} senses={self.stats.senses:,}")
        print("[embed] embedding all events...")
        t0 = time.time()
        self.embed_all()
        print(f"[embed] done in {time.time()-t0:.1f}s")

        senses = list(self.by_sense.keys())
        random.shuffle(senses)
        if self.cfg.max_senses:
            senses = senses[: self.cfg.max_senses]

        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)

        # resume support: if checkpoint exists, load written signatures to prevent duplicates
        written_sig: set[str] = set()
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    ck = json.load(f)
                written_sig = set(ck.get("written_sig", []))
                print(f"[resume] loaded checkpoint: {len(written_sig)} written signatures")
            except Exception:
                print("[resume] checkpoint unreadable, starting fresh")

        mode = "a" if os.path.exists(self.out_path) else "w"
        with open(self.out_path, mode, encoding="utf-8") as fout:
            for si, sense_id in enumerate(senses, start=1):
                # checkpoint cadence
                if self.checkpoint_path and (si % self.cfg.checkpoint_every == 0):
                    self._save_checkpoint(written_sig, si)

                # build candidates for this sense
                picks = []
                for _ in range(self.cfg.candidates_per_sense):
                    ap = self._pick_anchor_and_pos(sense_id)
                    if ap:
                        picks.append(ap)

                if not picks:
                    self.stats.skipped_no_pos += 1
                    continue

                self.stats.senses_with_pos += 1

                for (anchor_i, pos_i) in picks:
                    self.stats.candidates_tested += 1

                    # mine confuser pools
                    within = self._within_lemma_confusers(anchor_i, topk=self.cfg.lemma_topk)
                    global_pool = self._global_confusers(anchor_i, topk=self.cfg.global_topk)

                    if len(within) < max(3, self.cfg.negatives_per_lineup):
                        # if no within-lemma confusers exist, we can still try cross-domain,
                        # but tier3 is usually anchored in within-lemma ambiguity.
                        # We'll allow it, but require enough pool.
                        pass

                    if len(within) + len(global_pool) < self.cfg.negatives_per_lineup:
                        self.stats.skipped_no_neg_pool += 1
                        continue

                    # build multiple lineups (different negative sets)
                    lineup_records = []
                    passes = 0

                    for l in range(self.cfg.lineups_per_candidate):
                        negs = []

                        # ensure some within-lemma confusers if possible
                        n_within = min(3, self.cfg.negatives_per_lineup)
                        if within:
                            negs.extend(random.sample(within, k=min(n_within, len(within))))

                        # fill remainder from global pool (or from within if needed)
                        need = self.cfg.negatives_per_lineup - len(negs)
                        pool = [x for x in global_pool if x not in negs]
                        if need > 0 and pool:
                            negs.extend(random.sample(pool, k=min(need, len(pool))))
                        need = self.cfg.negatives_per_lineup - len(negs)
                        if need > 0 and within:
                            extra = [x for x in within if x not in negs]
                            if extra:
                                negs.extend(random.sample(extra, k=min(need, len(extra))))

                        if len(negs) < self.cfg.negatives_per_lineup:
                            continue

                        pos_sim, max_neg, danger = self._score_lineup(anchor_i, pos_i, negs)

                        lineup_records.append({
                            "neg_event_ids": [self.events[n]["event_id"] for n in negs],
                            "neg_sense_ids": [self.events[n]["sense_id"] for n in negs],
                            "pos_sim": pos_sim,
                            "max_neg_sim": max_neg,
                            "danger": danger,
                        })

                        if danger >= self.cfg.tier3_threshold:
                            passes += 1

                    if passes < self.cfg.min_lineups_to_pass:
                        continue

                    # Candidate qualifies as organic tier3
                    anchor = self.events[anchor_i]
                    pos = self.events[pos_i]

                    neg_sids_flat = []
                    for lr in lineup_records:
                        neg_sids_flat.extend(lr["neg_sense_ids"])
                    # for signature, use a representative set: top frequent neg sense ids across lineups
                    neg_sids_sorted = sorted(set(neg_sids_flat))

                    sig = self._bundle_signature(anchor["sense_id"], pos["sense_id"], neg_sids_sorted)
                    if sig in self.dedupe_signatures or sig in written_sig:
                        self.stats.dedup_hits += 1
                        continue

                    # Build a "canonical" negatives list from one of the passing lineups with max danger
                    best = max(lineup_records, key=lambda r: r["danger"])
                    neg_indices = []
                    # map best neg_event_ids back to indices for text
                    best_ids = set(best["neg_event_ids"])
                    for j, ev in enumerate(self.events):
                        if ev["event_id"] in best_ids:
                            neg_indices.append(j)

                    negatives = []
                    for n in neg_indices:
                        ev = self.events[n]
                        negatives.append({
                            "text": ev["text"],
                            "sense_id": ev["sense_id"],
                            "lemma": ev["lemma"],
                            "event_id": ev["event_id"],
                        })

                    bundle = {
                        "anchor": {
                            "text": anchor["text"],
                            "sense_id": anchor["sense_id"],
                            "lemma": anchor["lemma"],
                            "event_id": anchor["event_id"],
                        },
                        "positive": {
                            "text": pos["text"],
                            "sense_id": pos["sense_id"],
                            "lemma": pos["lemma"],
                            "event_id": pos["event_id"],
                        },
                        "negatives": negatives,
                        "metadata": {
                            "difficulty_tier": "tier3_adversarial",
                            "source": "tier3_organic_miner",
                            "tier3_threshold": self.cfg.tier3_threshold,
                            "passes": passes,
                            "min_lineups_to_pass": self.cfg.min_lineups_to_pass,
                            "lineups_per_candidate": self.cfg.lineups_per_candidate,
                            "lineup_dangers": [lr["danger"] for lr in lineup_records],
                            "best_lineup": best,
                            "timestamp": _now_iso(),
                        }
                    }

                    self.stats.candidates_passed += 1
                    written_sig.add(sig)
                    self._write_bundle(bundle, fout)

                    if self.stats.bundles_written >= self.cfg.max_output:
                        print("[stop] reached max_output cap")
                        self._save_checkpoint(written_sig, si)
                        self._save_stats()
                        return

                # light progress pulse
                if si % 100 == 0:
                    print(f"[progress] senses={si}/{len(senses)} written={self.stats.bundles_written} "
                          f"passed={self.stats.candidates_passed} dedupe={self.stats.dedup_hits}")

        self._save_checkpoint(written_sig, len(senses))
        self._save_stats()

    def _save_checkpoint(self, written_sig: set[str], sense_i: int):
        if not self.checkpoint_path:
            return
        tmp = self.checkpoint_path + ".tmp"
        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": _now_iso(),
                "sense_index": sense_i,
                "written_sig": list(written_sig),
                "stats": asdict(self.stats),
                "config": asdict(self.cfg),
            }, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.checkpoint_path)

    def _save_stats(self):
        os.makedirs(os.path.dirname(self.stats_path) or ".", exist_ok=True)
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.stats), f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True, help="canonicalized events jsonl (e.g., canonicalized_v21.jsonl)")
    p.add_argument("--existing-tier3", default=None, help="existing bundles jsonl to dedupe against (tier3 only)")
    p.add_argument("--out", required=True, help="output jsonl for mined organic tier3 bundles")
    p.add_argument("--stats", required=True, help="output stats json")
    p.add_argument("--checkpoint", default=None, help="checkpoint json for resume")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default="cuda")

    # knobs
    p.add_argument("--tier3-threshold", type=float, default=0.593)
    p.add_argument("--lemma-topk", type=int, default=40)
    p.add_argument("--global-topk", type=int, default=80)
    p.add_argument("--negatives-per-lineup", type=int, default=5)
    p.add_argument("--lineups-per-candidate", type=int, default=6)
    p.add_argument("--min-lineups-to-pass", type=int, default=3)
    p.add_argument("--candidates-per-sense", type=int, default=6)
    p.add_argument("--max-senses", type=int, default=None)
    p.add_argument("--max-output", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=200)

    args = p.parse_args()

    cfg = MinerConfig(
        tier3_threshold=args.tier3_threshold,
        lemma_topk=args.lemma_topk,
        global_topk=args.global_topk,
        negatives_per_lineup=args.negatives_per_lineup,
        lineups_per_candidate=args.lineups_per_candidate,
        min_lineups_to_pass=args.min_lineups_to_pass,
        candidates_per_sense=args.candidates_per_sense,
        max_senses=args.max_senses,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        max_output=args.max_output,
    )

    miner = OrganicTier3Miner(
        events_path=args.events,
        existing_bundles_path=args.existing_tier3,
        out_path=args.out,
        stats_path=args.stats,
        checkpoint_path=args.checkpoint,
        embed_model=args.embed_model,
        device=args.device,
        cfg=cfg,
    )
    miner.run()
    print("[done] wrote:", args.out)
    print("[done] stats:", args.stats)


if __name__ == "__main__":
    main()
