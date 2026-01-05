# Current Implementation Reference

> **Auto-generated** by `tools/extract_script_flags.py`
> Last updated: 2026-01-05 09:15:35

This document shows the actual flags available in the current scripts.
For the target `crucible` CLI interface, see [CLI.md](CLI.md).

## Training (v3, three-pool ratios)

**Script**: `scripts/train_curriculum_v3.py`

```
usage: train_curriculum_v3.py [-h] --bundles BUNDLES [--steps STEPS]
                              [--batch-size BATCH_SIZE] [--lr LR]
                              [--margin MARGIN] [--embed-dim EMBED_DIM]
                              [--phase-dim PHASE_DIM] [--seed SEED]
                              [--device DEVICE] [--output-dir OUTPUT_DIR]
                              [--max-samples MAX_SAMPLES]
                              [--curriculum {phased,phased_v2,uniform}]
                              [--tier3-legacy TIER3_LEGACY]
                              [--tier3-organic TIER3_ORGANIC]
                              [--tier3-expanded TIER3_EXPANDED]

Curriculum training with three-pool tier3 mixing

options:
  -h, --help            show this help message and exit
  --bundles BUNDLES     Path to bundles JSONL
  --steps STEPS
  --batch-size BATCH_SIZE
  --lr LR
  --margin MARGIN
  --embed-dim EMBED_DIM
  --phase-dim PHASE_DIM
  --seed SEED
  --device DEVICE
  --output-dir OUTPUT_DIR
  --max-samples MAX_SAMPLES
  --curriculum {phased,phased_v2,uniform}
  --tier3-legacy TIER3_LEGACY
                        Tier3 legacy (original killer) ratio. Default 0.80
  --tier3-organic TIER3_ORGANIC
                        Tier3 organic (newly mined) ratio. Default 0.10
  --tier3-expanded TIER3_EXPANDED
                        Tier3 expanded (vetted tier3x) ratio. Default 0.10
```

---

## Training (v2, single tier3-mix)

**Script**: `scripts/train_curriculum_v2.py`

```
usage: train_curriculum_v2.py [-h] --bundles BUNDLES [--steps STEPS]
                              [--batch-size BATCH_SIZE] [--lr LR]
                              [--margin MARGIN] [--embed-dim EMBED_DIM]
                              [--phase-dim PHASE_DIM] [--seed SEED]
                              [--device DEVICE] [--output-dir OUTPUT_DIR]
                              [--max-samples MAX_SAMPLES]
                              [--curriculum {phased,phased_v2,uniform}]
                              [--tier3-mix RATIO]

Curriculum training for v2.3 bundles

options:
  -h, --help            show this help message and exit
  --bundles BUNDLES     Path to v2.3 bundles JSONL
  --steps STEPS         Training steps
  --batch-size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --margin MARGIN       Contrastive margin
  --embed-dim EMBED_DIM
                        Embedding dimension
  --phase-dim PHASE_DIM
                        Phase dimension
  --seed SEED           Random seed
  --device DEVICE
  --output-dir OUTPUT_DIR
                        Output directory
  --max-samples MAX_SAMPLES
                        Max training samples
  --curriculum {phased,phased_v2,uniform}
                        Curriculum type: phased (C2), phased_v2 (C2b with
                        teeth), or uniform (C1 baseline)
  --tier3-mix RATIO     Tier3 core ratio (0.0-1.0). E.g., 0.7 = 70% core, 30%
                        expanded. None = no mixing.
```

---

## Evaluation

**Script**: `experiments/eval_capsules.py`

```
usage: eval_capsules.py [-h] --eval EVAL [--results-root RESULTS_ROOT]
                        [--device DEVICE] [--out OUT]

Evaluate capsules against frozen eval set

options:
  -h, --help            show this help message and exit
  --eval EVAL           Eval JSONL file
  --results-root RESULTS_ROOT
                        Root folder containing condition/seed capsules
  --device DEVICE
  --out OUT             Output scoreboard JSON
```

---

## CLI vs Scripts

| Feature | Target CLI (`crucible`) | Current Scripts |
|---------|------------------------|-----------------|
| Three-pool ratios | `--tier3-legacy/organic/expanded` | v3 only |
| Single tier3 mix | - | v2 `--tier3-mix` |
| Eval flags | `--checkpoint`, `--eval-set` | `--eval`, `--results-root` |

The `crucible` CLI documented in [CLI.md](CLI.md) is the target interface.
Use the scripts above for current implementation.