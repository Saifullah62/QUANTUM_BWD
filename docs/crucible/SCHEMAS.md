# Crucible Data Schemas

JSON schemas for all data formats used in Crucible.

## Canonical Event

The base unit of data. Represents a single sense-disambiguated text instance.

```json
{
  "event_id": "evt_001",
  "sense": "bank#financial",
  "text": "The bank approved the loan application.",
  "lemma": "bank",
  "embedding": [0.123, -0.456, ...]
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event_id` | string | Yes | Unique identifier |
| `sense` | string | Yes | Sense label (format: `lemma#sense_id`) |
| `text` | string | Yes | The text instance |
| `lemma` | string | No | Lemma extracted from sense |
| `embedding` | array[float] | No | Pre-computed embedding vector |

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["event_id", "sense", "text"],
  "properties": {
    "event_id": {
      "type": "string",
      "pattern": "^evt_[a-zA-Z0-9_]+$"
    },
    "sense": {
      "type": "string",
      "pattern": "^[a-z]+#[a-zA-Z0-9_]+$"
    },
    "text": {
      "type": "string",
      "minLength": 1
    },
    "lemma": {
      "type": "string"
    },
    "embedding": {
      "type": "array",
      "items": {"type": "number"}
    }
  }
}
```

---

## Contrastive Bundle

A training/evaluation unit with anchor, positive, and negatives.

```json
{
  "bundle_id": "bnd_001",
  "anchor": {
    "event_id": "evt_001",
    "sense": "bank#financial",
    "text": "The bank approved the loan."
  },
  "positive": {
    "event_id": "evt_002",
    "sense": "bank#financial",
    "text": "Check your bank balance."
  },
  "negatives": [
    {
      "event_id": "evt_003",
      "sense": "bank#river",
      "text": "We walked along the river bank.",
      "source": "within_lemma"
    },
    {
      "event_id": "evt_004",
      "sense": "deposit#geological",
      "text": "The mineral deposit was valuable.",
      "source": "cross_lemma"
    }
  ],
  "metadata": {
    "tier": "tier3_adversarial",
    "danger_score": 0.623,
    "source": "tier3_organic_miner",
    "passes": 8,
    "lineups_tested": 10,
    "mined_at": "2026-01-05T10:30:00Z"
  }
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `bundle_id` | string | Yes | Unique bundle identifier |
| `anchor` | Event | Yes | The anchor event |
| `positive` | Event | Yes | Same-sense positive |
| `negatives` | array[NegativeEvent] | Yes | Negative examples |
| `metadata` | BundleMetadata | Yes | Tier, scores, provenance |

### NegativeEvent

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event_id` | string | Yes | Event identifier |
| `sense` | string | Yes | Sense label |
| `text` | string | Yes | The text |
| `source` | string | No | Negative source type |

**Negative Source Types**:
- `within_lemma`: Same lemma, different sense
- `sibling`: Related sense in hierarchy
- `cross_lemma`: Different lemma, confusing context

### BundleMetadata

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tier` | string | Yes | Tier assignment |
| `danger_score` | float | Yes | Computed danger score |
| `source` | string | No | Provenance tag |
| `expansion` | string | No | Expansion method (for tier3x) |
| `passes` | int | No | Lineup passes (organic) |
| `lineups_tested` | int | No | Total lineups (organic) |
| `mined_at` | string | No | ISO timestamp |

### Tier Values

- `tier1_easy`
- `tier2_robust`
- `tier3_adversarial`

### Source Values

- `bundle_generator` (default)
- `tier3_organic_miner`
- `tier3_expansion`
- `manual_curation`

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["bundle_id", "anchor", "positive", "negatives", "metadata"],
  "properties": {
    "bundle_id": {"type": "string"},
    "anchor": {"$ref": "#/definitions/event"},
    "positive": {"$ref": "#/definitions/event"},
    "negatives": {
      "type": "array",
      "items": {"$ref": "#/definitions/negative_event"},
      "minItems": 1
    },
    "metadata": {"$ref": "#/definitions/bundle_metadata"}
  },
  "definitions": {
    "event": {
      "type": "object",
      "required": ["event_id", "sense", "text"],
      "properties": {
        "event_id": {"type": "string"},
        "sense": {"type": "string"},
        "text": {"type": "string"}
      }
    },
    "negative_event": {
      "allOf": [
        {"$ref": "#/definitions/event"},
        {
          "properties": {
            "source": {
              "type": "string",
              "enum": ["within_lemma", "sibling", "cross_lemma"]
            }
          }
        }
      ]
    },
    "bundle_metadata": {
      "type": "object",
      "required": ["tier", "danger_score"],
      "properties": {
        "tier": {
          "type": "string",
          "enum": ["tier1_easy", "tier2_robust", "tier3_adversarial"]
        },
        "danger_score": {"type": "number"},
        "source": {"type": "string"},
        "expansion": {"type": "string"},
        "passes": {"type": "integer", "minimum": 0},
        "lineups_tested": {"type": "integer", "minimum": 1},
        "mined_at": {"type": "string", "format": "date-time"}
      }
    }
  }
}
```

---

## Scoreboard

Evaluation results output.

```json
{
  "eval_header": {
    "eval_path": "evals/frozen_eval_v23.jsonl",
    "fingerprint": "a1b2c3d4e5f6g7h8|42|3786",
    "source_hash": "a1b2c3d4e5f6g7h8",
    "seed": 42,
    "item_count": 3786,
    "tier_distribution": {
      "tier1_easy": 1666,
      "tier2_robust": 1666,
      "tier3_adversarial": 454
    },
    "version": "v23"
  },
  "checkpoint": {
    "path": "results/baseline/checkpoint_final.pt",
    "hash": "SHA256:f1e2d3c4..."
  },
  "overall": {
    "n": 3786,
    "correct": 2739,
    "accuracy": 0.7234,
    "pass_rate": 0.7234,
    "mean_margin": 0.045,
    "median_margin": 0.067,
    "q10_margin": -0.089,
    "q90_margin": 0.156,
    "mean_rank": 1.34
  },
  "by_tier": {
    "tier1_easy": {
      "n": 1666,
      "correct": 1420,
      "accuracy": 0.8524,
      "pass_rate": 0.8524,
      "mean_margin": 0.112,
      "median_margin": 0.134,
      "q10_margin": 0.023,
      "q90_margin": 0.198,
      "mean_rank": 1.12
    },
    "tier2_robust": {
      "n": 1666,
      "correct": 1145,
      "accuracy": 0.6872,
      "pass_rate": 0.6872,
      "mean_margin": 0.031,
      "median_margin": 0.045,
      "q10_margin": -0.056,
      "q90_margin": 0.112,
      "mean_rank": 1.41
    },
    "tier3_adversarial": {
      "n": 454,
      "correct": 136,
      "accuracy": 0.2996,
      "pass_rate": 0.2996,
      "mean_margin": -0.089,
      "median_margin": -0.067,
      "q10_margin": -0.234,
      "q90_margin": 0.034,
      "mean_rank": 2.87
    }
  },
  "timestamp": "2026-01-05T10:30:00Z",
  "crucible_version": "1.0.0"
}
```

### TierStats Fields

| Field | Type | Description |
|-------|------|-------------|
| `n` | int | Items in tier |
| `correct` | int | Items with positive margin |
| `accuracy` | float | correct / n |
| `pass_rate` | float | Same as accuracy |
| `mean_margin` | float | Average margin |
| `median_margin` | float | Median margin |
| `q10_margin` | float | 10th percentile |
| `q90_margin` | float | 90th percentile |
| `mean_rank` | float | Average rank of positive |

---

## Pool Manifest

Tier3 pool composition tracking.

```json
{
  "pools": {
    "legacy": {
      "path": "data/tier3_legacy.jsonl",
      "count": 454,
      "hash": "SHA256:a1b2c3d4..."
    },
    "organic": {
      "path": "data/tier3_organic_train.jsonl",
      "count": 320,
      "hash": "SHA256:e5f6g7h8..."
    },
    "expanded": {
      "path": null,
      "count": 0,
      "hash": null
    }
  },
  "total_tier3": 774,
  "created_at": "2026-01-05T10:30:00Z"
}
```

---

## Mining Stats

Output from organic mining runs.

```json
{
  "candidates_tested": 25000,
  "candidates_accepted": 352,
  "pass_rate": 0.0141,
  "pass_distribution": {
    "2": 0,
    "3": 45,
    "4": 67,
    "5": 58,
    "6": 52,
    "7": 48,
    "8": 38,
    "9": 28,
    "10": 16
  },
  "parameters": {
    "threshold": 0.593,
    "lineups_per_candidate": 10,
    "min_lineups_to_pass": 3,
    "candidates_per_sense": 15
  },
  "output_path": "data/tier3_organic.jsonl",
  "duration_seconds": 3847,
  "timestamp": "2026-01-05T10:30:00Z"
}
```

---

## Capsule Metadata

Reproducibility metadata for packaged artifacts.

```json
{
  "capsule_version": "1.0",
  "created_at": "2026-01-05T10:30:00Z",
  "crucible_version": "1.0.0",
  "git": {
    "commit": "a1b2c3d4e5f6g7h8",
    "branch": "main",
    "dirty": false
  },
  "config_hash": "SHA256:f1e2d3c4...",
  "checkpoint_hash": "SHA256:a9b8c7d6...",
  "training": {
    "steps": 10000,
    "final_loss": 0.234,
    "final_accuracy": 0.876
  },
  "scoreboards": [
    {
      "name": "frozen",
      "path": "frozen_scoreboard.json",
      "fingerprint": "a1b2c3d4e5f6g7h8|42|3786"
    },
    {
      "name": "organic",
      "path": "organic_scoreboard.json",
      "fingerprint": "i9j0k1l2m3n4o5p6|42|80"
    }
  ]
}
```

---

## Sweep Results

Aggregated results from multi-condition experiments.

```json
{
  "sweep_type": "ratio",
  "conditions": ["R85_10_5", "R80_15_5", "R75_15_10", "R70_20_10"],
  "seeds": [42, 123, 456],
  "capsules": [
    {
      "condition": "R75_15_10",
      "seed": 42,
      "checkpoint": "R75_15_10/seed_42/checkpoint_final.pt",
      "overall": {
        "accuracy": 0.7234,
        "pass_rate": 0.7234,
        "mean_margin": 0.045
      },
      "by_tier": {
        "tier1_easy": {"pass_rate": 0.8524},
        "tier2_robust": {"pass_rate": 0.6872},
        "tier3_adversarial": {"pass_rate": 0.2996}
      }
    }
  ],
  "aggregated": {
    "R75_15_10": {
      "frozen_tier3": {"mean": 0.302, "std": 0.012},
      "organic_holdout": {"mean": 0.308, "std": 0.015}
    }
  },
  "pareto_frontier": [
    {"condition": "R75_15_10", "frozen_tier3": 0.302, "organic_holdout": 0.308}
  ],
  "timestamp": "2026-01-05T10:30:00Z"
}
```

---

## File Formats

### JSONL Files

All data files use JSON Lines format (one JSON object per line):

```
{"event_id": "evt_001", "sense": "bank#financial", "text": "..."}
{"event_id": "evt_002", "sense": "bank#river", "text": "..."}
```

### Naming Conventions

| Pattern | Description |
|---------|-------------|
| `events_*.jsonl` | Canonical events |
| `contrastive_bundles_*.jsonl` | Tiered bundles |
| `tier3_*.jsonl` | Tier3 pool files |
| `eval_*.jsonl` | Evaluation sets |
| `*_scoreboard.json` | Evaluation results |
| `checkpoint_*.pt` | Model checkpoints |

### Version Tags

Include version in filenames for data lineage:

- `contrastive_bundles_v25.jsonl`
- `eval_v23_contrastive_5k.jsonl`
- `tier3_organic_v2.jsonl`
