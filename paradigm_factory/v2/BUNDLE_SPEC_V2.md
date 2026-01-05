# Bundle Specification v2

## Overview

This spec defines the data formats and quality gates for the next-generation polysemy bundle pipeline. The goal is to produce bundles with:
- **Longer contexts** (15-40 tokens vs current 5-11)
- **Diverse context styles** (news, instructional, dialogue, technical, etc.)
- **Adversarial hard negatives** (lexical + embedding confusability)
- **Provenance tracking** for audit

---

## 1. Raw Usage Event Schema

Each scraped example is a "sense-labeled usage event" with provenance.

```json
{
  "event_id": "uuid-v4",
  "lemma": "bank",
  "pos": "NOUN",
  "sense_id": "bank#financial_institution",
  "sense_gloss": "a financial establishment that accepts deposits and lends money",

  "source": {
    "dataset": "wikipedia|news|books|web|wiktionary",
    "url": "https://...",
    "doc_id": "...",
    "license": "CC-BY-SA|public_domain|fair_use",
    "retrieved_at": "2026-01-04T12:34:56Z"
  },

  "context": {
    "sentence": "After the storm, the river bank collapsed near the footbridge.",
    "left": "After the storm, the river",
    "target": "bank",
    "right": "collapsed near the footbridge.",
    "span_char": [24, 28],
    "span_token": [5, 6],
    "full_window": "... preceding context ... After the storm, the river bank collapsed near the footbridge. ... following context ..."
  },

  "signals": {
    "disambiguators": ["river", "collapsed", "footbridge"],
    "cue_hits": ["river"],
    "anti_cue_hits": [],
    "topic_tags": ["geography", "weather"],
    "named_entities": [],
    "numbers_present": false,
    "has_prep_phrase": true,
    "has_verb_object": true
  },

  "quality": {
    "is_definition_like": false,
    "is_template_like": false,
    "context_len_tokens": 12,
    "window_len_tokens": 28,
    "contains_quote": false,
    "has_strong_disambiguator": true,
    "style": "news|instructional|dialogue|technical|narrative|forum"
  }
}
```

---

## 2. Sense Inventory Schema

Each lemma has a canonical sense catalog with cue/anti-cue bags.

```json
{
  "lemma": "bank",
  "pos": "NOUN",
  "senses": [
    {
      "sense_id": "bank#financial_institution",
      "label": "financial_institution",
      "gloss": "a financial establishment that accepts deposits and lends money",
      "definition_source": "wordnet|wiktionary|custom",
      "cues": {
        "keywords": ["money", "deposit", "loan", "account", "savings", "withdraw"],
        "collocates": ["at the bank", "bank account", "bank loan", "central bank"],
        "prepositions": ["at", "with", "from"],
        "argument_patterns": ["deposit_in", "withdraw_from", "open_account_at"]
      },
      "anti_cues": {
        "keywords": ["river", "slope", "shore", "edge", "side"],
        "collocates": ["river bank", "bank of the river", "steep bank"]
      },
      "domain": "finance",
      "frequency_rank": 1
    },
    {
      "sense_id": "bank#river_edge",
      "label": "river_edge",
      "gloss": "the sloping land beside a body of water",
      "definition_source": "wordnet",
      "cues": {
        "keywords": ["river", "stream", "water", "shore", "slope", "erosion"],
        "collocates": ["river bank", "along the bank", "steep bank", "opposite bank"],
        "prepositions": ["along", "on", "beside"],
        "argument_patterns": ["walk_along", "sit_on", "fish_from"]
      },
      "anti_cues": {
        "keywords": ["money", "account", "loan", "deposit", "savings"],
        "collocates": ["bank account", "bank loan"]
      },
      "domain": "geography",
      "frequency_rank": 2
    }
  ],
  "confusable_pairs": [
    {
      "sense_a": "bank#financial_institution",
      "sense_b": "bank#river_edge",
      "confusability_score": 0.3,
      "shared_collocates": ["the bank", "near the bank"],
      "differentiating_features": ["domain nouns", "prepositional phrases"]
    }
  ]
}
```

---

## 3. Bundle Schema v2

Extended from v1 with richer negative ladder and provenance.

```json
{
  "schema_version": "2.1",
  "bundle_id": "uuid-v4",
  "paradigm": "polysemy",

  "word": {
    "lemma": "bank",
    "pos": "NOUN",
    "language": "en"
  },

  "sense_catalog": [
    {
      "sense_id": "bank#financial_institution",
      "label": "financial_institution",
      "gloss": "a financial establishment...",
      "cues": ["money", "deposit", "loan"],
      "anti_cues": ["river", "shore"]
    }
  ],

  "items": [
    {
      "item_id": "anchor_001",
      "role": "anchor",
      "sense_id": "bank#financial_institution",
      "context": "She walked into the bank to open a new savings account for her daughter.",
      "context_window": "... extended context ...",
      "target_span": [21, 25],
      "disambiguators": ["savings", "account", "open"],
      "source_event_id": "raw-event-uuid",
      "difficulty": 0.3,
      "hardness": "medium",
      "style": "narrative"
    },
    {
      "item_id": "pos_001",
      "role": "positive",
      "sense_id": "bank#financial_institution",
      "context": "The central bank announced new interest rate policies yesterday.",
      "context_window": "...",
      "target_span": [12, 16],
      "disambiguators": ["central", "interest", "rate", "policies"],
      "source_event_id": "raw-event-uuid-2",
      "difficulty": 0.25,
      "hardness": "easy",
      "style": "news"
    },
    {
      "item_id": "neg_easy_001",
      "role": "negative",
      "sense_id": "bank#river_edge",
      "context": "The fisherman sat on the grassy bank of the river, watching the water flow by.",
      "negative_type": "easy",
      "lexical_overlap": 0.1,
      "embedding_similarity": 0.3
    },
    {
      "item_id": "neg_hard_001",
      "role": "hard_negative",
      "sense_id": "bank#river_edge",
      "context": "After the flood, the bank needed extensive repairs before it could be used again.",
      "negative_type": "hard_lexical",
      "lexical_overlap": 0.4,
      "embedding_similarity": 0.7,
      "confusion_reason": "shared_collocate:repairs,used"
    },
    {
      "item_id": "neg_hard_002",
      "role": "hard_negative",
      "sense_id": "bank#river_edge",
      "context": "The bank was crowded with people enjoying the sunny afternoon.",
      "negative_type": "hard_embedding",
      "lexical_overlap": 0.2,
      "embedding_similarity": 0.85,
      "confusion_reason": "embedding_nearest_neighbor"
    }
  ],

  "pairings": {
    "anchor_item_id": "anchor_001",
    "positives": ["pos_001"],
    "negatives": {
      "easy": ["neg_easy_001"],
      "medium": [],
      "hard_lexical": ["neg_hard_001"],
      "hard_embedding": ["neg_hard_002"]
    }
  },

  "contrastive_targets": {
    "margins": {
      "positive": 0.05,
      "easy_negative": 0.20,
      "medium_negative": 0.10,
      "hard_negative": 0.03
    }
  },

  "provenance": {
    "generator": "bundle_generator_v2",
    "generation_timestamp": "2026-01-04T12:00:00Z",
    "sense_inventory_version": "wordnet-3.1+custom",
    "quality_score": 0.92
  }
}
```

---

## 4. Quality Gates

### Gate 1: Raw Event Acceptance
- [ ] `context_len_tokens >= 15` (reject short contexts)
- [ ] `context_len_tokens <= 50` (reject overly long)
- [ ] `has_strong_disambiguator == true` (at least one non-target disambiguator)
- [ ] `is_definition_like == false` (not a dictionary definition)
- [ ] `is_template_like == false` (not "In terms of X, ..." pattern)
- [ ] `len(cue_hits) >= 1` (at least one cue word present)
- [ ] `len(anti_cue_hits) == 0` (no anti-cue words for this sense)

### Gate 2: Bundle Validity
- [ ] Anchor and positive share `sense_id`
- [ ] Anchor and positive come from different `source_event_id` (no near-duplicates)
- [ ] Hard negative has different `sense_id`
- [ ] Hard negative is not definition-like
- [ ] Hard negative has either:
  - `lexical_overlap > 0.3` (lexical confusability), OR
  - `embedding_similarity > 0.7` (embedding confusability)
- [ ] For hard negatives: `len(anti_cue_hits) >= 1` OR conflicting cue present

### Gate 3: Bundle Diversity
- [ ] At least 2 different context styles across items
- [ ] At least 1 easy negative and 1 hard negative
- [ ] Positive context differs from anchor by >= 3 non-stopword tokens

---

## 5. Context Style Categories

Each context should be tagged with one of:

| Style | Description | Example Pattern |
|-------|-------------|-----------------|
| `news` | News article prose | "The bank announced..." |
| `narrative` | Story/fiction | "She walked into the bank..." |
| `instructional` | How-to/educational | "To open an account, visit your local bank..." |
| `dialogue` | Conversation/quotes | "'I need to go to the bank,' she said." |
| `technical` | Domain-specific docs | "The bank's liquidity ratio..." |
| `forum` | Informal/social | "anyone know if the bank is open on sunday?" |
| `legal` | Formal/legal language | "The bank shall provide notice..." |

Target distribution: No single style > 40% of contexts.

---

## 6. Adversarial Negative Mining

### Method 1: Lexical Confusability
1. For each sense pair (A, B), compute shared collocates
2. Find contexts for sense B that contain collocates typical of sense A
3. Score by: `lexical_overlap = |shared_ngrams| / |total_ngrams|`
4. Accept as hard negative if `lexical_overlap > 0.3`

### Method 2: Embedding Confusability
1. Embed all contexts using frozen encoder (or yesterday's checkpoint)
2. For each anchor context, find k-nearest neighbors
3. Filter to neighbors with different `sense_id`
4. Score by cosine similarity
5. Accept as hard negative if `similarity > 0.7`

### Method 3: Retrieval Adversarial
1. Use anchor as query against corpus of all sense contexts
2. Find top-k retrievals with wrong sense
3. These are the "model's actual confusions"
4. Prioritize for hard negative inclusion

---

## 7. Target Metrics for v2 Dataset

| Metric | Current (v1) | Target (v2) |
|--------|--------------|-------------|
| Lemmas | 263 | 1,000+ |
| Sense IDs | 778 | 3,000+ |
| Total bundles | 15,780 | 50,000+ |
| Total items | 86,073 | 300,000+ |
| Median context length | 6 tokens | 25 tokens |
| Context length range | 5-11 tokens | 15-40 tokens |
| Cue phrase diversity | 2 patterns | 10+ patterns |
| Context styles | 1 (synthetic) | 6+ (natural) |
| Hard negatives per bundle | 1 | 2-3 |

---

## 8. Pipeline Stages

```
┌─────────────────┐
│ Sense Inventory │ ← WordNet + Wiktionary + custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Raw Event       │ ← Wikipedia, news, books, web scrape
│ Scraper         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quality Gate 1  │ ← Filter raw events
│ (Event Filter)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Negative Miner  │ ← Lexical + embedding adversarial
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bundle Builder  │ ← Assemble anchor/pos/neg
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quality Gate 2  │ ← Validate bundles
│ (Bundle Filter) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Final Dataset   │ → bundles_v2.jsonl
└─────────────────┘
```

---

*Spec Version: 2.0*
*Date: 2026-01-04*
