# Killer Negative Dossier

*Generated: 2026-01-03*
*Based on: v3 validation run, seed 42*

## Executive Summary

**115 killer negatives tracked**, of which:
- **53 (46%)** have inverted ordering (hard_neg MORE similar than positive)
- **35 (30%)** are near-misses (0 < gap < 0.05)
- **27 (24%)** have adequate gap but were worst in batch

## Critical Finding: Synthetic Context Problem

The training contexts are **template-generated** with nonsensical verbs:

| Template Verb | Frequency | Example |
|---------------|-----------|---------|
| evolved | High | "That lean evolved every occasion." |
| coalesced | High | "Near the party, the situation coalesced." |
| bifurcated | High | "The form was ineffable and bifurcated thoroughly." |
| oscillated | High | "The ostensible form seemed to oscillated thoroughly." |
| manifested | High | "After the play, everything manifested slowly." |
| attenuated | High | "Near the form, the approach attenuated." |
| fluctuated | High | "The pick was remarkable and fluctuated thoroughly." |

**Implication:** The contexts don't provide meaningful disambiguation signal. The model can only rely on:
1. Prepositional phrases ("In terms of battery", "Regarding the political")
2. Domain keywords when present
3. Syntactic frame (though often broken grammar)

## Top 15 Inversions (Full Context)

### 1. party#political_group vs party#participant
**Gap: -0.193** (worst inversion)

| Role | Sense | Context |
|------|-------|---------|
| Anchor | political_group | "Everyone persisted the remarkable party." |
| Positive | political_group | "Regarding the political, they evolved my party thoroughly." |
| Hard Neg | participant | "Near the party, the situation coalesced." |

**Analysis:** The positive has "Regarding the political" which should disambiguate, but the anchor has no clear signal. Hard negative's context is equally vague.

---

### 2. grade#quality_level vs grade#school_year
**Gap: -0.189**

| Role | Sense | Context |
|------|-------|---------|
| Anchor | quality_level | "The bright grade seemed to was carefully." |
| Positive | quality_level | "The bright grade belonged to the mechanism." |
| Hard Neg | school_year | "I found the grade to be inchoate." |

**Analysis:** Broken grammar in anchor ("seemed to was"). No disambiguation signal in any context.

---

### 3. print#to_reproduce_text vs print#pattern
**Gap: -0.164**

| Role | Sense | Context |
|------|-------|---------|
| Anchor | to_reproduce_text | "The peculiar print belonged to the structure." |
| Positive | to_reproduce_text | "She evolved my print suddenly." |
| Hard Neg | pattern | "The method bifurcated a liminal print." |

**Analysis:** No semantic content distinguishes these. Pure template noise.

---

### 4. lean#to_incline vs lean#meat_without_fat
**Gap: -0.159**

| Role | Sense | Context |
|------|-------|---------|
| Anchor | to_incline | "That lean evolved every occasion." |
| Positive | to_incline | "After the lean, everything manifested gradually." |
| Hard Neg | meat_without_fat | "Regarding the meat, the lean was ineffable and oscillated slowly." |

**Analysis:** Hard negative has "Regarding the meat" which is the ONLY contextual signal in the bundle. This signal points to the wrong class! The anchor/positive have no disambiguation.

---

### 5. gum#mouth_tissue vs gum#tree_resin
**Gap: -0.142**

| Role | Sense | Context |
|------|-------|---------|
| Anchor | mouth_tissue | "Regarding the mouth, that gum moved every time." |
| Positive | mouth_tissue | "The method's gum was old." |
| Hard Neg | tree_resin | "The approach required a liminal gum." |

**Analysis:** Anchor has "Regarding the mouth" but positive lacks it. Hard neg is neutral.

---

### 6. leave#permission vs leave#foliage
**Gap: -0.113**

| Role | Sense | Context |
|------|-------|---------|
| Anchor | permission | "Everyone fluctuated the unexpected leave." |
| Positive | permission | "Speaking of permissions, everyone transformed the familiar leave." |
| Hard Neg | foliage | "Everyone coalesced the ineffable leave." |

**Analysis:** Positive has "Speaking of permissions" — good signal! But anchor lacks any cue. Surface overlap "everyone" and "leave" appears in both anchor and hard neg.

---

### 7-15. [See full logs]

---

## Pattern Analysis

### Failure Categories

| Category | Count | % | Root Cause |
|----------|-------|---|------------|
| No disambiguation in anchor | 35 | 66% | Template generator didn't include sense keywords |
| Signal in wrong item | 8 | 15% | Hard neg has disambiguation phrase, anchor doesn't |
| Broken grammar | 5 | 9% | Template errors break syntax |
| Genuinely ambiguous | 5 | 9% | Senses too close without domain context |

### Most Problematic Lemmas

| Lemma | Inversions | Near-Misses | Total |
|-------|------------|-------------|-------|
| monitor | 2 | 2 | 4 |
| pick | 2 | 2 | 4 |
| host | 2 | 2 | 4 |
| party | 1 | 2 | 3 |
| present | 1 | 2 | 3 |
| hang | 1 | 2 | 3 |

### Surface Cue Overlap

Many inversions share surface tokens between anchor and hard negative:
- `grade` appears in all items for grade bundles
- `leave` appears in all items for leave bundles
- Template words ("everyone", "the", "carefully") create false similarity

---

## Recommendations

### 1. Context Quality (Data Fix)
- Regenerate bundles with more meaningful templates
- Ensure disambiguation phrases appear in anchors, not just positives
- Fix broken grammar issues

### 2. Attentive Pooling (Architecture Fix)
- Add SenseHead that learns to attend to disambiguating tokens
- Train to ignore template verbs ("evolved", "coalesced")
- Focus on prepositional phrases and domain keywords

### 3. Curriculum (Training Fix)
- Phase 1: Only train on bundles where anchor has clear disambiguation
- Phase 2: Add near-misses where positive has clear signal
- Phase 3: Attack inversions with multi-prototype or exclude

### 4. Bundle Filtering (Immediate)
- Flag bundles where hard_neg has better disambiguation than anchor
- These are "inverted signal" bundles that poison training
- Either fix or exclude from certification target

---

## Appendix: Template Verb Frequencies

```
the: 6272
of: 1107
to: 979
was: 898
near: 643
seemed: 441
belonged: 427
speaking: 405
regarding: 368
in terms of: 355
because: 347
after: 339
everything: 339
evolved: ~200
coalesced: ~150
bifurcated: ~100
oscillated: ~100
manifested: ~100
attenuated: ~100
fluctuated: ~100
```

The high frequency of these template verbs confirms synthetic generation. They carry no semantic content.
