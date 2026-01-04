"""
Evaluation Integrity Gate
=========================

Ensures eval items are meaningful measurements, not noisy traps.

Gates:
1. query_realism_gate: Query must come from sense-in-use sentence, not disambig surface
2. gold_alignment_gate: Correct passages must share sense_id lineage with query
3. context_richness_gate: Query must have sufficient disambiguating cues

Actions:
- PASS: Item is valid for eval
- QUARANTINE: Item has data bugs, exclude from eval
- REGEN: Item needs context regeneration (synthetic minimal pair)
"""

import json
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum


class GateStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


class ItemAction(Enum):
    KEEP = "keep"
    QUARANTINE = "quarantine"
    REGEN = "needs_context_regen"


@dataclass
class IntegrityResult:
    """Result of integrity check on a single eval item."""
    eval_id: str
    lemma: str
    query_realism: GateStatus
    gold_alignment: GateStatus
    context_richness: GateStatus
    action: ItemAction
    issues: List[str]
    suggestions: List[str]


class EvalIntegrityGate:
    """Validates and filters eval items for quality."""

    # Patterns that indicate disambig/list-style content
    DISAMBIG_PATTERNS = [
        r'may also refer to',
        r'see also',
        r'\(disambiguation\)',
        r'==\s*See also\s*==',
        r'==\s*References\s*==',
        r'\|\s*==',
        r'All pages with titles',
        r'Topics referred to by the same term',
    ]

    # Minimum cue token requirements by tier
    CUE_REQUIREMENTS = {
        'high_cue': 3,
        'medium_cue': 2,
        'low_cue': 1
    }

    def __init__(self):
        self.disambig_regex = re.compile(
            '|'.join(self.DISAMBIG_PATTERNS),
            re.IGNORECASE
        )

    def check_query_realism(self, item: Dict) -> Tuple[GateStatus, List[str]]:
        """
        Check if query is a realistic sense-in-use sentence.
        Fails if query looks like disambig/list content.
        """
        issues = []
        query_text = item.get('query', {}).get('text', '')

        # Check for disambig patterns
        if self.disambig_regex.search(query_text):
            issues.append("Query contains disambiguation patterns")
            return GateStatus.FAIL, issues

        # Check for list-like structure (multiple == sections)
        if query_text.count('==') >= 2:
            issues.append("Query contains wiki section markers")
            return GateStatus.FAIL, issues

        # Check for excessive proper noun density (list of names)
        words = query_text.split()
        if len(words) > 10:
            caps = sum(1 for w in words if w[0].isupper() if w)
            if caps / len(words) > 0.5:
                issues.append("Query has high proper noun density (likely a list)")
                return GateStatus.WARN, issues

        # Check minimum sentence structure
        if len(query_text) < 30:
            issues.append("Query too short for meaningful context")
            return GateStatus.WARN, issues

        if not any(c in query_text for c in '.!?'):
            issues.append("Query lacks sentence-ending punctuation")
            return GateStatus.WARN, issues

        return GateStatus.PASS, issues

    def check_gold_alignment(self, item: Dict) -> Tuple[GateStatus, List[str]]:
        """
        Check if correct passages share sense lineage with query.
        """
        issues = []
        query_sense = item.get('query', {}).get('sense_id', '')
        candidates = item.get('candidates', [])
        correct_ids = set(item.get('correct_passage_ids', []))

        if not query_sense:
            issues.append("Query has no sense_id")
            return GateStatus.FAIL, issues

        # Parse query sense components
        query_parts = self._parse_sense_id(query_sense)

        # Check each correct passage
        misaligned = []
        for cand in candidates:
            if cand.get('passage_id') in correct_ids:
                cand_sense = cand.get('sense_id', '')
                cand_parts = self._parse_sense_id(cand_sense)

                # Check lineage match
                if query_parts['lemma'] != cand_parts['lemma']:
                    misaligned.append(f"Lemma mismatch: {query_parts['lemma']} vs {cand_parts['lemma']}")
                elif query_parts['source'] != cand_parts['source']:
                    # Different source is OK if sense key matches
                    if query_parts['sense_key'] != cand_parts['sense_key']:
                        misaligned.append(f"Cross-source sense mismatch: {query_sense} vs {cand_sense}")

        if misaligned:
            issues.extend(misaligned)
            return GateStatus.FAIL, issues

        # Check for suspiciously many correct passages (multi-label ambiguity)
        if len(correct_ids) > 5:
            issues.append(f"Too many correct passages ({len(correct_ids)}) - query may be under-specified")
            return GateStatus.WARN, issues

        return GateStatus.PASS, issues

    def check_context_richness(self, item: Dict) -> Tuple[GateStatus, List[str]]:
        """
        Check if query has sufficient disambiguating cues.
        """
        issues = []
        query = item.get('query', {})
        cue_tier = query.get('cue_tier', 'medium_cue')
        cue_strength = query.get('cue_strength', 0.5)

        # Check cue strength
        if cue_strength < 0.3:
            issues.append(f"Very low cue strength ({cue_strength:.2f})")
            return GateStatus.FAIL, issues

        # Check if query text contains the lemma
        query_text = query.get('text', '').lower()
        lemma = item.get('lemma', '').lower()

        if lemma and lemma not in query_text:
            issues.append("Lemma not found in query text")
            return GateStatus.FAIL, issues

        # Check for minimal disambiguating context
        span = query.get('span', {})
        left_context = query_text[:span.get('start', 0)]
        right_context = query_text[span.get('end', len(query_text)):]

        total_context = len(left_context.split()) + len(right_context.split())
        if total_context < 5:
            issues.append(f"Minimal context around lemma ({total_context} words)")
            return GateStatus.WARN, issues

        return GateStatus.PASS, issues

    def _parse_sense_id(self, sense_id: str) -> Dict:
        """Parse sense_id into components."""
        # Format: source:lemma.pos.sense_key
        result = {'source': '', 'lemma': '', 'pos': '', 'sense_key': '', 'full': sense_id}

        if ':' in sense_id:
            parts = sense_id.split(':', 1)
            result['source'] = parts[0]
            rest = parts[1]
        else:
            rest = sense_id

        subparts = rest.split('.')
        if len(subparts) >= 1:
            result['lemma'] = subparts[0]
        if len(subparts) >= 2:
            result['pos'] = subparts[1]
        if len(subparts) >= 3:
            result['sense_key'] = '.'.join(subparts[2:])

        return result

    def check_item(self, item: Dict) -> IntegrityResult:
        """Run all integrity checks on a single eval item."""
        eval_id = item.get('eval_id', 'unknown')
        lemma = item.get('lemma', '')

        issues = []
        suggestions = []

        # Run gates
        realism_status, realism_issues = self.check_query_realism(item)
        issues.extend(realism_issues)

        alignment_status, alignment_issues = self.check_gold_alignment(item)
        issues.extend(alignment_issues)

        richness_status, richness_issues = self.check_context_richness(item)
        issues.extend(richness_issues)

        # Determine action
        if realism_status == GateStatus.FAIL or alignment_status == GateStatus.FAIL:
            action = ItemAction.QUARANTINE
            suggestions.append("Regenerate query from sense page sentence, not disambig surface")
        elif richness_status == GateStatus.FAIL:
            action = ItemAction.REGEN
            suggestions.append("Generate synthetic minimal-pair context with stronger cues")
        elif realism_status == GateStatus.WARN or alignment_status == GateStatus.WARN or richness_status == GateStatus.WARN:
            action = ItemAction.REGEN
            suggestions.append("Review and potentially regenerate with better cues")
        else:
            action = ItemAction.KEEP

        return IntegrityResult(
            eval_id=eval_id,
            lemma=lemma,
            query_realism=realism_status,
            gold_alignment=alignment_status,
            context_richness=richness_status,
            action=action,
            issues=issues,
            suggestions=suggestions
        )

    def process_eval_file(self, input_file: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process an eval file and split into keep/quarantine/regen buckets.

        Returns:
            (valid_items, quarantined_items, needs_regen_items)
        """
        valid = []
        quarantined = []
        needs_regen = []

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)
                result = self.check_item(item)

                # Add integrity metadata to item
                item['integrity'] = {
                    'query_realism_gate': result.query_realism.value,
                    'gold_alignment_gate': result.gold_alignment.value,
                    'context_richness_gate': result.context_richness.value,
                    'action': result.action.value,
                    'issues': result.issues,
                    'suggestions': result.suggestions
                }

                if result.action == ItemAction.KEEP:
                    valid.append(item)
                elif result.action == ItemAction.QUARANTINE:
                    quarantined.append(item)
                else:
                    needs_regen.append(item)

        return valid, quarantined, needs_regen


class MinimalPairGenerator:
    """
    Generates minimal-pair synthetic contexts for items that need regeneration.
    """

    # Domain-specific cue templates
    DOMAIN_CUES = {
        'game_engine': ['renderer', 'scene', 'GameObject', 'runtime', 'shader', 'Unity editor'],
        'abstract_concept': ['cohesion', 'solidarity', 'purpose', 'agreement', 'togetherness'],
        'video_game': ['player', 'score', 'level', 'game', 'Nokia', 'mobile', 'screen'],
        'reptile': ['scales', 'venom', 'slither', 'species', 'habitat', 'cold-blooded'],
        'finance': ['stock', 'tax', 'loss', 'investment', 'IRS', 'deduction', 'securities'],
        'geography': ['bay', 'coast', 'estuary', 'tidal', 'England', 'shore'],
        'medical': ['muscle', 'tissue', 'organ', 'patient', 'diagnosis', 'treatment'],
        'technology': ['data', 'algorithm', 'system', 'processor', 'memory', 'code'],
        'legal': ['court', 'ruling', 'statute', 'defendant', 'plaintiff', 'jurisdiction'],
    }

    SENTENCE_TEMPLATES = [
        "The {domain_adj} {lemma} in this context refers to {gloss_fragment}.",
        "When discussing {domain_topic}, the term {lemma} specifically means {gloss_fragment}.",
        "In {domain_field}, a {lemma} is defined as {gloss_fragment}.",
        "The {lemma} mentioned here relates to {domain_topic}, indicating {gloss_fragment}.",
    ]

    def generate_synthetic_query(self, item: Dict, sense_inventory: Dict = None) -> Dict:
        """
        Generate a synthetic minimal-pair query for an item that needs regeneration.
        """
        lemma = item.get('lemma', '')
        query_sense = item.get('query', {}).get('sense_id', '')
        sense_gloss = item.get('query', {}).get('sense_gloss', '')

        # Infer domain from sense_id
        domain = self._infer_domain(query_sense, sense_gloss)
        cues = self.DOMAIN_CUES.get(domain, [])

        # Generate synthetic sentence
        import random
        template = random.choice(self.SENTENCE_TEMPLATES)

        # Extract gloss fragment
        gloss_fragment = sense_gloss[:50] if sense_gloss else "this particular meaning"

        synthetic_text = template.format(
            lemma=lemma,
            domain_adj=domain.replace('_', ' '),
            domain_topic=domain.replace('_', ' '),
            domain_field=domain.replace('_', ' '),
            gloss_fragment=gloss_fragment
        )

        # Find lemma position in synthetic text
        lemma_start = synthetic_text.lower().find(lemma.lower())
        if lemma_start == -1:
            lemma_start = 0
        lemma_end = lemma_start + len(lemma)

        new_query = {
            'text': synthetic_text,
            'span': {
                'start': lemma_start,
                'end': lemma_end,
                'surface': lemma
            },
            'sense_id': query_sense,
            'sense_gloss': sense_gloss,
            'cue_strength': 0.85,  # Synthetic contexts are designed to be cue-rich
            'cue_tier': 'high_cue',
            'provenance': {
                'source_type': 'synthetic_minimal_pair',
                'domain_cues_used': cues[:3],
                'template_used': template
            }
        }

        return new_query

    def _infer_domain(self, sense_id: str, gloss: str) -> str:
        """Infer the domain from sense_id and gloss."""
        combined = (sense_id + ' ' + gloss).lower()

        domain_keywords = {
            'game_engine': ['unity', 'engine', 'game development', 'gameobject'],
            'video_game': ['game', 'player', 'nokia', 'mobile game', 'video game'],
            'reptile': ['snake', 'reptile', 'venom', 'scales', 'animal'],
            'finance': ['stock', 'tax', 'wash sale', 'investment', 'securities', 'loss'],
            'geography': ['bay', 'coast', 'wash', 'estuary', 'england'],
            'medical': ['muscle', 'organ', 'medical', 'patient', 'health'],
            'technology': ['data', 'code', 'algorithm', 'software', 'computer'],
            'legal': ['court', 'law', 'legal', 'statute', 'ruling'],
            'abstract_concept': ['unity', 'cohesion', 'concept', 'state of being'],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in combined for kw in keywords):
                return domain

        return 'general'


def run_integrity_gate(eval_file: Path, output_dir: Path = None) -> Dict:
    """Run the integrity gate on an eval file."""
    print("=" * 70)
    print("EVALUATION INTEGRITY GATE")
    print("=" * 70)

    gate = EvalIntegrityGate()
    generator = MinimalPairGenerator()

    print(f"\nProcessing: {eval_file}")
    valid, quarantined, needs_regen = gate.process_eval_file(eval_file)

    print(f"\n--- Results ---")
    print(f"  Valid (KEEP):       {len(valid)}")
    print(f"  Quarantined:        {len(quarantined)}")
    print(f"  Needs Regeneration: {len(needs_regen)}")

    # Analyze issues
    issue_counts = defaultdict(int)
    for item in quarantined + needs_regen:
        for issue in item.get('integrity', {}).get('issues', []):
            issue_counts[issue] += 1

    print(f"\n--- Top Issues ---")
    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:4d}: {issue}")

    # Generate synthetic queries for needs_regen items
    print(f"\n--- Generating Synthetic Contexts ---")
    regenerated = []
    for item in needs_regen:
        new_query = generator.generate_synthetic_query(item)
        item['query'] = new_query
        item['integrity']['regenerated'] = True
        regenerated.append(item)
    print(f"  Generated {len(regenerated)} synthetic minimal-pair queries")

    # Save outputs
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Valid items
        valid_file = output_dir / "eval_valid.jsonl"
        with open(valid_file, 'w', encoding='utf-8') as f:
            for item in valid:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\n  Saved valid items: {valid_file}")

        # Quarantined items
        quarantine_file = output_dir / "eval_quarantined.jsonl"
        with open(quarantine_file, 'w', encoding='utf-8') as f:
            for item in quarantined:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  Saved quarantined items: {quarantine_file}")

        # Regenerated items
        regen_file = output_dir / "eval_regenerated.jsonl"
        with open(regen_file, 'w', encoding='utf-8') as f:
            for item in regenerated:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  Saved regenerated items: {regen_file}")

        # Combined clean eval (valid + regenerated)
        clean_file = output_dir / "eval_clean.jsonl"
        clean_items = valid + regenerated
        with open(clean_file, 'w', encoding='utf-8') as f:
            for item in clean_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  Saved clean eval set: {clean_file} ({len(clean_items)} items)")

    stats = {
        'input_file': str(eval_file),
        'total_items': len(valid) + len(quarantined) + len(needs_regen),
        'valid': len(valid),
        'quarantined': len(quarantined),
        'needs_regen': len(needs_regen),
        'regenerated': len(regenerated),
        'clean_total': len(valid) + len(regenerated),
        'top_issues': dict(sorted(issue_counts.items(), key=lambda x: -x[1])[:10])
    }

    return stats


if __name__ == "__main__":
    eval_dir = Path("paradigm_factory/v2/evals")
    output_dir = Path("paradigm_factory/v2/evals_gated")

    # Process retrieval eval
    stats = run_integrity_gate(
        eval_file=eval_dir / "eval_multi_sense_retrieval.jsonl",
        output_dir=output_dir
    )

    print("\n" + "=" * 70)
    print("INTEGRITY GATE COMPLETE")
    print("=" * 70)
    print(f"\n  Clean eval items: {stats['clean_total']}/{stats['total_items']}")
    print(f"  Quarantine rate: {100*stats['quarantined']/stats['total_items']:.1f}%")
