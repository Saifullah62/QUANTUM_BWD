#!/usr/bin/env python3
"""
Killer Enhancement Pipeline
============================

Deterministic augmentation pass for killer items.
Applies rules from killer_rewrite_rules_v1.yaml.

Actions:
- regenerate_query: Generate new query from in-page sentence
- quarantine: Block item (never regenerate silently)
- add_minimal_pairs: Generate 2-3 query variants with single cue injection
- flag_for_review: Mark for manual review

Usage:
  python paradigm_factory/v2/killers/enhance_killers.py --input killers.jsonl --output enhanced/
"""

import json
import yaml
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ActionType(Enum):
    QUARANTINE = "quarantine"
    REGENERATE_QUERY = "regenerate_query"
    ADD_MINIMAL_PAIRS = "add_minimal_pairs"
    FLAG_FOR_REVIEW = "flag_for_review"
    KEEP = "keep"


@dataclass
class EnhancementResult:
    """Result of applying rules to a killer item."""
    original_id: str
    action: ActionType
    rule_id: str
    reason: str
    items: List[Dict]  # Enhanced items (may be multiple for minimal pairs)
    audit: Dict


class KillerEnhancer:
    """Applies rewrite rules to killer items."""

    def __init__(self, rules_file: Path):
        self.rules_file = rules_file
        self.rules = self._load_rules()
        self.cue_phrases = self.rules.get('cue_phrases', {})

    def _load_rules(self) -> Dict:
        """Load rules from YAML file."""
        with open(self.rules_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _check_condition(self, item: Dict, condition: Dict) -> bool:
        """Check if an item matches a rule condition."""
        query = item.get('query', {})
        query_text = query.get('text', '')

        # query_contains_any
        if 'query_contains_any' in condition:
            patterns = condition['query_contains_any']
            if any(p.lower() in query_text.lower() for p in patterns):
                return True

        # query_matches_pattern
        if 'query_matches_pattern' in condition:
            pattern = condition['query_matches_pattern']
            if re.search(pattern, query_text):
                return True

        # cue_strength_below
        if 'cue_strength_below' in condition:
            threshold = condition['cue_strength_below']
            cue_strength = query.get('cue_strength', 0.5)
            if cue_strength < threshold:
                # Also check has_valid_sense_page if required
                if condition.get('has_valid_sense_page', False):
                    provenance = query.get('provenance', {})
                    if provenance.get('source_page'):
                        return True
                else:
                    return True

        # cue_strength_above (for minimal pairs)
        if 'cue_strength_above' in condition:
            threshold = condition['cue_strength_above']
            cue_strength = query.get('cue_strength', 0.5)
            if cue_strength <= threshold:
                return False
            # Check reason_code if also specified
            if 'reason_code' in condition:
                killer_meta = item.get('killer_metadata', {})
                if killer_meta.get('reason_code') != condition['reason_code']:
                    return False
            return True

        # reason_code alone
        if 'reason_code' in condition and 'cue_strength_above' not in condition:
            killer_meta = item.get('killer_metadata', {})
            if killer_meta.get('reason_code') == condition['reason_code']:
                return True

        # gold_alignment_failed
        if 'gold_alignment_failed' in condition:
            integrity = item.get('killer_metadata', {}).get('integrity', {})
            if integrity.get('gold_alignment_gate') == 'fail':
                return True

        # margin_below
        if 'margin_below' in condition:
            # Would need margin from scoring - check if present
            margin = item.get('margin', item.get('margin_at_decision', 1.0))
            if margin < condition['margin_below']:
                return True

        # query_length_below
        if 'query_length_below' in condition:
            if len(query_text) < condition['query_length_below']:
                return True

        return False

    def _apply_action(self, item: Dict, rule: Dict) -> EnhancementResult:
        """Apply a rule's action to an item."""
        action_str = rule['action']
        rule_id = rule['id']
        reason = rule.get('reason', rule_id)

        audit = {
            'timestamp': datetime.now().isoformat(),
            'rule_id': rule_id,
            'rule_version': self.rules.get('version', 'unknown'),
            'original_item': item
        }

        if action_str == 'quarantine':
            return EnhancementResult(
                original_id=item.get('eval_id', 'unknown'),
                action=ActionType.QUARANTINE,
                rule_id=rule_id,
                reason=reason,
                items=[],  # Quarantined items produce no output
                audit=audit
            )

        elif action_str == 'regenerate_query':
            # Generate new query from sense page
            regenerated = self._regenerate_query(item, rule)
            return EnhancementResult(
                original_id=item.get('eval_id', 'unknown'),
                action=ActionType.REGENERATE_QUERY,
                rule_id=rule_id,
                reason=reason,
                items=[regenerated] if regenerated else [],
                audit=audit
            )

        elif action_str == 'add_minimal_pairs':
            # Generate query variants with cue injection
            num_variants = rule.get('num_variants', 3)
            variants = self._generate_minimal_pairs(item, num_variants)
            return EnhancementResult(
                original_id=item.get('eval_id', 'unknown'),
                action=ActionType.ADD_MINIMAL_PAIRS,
                rule_id=rule_id,
                reason=reason,
                items=variants,
                audit=audit
            )

        elif action_str == 'flag_for_review':
            # Keep item but mark for review
            flagged = item.copy()
            flagged['_review'] = {
                'flagged_by': rule_id,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            return EnhancementResult(
                original_id=item.get('eval_id', 'unknown'),
                action=ActionType.FLAG_FOR_REVIEW,
                rule_id=rule_id,
                reason=reason,
                items=[flagged],
                audit=audit
            )

        else:
            # Unknown action, keep as-is
            return EnhancementResult(
                original_id=item.get('eval_id', 'unknown'),
                action=ActionType.KEEP,
                rule_id='none',
                reason='no_matching_rule',
                items=[item],
                audit=audit
            )

    def _regenerate_query(self, item: Dict, rule: Dict) -> Optional[Dict]:
        """Regenerate query from sense page sentence."""
        # In a real implementation, this would:
        # 1. Look up the sense page from provenance
        # 2. Extract a natural sentence containing the lemma
        # 3. Replace the query text
        # For now, mark for manual regeneration
        regenerated = item.copy()
        regenerated['_regen'] = {
            'source': rule.get('source', 'sense_page_sentence'),
            'status': 'pending',
            'original_query': item.get('query', {}).get('text', ''),
            'timestamp': datetime.now().isoformat()
        }
        regenerated['eval_id'] = f"{item.get('eval_id', 'unknown')}_regen"
        return regenerated

    def _generate_minimal_pairs(self, item: Dict, num_variants: int) -> List[Dict]:
        """Generate query variants with single cue injection."""
        variants = []
        query = item.get('query', {})
        sense_id = query.get('sense_id', '')

        # Find cue phrases for this sense type
        sense_type = sense_id.split('#')[-1] if '#' in sense_id else 'unknown'
        cues = self.cue_phrases.get(sense_type, [])

        if not cues:
            # No cues defined for this sense type
            return [item]

        # Generate variants by injecting cues
        original_text = query.get('text', '')

        for i, cue in enumerate(cues[:num_variants]):
            variant = item.copy()
            variant['query'] = query.copy()

            # Simple cue injection: prepend cue phrase
            # In production, would use smarter insertion
            variant['query']['text'] = f"[{cue}] {original_text}"
            variant['query']['injected_cue'] = cue
            variant['eval_id'] = f"{item.get('eval_id', 'unknown')}_mp{i+1}"
            variant['_minimal_pair'] = {
                'parent_id': item.get('eval_id', 'unknown'),
                'variant_index': i + 1,
                'cue_injected': cue
            }
            variants.append(variant)

        return variants

    def enhance_item(self, item: Dict) -> EnhancementResult:
        """Apply rules to a single item."""
        # Sort rules by priority
        rules = sorted(
            self.rules.get('rules', []),
            key=lambda r: r.get('priority', 50)
        )

        # Find first matching rule
        for rule in rules:
            condition = rule.get('condition', {})
            if self._check_condition(item, condition):
                return self._apply_action(item, rule)

        # No rule matched, keep as-is
        return EnhancementResult(
            original_id=item.get('eval_id', 'unknown'),
            action=ActionType.KEEP,
            rule_id='none',
            reason='no_matching_rule',
            items=[item],
            audit={'timestamp': datetime.now().isoformat(), 'original_item': item}
        )

    def enhance_file(self, input_file: Path, output_dir: Path) -> Dict:
        """Enhance all items in a file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load items
        items = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))

        # Process items
        enhanced = []
        quarantined = []
        flagged = []
        audit_log = []

        stats = {
            'total': len(items),
            'kept': 0,
            'quarantined': 0,
            'regenerated': 0,
            'minimal_pairs_added': 0,
            'flagged': 0
        }

        for item in items:
            result = self.enhance_item(item)
            audit_log.append({
                'original_id': result.original_id,
                'action': result.action.value,
                'rule_id': result.rule_id,
                'reason': result.reason,
                'output_count': len(result.items)
            })

            if result.action == ActionType.QUARANTINE:
                quarantined.append(result.audit['original_item'])
                stats['quarantined'] += 1
            elif result.action == ActionType.REGENERATE_QUERY:
                enhanced.extend(result.items)
                stats['regenerated'] += 1
            elif result.action == ActionType.ADD_MINIMAL_PAIRS:
                enhanced.extend(result.items)
                stats['minimal_pairs_added'] += len(result.items)
            elif result.action == ActionType.FLAG_FOR_REVIEW:
                flagged.extend(result.items)
                stats['flagged'] += 1
            else:
                enhanced.extend(result.items)
                stats['kept'] += 1

        # Write outputs
        enhanced_file = output_dir / "enhanced.jsonl"
        quarantined_file = output_dir / "quarantined.jsonl"
        flagged_file = output_dir / "flagged_for_review.jsonl"
        audit_file = output_dir / "enhancement_audit.jsonl"

        for filepath, data in [
            (enhanced_file, enhanced),
            (quarantined_file, quarantined),
            (flagged_file, flagged)
        ]:
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        with open(audit_file, 'w', encoding='utf-8') as f:
            for entry in audit_log:
                f.write(json.dumps(entry) + '\n')

        # Write summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_file),
            'rules_version': self.rules.get('version', 'unknown'),
            'stats': stats,
            'output_files': {
                'enhanced': str(enhanced_file),
                'quarantined': str(quarantined_file),
                'flagged': str(flagged_file),
                'audit': str(audit_file)
            }
        }

        summary_file = output_dir / "enhancement_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        return summary


def main():
    parser = argparse.ArgumentParser(description="Enhance killer items")
    parser.add_argument("--input", type=Path, required=True, help="Input killers JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--rules", type=Path,
                       default=Path(__file__).parent / "killer_rewrite_rules_v1.yaml",
                       help="Rules YAML file")
    args = parser.parse_args()

    print("=" * 60)
    print("KILLER ENHANCEMENT PIPELINE")
    print("=" * 60)

    enhancer = KillerEnhancer(args.rules)
    summary = enhancer.enhance_file(args.input, args.output)

    print(f"\nInput: {args.input}")
    print(f"Rules: {args.rules} (v{enhancer.rules.get('version', '?')})")
    print(f"\nStats:")
    for key, value in summary['stats'].items():
        print(f"  {key}: {value}")
    print(f"\nOutputs written to: {args.output}")


if __name__ == "__main__":
    main()
