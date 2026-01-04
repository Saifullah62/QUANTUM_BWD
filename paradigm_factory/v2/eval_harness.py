"""
Evaluation Harness & Autopilot Dashboard
=========================================

Scoring signals:
1. Multi-Sense Retrieval: Top-k hit rate, best-correct rank, margin at decision
2. Multi-Step Coherence: per-step match, transition P/R/F1, final-state accuracy

Autopilot Dashboard:
- Section A: Gates (canary pass, slack pass, integrity preflight)
- Section B: Retrieval metrics by tier (real vs synthetic split)
- Section C: Coherence metrics by shift count
- Section D: Top 20 new killers with reason codes

Reason Codes:
- UNDERDETERMINED_CONTEXT: weak evidence tokens
- WITHIN_LEMMA_COLLISION: same lemma, same POS confusion
- CROSS_POS_COLLISION: same lemma, different POS confusion
- CROSS_DOMAIN_CONFUSABLE: topic tag overlap with wrong sense
- STYLE_MISMATCH: positive stylistically unlike anchor

Integrity Gate Integration:
- Runs as mandatory preflight before scoring
- Validates gold file hash for measurement identity
- Separates real-usage vs synthetic eval sets in output
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum

# Import integrity gate
try:
    from paradigm_factory.v2.eval_integrity_gate import EvalIntegrityGate, ItemAction as IntegrityAction
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from paradigm_factory.v2.eval_integrity_gate import EvalIntegrityGate, ItemAction as IntegrityAction

# =============================================================================
# TRUST ANCHORS
# =============================================================================
# Gold file hash (stored in code, not just in file). Prevents silent edits.
EXPECTED_GOLD_HASH = "e1d122539e92e5ddd845549f5a23a0835b996515d4c1b74d97ee3bd4c88247aa"
GOLD_FILE_VERSION = "v1"

# Integrity ruleset version. Increment when changing gate thresholds/filters.
# This makes "what counts as valid" evolution transparent and auditable.
INTEGRITY_RULESET = "v1.0"
INTEGRITY_RULESET_CHANGELOG = {
    "v1.0": "Initial ruleset: quarantine_threshold=0.20, disambig_patterns, gold_alignment_check, cue_strength>=0.3"
}


class ReasonCode(Enum):
    """Failure reason codes for killer mining."""
    UNDERDETERMINED_CONTEXT = "underdetermined_context"
    WITHIN_LEMMA_COLLISION = "within_lemma_collision"
    CROSS_POS_COLLISION = "cross_pos_collision"
    CROSS_DOMAIN_CONFUSABLE = "cross_domain_confusable"
    STYLE_MISMATCH = "style_mismatch"
    UNKNOWN = "unknown"


@dataclass
class IntegrityPreflightResult:
    """Result of the integrity preflight check."""
    passed: bool
    gold_file_valid: bool
    gold_hash_verified: bool
    code_hash_match: bool  # True if actual hash matches EXPECTED_GOLD_HASH constant
    expected_hash: Optional[str]
    actual_hash: Optional[str]
    items_valid: int
    items_quarantined: int
    items_regenerated: int
    quarantine_rate: float
    gold_version: str = GOLD_FILE_VERSION
    error_message: Optional[str] = None

    def fingerprint(self) -> str:
        """Generate a short integrity fingerprint for provenance tracking."""
        hash_short = self.actual_hash[:12] if self.actual_hash else "none"
        return f"gold={self.gold_version}|rules={INTEGRITY_RULESET}|hash={hash_short}|q={self.quarantine_rate:.1%}|r={self.items_regenerated}"


@dataclass
class RetrievalResult:
    """Result from scoring a single retrieval eval item."""
    eval_id: str
    lemma: str
    cue_tier: str
    top_1_hit: bool
    top_3_hit: bool
    top_5_hit: bool
    best_correct_rank: int  # 1-indexed, 999 if no correct found
    margin_at_decision: float  # score gap between top correct and top incorrect
    failure_reason: Optional[str] = None
    wrong_passage_id: Optional[str] = None
    wrong_sense_id: Optional[str] = None


@dataclass
class CoherenceResult:
    """Result from scoring a single coherence eval item."""
    eval_id: str
    lemma: str
    num_steps: int
    num_sense_shifts: int
    per_step_matches: List[bool]
    transitions_expected: int
    transitions_detected: int
    transitions_false_positive: int
    final_state_correct: bool
    failure_reason: Optional[str] = None


@dataclass
class DashboardReport:
    """Complete autopilot dashboard report."""
    timestamp: str
    run_id: str

    # Section A: Gates
    canary_pass: bool
    canary_score: float
    slack_pass: bool
    slack_value: float
    integrity_preflight: Optional[IntegrityPreflightResult] = None

    # Section B: Retrieval (overall)
    retrieval_top1: float = 0.0
    retrieval_top3: float = 0.0
    retrieval_top5: float = 0.0
    retrieval_rank_median: float = 999.0
    retrieval_rank_p90: float = 999.0
    retrieval_by_tier: Dict[str, Dict[str, float]] = field(default_factory=dict)
    retrieval_margin_mean: float = 0.0
    retrieval_margin_p10: float = 0.0  # Low margin = brittle

    # Section B.1: Retrieval - Real Usage Set
    retrieval_real_top1: float = 0.0
    retrieval_real_top3: float = 0.0
    retrieval_real_count: int = 0

    # Section B.2: Retrieval - Synthetic Set
    retrieval_synth_top1: float = 0.0
    retrieval_synth_top3: float = 0.0
    retrieval_synth_count: int = 0

    # Section C: Coherence
    coherence_per_step_accuracy: float = 0.0
    coherence_transition_precision: float = 0.0
    coherence_transition_recall: float = 0.0
    coherence_transition_f1: float = 0.0
    coherence_final_state_accuracy: float = 0.0
    coherence_by_shifts: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Section D: New Killers
    top_killers: List[Dict] = field(default_factory=list)

    # Composite score
    composite_score: float = 0.0


class IntegrityPreflight:
    """Runs integrity checks before evaluation scoring."""

    GOLD_FILE_PATH = Path(__file__).parent / "evals_gated" / "EVAL_INTEGRITY_GOLD_v1.json"

    def __init__(self, gold_file: Path = None):
        self.gold_file = gold_file or self.GOLD_FILE_PATH
        self.gate = EvalIntegrityGate()

    def verify_gold_hash(self) -> Tuple[bool, bool, Optional[str], Optional[str]]:
        """
        Verify the gold file hash matches BOTH:
        1. The hash embedded in the file metadata
        2. The EXPECTED_GOLD_HASH constant in code (trust anchor)

        Returns:
            (file_hash_match, code_hash_match, embedded_hash, actual_hash)
        """
        if not self.gold_file.exists():
            return False, False, None, None

        with open(self.gold_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        embedded_hash = data.get('_metadata', {}).get('content_hash')
        if not embedded_hash:
            return False, False, None, None

        # Compute actual hash of items
        items_json = json.dumps(data['items'], sort_keys=True, separators=(',', ':'))
        actual_hash = hashlib.sha256(items_json.encode('utf-8')).hexdigest()

        # Check both anchors
        file_hash_match = (embedded_hash == actual_hash)
        code_hash_match = (actual_hash == EXPECTED_GOLD_HASH)

        return file_hash_match, code_hash_match, embedded_hash, actual_hash

    def run_preflight(self, eval_items: List[Dict]) -> IntegrityPreflightResult:
        """
        Run full integrity preflight check.

        Args:
            eval_items: List of eval items to validate

        Returns:
            IntegrityPreflightResult with validation outcomes
        """
        # Check gold file
        gold_valid = self.gold_file.exists()
        file_hash_match, code_hash_match, embedded_hash, actual_hash = self.verify_gold_hash()

        if not gold_valid:
            return IntegrityPreflightResult(
                passed=False,
                gold_file_valid=False,
                gold_hash_verified=False,
                code_hash_match=False,
                expected_hash=None,
                actual_hash=None,
                items_valid=0,
                items_quarantined=0,
                items_regenerated=0,
                quarantine_rate=0.0,
                error_message=f"Gold file not found: {self.gold_file}"
            )

        if not file_hash_match:
            return IntegrityPreflightResult(
                passed=False,
                gold_file_valid=True,
                gold_hash_verified=False,
                code_hash_match=code_hash_match,
                expected_hash=embedded_hash,
                actual_hash=actual_hash,
                items_valid=0,
                items_quarantined=0,
                items_regenerated=0,
                quarantine_rate=0.0,
                error_message="Gold file hash mismatch - embedded hash doesn't match content"
            )

        if not code_hash_match:
            return IntegrityPreflightResult(
                passed=False,
                gold_file_valid=True,
                gold_hash_verified=True,
                code_hash_match=False,
                expected_hash=EXPECTED_GOLD_HASH,
                actual_hash=actual_hash,
                items_valid=0,
                items_quarantined=0,
                items_regenerated=0,
                quarantine_rate=0.0,
                error_message=f"TRUST ANCHOR MISMATCH: Gold file hash differs from EXPECTED_GOLD_HASH constant. "
                             f"If intentional, update EXPECTED_GOLD_HASH in eval_harness.py"
            )

        # Run integrity gate on eval items
        valid_count = 0
        quarantine_count = 0
        regen_count = 0

        for item in eval_items:
            result = self.gate.check_item(item)
            if result.action == IntegrityAction.KEEP:
                valid_count += 1
            elif result.action == IntegrityAction.QUARANTINE:
                quarantine_count += 1
            elif result.action == IntegrityAction.REGEN:
                regen_count += 1

        total = len(eval_items)
        quarantine_rate = quarantine_count / total if total > 0 else 0.0

        # Preflight passes if quarantine rate is below 20%
        passed = quarantine_rate < 0.20

        return IntegrityPreflightResult(
            passed=passed,
            gold_file_valid=True,
            gold_hash_verified=True,
            code_hash_match=True,
            expected_hash=EXPECTED_GOLD_HASH,
            actual_hash=actual_hash,
            items_valid=valid_count,
            items_quarantined=quarantine_count,
            items_regenerated=regen_count,
            quarantine_rate=quarantine_rate,
            error_message=None if passed else f"High quarantine rate: {quarantine_rate:.1%}"
        )

    def classify_item_source(self, item: Dict) -> str:
        """
        Classify whether an item is from real usage or synthetic generation.

        Returns: 'real' or 'synthetic'
        """
        # Check provenance for synthetic markers
        query = item.get('query', {})
        provenance = query.get('provenance', {})
        source_type = provenance.get('source_type', '')

        # Synthetic markers
        if 'synthetic' in source_type.lower():
            return 'synthetic'
        if 'minimal_pair' in source_type.lower():
            return 'synthetic'
        if 'regen' in source_type.lower():
            return 'synthetic'

        # Check for killfix prefix in eval_id (manually curated fixes)
        eval_id = item.get('eval_id', '')
        if eval_id.startswith('killfix_'):
            return 'synthetic'

        # Default to real
        return 'real'


class RetrievalScorer:
    """Scores multi-sense retrieval eval items."""

    def __init__(self, similarity_fn=None):
        """
        Args:
            similarity_fn: Function(query_text, passage_text) -> float
                          If None, uses simple word overlap as placeholder
        """
        self.similarity_fn = similarity_fn or self._default_similarity

    def _default_similarity(self, query: str, passage: str) -> float:
        """Simple word overlap similarity (placeholder for real embeddings)."""
        q_words = set(query.lower().split())
        p_words = set(passage.lower().split())
        if not q_words or not p_words:
            return 0.0
        intersection = len(q_words & p_words)
        union = len(q_words | p_words)
        return intersection / union if union > 0 else 0.0

    def score_item(self, item: Dict, model_rankings: List[str] = None) -> RetrievalResult:
        """
        Score a single retrieval eval item.

        Args:
            item: The eval item dict
            model_rankings: Optional pre-computed ranking of passage_ids.
                           If None, uses similarity_fn to compute.
        """
        query = item['query']
        candidates = item['candidates']
        correct_ids = set(item['correct_passage_ids'])

        # Compute rankings if not provided
        if model_rankings is None:
            scores = []
            for cand in candidates:
                sim = self.similarity_fn(query['text'], cand['text'])
                scores.append((cand['passage_id'], sim, cand))
            scores.sort(key=lambda x: -x[1])  # Descending
            model_rankings = [s[0] for s in scores]
            score_map = {s[0]: s[1] for s in scores}
        else:
            score_map = {}

        # Compute metrics
        top_1_hit = model_rankings[0] in correct_ids if model_rankings else False
        top_3_hit = bool(set(model_rankings[:3]) & correct_ids) if model_rankings else False
        top_5_hit = bool(set(model_rankings[:5]) & correct_ids) if model_rankings else False

        # Best correct rank
        best_rank = 999
        for i, pid in enumerate(model_rankings):
            if pid in correct_ids:
                best_rank = i + 1  # 1-indexed
                break

        # Margin at decision
        margin = 0.0
        if score_map:
            correct_scores = [score_map[pid] for pid in correct_ids if pid in score_map]
            incorrect_scores = [score_map[pid] for pid in score_map if pid not in correct_ids]
            if correct_scores and incorrect_scores:
                best_correct = max(correct_scores)
                best_incorrect = max(incorrect_scores)
                margin = best_correct - best_incorrect

        # Failure reason
        failure_reason = None
        wrong_passage_id = None
        wrong_sense_id = None

        if not top_1_hit and model_rankings:
            wrong_passage_id = model_rankings[0]
            wrong_cand = next((c for c in candidates if c['passage_id'] == wrong_passage_id), None)
            if wrong_cand:
                wrong_sense_id = wrong_cand.get('sense_id', '')
                failure_reason = self._classify_failure(query, wrong_cand, item)

        return RetrievalResult(
            eval_id=item['eval_id'],
            lemma=item['lemma'],
            cue_tier=query.get('cue_tier', 'unknown'),
            top_1_hit=top_1_hit,
            top_3_hit=top_3_hit,
            top_5_hit=top_5_hit,
            best_correct_rank=best_rank,
            margin_at_decision=margin,
            failure_reason=failure_reason,
            wrong_passage_id=wrong_passage_id,
            wrong_sense_id=wrong_sense_id
        )

    def _classify_failure(self, query: Dict, wrong_cand: Dict, item: Dict) -> str:
        """Classify the reason for a retrieval failure."""
        query_sense = query.get('sense_id', '')
        wrong_sense = wrong_cand.get('sense_id', '')
        query_cue = query.get('cue_strength', 0.5)

        # Parse sense IDs to extract components
        def parse_sense(sid):
            # Format: source:lemma.pos.key
            parts = sid.split(':')
            if len(parts) == 2:
                source = parts[0]
                rest = parts[1]
            else:
                source = 'unknown'
                rest = sid

            subparts = rest.split('.')
            lemma = subparts[0] if subparts else ''
            pos = subparts[1] if len(subparts) > 1 else ''
            return {'source': source, 'lemma': lemma, 'pos': pos, 'full': sid}

        q_parsed = parse_sense(query_sense)
        w_parsed = parse_sense(wrong_sense)

        # Underdetermined context
        if query_cue < 0.4:
            return ReasonCode.UNDERDETERMINED_CONTEXT.value

        # Within-lemma collision (same lemma)
        if q_parsed['lemma'] == w_parsed['lemma']:
            if q_parsed['pos'] == w_parsed['pos']:
                return ReasonCode.WITHIN_LEMMA_COLLISION.value
            else:
                return ReasonCode.CROSS_POS_COLLISION.value

        # Cross-domain confusable (different lemma but shared topic)
        # This would need topic tag comparison - simplified here
        return ReasonCode.CROSS_DOMAIN_CONFUSABLE.value


class CoherenceScorer:
    """Scores multi-step coherence eval items."""

    def score_item(self, item: Dict, model_outputs: List[Dict] = None) -> CoherenceResult:
        """
        Score a single coherence eval item.

        Args:
            item: The eval item dict
            model_outputs: List of dicts with 'active_sense_id' and 'evidence' per step.
                          If None, simulates perfect performance for testing.
        """
        steps = item['steps']
        checks = item['coherence_checks']

        # If no model outputs, simulate for testing
        if model_outputs is None:
            # Placeholder: assume model gets it right (for harness testing)
            model_outputs = [{'active_sense_id': s['expected_sense'], 'evidence': ''} for s in steps]

        # Score per-step matches
        per_step_matches = []
        for i, step in enumerate(steps):
            expected = step['expected_sense']
            actual = model_outputs[i]['active_sense_id'] if i < len(model_outputs) else ''
            per_step_matches.append(expected == actual)

        # Score transitions
        transitions_expected = 0
        transitions_detected = 0
        transitions_false_positive = 0

        for i in range(1, len(steps)):
            prev_expected = steps[i-1]['expected_sense']
            curr_expected = steps[i]['expected_sense']
            expected_shift = prev_expected != curr_expected

            prev_actual = model_outputs[i-1]['active_sense_id'] if i-1 < len(model_outputs) else ''
            curr_actual = model_outputs[i]['active_sense_id'] if i < len(model_outputs) else ''
            actual_shift = prev_actual != curr_actual

            if expected_shift:
                transitions_expected += 1
                if actual_shift and curr_actual == curr_expected:
                    transitions_detected += 1
            else:
                if actual_shift:
                    transitions_false_positive += 1

        # Final state
        final_expected = steps[-1]['expected_sense']
        final_actual = model_outputs[-1]['active_sense_id'] if model_outputs else ''
        final_state_correct = final_expected == final_actual

        # Failure reason
        failure_reason = None
        if not all(per_step_matches):
            # Find first failure
            for i, match in enumerate(per_step_matches):
                if not match:
                    failure_reason = f"step_{i+1}_mismatch"
                    break

        return CoherenceResult(
            eval_id=item['eval_id'],
            lemma=item['lemma'],
            num_steps=item['num_steps'],
            num_sense_shifts=item['num_sense_shifts'],
            per_step_matches=per_step_matches,
            transitions_expected=transitions_expected,
            transitions_detected=transitions_detected,
            transitions_false_positive=transitions_false_positive,
            final_state_correct=final_state_correct,
            failure_reason=failure_reason
        )


class AutopilotDashboard:
    """Generates the autopilot dashboard report."""

    def __init__(self, run_id: str = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.retrieval_results: List[RetrievalResult] = []
        self.coherence_results: List[CoherenceResult] = []
        self.canary_score = 0.0
        self.slack_value = 0.0
        self.integrity_preflight: Optional[IntegrityPreflightResult] = None
        # Maps eval_id -> source type ('real' or 'synthetic')
        self.item_source_map: Dict[str, str] = {}

    def set_gates(self, canary_score: float, slack_value: float):
        """Set the gate values."""
        self.canary_score = canary_score
        self.slack_value = slack_value

    def set_integrity_preflight(self, result: IntegrityPreflightResult):
        """Set the integrity preflight result."""
        self.integrity_preflight = result

    def add_retrieval_results(self, results: List[RetrievalResult], source_map: Dict[str, str] = None):
        """Add retrieval results with optional source classification."""
        self.retrieval_results.extend(results)
        if source_map:
            self.item_source_map.update(source_map)

    def add_coherence_results(self, results: List[CoherenceResult]):
        """Add coherence results."""
        self.coherence_results.extend(results)

    def generate_report(self, canary_threshold: float = 0.90, slack_threshold: float = 0.0) -> DashboardReport:
        """Generate the complete dashboard report."""

        # Section A: Gates
        canary_pass = self.canary_score >= canary_threshold
        slack_pass = self.slack_value >= slack_threshold

        # Section B: Retrieval metrics
        retrieval_real_top1 = retrieval_real_top3 = 0.0
        retrieval_real_count = 0
        retrieval_synth_top1 = retrieval_synth_top3 = 0.0
        retrieval_synth_count = 0

        if self.retrieval_results:
            retrieval_top1 = np.mean([r.top_1_hit for r in self.retrieval_results])
            retrieval_top3 = np.mean([r.top_3_hit for r in self.retrieval_results])
            retrieval_top5 = np.mean([r.top_5_hit for r in self.retrieval_results])

            ranks = [r.best_correct_rank for r in self.retrieval_results if r.best_correct_rank < 999]
            retrieval_rank_median = np.median(ranks) if ranks else 999
            retrieval_rank_p90 = np.percentile(ranks, 90) if ranks else 999

            margins = [r.margin_at_decision for r in self.retrieval_results]
            retrieval_margin_mean = np.mean(margins) if margins else 0.0
            retrieval_margin_p10 = np.percentile(margins, 10) if margins else 0.0

            # By tier
            retrieval_by_tier = {}
            for tier in ['high_cue', 'medium_cue', 'low_cue']:
                tier_results = [r for r in self.retrieval_results if r.cue_tier == tier]
                if tier_results:
                    retrieval_by_tier[tier] = {
                        'top1': np.mean([r.top_1_hit for r in tier_results]),
                        'top3': np.mean([r.top_3_hit for r in tier_results]),
                        'top5': np.mean([r.top_5_hit for r in tier_results]),
                        'count': len(tier_results)
                    }

            # Real vs Synthetic split
            real_results = [r for r in self.retrieval_results
                           if self.item_source_map.get(r.eval_id, 'real') == 'real']
            synth_results = [r for r in self.retrieval_results
                            if self.item_source_map.get(r.eval_id, 'real') == 'synthetic']

            if real_results:
                retrieval_real_top1 = np.mean([r.top_1_hit for r in real_results])
                retrieval_real_top3 = np.mean([r.top_3_hit for r in real_results])
                retrieval_real_count = len(real_results)

            if synth_results:
                retrieval_synth_top1 = np.mean([r.top_1_hit for r in synth_results])
                retrieval_synth_top3 = np.mean([r.top_3_hit for r in synth_results])
                retrieval_synth_count = len(synth_results)
        else:
            retrieval_top1 = retrieval_top3 = retrieval_top5 = 0.0
            retrieval_rank_median = retrieval_rank_p90 = 999
            retrieval_margin_mean = retrieval_margin_p10 = 0.0
            retrieval_by_tier = {}

        # Section C: Coherence metrics
        if self.coherence_results:
            # Per-step accuracy
            all_step_matches = []
            for r in self.coherence_results:
                all_step_matches.extend(r.per_step_matches)
            coherence_per_step_accuracy = np.mean(all_step_matches) if all_step_matches else 0.0

            # Transition P/R/F1
            total_expected = sum(r.transitions_expected for r in self.coherence_results)
            total_detected = sum(r.transitions_detected for r in self.coherence_results)
            total_fp = sum(r.transitions_false_positive for r in self.coherence_results)

            precision = total_detected / (total_detected + total_fp) if (total_detected + total_fp) > 0 else 0.0
            recall = total_detected / total_expected if total_expected > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            coherence_transition_precision = precision
            coherence_transition_recall = recall
            coherence_transition_f1 = f1

            # Final state accuracy
            coherence_final_state_accuracy = np.mean([r.final_state_correct for r in self.coherence_results])

            # By shift count
            coherence_by_shifts = {}
            for shifts in [1, 2]:
                shift_results = [r for r in self.coherence_results if r.num_sense_shifts == shifts]
                if shift_results:
                    coherence_by_shifts[f"{shifts}_shift"] = {
                        'per_step_acc': np.mean([m for r in shift_results for m in r.per_step_matches]),
                        'final_state_acc': np.mean([r.final_state_correct for r in shift_results]),
                        'count': len(shift_results)
                    }
            # 2+ shifts
            multi_shift = [r for r in self.coherence_results if r.num_sense_shifts >= 2]
            if multi_shift:
                coherence_by_shifts['2+_shifts'] = {
                    'per_step_acc': np.mean([m for r in multi_shift for m in r.per_step_matches]),
                    'final_state_acc': np.mean([r.final_state_correct for r in multi_shift]),
                    'count': len(multi_shift)
                }
        else:
            coherence_per_step_accuracy = 0.0
            coherence_transition_precision = coherence_transition_recall = coherence_transition_f1 = 0.0
            coherence_final_state_accuracy = 0.0
            coherence_by_shifts = {}

        # Section D: Top killers
        top_killers = self._extract_top_killers(20)

        # Composite score (weighted)
        # Weights: Top-1 most important, final state matters, canary is gate
        composite = (
            0.30 * retrieval_top1 +
            0.15 * retrieval_top3 +
            0.05 * retrieval_top5 +
            0.20 * coherence_final_state_accuracy +
            0.15 * coherence_per_step_accuracy +
            0.15 * coherence_transition_f1
        ) * 100  # Scale to 0-100

        return DashboardReport(
            timestamp=datetime.now().isoformat(),
            run_id=self.run_id,
            canary_pass=canary_pass,
            canary_score=self.canary_score,
            slack_pass=slack_pass,
            slack_value=self.slack_value,
            integrity_preflight=self.integrity_preflight,
            retrieval_top1=retrieval_top1,
            retrieval_top3=retrieval_top3,
            retrieval_top5=retrieval_top5,
            retrieval_rank_median=retrieval_rank_median,
            retrieval_rank_p90=retrieval_rank_p90,
            retrieval_by_tier=retrieval_by_tier,
            retrieval_margin_mean=retrieval_margin_mean,
            retrieval_margin_p10=retrieval_margin_p10,
            retrieval_real_top1=retrieval_real_top1,
            retrieval_real_top3=retrieval_real_top3,
            retrieval_real_count=retrieval_real_count,
            retrieval_synth_top1=retrieval_synth_top1,
            retrieval_synth_top3=retrieval_synth_top3,
            retrieval_synth_count=retrieval_synth_count,
            coherence_per_step_accuracy=coherence_per_step_accuracy,
            coherence_transition_precision=coherence_transition_precision,
            coherence_transition_recall=coherence_transition_recall,
            coherence_transition_f1=coherence_transition_f1,
            coherence_final_state_accuracy=coherence_final_state_accuracy,
            coherence_by_shifts=coherence_by_shifts,
            top_killers=top_killers,
            composite_score=composite
        )

    def _extract_top_killers(self, n: int) -> List[Dict]:
        """Extract top N killers with reason codes."""
        killers = []

        # From retrieval failures
        for r in self.retrieval_results:
            if not r.top_1_hit:
                killers.append({
                    'type': 'retrieval',
                    'eval_id': r.eval_id,
                    'lemma': r.lemma,
                    'cue_tier': r.cue_tier,
                    'best_correct_rank': r.best_correct_rank,
                    'margin': r.margin_at_decision,
                    'reason_code': r.failure_reason or ReasonCode.UNKNOWN.value,
                    'wrong_sense_id': r.wrong_sense_id,
                    'severity': (1.0 - r.margin_at_decision) * (1.0 if r.cue_tier == 'high_cue' else 0.7)
                })

        # From coherence failures
        for r in self.coherence_results:
            if not r.final_state_correct:
                killers.append({
                    'type': 'coherence',
                    'eval_id': r.eval_id,
                    'lemma': r.lemma,
                    'num_steps': r.num_steps,
                    'num_sense_shifts': r.num_sense_shifts,
                    'step_accuracy': np.mean(r.per_step_matches),
                    'reason_code': r.failure_reason or ReasonCode.UNKNOWN.value,
                    'severity': (1.0 - np.mean(r.per_step_matches)) * r.num_sense_shifts
                })

        # Sort by severity (highest first)
        killers.sort(key=lambda x: -x.get('severity', 0))

        return killers[:n]

    def render_text_report(self, report: DashboardReport) -> str:
        """Render the report as formatted text."""
        lines = []

        lines.append("=" * 70)
        lines.append(f"AUTOPILOT DASHBOARD - Run: {report.run_id}")
        lines.append(f"Timestamp: {report.timestamp}")
        if report.integrity_preflight:
            lines.append(f"Integrity: {report.integrity_preflight.fingerprint()}")
        lines.append("=" * 70)

        # Section A: Gates
        lines.append("\n[SECTION A: GATES]")
        lines.append("-" * 40)
        canary_status = "PASS" if report.canary_pass else "FAIL"
        slack_status = "PASS" if report.slack_pass else "FAIL"
        lines.append(f"  Canary:  {canary_status} (score: {report.canary_score:.3f})")
        lines.append(f"  Slack:   {slack_status} (value: {report.slack_value:.4f})")

        # Integrity Preflight
        if report.integrity_preflight:
            pf = report.integrity_preflight
            pf_status = "PASS" if pf.passed else "FAIL"
            lines.append(f"  Integrity Preflight: {pf_status}")
            lines.append(f"    Gold Hash (file):   {'Match' if pf.gold_hash_verified else 'MISMATCH'}")
            lines.append(f"    Gold Hash (code):   {'Match' if pf.code_hash_match else 'MISMATCH'}")
            lines.append(f"    Items Valid: {pf.items_valid}")
            lines.append(f"    Items Quarantined: {pf.items_quarantined} ({pf.quarantine_rate:.1%})")
            lines.append(f"    Items Regenerated: {pf.items_regenerated}")
            lines.append(f"    Fingerprint: {pf.fingerprint()}")
            if pf.error_message:
                lines.append(f"    Error: {pf.error_message}")

        # Section B: Retrieval
        lines.append("\n[SECTION B: RETRIEVAL]")
        lines.append("-" * 40)
        lines.append(f"  Top-1 Hit Rate:     {report.retrieval_top1:.1%}")
        lines.append(f"  Top-3 Hit Rate:     {report.retrieval_top3:.1%}")
        lines.append(f"  Top-5 Hit Rate:     {report.retrieval_top5:.1%}")
        lines.append(f"  Best-Rank Median:   {report.retrieval_rank_median:.1f}")
        lines.append(f"  Best-Rank P90:      {report.retrieval_rank_p90:.1f}")
        lines.append(f"  Margin Mean:        {report.retrieval_margin_mean:.4f}")
        lines.append(f"  Margin P10:         {report.retrieval_margin_p10:.4f}")

        lines.append("\n  By Cue Tier:")
        for tier, metrics in report.retrieval_by_tier.items():
            lines.append(f"    {tier}: Top-1={metrics['top1']:.1%}, Top-3={metrics['top3']:.1%} (n={metrics['count']})")

        # Real vs Synthetic split
        lines.append("\n  By Source Type:")
        if report.retrieval_real_count > 0:
            lines.append(f"    real_usage: Top-1={report.retrieval_real_top1:.1%}, Top-3={report.retrieval_real_top3:.1%} (n={report.retrieval_real_count})")
        if report.retrieval_synth_count > 0:
            lines.append(f"    synthetic:  Top-1={report.retrieval_synth_top1:.1%}, Top-3={report.retrieval_synth_top3:.1%} (n={report.retrieval_synth_count})")

        # Section C: Coherence
        lines.append("\n[SECTION C: COHERENCE]")
        lines.append("-" * 40)
        lines.append(f"  Per-Step Accuracy:      {report.coherence_per_step_accuracy:.1%}")
        lines.append(f"  Transition Precision:   {report.coherence_transition_precision:.1%}")
        lines.append(f"  Transition Recall:      {report.coherence_transition_recall:.1%}")
        lines.append(f"  Transition F1:          {report.coherence_transition_f1:.1%}")
        lines.append(f"  Final State Accuracy:   {report.coherence_final_state_accuracy:.1%}")

        lines.append("\n  By Sense Shifts:")
        for shifts, metrics in report.coherence_by_shifts.items():
            lines.append(f"    {shifts}: Step={metrics['per_step_acc']:.1%}, Final={metrics['final_state_acc']:.1%} (n={metrics['count']})")

        # Section D: Top Killers
        lines.append("\n[SECTION D: TOP KILLERS]")
        lines.append("-" * 40)
        for i, killer in enumerate(report.top_killers[:20], 1):
            if killer['type'] == 'retrieval':
                lines.append(f"  {i:2d}. [{killer['type']}] {killer['lemma']} | tier={killer['cue_tier']} | rank={killer['best_correct_rank']} | reason={killer['reason_code']}")
            else:
                lines.append(f"  {i:2d}. [{killer['type']}] {killer['lemma']} | steps={killer['num_steps']} | shifts={killer['num_sense_shifts']} | reason={killer['reason_code']}")

        # Composite Score
        lines.append("\n" + "=" * 70)
        lines.append(f"COMPOSITE SCORE: {report.composite_score:.1f}/100")
        lines.append("=" * 70)

        return "\n".join(lines)


def run_eval_harness(retrieval_file: Path, coherence_file: Path,
                     canary_score: float = 0.95, slack_value: float = 0.05,
                     output_dir: Path = None,
                     skip_integrity_check: bool = False) -> DashboardReport:
    """
    Run the full evaluation harness with integrity preflight.

    Args:
        retrieval_file: Path to retrieval eval JSONL
        coherence_file: Path to coherence eval JSONL
        canary_score: Threshold for canary gate
        slack_value: Threshold for slack gate
        output_dir: Directory for output reports
        skip_integrity_check: If True, skip integrity preflight (for testing only)
    """

    print("=" * 70)
    print("EVALUATION HARNESS")
    print("=" * 70)

    # Load eval items
    print("\nLoading evaluation items...")

    retrieval_items = []
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                retrieval_items.append(json.loads(line))
    print(f"  Retrieval items: {len(retrieval_items)}")

    coherence_items = []
    with open(coherence_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                coherence_items.append(json.loads(line))
    print(f"  Coherence items: {len(coherence_items)}")

    # MANDATORY PREFLIGHT: Run integrity gate
    print("\n[PREFLIGHT] Running integrity gate...")
    preflight = IntegrityPreflight()
    preflight_result = None

    if not skip_integrity_check:
        preflight_result = preflight.run_preflight(retrieval_items)
        pf_status = "PASS" if preflight_result.passed else "FAIL"
        print(f"  Integrity Preflight: {pf_status}")
        print(f"    Gold Hash (file): {'Match' if preflight_result.gold_hash_verified else 'MISMATCH'}")
        print(f"    Gold Hash (code): {'Match' if preflight_result.code_hash_match else 'MISMATCH'}")
        print(f"    Items Valid: {preflight_result.items_valid}")
        print(f"    Items Quarantined: {preflight_result.items_quarantined} ({preflight_result.quarantine_rate:.1%})")
        print(f"    Items Regenerated: {preflight_result.items_regenerated}")
        print(f"    Fingerprint: {preflight_result.fingerprint()}")

        if not preflight_result.passed:
            print(f"\n  WARNING: Integrity preflight FAILED - {preflight_result.error_message}")
            print("  Proceeding with scoring, but results may be unreliable.")
    else:
        print("  Skipping integrity check (skip_integrity_check=True)")

    # Classify items by source type
    print("\nClassifying items by source type...")
    source_map = {}
    real_count = 0
    synth_count = 0
    for item in retrieval_items:
        source_type = preflight.classify_item_source(item)
        source_map[item['eval_id']] = source_type
        if source_type == 'real':
            real_count += 1
        else:
            synth_count += 1
    print(f"  Real usage items: {real_count}")
    print(f"  Synthetic items: {synth_count}")

    # Score items
    print("\nScoring items...")

    retrieval_scorer = RetrievalScorer()
    retrieval_results = [retrieval_scorer.score_item(item) for item in retrieval_items]

    coherence_scorer = CoherenceScorer()
    coherence_results = [coherence_scorer.score_item(item) for item in coherence_items]

    # Generate dashboard
    print("\nGenerating dashboard...")

    dashboard = AutopilotDashboard()
    dashboard.set_gates(canary_score, slack_value)
    if preflight_result:
        dashboard.set_integrity_preflight(preflight_result)
    dashboard.add_retrieval_results(retrieval_results, source_map)
    dashboard.add_coherence_results(coherence_results)

    report = dashboard.generate_report()

    # Render text report
    text_report = dashboard.render_text_report(report)
    print("\n" + text_report)

    # Save reports
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON report
        json_file = output_dir / f"dashboard_{report.run_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"\nSaved JSON report: {json_file}")

        # Text report
        text_file = output_dir / f"dashboard_{report.run_id}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        print(f"Saved text report: {text_file}")

        # Killers file (for next iteration)
        killers_file = output_dir / f"killers_{report.run_id}.jsonl"
        with open(killers_file, 'w', encoding='utf-8') as f:
            for killer in report.top_killers:
                f.write(json.dumps(killer) + '\n')
        print(f"Saved killers list: {killers_file}")

    return report


if __name__ == "__main__":
    eval_dir = Path("paradigm_factory/v2/evals")
    output_dir = Path("paradigm_factory/v2/dashboard_reports")

    report = run_eval_harness(
        retrieval_file=eval_dir / "eval_multi_sense_retrieval.jsonl",
        coherence_file=eval_dir / "eval_multi_step_coherence.jsonl",
        canary_score=0.95,  # Placeholder - would come from actual canary eval
        slack_value=0.05,   # Placeholder - would come from SenseHead metrics
        output_dir=output_dir
    )
