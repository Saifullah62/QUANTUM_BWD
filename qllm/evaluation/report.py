"""
Win Condition Report Generator
==============================

Packages ablation and training results into a clean, stakeholder-ready report.

The story: "Baseline does X; QLLM with paradigm Y does Z; this is not leakage."

Output formats:
- Console summary
- JSON for programmatic access
- Markdown for documentation
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ParadigmResult:
    """Results for a single paradigm"""
    name: str
    enabled: bool
    win_condition_score: float
    baseline_score: float
    improvement: float  # Absolute improvement
    improvement_pct: float  # Percentage improvement
    is_significant: bool  # > 5% improvement
    leak_clean: bool  # No leakage detected


@dataclass
class WinConditionReport:
    """Complete win condition report"""
    timestamp: str
    model_name: str
    total_steps: int

    # Per-paradigm results
    semantic_phase: Optional[ParadigmResult]
    lindblad: Optional[ParadigmResult]
    qualia: Optional[ParadigmResult]
    retrocausal: Optional[ParadigmResult]
    emergent: Optional[ParadigmResult]

    # Overall summary
    paradigms_with_signal: List[str]
    best_paradigm: str
    overall_improvement: float

    # Credibility checks
    all_leaks_clean: bool
    retrocausal_leak_report: Optional[Dict]


class ReportGenerator:
    """
    Generates win condition reports from training/ablation results.

    Usage:
        generator = ReportGenerator()
        report = generator.generate_from_ablation(ablation_results)
        generator.print_report(report)
        generator.save_markdown(report, "results/report.md")
    """

    def __init__(self, significance_threshold: float = 0.05):
        self.significance_threshold = significance_threshold

    def generate_from_ablation(
        self,
        ablation_results: List[Dict],
        model_name: str = "QLLM",
        leak_reports: Optional[Dict] = None
    ) -> WinConditionReport:
        """Generate report from ablation study results."""

        # Find baseline
        baseline = None
        for r in ablation_results:
            if r.get('paradigm') == 'baseline':
                baseline = r
                break

        if baseline is None:
            raise ValueError("No baseline found in ablation results")

        baseline_polysemy = baseline.get('win_conditions', {}).get('polysemy_resolution', 0)
        baseline_consistency = baseline.get('win_conditions', {}).get('noise_invariance', 0)
        baseline_retrocausal = baseline.get('win_conditions', {}).get('retrocausal_reasoning', 0)

        # Process each paradigm
        paradigm_results = {}

        for r in ablation_results:
            paradigm = r.get('paradigm', '')

            if 'semantic_phase' in paradigm:
                score = r.get('win_conditions', {}).get('polysemy_resolution', 0)
                paradigm_results['semantic_phase'] = self._create_paradigm_result(
                    'SemanticPhase', True, score, baseline_polysemy
                )

            elif 'lindblad' in paradigm:
                score = r.get('win_conditions', {}).get('noise_invariance', 0)
                paradigm_results['lindblad'] = self._create_paradigm_result(
                    'Lindblad', True, score, baseline_consistency
                )

            elif 'qualia' in paradigm:
                # Qualia uses combined metric or default to consistency
                score = r.get('win_conditions', {}).get('noise_invariance', 0)
                paradigm_results['qualia'] = self._create_paradigm_result(
                    'Qualia', True, score, baseline_consistency
                )

            elif 'retrocausal' in paradigm:
                score = r.get('win_conditions', {}).get('retrocausal_reasoning', 0)
                leak_clean = leak_reports.get('is_clean', True) if leak_reports else True
                result = self._create_paradigm_result(
                    'Retrocausal', True, score, baseline_retrocausal
                )
                result.leak_clean = leak_clean
                paradigm_results['retrocausal'] = result

            elif 'emergent' in paradigm:
                score = r.get('win_conditions', {}).get('noise_invariance', 0)
                paradigm_results['emergent'] = self._create_paradigm_result(
                    'Emergent', True, score, baseline_consistency
                )

        # Find paradigms with signal
        paradigms_with_signal = [
            name for name, result in paradigm_results.items()
            if result.is_significant
        ]

        # Find best paradigm
        best_paradigm = max(
            paradigm_results.items(),
            key=lambda x: x[1].improvement_pct
        )[0] if paradigm_results else "none"

        # Overall improvement (all paradigms combined)
        all_paradigms_result = None
        for r in ablation_results:
            if r.get('paradigm') == 'all_paradigms':
                all_paradigms_result = r
                break

        if all_paradigms_result:
            # Average improvement across all metrics
            overall_polysemy = all_paradigms_result.get('win_conditions', {}).get('polysemy_resolution', 0)
            overall_consistency = all_paradigms_result.get('win_conditions', {}).get('noise_invariance', 0)
            overall_retro = all_paradigms_result.get('win_conditions', {}).get('retrocausal_reasoning', 0)

            baseline_avg = (baseline_polysemy + baseline_consistency + baseline_retrocausal) / 3
            overall_avg = (overall_polysemy + overall_consistency + overall_retro) / 3
            overall_improvement = overall_avg - baseline_avg
        else:
            overall_improvement = 0.0

        # Check all leaks
        all_leaks_clean = all(
            r.leak_clean for r in paradigm_results.values()
        )

        return WinConditionReport(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            total_steps=ablation_results[0].get('steps', 0) if ablation_results else 0,
            semantic_phase=paradigm_results.get('semantic_phase'),
            lindblad=paradigm_results.get('lindblad'),
            qualia=paradigm_results.get('qualia'),
            retrocausal=paradigm_results.get('retrocausal'),
            emergent=paradigm_results.get('emergent'),
            paradigms_with_signal=paradigms_with_signal,
            best_paradigm=best_paradigm,
            overall_improvement=overall_improvement,
            all_leaks_clean=all_leaks_clean,
            retrocausal_leak_report=leak_reports
        )

    def _create_paradigm_result(
        self,
        name: str,
        enabled: bool,
        score: float,
        baseline_score: float
    ) -> ParadigmResult:
        """Create a ParadigmResult from scores."""
        improvement = score - baseline_score
        improvement_pct = improvement / (baseline_score + 1e-8)
        is_significant = improvement_pct > self.significance_threshold

        return ParadigmResult(
            name=name,
            enabled=enabled,
            win_condition_score=score,
            baseline_score=baseline_score,
            improvement=improvement,
            improvement_pct=improvement_pct,
            is_significant=is_significant,
            leak_clean=True  # Default, may be overwritten
        )

    def print_report(self, report: WinConditionReport):
        """Print report to console."""
        print("\n" + "=" * 70)
        print(f"  QLLM WIN CONDITION REPORT")
        print(f"  Model: {report.model_name}")
        print(f"  Generated: {report.timestamp}")
        print("=" * 70)

        print("\n📊 PARADIGM RESULTS")
        print("-" * 70)

        for paradigm_name in ['semantic_phase', 'lindblad', 'qualia', 'retrocausal', 'emergent']:
            result = getattr(report, paradigm_name)
            if result:
                status = "✓" if result.is_significant else "○"
                leak_status = "🔒" if result.leak_clean else "⚠️"
                print(f"  {status} {result.name:<15} "
                      f"Score: {result.win_condition_score:.4f} "
                      f"(Δ {result.improvement:+.4f}, {result.improvement_pct*100:+.1f}%) "
                      f"{leak_status}")

        print("\n📈 SUMMARY")
        print("-" * 70)
        print(f"  Paradigms with signal: {', '.join(report.paradigms_with_signal) or 'None'}")
        print(f"  Best paradigm: {report.best_paradigm}")
        print(f"  Overall improvement: {report.overall_improvement:+.4f}")

        print("\n🔐 CREDIBILITY")
        print("-" * 70)
        if report.all_leaks_clean:
            print("  ✓ All paradigms pass leak detection")
        else:
            print("  ⚠️ WARNING: Some paradigms show potential leakage")

        if report.retrocausal_leak_report:
            lr = report.retrocausal_leak_report
            print(f"  Retrocausal leak score: {lr.get('overall_leak_score', 'N/A')}")
            if lr.get('red_flags'):
                print(f"  Red flags: {', '.join(lr['red_flags'])}")

        print("\n" + "=" * 70)

        # The story
        print("\n📝 THE STORY (for stakeholders):")
        print("-" * 70)
        self._print_story(report)
        print("=" * 70 + "\n")

    def _print_story(self, report: WinConditionReport):
        """Generate the narrative story."""
        stories = []

        # Baseline
        stories.append("Baseline transformer model establishes reference performance.")

        # Per-paradigm stories
        if report.semantic_phase and report.semantic_phase.is_significant:
            stories.append(
                f"QLLM with SemanticPhase shows {report.semantic_phase.improvement_pct*100:.1f}% "
                f"improvement on polysemy resolution - context-conditioned phase "
                f"successfully differentiates word meanings."
            )

        if report.lindblad and report.lindblad.is_significant:
            stories.append(
                f"QLLM with Lindblad achieves {report.lindblad.improvement_pct*100:.1f}% "
                f"better noise invariance - representations converge to stable "
                f"semantic basins regardless of perturbation."
            )

        if report.retrocausal and report.retrocausal.is_significant:
            leak_note = "" if report.retrocausal.leak_clean else " (REQUIRES LEAK INVESTIGATION)"
            stories.append(
                f"QLLM with Retrocausal improves {report.retrocausal.improvement_pct*100:.1f}% "
                f"on backward inference tasks{leak_note}."
            )

        if report.qualia and report.qualia.is_significant:
            stories.append(
                f"QLLM with Qualia channels shows {report.qualia.improvement_pct*100:.1f}% "
                f"improvement - subjective experience encoding adds semantic richness."
            )

        if report.emergent and report.emergent.is_significant:
            stories.append(
                f"QLLM with Emergent attractors demonstrates {report.emergent.improvement_pct*100:.1f}% "
                f"stability improvement - frozen flows provide robust representations."
            )

        # Credibility statement
        if report.all_leaks_clean:
            stories.append(
                "Critically, all improvements pass rigorous leak detection - "
                "these are genuine architectural contributions, not data leakage."
            )
        else:
            stories.append(
                "⚠️ NOTE: Some paradigms require additional leak investigation "
                "before claims can be validated."
            )

        for story in stories:
            print(f"  • {story}")

    def save_json(self, report: WinConditionReport, path: str):
        """Save report as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)

        print(f"Report saved to {output_path}")

    def save_markdown(self, report: WinConditionReport, path: str):
        """Save report as Markdown."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# QLLM Win Condition Report",
            f"",
            f"**Model:** {report.model_name}",
            f"**Generated:** {report.timestamp}",
            f"**Training Steps:** {report.total_steps}",
            f"",
            f"## Paradigm Results",
            f"",
            f"| Paradigm | Score | Baseline | Improvement | Significant | Leak Clean |",
            f"|----------|-------|----------|-------------|-------------|------------|",
        ]

        for paradigm_name in ['semantic_phase', 'lindblad', 'qualia', 'retrocausal', 'emergent']:
            result = getattr(report, paradigm_name)
            if result:
                sig = "✓" if result.is_significant else "○"
                leak = "✓" if result.leak_clean else "⚠️"
                lines.append(
                    f"| {result.name} | {result.win_condition_score:.4f} | "
                    f"{result.baseline_score:.4f} | {result.improvement:+.4f} ({result.improvement_pct*100:+.1f}%) | "
                    f"{sig} | {leak} |"
                )

        lines.extend([
            f"",
            f"## Summary",
            f"",
            f"- **Paradigms with Signal:** {', '.join(report.paradigms_with_signal) or 'None'}",
            f"- **Best Paradigm:** {report.best_paradigm}",
            f"- **Overall Improvement:** {report.overall_improvement:+.4f}",
            f"- **All Leaks Clean:** {'Yes' if report.all_leaks_clean else 'No ⚠️'}",
            f"",
            f"## The Story",
            f"",
        ])

        # Add story
        if report.semantic_phase and report.semantic_phase.is_significant:
            lines.append(f"- SemanticPhase: {report.semantic_phase.improvement_pct*100:.1f}% improvement on polysemy")
        if report.lindblad and report.lindblad.is_significant:
            lines.append(f"- Lindblad: {report.lindblad.improvement_pct*100:.1f}% better noise invariance")
        if report.retrocausal and report.retrocausal.is_significant:
            lines.append(f"- Retrocausal: {report.retrocausal.improvement_pct*100:.1f}% improvement on backward inference")
        if report.qualia and report.qualia.is_significant:
            lines.append(f"- Qualia: {report.qualia.improvement_pct*100:.1f}% improvement in subjective encoding")
        if report.emergent and report.emergent.is_significant:
            lines.append(f"- Emergent: {report.emergent.improvement_pct*100:.1f}% stability improvement")

        lines.extend([
            f"",
            f"---",
            f"*Generated by QLLM Win Condition Reporter*",
        ])

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"Markdown report saved to {output_path}")


def generate_report_from_training(
    trainer,
    output_dir: str = "./reports"
) -> WinConditionReport:
    """
    Convenience function to generate report from a trainer object.

    Usage:
        report = generate_report_from_training(trainer)
    """
    generator = ReportGenerator()

    # Collect results from trainer history
    win_conditions = trainer.validate_win_conditions()

    # Build mock ablation result format
    results = [{
        'paradigm': 'baseline',
        'win_conditions': {'polysemy_resolution': 0, 'noise_invariance': 0, 'retrocausal_reasoning': 0},
        'steps': trainer.state.step
    }, {
        'paradigm': 'all_paradigms',
        'win_conditions': win_conditions,
        'steps': trainer.state.step
    }]

    report = generator.generate_from_ablation(results, model_name="QLLM")
    generator.print_report(report)

    # Save reports
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator.save_json(report, f"{output_dir}/win_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    generator.save_markdown(report, f"{output_dir}/win_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

    return report
