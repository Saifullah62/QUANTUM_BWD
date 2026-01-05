"""
Paradigm Analysis
=================

Deep analysis of QLLM paradigm effectiveness through:
- Layer activation analysis
- Paradigm contribution tracking
- Ablation studies
- Comparative analysis with baseline models
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LayerActivation:
    """Activation data from a model layer"""
    layer_name: str
    paradigm: str
    mean_activation: float
    std_activation: float
    sparsity: float  # Fraction of zeros
    entropy: float
    raw_data: Optional[torch.Tensor] = None


class ParadigmAnalyzer:
    """
    Analyze how each paradigm contributes to model behavior.

    Provides:
    1. Activation analysis per paradigm layer
    2. Contribution tracking during inference
    3. Ablation study support
    4. Visualization of paradigm dynamics
    """

    PARADIGM_LAYERS = {
        'semantic_phase': ['embeddings', 'phase_modulator', 'interference'],
        'retrocausal': ['retrocausal_attn', 'two_state_vector', 'weak_value'],
        'lindblad': ['lindblad', 'dissipative_norm', 'entropy_gate'],
        'qualia': ['qualia_encoder', 'qualia_head'],
        'emergent': ['attractor', 'complexity_time', 'informational_flow']
    }

    def __init__(self, model):
        self.model = model
        self.activations: Dict[str, List[LayerActivation]] = {}
        self.hooks = []

    def _register_hooks(self):
        """Register forward hooks to capture activations"""
        self.activations = {p: [] for p in self.PARADIGM_LAYERS}

        def make_hook(layer_name, paradigm):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation = LayerActivation(
                        layer_name=layer_name,
                        paradigm=paradigm,
                        mean_activation=output.abs().mean().item(),
                        std_activation=output.std().item(),
                        sparsity=(output == 0).float().mean().item(),
                        entropy=self._compute_entropy(output),
                        raw_data=output.detach().clone() if output.numel() < 10000 else None
                    )
                    self.activations[paradigm].append(activation)
            return hook

        for name, module in self.model.named_modules():
            for paradigm, layer_names in self.PARADIGM_LAYERS.items():
                if any(ln in name.lower() for ln in layer_names):
                    hook = module.register_forward_hook(make_hook(name, paradigm))
                    self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _compute_entropy(self, x: torch.Tensor) -> float:
        """Compute entropy of activation distribution"""
        flat = x.view(-1)
        probs = F.softmax(flat, dim=0)
        entropy = -(probs * (probs + 1e-10).log()).sum()
        return entropy.item()

    def analyze_forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single forward pass through the model.

        Returns activation statistics for each paradigm.
        """
        self._register_hooks()

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

            # Aggregate activations by paradigm
            analysis = {}
            for paradigm, activations in self.activations.items():
                if activations:
                    analysis[paradigm] = {
                        'num_layers': len(activations),
                        'mean_activation': np.mean([a.mean_activation for a in activations]),
                        'std_activation': np.mean([a.std_activation for a in activations]),
                        'mean_sparsity': np.mean([a.sparsity for a in activations]),
                        'mean_entropy': np.mean([a.entropy for a in activations]),
                        'layers': [a.layer_name for a in activations]
                    }
                else:
                    analysis[paradigm] = {'num_layers': 0, 'mean_activation': 0}

            return analysis

        finally:
            self._remove_hooks()

    def run_ablation_study(
        self,
        test_inputs: List[torch.Tensor],
        paradigm_to_ablate: str
    ) -> Dict[str, Any]:
        """
        Run ablation study by disabling a paradigm's layers.

        Compares performance with and without the paradigm.
        """
        results = {
            'paradigm': paradigm_to_ablate,
            'baseline': [],
            'ablated': []
        }

        # Baseline (all paradigms active)
        for input_ids in test_inputs:
            with torch.no_grad():
                outputs = self.model(input_ids)
            results['baseline'].append({
                'loss': outputs.get('loss', 0),
                'perplexity': torch.exp(outputs.get('loss', torch.tensor(0))).item()
            })

        # Ablated (target paradigm disabled)
        # Store original states
        original_states = {}
        for name, module in self.model.named_modules():
            if any(ln in name.lower() for ln in self.PARADIGM_LAYERS.get(paradigm_to_ablate, [])):
                original_states[name] = {
                    'training': module.training,
                    'requires_grad': {n: p.requires_grad for n, p in module.named_parameters()}
                }
                # Disable by setting eval mode and zeroing gradients
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        try:
            for input_ids in test_inputs:
                with torch.no_grad():
                    outputs = self.model(input_ids)
                results['ablated'].append({
                    'loss': outputs.get('loss', 0),
                    'perplexity': torch.exp(outputs.get('loss', torch.tensor(0))).item()
                })
        finally:
            # Restore original states
            for name, module in self.model.named_modules():
                if name in original_states:
                    if original_states[name]['training']:
                        module.train()
                    for n, p in module.named_parameters():
                        p.requires_grad = original_states[name]['requires_grad'].get(n, True)

        # Compute impact
        baseline_loss = np.mean([r['loss'] for r in results['baseline'] if r['loss']])
        ablated_loss = np.mean([r['loss'] for r in results['ablated'] if r['loss']])

        results['impact'] = {
            'loss_increase': ablated_loss - baseline_loss,
            'relative_impact': (ablated_loss - baseline_loss) / (baseline_loss + 1e-10)
        }

        return results

    def analyze_paradigm_interactions(
        self,
        input_ids: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze how paradigms interact with each other.

        Measures correlation between paradigm activations.
        """
        self._register_hooks()

        try:
            with torch.no_grad():
                self.model(input_ids)

            # Compute correlations between paradigm activations
            interactions = {}
            paradigms = list(self.PARADIGM_LAYERS.keys())

            for i, p1 in enumerate(paradigms):
                interactions[p1] = {}
                acts1 = self.activations.get(p1, [])
                if not acts1:
                    continue

                mean1 = np.mean([a.mean_activation for a in acts1])

                for j, p2 in enumerate(paradigms):
                    if i >= j:
                        continue

                    acts2 = self.activations.get(p2, [])
                    if not acts2:
                        continue

                    mean2 = np.mean([a.mean_activation for a in acts2])

                    # Compute correlation coefficient
                    # Using mean activations as proxy
                    correlation = min(mean1, mean2) / (max(mean1, mean2) + 1e-10)
                    interactions[p1][p2] = correlation
                    if p2 not in interactions:
                        interactions[p2] = {}
                    interactions[p2][p1] = correlation

            return interactions

        finally:
            self._remove_hooks()

    def generate_paradigm_report(
        self,
        test_inputs: List[torch.Tensor],
        output_dir: str = "./analysis"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive paradigm analysis report.

        Creates visualizations and detailed statistics.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report = {
            'activation_analysis': [],
            'ablation_studies': {},
            'paradigm_interactions': None
        }

        # Activation analysis for each input
        print("Running activation analysis...")
        for i, input_ids in enumerate(test_inputs):
            analysis = self.analyze_forward_pass(input_ids)
            report['activation_analysis'].append(analysis)

        # Ablation studies for each paradigm
        print("Running ablation studies...")
        for paradigm in self.PARADIGM_LAYERS.keys():
            print(f"  Ablating {paradigm}...")
            report['ablation_studies'][paradigm] = self.run_ablation_study(
                test_inputs[:5],  # Use subset for efficiency
                paradigm
            )

        # Paradigm interactions
        print("Analyzing paradigm interactions...")
        report['paradigm_interactions'] = self.analyze_paradigm_interactions(test_inputs[0])

        # Generate visualizations
        self._generate_visualizations(report, output_path)

        # Save report
        report_path = output_path / "paradigm_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Report saved to {report_path}")
        return report

    def _generate_visualizations(self, report: Dict, output_path: Path):
        """Generate visualization plots"""

        # 1. Activation heatmap by paradigm
        paradigms = list(self.PARADIGM_LAYERS.keys())
        if report['activation_analysis']:
            fig, ax = plt.subplots(figsize=(10, 6))

            data = []
            for analysis in report['activation_analysis']:
                row = [analysis.get(p, {}).get('mean_activation', 0) for p in paradigms]
                data.append(row)

            if data:
                im = ax.imshow(data, aspect='auto', cmap='viridis')
                ax.set_xticks(range(len(paradigms)))
                ax.set_xticklabels(paradigms, rotation=45, ha='right')
                ax.set_ylabel('Sample')
                ax.set_title('Paradigm Activation Heatmap')
                plt.colorbar(im, ax=ax, label='Mean Activation')
                plt.tight_layout()
                plt.savefig(output_path / 'activation_heatmap.png', dpi=150)
                plt.close()

        # 2. Ablation impact bar chart
        if report['ablation_studies']:
            fig, ax = plt.subplots(figsize=(10, 6))

            paradigms_ablated = list(report['ablation_studies'].keys())
            impacts = [
                report['ablation_studies'][p].get('impact', {}).get('relative_impact', 0)
                for p in paradigms_ablated
            ]

            colors = ['red' if i > 0 else 'green' for i in impacts]
            ax.bar(paradigms_ablated, impacts, color=colors)
            ax.set_ylabel('Relative Loss Increase')
            ax.set_title('Paradigm Ablation Impact')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'ablation_impact.png', dpi=150)
            plt.close()

        # 3. Paradigm interaction network
        if report['paradigm_interactions']:
            fig, ax = plt.subplots(figsize=(8, 8))

            # Create adjacency matrix
            paradigms = list(self.PARADIGM_LAYERS.keys())
            n = len(paradigms)
            matrix = np.zeros((n, n))

            for i, p1 in enumerate(paradigms):
                for j, p2 in enumerate(paradigms):
                    if p1 in report['paradigm_interactions'] and p2 in report['paradigm_interactions'][p1]:
                        matrix[i, j] = report['paradigm_interactions'][p1][p2]

            im = ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(paradigms, rotation=45, ha='right')
            ax.set_yticklabels(paradigms)
            ax.set_title('Paradigm Interactions')
            plt.colorbar(im, ax=ax, label='Correlation')
            plt.tight_layout()
            plt.savefig(output_path / 'paradigm_interactions.png', dpi=150)
            plt.close()


class ComparativeAnalyzer:
    """
    Compare QLLM with baseline models to measure paradigm benefits.
    """

    def __init__(self, qllm_model, baseline_model, tokenizer):
        self.qllm = qllm_model
        self.baseline = baseline_model
        self.tokenizer = tokenizer

    def compare_on_task(
        self,
        task_name: str,
        test_cases: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Compare models on a specific task.

        Args:
            task_name: Name of the task (e.g., 'ambiguity_resolution')
            test_cases: List of {input, expected_output} dicts
        """
        results = {
            'task': task_name,
            'qllm': {'scores': [], 'outputs': []},
            'baseline': {'scores': [], 'outputs': []}
        }

        for case in test_cases:
            input_text = case['input']
            expected = case['expected_output']

            # Generate with both models
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt')['input_ids']

            with torch.no_grad():
                qllm_out = self.qllm.generate(input_ids, max_new_tokens=50)
                baseline_out = self.baseline.generate(input_ids, max_new_tokens=50)

            qllm_text = self.tokenizer.decode(qllm_out[0])
            baseline_text = self.tokenizer.decode(baseline_out[0])

            # Score outputs (simple word overlap for now)
            qllm_score = self._score_output(qllm_text, expected)
            baseline_score = self._score_output(baseline_text, expected)

            results['qllm']['scores'].append(qllm_score)
            results['qllm']['outputs'].append(qllm_text)
            results['baseline']['scores'].append(baseline_score)
            results['baseline']['outputs'].append(baseline_text)

        # Compute summary
        results['summary'] = {
            'qllm_mean': np.mean(results['qllm']['scores']),
            'baseline_mean': np.mean(results['baseline']['scores']),
            'improvement': np.mean(results['qllm']['scores']) - np.mean(results['baseline']['scores'])
        }

        return results

    def _score_output(self, output: str, expected: str) -> float:
        """Simple scoring based on word overlap"""
        out_words = set(output.lower().split())
        exp_words = set(expected.lower().split())
        if not exp_words:
            return 0
        overlap = len(out_words & exp_words)
        return overlap / len(exp_words)
