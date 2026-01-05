#!/usr/bin/env python3
"""
Paradigm Factory - Checkpoint Evaluator
========================================

Runs eval pack against local model checkpoints to compare
paradigm effects on actual behaviors (not just loss).

Tests:
1. Polysemy disambiguation accuracy
2. Lindblad consistency (clean vs noisy output agreement)
"""

import argparse
import json
import re
import torch
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qllm.core.model import QLLM
from qllm.core.config import QLLMConfig


class SimpleTokenizer:
    """Simple character-level tokenizer for evaluation."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        # Build basic vocab: special tokens + ASCII printable
        self.char_to_id = {chr(i): (i - 31) for i in range(32, 127)}
        self.char_to_id['\n'] = 96
        self.char_to_id['\t'] = 97
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

    def encode(self, text: str) -> List[int]:
        """Convert text to token ids."""
        ids = []
        for c in text:
            if c in self.char_to_id:
                ids.append(self.char_to_id[c])
            else:
                ids.append(1)  # UNK
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token ids back to text."""
        chars = []
        for i in ids:
            if i in self.id_to_char:
                chars.append(self.id_to_char[i])
            elif i == 0:
                break  # EOS
            else:
                chars.append('?')
        return ''.join(chars)


def load_checkpoint(checkpoint_dir: Path, device: str = 'cuda') -> QLLM:
    """Load model from checkpoint directory."""
    # Find the model file
    model_files = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))

    if not model_files:
        # Try to find in subdirectories
        model_files = list(checkpoint_dir.glob("**/model*.pt")) + list(checkpoint_dir.glob("**/checkpoint*.pt"))

    if not model_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    checkpoint_path = model_files[0]
    print(f"  Loading from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same config used in training
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Config matching run_proof_ablation_v2.py defaults
        config = QLLMConfig(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            intermediate_dim=256,  # hidden_dim * 2
            max_seq_length=128,
            use_semantic_phase=True,
            semantic_phase_dim=128,
            use_retrocausal_attention=True,
            retrocausal_layers=[0, 2],  # [0, num_layers // 2]
            use_lindblad_layers=True,
            lindblad_every_n_layers=2,
            use_qualia_output=True,
            num_qualia_channels=8,
            use_emergent_init=True
        )

    model = QLLM(config)

    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Fix shape mismatch: prev_complexity is scalar in checkpoint but [1] in model
    if 'complexity_time.prev_complexity' in state_dict:
        val = state_dict['complexity_time.prev_complexity']
        if val.dim() == 0:  # scalar
            state_dict['complexity_time.prev_complexity'] = val.unsqueeze(0)

    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def generate_response(
    model: QLLM,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = 'cuda'
) -> str:
    """Generate a response from the model."""
    # Tokenize
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)

    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_tensor)
            logits = outputs['logits']
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

            # Stop at EOS or max length
            if next_token.item() == tokenizer.eos_token_id:
                break
            if input_tensor.size(1) > 200:
                break

    # Decode
    output_ids = input_tensor[0].tolist()
    return tokenizer.decode(output_ids[len(input_ids):])


def compute_sequence_logprob(model: QLLM, input_ids: torch.Tensor) -> float:
    """Compute average log probability of a sequence under the model."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.get('logits')

        if logits is None:
            return float('-inf')

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Average over sequence (excluding padding)
        mask = shift_labels != 0
        if mask.sum() > 0:
            avg_log_prob = (token_log_probs * mask).sum() / mask.sum()
        else:
            avg_log_prob = token_log_probs.mean()

        return avg_log_prob.item()


def eval_polysemy_item(
    model: QLLM,
    tokenizer: SimpleTokenizer,
    item: Dict,
    device: str = 'cuda'
) -> Dict:
    """Evaluate polysemy disambiguation using log-prob preference scoring.

    Compares model's probability of context+correct_sense vs context+distractor.
    Higher probability for correct sense = successful disambiguation.
    """
    word = item.get('word', '')
    # Extract a clean context snippet
    raw_context = item.get('context_with_blank', '')
    # Try to find actual sentence content
    if 'full_solution' in raw_context:
        sentences = re.findall(r'[A-Z][^.!?]*[.!?]', raw_context)
        context = sentences[0] if sentences else raw_context[:60]
    else:
        context = raw_context[:60]

    correct_sense = item.get('correct_sense', '')[:40]
    distractor = item.get('distractor_senses', [''])[0][:40] if item.get('distractor_senses') else ''

    if not distractor:
        return {'correct': True, 'response': 'no_distractor', 'margin': 0.0}

    try:
        # Create paired sequences: context + sense description
        correct_seq = f"{context} ({correct_sense})"
        distractor_seq = f"{context} ({distractor})"

        # Encode
        correct_ids = tokenizer.encode(correct_seq)[:64]
        distractor_ids = tokenizer.encode(distractor_seq)[:64]

        correct_tensor = torch.tensor([correct_ids], device=device)
        distractor_tensor = torch.tensor([distractor_ids], device=device)

        # Compute log probabilities
        correct_logprob = compute_sequence_logprob(model, correct_tensor)
        distractor_logprob = compute_sequence_logprob(model, distractor_tensor)

        # Model should assign higher probability to correct sense
        is_correct = correct_logprob > distractor_logprob
        margin = correct_logprob - distractor_logprob

        response = f"correct={correct_logprob:.3f} dist={distractor_logprob:.3f} margin={margin:.3f}"

    except Exception as e:
        is_correct = False
        margin = 0.0
        response = f"error: {str(e)[:50]}"

    return {
        'correct': is_correct,
        'response': response,
        'margin': margin,
        'expected': 'correct_logprob > distractor_logprob'
    }


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between two distributions.

    JS divergence is symmetric and bounded [0, ln(2)] ≈ [0, 0.693].
    """
    # Ensure valid probability distributions
    p = torch.nn.functional.softmax(p, dim=-1)
    q = torch.nn.functional.softmax(q, dim=-1)

    # Compute midpoint
    m = 0.5 * (p + q)

    # KL(P || M) and KL(Q || M)
    kl_pm = torch.nn.functional.kl_div(m.log(), p, reduction='none').sum(dim=-1)
    kl_qm = torch.nn.functional.kl_div(m.log(), q, reduction='none').sum(dim=-1)

    # JS = 0.5 * (KL(P||M) + KL(Q||M))
    js = 0.5 * (kl_pm + kl_qm)
    return js


def eval_lindblad_item(
    model: QLLM,
    tokenizer: SimpleTokenizer,
    item: Dict,
    device: str = 'cuda'
) -> Dict:
    """Evaluate Lindblad consistency using logit-based JS divergence.

    Measures how much the next-token distributions diverge between
    clean and noisy inputs. Lower divergence = better noise invariance.
    """
    clean_text = item.get('clean_text', '')[:80]
    noisy_text = item.get('noisy_text', '')[:80]

    try:
        with torch.no_grad():
            # Encode inputs
            clean_ids = tokenizer.encode(clean_text)[:60]
            noisy_ids = tokenizer.encode(noisy_text)[:60]

            # Pad to same length for fair comparison
            max_len = max(len(clean_ids), len(noisy_ids))
            clean_ids = clean_ids + [0] * (max_len - len(clean_ids))
            noisy_ids = noisy_ids + [0] * (max_len - len(noisy_ids))

            clean_tensor = torch.tensor([clean_ids], device=device)
            noisy_tensor = torch.tensor([noisy_ids], device=device)

            # Get logits
            clean_out = model(input_ids=clean_tensor)
            noisy_out = model(input_ids=noisy_tensor)

            clean_logits = clean_out.get('logits')
            noisy_logits = noisy_out.get('logits')

            if clean_logits is not None and noisy_logits is not None:
                # Compare distributions at last N positions (where meaning matters)
                n_positions = min(8, clean_logits.size(1))
                clean_final = clean_logits[0, -n_positions:, :]
                noisy_final = noisy_logits[0, -n_positions:, :]

                # Compute JS divergence at each position, then average
                js_divs = jensen_shannon_divergence(clean_final, noisy_final)
                avg_js = js_divs.mean().item()

                # Normalize to [0, 1] range (max JS is ln(2) ≈ 0.693)
                normalized_js = min(avg_js / 0.693, 1.0)
            else:
                avg_js = 1.0
                normalized_js = 1.0

    except Exception as e:
        avg_js = 1.0
        normalized_js = 1.0

    # Lower divergence = more consistent. Threshold at 0.3 (less than half of max)
    is_consistent = normalized_js < 0.3

    return {
        'consistent': is_consistent,
        'overlap': 1.0 - normalized_js,  # Convert to similarity-like metric
        'js_divergence': avg_js,
        'normalized_js': normalized_js,
        'corruption_type': item.get('corruption_type', 'unknown')
    }


def evaluate_checkpoint(
    checkpoint_dir: Path,
    eval_pack: Dict,
    device: str = 'cuda',
    max_polysemy: int = 20,
    max_lindblad: int = 20
) -> Dict:
    """Evaluate a single checkpoint against the eval pack."""
    # Load model
    try:
        model = load_checkpoint(checkpoint_dir, device)
    except Exception as e:
        print(f"  Failed to load checkpoint: {e}")
        return {'error': str(e)}

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)

    results = {
        'checkpoint': str(checkpoint_dir),
        'polysemy_results': [],
        'lindblad_results': [],
        'metrics': {}
    }

    # Evaluate polysemy items
    polysemy_items = eval_pack.get('polysemy_items', [])[:max_polysemy]
    polysemy_correct = 0

    print(f"  Evaluating {len(polysemy_items)} polysemy items...")
    for item in polysemy_items:
        result = eval_polysemy_item(model, tokenizer, item, device)
        results['polysemy_results'].append(result)
        if result['correct']:
            polysemy_correct += 1

    # Evaluate Lindblad items
    lindblad_items = eval_pack.get('lindblad_items', [])[:max_lindblad]
    lindblad_consistent = 0

    print(f"  Evaluating {len(lindblad_items)} Lindblad items...")
    for item in lindblad_items:
        result = eval_lindblad_item(model, tokenizer, item, device)
        results['lindblad_results'].append(result)
        if result['consistent']:
            lindblad_consistent += 1

    # Compute metrics
    polysemy_margins = [r.get('margin', 0) for r in results['polysemy_results']]
    lindblad_js = [r.get('js_divergence', 1.0) for r in results['lindblad_results']]

    results['metrics'] = {
        'polysemy_accuracy': polysemy_correct / len(polysemy_items) if polysemy_items else 0,
        'polysemy_correct': polysemy_correct,
        'polysemy_total': len(polysemy_items),
        'avg_polysemy_margin': sum(polysemy_margins) / len(polysemy_margins) if polysemy_margins else 0,
        'lindblad_consistency': lindblad_consistent / len(lindblad_items) if lindblad_items else 0,
        'lindblad_consistent': lindblad_consistent,
        'lindblad_total': len(lindblad_items),
        'avg_js_divergence': sum(lindblad_js) / len(lindblad_js) if lindblad_js else 1.0,
        'avg_lindblad_overlap': sum(r['overlap'] for r in results['lindblad_results']) / len(results['lindblad_results']) if results['lindblad_results'] else 0
    }

    return results


def evaluate_all_checkpoints(
    run_dir: Path,
    eval_pack_path: Path,
    output_path: Optional[Path] = None,
    device: str = 'cuda'
) -> Dict:
    """Evaluate all ablation checkpoints from a run."""

    # Load eval pack
    with open(eval_pack_path, 'r') as f:
        eval_pack = json.load(f)

    print(f"Loaded eval pack: {eval_pack.get('name', 'unknown')}")
    print(f"  Polysemy items: {len(eval_pack.get('polysemy_items', []))}")
    print(f"  Lindblad items: {len(eval_pack.get('lindblad_items', []))}")

    # Find ablation directories
    ablations = ['baseline', 'semantic_phase', 'lindblad', 'qualia', 'all_paradigms']

    all_results = {
        'run_dir': str(run_dir),
        'eval_pack': eval_pack_path.name,
        'timestamp': datetime.now().isoformat(),
        'ablations': {}
    }

    for ablation in ablations:
        checkpoint_dir = run_dir / ablation
        if not checkpoint_dir.exists():
            print(f"\nSkipping {ablation} - directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {ablation}")
        print('='*60)

        results = evaluate_checkpoint(checkpoint_dir, eval_pack, device)
        all_results['ablations'][ablation] = results

        if 'metrics' in results:
            m = results['metrics']
            print(f"  Polysemy: {m['polysemy_accuracy']:.1%} ({m['polysemy_correct']}/{m['polysemy_total']}) margin={m['avg_polysemy_margin']:.3f}")
            print(f"  Lindblad: {m['lindblad_consistency']:.1%} ({m['lindblad_consistent']}/{m['lindblad_total']}) JS={m['avg_js_divergence']:.4f}")

    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY (lower JS = better noise invariance)")
    print("="*70)
    print(f"{'Ablation':<18} {'Polysemy':<10} {'Margin':<10} {'Lindblad':<10} {'JS Div':<10}")
    print("-"*70)

    for ablation in ablations:
        if ablation in all_results['ablations'] and 'metrics' in all_results['ablations'][ablation]:
            m = all_results['ablations'][ablation]['metrics']
            print(f"{ablation:<18} {m['polysemy_accuracy']:>6.1%}    {m['avg_polysemy_margin']:>+7.3f}   {m['lindblad_consistency']:>6.1%}    {m['avg_js_divergence']:>7.4f}")

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints against eval pack")
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Directory containing ablation checkpoints')
    parser.add_argument('--eval-pack', type=str, required=True,
                        help='Path to eval pack JSON')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    evaluate_all_checkpoints(
        run_dir=Path(args.run_dir),
        eval_pack_path=Path(args.eval_pack),
        output_path=Path(args.output) if args.output else None,
        device=args.device
    )


if __name__ == "__main__":
    main()
