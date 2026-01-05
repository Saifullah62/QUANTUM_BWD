#!/usr/bin/env python3
"""
Quick test of the v2 bundle pipeline without swarm.
Creates sample bundles and runs a few training steps to validate the dashboard.
"""

import sys
import os
import torch
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from paradigm_factory.polysemy_bundle_v2 import create_bundle, save_bundles, SENSE_CATALOG
from paradigm_factory.bundle_dataset import create_bundle_dataloader, compute_bundle_contrastive_loss


def create_test_bundles(output_path: Path, n_bundles: int = 20):
    """Create test bundles from SENSE_CATALOG without swarm."""
    bundles = []
    bundle_idx = 0

    # Sample contexts for each word (manually curated for testing)
    test_contexts = {
        "present": {
            "gift": [
                "She wrapped the birthday present carefully.",
                "The present was hidden under the tree.",
                "He bought her a present for their anniversary."
            ],
            "current_time": [
                "At present, sales are rising faster than expected.",
                "The present situation requires immediate attention."
            ],
            "to_show": [
                "The report will present the findings at noon.",
                "She will present her research at the conference."
            ]
        },
        "bank": {
            "financial_institution": [
                "I deposited my paycheck at the bank this morning.",
                "The bank approved my loan application yesterday.",
                "She works as a teller at the local bank."
            ],
            "river_edge": [
                "We sat on the grassy bank watching the river flow.",
                "The deer came down to drink at the river bank."
            ]
        },
        "spring": {
            "season": [
                "The flowers bloom beautifully in spring.",
                "Spring is my favorite time of year."
            ],
            "water_source": [
                "We hiked to the mountain spring for fresh water.",
                "The natural spring provided cold, clear water."
            ],
            "coiled_metal": [
                "The spring in the mattress was broken.",
                "The old clock needed a new spring mechanism."
            ]
        },
        "wave": {
            "ocean_motion": [
                "A huge wave crashed against the rocks.",
                "The surfer rode the wave expertly."
            ],
            "hand_gesture": [
                "He returned the wave with a smile.",
                "She gave a friendly wave as she left."
            ]
        }
    }

    for word, senses_dict in test_contexts.items():
        senses = list(senses_dict.keys())
        if word not in SENSE_CATALOG:
            continue

        sense_defs = SENSE_CATALOG[word]

        for anchor_sense in senses:
            if anchor_sense not in senses_dict or len(senses_dict[anchor_sense]) < 2:
                continue

            anchor_def = next((s for s in sense_defs if s.label == anchor_sense), None)
            if not anchor_def:
                continue

            anchor_context = senses_dict[anchor_sense][0]
            positive_contexts = [(anchor_def.sense_id, ctx) for ctx in senses_dict[anchor_sense][1:]]

            negative_contexts = []
            hard_negative_contexts = []

            for other_sense, other_contexts in senses_dict.items():
                if other_sense == anchor_sense:
                    continue
                other_def = next((s for s in sense_defs if s.label == other_sense), None)
                if not other_def or not other_contexts:
                    continue

                negative_contexts.append((other_def.sense_id, other_contexts[0]))
                if len(other_contexts) > 1:
                    hard_negative_contexts.append(
                        (other_def.sense_id, other_contexts[1], "Similar structure, different sense")
                    )

            if positive_contexts and negative_contexts:
                bundle = create_bundle(
                    word=word,
                    anchor_sense_id=anchor_def.sense_id,
                    anchor_context=anchor_context,
                    positive_contexts=positive_contexts,
                    negative_contexts=negative_contexts,
                    hard_negative_contexts=hard_negative_contexts[:1],
                    bundle_index=bundle_idx,
                    seed=42
                )
                bundles.append(bundle)
                bundle_idx += 1

            if len(bundles) >= n_bundles:
                break
        if len(bundles) >= n_bundles:
            break

    save_bundles(bundles, output_path)
    print(f"Created {len(bundles)} test bundles at {output_path}")
    return bundles


def test_training_loop(bundle_path: Path, steps: int = 10):
    """Run a few training steps to validate the dashboard."""
    from qllm.core.config import QLLMConfig
    from qllm.core.model import QLLM

    print("\n" + "=" * 60)
    print("  Testing Training Loop")
    print("=" * 60)

    # Create small model (num_heads must be >= 4 for PhaseModulator)
    model_config = QLLMConfig(
        vocab_size=500,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        intermediate_dim=128,
        max_seq_length=64,
        use_semantic_phase=True,
        semantic_phase_dim=64,
        use_emergent_init=True
    )

    model = QLLM(model_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create dataloader
    dataloader = create_bundle_dataloader(
        bundle_path=bundle_path,
        batch_size=4,
        max_length=64,
        vocab_size=500
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"\nDevice: {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nRunning {steps} training steps...\n")

    dataloader_iter = iter(dataloader)

    for step in range(1, steps + 1):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        model.train()
        optimizer.zero_grad()

        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get embeddings for contrastive loss
        if 'hidden_states' in outputs and outputs['hidden_states'] is not None:
            embeddings = outputs['hidden_states'].mean(dim=1)
        else:
            embeddings = outputs.get('logits', torch.zeros(input_ids.size(0), 64, device=device)).mean(dim=1)

        # Compute contrastive metrics
        metrics = compute_bundle_contrastive_loss(
            embeddings=embeddings,
            batch=batch,
            device=device
        )

        loss = metrics['contrastive_loss']
        loss.backward()
        optimizer.step()

        # Print dashboard
        msr_e = metrics.get('msr_easy', 0)
        msr_h = metrics.get('msr_hard', 0)
        gap_e = metrics.get('gap_easy', 0)
        gap_h = metrics.get('gap_hard', 0)
        slack_e = metrics.get('avg_slack_easy', 0)
        slack_h = metrics.get('avg_slack_hard', 0)

        print(f"Step {step:3d} | Loss: {loss.item():.4f} | "
              f"MSR: {msr_e:.0%}/{msr_h:.0%} | "
              f"Gap: {gap_e:+.3f}/{gap_h:+.3f} | "
              f"Slack: {slack_e:+.3f}/{slack_h:+.3f}")

    print("\n[OK] Training loop test passed!")


def main():
    print("=" * 60)
    print("  V2 Bundle Pipeline Test")
    print("=" * 60)

    # Create test bundles
    test_path = Path("paradigm_factory/output/test_bundles_pipeline.jsonl")
    create_test_bundles(test_path, n_bundles=15)

    # Test training loop
    test_training_loop(test_path, steps=10)

    print("\n" + "=" * 60)
    print("  All Tests Passed!")
    print("=" * 60)
    print("\nReady for production:")
    print("  1. Generate bundles: python paradigm_factory/bundle_factory.py --words 50")
    print("  2. Train: python scripts/train_semantic_phase_v2.py --bundles <path>")
    print("  3. Multi-seed: python scripts/train_semantic_phase_v2.py --bundles <path> --multi-seed")


if __name__ == "__main__":
    main()
