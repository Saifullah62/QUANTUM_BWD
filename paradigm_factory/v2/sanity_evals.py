"""
Sanity Evaluation Tasks
=======================

Two evaluation formats that test real-world polysemy handling:
1. Retrieval Sense Selection - Does retrieval pick the right meaning under ambiguity?
2. Workflow Consistency - Does the agent maintain sense consistency across a workflow?
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from collections import defaultdict


# =============================================================================
# Eval 1: Retrieval Sense Selection
# =============================================================================

@dataclass
class RetrievalCandidate:
    """A document candidate for retrieval."""
    doc_id: str
    content: str
    sense_id: str
    is_correct: bool  # Whether this matches the query's intended sense
    difficulty: str  # easy, medium, hard


@dataclass
class RetrievalSenseTask:
    """
    A retrieval task where the query is ambiguous and the system must
    select documents with the correct sense.
    """
    task_id: str
    lemma: str
    query: str
    intended_sense_id: str
    intended_sense_gloss: str
    candidates: List[RetrievalCandidate]
    expected_top_k: List[str]  # doc_ids that should be in top-k
    distractor_count: int  # Number of wrong-sense distractors

    def to_dict(self) -> Dict:
        return {
            "task_type": "retrieval_sense_selection",
            "task_id": self.task_id,
            "lemma": self.lemma,
            "query": self.query,
            "intended_sense": {
                "sense_id": self.intended_sense_id,
                "gloss": self.intended_sense_gloss
            },
            "candidates": [asdict(c) for c in self.candidates],
            "evaluation": {
                "expected_top_k": self.expected_top_k,
                "distractor_count": self.distractor_count,
                "k": len(self.expected_top_k)
            }
        }


class RetrievalEvalBuilder:
    """Build retrieval sense selection evaluation tasks."""

    def __init__(self, bundles_path: Path):
        self.bundles = self._load_bundles(bundles_path)

    def _load_bundles(self, path: Path) -> List[Dict]:
        bundles = []
        with open(path) as f:
            for line in f:
                bundles.append(json.loads(line))
        return bundles

    def build_task_from_bundle(self, bundle: Dict) -> Optional[RetrievalSenseTask]:
        """Create a retrieval task from a bundle."""
        items = bundle.get('items', [])
        if len(items) < 3:
            return None

        # Get anchor as query basis
        anchor = next((i for i in items if i['role'] == 'anchor'), None)
        if not anchor:
            return None

        # Get correct-sense items (anchor + positives)
        correct_items = [i for i in items if i['sense_id'] == anchor['sense_id']]

        # Get wrong-sense items (negatives)
        wrong_items = [i for i in items if i['sense_id'] != anchor['sense_id']]

        if not wrong_items:
            return None

        # Build query from anchor context (simplified)
        query = anchor['context']

        # Build candidates
        candidates = []

        # Correct sense documents
        for item in correct_items:
            candidates.append(RetrievalCandidate(
                doc_id=item['item_id'],
                content=item['context'],
                sense_id=item['sense_id'],
                is_correct=True,
                difficulty=item.get('hardness', 'medium')
            ))

        # Wrong sense documents (distractors)
        for item in wrong_items:
            candidates.append(RetrievalCandidate(
                doc_id=item['item_id'],
                content=item['context'],
                sense_id=item['sense_id'],
                is_correct=False,
                difficulty=item.get('hardness', 'medium')
            ))

        # Get sense info
        sense_catalog = bundle.get('sense_catalog', [])
        anchor_sense = next(
            (s for s in sense_catalog if s['sense_id'] == anchor['sense_id']),
            {'sense_id': anchor['sense_id'], 'gloss': ''}
        )

        return RetrievalSenseTask(
            task_id=f"ret_{bundle['bundle_id']}",
            lemma=bundle['word']['lemma'],
            query=query,
            intended_sense_id=anchor['sense_id'],
            intended_sense_gloss=anchor_sense.get('gloss', ''),
            candidates=candidates,
            expected_top_k=[c.doc_id for c in candidates if c.is_correct],
            distractor_count=len(wrong_items)
        )

    def build_eval_set(self, n_tasks: int = 100) -> List[RetrievalSenseTask]:
        """Build a set of retrieval evaluation tasks."""
        tasks = []

        for bundle in self.bundles:
            if len(tasks) >= n_tasks:
                break

            task = self.build_task_from_bundle(bundle)
            if task and task.distractor_count >= 1:
                tasks.append(task)

        return tasks

    def save_eval(self, tasks: List[RetrievalSenseTask], output_path: Path):
        """Save evaluation tasks to JSONL."""
        with open(output_path, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task.to_dict()) + '\n')
        print(f"Saved {len(tasks)} retrieval tasks to {output_path}")


# =============================================================================
# Eval 2: Workflow Consistency
# =============================================================================

@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    step_id: int
    instruction: str
    uses_lemma: str
    intended_sense_id: str
    sense_label: str
    context_hint: str  # What in the instruction disambiguates


@dataclass
class WorkflowConsistencyTask:
    """
    A multi-step workflow where the same lemma appears with different senses.
    The system should maintain consistency (same sense when context matches,
    different sense when context changes).
    """
    task_id: str
    title: str
    description: str
    lemma: str
    steps: List[WorkflowStep]
    consistency_checks: List[Dict]  # Pairs of steps to check for consistency

    def to_dict(self) -> Dict:
        return {
            "task_type": "workflow_consistency",
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "lemma": self.lemma,
            "steps": [asdict(s) for s in self.steps],
            "consistency_checks": self.consistency_checks,
            "evaluation": {
                "total_steps": len(self.steps),
                "lemma_occurrences": len([s for s in self.steps if s.uses_lemma]),
                "n_checks": len(self.consistency_checks)
            }
        }


class WorkflowEvalBuilder:
    """Build workflow consistency evaluation tasks."""

    # Template workflows for common polysemous words
    WORKFLOW_TEMPLATES = {
        "bank": {
            "title": "Financial and Geographic Planning",
            "description": "Plan a riverside development project with funding",
            "senses": {
                "bank#financial_institution": "financial_institution",
                "bank#river_edge": "river_edge"
            },
            "steps": [
                {
                    "instruction": "Contact the bank to arrange a construction loan for the project.",
                    "sense": "bank#financial_institution",
                    "hint": "loan, construction"
                },
                {
                    "instruction": "Survey the river bank where the new park will be built.",
                    "sense": "bank#river_edge",
                    "hint": "river, park"
                },
                {
                    "instruction": "Submit the loan application to the bank with project plans.",
                    "sense": "bank#financial_institution",
                    "hint": "loan, application"
                },
                {
                    "instruction": "Assess erosion damage along the bank from last year's flooding.",
                    "sense": "bank#river_edge",
                    "hint": "erosion, flooding"
                },
                {
                    "instruction": "Request a bank statement showing available project funds.",
                    "sense": "bank#financial_institution",
                    "hint": "statement, funds"
                }
            ]
        },
        "spring": {
            "title": "Seasonal Equipment Maintenance",
            "description": "Prepare outdoor equipment for the spring season",
            "senses": {
                "spring#season": "season",
                "spring#coiled_device": "coiled_device",
                "spring#water_source": "water_source"
            },
            "steps": [
                {
                    "instruction": "Check the spring tension on all trampoline frames.",
                    "sense": "spring#coiled_device",
                    "hint": "tension, frames"
                },
                {
                    "instruction": "Plan the spring planting schedule for the garden.",
                    "sense": "spring#season",
                    "hint": "planting, garden"
                },
                {
                    "instruction": "Replace worn springs in the patio furniture.",
                    "sense": "spring#coiled_device",
                    "hint": "worn, furniture"
                },
                {
                    "instruction": "Test water quality at the natural spring on the property.",
                    "sense": "spring#water_source",
                    "hint": "water, natural"
                },
                {
                    "instruction": "Order new spring bulbs for early spring flowering.",
                    "sense": "spring#season",
                    "hint": "bulbs, flowering"
                }
            ]
        },
        "draft": {
            "title": "Document and Environment Management",
            "description": "Prepare documents while managing office comfort",
            "senses": {
                "draft#preliminary_version": "preliminary_version",
                "draft#air_current": "air_current"
            },
            "steps": [
                {
                    "instruction": "Write the first draft of the project proposal.",
                    "sense": "draft#preliminary_version",
                    "hint": "write, proposal"
                },
                {
                    "instruction": "Close the window to stop the cold draft in the office.",
                    "sense": "draft#air_current",
                    "hint": "cold, window"
                },
                {
                    "instruction": "Send the draft to the team for initial feedback.",
                    "sense": "draft#preliminary_version",
                    "hint": "send, feedback"
                },
                {
                    "instruction": "Adjust the vent to reduce the draft near the workstation.",
                    "sense": "draft#air_current",
                    "hint": "vent, workstation"
                },
                {
                    "instruction": "Revise the draft based on reviewer comments.",
                    "sense": "draft#preliminary_version",
                    "hint": "revise, comments"
                },
                {
                    "instruction": "Check for drafts around the door seals before winter.",
                    "sense": "draft#air_current",
                    "hint": "door, seals"
                }
            ]
        },
        "run": {
            "title": "Software and Athletic Training",
            "description": "Manage both software deployments and fitness routines",
            "senses": {
                "run#execute_software": "execute_software",
                "run#physical_movement": "physical_movement"
            },
            "steps": [
                {
                    "instruction": "Run the test suite before deploying to production.",
                    "sense": "run#execute_software",
                    "hint": "test, deploying"
                },
                {
                    "instruction": "Go for a morning run to clear your head.",
                    "sense": "run#physical_movement",
                    "hint": "morning, head"
                },
                {
                    "instruction": "Run the database migration script overnight.",
                    "sense": "run#execute_software",
                    "hint": "database, script"
                },
                {
                    "instruction": "Track your run distance with the fitness app.",
                    "sense": "run#physical_movement",
                    "hint": "distance, fitness"
                },
                {
                    "instruction": "Run the performance benchmarks on the new server.",
                    "sense": "run#execute_software",
                    "hint": "benchmarks, server"
                }
            ]
        }
    }

    def __init__(self, sense_inventory: Dict = None):
        self.sense_inventory = sense_inventory or {}

    def build_task_from_template(
        self,
        lemma: str,
        template: Dict
    ) -> WorkflowConsistencyTask:
        """Build a workflow task from a template."""

        steps = []
        for i, step_def in enumerate(template['steps'], 1):
            steps.append(WorkflowStep(
                step_id=i,
                instruction=step_def['instruction'],
                uses_lemma=lemma,
                intended_sense_id=step_def['sense'],
                sense_label=template['senses'].get(step_def['sense'], ''),
                context_hint=step_def['hint']
            ))

        # Build consistency checks
        checks = []

        # Same-sense pairs should be consistent
        for i, s1 in enumerate(steps):
            for s2 in steps[i+1:]:
                if s1.intended_sense_id == s2.intended_sense_id:
                    checks.append({
                        "step_a": s1.step_id,
                        "step_b": s2.step_id,
                        "expected": "same_sense",
                        "sense_id": s1.intended_sense_id
                    })

        # Different-sense pairs should differ
        for i, s1 in enumerate(steps):
            for s2 in steps[i+1:]:
                if s1.intended_sense_id != s2.intended_sense_id:
                    checks.append({
                        "step_a": s1.step_id,
                        "step_b": s2.step_id,
                        "expected": "different_sense",
                        "sense_a": s1.intended_sense_id,
                        "sense_b": s2.intended_sense_id
                    })

        return WorkflowConsistencyTask(
            task_id=f"wf_{lemma}_{uuid.uuid4().hex[:6]}",
            title=template['title'],
            description=template['description'],
            lemma=lemma,
            steps=steps,
            consistency_checks=checks
        )

    def build_eval_set(self) -> List[WorkflowConsistencyTask]:
        """Build workflow evaluation tasks from all templates."""
        tasks = []

        for lemma, template in self.WORKFLOW_TEMPLATES.items():
            task = self.build_task_from_template(lemma, template)
            tasks.append(task)

        return tasks

    def save_eval(self, tasks: List[WorkflowConsistencyTask], output_path: Path):
        """Save evaluation tasks to JSONL."""
        with open(output_path, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task.to_dict()) + '\n')
        print(f"Saved {len(tasks)} workflow tasks to {output_path}")


# =============================================================================
# Combined Eval Suite
# =============================================================================

def build_sanity_eval_suite(
    bundles_path: Path,
    output_dir: Path = None
) -> Dict[str, Path]:
    """Build the complete sanity evaluation suite."""

    output_dir = output_dir or Path("paradigm_factory/v2/evals")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Build retrieval eval
    print("\nBuilding Retrieval Sense Selection eval...")
    ret_builder = RetrievalEvalBuilder(bundles_path)
    ret_tasks = ret_builder.build_eval_set(n_tasks=100)
    ret_path = output_dir / "retrieval_sense_selection.jsonl"
    ret_builder.save_eval(ret_tasks, ret_path)
    outputs['retrieval'] = ret_path

    # Build workflow eval
    print("\nBuilding Workflow Consistency eval...")
    wf_builder = WorkflowEvalBuilder()
    wf_tasks = wf_builder.build_eval_set()
    wf_path = output_dir / "workflow_consistency.jsonl"
    wf_builder.save_eval(wf_tasks, wf_path)
    outputs['workflow'] = wf_path

    # Write summary
    summary = {
        "eval_suite": "polysemy_sanity_v1",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "evaluations": {
            "retrieval_sense_selection": {
                "path": str(ret_path),
                "n_tasks": len(ret_tasks),
                "description": "Tests whether retrieval picks correct sense under ambiguity"
            },
            "workflow_consistency": {
                "path": str(wf_path),
                "n_tasks": len(wf_tasks),
                "description": "Tests sense consistency across multi-step workflows"
            }
        }
    }

    summary_path = output_dir / "eval_suite_manifest.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    outputs['manifest'] = summary_path

    print(f"\nEval suite saved to {output_dir}")
    return outputs


def main():
    """Build the sanity eval suite."""
    # Check for bundles
    bundles_path = Path("paradigm_factory/v2/bundles/bundles_v2_test.jsonl")

    if not bundles_path.exists():
        # Create minimal test bundles
        print("No bundles found. Creating minimal test set...")
        bundles_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a few test bundles manually
        test_bundles = [
            {
                "bundle_id": "test001",
                "word": {"lemma": "bank", "pos": "NOUN"},
                "sense_catalog": [
                    {"sense_id": "bank#financial", "gloss": "financial institution"},
                    {"sense_id": "bank#river", "gloss": "river edge"}
                ],
                "items": [
                    {"item_id": "a1", "role": "anchor", "sense_id": "bank#financial",
                     "context": "I deposited money at the bank today.", "hardness": "easy"},
                    {"item_id": "p1", "role": "positive", "sense_id": "bank#financial",
                     "context": "The bank approved my loan application.", "hardness": "easy"},
                    {"item_id": "n1", "role": "hard_negative", "sense_id": "bank#river",
                     "context": "We walked along the muddy bank of the river.", "hardness": "hard"}
                ]
            }
        ]

        with open(bundles_path, 'w') as f:
            for b in test_bundles:
                f.write(json.dumps(b) + '\n')

    # Build eval suite
    outputs = build_sanity_eval_suite(bundles_path)

    print("\n" + "=" * 60)
    print("  Sanity Eval Suite Complete")
    print("=" * 60)
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
