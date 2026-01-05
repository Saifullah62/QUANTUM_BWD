"""
Quantum Paradigm Dataset Generator
===================================

Generates training data that embodies quantum paradigm concepts:
1. Semantic ambiguity resolution (phase collapse)
2. Retrocausal reasoning (future-informed inference)
3. Contextual meaning modulation
4. Qualia-aware responses

Uses the GPU cluster's swarm controller to generate high-quality data.
"""

import json
import asyncio
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import random


@dataclass
class DataExample:
    """Single training example"""
    input_text: str
    output_text: str
    paradigm: str  # Which paradigm this exemplifies
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'input': self.input_text,
            'output': self.output_text,
            'paradigm': self.paradigm,
            'metadata': self.metadata
        }


class DatasetGenerator:
    """
    Generate quantum paradigm training data using the swarm.

    Uses the fleet's swarm controller for:
    - Debate (multiple perspectives)
    - Consensus (resolved meaning)
    - Chain-of-thought (step-by-step reasoning)
    - Parallel exploration (diverse approaches)
    """

    SWARM_URL = "http://159.203.35.45:8007"

    # Paradigm-specific prompts
    PARADIGM_PROMPTS = {
        'semantic_phase': [
            {
                'template': 'The word "{word}" can mean different things. In the context of "{context1}", it means one thing. In the context of "{context2}", it means something else. Explain how the meaning changes based on context.',
                'words': [
                    ('bank', 'I walked along the river bank', 'I need to go to the bank to deposit money'),
                    ('light', 'The light was too bright', 'The bag was light enough to carry'),
                    ('spring', 'The spring flowers bloomed', 'The spring in the mattress broke'),
                    ('bark', 'The dog started to bark', 'The bark of the tree was rough'),
                    ('date', 'What is today\'s date?', 'I have a date tonight'),
                    ('match', 'Light the match carefully', 'It was a close match'),
                    ('wave', 'The ocean wave crashed', 'She gave a friendly wave'),
                    ('present', 'I bought a present', 'Please be present at the meeting'),
                    ('lead', 'Lead the way forward', 'The pipe was made of lead'),
                    ('tear', 'A tear rolled down her cheek', 'Don\'t tear the paper'),
                ]
            },
            {
                'template': 'Resolve the ambiguity: "{ambiguous_sentence}" Given the context: "{context}", what is the most coherent interpretation?',
                'examples': [
                    ('I saw her duck', 'We were at a pond', 'She has a pet duck that I observed'),
                    ('Flying planes can be dangerous', 'He is a pilot', 'The act of piloting aircraft carries risks'),
                    ('The chicken is ready to eat', 'The chef finished cooking', 'The chicken dish is prepared for consumption'),
                    ('Time flies like an arrow', 'A philosophy lecture', 'Time passes quickly, metaphorically like an arrow'),
                ]
            }
        ],
        'retrocausal': [
            {
                'template': 'Given the outcome "{outcome}", what sequence of events most likely led to this result? Think backwards from the end.',
                'examples': [
                    ('The company went bankrupt', 'Poor financial decisions, declining sales, failed to adapt to market changes'),
                    ('She won the Nobel Prize', 'Years of dedicated research, breakthrough discovery, peer recognition'),
                    ('The experiment succeeded', 'Careful hypothesis, rigorous methodology, persistent iteration'),
                    ('The relationship ended', 'Growing apart, unresolved conflicts, different life goals'),
                ]
            },
            {
                'template': 'If we want to achieve "{goal}" in the future, what must be true now? Reason from the future backward to the present.',
                'examples': [
                    ('Carbon neutrality by 2050', 'Massive renewable investment, policy changes, behavioral shifts starting now'),
                    ('Fluency in a new language', 'Daily practice, immersion, systematic vocabulary building'),
                    ('A successful startup', 'Strong team, validated problem, sustainable business model'),
                ]
            }
        ],
        'lindblad': [
            {
                'template': 'In a noisy environment with conflicting information: {noise}. Find the stable, coherent interpretation that emerges despite the noise.',
                'examples': [
                    ('Some say the economy is growing, others say it\'s shrinking, data is mixed', 'The economy shows uneven growth across sectors'),
                    ('Multiple witnesses give contradictory accounts of the event', 'The core facts that all accounts agree on form the stable truth'),
                    ('The experiment has high variance but a clear trend', 'The underlying pattern is robust despite measurement noise'),
                ]
            },
            {
                'template': 'Transform this chaotic input into a stable, organized output: "{chaotic_input}"',
                'examples': [
                    ('meeting tomorrow maybe 3pm or 4 depends on bob call him first agenda tbd', 'Meeting: Tomorrow at 3-4pm (pending Bob\'s confirmation). Action: Call Bob. Agenda: To be determined.'),
                    ('buy milk eggs maybe bread if sale also dog food low', 'Shopping list: 1. Milk 2. Eggs 3. Bread (if on sale) 4. Dog food (running low)'),
                ]
            }
        ],
        'qualia': [
            {
                'template': 'Describe not just what "{experience}" is, but what it feels like to experience it. Include the qualitative, subjective dimension.',
                'examples': [
                    ('seeing the color red', 'A warm, urgent sensation that draws attention. It feels active and present, different from the cool retreat of blue.'),
                    ('the taste of coffee', 'A complex bitter-sweet warmth that spreads through awareness. It carries alertness in its flavor.'),
                    ('hearing a minor chord', 'A sound that pulls inward, creating a space of contemplation. It has weight and depth.'),
                ]
            },
            {
                'template': 'Rate this situation on these qualitative dimensions: valence (positive/negative), arousal (intensity), certainty (confidence). Situation: "{situation}"',
                'examples': [
                    ('Receiving unexpected good news', 'Valence: Strongly positive. Arousal: High - excitement and surprise. Certainty: Initially uncertain, then confirmed.'),
                    ('Waiting for important results', 'Valence: Neutral-anxious. Arousal: Moderate-high anticipation. Certainty: Low - outcome unknown.'),
                ]
            }
        ],
        'emergent': [
            {
                'template': 'What stable patterns emerge from these diverse observations? Find the "frozen flows" - the constants that persist. Observations: {observations}',
                'examples': [
                    ('Markets cycle, empires rise and fall, technologies disrupt and stabilize', 'Change is constant, but patterns of adaptation and selection persist across domains.'),
                    ('Different cultures independently developed similar myths, moral codes, social structures', 'Certain human universals emerge from our shared cognitive and social architecture.'),
                ]
            },
            {
                'template': 'As complexity increases in "{system}", what new properties emerge that weren\'t present in the simpler components?',
                'examples': [
                    ('neural networks', 'Simple neurons create complex cognition. Emergence: learning, abstraction, creativity.'),
                    ('ecosystems', 'Individual organisms create complex webs. Emergence: resilience, nutrient cycling, biodiversity.'),
                    ('cities', 'Individual people create urban systems. Emergence: culture, innovation, economic specialization.'),
                ]
            }
        ]
    }

    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def call_swarm(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call the swarm controller API"""
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.SWARM_URL}{endpoint}",
                json=data
            )
            return response.json()

    async def generate_semantic_phase_examples(
        self,
        num_examples: int = 100
    ) -> List[DataExample]:
        """Generate examples for semantic phase understanding"""
        examples = []

        for prompt_config in self.PARADIGM_PROMPTS['semantic_phase']:
            if 'words' in prompt_config:
                for word, ctx1, ctx2 in prompt_config['words']:
                    # Use swarm to generate explanation
                    result = await self.call_swarm('/swarm/think', {
                        'problem': prompt_config['template'].format(
                            word=word, context1=ctx1, context2=ctx2
                        )
                    })

                    if result.get('status') == 'SUCCESS':
                        examples.append(DataExample(
                            input_text=f"Context 1: {ctx1}\nContext 2: {ctx2}\n\nExplain how the meaning of '{word}' changes between these contexts.",
                            output_text=result.get('answer', result.get('full_response', '')),
                            paradigm='semantic_phase',
                            metadata={'word': word}
                        ))

            elif 'examples' in prompt_config:
                for sentence, context, interpretation in prompt_config['examples']:
                    examples.append(DataExample(
                        input_text=f"Ambiguous: {sentence}\nContext: {context}\n\nProvide the correct interpretation.",
                        output_text=interpretation,
                        paradigm='semantic_phase',
                        metadata={'ambiguous_sentence': sentence}
                    ))

        return examples[:num_examples]

    async def generate_retrocausal_examples(
        self,
        num_examples: int = 100
    ) -> List[DataExample]:
        """Generate examples for retrocausal reasoning"""
        examples = []

        for prompt_config in self.PARADIGM_PROMPTS['retrocausal']:
            if 'examples' in prompt_config:
                for outcome, reasoning in prompt_config['examples'][:2]:
                    # Use swarm debate for multiple perspectives
                    result = await self.call_swarm('/swarm/explore', {
                        'problem': prompt_config['template'].format(
                            outcome=outcome
                        ) if 'outcome' in prompt_config['template'] else prompt_config['template'].format(
                            goal=outcome
                        ),
                        'num_explorers': 3
                    })

                    if result.get('status') == 'SUCCESS':
                        examples.append(DataExample(
                            input_text=f"Outcome: {outcome}\n\nWhat events led to this outcome? Think backwards.",
                            output_text=result.get('synthesis', reasoning),
                            paradigm='retrocausal',
                            metadata={'outcome': outcome}
                        ))

        return examples[:num_examples]

    async def generate_lindblad_examples(
        self,
        num_examples: int = 100
    ) -> List[DataExample]:
        """Generate examples for noise-to-signal transformation"""
        examples = []

        for prompt_config in self.PARADIGM_PROMPTS['lindblad']:
            for noise, stable in prompt_config['examples']:
                examples.append(DataExample(
                    input_text=f"Noisy input: {noise}\n\nFind the stable, coherent interpretation.",
                    output_text=stable,
                    paradigm='lindblad',
                    metadata={'noise_level': 'high'}
                ))

        return examples[:num_examples]

    async def generate_qualia_examples(
        self,
        num_examples: int = 100
    ) -> List[DataExample]:
        """Generate examples for qualia-aware responses"""
        examples = []

        for prompt_config in self.PARADIGM_PROMPTS['qualia']:
            for experience, qualia_response in prompt_config['examples']:
                examples.append(DataExample(
                    input_text=f"Experience: {experience}\n\nDescribe the subjective, qualitative aspects.",
                    output_text=qualia_response,
                    paradigm='qualia',
                    metadata={'experience_type': experience}
                ))

        return examples[:num_examples]

    async def generate_emergent_examples(
        self,
        num_examples: int = 100
    ) -> List[DataExample]:
        """Generate examples for emergent computation understanding"""
        examples = []

        for prompt_config in self.PARADIGM_PROMPTS['emergent']:
            for system, emergence in prompt_config['examples']:
                result = await self.call_swarm('/swarm/think', {
                    'problem': f"What properties emerge from complex {system} that aren't present in simpler components?"
                })

                if result.get('status') == 'SUCCESS':
                    examples.append(DataExample(
                        input_text=f"System: {system}\n\nWhat emergent properties arise from increasing complexity?",
                        output_text=result.get('answer', emergence),
                        paradigm='emergent',
                        metadata={'system': system}
                    ))

        return examples[:num_examples]

    async def generate_full_dataset(
        self,
        examples_per_paradigm: int = 100,
        output_file: str = "quantum_paradigm_data.jsonl"
    ) -> Path:
        """Generate complete dataset for all paradigms"""
        all_examples = []

        print("Generating Semantic Phase examples...")
        all_examples.extend(await self.generate_semantic_phase_examples(examples_per_paradigm))

        print("Generating Retrocausal examples...")
        all_examples.extend(await self.generate_retrocausal_examples(examples_per_paradigm))

        print("Generating Lindblad examples...")
        all_examples.extend(await self.generate_lindblad_examples(examples_per_paradigm))

        print("Generating Qualia examples...")
        all_examples.extend(await self.generate_qualia_examples(examples_per_paradigm))

        print("Generating Emergent examples...")
        all_examples.extend(await self.generate_emergent_examples(examples_per_paradigm))

        # Shuffle
        random.shuffle(all_examples)

        # Write to file
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            for example in all_examples:
                f.write(json.dumps(example.to_dict()) + '\n')

        print(f"Generated {len(all_examples)} examples -> {output_path}")
        return output_path


class QuantumParadigmDataset:
    """
    PyTorch-compatible dataset for training QLLM.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 2048
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.examples = []
        with open(self.data_path, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Format as instruction-following
        text = f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze(),  # Causal LM
            'paradigm': example['paradigm']
        }


# CLI interface
async def main():
    """Generate dataset from command line"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Quantum Paradigm Dataset")
    parser.add_argument('--output-dir', default='./data', help='Output directory')
    parser.add_argument('--examples-per-paradigm', type=int, default=50, help='Examples per paradigm')
    parser.add_argument('--output-file', default='quantum_paradigm_data.jsonl', help='Output filename')

    args = parser.parse_args()

    generator = DatasetGenerator(output_dir=args.output_dir)
    await generator.generate_full_dataset(
        examples_per_paradigm=args.examples_per_paradigm,
        output_file=args.output_file
    )


if __name__ == "__main__":
    asyncio.run(main())
