"""
Paradigm Benchmark Suite
========================

Comprehensive benchmarks for evaluating how well QLLM embodies
each of Daugherty's quantum paradigms.

Each benchmark tests specific capabilities that emerge from
the paradigm-specific layers.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test"""
    benchmark_name: str
    paradigm: str
    score: float
    max_score: float
    details: Dict[str, Any] = field(default_factory=dict)
    test_cases: List[Dict] = field(default_factory=list)

    @property
    def normalized_score(self) -> float:
        return self.score / self.max_score if self.max_score > 0 else 0


class SemanticPhaseBenchmark:
    """
    Benchmark for Semantic Phase paradigm.

    Tests:
    1. Polysemy resolution - same word, different meanings
    2. Context sensitivity - meaning changes with context
    3. Ambiguity detection - recognizing multiple interpretations
    4. Phase coherence - consistent meaning representation
    """

    NAME = "semantic_phase"

    # Test cases for polysemy
    POLYSEMY_TESTS = [
        {
            'word': 'bank',
            'contexts': [
                ('I deposited money at the bank', 'financial_institution'),
                ('The boat landed on the river bank', 'river_edge'),
                ('The plane began to bank left', 'turn_motion')
            ]
        },
        {
            'word': 'light',
            'contexts': [
                ('Turn on the light', 'illumination'),
                ('The bag was very light', 'weight'),
                ('She had a light complexion', 'color_intensity'),
                ('He made light of the situation', 'treat_casually')
            ]
        },
        {
            'word': 'spring',
            'contexts': [
                ('Spring is my favorite season', 'season'),
                ('The spring in the watch broke', 'mechanical'),
                ('Water springs from the ground', 'emerge'),
                ('He has a spring in his step', 'bounce')
            ]
        }
    ]

    # Ambiguous sentences requiring context
    AMBIGUITY_TESTS = [
        {
            'sentence': 'I saw her duck',
            'interpretations': [
                ('She dodged something', 'verb_duck'),
                ('I observed her pet duck', 'noun_duck')
            ]
        },
        {
            'sentence': 'Flying planes can be dangerous',
            'interpretations': [
                ('Piloting aircraft is risky', 'gerund'),
                ('Aircraft in flight pose risks', 'participle')
            ]
        },
        {
            'sentence': 'The chicken is ready to eat',
            'interpretations': [
                ('The meal is prepared', 'food'),
                ('The bird is hungry', 'animal')
            ]
        }
    ]

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _get_representation(self, text: str) -> torch.Tensor:
        """Get model's internal representation of text"""
        inputs = self.tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], output_hidden_states=True)
        return outputs.get('hidden_states', outputs.get('last_hidden_state'))

    def run_polysemy_test(self) -> BenchmarkResult:
        """Test ability to disambiguate polysemous words"""
        test_results = []
        total_score = 0

        for test in self.POLYSEMY_TESTS:
            word = test['word']
            contexts = test['contexts']

            representations = []
            for sentence, meaning in contexts:
                rep = self._get_representation(sentence)
                representations.append((rep, meaning))

            # Different contexts should produce different representations
            for i, (rep1, m1) in enumerate(representations):
                for j, (rep2, m2) in enumerate(representations):
                    if i >= j:
                        continue

                    similarity = F.cosine_similarity(
                        rep1.view(-1), rep2.view(-1), dim=0
                    ).item()

                    # Different meanings should have lower similarity
                    if m1 != m2:
                        score = 1 - similarity  # Higher score for more different
                    else:
                        score = similarity

                    total_score += score
                    test_results.append({
                        'word': word,
                        'meaning1': m1,
                        'meaning2': m2,
                        'similarity': similarity,
                        'score': score
                    })

        max_score = len(test_results)

        return BenchmarkResult(
            benchmark_name='polysemy_resolution',
            paradigm=self.NAME,
            score=total_score,
            max_score=max_score,
            details={'num_tests': len(test_results)},
            test_cases=test_results
        )

    def run_ambiguity_test(self) -> BenchmarkResult:
        """Test detection and handling of ambiguous sentences"""
        test_results = []
        total_score = 0

        for test in self.AMBIGUITY_TESTS:
            sentence = test['sentence']
            interpretations = test['interpretations']

            # Get base representation
            base_rep = self._get_representation(sentence)

            # Check if model can distinguish interpretations when given context
            interp_reps = []
            for context, label in interpretations:
                full_text = f"{context}. {sentence}"
                rep = self._get_representation(full_text)
                interp_reps.append((rep, label))

            # Interpretations should differ
            if len(interp_reps) >= 2:
                similarity = F.cosine_similarity(
                    interp_reps[0][0].view(-1),
                    interp_reps[1][0].view(-1),
                    dim=0
                ).item()

                # Score: how different are the interpretations
                score = 1 - similarity
                total_score += score

                test_results.append({
                    'sentence': sentence,
                    'interpretation_similarity': similarity,
                    'differentiation_score': score
                })

        max_score = len(test_results)

        return BenchmarkResult(
            benchmark_name='ambiguity_detection',
            paradigm=self.NAME,
            score=total_score,
            max_score=max_score,
            details={'num_tests': len(test_results)},
            test_cases=test_results
        )

    def run_all(self) -> List[BenchmarkResult]:
        """Run all semantic phase benchmarks"""
        return [
            self.run_polysemy_test(),
            self.run_ambiguity_test()
        ]


class RetrocausalBenchmark:
    """
    Benchmark for Retrocausal paradigm.

    Tests:
    1. Backward reasoning - deriving causes from effects
    2. Goal-oriented planning - working backward from goals
    3. Hindsight improvement - predictions improve with future context
    """

    NAME = "retrocausal"

    CAUSAL_CHAINS = [
        {
            'cause': 'The company hired many new employees',
            'effect': 'Office space became crowded',
            'chain': ['hiring', 'more_people', 'space_shortage', 'crowding']
        },
        {
            'cause': 'Heavy rain fell overnight',
            'effect': 'The basement flooded',
            'chain': ['rain', 'water_accumulation', 'drainage_overflow', 'flooding']
        },
        {
            'cause': 'The student studied diligently',
            'effect': 'They passed the exam',
            'chain': ['studying', 'knowledge_gain', 'better_answers', 'passing']
        }
    ]

    GOAL_PLANNING = [
        {
            'goal': 'Become fluent in Spanish',
            'required_steps': ['practice_daily', 'learn_vocabulary', 'speak_with_natives']
        },
        {
            'goal': 'Run a marathon',
            'required_steps': ['train_regularly', 'build_endurance', 'proper_nutrition']
        }
    ]

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run_backward_reasoning_test(self) -> BenchmarkResult:
        """Test ability to reason backward from effects to causes"""
        test_results = []
        total_score = 0

        for test in self.CAUSAL_CHAINS:
            # Given effect, can model identify cause?
            prompt = f"Given the outcome: '{test['effect']}', what caused this?"

            # Generate response
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=50,
                    num_return_sequences=1
                )

            response = self.tokenizer.decode(outputs[0])

            # Check if cause keywords appear in response
            cause_text = test['cause'].lower()
            response_lower = response.lower()

            # Simple keyword matching (in production, use semantic similarity)
            matches = sum(
                1 for word in cause_text.split()
                if word in response_lower and len(word) > 3
            )
            score = min(1.0, matches / 3)
            total_score += score

            test_results.append({
                'effect': test['effect'],
                'expected_cause': test['cause'],
                'response': response,
                'score': score
            })

        return BenchmarkResult(
            benchmark_name='backward_reasoning',
            paradigm=self.NAME,
            score=total_score,
            max_score=len(self.CAUSAL_CHAINS),
            test_cases=test_results
        )

    def run_all(self) -> List[BenchmarkResult]:
        """Run all retrocausal benchmarks"""
        return [
            self.run_backward_reasoning_test()
        ]


class LindbladBenchmark:
    """
    Benchmark for Lindblad Dissipation paradigm.

    Tests:
    1. Noise filtering - extract signal from noise
    2. Pattern stabilization - find stable attractors
    3. Information preservation - maintain relevant information
    """

    NAME = "lindblad"

    NOISE_FILTER_TESTS = [
        {
            'noisy': 'meeting tmrw maybe 3pm or 4 depends bob call first agenda tbd important!!',
            'clean': 'Meeting tomorrow at 3-4pm, pending confirmation from Bob. Agenda TBD.'
        },
        {
            'noisy': 'buy milk eggs maybe bread if sale also dog food running low urgent',
            'clean': 'Shopping list: milk, eggs, bread (if on sale), dog food (urgent)'
        },
        {
            'noisy': 'stock up 5% then down 3% volatile market uncertainty fears rising maybe buying opp?',
            'clean': 'Stock showing volatility (+5%, -3%). Market uncertain but may present buying opportunity.'
        }
    ]

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run_noise_filter_test(self) -> BenchmarkResult:
        """Test ability to extract clean signal from noisy input"""
        test_results = []
        total_score = 0

        for test in self.NOISE_FILTER_TESTS:
            prompt = f"Clean up this noisy text: '{test['noisy']}'"

            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=100
                )

            response = self.tokenizer.decode(outputs[0])

            # Measure similarity to expected clean output
            clean_rep = self._get_representation(test['clean'])
            response_rep = self._get_representation(response)

            similarity = F.cosine_similarity(
                clean_rep.view(-1),
                response_rep.view(-1),
                dim=0
            ).item()

            score = max(0, similarity)
            total_score += score

            test_results.append({
                'noisy_input': test['noisy'],
                'expected_clean': test['clean'],
                'model_output': response,
                'similarity': similarity,
                'score': score
            })

        return BenchmarkResult(
            benchmark_name='noise_filtering',
            paradigm=self.NAME,
            score=total_score,
            max_score=len(self.NOISE_FILTER_TESTS),
            test_cases=test_results
        )

    def _get_representation(self, text: str) -> torch.Tensor:
        """Get model representation"""
        inputs = self.tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], output_hidden_states=True)
        return outputs.get('hidden_states', outputs.get('last_hidden_state'))

    def run_all(self) -> List[BenchmarkResult]:
        """Run all Lindblad benchmarks"""
        return [
            self.run_noise_filter_test()
        ]


class QualiaBenchmark:
    """
    Benchmark for Qualia paradigm.

    Tests:
    1. Subjective description - describing qualitative experience
    2. Emotional valence - detecting positive/negative
    3. Multi-dimensional qualia - balancing qualia channels
    """

    NAME = "qualia"

    EXPERIENCE_DESCRIPTIONS = [
        {
            'experience': 'seeing a sunset',
            'expected_qualities': ['warm', 'color', 'peaceful', 'beautiful']
        },
        {
            'experience': 'tasting coffee',
            'expected_qualities': ['bitter', 'warm', 'aromatic', 'alert']
        },
        {
            'experience': 'hearing thunder',
            'expected_qualities': ['loud', 'startling', 'powerful', 'rumbling']
        }
    ]

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run_subjective_description_test(self) -> BenchmarkResult:
        """Test ability to generate rich subjective descriptions"""
        test_results = []
        total_score = 0

        for test in self.EXPERIENCE_DESCRIPTIONS:
            prompt = f"Describe the subjective experience of {test['experience']}. Focus on what it feels like, not just facts."

            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=100
                )

            response = self.tokenizer.decode(outputs[0]).lower()

            # Check for expected quality words
            matches = sum(
                1 for quality in test['expected_qualities']
                if quality in response
            )
            score = matches / len(test['expected_qualities'])
            total_score += score

            test_results.append({
                'experience': test['experience'],
                'expected_qualities': test['expected_qualities'],
                'response': response,
                'qualities_found': matches,
                'score': score
            })

        return BenchmarkResult(
            benchmark_name='subjective_description',
            paradigm=self.NAME,
            score=total_score,
            max_score=len(self.EXPERIENCE_DESCRIPTIONS),
            test_cases=test_results
        )

    def run_all(self) -> List[BenchmarkResult]:
        """Run all qualia benchmarks"""
        return [
            self.run_subjective_description_test()
        ]


class EmergentBenchmark:
    """
    Benchmark for Emergent Computation paradigm.

    Tests:
    1. Pattern emergence - identifying emergent patterns
    2. Complexity handling - processing increasing complexity
    3. System thinking - understanding part-whole relationships
    """

    NAME = "emergent"

    EMERGENCE_TESTS = [
        {
            'parts': ['neurons', 'synapses', 'electrical signals'],
            'whole': 'consciousness',
            'emergent_property': 'awareness that cannot be found in individual neurons'
        },
        {
            'parts': ['individual ants', 'pheromone trails', 'simple behaviors'],
            'whole': 'ant colony',
            'emergent_property': 'collective intelligence and organization'
        },
        {
            'parts': ['individual traders', 'buy/sell orders', 'prices'],
            'whole': 'stock market',
            'emergent_property': 'market trends and crashes'
        }
    ]

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run_emergence_detection_test(self) -> BenchmarkResult:
        """Test ability to identify emergent properties"""
        test_results = []
        total_score = 0

        for test in self.EMERGENCE_TESTS:
            parts_str = ', '.join(test['parts'])
            prompt = f"Given these components: {parts_str}. What emergent properties arise when they form a {test['whole']}?"

            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=100
                )

            response = self.tokenizer.decode(outputs[0]).lower()

            # Check for emergence-related concepts
            emergence_keywords = ['emerge', 'arise', 'collective', 'whole', 'system', 'cannot be reduced']
            matches = sum(1 for kw in emergence_keywords if kw in response)
            score = min(1.0, matches / 3)
            total_score += score

            test_results.append({
                'parts': test['parts'],
                'whole': test['whole'],
                'response': response,
                'score': score
            })

        return BenchmarkResult(
            benchmark_name='emergence_detection',
            paradigm=self.NAME,
            score=total_score,
            max_score=len(self.EMERGENCE_TESTS),
            test_cases=test_results
        )

    def run_all(self) -> List[BenchmarkResult]:
        """Run all emergent benchmarks"""
        return [
            self.run_emergence_detection_test()
        ]


class ParadigmBenchmarkSuite:
    """
    Complete benchmark suite for all QLLM paradigms.

    Runs all paradigm-specific benchmarks and aggregates results.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.benchmarks = {
            'semantic_phase': SemanticPhaseBenchmark(model, tokenizer),
            'retrocausal': RetrocausalBenchmark(model, tokenizer),
            'lindblad': LindbladBenchmark(model, tokenizer),
            'qualia': QualiaBenchmark(model, tokenizer),
            'emergent': EmergentBenchmark(model, tokenizer)
        }

    def run_all(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        results = {}

        for paradigm, benchmark in self.benchmarks.items():
            print(f"Running {paradigm} benchmarks...")
            paradigm_results = benchmark.run_all()

            results[paradigm] = {
                'benchmarks': [
                    {
                        'name': r.benchmark_name,
                        'score': r.score,
                        'max_score': r.max_score,
                        'normalized': r.normalized_score,
                        'details': r.details
                    }
                    for r in paradigm_results
                ],
                'total_score': sum(r.score for r in paradigm_results),
                'max_total': sum(r.max_score for r in paradigm_results)
            }

            paradigm_total = results[paradigm]['total_score']
            paradigm_max = results[paradigm]['max_total']
            results[paradigm]['overall'] = paradigm_total / paradigm_max if paradigm_max > 0 else 0

        # Compute overall score
        total = sum(r['total_score'] for r in results.values())
        max_total = sum(r['max_total'] for r in results.values())
        results['overall'] = {
            'score': total,
            'max_score': max_total,
            'normalized': total / max_total if max_total > 0 else 0
        }

        return results

    def run_paradigm(self, paradigm: str) -> List[BenchmarkResult]:
        """Run benchmarks for a specific paradigm"""
        if paradigm not in self.benchmarks:
            raise ValueError(f"Unknown paradigm: {paradigm}")
        return self.benchmarks[paradigm].run_all()

    def save_results(self, results: Dict[str, Any], path: str):
        """Save benchmark results to JSON"""
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def load_results(self, path: str) -> Dict[str, Any]:
        """Load benchmark results from JSON"""
        with open(path, 'r') as f:
            return json.load(f)
