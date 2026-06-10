"""
Evolutionary search strategy for NAS.

Uses tournament selection, mutation, and elitist replacement.
"""

import random
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator


class EvolutionarySearch:
    """Evolutionary architecture search."""

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Evaluator,
        population_size: int = 20,
        n_generations: int = 50,
        tournament_size: int = 3,
        mutation_rate: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Args:
            search_space: Architecture search space
            evaluator: Architecture evaluator
            population_size: Size of population
            n_generations: Number of generations to evolve
            tournament_size: Tournament size for selection
            mutation_rate: Mutation probability per hyperparameter
            seed: Random seed
        """
        self.search_space = search_space
        self.evaluator = evaluator
        self.population_size = population_size
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.population = []
        self.history = []

    def search(self) -> List[Dict]:
        """Run evolutionary search.

        Returns:
            results: Best architectures found
        """
        print(f"Starting evolutionary search...")
        print(f"  Population: {self.population_size}")
        print(f"  Generations: {self.n_generations}")
        print(f"  Tournament size: {self.tournament_size}")
        print(f"  Mutation rate: {self.mutation_rate}\n")

        # Initialize population
        self.initialize_population()

        # Track history for analysis
        self.history = [{
            'generation': 0,
            'best_acc': self.population[0]['val_acc'],
            'mean_acc': np.mean([ind['val_acc'] for ind in self.population]),
            'diversity': self.compute_diversity()
        }]

        # Evolution loop
        for gen in tqdm(range(1, self.n_generations + 1), desc="Evolution"):
            # Tournament selection
            parent_config = self.tournament_selection()

            # Mutate to create child
            child_config = self.search_space.mutate(parent_config, self.mutation_rate)

            # Evaluate child
            child_model = self.search_space.build_model(child_config)
            child_result = self.evaluator.evaluate(child_model, child_config)

            # Elitist replacement: replace worst if child is better
            worst_idx = min(range(len(self.population)),
                          key=lambda i: self.population[i]['val_acc'])

            if child_result['val_acc'] > self.population[worst_idx]['val_acc']:
                self.population[worst_idx] = child_result
                # Re-sort population
                self.population.sort(key=lambda x: x['val_acc'], reverse=True)

            # Track generation stats
            gen_stats = {
                'generation': gen,
                'best_acc': self.population[0]['val_acc'],
                'mean_acc': np.mean([ind['val_acc'] for ind in self.population]),
                'diversity': self.compute_diversity()
            }
            self.history.append(gen_stats)

            # Log progress
            if gen % 10 == 0:
                print(f"\nGen {gen}: Best={gen_stats['best_acc']:.4f}, "
                      f"Mean={gen_stats['mean_acc']:.4f}, "
                      f"Diversity={gen_stats['diversity']:.2f}")

        print(f"\nEvolutionary search complete!")
        print(f"Best architecture: {self.population[0]['val_acc']:.4f} accuracy")

        return self.population

    def initialize_population(self):
        """Initialize population with random architectures."""
        print(f"Initializing population of {self.population_size}...")
        self.population = []

        for i in tqdm(range(self.population_size), desc="Population Init"):
            # Sample random architecture
            config = self.search_space.sample_random()

            # Build and evaluate
            model = self.search_space.build_model(config)
            result = self.evaluator.evaluate(model, config)

            self.population.append(result)

        # Sort by fitness (validation accuracy)
        self.population.sort(key=lambda x: x['val_acc'], reverse=True)

        print(f"Population initialized. Best: {self.population[0]['val_acc']:.4f}")

    def tournament_selection(self) -> Dict:
        """Select parent via tournament selection.

        Returns:
            parent: Selected architecture config
        """
        # Randomly sample tournament_size individuals
        tournament = random.sample(self.population, self.tournament_size)

        # Return the one with best fitness (validation accuracy)
        winner = max(tournament, key=lambda x: x['val_acc'])
        return winner['config']

    def compute_diversity(self) -> float:
        """Compute population diversity.

        Returns:
            diversity: Average pairwise Hamming distance
        """
        # Encode all individuals as token sequences
        encodings = [self.search_space.encode(ind['config']) for ind in self.population]

        # Compute average Hamming distance between all pairs
        total_distance = 0
        n_pairs = 0

        for i in range(len(encodings)):
            for j in range(i + 1, len(encodings)):
                # Hamming distance: count different positions
                distance = sum(a != b for a, b in zip(encodings[i], encodings[j]))
                total_distance += distance
                n_pairs += 1

        if n_pairs == 0:
            return 0.0

        return total_distance / n_pairs

    def get_best_architecture(self) -> Dict:
        """Get best architecture found.

        Returns:
            best_result: Best evaluation result
        """
        if not self.population:
            raise ValueError("Population not initialized")
        return max(self.population, key=lambda x: x['val_acc'])
