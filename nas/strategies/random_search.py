"""
Random search strategy for NAS.

Baseline: uniformly sample architectures from search space.
"""

import random
import time
from typing import List, Dict, Optional
from tqdm import tqdm

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator


class RandomSearch:
    """Random architecture search."""

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Evaluator,
        n_evaluations: int = 100,
        seed: Optional[int] = None
    ):
        """
        Args:
            search_space: Architecture search space
            evaluator: Architecture evaluator
            n_evaluations: Number of architectures to evaluate
            seed: Random seed
        """
        self.search_space = search_space
        self.evaluator = evaluator
        self.n_evaluations = n_evaluations

        if seed is not None:
            random.seed(seed)

        self.results = []

    def search(self) -> List[Dict]:
        """Run random search.

        Returns:
            results: List of evaluation results sorted by accuracy (descending)
        """
        print(f"Starting random search with {self.n_evaluations} evaluations...")

        for i in tqdm(range(self.n_evaluations), desc="Random Search"):
            # Sample random architecture
            config = self.search_space.sample_random()

            # Build model
            model = self.search_space.build_model(config)

            # Evaluate
            result = self.evaluator.evaluate(model, config)

            # Store result
            self.results.append(result)

            # Log progress
            if (i + 1) % 10 == 0 or i == 0:
                best_so_far = max(self.results, key=lambda x: x['val_acc'])
                print(f"\nEval {i+1}/{self.n_evaluations}: "
                      f"Current acc={result['val_acc']:.4f}, "
                      f"Best so far={best_so_far['val_acc']:.4f}")

        # Sort results by validation accuracy (descending)
        self.results.sort(key=lambda x: x['val_acc'], reverse=True)

        print(f"\nRandom search complete!")
        print(f"Best architecture: {self.results[0]['val_acc']:.4f} accuracy")

        return self.results

    def get_best_architecture(self) -> Dict:
        """Get best architecture found.

        Returns:
            best_result: Evaluation result of best architecture
        """
        if not self.results:
            raise ValueError("No architectures evaluated yet")
        return max(self.results, key=lambda x: x['val_acc'])
