"""
Tests for search strategies.
"""

import pytest
import tempfile
from nas.search_space import SearchSpace
from nas.evaluator import Evaluator
from nas.strategies.random_search import RandomSearch
from nas.strategies.evolutionary import EvolutionarySearch


def test_random_search():
    """Test random search runs."""
    space = SearchSpace(input_size=784, output_size=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = Evaluator(
            dataset='mnist',
            train_budget=1,
            batch_size=256,
            cache_dir=tmpdir
        )

        # Run random search with small n_evaluations
        search = RandomSearch(
            search_space=space,
            evaluator=evaluator,
            n_evaluations=3,
            seed=42
        )

        results = search.search()

        # Assert results are returned
        assert len(results) == 3, "Should return 3 results"

        # Assert results are sorted by accuracy (descending)
        for i in range(len(results) - 1):
            assert results[i]['val_acc'] >= results[i+1]['val_acc'], \
                "Results should be sorted by accuracy (descending)"


def test_tournament_selection():
    """Test tournament selection."""
    space = SearchSpace(input_size=784, output_size=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = Evaluator(
            dataset='mnist',
            train_budget=1,
            batch_size=256,
            cache_dir=tmpdir
        )

        search = EvolutionarySearch(
            search_space=space,
            evaluator=evaluator,
            population_size=5,
            n_generations=1,
            tournament_size=3,
            seed=42
        )

        # Initialize population
        search.initialize_population()

        # Get accuracies before selection
        accs = [ind['val_acc'] for ind in search.population]

        # Run tournament selection multiple times
        for _ in range(10):
            parent_config = search.tournament_selection()
            # Parent should be valid config
            assert search.search_space.is_valid(parent_config), \
                "Selected parent should be valid"


def test_population_diversity():
    """Test diversity computation."""
    space = SearchSpace(input_size=784, output_size=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = Evaluator(
            dataset='mnist',
            train_budget=1,
            batch_size=256,
            cache_dir=tmpdir
        )

        search = EvolutionarySearch(
            search_space=space,
            evaluator=evaluator,
            population_size=5,
            n_generations=1,
            seed=42
        )

        # Create population with identical individuals
        config = space.sample_random()
        model = space.build_model(config)
        result = evaluator.evaluate(model, config)
        search.population = [result] * 5

        # Compute diversity
        diversity = search.compute_diversity()

        # Assert diversity is 0 (all identical)
        assert diversity == 0.0, "Diversity should be 0 for identical individuals"

        # Create population with different individuals
        search.initialize_population()

        # Compute diversity
        diversity = search.compute_diversity()

        # Assert diversity > 0
        assert diversity > 0, "Diversity should be > 0 for different individuals"


def test_rl_controller_sampling():
    """Test RL controller can sample architectures."""
    from nas.strategies.rl_controller import LSTMController
    import torch

    # Create controller
    controller = LSTMController(hidden_size=32)

    # Sample architectures
    tokens, log_probs, entropy = controller.sample(n_samples=3)

    # Check shapes
    assert tokens.shape == (3, 13), f"Expected shape (3, 13), got {tokens.shape}"
    assert log_probs.shape == (3,), f"Expected shape (3,), got {log_probs.shape}"
    assert entropy.shape == (3,), f"Expected shape (3,), got {entropy.shape}"

    # Check tokens are in valid range
    for i, vocab_size in enumerate(controller.vocab_sizes):
        assert (tokens[:, i] < vocab_size).all(), \
            f"Token position {i} has invalid values"

    # Check entropy is positive
    assert (entropy > 0).all(), "Entropy should be positive"
