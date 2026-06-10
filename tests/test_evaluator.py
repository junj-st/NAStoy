"""
Tests for architecture evaluation and caching.
"""

import pytest
import tempfile
import time
from pathlib import Path
from nas.search_space import SearchSpace
from nas.evaluator import Evaluator


def test_cache_key_generation():
    """Test that cache keys are deterministic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = Evaluator(dataset='mnist', cache_dir=tmpdir, train_budget=1)

        config = {
            'n_layers': 2,
            'hidden_sizes': [64, 32],
            'activations': ['relu', 'relu'],
            'dropouts': [0.1, 0.0]
        }

        # Generate cache key twice
        key1 = evaluator._get_cache_key(config)
        key2 = evaluator._get_cache_key(config)

        # Assert keys are equal
        assert key1 == key2, "Cache keys should be deterministic"
        assert len(key1) == 64, "SHA256 hash should be 64 hex characters"


def test_caching():
    """Test that evaluation results are cached."""
    space = SearchSpace(input_size=784, output_size=10)
    config = {'n_layers': 1, 'hidden_sizes': [32], 'activations': ['relu'], 'dropouts': [0.0]}
    model = space.build_model(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = Evaluator(
            dataset='mnist',
            cache_dir=tmpdir,
            train_budget=1,
            batch_size=256
        )

        # First evaluation
        start = time.time()
        result1 = evaluator.evaluate(model, config)
        time1 = time.time() - start

        # Second evaluation (should use cache)
        start = time.time()
        result2 = evaluator.evaluate(model, config)
        time2 = time.time() - start

        # Results should be identical
        assert result1 == result2, "Cached results should match original"

        # Second evaluation should be much faster (cached)
        assert time2 < time1 / 10, f"Cached eval ({time2:.4f}s) should be much faster than original ({time1:.4f}s)"


def test_training_loop():
    """Test training loop executes."""
    space = SearchSpace(input_size=784, output_size=10)
    config = {'n_layers': 1, 'hidden_sizes': [64], 'activations': ['relu'], 'dropouts': [0.0]}
    model = space.build_model(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = Evaluator(
            dataset='mnist',
            cache_dir=tmpdir,
            train_budget=2,
            batch_size=256
        )

        result = evaluator.evaluate(model, config)

        # Assert result has expected fields
        assert 'val_acc' in result
        assert 'train_acc' in result
        assert 'train_time' in result
        assert 'n_params' in result
        assert 'config' in result
        assert 'best_epoch' in result

        # Assert val_acc > 0 and reasonable
        assert result['val_acc'] > 0, "Validation accuracy should be positive"
        assert result['val_acc'] <= 1.0, "Validation accuracy should be <= 1.0"

        # Assert training time is reasonable
        assert result['train_time'] > 0, "Training time should be positive"


def test_count_parameters():
    """Test parameter counting."""
    space = SearchSpace(input_size=784, output_size=10)

    # Simple 1-layer model: 784 -> 64 -> 10
    # Params: (784 * 64 + 64) + (64 * 10 + 10) = 50,240 + 650 = 50,890
    config = {'n_layers': 1, 'hidden_sizes': [64], 'activations': ['relu'], 'dropouts': [0.0]}
    model = space.build_model(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = Evaluator(dataset='mnist', cache_dir=tmpdir)
        n_params = evaluator.count_parameters(model)

        expected = (784 * 64 + 64) + (64 * 10 + 10)
        assert n_params == expected, f"Expected {expected} parameters, got {n_params}"
