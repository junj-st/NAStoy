"""
Tests for search space encoding/decoding and model building.
"""

import pytest
import torch
from nas.search_space import SearchSpace


def test_encode_decode():
    """Test that encoding and decoding are inverse operations."""
    space = SearchSpace(input_size=784, output_size=10)

    # Create sample config
    config = {
        'n_layers': 3,
        'hidden_sizes': [128, 64, 32],
        'activations': ['relu', 'tanh', 'sigmoid'],
        'dropouts': [0.1, 0.2, 0.0]
    }

    # Encode to tokens
    tokens = space.encode(config)
    assert len(tokens) == 13, "Token sequence should have 13 elements"

    # Decode back to config
    decoded = space.decode(tokens)

    # Assert configs are equal
    assert config == decoded, "Decoded config should match original"


def test_sample_random():
    """Test random architecture sampling."""
    space = SearchSpace(input_size=784, output_size=10)

    # Sample architecture
    config = space.sample_random()

    # Check it's valid
    assert space.is_valid(config), "Sampled config should be valid"

    # Check all fields present
    assert 'n_layers' in config
    assert 'hidden_sizes' in config
    assert 'activations' in config
    assert 'dropouts' in config

    # Check list lengths match n_layers
    assert len(config['hidden_sizes']) == config['n_layers']
    assert len(config['activations']) == config['n_layers']
    assert len(config['dropouts']) == config['n_layers']


def test_mutate():
    """Test architecture mutation."""
    space = SearchSpace(input_size=784, output_size=10)

    # Create config
    config = {
        'n_layers': 2,
        'hidden_sizes': [64, 64],
        'activations': ['relu', 'relu'],
        'dropouts': [0.1, 0.1]
    }

    # Mutate with high rate (should change something)
    mutated = space.mutate(config, mutation_rate=1.0)

    # Check result is valid
    assert space.is_valid(mutated), "Mutated config should be valid"

    # With mutation_rate=1.0, something should change
    # (not guaranteed to be different due to random choices, but very likely)


def test_build_model():
    """Test model building from config."""
    space = SearchSpace(input_size=784, output_size=10)

    # Create config
    config = {
        'n_layers': 2,
        'hidden_sizes': [128, 64],
        'activations': ['relu', 'tanh'],
        'dropouts': [0.1, 0.0]
    }

    # Build model
    model = space.build_model(config)

    # Check model can forward pass
    x = torch.randn(4, 784)
    y = model(x)

    # Check output shape is correct
    assert y.shape == (4, 10), f"Expected shape (4, 10), got {y.shape}"


def test_is_valid():
    """Test config validation."""
    space = SearchSpace(input_size=784, output_size=10)

    # Valid config
    valid_config = {
        'n_layers': 2,
        'hidden_sizes': [64, 128],
        'activations': ['relu', 'tanh'],
        'dropouts': [0.1, 0.2]
    }
    assert space.is_valid(valid_config), "Valid config should pass validation"

    # Invalid: missing key
    invalid_config1 = {
        'n_layers': 2,
        'hidden_sizes': [64, 128],
        'activations': ['relu', 'tanh']
        # missing dropouts
    }
    assert not space.is_valid(invalid_config1), "Config with missing key should fail"

    # Invalid: mismatched lengths
    invalid_config2 = {
        'n_layers': 2,
        'hidden_sizes': [64],  # Wrong length
        'activations': ['relu', 'tanh'],
        'dropouts': [0.1, 0.2]
    }
    assert not space.is_valid(invalid_config2), "Config with mismatched lengths should fail"

    # Invalid: out of range value
    invalid_config3 = {
        'n_layers': 2,
        'hidden_sizes': [64, 512],  # 512 not in HIDDEN_SIZES
        'activations': ['relu', 'tanh'],
        'dropouts': [0.1, 0.2]
    }
    assert not space.is_valid(invalid_config3), "Config with invalid hidden size should fail"
