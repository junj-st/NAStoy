"""
Search space definition for MLP architectures.

Encodes architectures as 13-token sequences:
[n_layers, h1, h2, h3, h4, act1, act2, act3, act4, drop1, drop2, drop3, drop4]

Example: 3-layer network [128, 64, 32] with ReLU and dropout 0.1
Config: {'n_layers': 3, 'hidden_sizes': [128, 64, 32], 'activations': ['relu', 'relu', 'relu'], 'dropouts': [0.1, 0.1, 0.1]}
Tokens: [2, 3, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
"""

import random
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class SearchSpace:
    """Defines the MLP architecture search space."""

    # Hyperparameter choices
    HIDDEN_SIZES = [32, 64, 128, 256]
    ACTIVATIONS = ['relu', 'tanh', 'sigmoid']
    DROPOUTS = [0.0, 0.1, 0.2, 0.3]
    MAX_LAYERS = 4

    def __init__(self, input_size: int = 784, output_size: int = 10):
        """
        Args:
            input_size: Input feature dimension (784 for MNIST, 3072 for CIFAR-10)
            output_size: Number of classes
        """
        self.input_size = input_size
        self.output_size = output_size

    def sample_random(self) -> Dict:
        """Sample a random valid architecture configuration.

        Returns:
            config: Dictionary with keys 'n_layers', 'hidden_sizes', 'activations', 'dropouts'
        """
        n_layers = random.randint(1, self.MAX_LAYERS)
        hidden_sizes = [random.choice(self.HIDDEN_SIZES) for _ in range(n_layers)]
        activations = [random.choice(self.ACTIVATIONS) for _ in range(n_layers)]
        dropouts = [random.choice(self.DROPOUTS) for _ in range(n_layers)]

        return {
            'n_layers': n_layers,
            'hidden_sizes': hidden_sizes,
            'activations': activations,
            'dropouts': dropouts
        }

    def encode(self, config: Dict) -> List[int]:
        """Convert architecture config to token sequence.

        Args:
            config: Architecture configuration dict

        Returns:
            tokens: List of 13 integers
        """
        n_layers = config['n_layers']
        tokens = [n_layers - 1]  # 0-3 for 1-4 layers

        # Encode hidden sizes (pad with 0 for unused layers)
        for i in range(self.MAX_LAYERS):
            if i < n_layers:
                tokens.append(self.HIDDEN_SIZES.index(config['hidden_sizes'][i]))
            else:
                tokens.append(0)

        # Encode activations (pad with 0 for unused layers)
        for i in range(self.MAX_LAYERS):
            if i < n_layers:
                tokens.append(self.ACTIVATIONS.index(config['activations'][i]))
            else:
                tokens.append(0)

        # Encode dropouts (pad with 0 for unused layers)
        for i in range(self.MAX_LAYERS):
            if i < n_layers:
                tokens.append(self.DROPOUTS.index(config['dropouts'][i]))
            else:
                tokens.append(0)

        return tokens

    def decode(self, tokens: List[int]) -> Dict:
        """Convert token sequence to architecture config.

        Args:
            tokens: List of 13 integers

        Returns:
            config: Architecture configuration dict
        """
        n_layers = tokens[0] + 1  # Convert 0-3 back to 1-4

        # Decode hidden sizes (only first n_layers matter)
        hidden_sizes = [self.HIDDEN_SIZES[tokens[1 + i]] for i in range(n_layers)]

        # Decode activations (only first n_layers matter)
        activations = [self.ACTIVATIONS[tokens[5 + i]] for i in range(n_layers)]

        # Decode dropouts (only first n_layers matter)
        dropouts = [self.DROPOUTS[tokens[9 + i]] for i in range(n_layers)]

        return {
            'n_layers': n_layers,
            'hidden_sizes': hidden_sizes,
            'activations': activations,
            'dropouts': dropouts
        }

    def mutate(self, config: Dict, mutation_rate: float = 0.3) -> Dict:
        """Apply random mutations to an architecture.

        Args:
            config: Original architecture configuration
            mutation_rate: Probability of mutating each hyperparameter

        Returns:
            mutated_config: New architecture configuration
        """
        mutated = {
            'n_layers': config['n_layers'],
            'hidden_sizes': config['hidden_sizes'][:],
            'activations': config['activations'][:],
            'dropouts': config['dropouts'][:]
        }

        # Mutate number of layers
        if random.random() < mutation_rate:
            old_n_layers = mutated['n_layers']
            mutated['n_layers'] = random.randint(1, self.MAX_LAYERS)

            # Adjust lists if number of layers changed
            if mutated['n_layers'] > old_n_layers:
                # Add new layers
                for _ in range(mutated['n_layers'] - old_n_layers):
                    mutated['hidden_sizes'].append(random.choice(self.HIDDEN_SIZES))
                    mutated['activations'].append(random.choice(self.ACTIVATIONS))
                    mutated['dropouts'].append(random.choice(self.DROPOUTS))
            elif mutated['n_layers'] < old_n_layers:
                # Remove layers
                mutated['hidden_sizes'] = mutated['hidden_sizes'][:mutated['n_layers']]
                mutated['activations'] = mutated['activations'][:mutated['n_layers']]
                mutated['dropouts'] = mutated['dropouts'][:mutated['n_layers']]

        # Mutate each layer's parameters
        for i in range(mutated['n_layers']):
            if random.random() < mutation_rate:
                mutated['hidden_sizes'][i] = random.choice(self.HIDDEN_SIZES)
            if random.random() < mutation_rate:
                mutated['activations'][i] = random.choice(self.ACTIVATIONS)
            if random.random() < mutation_rate:
                mutated['dropouts'][i] = random.choice(self.DROPOUTS)

        return mutated

    def is_valid(self, config: Dict) -> bool:
        """Check if architecture configuration is valid.

        Args:
            config: Architecture configuration to validate

        Returns:
            valid: True if config is valid
        """
        # Check required keys
        required_keys = ['n_layers', 'hidden_sizes', 'activations', 'dropouts']
        if not all(key in config for key in required_keys):
            return False

        n_layers = config['n_layers']

        # Check n_layers in valid range
        if not (1 <= n_layers <= self.MAX_LAYERS):
            return False

        # Check list lengths match n_layers
        if len(config['hidden_sizes']) != n_layers:
            return False
        if len(config['activations']) != n_layers:
            return False
        if len(config['dropouts']) != n_layers:
            return False

        # Check all values are valid
        for h in config['hidden_sizes']:
            if h not in self.HIDDEN_SIZES:
                return False

        for a in config['activations']:
            if a not in self.ACTIVATIONS:
                return False

        for d in config['dropouts']:
            if d not in self.DROPOUTS:
                return False

        return True

    def build_model(self, config: Dict) -> nn.Module:
        """Build PyTorch model from architecture configuration.

        Args:
            config: Architecture configuration

        Returns:
            model: PyTorch nn.Module
        """
        layers = []
        input_dim = self.input_size

        for i in range(config['n_layers']):
            hidden_size = config['hidden_sizes'][i]
            activation = config['activations'][i]
            dropout = config['dropouts'][i]

            # Linear layer
            layers.append(nn.Linear(input_dim, hidden_size))

            # Activation
            layers.append(self.get_activation(activation))

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_size

        # Output layer
        layers.append(nn.Linear(input_dim, self.output_size))

        return nn.Sequential(*layers)

    def get_activation(self, name: str) -> nn.Module:
        """Get activation module by name.

        Args:
            name: Activation name ('relu', 'tanh', 'sigmoid')

        Returns:
            activation: PyTorch activation module
        """
        if name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {name}")
