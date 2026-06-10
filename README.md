A minimal, educational implementation of Neural Architecture Search (NAS) for exploring different search strategies on toy datasets.

## Overview

This project implements three NAS strategies:
- **Random Search**: Baseline uniform sampling
- **Evolutionary Search**: Population-based with mutation and selection
- **RL Controller**: REINFORCE-based autoregressive architecture generation

## Search Space

The search space consists of fully-connected MLPs with the following hyperparameters:
- **Number of layers**: 1-4 hidden layers
- **Hidden sizes**: {32, 64, 128, 256} per layer
- **Activations**: {ReLU, Tanh, Sigmoid} per layer
- **Dropout rates**: {0.0, 0.1, 0.2, 0.3} per layer

Architectures are encoded as 13-token sequences:
```
[n_layers, h1, h2, h3, h4, act1, act2, act3, act4, drop1, drop2, drop3, drop4]
```

Example: A 3-layer network [128, 64, 32] with ReLU and dropout 0.1:
```
[2, 3, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run random search on MNIST:
```bash
python experiments/mnist_random.py
```

Run evolutionary search on MNIST:
```bash
python experiments/mnist_evolutionary.py
```

Run RL-based search on MNIST:
```bash
python experiments/mnist_rl.py
```

## Project Structure

```
nas-toy/
├── nas/                      # Core NAS implementation
│   ├── search_space.py       # Architecture encoding and model building
│   ├── evaluator.py          # Training and evaluation with caching
│   ├── proxy.py              # Zero-cost proxy estimators
│   ├── viz.py                # Visualization utilities
│   └── strategies/           # Search strategies
│       ├── random_search.py
│       ├── evolutionary.py
│       └── rl_controller.py
├── experiments/              # Experiment scripts
├── configs/                  # YAML configuration files
├── tests/                    # Unit tests
└── results/                  # Generated results and plots
```

## Analysis

See [FINDINGS.md](FINDINGS.md) for detailed analysis and research insights.

## References

- Zoph & Le (2017). Neural Architecture Search with Reinforcement Learning
- Real et al. (2019). Regularized Evolution for Image Classifier Architecture Search
- Abdelfattah et al. (2021). Zero-Cost Proxies for Lightweight NAS
