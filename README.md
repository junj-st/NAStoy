A minimal, educational implementation of Neural Architecture Search (NAS) for exploring different search strategies on toy datasets.

## Overview

This project implements three NAS strategies:
- **Random Search**: Baseline uniform sampling
- **Evolutionary Search**: Population-based with mutation and selection
- **RL Controller**: REINFORCE-based autoregressive architecture generation
- **Predictor-Guided Search**: Uses a surrogate model to filter candidates before evaluation
- **Early Stopping Predictor**: LSTM-based predictor to estimate final accuracy from partial training curves
- **Multi-Fidelity Optimization**: Successive Halving for efficient resource allocation
- **DARTS**: Gradient-based differentiable architecture search
- **Hardware-Aware NAS**: Multi-objective optimization for accuracy, latency, size, and FLOPs

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

Run random search on CIFAR-10:
```bash
python experiments/cifar10_random.py
```

Run evolutionary search on CIFAR-10:
```bash
python experiments/cifar10_evolutionary.py
```

Run RL-based search on CIFAR-10:
```bash
python experiments/cifar10_rl.py
```

Advanced experiments:
```bash
python experiments/predictor_guided_search.py    # Surrogate-based search
python experiments/early_stopping_predictor.py   # Learning curve prediction
python experiments/multi_fidelity_search.py      # Successive Halving
python experiments/darts_search.py               # Gradient-based NAS
python experiments/hardware_aware_nas.py         # Hardware-aware optimization
```

## Analysis

See [FINDINGS.md](FINDINGS.md) for detailed analysis and research insights.

## References

- Zoph & Le (2017). Neural Architecture Search with Reinforcement Learning
- Real et al. (2019). Regularized Evolution for Image Classifier Architecture Search
- Abdelfattah et al. (2021). Zero-Cost Proxies for Lightweight NAS
- Liu et al. (2019). DARTS: Differentiable Architecture Search
- Li et al. (2020). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization
- Cai et al. (2020). Once-for-All: Train One Network and Specialize it for Efficient Deployment
- Baker et al. (2018). Accelerating Neural Architecture Search using Performance Prediction
