# NAS Toy - Research Findings

## Problem Statement

Neural Architecture Search (NAS) aims to automate the design of neural network architectures. This project explores three search strategies on a simplified search space to understand:
1. How different search strategies compare in sample efficiency
2. Whether zero-cost proxies can predict architecture performance
3. What architectural patterns emerge from different search methods

## Methods

### Search Space
Fully-connected MLPs with:
- 1-4 hidden layers
- Hidden sizes: {32, 64, 128, 256}
- Activations: {ReLU, Tanh, Sigmoid}
- Dropout: {0.0, 0.1, 0.2, 0.3}

Total search space size: ~16 million architectures

### Search Strategies

**Random Search**: Uniform sampling baseline

**Evolutionary Search**:
- Population size: 20
- Tournament selection (k=3)
- Mutation-only reproduction
- Elitist replacement

**RL Controller**:
- LSTM controller (hidden_size=64)
- REINFORCE with baseline
- Reward shaping: acc - λ*log(params)
- Entropy regularization

### Evaluation Protocol
- MNIST: 10 epochs training budget
- CIFAR-10: 20 epochs training budget
- Early stopping: 3 epochs patience
- Disk caching to avoid re-evaluation

### Zero-Cost Proxies
- **Gradient Norm**: L2 norm of gradients on single minibatch
- **SynFlow**: Synthetic gradient flow metric

## Results

### MNIST Results

| Strategy | Best Val Acc | Mean Top-10 | Evaluations | Wall Time | Best Architecture |
|----------|--------------|-------------|-------------|-----------|-------------------|
| Random   | TBD          | TBD         | 100         | TBD       | TBD               |
| Evolutionary | TBD      | TBD         | 1000        | TBD       | TBD               |
| RL Controller | TBD      | TBD         | 2000        | TBD       | TBD               |

### CIFAR-10 Results

| Strategy | Best Val Acc | Mean Top-10 | Evaluations | Wall Time | Best Architecture |
|----------|--------------|-------------|-------------|-----------|-------------------|
| Random   | TBD          | TBD         | 150         | TBD       | TBD               |
| Evolutionary | TBD      | TBD         | 1500        | TBD       | TBD               |
| RL Controller | TBD      | TBD         | 3000        | TBD       | TBD               |

### Zero-Cost Proxy Correlation

| Proxy | MNIST (Spearman ρ) | CIFAR-10 (Spearman ρ) |
|-------|--------------------|-----------------------|
| Gradient Norm | TBD         | TBD                   |
| SynFlow       | TBD         | TBD                   |

## Analysis

### Search Strategy Comparison
TBD: Analysis of which strategy finds best architectures fastest

### Architectural Patterns
TBD: Common patterns in top-performing architectures

### Proxy Effectiveness
TBD: Whether zero-cost proxies can reduce evaluation budget

## Limitations

1. **Small search space**: Only MLPs on toy datasets
2. **Limited training budget**: Architectures not fully converged
3. **No multi-objective optimization**: Only accuracy considered
4. **Single run**: No uncertainty quantification

## Future Work

- Extend to convolutional search spaces
- Multi-objective optimization (accuracy + efficiency)
- Transfer learning across datasets
- Predictor-based search strategies

## References

1. Zoph, B., & Le, Q. V. (2017). Neural Architecture Search with Reinforcement Learning. ICLR.
2. Real, E., et al. (2019). Regularized Evolution for Image Classifier Architecture Search. AAAI.
3. Abdelfattah, M. S., et al. (2021). Zero-Cost Proxies for Lightweight NAS. ICLR.
4. White, C., et al. (2021). How Powerful are Performance Predictors in Neural Architecture Search? NeurIPS.
