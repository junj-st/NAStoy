# Neural Architecture Search - Experimental Findings

## Executive Summary

This project implements and compares three NAS strategies on a simplified MLP search space:

**Main Results:**
- **Best accuracy**: RL Controller (98.40% on MNIST)
- **Most efficient**: Evolutionary search (98.27% in 70 evals)
- **Best baseline**: Random search (98.16% in 100 evals)
- **Zero-cost proxies**: Ineffective for this search space (ρ < 0.3)

**Key Insight**: For small search spaces, evolutionary search offers the best quality/cost trade-off. Zero-cost proxies show limited predictive power on simple MLPs.

---

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

| Strategy | Best Val Acc | Evaluations | Wall Time | Best Architecture |
|----------|--------------|-------------|-----------|-------------------|
| Random   | **98.16%**   | 100         | ~42 min   | 2L: [256, 256], sigmoid→relu, drop 0.3 |
| Evolutionary | **98.27%** | 70         | ~20 min   | 2L: [256, 64], relu→tanh, drop 0.1 |
| RL Controller | **98.40%** ✓ | 2,000      | ~8.7 hrs  | 2L: [256, 256], relu→relu, drop 0.2→0.3 |

**Key Findings:**
- RL Controller achieved highest accuracy (98.40%) but required 20x more evaluations
- Evolutionary search found competitive results (98.27%) with only 70 evaluations
- Random search baseline reached 98.16% with 100 evaluations
- All strategies converged to 2-layer architectures with 256 hidden units

### CIFAR-10 Results

**Note**: CIFAR-10 experiments encountered cache collision bug where dataset name was not included in cache keys, causing MNIST results to be reused. This has been fixed but clean results are pending.

| Strategy | Status | Notes |
|----------|--------|-------|
| Random   | Buggy cache | Mixed MNIST/CIFAR-10 results (98% vs 43-52%) |
| Evolutionary | Not run | - |
| RL Controller | Not run | - |

### Zero-Cost Proxy Correlation (MNIST)

| Proxy | Spearman ρ | p-value | Interpretation |
|-------|------------|---------|----------------|
| Gradient Norm | 0.0544 | 0.52    | **No correlation** (not significant) |
| SynFlow       | 0.2976 | 0.0003  | **Weak correlation** (significant) |

**Key Findings:**
- Tested on 141 unique architectures from all three search strategies
- Gradient norm shows no predictive power for MLP performance
- SynFlow shows weak but statistically significant correlation
- Neither proxy is reliable for architecture ranking on this search space
- Full training remains necessary for accurate evaluation

## Analysis

### Search Strategy Comparison

**Sample Efficiency:**
1. **Evolutionary (Winner for efficiency)**: Best accuracy/evaluations ratio
   - Reached 98.27% in just 70 evaluations (~20 minutes)
   - Population-based approach leverages mutation to explore promising regions
   - Diversity decreased from 6.67 → 1.88, showing convergence to good solutions

2. **Random (Strong baseline)**: Simple but effective
   - Achieved 98.16% with 100 evaluations
   - No learning overhead, pure exploration
   - Surprisingly competitive on this simple search space

3. **RL Controller (Best final accuracy)**: Highest quality but computationally expensive
   - Achieved 98.40% (best overall) after 2,000 evaluations
   - Learned to generate high-performing architectures (avg accuracy improved 97.17% → 98.02%)
   - Entropy decay (1.29 → 1.10) shows successful exploration-to-exploitation transition
   - Required 8.7 hours vs ~20 minutes for evolutionary

**Verdict**: For small search spaces like this, evolutionary search offers the best trade-off between quality and computational cost. RL controller is overkill unless you need the absolute best architecture.

### Architectural Patterns

**Convergence across strategies:**
- All three strategies converged to **2-layer architectures**
- **256 hidden units** in first layer was universally preferred
- Deeper networks (3-4 layers) were less successful
- Smaller networks (32-64 units) underperformed

**Best architecture (RL Controller):**
```python
{
  'n_layers': 2,
  'hidden_sizes': [256, 256],
  'activations': ['relu', 'relu'],
  'dropouts': [0.2, 0.3],
  'params': 269,322,
  'accuracy': 98.40%
}
```

**Insights:**
- MNIST is simple enough that 2 layers suffice
- Large first layer (256 units) captures most discriminative features
- ReLU consistently outperformed Tanh and Sigmoid
- Moderate dropout (0.1-0.3) helped regularization
- Parameter count sweet spot: ~200K-270K parameters

### Proxy Effectiveness

**Gradient Norm (Failed):**
- ρ = 0.0544, p = 0.52 (not statistically significant)
- No predictive power for architecture quality
- Theory: Gradient magnitude alone doesn't capture trainability

**SynFlow (Limited success):**
- ρ = 0.2976, p < 0.001 (statistically significant)
- Weak correlation suggests some signal but not actionable
- Could potentially be used for coarse filtering (top 50%) but not fine-grained ranking

**Why proxies failed:**
1. **Search space too simple**: MLPs on MNIST have narrow performance variance
2. **Proxies designed for CNNs**: SynFlow was developed for convolutional architectures
3. **Dataset too easy**: MNIST allows many architectures to succeed
4. **Limited dynamic range**: Most architectures scored 95-98%, leaving little signal

**Conclusion**: Zero-cost proxies are not effective for this search space. Full training remains necessary. Proxies might show stronger correlations on harder datasets (CIFAR-10, ImageNet) or more complex search spaces (CNNs, Transformers).

## Limitations

1. **Simple search space**: Only fully-connected MLPs tested
   - Real-world NAS typically uses convolutional or transformer search spaces
   - Limited architectural diversity (just layers, widths, activations, dropout)

2. **Easy dataset**: MNIST allows many architectures to succeed
   - Narrow performance variance (95-98%) makes ranking difficult
   - Zero-cost proxies might show stronger correlations on CIFAR-10/ImageNet

3. **Limited training budget**: 10 epochs for MNIST
   - Architectures not fully converged
   - Early stopping after 3 epochs without improvement
   - May favor fast-learning architectures over those that need more training

4. **Single seed**: No statistical significance testing
   - Results could vary with different random seeds
   - No confidence intervals or error bars
   - Evolutionary and RL results particularly sensitive to initialization

5. **Cache collision bug**: CIFAR-10 results contaminated
   - Fixed in code but clean results not yet generated
   - Prevented cross-dataset analysis

6. **No multi-objective optimization**: Only accuracy considered
   - Real applications care about latency, memory, FLOPs
   - Best architecture (269K params) may not be most efficient

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
