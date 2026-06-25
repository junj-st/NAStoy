"""
Predictor Generalization Test

Tests if predictors trained on one dataset generalize to other datasets.
- Can MNIST predictor predict Fashion-MNIST performance?
- Cross-dataset transfer learning for NAS predictors
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator


def train_source_predictor(source_dataset='mnist'):
    """Train predictor on source dataset.

    Args:
        source_dataset: 'mnist' or 'cifar10'

    Returns:
        predictor, search_space, train_data
    """
    print(f"Training predictor on {source_dataset.upper()}...")

    # Load data
    if source_dataset == 'mnist':
        result_files = [
            'results/mnist_random.json',
            'results/mnist_evolutionary.json',
            'results/mnist_rl.json'
        ]
        input_size = 784
    else:
        result_files = [
            'results/cifar10_random.json',
            'results/cifar10_evolutionary.json'
        ]
        input_size = 3072

    all_results = []
    for file in result_files:
        if Path(file).exists():
            with open(file, 'r') as f:
                data = json.load(f)
                all_results.extend(data['results'])

    # Deduplicate
    import hashlib
    seen = set()
    unique_results = []
    for result in all_results:
        config_str = json.dumps(result['config'], sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        if config_hash not in seen:
            seen.add(config_hash)
            unique_results.append(result)

    # Encode and train
    search_space = SearchSpace(input_size=input_size, output_size=10)
    X = np.array([search_space.encode(r['config']) for r in unique_results])
    y = np.array([r['val_acc'] for r in unique_results])

    predictor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    predictor.fit(X, y)

    print(f"  Trained on {len(X)} architectures")
    print(f"  Mean accuracy: {y.mean():.4f}, Std: {y.std():.4f}")

    return predictor, search_space, (X, y)


def evaluate_on_target(predictor, source_space, target_dataset='fashionmnist', n_samples=50):
    """Evaluate source predictor on target dataset.

    Args:
        predictor: Trained predictor
        source_space: SearchSpace used for encoding
        target_dataset: Target dataset name
        n_samples: Number of architectures to evaluate on target

    Returns:
        results: List of (predicted, actual, config) tuples
    """
    print(f"\nEvaluating on {target_dataset.upper()}...")
    print(f"  Sampling {n_samples} random architectures...")

    # Setup evaluator for target dataset
    evaluator = Evaluator(
        dataset=target_dataset,
        train_budget=10,  # Same as MNIST
        cache_dir='results/cache'
    )

    # Sample and evaluate architectures
    results = []
    for i in tqdm(range(n_samples), desc="Evaluating"):
        # Sample random architecture
        config = source_space.sample_random()
        encoding = source_space.encode(config)

        # Predict with source predictor
        predicted = predictor.predict([encoding])[0]

        # Actually evaluate on target dataset
        model = source_space.build_model(config)
        eval_result = evaluator.evaluate(model, config)
        actual = eval_result['val_acc']

        results.append({
            'predicted': predicted,
            'actual': actual,
            'config': config,
            'n_params': eval_result['n_params']
        })

    return results


def analyze_generalization(source_data, target_results, source_name, target_name):
    """Analyze how well predictor generalizes.

    Args:
        source_data: (X, y) from source dataset
        target_results: Results from target evaluation
        source_name: Name of source dataset
        target_name: Name of target dataset
    """
    print(f"\n{'='*60}")
    print(f"GENERALIZATION ANALYSIS: {source_name.upper()} → {target_name.upper()}")
    print(f"{'='*60}")

    # Extract arrays
    predicted = np.array([r['predicted'] for r in target_results])
    actual = np.array([r['actual'] for r in target_results])

    # Compute metrics
    spearman_rho, spearman_p = spearmanr(predicted, actual)
    pearson_r, pearson_p = pearsonr(predicted, actual)
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)

    # Statistics
    print(f"\nTarget Dataset Statistics:")
    print(f"  Mean accuracy: {actual.mean():.4f}")
    print(f"  Std accuracy:  {actual.std():.4f}")
    print(f"  Min accuracy:  {actual.min():.4f}")
    print(f"  Max accuracy:  {actual.max():.4f}")

    print(f"\nPrediction Statistics:")
    print(f"  Mean predicted: {predicted.mean():.4f}")
    print(f"  Std predicted:  {predicted.std():.4f}")

    print(f"\nGeneralization Metrics:")
    print(f"  Spearman ρ:  {spearman_rho:.4f} (p={spearman_p:.4e})")
    print(f"  Pearson r:   {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"  R² score:    {r2:.4f}")
    print(f"  MAE:         {mae:.4f}")

    # Interpretation
    if abs(spearman_rho) > 0.7:
        strength = "STRONG"
    elif abs(spearman_rho) > 0.4:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    print(f"\n→ {strength} generalization from {source_name.upper()} to {target_name.upper()}")

    return {
        'spearman_rho': spearman_rho,
        'pearson_r': pearson_r,
        'r2': r2,
        'mae': mae,
        'strength': strength
    }


def plot_generalization(target_results, source_name, target_name, output_dir='plots'):
    """Plot predicted vs actual for target dataset."""
    Path(output_dir).mkdir(exist_ok=True)

    predicted = np.array([r['predicted'] for r in target_results])
    actual = np.array([r['actual'] for r in target_results])

    # Compute correlation for title
    rho, _ = spearmanr(predicted, actual)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted, actual, alpha=0.6, s=100, edgecolors='k', linewidths=0.5)

    # Add perfect prediction line
    min_val = min(predicted.min(), actual.min())
    max_val = max(predicted.max(), actual.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    # Best fit line
    z = np.polyfit(predicted, actual, 1)
    p = np.poly1d(z)
    x_line = np.linspace(predicted.min(), predicted.max(), 100)
    plt.plot(x_line, p(x_line), 'b-', alpha=0.8, linewidth=2, label='Best fit')

    plt.xlabel(f'Predicted on {source_name.upper()} Predictor', fontsize=12)
    plt.ylabel(f'Actual on {target_name.upper()}', fontsize=12)
    plt.title(f'Predictor Generalization: {source_name.upper()} → {target_name.upper()}\nSpearman ρ = {rho:.4f}',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = f'{output_dir}/generalization_{source_name}_to_{target_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_file}")
    plt.close()


def main():
    print("="*60)
    print("Predictor Generalization Test")
    print("="*60)

    # Test 1: MNIST → Fashion-MNIST
    print(f"\n{'='*60}")
    print("Test 1: MNIST → Fashion-MNIST")
    print(f"{'='*60}")

    predictor_mnist, space_mnist, data_mnist = train_source_predictor('mnist')
    results_fashion = evaluate_on_target(
        predictor_mnist,
        space_mnist,
        target_dataset='fashionmnist',
        n_samples=50
    )

    metrics_fashion = analyze_generalization(
        data_mnist,
        results_fashion,
        'mnist',
        'fashionmnist'
    )

    plot_generalization(results_fashion, 'mnist', 'fashionmnist')

    # Save results
    output_file = 'results/generalization_test.json'
    with open(output_file, 'w') as f:
        json.dump({
            'tests': [
                {
                    'source': 'mnist',
                    'target': 'fashionmnist',
                    'n_samples': len(results_fashion),
                    'metrics': metrics_fashion,
                    'results': results_fashion
                }
            ]
        }, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    print(f"\n{'='*60}")
    print("✓ Generalization Test Complete!")
    print(f"{'='*60}")

    # Summary
    print(f"\nSUMMARY:")
    print(f"  MNIST → Fashion-MNIST: {metrics_fashion['strength']} (ρ = {metrics_fashion['spearman_rho']:.3f})")


if __name__ == '__main__':
    main()
