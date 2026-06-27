"""
Meta-Learning Predictor

Trains a predictor on multiple datasets simultaneously to learn
cross-dataset architectural patterns. Tests if this improves generalization.
"""

import json
import numpy as np
import hashlib
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator


def load_multi_dataset_data(datasets=['mnist', 'cifar10']):
    """Load data from multiple datasets for meta-learning.

    Args:
        datasets: List of dataset names

    Returns:
        X, y, dataset_ids: Features, targets, dataset indicators
        search_space: SearchSpace object (shared)
    """
    print("Loading multi-dataset data...")

    all_X = []
    all_y = []
    all_dataset_ids = []

    for dataset in datasets:
        print(f"\n  Loading {dataset.upper()}...")

        if dataset == 'mnist':
            result_files = [
                'results/mnist_random.json',
                'results/mnist_evolutionary.json',
                'results/mnist_rl.json'
            ]
            input_size = 784
        elif dataset == 'cifar10':
            result_files = [
                'results/cifar10_random.json',
                'results/cifar10_evolutionary.json'
            ]
            input_size = 3072
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Load results
        results = []
        for file in result_files:
            if Path(file).exists():
                with open(file, 'r') as f:
                    data = json.load(f)
                    results.extend(data['results'])

        # Deduplicate
        seen = set()
        unique_results = []
        for result in results:
            config_str = json.dumps(result['config'], sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()
            if config_hash not in seen:
                seen.add(config_hash)
                unique_results.append(result)

        # Encode (use same search space for all - 784 input for simplicity)
        search_space = SearchSpace(input_size=784, output_size=10)
        X = np.array([search_space.encode(r['config']) for r in unique_results])
        y = np.array([r['val_acc'] for r in unique_results])

        print(f"    {len(unique_results)} unique architectures")
        print(f"    Mean accuracy: {y.mean():.4f} ± {y.std():.4f}")

        all_X.append(X)
        all_y.append(y)
        all_dataset_ids.extend([dataset] * len(X))

    # Combine
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)

    print(f"\n  Combined: {len(X_combined)} total architectures")

    return X_combined, y_combined, all_dataset_ids, search_space


def train_meta_predictor(datasets=['mnist', 'cifar10']):
    """Train meta-learning predictor on multiple datasets.

    Args:
        datasets: List of dataset names

    Returns:
        predictor: Trained model
        search_space: SearchSpace
        train_data: Training data info
    """
    print(f"\n{'='*60}")
    print("Meta-Learning Predictor")
    print(f"{'='*60}")
    print(f"  Datasets: {', '.join([d.upper() for d in datasets])}")

    # Load data
    X, y, dataset_ids, search_space = load_multi_dataset_data(datasets)

    # Add dataset indicator as feature (one-hot encoding)
    unique_datasets = list(set(dataset_ids))
    dataset_features = np.zeros((len(X), len(unique_datasets)))
    for i, ds_id in enumerate(dataset_ids):
        ds_idx = unique_datasets.index(ds_id)
        dataset_features[i, ds_idx] = 1

    # Augment features with dataset indicators
    X_augmented = np.hstack([X, dataset_features])

    print(f"\n  Feature dimension: {X_augmented.shape[1]}")
    print(f"    Architecture encoding: {X.shape[1]}")
    print(f"    Dataset indicators: {dataset_features.shape[1]}")

    # Train
    print(f"\n  Training meta-predictor...")
    predictor = GradientBoostingRegressor(
        n_estimators=150,  # More trees for meta-learning
        max_depth=6,        # Deeper trees
        random_state=42
    )
    predictor.fit(X_augmented, y)

    print(f"  ✓ Meta-predictor trained on {len(X)} architectures")

    return predictor, search_space, {
        'n_architectures': len(X),
        'datasets': unique_datasets,
        'X_shape': X_augmented.shape
    }


def test_generalization(meta_predictor, search_space, train_datasets,
                       test_dataset='fashionmnist', n_samples=50):
    """Test meta-predictor generalization to new dataset.

    Args:
        meta_predictor: Trained meta-predictor
        search_space: SearchSpace for encoding
        train_datasets: Datasets used for training
        test_dataset: New dataset to test on
        n_samples: Number of test samples

    Returns:
        results: Test results
    """
    print(f"\n{'='*60}")
    print(f"Testing Generalization: {'+'.join([d.upper() for d in train_datasets])} → {test_dataset.upper()}")
    print(f"{'='*60}")

    # Setup evaluator
    evaluator = Evaluator(
        dataset=test_dataset,
        train_budget=10,
        cache_dir='results/cache'
    )

    # Generate test architectures
    print(f"\n  Sampling {n_samples} random architectures...")
    results = []
    for _ in tqdm(range(n_samples), desc="Evaluating"):
        config = search_space.sample_random()
        encoding = search_space.encode(config)

        # Augment with dataset indicator (all zeros for new dataset)
        dataset_features = np.zeros(len(train_datasets))
        encoding_augmented = np.hstack([encoding, dataset_features])

        # Predict
        predicted = meta_predictor.predict([encoding_augmented])[0]

        # Evaluate
        model = search_space.build_model(config)
        eval_result = evaluator.evaluate(model, config)
        actual = eval_result['val_acc']

        results.append({
            'predicted': predicted,
            'actual': actual,
            'config': config
        })

    # Compute metrics
    predicted = np.array([r['predicted'] for r in results])
    actual = np.array([r['actual'] for r in results])

    spearman_rho, spearman_p = spearmanr(predicted, actual)
    pearson_r, pearson_p = pearsonr(predicted, actual)
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)

    print(f"\n  Generalization Metrics:")
    print(f"    Spearman ρ: {spearman_rho:.4f} (p={spearman_p:.4e})")
    print(f"    Pearson r:  {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"    R²:         {r2:.4f}")
    print(f"    MAE:        {mae:.4f}")

    # Interpretation
    if abs(spearman_rho) > 0.7:
        strength = "STRONG"
    elif abs(spearman_rho) > 0.4:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    print(f"\n  → {strength} generalization to {test_dataset.upper()}")

    return {
        'results': results,
        'metrics': {
            'spearman_rho': spearman_rho,
            'pearson_r': pearson_r,
            'r2': r2,
            'mae': mae,
            'strength': strength
        }
    }


def compare_with_single_dataset(test_dataset='fashionmnist'):
    """Compare meta-predictor with single-dataset predictor.

    Args:
        test_dataset: Test dataset

    Returns:
        comparison: Comparison results
    """
    print(f"\n{'='*60}")
    print("COMPARISON: Meta-Learning vs Single-Dataset")
    print(f"{'='*60}")

    # Load single-dataset results (from previous generalization test)
    single_dataset_file = 'results/generalization_test.json'
    if Path(single_dataset_file).exists():
        with open(single_dataset_file, 'r') as f:
            single_data = json.load(f)

        single_rho = single_data['tests'][0]['metrics']['spearman_rho']
        single_mae = single_data['tests'][0]['metrics']['mae']

        print(f"\n  Single-dataset (MNIST only):")
        print(f"    Spearman ρ: {single_rho:.4f}")
        print(f"    MAE:        {single_mae:.4f}")

        return single_rho, single_mae
    else:
        print("  ⚠ No single-dataset baseline found")
        return None, None


def visualize_meta_learning(meta_results, single_rho=None):
    """Visualize meta-learning results."""
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)

    predicted = np.array([r['predicted'] for r in meta_results['results']])
    actual = np.array([r['actual'] for r in meta_results['results']])

    rho = meta_results['metrics']['spearman_rho']

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted, actual, alpha=0.6, s=100,
               edgecolors='k', linewidths=0.5, color='purple')

    # Perfect line
    min_val = min(predicted.min(), actual.min())
    max_val = max(predicted.max(), actual.max())
    plt.plot([min_val, max_val], [min_val, max_val],
            'r--', linewidth=2, label='Perfect')

    # Best fit
    z = np.polyfit(predicted, actual, 1)
    p = np.poly1d(z)
    x_line = np.linspace(predicted.min(), predicted.max(), 100)
    plt.plot(x_line, p(x_line), 'b-', alpha=0.8,
            linewidth=2, label='Best fit')

    plt.xlabel('Predicted (Meta-Predictor)', fontsize=12)
    plt.ylabel('Actual (Fashion-MNIST)', fontsize=12)

    title = f'Meta-Learning Predictor: MNIST+CIFAR10 → Fashion-MNIST\nSpearman ρ = {rho:.4f}'
    if single_rho is not None:
        improvement = rho - single_rho
        title += f' (vs single: {single_rho:.4f}, improvement: {improvement:+.4f})'

    plt.title(title, fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = f'{output_dir}/meta_learning_predictor.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_file}")
    plt.close()


def main():
    print("="*60)
    print("Meta-Learning Predictor")
    print("="*60)

    # Train meta-predictor on MNIST + CIFAR-10
    meta_predictor, search_space, train_info = train_meta_predictor(
        datasets=['mnist', 'cifar10']
    )

    # Test on Fashion-MNIST
    meta_results = test_generalization(
        meta_predictor,
        search_space,
        train_datasets=['mnist', 'cifar10'],
        test_dataset='fashionmnist',
        n_samples=50
    )

    # Compare with single-dataset
    single_rho, single_mae = compare_with_single_dataset('fashionmnist')

    if single_rho is not None:
        meta_rho = meta_results['metrics']['spearman_rho']
        improvement = meta_rho - single_rho

        print(f"\n{'='*60}")
        if improvement > 0:
            print(f"✓ Meta-learning IMPROVED by {improvement:+.4f}")
        else:
            print(f"✗ Meta-learning did not improve ({improvement:+.4f})")
        print(f"{'='*60}")

    # Visualize
    visualize_meta_learning(meta_results, single_rho)

    # Save
    output_file = 'results/meta_predictor.json'
    with open(output_file, 'w') as f:
        json.dump({
            'train_datasets': ['mnist', 'cifar10'],
            'test_dataset': 'fashionmnist',
            'train_info': train_info,
            'test_results': meta_results,
            'comparison': {
                'meta_rho': meta_results['metrics']['spearman_rho'],
                'single_rho': single_rho,
                'improvement': meta_results['metrics']['spearman_rho'] - single_rho if single_rho else None
            }
        }, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    print(f"\n{'='*60}")
    print("✓ Meta-Learning Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
