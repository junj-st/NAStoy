"""
Architecture Performance Predictor

Trains models to predict architecture accuracy from encoding.
Compares learned predictors with zero-cost proxies.
"""

import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr

from nas.search_space import SearchSpace


def load_all_architectures(dataset='mnist'):
    """Load all evaluated architectures for a dataset.

    Args:
        dataset: 'mnist' or 'cifar10'

    Returns:
        architectures: List of (encoding, accuracy, config) tuples
    """
    # Result files for this dataset
    if dataset == 'mnist':
        result_files = [
            'results/mnist_random.json',
            'results/mnist_evolutionary.json',
            'results/mnist_rl.json'
        ]
    else:  # cifar10
        result_files = [
            'results/cifar10_random.json',
            'results/cifar10_evolutionary.json'
        ]

    # Load all results
    all_results = []
    for file in result_files:
        if Path(file).exists():
            with open(file, 'r') as f:
                data = json.load(f)
                all_results.extend(data['results'])

    # Deduplicate by config
    seen_configs = set()
    unique_results = []
    for result in all_results:
        config_str = json.dumps(result['config'], sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        if config_hash not in seen_configs:
            seen_configs.add(config_hash)
            unique_results.append(result)

    # Encode architectures
    search_space = SearchSpace(input_size=784 if dataset == 'mnist' else 3072, output_size=10)
    architectures = []

    for result in unique_results:
        encoding = search_space.encode(result['config'])
        accuracy = result['val_acc']
        architectures.append((encoding, accuracy, result['config']))

    return architectures


def train_predictors(X_train, y_train, X_test, y_test):
    """Train multiple predictor models.

    Returns:
        results: Dict of model_name -> {predictions, metrics}
    """
    models = {
        'Linear (Ridge)': Ridge(alpha=1.0),
        'Linear (Lasso)': Lasso(alpha=0.001, max_iter=5000),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_spearman, _ = spearmanr(test_pred, y_test)
        test_pearson, _ = pearsonr(test_pred, y_test)

        results[name] = {
            'model': model,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_spearman': test_spearman,
            'test_pearson': test_pearson
        }

        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test Spearman ρ: {test_spearman:.4f}")

    return results


def plot_predictions(results, y_test, dataset_name, output_dir='plots'):
    """Plot predicted vs actual for all models."""
    Path(output_dir).mkdir(exist_ok=True)

    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]

        test_pred = result['test_pred']

        # Scatter plot
        ax.scatter(y_test, test_pred, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

        # Perfect prediction line
        min_val, max_val = y_test.min(), y_test.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        # Labels
        ax.set_xlabel('Actual Accuracy', fontsize=11)
        ax.set_ylabel('Predicted Accuracy', fontsize=11)
        ax.set_title(f'{name}\nR² = {result["test_r2"]:.3f}, ρ = {result["test_spearman"]:.3f}',
                    fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide extra subplot
    if n_models < 6:
        for idx in range(n_models, 6):
            axes[idx].axis('off')

    plt.suptitle(f'Architecture Performance Prediction - {dataset_name.upper()}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = f'{output_dir}/predictor_{dataset_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved predictions plot to {output_file}")
    plt.close()


def analyze_feature_importance(model, feature_names, dataset_name, output_dir='plots'):
    """Analyze and plot feature importance for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Architecture Feature', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title(f'Feature Importance - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = f'{output_dir}/feature_importance_{dataset_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved feature importance to {output_file}")
    plt.close()

    # Print top features
    print(f"\nTop 5 most important features:")
    for i in range(min(5, len(importances))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


def compare_with_proxies(dataset_name='mnist'):
    """Compare predictor with zero-cost proxies."""
    proxy_file = 'results/proxy_correlation.json'

    if not Path(proxy_file).exists():
        print("\nProxy correlation data not found. Skipping comparison.")
        return

    with open(proxy_file, 'r') as f:
        proxy_data = json.load(f)

    print(f"\n{'='*60}")
    print("COMPARISON: Learned Predictors vs Zero-Cost Proxies")
    print(f"{'='*60}")
    print(f"\nZero-Cost Proxies (MNIST only):")
    print(f"  Gradient Norm:  ρ = {proxy_data['correlations']['grad_norm']['rho']:.4f}")
    print(f"  SynFlow:        ρ = {proxy_data['correlations']['synflow']['rho']:.4f}")


def main():
    print("="*60)
    print("Architecture Performance Predictor")
    print("="*60)

    # Feature names for interpretability
    feature_names = [
        'n_layers',
        'h1', 'h2', 'h3', 'h4',
        'act1', 'act2', 'act3', 'act4',
        'drop1', 'drop2', 'drop3', 'drop4'
    ]

    # Train predictors for both datasets
    for dataset in ['mnist', 'cifar10']:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")

        # Load data
        print(f"\nLoading {dataset.upper()} architectures...")
        architectures = load_all_architectures(dataset)
        print(f"Loaded {len(architectures)} unique architectures")

        # Prepare data
        X = np.array([enc for enc, _, _ in architectures])
        y = np.array([acc for _, acc, _ in architectures])

        print(f"\nArchitecture Statistics:")
        print(f"  Mean accuracy: {y.mean():.4f}")
        print(f"  Std accuracy:  {y.std():.4f}")
        print(f"  Min accuracy:  {y.min():.4f}")
        print(f"  Max accuracy:  {y.max():.4f}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

        # Train models
        results = train_predictors(X_train, y_train, X_test, y_test)

        # Plot predictions
        plot_predictions(results, y_test, dataset)

        # Feature importance (for Random Forest)
        if 'Random Forest' in results:
            analyze_feature_importance(
                results['Random Forest']['model'],
                feature_names,
                dataset
            )

        # Save results
        output_file = f'results/predictor_{dataset}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'dataset': dataset,
                'n_architectures': len(architectures),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'models': {
                    name: {
                        'train_r2': res['train_r2'],
                        'test_r2': res['test_r2'],
                        'test_mae': res['test_mae'],
                        'test_spearman': res['test_spearman'],
                        'test_pearson': res['test_pearson']
                    }
                    for name, res in results.items()
                }
            }, f, indent=2)
        print(f"\n✓ Saved results to {output_file}")

    # Compare with zero-cost proxies
    compare_with_proxies()

    print(f"\n{'='*60}")
    print("✓ Architecture Predictor Analysis Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
