"""
Ensemble Predictor

Combines multiple predictor models (Random Forest, Gradient Boosting, Neural Net)
to improve prediction accuracy through ensemble averaging.
"""

import json
import numpy as np
import hashlib
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from nas.search_space import SearchSpace


class EnsemblePredictor:
    """Ensemble of multiple predictor models."""

    def __init__(self, models=None):
        """Initialize ensemble.

        Args:
            models: List of (name, model) tuples. If None, use defaults.
        """
        if models is None:
            self.models = [
                ('Ridge', Ridge(alpha=1.0)),
                ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
                ('Neural Net', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
            ]
        else:
            self.models = models

        self.weights = None  # Will be learned

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models and learn ensemble weights.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data for weight learning (optional)
        """
        print("Training ensemble models...")

        # Train each model
        for name, model in self.models:
            print(f"  Training {name}...")
            model.fit(X_train, y_train)

        # Learn weights
        if X_val is not None and y_val is not None:
            print("  Learning ensemble weights...")
            self._learn_weights(X_val, y_val)
        else:
            # Equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
            print(f"  Using equal weights: {self.weights}")

    def _learn_weights(self, X_val, y_val):
        """Learn optimal ensemble weights based on validation performance."""
        # Get predictions from each model
        predictions = np.array([model.predict(X_val) for _, model in self.models])

        # Optimize weights to minimize MAE on validation set
        from scipy.optimize import minimize

        def objective(weights):
            # Weighted average
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            return mean_absolute_error(y_val, ensemble_pred)

        # Constraints: weights sum to 1, all non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]

        # Initialize with equal weights
        w0 = np.ones(len(self.models)) / len(self.models)

        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        self.weights = result.x
        print(f"  Learned weights: {self.weights}")

    def predict(self, X):
        """Predict using weighted ensemble.

        Args:
            X: Features

        Returns:
            predictions: Weighted average of all models
        """
        predictions = np.array([model.predict(X) for _, model in self.models])
        return np.average(predictions, axis=0, weights=self.weights)

    def predict_individual(self, X):
        """Get predictions from each model.

        Args:
            X: Features

        Returns:
            predictions: Dict mapping model name to predictions
        """
        return {name: model.predict(X) for name, model in self.models}


def load_data(dataset='cifar10'):
    """Load training data for predictor.

    Args:
        dataset: Dataset name

    Returns:
        X, y: Features and targets
        search_space: SearchSpace object
    """
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
    all_results = []
    for file in result_files:
        if Path(file).exists():
            with open(file, 'r') as f:
                data = json.load(f)
                all_results.extend(data['results'])

    # Deduplicate
    seen = set()
    unique_results = []
    for result in all_results:
        config_str = json.dumps(result['config'], sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        if config_hash not in seen:
            seen.add(config_hash)
            unique_results.append(result)

    # Encode
    search_space = SearchSpace(input_size=input_size, output_size=10)
    X = np.array([search_space.encode(r['config']) for r in unique_results])
    y = np.array([r['val_acc'] for r in unique_results])

    return X, y, search_space


def evaluate_ensemble(dataset='cifar10'):
    """Train and evaluate ensemble predictor.

    Args:
        dataset: Dataset to train on

    Returns:
        ensemble: Trained ensemble
        metrics: Performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Ensemble Predictor on {dataset.upper()}")
    print(f"{'='*60}")

    # Load data
    X, y, search_space = load_data(dataset)
    print(f"\nLoaded {len(X)} unique architectures")

    # Train/val split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    # Train ensemble
    ensemble = EnsemblePredictor()
    ensemble.fit(X_train, y_train, X_val, y_val)

    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    # Ensemble predictions
    y_pred_ensemble = ensemble.predict(X_val)

    # Individual model predictions
    individual_preds = ensemble.predict_individual(X_val)

    # Compute metrics for each
    print(f"\nIndividual Models:")
    individual_metrics = {}
    for name, y_pred in individual_preds.items():
        spearman_rho, _ = spearmanr(y_val, y_pred)
        pearson_r, _ = pearsonr(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)

        print(f"\n  {name}:")
        print(f"    Spearman ρ: {spearman_rho:.4f}")
        print(f"    Pearson r:  {pearson_r:.4f}")
        print(f"    R²:         {r2:.4f}")
        print(f"    MAE:        {mae:.4f}")

        individual_metrics[name] = {
            'spearman_rho': spearman_rho,
            'pearson_r': pearson_r,
            'r2': r2,
            'mae': mae
        }

    # Ensemble metrics
    spearman_rho, _ = spearmanr(y_val, y_pred_ensemble)
    pearson_r, _ = pearsonr(y_val, y_pred_ensemble)
    r2 = r2_score(y_val, y_pred_ensemble)
    mae = mean_absolute_error(y_val, y_pred_ensemble)

    print(f"\n  ENSEMBLE (weighted):")
    print(f"    Spearman ρ: {spearman_rho:.4f}")
    print(f"    Pearson r:  {pearson_r:.4f}")
    print(f"    R²:         {r2:.4f}")
    print(f"    MAE:        {mae:.4f}")
    print(f"    Weights:    {ensemble.weights}")

    ensemble_metrics = {
        'spearman_rho': spearman_rho,
        'pearson_r': pearson_r,
        'r2': r2,
        'mae': mae,
        'weights': ensemble.weights.tolist()
    }

    # Check if ensemble improved over best individual
    best_individual_rho = max(m['spearman_rho'] for m in individual_metrics.values())
    improvement = spearman_rho - best_individual_rho

    print(f"\n{'='*60}")
    if improvement > 0:
        print(f"✓ Ensemble IMPROVED over best individual by {improvement:.4f}")
    else:
        print(f"✗ Ensemble did not improve (difference: {improvement:.4f})")
    print(f"{'='*60}")

    return ensemble, {
        'individual': individual_metrics,
        'ensemble': ensemble_metrics,
        'improvement': improvement
    }


def visualize_ensemble(dataset='cifar10'):
    """Visualize ensemble predictions."""
    # Load data
    X, y, _ = load_data(dataset)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    ensemble = EnsemblePredictor()
    ensemble.fit(X_train, y_train, X_val, y_val)

    # Predictions
    y_pred_ensemble = ensemble.predict(X_val)
    individual_preds = ensemble.predict_individual(X_val)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Individual models
    for i, (name, y_pred) in enumerate(individual_preds.items()):
        ax = axes[i]
        rho, _ = spearmanr(y_val, y_pred)
        ax.scatter(y_pred, y_val, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

        # Perfect line
        min_val = min(y_pred.min(), y_val.min())
        max_val = max(y_pred.max(), y_val.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'{name}\nρ = {rho:.4f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Ensemble
    ax = axes[4]
    rho, _ = spearmanr(y_val, y_pred_ensemble)
    ax.scatter(y_pred_ensemble, y_val, alpha=0.6, s=50,
              edgecolors='k', linewidths=0.5, color='green')

    min_val = min(y_pred_ensemble.min(), y_val.min())
    max_val = max(y_pred_ensemble.max(), y_val.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f'ENSEMBLE\nρ = {rho:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Weight visualization
    ax = axes[5]
    model_names = [name for name, _ in ensemble.models]
    ax.bar(range(len(ensemble.weights)), ensemble.weights,
           color='steelblue', edgecolor='k', linewidth=1.5)
    ax.set_xticks(range(len(ensemble.weights)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Weight', fontsize=11)
    ax.set_title('Ensemble Weights', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    output_file = f'{output_dir}/ensemble_predictor_{dataset}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_file}")
    plt.close()


def main():
    print("="*60)
    print("Ensemble Predictor")
    print("="*60)

    # Evaluate on CIFAR-10
    ensemble, metrics = evaluate_ensemble('cifar10')

    # Visualize
    visualize_ensemble('cifar10')

    # Save
    output_file = 'results/ensemble_predictor.json'
    with open(output_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        metrics_serializable = {
            'individual': {
                name: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in model_metrics.items()}
                for name, model_metrics in metrics['individual'].items()
            },
            'ensemble': {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics['ensemble'].items()
            },
            'improvement': float(metrics['improvement'])
        }

        json.dump({
            'dataset': 'cifar10',
            'metrics': metrics_serializable
        }, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    print(f"\n{'='*60}")
    print("✓ Ensemble Predictor Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
