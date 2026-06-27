"""
Early Stopping Predictor

Predicts final accuracy from partial training curves (first 2-3 epochs).
Can 5-10x NAS speed by stopping unpromising architectures early.
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator


class LearningCurvePredictor(nn.Module):
    """LSTM-based predictor for learning curves."""

    def __init__(self, input_size=2, hidden_size=32, num_layers=2):
        """
        Args:
            input_size: Number of features per timestep (train_acc, val_acc)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)

        Returns:
            predictions: (batch_size, 1)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_size)

        # Predict final accuracy
        output = self.fc(last_hidden)

        return output


def collect_learning_curves(dataset='cifar10', n_samples=100):
    """Collect learning curves from random architectures.

    Args:
        dataset: Dataset to train on
        n_samples: Number of architectures to evaluate

    Returns:
        data: List of {config, learning_curve, final_acc}
    """
    print(f"\n{'='*60}")
    print(f"Collecting Learning Curves on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Samples: {n_samples}")

    input_size = 3072 if dataset == 'cifar10' else 784
    search_space = SearchSpace(input_size=input_size, output_size=10)

    evaluator = Evaluator(
        dataset=dataset,
        train_budget=20 if dataset == 'cifar10' else 10,
        cache_dir='results/cache_curves',  # Separate cache for curves
        save_curves=True  # Enable curve tracking
    )

    data = []
    for i in tqdm(range(n_samples), desc="Evaluating"):
        config = search_space.sample_random()
        model = search_space.build_model(config)
        result = evaluator.evaluate(model, config)

        data.append({
            'config': result['config'],
            'learning_curve': result['learning_curve'],
            'final_acc': result['val_acc'],
            'n_params': result['n_params']
        })

    print(f"\n  Collected {len(data)} learning curves")
    return data


def prepare_curve_data(data, early_stop_epoch=2):
    """Prepare data for learning curve prediction.

    Args:
        data: List of architecture results with learning curves
        early_stop_epoch: Use curves up to this epoch as input

    Returns:
        X: Input sequences (n_samples, seq_len, 2)
        y: Target final accuracies (n_samples,)
    """
    X = []
    y = []

    for sample in data:
        curve = sample['learning_curve']

        # Skip if curve is too short
        if len(curve) < early_stop_epoch + 1:
            continue

        # Extract features (train_acc, val_acc) for first N epochs
        features = []
        for epoch_data in curve[:early_stop_epoch + 1]:
            features.append([
                epoch_data['train_acc'],
                epoch_data['val_acc']
            ])

        X.append(features)
        y.append(sample['final_acc'])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def train_curve_predictor(X_train, y_train, X_val, y_val, epochs=100):
    """Train LSTM curve predictor.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Training epochs

    Returns:
        model: Trained predictor
        history: Training history
    """
    print("\nTraining Learning Curve Predictor...")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    # Model
    model = LearningCurvePredictor(input_size=2, hidden_size=32, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = criterion(y_val_pred, y_val_t)

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={loss.item():.6f}, Val Loss={val_loss.item():.6f}")

    print(f"  Best Val Loss: {best_val_loss:.6f}")

    return model, history


def evaluate_predictor(model, X_test, y_test, early_stop_epoch=2):
    """Evaluate curve predictor.

    Args:
        model: Trained predictor
        X_test, y_test: Test data
        early_stop_epoch: Number of epochs used for prediction

    Returns:
        metrics: Evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Predictor (using first {early_stop_epoch+1} epochs)")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        y_pred_t = model(X_test_t)
        y_pred = y_pred_t.squeeze().numpy()

    # Metrics
    spearman_rho, _ = spearmanr(y_test, y_pred)
    pearson_r, _ = pearsonr(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n  Prediction Quality:")
    print(f"    Spearman ρ: {spearman_rho:.4f}")
    print(f"    Pearson r:  {pearson_r:.4f}")
    print(f"    MAE:        {mae:.4f}")
    print(f"    R²:         {r2:.4f}")

    return {
        'spearman_rho': spearman_rho,
        'pearson_r': pearson_r,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred,
        'actuals': y_test
    }


def test_early_stopping_strategy(model, data, early_stop_epoch=2, stop_fraction=0.5):
    """Test early stopping strategy.

    Args:
        model: Trained predictor
        data: Architecture data with learning curves
        early_stop_epoch: Epoch to make early stopping decision
        stop_fraction: Fraction of architectures to stop early

    Returns:
        analysis: Early stopping analysis
    """
    print(f"\n{'='*60}")
    print(f"Early Stopping Strategy Analysis")
    print(f"{'='*60}")
    print(f"  Decision point: Epoch {early_stop_epoch}")
    print(f"  Stop fraction: {stop_fraction:.0%}")

    # Prepare predictions
    X, y = prepare_curve_data(data, early_stop_epoch)

    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        y_pred_t = model(X_t)
        predictions = y_pred_t.squeeze().numpy()

    # Sort by predicted accuracy
    sorted_indices = np.argsort(predictions)

    # Bottom fraction would be stopped
    n_stop = int(len(sorted_indices) * stop_fraction)
    stopped_indices = sorted_indices[:n_stop]
    continued_indices = sorted_indices[n_stop:]

    # Analysis
    stopped_actual = y[stopped_indices]
    continued_actual = y[continued_indices]

    # What if we stopped bottom 50%?
    best_in_stopped = stopped_actual.max() if len(stopped_actual) > 0 else 0
    best_in_continued = continued_actual.max() if len(continued_actual) > 0 else 0

    # Compute savings
    # If we train full budget on continued, but stop early on stopped
    full_budget = 20  # CIFAR-10
    compute_saved = (n_stop * (full_budget - early_stop_epoch - 1)) / (len(data) * full_budget)

    print(f"\n  Results:")
    print(f"    Stopped:   {len(stopped_indices)} architectures (bottom {stop_fraction:.0%})")
    print(f"    Continued: {len(continued_indices)} architectures")
    print(f"\n  Best accuracy in stopped:   {best_in_stopped:.4f}")
    print(f"  Best accuracy in continued: {best_in_continued:.4f}")
    print(f"\n  Compute saved: {compute_saved:.1%}")

    # Did we miss the best architecture?
    overall_best = y.max()
    missed_best = best_in_stopped >= overall_best * 0.99  # Within 1% of best

    if missed_best:
        print(f"\n  ⚠ WARNING: Stopped an architecture within 1% of best!")
    else:
        print(f"\n  ✓ Did not miss best architecture")

    # Precision/Recall analysis
    # "Bad" = bottom 50% by actual accuracy
    actual_sorted = np.argsort(y)
    actually_bad = set(actual_sorted[:n_stop])
    predicted_bad = set(stopped_indices)

    true_positives = len(actually_bad & predicted_bad)
    false_positives = len(predicted_bad - actually_bad)
    false_negatives = len(actually_bad - predicted_bad)

    precision = true_positives / len(predicted_bad) if len(predicted_bad) > 0 else 0
    recall = true_positives / len(actually_bad) if len(actually_bad) > 0 else 0

    print(f"\n  Early Stopping Accuracy:")
    print(f"    Precision: {precision:.2%} (stopped were actually bad)")
    print(f"    Recall:    {recall:.2%} (caught bad architectures)")

    return {
        'stopped': len(stopped_indices),
        'continued': len(continued_indices),
        'best_in_stopped': best_in_stopped,
        'best_in_continued': best_in_continued,
        'compute_saved': compute_saved,
        'missed_best': missed_best,
        'precision': precision,
        'recall': recall
    }


def visualize_results(model, X_test, y_test, early_stop_epoch, dataset='cifar10'):
    """Visualize predictor results."""
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)

    # Get predictions
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        y_pred_t = model(X_test_t)
        y_pred = y_pred_t.squeeze().numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Predicted vs Actual
    ax = axes[0]
    spearman_rho, _ = spearmanr(y_test, y_pred)

    ax.scatter(y_pred, y_test, alpha=0.6, s=80, edgecolors='k', linewidths=0.5)

    # Perfect line
    min_val = min(y_pred.min(), y_test.min())
    max_val = max(y_pred.max(), y_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    # Best fit
    z = np.polyfit(y_pred, y_test, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_pred.min(), y_pred.max(), 100)
    ax.plot(x_line, p(x_line), 'b-', alpha=0.8, linewidth=2, label='Best fit')

    ax.set_xlabel(f'Predicted from {early_stop_epoch+1} Epochs', fontsize=12)
    ax.set_ylabel('Actual Final Accuracy', fontsize=12)
    ax.set_title(f'Early Stopping Predictor: {dataset.upper()}\nSpearman ρ = {spearman_rho:.4f}',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Error distribution
    ax = axes[1]
    errors = y_test - y_pred
    ax.hist(errors, bins=30, edgecolor='k', alpha=0.7, color='steelblue')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Error Distribution\nMAE = {np.abs(errors).mean():.4f}',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_file = f'{output_dir}/early_stopping_predictor_{dataset}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_file}")
    plt.close()


def main():
    print("="*60)
    print("Early Stopping Predictor")
    print("="*60)

    dataset = 'cifar10'
    n_samples = 100
    early_stop_epoch = 2  # Use epochs 0, 1, 2 to predict final

    # Collect learning curves
    data = collect_learning_curves(dataset=dataset, n_samples=n_samples)

    # Prepare data
    X, y = prepare_curve_data(data, early_stop_epoch)
    print(f"\n  Prepared data: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train predictor
    model, history = train_curve_predictor(X_train, y_train, X_val, y_val, epochs=100)

    # Evaluate
    metrics = evaluate_predictor(model, X_test, y_test, early_stop_epoch)

    # Test early stopping strategy
    strategy_analysis = test_early_stopping_strategy(
        model, data, early_stop_epoch, stop_fraction=0.5
    )

    # Visualize
    visualize_results(model, X_test, y_test, early_stop_epoch, dataset)

    # Save results
    output_file = 'results/early_stopping_predictor.json'
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': dataset,
            'n_samples': n_samples,
            'early_stop_epoch': early_stop_epoch,
            'metrics': {
                'spearman_rho': float(metrics['spearman_rho']),
                'pearson_r': float(metrics['pearson_r']),
                'mae': float(metrics['mae']),
                'r2': float(metrics['r2'])
            },
            'early_stopping_analysis': {
                'stopped': int(strategy_analysis['stopped']),
                'continued': int(strategy_analysis['continued']),
                'best_in_stopped': float(strategy_analysis['best_in_stopped']),
                'best_in_continued': float(strategy_analysis['best_in_continued']),
                'compute_saved': float(strategy_analysis['compute_saved']),
                'missed_best': bool(strategy_analysis['missed_best']),
                'precision': float(strategy_analysis['precision']),
                'recall': float(strategy_analysis['recall'])
            }
        }, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Prediction Quality (from {early_stop_epoch+1} epochs):")
    print(f"    Spearman ρ: {metrics['spearman_rho']:.4f}")
    print(f"    MAE:        {metrics['mae']:.4f}")
    print(f"\n  Early Stopping Efficiency:")
    print(f"    Compute saved: {strategy_analysis['compute_saved']:.1%}")
    print(f"    Precision:     {strategy_analysis['precision']:.1%}")
    print(f"    Recall:        {strategy_analysis['recall']:.1%}")
    print(f"    Missed best:   {'Yes ⚠' if strategy_analysis['missed_best'] else 'No ✓'}")

    print(f"\n{'='*60}")
    print("✓ Early Stopping Predictor Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
