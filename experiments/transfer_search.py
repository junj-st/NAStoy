"""
Transfer Predictor-Guided Search

Tests if a predictor trained on one dataset can accelerate search on another.
Uses MNIST predictor to guide Fashion-MNIST search.
"""

import json
import numpy as np
import hashlib
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator
from sklearn.ensemble import GradientBoostingRegressor


def train_source_predictor(source_dataset='mnist'):
    """Train predictor on source dataset.

    Args:
        source_dataset: Dataset to train on

    Returns:
        predictor, search_space
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
        raise ValueError(f"Unsupported source dataset: {source_dataset}")

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

    # Encode and train
    search_space = SearchSpace(input_size=input_size, output_size=10)
    X = np.array([search_space.encode(r['config']) for r in unique_results])
    y = np.array([r['val_acc'] for r in unique_results])

    predictor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    predictor.fit(X, y)

    print(f"  Trained on {len(X)} architectures")
    print(f"  Source accuracy: {y.mean():.4f} ± {y.std():.4f}")

    return predictor, search_space


def transfer_predictor_search(
    predictor,
    source_space,
    target_dataset='fashionmnist',
    n_candidates=5000,
    top_k=20
):
    """Use source predictor to guide target dataset search.

    Args:
        predictor: Trained predictor from source dataset
        source_space: SearchSpace used for encoding
        target_dataset: Target dataset to search
        n_candidates: Number of candidates to generate
        top_k: Number to actually evaluate

    Returns:
        results: Evaluated top-K architectures
        all_predictions: Predictions for all candidates
    """
    print(f"\n{'='*60}")
    print(f"Transfer Search: MNIST Predictor → {target_dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Candidates: {n_candidates:,}")
    print(f"  Top-K to evaluate: {top_k}")

    # Generate candidates
    print(f"\nGenerating {n_candidates:,} random architectures...")
    candidates = []
    for _ in tqdm(range(n_candidates), desc="Generating"):
        config = source_space.sample_random()
        encoding = source_space.encode(config)
        candidates.append((encoding, config))

    # Predict scores
    print(f"\nPredicting performance with MNIST predictor...")
    X_candidates = np.array([enc for enc, _ in candidates])
    predictions = predictor.predict(X_candidates)

    # Sort by predicted performance
    sorted_indices = np.argsort(predictions)[::-1]
    top_indices = sorted_indices[:top_k]

    print(f"\nTop {top_k} predicted architectures:")
    for i, idx in enumerate(top_indices[:5]):
        print(f"  {i+1}. Predicted: {predictions[idx]:.4f}")

    # Evaluate top-K on target dataset
    print(f"\nEvaluating top {top_k} on {target_dataset.upper()}...")
    evaluator = Evaluator(
        dataset=target_dataset,
        train_budget=10,
        cache_dir='results/cache'
    )

    results = []
    for i, idx in enumerate(tqdm(top_indices, desc="Evaluating")):
        encoding, config = candidates[idx]
        model = source_space.build_model(config)
        result = evaluator.evaluate(model, config)
        result['predicted_acc'] = predictions[idx]
        result['rank'] = i + 1
        results.append(result)

    # Sort by actual performance
    results.sort(key=lambda x: x['val_acc'], reverse=True)

    return results, predictions, top_indices


def compare_with_baseline(transfer_results, target_dataset='fashionmnist'):
    """Compare transfer search with random baseline.

    Args:
        transfer_results: Results from transfer search
        target_dataset: Target dataset name
    """
    print(f"\n{'='*60}")
    print("COMPARISON WITH RANDOM BASELINE")
    print(f"{'='*60}")

    # Run random baseline for comparison
    print(f"\nRunning random baseline on {target_dataset.upper()}...")
    search_space = SearchSpace(input_size=784, output_size=10)
    evaluator = Evaluator(
        dataset=target_dataset,
        train_budget=10,
        cache_dir='results/cache'
    )

    top_k = len(transfer_results)
    baseline_results = []
    for _ in tqdm(range(top_k), desc="Random baseline"):
        config = search_space.sample_random()
        model = search_space.build_model(config)
        result = evaluator.evaluate(model, config)
        baseline_results.append(result)

    # Compare
    transfer_accs = [r['val_acc'] for r in transfer_results]
    baseline_accs = [r['val_acc'] for r in baseline_results]

    transfer_best = max(transfer_accs)
    transfer_mean = np.mean(transfer_accs)

    baseline_best = max(baseline_accs)
    baseline_mean = np.mean(baseline_accs)

    print(f"\nTransfer Search (MNIST predictor, top-{top_k} from 5K):")
    print(f"  Best:  {transfer_best:.4f}")
    print(f"  Mean:  {transfer_mean:.4f}")
    print(f"  Std:   {np.std(transfer_accs):.4f}")

    print(f"\nRandom Baseline ({top_k} random):")
    print(f"  Best:  {baseline_best:.4f}")
    print(f"  Mean:  {baseline_mean:.4f}")
    print(f"  Std:   {np.std(baseline_accs):.4f}")

    print(f"\nImprovement:")
    print(f"  Best:  {(transfer_best - baseline_best)*100:+.2f}%")
    print(f"  Mean:  {(transfer_mean - baseline_mean)*100:+.2f}%")

    # Prediction quality on evaluated set
    pred_vals = np.array([r['predicted_acc'] for r in transfer_results])
    actual_vals = np.array([r['val_acc'] for r in transfer_results])

    correlation, p_value = spearmanr(pred_vals, actual_vals)
    print(f"\nPrediction Quality on Evaluated Set:")
    print(f"  Spearman ρ: {correlation:.4f} (p={p_value:.4e})")
    print(f"  (How well predictor ranked the top-K candidates)")

    return baseline_results


def main():
    print("="*60)
    print("Transfer Predictor-Guided Search")
    print("="*60)

    # Train MNIST predictor
    predictor, search_space = train_source_predictor('mnist')

    # Use it to guide Fashion-MNIST search
    results, predictions, top_indices = transfer_predictor_search(
        predictor,
        search_space,
        target_dataset='fashionmnist',
        n_candidates=5000,
        top_k=20
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nTop 5 Architectures on Fashion-MNIST:")
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. Accuracy: {result['val_acc']:.4f} (predicted: {result['predicted_acc']:.4f})")
        print(f"   Architecture: {result['config']['n_layers']+1}L {result['config']['hidden_sizes'][:result['config']['n_layers']+1]}")
        print(f"   Activations: {result['config']['activations'][:result['config']['n_layers']+1]}")

    # Compare with baseline
    baseline_results = compare_with_baseline(results, target_dataset='fashionmnist')

    # Save results
    output_file = 'results/transfer_search.json'
    with open(output_file, 'w') as f:
        json.dump({
            'source_dataset': 'mnist',
            'target_dataset': 'fashionmnist',
            'n_candidates': 5000,
            'top_k_evaluated': len(results),
            'transfer_results': results,
            'baseline_results': baseline_results,
            'summary': {
                'transfer_best': max(r['val_acc'] for r in results),
                'transfer_mean': np.mean([r['val_acc'] for r in results]),
                'baseline_best': max(r['val_acc'] for r in baseline_results),
                'baseline_mean': np.mean([r['val_acc'] for r in baseline_results]),
                'improvement_best': (max(r['val_acc'] for r in results) -
                                   max(r['val_acc'] for r in baseline_results)) * 100,
                'improvement_mean': (np.mean([r['val_acc'] for r in results]) -
                                   np.mean([r['val_acc'] for r in baseline_results])) * 100
            }
        }, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    print(f"\n{'='*60}")
    print("✓ Transfer Search Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
