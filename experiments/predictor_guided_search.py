"""
Predictor-Guided Architecture Search

Uses the trained predictor to pre-filter candidates before expensive evaluation.
Compares with random search to measure acceleration.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def train_predictor(dataset='cifar10'):
    """Train predictor on existing data.

    Args:
        dataset: 'mnist' or 'cifar10'

    Returns:
        predictor: Trained model
        search_space: SearchSpace object
    """
    print(f"Training predictor on {dataset.upper()} data...")

    # Load architectures (reuse code from architecture_predictor.py)
    if dataset == 'mnist':
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

    # Encode
    search_space = SearchSpace(input_size=input_size, output_size=10)
    X = np.array([search_space.encode(r['config']) for r in unique_results])
    y = np.array([r['val_acc'] for r in unique_results])

    # Train predictor
    predictor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    predictor.fit(X, y)

    print(f"  Trained on {len(X)} architectures")

    return predictor, search_space


def predictor_guided_search(dataset='cifar10', n_candidates=10000, top_k=10):
    """Search using predictor to pre-filter.

    Args:
        dataset: Dataset to search on
        n_candidates: Number of random architectures to generate
        top_k: Number of top predicted architectures to actually evaluate

    Returns:
        results: List of evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Predictor-Guided Search on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Candidates: {n_candidates:,}")
    print(f"  Top-K to evaluate: {top_k}")

    # Train predictor
    predictor, search_space = train_predictor(dataset)

    # Generate candidates
    print(f"\nGenerating {n_candidates:,} random architectures...")
    candidates = []
    for _ in tqdm(range(n_candidates), desc="Generating"):
        config = search_space.sample_random()
        encoding = search_space.encode(config)
        candidates.append((encoding, config))

    # Predict scores
    print(f"\nPredicting performance for all candidates...")
    X_candidates = np.array([enc for enc, _ in candidates])
    predictions = predictor.predict(X_candidates)

    # Sort by predicted performance
    sorted_indices = np.argsort(predictions)[::-1]
    top_indices = sorted_indices[:top_k]

    print(f"\nTop {top_k} predicted architectures:")
    for i, idx in enumerate(top_indices[:5]):  # Show top 5
        print(f"  {i+1}. Predicted: {predictions[idx]:.4f}")

    # Evaluate top-K
    print(f"\nEvaluating top {top_k} architectures...")
    evaluator = Evaluator(
        dataset=dataset,
        train_budget=20 if dataset == 'cifar10' else 10,
        cache_dir='results/cache'
    )

    results = []
    for i, idx in enumerate(tqdm(top_indices, desc="Evaluating")):
        encoding, config = candidates[idx]
        model = search_space.build_model(config)
        result = evaluator.evaluate(model, config)
        result['predicted_acc'] = predictions[idx]
        result['rank'] = i + 1
        results.append(result)

    # Sort by actual performance
    results.sort(key=lambda x: x['val_acc'], reverse=True)

    return results, predictions, top_indices


def compare_with_baseline(predictor_results, dataset='cifar10'):
    """Compare predictor-guided search with baseline random search.

    Args:
        predictor_results: Results from predictor-guided search
        dataset: Dataset name
    """
    print(f"\n{'='*60}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*60}")

    # Load baseline random search results
    baseline_file = f'results/{dataset}_random.json'
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)

    baseline_results = baseline_data['results']

    # Get top-K from baseline
    baseline_results_sorted = sorted(baseline_results, key=lambda x: x['val_acc'], reverse=True)
    top_k = len(predictor_results)
    baseline_topk = baseline_results_sorted[:top_k]

    # Best architectures
    predictor_best = max(predictor_results, key=lambda x: x['val_acc'])['val_acc']
    baseline_best = baseline_topk[0]['val_acc']

    # Mean of top-K
    predictor_mean = np.mean([r['val_acc'] for r in predictor_results])
    baseline_mean = np.mean([r['val_acc'] for r in baseline_topk])

    print(f"\nPredictor-Guided (Top-{top_k} from 10K candidates):")
    print(f"  Best:  {predictor_best:.4f}")
    print(f"  Mean:  {predictor_mean:.4f}")

    print(f"\nBaseline Random (Top-{top_k} from {len(baseline_results)} random):")
    print(f"  Best:  {baseline_best:.4f}")
    print(f"  Mean:  {baseline_mean:.4f}")

    print(f"\nImprovement:")
    print(f"  Best:  {(predictor_best - baseline_best)*100:+.2f}%")
    print(f"  Mean:  {(predictor_mean - baseline_mean)*100:+.2f}%")

    # Prediction accuracy
    predictions_vs_actual = [
        (r['predicted_acc'], r['val_acc'])
        for r in predictor_results
    ]

    pred_vals = np.array([p for p, _ in predictions_vs_actual])
    actual_vals = np.array([a for _, a in predictions_vs_actual])

    from scipy.stats import spearmanr
    correlation, _ = spearmanr(pred_vals, actual_vals)

    print(f"\nPrediction Quality:")
    print(f"  Spearman ρ: {correlation:.4f}")
    print(f"  (How well predictions ranked architectures)")


def main():
    print("="*60)
    print("Predictor-Guided Architecture Search")
    print("="*60)

    # Run on CIFAR-10
    results, predictions, top_indices = predictor_guided_search(
        dataset='cifar10',
        n_candidates=10000,
        top_k=10
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nTop 5 Architectures (by actual performance):")
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. Accuracy: {result['val_acc']:.4f} (predicted: {result['predicted_acc']:.4f})")
        print(f"   Architecture: {result['config']['n_layers']+1}L {result['config']['hidden_sizes'][:result['config']['n_layers']+1]}")
        print(f"   Activations: {result['config']['activations'][:result['config']['n_layers']+1]}")

    # Compare with baseline
    compare_with_baseline(results, dataset='cifar10')

    # Save results
    output_file = 'results/predictor_guided_search.json'
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': 'cifar10',
            'n_candidates': 10000,
            'top_k_evaluated': len(results),
            'results': results,
            'summary': {
                'best_acc': max(r['val_acc'] for r in results),
                'mean_acc': np.mean([r['val_acc'] for r in results]),
                'prediction_quality': float(np.corrcoef(
                    [r['predicted_acc'] for r in results],
                    [r['val_acc'] for r in results]
                )[0, 1])
            }
        }, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    print(f"\n{'='*60}")
    print("✓ Predictor-Guided Search Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
