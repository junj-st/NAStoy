"""
Multi-Fidelity Optimization for NAS

Uses cheap approximations (fewer epochs, less data) to filter candidates
before expensive full evaluation. Implements Successive Halving (ASHA).
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import time

import torch
from torch.utils.data import DataLoader, Subset

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator


class MultiFidelityEvaluator:
    """Evaluator with configurable fidelity levels."""

    def __init__(self, dataset='cifar10', device=None):
        """
        Args:
            dataset: Dataset name
            device: Device to train on
        """
        self.dataset = dataset
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Fidelity definitions
        self.fidelities = {
            'low': {'epochs': 2, 'data_fraction': 0.25},
            'medium': {'epochs': 5, 'data_fraction': 0.50},
            'high': {'epochs': 10, 'data_fraction': 1.0},
            'full': {'epochs': 20, 'data_fraction': 1.0}
        }

    def evaluate_at_fidelity(self, model, config, fidelity='full'):
        """Evaluate architecture at specified fidelity.

        Args:
            model: PyTorch model
            config: Architecture config
            fidelity: 'low', 'medium', 'high', or 'full'

        Returns:
            result: Evaluation result with val_acc and cost
        """
        fid_config = self.fidelities[fidelity]
        epochs = fid_config['epochs']
        data_fraction = fid_config['data_fraction']

        # Create evaluator with fidelity settings
        evaluator = Evaluator(
            dataset=self.dataset,
            train_budget=epochs,
            cache_dir=None,  # No caching for multi-fidelity
            device=self.device
        )

        # Subsample data if needed
        if data_fraction < 1.0:
            # Monkey-patch data loaders to use subset
            original_train_loader = evaluator.train_loader
            original_val_loader = evaluator.val_loader

            # Create subset indices
            n_train = len(original_train_loader.dataset)
            n_val = len(original_val_loader.dataset)
            train_indices = np.random.choice(n_train, int(n_train * data_fraction), replace=False)
            val_indices = np.random.choice(n_val, int(n_val * data_fraction), replace=False)

            # Create subset datasets
            train_subset = Subset(original_train_loader.dataset, train_indices)
            val_subset = Subset(original_val_loader.dataset, val_indices)

            evaluator.train_loader = DataLoader(
                train_subset,
                batch_size=evaluator.batch_size,
                shuffle=True,
                num_workers=0
            )
            evaluator.val_loader = DataLoader(
                val_subset,
                batch_size=evaluator.batch_size,
                shuffle=False,
                num_workers=0
            )

        # Evaluate
        start_time = time.time()
        result = evaluator.evaluate(model, config)
        eval_time = time.time() - start_time

        result['fidelity'] = fidelity
        result['epochs'] = epochs
        result['data_fraction'] = data_fraction
        result['eval_time'] = eval_time

        return result


def test_fidelity_correlation(n_samples=50, dataset='cifar10'):
    """Test correlation between different fidelities and full accuracy.

    Args:
        n_samples: Number of architectures to test
        dataset: Dataset name

    Returns:
        data: List of results at all fidelities
        correlations: Correlation metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing Fidelity Correlation on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Samples: {n_samples}")

    input_size = 3072 if dataset == 'cifar10' else 784
    search_space = SearchSpace(input_size=input_size, output_size=10)
    evaluator = MultiFidelityEvaluator(dataset=dataset)

    data = []

    for i in tqdm(range(n_samples), desc="Evaluating"):
        config = search_space.sample_random()

        results = {}
        for fidelity in ['low', 'medium', 'high', 'full']:
            model = search_space.build_model(config)
            result = evaluator.evaluate_at_fidelity(model, config, fidelity)
            results[fidelity] = {
                'val_acc': result['val_acc'],
                'eval_time': result['eval_time']
            }

        data.append({
            'config': config,
            'results': results
        })

    # Compute correlations
    print(f"\n{'='*60}")
    print("FIDELITY CORRELATIONS")
    print(f"{'='*60}")

    correlations = {}
    full_accs = [d['results']['full']['val_acc'] for d in data]

    for fidelity in ['low', 'medium', 'high']:
        fid_accs = [d['results'][fidelity]['val_acc'] for d in data]

        spearman_rho, sp_p = spearmanr(fid_accs, full_accs)
        pearson_r, pr_p = pearsonr(fid_accs, full_accs)

        avg_time = np.mean([d['results'][fidelity]['eval_time'] for d in data])
        full_time = np.mean([d['results']['full']['eval_time'] for d in data])
        speedup = full_time / avg_time

        correlations[fidelity] = {
            'spearman_rho': spearman_rho,
            'pearson_r': pearson_r,
            'avg_time': avg_time,
            'speedup': speedup
        }

        print(f"\n  {fidelity.upper()}:")
        print(f"    Spearman ρ: {spearman_rho:.4f}")
        print(f"    Pearson r:  {pearson_r:.4f}")
        print(f"    Avg time:   {avg_time:.1f}s")
        print(f"    Speedup:    {speedup:.1f}x")

    return data, correlations


def successive_halving(
    n_candidates=81,
    dataset='cifar10',
    reduction_factor=3,
    min_fidelity='low',
    max_fidelity='full'
):
    """Successive Halving Algorithm (SHA/ASHA).

    Args:
        n_candidates: Number of initial candidates
        dataset: Dataset name
        reduction_factor: Fraction to keep at each rung
        min_fidelity: Starting fidelity
        max_fidelity: Final fidelity

    Returns:
        results: Final evaluation results
        history: Full search history
    """
    print(f"\n{'='*60}")
    print(f"Successive Halving on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Initial candidates: {n_candidates}")
    print(f"  Reduction factor: {reduction_factor}")

    input_size = 3072 if dataset == 'cifar10' else 784
    search_space = SearchSpace(input_size=input_size, output_size=10)
    evaluator = MultiFidelityEvaluator(dataset=dataset)

    # Define rungs (fidelity levels)
    fidelity_sequence = ['low', 'medium', 'high', 'full']
    start_idx = fidelity_sequence.index(min_fidelity)
    end_idx = fidelity_sequence.index(max_fidelity)
    rungs = fidelity_sequence[start_idx:end_idx+1]

    print(f"  Fidelity rungs: {' → '.join(rungs)}")

    # Generate candidates
    print(f"\n  Generating {n_candidates} candidates...")
    candidates = []
    for _ in range(n_candidates):
        config = search_space.sample_random()
        candidates.append({
            'config': config,
            'results': {}
        })

    history = {
        'rungs': [],
        'total_cost': 0
    }

    active_candidates = candidates
    total_cost = 0

    # Run successive halving
    for rung_idx, fidelity in enumerate(rungs):
        n_active = len(active_candidates)
        print(f"\n  Rung {rung_idx+1}/{len(rungs)}: {fidelity.upper()} ({n_active} candidates)")

        # Evaluate at current fidelity
        for candidate in tqdm(active_candidates, desc=f"  Evaluating at {fidelity}"):
            model = search_space.build_model(candidate['config'])
            result = evaluator.evaluate_at_fidelity(model, candidate['config'], fidelity)

            candidate['results'][fidelity] = {
                'val_acc': result['val_acc'],
                'eval_time': result['eval_time']
            }
            total_cost += result['eval_time']

        # Sort by current fidelity performance
        active_candidates.sort(
            key=lambda x: x['results'][fidelity]['val_acc'],
            reverse=True
        )

        # Record rung
        history['rungs'].append({
            'fidelity': fidelity,
            'n_evaluated': n_active,
            'best_acc': active_candidates[0]['results'][fidelity]['val_acc'],
            'rung_cost': sum(c['results'][fidelity]['eval_time'] for c in active_candidates)
        })

        # Promote top candidates to next rung (except at final rung)
        if rung_idx < len(rungs) - 1:
            n_promote = max(1, n_active // reduction_factor)
            active_candidates = active_candidates[:n_promote]
            print(f"    → Promoting top {n_promote} to next rung")

    history['total_cost'] = total_cost

    print(f"\n  Total compute time: {total_cost:.1f}s ({total_cost/60:.1f} min)")

    # Final results
    final_results = active_candidates
    final_results.sort(key=lambda x: x['results']['full']['val_acc'], reverse=True)

    return final_results, history


def compare_with_baseline(sha_results, sha_history, dataset='cifar10'):
    """Compare SHA with random search baseline.

    Args:
        sha_results: SHA final results
        sha_history: SHA search history
        dataset: Dataset name

    Returns:
        comparison: Comparison metrics
    """
    print(f"\n{'='*60}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*60}")

    # SHA statistics
    sha_best = sha_results[0]['results']['full']['val_acc']
    sha_mean = np.mean([r['results']['full']['val_acc'] for r in sha_results])
    sha_cost = sha_history['total_cost']
    sha_n_full = len(sha_results)

    print(f"\n  Successive Halving:")
    print(f"    Best accuracy: {sha_best:.4f}")
    print(f"    Mean accuracy: {sha_mean:.4f}")
    print(f"    Total cost:    {sha_cost:.1f}s ({sha_cost/60:.1f} min)")
    print(f"    Full evals:    {sha_n_full}")

    # Baseline: How many full evaluations can we do with same budget?
    input_size = 3072 if dataset == 'cifar10' else 784
    search_space = SearchSpace(input_size=input_size, output_size=10)
    evaluator = MultiFidelityEvaluator(dataset=dataset)

    # Estimate cost of one full evaluation
    print(f"\n  Running baseline (random search at full fidelity)...")
    sample_config = search_space.sample_random()
    sample_model = search_space.build_model(sample_config)
    sample_result = evaluator.evaluate_at_fidelity(sample_model, sample_config, 'full')
    full_eval_time = sample_result['eval_time']

    n_baseline = int(sha_cost / full_eval_time)
    print(f"    Budget allows ~{n_baseline} full evaluations")

    # Run baseline
    baseline_results = []
    baseline_cost = 0

    for _ in tqdm(range(n_baseline), desc="  Random baseline"):
        config = search_space.sample_random()
        model = search_space.build_model(config)
        result = evaluator.evaluate_at_fidelity(model, config, 'full')

        baseline_results.append({
            'val_acc': result['val_acc'],
            'eval_time': result['eval_time']
        })
        baseline_cost += result['eval_time']

    baseline_best = max(r['val_acc'] for r in baseline_results)
    baseline_mean = np.mean([r['val_acc'] for r in baseline_results])

    print(f"\n  Random Baseline:")
    print(f"    Best accuracy: {baseline_best:.4f}")
    print(f"    Mean accuracy: {baseline_mean:.4f}")
    print(f"    Total cost:    {baseline_cost:.1f}s ({baseline_cost/60:.1f} min)")
    print(f"    Full evals:    {len(baseline_results)}")

    print(f"\n  Improvement:")
    print(f"    Best: {(sha_best - baseline_best)*100:+.2f}%")
    print(f"    Mean: {(sha_mean - baseline_mean)*100:+.2f}%")

    return {
        'sha': {
            'best': sha_best,
            'mean': sha_mean,
            'cost': sha_cost,
            'n_full': sha_n_full
        },
        'baseline': {
            'best': baseline_best,
            'mean': baseline_mean,
            'cost': baseline_cost,
            'n_full': len(baseline_results)
        },
        'improvement': {
            'best': (sha_best - baseline_best) * 100,
            'mean': (sha_mean - baseline_mean) * 100
        }
    }


def visualize_results(fidelity_data, sha_history, dataset='cifar10'):
    """Visualize multi-fidelity results."""
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fidelity correlations
    ax = axes[0]
    full_accs = [d['results']['full']['val_acc'] for d in fidelity_data]

    for fidelity, color, marker in [
        ('low', 'red', 'o'),
        ('medium', 'orange', 's'),
        ('high', 'green', '^')
    ]:
        fid_accs = [d['results'][fidelity]['val_acc'] for d in fidelity_data]
        rho, _ = spearmanr(fid_accs, full_accs)

        ax.scatter(fid_accs, full_accs, alpha=0.6, s=60,
                  label=f'{fidelity.capitalize()} (ρ={rho:.3f})',
                  color=color, marker=marker, edgecolors='k', linewidths=0.5)

    # Perfect line
    all_accs = full_accs + [d['results']['low']['val_acc'] for d in fidelity_data]
    min_val, max_val = min(all_accs), max(all_accs)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Fidelity Accuracy', fontsize=12)
    ax.set_ylabel('Full Accuracy', fontsize=12)
    ax.set_title(f'Fidelity Correlation: {dataset.upper()}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # SHA progression
    ax = axes[1]
    rungs = sha_history['rungs']
    fidelities = [r['fidelity'] for r in rungs]
    best_accs = [r['best_acc'] for r in rungs]
    n_evaluated = [r['n_evaluated'] for r in rungs]

    ax2 = ax.twinx()

    # Best accuracy progression
    ax.plot(range(len(rungs)), best_accs, 'b-o', linewidth=2, markersize=8,
           label='Best Accuracy', markeredgecolor='k', markeredgewidth=0.5)
    ax.set_xlabel('Rung (Fidelity Level)', fontsize=12)
    ax.set_ylabel('Best Accuracy', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([f.capitalize() for f in fidelities], rotation=45)

    # Number of candidates
    ax2.plot(range(len(rungs)), n_evaluated, 'r-s', linewidth=2, markersize=8,
            label='# Candidates', markeredgecolor='k', markeredgewidth=0.5)
    ax2.set_ylabel('# Candidates Evaluated', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax.set_title('Successive Halving Progression', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.tight_layout()

    output_file = f'{output_dir}/multi_fidelity_{dataset}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_file}")
    plt.close()


def main():
    print("="*60)
    print("Multi-Fidelity Optimization")
    print("="*60)

    dataset = 'cifar10'

    # Test fidelity correlations
    fidelity_data, correlations = test_fidelity_correlation(n_samples=50, dataset=dataset)

    # Run successive halving
    sha_results, sha_history = successive_halving(
        n_candidates=81,
        dataset=dataset,
        reduction_factor=3,
        min_fidelity='low',
        max_fidelity='full'
    )

    # Print top results
    print(f"\n{'='*60}")
    print("TOP 5 FINAL ARCHITECTURES")
    print(f"{'='*60}")

    for i, result in enumerate(sha_results[:5]):
        print(f"\n{i+1}. Accuracy: {result['results']['full']['val_acc']:.4f}")
        config = result['config']
        print(f"   Architecture: {config['n_layers']+1}L {config['hidden_sizes'][:config['n_layers']+1]}")
        print(f"   Activations: {config['activations'][:config['n_layers']+1]}")

    # Compare with baseline
    comparison = compare_with_baseline(sha_results, sha_history, dataset)

    # Visualize
    visualize_results(fidelity_data, sha_history, dataset)

    # Save results
    output_file = 'results/multi_fidelity.json'
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON
        correlations_json = {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in correlations.items()
        }

        json.dump({
            'dataset': dataset,
            'fidelity_correlations': correlations_json,
            'sha_results': [{
                'config': r['config'],
                'full_acc': float(r['results']['full']['val_acc'])
            } for r in sha_results],
            'sha_history': {
                'rungs': [{
                    'fidelity': r['fidelity'],
                    'n_evaluated': int(r['n_evaluated']),
                    'best_acc': float(r['best_acc']),
                    'rung_cost': float(r['rung_cost'])
                } for r in sha_history['rungs']],
                'total_cost': float(sha_history['total_cost'])
            },
            'comparison': {
                'sha': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                       for k, v in comparison['sha'].items()},
                'baseline': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in comparison['baseline'].items()},
                'improvement': {k: float(v) for k, v in comparison['improvement'].items()}
            }
        }, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    print(f"\n{'='*60}")
    print("✓ Multi-Fidelity Optimization Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
