"""
Evolutionary search experiment on CIFAR-10.
"""

import json
import yaml
from pathlib import Path

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator
from nas.strategies.evolutionary import EvolutionarySearch
from nas import viz


def main():
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'cifar10_evolutionary.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Evolutionary Search on CIFAR-10")
    print("=" * 60)
    print(f"Config: {config}\n")

    # Initialize search space (CIFAR-10: 32x32x3 = 3072 input features)
    search_space = SearchSpace(input_size=3072, output_size=10)

    # Initialize evaluator
    evaluator = Evaluator(
        dataset=config['dataset'],
        train_budget=config['train_budget'],
        cache_dir=config['cache_dir']
    )

    # Initialize evolutionary search
    search = EvolutionarySearch(
        search_space=search_space,
        evaluator=evaluator,
        population_size=config['population_size'],
        n_generations=config['n_generations'],
        tournament_size=config['tournament_size'],
        mutation_rate=config['mutation_rate'],
        seed=config.get('seed')
    )

    # Run search
    population = search.search()

    # Prepare results for saving
    results_data = {
        'strategy': config['strategy'],
        'dataset': config['dataset'],
        'population_size': config['population_size'],
        'n_generations': config['n_generations'],
        'train_budget': config['train_budget'],
        'results': population,
        'history': search.history,
        'best_architecture': population[0]
    }

    # Save results
    results_file = Path(config['results_file'])
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")

    # Generate visualizations
    plot_dir = Path('plots')
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Trajectory plot
    trajectory_plot = plot_dir / 'cifar10_evolutionary_trajectory.png'
    viz.plot_search_trajectory(str(results_file), str(trajectory_plot))

    # Diversity plot
    diversity_plot = plot_dir / 'cifar10_evolutionary_diversity.png'
    viz.plot_population_diversity(str(results_file), str(diversity_plot))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best = population[0]
    print(f"Best architecture found:")
    print(f"  Validation accuracy: {best['val_acc']:.4f}")
    print(f"  Train accuracy: {best['train_acc']:.4f}")
    print(f"  Parameters: {best['n_params']:,}")
    print(f"  Training time: {best['train_time']:.2f}s")
    print(f"  Architecture:")
    print(f"    Layers: {best['config']['n_layers']}")
    print(f"    Hidden sizes: {best['config']['hidden_sizes']}")
    print(f"    Activations: {best['config']['activations']}")
    print(f"    Dropouts: {best['config']['dropouts']}")

    # Print generation stats
    print(f"\nEvolution Progress:")
    print(f"  Initial best: {search.history[0]['best_acc']:.4f}")
    print(f"  Final best: {search.history[-1]['best_acc']:.4f}")
    print(f"  Improvement: +{(search.history[-1]['best_acc'] - search.history[0]['best_acc']):.4f}")
    print(f"  Final diversity: {search.history[-1]['diversity']:.2f}")

    # Print top 5
    print(f"\nTop 5 architectures in final population:")
    for i, r in enumerate(population[:5], 1):
        print(f"  {i}. Acc: {r['val_acc']:.4f}, "
              f"Params: {r['n_params']:,}, "
              f"Layers: {r['config']['n_layers']}")

    print("\n✓ Evolutionary search on CIFAR-10 completed!")


if __name__ == '__main__':
    main()
