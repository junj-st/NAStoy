"""
Random search experiment on MNIST.
"""

import json
import yaml
from pathlib import Path

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator
from nas.strategies.random_search import RandomSearch
from nas import viz


def main():
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'mnist_random.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Random Search on MNIST")
    print("=" * 60)
    print(f"Config: {config}\n")

    # Initialize search space
    search_space = SearchSpace(input_size=784, output_size=10)

    # Initialize evaluator
    evaluator = Evaluator(
        dataset=config['dataset'],
        train_budget=config['train_budget'],
        cache_dir=config['cache_dir']
    )

    # Initialize random search
    search = RandomSearch(
        search_space=search_space,
        evaluator=evaluator,
        n_evaluations=config['n_evaluations'],
        seed=config.get('seed')
    )

    # Run search
    results = search.search()

    # Prepare results for saving
    results_data = {
        'strategy': config['strategy'],
        'dataset': config['dataset'],
        'n_evaluations': config['n_evaluations'],
        'train_budget': config['train_budget'],
        'results': results,
        'best_architecture': results[0]
    }

    # Save results
    results_file = Path(config['results_file'])
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")

    # Generate visualization
    plot_file = Path('plots') / 'mnist_random_trajectory.png'
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    viz.plot_search_trajectory(str(results_file), str(plot_file))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best = results[0]
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

    # Print top 5
    print(f"\nTop 5 architectures:")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. Acc: {r['val_acc']:.4f}, "
              f"Params: {r['n_params']:,}, "
              f"Layers: {r['config']['n_layers']}")

    print("\n✓ Random search completed!")


if __name__ == '__main__':
    main()
