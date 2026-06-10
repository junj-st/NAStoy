"""
RL controller experiment on MNIST.
"""

import json
import yaml
from pathlib import Path

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator
from nas.strategies.rl_controller import RLController
from nas import viz


def main():
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'mnist_rl.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("RL Controller Search on MNIST")
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

    # Initialize RL controller
    search = RLController(
        search_space=search_space,
        evaluator=evaluator,
        n_episodes=config['n_episodes'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        hidden_size=config['hidden_size'],
        entropy_coeff=config['entropy_coeff'],
        entropy_decay=config['entropy_decay'],
        reward_lambda=config['reward_lambda'],
        seed=config.get('seed')
    )

    # Run search
    results = search.search()

    # Prepare results for saving
    results_data = {
        'strategy': config['strategy'],
        'dataset': config['dataset'],
        'n_episodes': config['n_episodes'],
        'batch_size': config['batch_size'],
        'train_budget': config['train_budget'],
        'results': results[:100],  # Save top 100 architectures
        'history': search.history,  # Episode-by-episode stats
        'best_architecture': results[0]
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
    trajectory_plot = plot_dir / 'mnist_rl_trajectory.png'
    viz.plot_search_trajectory(str(results_file), str(trajectory_plot))

    # RL metrics plot
    rl_metrics_plot = plot_dir / 'mnist_rl_metrics.png'
    viz.plot_rl_metrics(str(results_file), str(rl_metrics_plot))

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

    # Print RL training stats
    print(f"\nRL Training Progress:")
    print(f"  Initial reward: {search.history[0]['mean_reward']:.4f}")
    print(f"  Final reward: {search.history[-1]['mean_reward']:.4f}")
    print(f"  Initial accuracy: {search.history[0]['mean_acc']:.4f}")
    print(f"  Final accuracy: {search.history[-1]['mean_acc']:.4f}")
    print(f"  Final entropy: {search.history[-1]['entropy']:.4f}")

    # Print top 5
    print(f"\nTop 5 architectures found:")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. Acc: {r['val_acc']:.4f}, "
              f"Params: {r['n_params']:,}, "
              f"Layers: {r['config']['n_layers']}")

    print("\n✓ RL search completed!")


if __name__ == '__main__':
    main()
