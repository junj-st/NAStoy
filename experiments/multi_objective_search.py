"""
Multi-Objective Architecture Search

Finds architectures that balance multiple objectives:
- Maximize accuracy
- Minimize parameters
- Minimize training time

Uses NSGA-II style Pareto optimization with evolutionary search.
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator


def dominates(a, b, objectives):
    """Check if solution a Pareto-dominates solution b.

    Args:
        a, b: Solutions with objective values
        objectives: List of (name, direction) tuples
            direction: 'max' or 'min'

    Returns:
        True if a dominates b
    """
    better_in_one = False

    for obj_name, direction in objectives:
        a_val = a[obj_name]
        b_val = b[obj_name]

        if direction == 'max':
            if a_val < b_val:
                return False
            if a_val > b_val:
                better_in_one = True
        else:  # min
            if a_val > b_val:
                return False
            if a_val < b_val:
                better_in_one = True

    return better_in_one


def compute_pareto_front(population, objectives):
    """Compute Pareto frontier from population.

    Args:
        population: List of solutions
        objectives: List of (name, direction) tuples

    Returns:
        pareto_front: List of non-dominated solutions
    """
    pareto_front = []

    for candidate in population:
        dominated = False
        for other in population:
            if dominates(other, candidate, objectives):
                dominated = True
                break

        if not dominated:
            pareto_front.append(candidate)

    return pareto_front


def crowding_distance(front, objectives):
    """Compute crowding distance for diversity preservation.

    Args:
        front: List of solutions in the front
        objectives: List of (name, direction) tuples

    Returns:
        distances: Dict mapping solution index to crowding distance
    """
    if len(front) <= 2:
        return {i: float('inf') for i in range(len(front))}

    distances = {i: 0.0 for i in range(len(front))}

    for obj_name, direction in objectives:
        # Sort by this objective
        sorted_indices = sorted(range(len(front)),
                               key=lambda i: front[i][obj_name])

        # Boundary points get infinite distance
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        # Normalize range
        obj_min = front[sorted_indices[0]][obj_name]
        obj_max = front[sorted_indices[-1]][obj_name]
        obj_range = obj_max - obj_min

        if obj_range == 0:
            continue

        # Compute crowding distance
        for i in range(1, len(front) - 1):
            idx = sorted_indices[i]
            idx_prev = sorted_indices[i - 1]
            idx_next = sorted_indices[i + 1]

            distances[idx] += (front[idx_next][obj_name] -
                             front[idx_prev][obj_name]) / obj_range

    return distances


def multi_objective_evolutionary_search(
    dataset='cifar10',
    population_size=20,
    n_generations=30,
    mutation_rate=0.3
):
    """Multi-objective evolutionary search.

    Args:
        dataset: Dataset to search on
        population_size: Population size
        n_generations: Number of generations
        mutation_rate: Mutation probability

    Returns:
        pareto_front: Final Pareto frontier
        all_evaluated: All evaluated architectures
    """
    print(f"\n{'='*60}")
    print(f"Multi-Objective Evolutionary Search on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Population size: {population_size}")
    print(f"  Generations: {n_generations}")
    print(f"  Mutation rate: {mutation_rate}")

    # Define objectives
    objectives = [
        ('val_acc', 'max'),      # Maximize accuracy
        ('n_params', 'min'),     # Minimize parameters
        ('train_time', 'min')    # Minimize training time
    ]

    # Setup
    input_size = 3072 if dataset == 'cifar10' else 784
    search_space = SearchSpace(input_size=input_size, output_size=10)
    evaluator = Evaluator(
        dataset=dataset,
        train_budget=20 if dataset == 'cifar10' else 10,
        cache_dir='results/cache'
    )

    # Initialize population
    print(f"\nInitializing population...")
    population = []
    for _ in tqdm(range(population_size), desc="Evaluating initial population"):
        config = search_space.sample_random()
        model = search_space.build_model(config)
        result = evaluator.evaluate(model, config)
        population.append(result)

    all_evaluated = list(population)

    # Evolution
    print(f"\nEvolving...")
    for gen in range(n_generations):
        # Compute Pareto fronts
        pareto_front = compute_pareto_front(population, objectives)

        # Compute crowding distances
        distances = crowding_distance(pareto_front, objectives)

        # Selection: tournament selection favoring Pareto front
        offspring = []
        n_offspring = population_size // 2

        for _ in range(n_offspring):
            # Tournament: pick 2 random, prefer Pareto member
            idx1, idx2 = np.random.choice(len(population), 2, replace=False)
            parent1 = population[idx1]
            parent2 = population[idx2]

            # Prefer Pareto front members
            p1_in_pareto = parent1 in pareto_front
            p2_in_pareto = parent2 in pareto_front

            if p1_in_pareto and not p2_in_pareto:
                parent = parent1
            elif p2_in_pareto and not p1_in_pareto:
                parent = parent2
            elif p1_in_pareto and p2_in_pareto:
                # Both in front, pick by crowding distance
                p1_idx = pareto_front.index(parent1)
                p2_idx = pareto_front.index(parent2)
                if distances[p1_idx] > distances[p2_idx]:
                    parent = parent1
                else:
                    parent = parent2
            else:
                # Neither in front, pick randomly
                parent = parent1 if np.random.random() < 0.5 else parent2

            # Mutate
            config = parent['config']
            if np.random.random() < mutation_rate:
                config = search_space.mutate(config)

            # Evaluate
            model = search_space.build_model(config)
            result = evaluator.evaluate(model, config)
            offspring.append(result)
            all_evaluated.append(result)

        # Combine population and offspring
        combined = population + offspring

        # Select next generation (Pareto fronts + crowding distance)
        fronts = []
        remaining = list(combined)

        while remaining and len(fronts) < population_size:
            front = compute_pareto_front(remaining, objectives)
            fronts.append(front)
            remaining = [x for x in remaining if x not in front]

        # Fill population with fronts
        population = []
        for front in fronts:
            if len(population) + len(front) <= population_size:
                population.extend(front)
            else:
                # Partial front - select by crowding distance
                distances = crowding_distance(front, objectives)
                sorted_front = sorted(enumerate(front),
                                    key=lambda x: distances[x[0]],
                                    reverse=True)
                n_needed = population_size - len(population)
                population.extend([x[1] for x in sorted_front[:n_needed]])
                break

        # Stats
        pareto_accs = [x['val_acc'] for x in pareto_front]
        pareto_params = [x['n_params'] for x in pareto_front]

        print(f"Gen {gen+1:2d}: Pareto size={len(pareto_front):2d}, "
              f"Acc=[{min(pareto_accs):.4f}, {max(pareto_accs):.4f}], "
              f"Params=[{min(pareto_params)}, {max(pareto_params)}]")

    # Final Pareto front
    final_pareto = compute_pareto_front(population, objectives)

    print(f"\nFinal Pareto frontier: {len(final_pareto)} solutions")
    print(f"Total evaluations: {len(all_evaluated)}")

    return final_pareto, all_evaluated, objectives


def visualize_pareto_front(pareto_front, all_evaluated, dataset='cifar10'):
    """Visualize Pareto frontier."""
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)

    # Extract values
    all_accs = [x['val_acc'] for x in all_evaluated]
    all_params = [x['n_params'] for x in all_evaluated]

    pareto_accs = [x['val_acc'] for x in pareto_front]
    pareto_params = [x['n_params'] for x in pareto_front]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy vs Parameters
    axes[0].scatter(all_params, all_accs, alpha=0.4, s=50,
                   label='All evaluated', color='lightblue', edgecolors='k', linewidths=0.5)
    axes[0].scatter(pareto_params, pareto_accs, s=150,
                   label='Pareto front', color='red', edgecolors='k', linewidths=1.5, marker='*')
    axes[0].set_xlabel('Parameters', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title(f'Pareto Frontier: Accuracy vs Parameters\n{dataset.upper()}',
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy vs Training Time
    all_times = [x['train_time'] for x in all_evaluated]
    pareto_times = [x['train_time'] for x in pareto_front]

    axes[1].scatter(all_times, all_accs, alpha=0.4, s=50,
                   label='All evaluated', color='lightgreen', edgecolors='k', linewidths=0.5)
    axes[1].scatter(pareto_times, pareto_accs, s=150,
                   label='Pareto front', color='red', edgecolors='k', linewidths=1.5, marker='*')
    axes[1].set_xlabel('Training Time (s)', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'Pareto Frontier: Accuracy vs Training Time\n{dataset.upper()}',
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = f'{output_dir}/multi_objective_{dataset}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_file}")
    plt.close()


def main():
    print("="*60)
    print("Multi-Objective Architecture Search")
    print("="*60)

    # Run on CIFAR-10
    pareto_front, all_evaluated, objectives = multi_objective_evolutionary_search(
        dataset='cifar10',
        population_size=20,
        n_generations=30,
        mutation_rate=0.3
    )

    # Print Pareto solutions
    print(f"\n{'='*60}")
    print("PARETO FRONTIER")
    print(f"{'='*60}")

    # Sort by accuracy
    sorted_pareto = sorted(pareto_front, key=lambda x: x['val_acc'], reverse=True)

    print(f"\nTop 5 Pareto-optimal architectures:")
    for i, result in enumerate(sorted_pareto[:5]):
        print(f"\n{i+1}. Accuracy: {result['val_acc']:.4f}")
        print(f"   Parameters: {result['n_params']:,}")
        print(f"   Train time: {result['train_time']:.1f}s")
        print(f"   Architecture: {result['config']['n_layers']+1}L {result['config']['hidden_sizes'][:result['config']['n_layers']+1]}")
        print(f"   Activations: {result['config']['activations'][:result['config']['n_layers']+1]}")

    # Visualize
    visualize_pareto_front(pareto_front, all_evaluated, dataset='cifar10')

    # Save results
    output_file = 'results/multi_objective_search.json'
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': 'cifar10',
            'population_size': 20,
            'n_generations': 30,
            'pareto_front': sorted_pareto,
            'all_evaluated': all_evaluated,
            'summary': {
                'n_pareto': len(pareto_front),
                'n_total_evaluated': len(all_evaluated),
                'best_acc': max(x['val_acc'] for x in pareto_front),
                'min_params': min(x['n_params'] for x in pareto_front),
                'min_time': min(x['train_time'] for x in pareto_front)
            }
        }, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    print(f"\n{'='*60}")
    print("✓ Multi-Objective Search Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
