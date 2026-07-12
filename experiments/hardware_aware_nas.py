"""
Hardware-Aware Neural Architecture Search

Optimizes for real-world deployment constraints:
- Inference latency (measured on actual hardware)
- Model size (memory footprint)
- FLOPs (computational complexity)
- Accuracy (validation performance)

Finds Pareto-optimal architectures for different deployment scenarios.
"""

import json
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from thop import profile  # For FLOPs calculation

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator


def measure_hardware_metrics(model, input_size, device='cpu', n_trials=100):
    """Measure hardware performance metrics.

    Args:
        model: PyTorch model
        input_size: Input dimension
        device: Device to measure on
        n_trials: Number of trials for latency measurement

    Returns:
        metrics: Dict with latency, size, flops, memory
    """
    model = model.to(device)
    model.eval()

    # Sample input
    dummy_input = torch.randn(1, input_size).to(device)

    # 1. Measure inference latency
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Measure
    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latency_mean = np.mean(latencies)
    latency_std = np.std(latencies)

    # 2. Model size (MB)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)

    # 3. FLOPs
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops_m = flops / 1e6  # Convert to millions

    # 4. Memory usage (approximate from parameters)
    memory_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2)  # Assume float32

    return {
        'latency_ms': latency_mean,
        'latency_std': latency_std,
        'size_mb': size_mb,
        'flops_m': flops_m,
        'memory_mb': memory_mb,
        'n_params': params
    }


def evaluate_with_hardware(model, config, dataset='cifar10', device='cpu'):
    """Evaluate architecture with both accuracy and hardware metrics.

    Args:
        model: PyTorch model
        config: Architecture config
        dataset: Dataset name
        device: Device for evaluation

    Returns:
        result: Dict with accuracy and hardware metrics
    """
    # Accuracy evaluation
    evaluator = Evaluator(
        dataset=dataset,
        train_budget=10 if dataset == 'cifar10' else 5,
        cache_dir='results/cache_hardware',
        device=device
    )

    accuracy_result = evaluator.evaluate(model, config)

    # Hardware metrics
    input_size = 3072 if dataset == 'cifar10' else 784
    hw_metrics = measure_hardware_metrics(model, input_size, device)

    # Combine
    result = {
        'config': config,
        'val_acc': accuracy_result['val_acc'],
        'train_time': accuracy_result['train_time'],
        **hw_metrics
    }

    return result


def pareto_dominates(a, b, objectives):
    """Check if solution a Pareto-dominates solution b.

    Args:
        a, b: Solutions
        objectives: List of (name, direction) tuples

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


def compute_pareto_frontier(population, objectives):
    """Compute Pareto frontier.

    Args:
        population: List of evaluated architectures
        objectives: List of (name, direction) tuples

    Returns:
        frontier: Non-dominated solutions
    """
    frontier = []

    for candidate in population:
        dominated = False
        for other in population:
            if pareto_dominates(other, candidate, objectives):
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return frontier


def hardware_aware_evolutionary_search(
    dataset='cifar10',
    population_size=30,
    n_generations=20,
    device='cpu'
):
    """Hardware-aware evolutionary search.

    Args:
        dataset: Dataset name
        population_size: Population size
        n_generations: Number of generations
        device: Device for evaluation

    Returns:
        pareto_frontier: Final Pareto-optimal solutions
        all_evaluated: All evaluated architectures
    """
    print(f"\n{'='*60}")
    print(f"Hardware-Aware NAS on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Population: {population_size}")
    print(f"  Generations: {n_generations}")
    print(f"  Device: {device}")

    # Define objectives
    objectives = [
        ('val_acc', 'max'),      # Maximize accuracy
        ('latency_ms', 'min'),   # Minimize latency
        ('size_mb', 'min'),      # Minimize size
        ('flops_m', 'min')       # Minimize FLOPs
    ]

    # Setup
    input_size = 3072 if dataset == 'cifar10' else 784
    search_space = SearchSpace(input_size=input_size, output_size=10)

    # Initialize population
    print(f"\nInitializing population...")
    population = []
    for _ in tqdm(range(population_size), desc="Evaluating"):
        config = search_space.sample_random()
        model = search_space.build_model(config)
        result = evaluate_with_hardware(model, config, dataset, device)
        population.append(result)

    all_evaluated = list(population)

    # Evolution
    print(f"\nEvolving...")
    for gen in range(n_generations):
        # Compute Pareto frontier
        pareto_front = compute_pareto_frontier(population, objectives)

        # Generate offspring (mutations)
        offspring = []
        n_offspring = population_size // 2

        for _ in range(n_offspring):
            # Select parent (prefer Pareto front members)
            if np.random.random() < 0.7 and len(pareto_front) > 0:
                parent = pareto_front[np.random.randint(len(pareto_front))]
            else:
                parent = population[np.random.randint(len(population))]

            # Mutate
            config = search_space.mutate(parent['config'])
            model = search_space.build_model(config)
            result = evaluate_with_hardware(model, config, dataset, device)

            offspring.append(result)
            all_evaluated.append(result)

        # Selection: Keep Pareto fronts
        combined = population + offspring
        fronts = []
        remaining = list(combined)

        while remaining and len(fronts) < population_size:
            front = compute_pareto_frontier(remaining, objectives)
            fronts.append(front)
            remaining = [x for x in remaining if x not in front]

        # Fill population
        population = []
        for front in fronts:
            if len(population) + len(front) <= population_size:
                population.extend(front)
            else:
                # Randomly select from partial front
                n_needed = population_size - len(population)
                population.extend(np.random.choice(front, n_needed, replace=False).tolist())
                break

        # Stats
        pareto_accs = [x['val_acc'] for x in pareto_front]
        pareto_latencies = [x['latency_ms'] for x in pareto_front]

        print(f"Gen {gen+1:2d}: Pareto={len(pareto_front):2d}, "
              f"Acc=[{min(pareto_accs):.3f}, {max(pareto_accs):.3f}], "
              f"Latency=[{min(pareto_latencies):.2f}, {max(pareto_latencies):.2f}]ms")

    # Final Pareto frontier
    final_pareto = compute_pareto_frontier(population, objectives)

    print(f"\nFinal Pareto frontier: {len(final_pareto)} solutions")
    print(f"Total evaluations: {len(all_evaluated)}")

    return final_pareto, all_evaluated, objectives


def find_deployment_solutions(pareto_frontier, constraints):
    """Find best solutions for different deployment scenarios.

    Args:
        pareto_frontier: Pareto-optimal solutions
        constraints: Dict of deployment constraints

    Returns:
        solutions: Solutions meeting constraints, sorted by accuracy
    """
    valid = []

    for solution in pareto_frontier:
        meets_constraints = True

        for constraint_name, max_value in constraints.items():
            if solution.get(constraint_name, float('inf')) > max_value:
                meets_constraints = False
                break

        if meets_constraints:
            valid.append(solution)

    # Sort by accuracy
    valid.sort(key=lambda x: x['val_acc'], reverse=True)

    return valid


def analyze_deployment_scenarios(pareto_frontier):
    """Analyze solutions for different deployment scenarios.

    Args:
        pareto_frontier: Pareto-optimal solutions

    Returns:
        scenarios: Dict of deployment scenario analyses
    """
    print(f"\n{'='*60}")
    print("DEPLOYMENT SCENARIO ANALYSIS")
    print(f"{'='*60}")

    scenarios = {
        'edge': {
            'name': 'Edge Device',
            'constraints': {'size_mb': 2.0, 'latency_ms': 10.0},
            'description': 'IoT/embedded devices'
        },
        'mobile': {
            'name': 'Mobile Device',
            'constraints': {'size_mb': 10.0, 'latency_ms': 50.0},
            'description': 'Smartphones/tablets'
        },
        'server': {
            'name': 'Server/Cloud',
            'constraints': {'size_mb': 100.0, 'latency_ms': 500.0},
            'description': 'Datacenter deployment'
        }
    }

    results = {}

    for scenario_key, scenario in scenarios.items():
        print(f"\n{scenario['name']} ({scenario['description']})")
        print(f"  Constraints: Size < {scenario['constraints']['size_mb']:.1f}MB, "
              f"Latency < {scenario['constraints']['latency_ms']:.1f}ms")

        solutions = find_deployment_solutions(pareto_frontier, scenario['constraints'])

        if solutions:
            best = solutions[0]
            print(f"  Best solution:")
            print(f"    Accuracy: {best['val_acc']:.4f}")
            print(f"    Latency:  {best['latency_ms']:.2f}ms")
            print(f"    Size:     {best['size_mb']:.2f}MB")
            print(f"    FLOPs:    {best['flops_m']:.1f}M")
            print(f"    Arch:     {best['config']['n_layers']+1}L {best['config']['hidden_sizes'][:best['config']['n_layers']+1]}")

            results[scenario_key] = {
                'best': best,
                'n_solutions': len(solutions)
            }
        else:
            print(f"  No solutions meet constraints!")
            results[scenario_key] = None

    return results


def visualize_hardware_aware(pareto_frontier, all_evaluated, dataset='cifar10'):
    """Visualize hardware-aware results."""
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Extract data
    all_accs = [x['val_acc'] for x in all_evaluated]
    all_latencies = [x['latency_ms'] for x in all_evaluated]
    all_sizes = [x['size_mb'] for x in all_evaluated]
    all_flops = [x['flops_m'] for x in all_evaluated]

    pareto_accs = [x['val_acc'] for x in pareto_frontier]
    pareto_latencies = [x['latency_ms'] for x in pareto_frontier]
    pareto_sizes = [x['size_mb'] for x in pareto_frontier]
    pareto_flops = [x['flops_m'] for x in pareto_frontier]

    # 1. Accuracy vs Latency
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(all_latencies, all_accs, alpha=0.4, s=30, label='All', color='lightblue')
    ax.scatter(pareto_latencies, pareto_accs, s=150, label='Pareto',
              color='red', marker='*', edgecolors='k', linewidths=1.5)
    ax.set_xlabel('Latency (ms)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Accuracy vs Latency', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Accuracy vs Size
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(all_sizes, all_accs, alpha=0.4, s=30, label='All', color='lightgreen')
    ax.scatter(pareto_sizes, pareto_accs, s=150, label='Pareto',
              color='red', marker='*', edgecolors='k', linewidths=1.5)
    ax.set_xlabel('Model Size (MB)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Accuracy vs Model Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Accuracy vs FLOPs
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(all_flops, all_accs, alpha=0.4, s=30, label='All', color='lightyellow')
    ax.scatter(pareto_flops, pareto_accs, s=150, label='Pareto',
              color='red', marker='*', edgecolors='k', linewidths=1.5)
    ax.set_xlabel('FLOPs (M)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Accuracy vs FLOPs', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Latency vs Size (colored by accuracy)
    ax = fig.add_subplot(gs[1, 0])
    scatter = ax.scatter(all_latencies, all_sizes, c=all_accs, alpha=0.6,
                        s=30, cmap='viridis', edgecolors='k', linewidths=0.3)
    ax.scatter(pareto_latencies, pareto_sizes, s=150,
              color='red', marker='*', edgecolors='k', linewidths=1.5, zorder=5)
    plt.colorbar(scatter, ax=ax, label='Accuracy')
    ax.set_xlabel('Latency (ms)', fontsize=11)
    ax.set_ylabel('Model Size (MB)', fontsize=11)
    ax.set_title('Latency vs Size (colored by Accuracy)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. Deployment scenarios
    ax = fig.add_subplot(gs[1, 1:])

    # Sort Pareto by accuracy
    sorted_pareto = sorted(pareto_frontier, key=lambda x: x['val_acc'], reverse=True)

    labels = [f"Arch {i+1}" for i in range(len(sorted_pareto))]
    accs = [x['val_acc'] for x in sorted_pareto]
    latencies = [x['latency_ms'] for x in sorted_pareto]
    sizes = [x['size_mb'] for x in sorted_pareto]

    x = np.arange(len(labels))
    width = 0.25

    # Normalize for visualization
    latencies_norm = np.array(latencies) / max(latencies)
    sizes_norm = np.array(sizes) / max(sizes)

    ax.bar(x - width, accs, width, label='Accuracy', color='steelblue')
    ax.bar(x, latencies_norm, width, label='Latency (normalized)', color='orange')
    ax.bar(x + width, sizes_norm, width, label='Size (normalized)', color='green')

    ax.set_xlabel('Architecture', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Pareto-Optimal Solutions Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Hardware-Aware NAS: {dataset.upper()}', fontsize=14, fontweight='bold', y=0.995)

    output_file = f'{output_dir}/hardware_aware_{dataset}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_file}")
    plt.close()


def main():
    print("="*60)
    print("Hardware-Aware Neural Architecture Search")
    print("="*60)

    dataset = 'cifar10'
    device = 'cpu'

    # Run hardware-aware search (reduced for faster execution)
    pareto_frontier, all_evaluated, objectives = hardware_aware_evolutionary_search(
        dataset=dataset,
        population_size=15,
        n_generations=10,
        device=device
    )

    # Print top results
    print(f"\n{'='*60}")
    print("PARETO-OPTIMAL ARCHITECTURES")
    print(f"{'='*60}")

    # Sort by accuracy
    sorted_pareto = sorted(pareto_frontier, key=lambda x: x['val_acc'], reverse=True)

    print(f"\nTop 5 by accuracy:")
    for i, result in enumerate(sorted_pareto[:5]):
        print(f"\n{i+1}. Accuracy: {result['val_acc']:.4f}")
        print(f"   Latency:  {result['latency_ms']:.2f}ms")
        print(f"   Size:     {result['size_mb']:.2f}MB")
        print(f"   FLOPs:    {result['flops_m']:.1f}M")
        print(f"   Arch:     {result['config']['n_layers']+1}L {result['config']['hidden_sizes'][:result['config']['n_layers']+1]}")

    # Analyze deployment scenarios
    deployment_results = analyze_deployment_scenarios(pareto_frontier)

    # Visualize
    visualize_hardware_aware(pareto_frontier, all_evaluated, dataset)

    # Save results
    output_file = 'results/hardware_aware_nas.json'
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': dataset,
            'device': device,
            'pareto_frontier': [{
                'config': r['config'],
                'val_acc': float(r['val_acc']),
                'latency_ms': float(r['latency_ms']),
                'size_mb': float(r['size_mb']),
                'flops_m': float(r['flops_m']),
                'n_params': int(r['n_params'])
            } for r in sorted_pareto],
            'deployment_scenarios': {
                k: {
                    'best': {
                        'val_acc': float(v['best']['val_acc']),
                        'latency_ms': float(v['best']['latency_ms']),
                        'size_mb': float(v['best']['size_mb']),
                        'config': v['best']['config']
                    } if v else None,
                    'n_solutions': v['n_solutions'] if v else 0
                } if v else None
                for k, v in deployment_results.items()
            },
            'summary': {
                'n_pareto': len(pareto_frontier),
                'n_evaluated': len(all_evaluated),
                'best_accuracy': max(r['val_acc'] for r in pareto_frontier),
                'min_latency': min(r['latency_ms'] for r in pareto_frontier),
                'min_size': min(r['size_mb'] for r in pareto_frontier)
            }
        }, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    print(f"\n{'='*60}")
    print("✓ Hardware-Aware NAS Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
