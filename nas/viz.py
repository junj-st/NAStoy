"""
Visualization utilities for NAS results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


def plot_search_trajectory(results_file: str, output_file: Optional[str] = None):
    """Plot search trajectory showing best accuracy over evaluations.

    Args:
        results_file: Path to results JSON file
        output_file: Path to save plot (if None, display instead)
    """
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data['results']

    # Calculate best accuracy so far at each evaluation
    best_acc_so_far = []
    current_best = 0.0

    for result in results:
        val_acc = result['val_acc']
        current_best = max(current_best, val_acc)
        best_acc_so_far.append(current_best)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_acc_so_far) + 1), best_acc_so_far, 'b-', linewidth=2)
    plt.xlabel('Evaluation Number', fontsize=12)
    plt.ylabel('Best Validation Accuracy', fontsize=12)
    plt.title(f'Search Trajectory - {data.get("strategy", "Unknown")} on {data.get("dataset", "Unknown")}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_population_diversity(results_file: str, output_file: Optional[str] = None):
    """Plot population diversity over generations (for evolutionary search).

    Args:
        results_file: Path to results JSON file
        output_file: Path to save plot
    """
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)

    history = data['history']

    # Extract generation stats
    generations = [h['generation'] for h in history]
    diversities = [h['diversity'] for h in history]
    best_accs = [h['best_acc'] for h in history]
    mean_accs = [h['mean_acc'] for h in history]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Accuracy over generations
    ax1.plot(generations, best_accs, 'b-', linewidth=2, label='Best')
    ax1.plot(generations, mean_accs, 'g--', linewidth=2, label='Mean')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title(f'Evolutionary Search - Accuracy Progress', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Diversity over generations
    ax2.plot(generations, diversities, 'r-', linewidth=2)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Population Diversity (Hamming Distance)', fontsize=12)
    ax2.set_title('Population Diversity Over Generations', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved diversity plot to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_rl_metrics(results_file: str, output_file: Optional[str] = None):
    """Plot RL controller metrics (entropy, reward).

    Args:
        results_file: Path to results JSON file
        output_file: Path to save plot
    """
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)

    history = data['history']

    # Extract episode stats
    episodes = [h['episode'] for h in history]
    mean_rewards = [h['mean_reward'] for h in history]
    mean_accs = [h['mean_acc'] for h in history]
    entropies = [h['entropy'] for h in history]
    baselines = [h['baseline'] for h in history]

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # Plot 1: Rewards over episodes
    ax1.plot(episodes, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
    ax1.plot(episodes, baselines, 'r--', linewidth=2, label='Baseline')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('RL Controller - Reward Progress', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy over episodes
    ax2.plot(episodes, mean_accs, 'g-', linewidth=2)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Mean Validation Accuracy', fontsize=12)
    ax2.set_title('Accuracy Progress', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Entropy over episodes
    ax3.plot(episodes, entropies, 'purple', linewidth=2)
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Controller Entropy', fontsize=12)
    ax3.set_title('Exploration (Entropy) Over Episodes', fontsize=14)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved RL metrics plot to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_proxy_correlation(architectures: List[Dict], proxy_name: str, correlation: float, output_file: Optional[str] = None):
    """Plot correlation between proxy scores and actual accuracy.

    Args:
        architectures: List of evaluated architectures with scores
        proxy_name: Name of proxy ('grad_norm' or 'synflow')
        correlation: Spearman correlation coefficient
        output_file: Path to save plot
    """
    # Extract data
    proxy_scores = np.array([arch[proxy_name] for arch in architectures])
    accuracies = np.array([arch['val_acc'] for arch in architectures])

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(proxy_scores, accuracies, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

    # Add best fit line
    z = np.polyfit(proxy_scores, accuracies, 1)
    p = np.poly1d(z)
    x_line = np.linspace(proxy_scores.min(), proxy_scores.max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit')

    # Labels and title
    proxy_label = 'Gradient Norm' if proxy_name == 'grad_norm' else 'SynFlow Score'
    plt.xlabel(proxy_label, fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title(f'{proxy_label} vs Validation Accuracy\nSpearman ρ = {correlation:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_search_space_heatmap(architectures: List[Dict], output_file: Optional[str] = None):
    """Plot heatmap of search space exploration.

    Shows which regions of search space (n_layers × hidden_size) were explored.

    Args:
        architectures: List of evaluated architectures
        output_file: Path to save plot
    """
    # TODO: Implement search space heatmap
    # X-axis: primary hidden size
    # Y-axis: number of layers
    # Color: average accuracy in that region
    raise NotImplementedError


def plot_architecture_diagram(config: Dict, output_file: Optional[str] = None):
    """Render architecture as a diagram.

    Args:
        config: Architecture configuration
        output_file: Path to save diagram
    """
    # TODO: Implement architecture diagram
    # Simple visualization of layer structure
    raise NotImplementedError
