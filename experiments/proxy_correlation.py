"""
Zero-cost proxy correlation analysis.

Loads all evaluated architectures and computes correlation between
proxy scores and actual validation accuracy.
"""

import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from nas.search_space import SearchSpace
from nas.proxy import compute_proxies
from nas import viz


def main():
    print("=" * 60)
    print("Zero-Cost Proxy Correlation Analysis")
    print("=" * 60)

    # Paths to result files
    result_files = [
        'results/mnist_random.json',
        'results/mnist_evolutionary.json',
        'results/mnist_rl.json'
    ]

    # Load all architectures
    print("\nLoading evaluated architectures...")
    all_results = []
    for file in result_files:
        if Path(file).exists():
            with open(file, 'r') as f:
                data = json.load(f)
                all_results.extend(data['results'])
            print(f"  Loaded {len(data['results'])} from {file}")

    print(f"\nTotal evaluations: {len(all_results)}")

    # Deduplicate architectures by config
    seen_configs = set()
    unique_results = []
    for result in all_results:
        # Create config hash
        config_str = json.dumps(result['config'], sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()

        if config_hash not in seen_configs:
            seen_configs.add(config_hash)
            unique_results.append(result)

    print(f"Unique architectures: {len(unique_results)}")

    # Initialize search space and data loader
    search_space = SearchSpace(input_size=784, output_size=10)

    # Load MNIST data for proxies
    print("\nLoading MNIST data for proxy computation...")
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    # Compute proxies for each architecture
    print("\nComputing proxies for each architecture...")
    proxy_results = []

    for result in tqdm(unique_results, desc="Computing proxies"):
        config = result['config']
        model = search_space.build_model(config)

        # Compute proxies
        proxies = compute_proxies(model, data_loader, input_shape=784, device='cpu')

        proxy_results.append({
            'val_acc': result['val_acc'],
            'grad_norm': proxies['grad_norm'],
            'synflow': proxies['synflow'],
            'n_params': result['n_params']
        })

    # Extract arrays for correlation
    accuracies = np.array([r['val_acc'] for r in proxy_results])
    grad_norms = np.array([r['grad_norm'] for r in proxy_results])
    synflows = np.array([r['synflow'] for r in proxy_results])

    # Compute Spearman correlations
    grad_norm_corr, grad_norm_p = spearmanr(grad_norms, accuracies)
    synflow_corr, synflow_p = spearmanr(synflows, accuracies)

    # Print results
    print("\n" + "=" * 60)
    print("CORRELATION RESULTS")
    print("=" * 60)
    print(f"Gradient Norm:")
    print(f"  Spearman ρ = {grad_norm_corr:.4f} (p = {grad_norm_p:.4e})")
    print(f"\nSynFlow:")
    print(f"  Spearman ρ = {synflow_corr:.4f} (p = {synflow_p:.4e})")
    print()

    # Interpret correlations
    def interpret_correlation(rho):
        if abs(rho) < 0.3:
            return "weak"
        elif abs(rho) < 0.7:
            return "moderate"
        else:
            return "strong"

    print(f"Gradient Norm shows {interpret_correlation(grad_norm_corr)} correlation with accuracy")
    print(f"SynFlow shows {interpret_correlation(synflow_corr)} correlation with accuracy")

    # Save results
    output_file = 'results/proxy_correlation.json'
    with open(output_file, 'w') as f:
        json.dump({
            'n_architectures': len(proxy_results),
            'correlations': {
                'grad_norm': {
                    'rho': grad_norm_corr,
                    'p_value': grad_norm_p
                },
                'synflow': {
                    'rho': synflow_corr,
                    'p_value': synflow_p
                }
            },
            'data': proxy_results
        }, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    # Generate scatter plots
    print("\nGenerating scatter plots...")
    Path('plots').mkdir(exist_ok=True)

    viz.plot_proxy_correlation(
        proxy_results,
        'grad_norm',
        grad_norm_corr,
        'plots/proxy_grad_norm.png'
    )
    print("  Saved plots/proxy_grad_norm.png")

    viz.plot_proxy_correlation(
        proxy_results,
        'synflow',
        synflow_corr,
        'plots/proxy_synflow.png'
    )
    print("  Saved plots/proxy_synflow.png")

    print("\n✓ Proxy correlation analysis completed!")


if __name__ == '__main__':
    main()
