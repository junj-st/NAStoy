"""
DARTS: Differentiable Architecture Search

Adapts DARTS to MLP search space. Uses continuous relaxation and
gradient-based optimization to find architectures efficiently.

Key differences from original DARTS:
- Designed for MLPs instead of CNNs
- Simpler operation space: linear transformations and activations
- Adapted for classification on MNIST/CIFAR-10
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Operation choices for DARTS
LAYER_OPS = ['linear', 'skip', 'zero']  # Skip = identity, zero = skip layer
ACTIVATION_OPS = ['relu', 'tanh', 'sigmoid', 'none']


class MixedLayerOp(nn.Module):
    """Mixed operation for layer (linear, skip, or zero)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Define operations
        self.ops = nn.ModuleDict({
            'linear': nn.Linear(in_features, out_features),
            'skip': nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features),
            'zero': Zero(out_features)
        })

    def forward(self, x, weights):
        """
        Args:
            x: Input tensor
            weights: Softmax weights for each operation [3]

        Returns:
            Mixed output
        """
        return sum(w * op(x) for w, op in zip(weights, self.ops.values()))


class MixedActivation(nn.Module):
    """Mixed activation function."""

    def __init__(self):
        super().__init__()

        self.ops = {
            'relu': lambda x: F.relu(x),
            'tanh': lambda x: torch.tanh(x),
            'sigmoid': lambda x: torch.sigmoid(x),
            'none': lambda x: x
        }

    def forward(self, x, weights):
        """
        Args:
            x: Input tensor
            weights: Softmax weights for each activation [4]

        Returns:
            Mixed output
        """
        return sum(w * op(x) for w, op in zip(weights, self.ops.values()))


class Zero(nn.Module):
    """Zero operation (effectively removes the layer)."""

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        # Return zeros with correct output dimension
        return torch.zeros(x.size(0), self.out_features, device=x.device, dtype=x.dtype)


class DARTSNetwork(nn.Module):
    """Differentiable architecture search network for MLPs."""

    def __init__(self, input_size, hidden_sizes, output_size, n_layers=3):
        """
        Args:
            input_size: Input dimension
            hidden_sizes: List of hidden sizes for each layer
            output_size: Output dimension (number of classes)
            n_layers: Number of searchable layers
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.n_layers = n_layers

        # Architecture parameters (to be optimized)
        self.arch_params_layers = nn.ParameterList([
            nn.Parameter(torch.randn(len(LAYER_OPS)))
            for _ in range(n_layers)
        ])

        self.arch_params_activations = nn.ParameterList([
            nn.Parameter(torch.randn(len(ACTIVATION_OPS)))
            for _ in range(n_layers)
        ])

        # Mixed operations
        sizes = [input_size] + hidden_sizes
        self.layers = nn.ModuleList([
            MixedLayerOp(sizes[i], sizes[i+1])
            for i in range(n_layers)
        ])

        self.activations = nn.ModuleList([
            MixedActivation()
            for _ in range(n_layers)
        ])

        # Final classification layer (not searchable)
        self.classifier = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        """Forward pass with mixed operations."""
        # Get architecture weights
        layer_weights = [F.softmax(alpha, dim=0) for alpha in self.arch_params_layers]
        activation_weights = [F.softmax(alpha, dim=0) for alpha in self.arch_params_activations]

        # Forward through layers
        for layer, activation, l_weights, a_weights in zip(
            self.layers, self.activations, layer_weights, activation_weights
        ):
            x = layer(x, l_weights)
            x = activation(x, a_weights)

        # Classification
        x = self.classifier(x)
        return x

    def get_architecture(self):
        """Get discrete architecture (argmax of architecture parameters)."""
        arch = {
            'layers': [],
            'activations': []
        }

        for alpha_layer in self.arch_params_layers:
            op_idx = alpha_layer.argmax().item()
            arch['layers'].append(LAYER_OPS[op_idx])

        for alpha_act in self.arch_params_activations:
            op_idx = alpha_act.argmax().item()
            arch['activations'].append(ACTIVATION_OPS[op_idx])

        return arch

    def get_architecture_weights(self):
        """Get softmax weights for visualization."""
        layer_weights = [F.softmax(alpha, dim=0).detach().cpu().numpy()
                        for alpha in self.arch_params_layers]
        activation_weights = [F.softmax(alpha, dim=0).detach().cpu().numpy()
                             for alpha in self.arch_params_activations]

        return layer_weights, activation_weights

    def arch_parameters(self):
        """Return architecture parameters for optimization."""
        for param in self.arch_params_layers:
            yield param
        for param in self.arch_params_activations:
            yield param


def darts_search(
    dataset='mnist',
    n_layers=3,
    hidden_sizes=[256, 256, 256],
    epochs=50,
    batch_size=256
):
    """
    Run DARTS architecture search.

    Args:
        dataset: Dataset name
        n_layers: Number of searchable layers
        hidden_sizes: Hidden sizes for each layer
        epochs: Number of search epochs
        batch_size: Batch size

    Returns:
        best_arch: Best architecture found
        history: Search history
    """
    print(f"\n{'='*60}")
    print(f"DARTS Search on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Epochs: {epochs}")

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Load data
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST('./data', train=False, transform=transform)
        input_size = 784
        output_size = 10
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        input_size = 3072
        output_size = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Split train into train/val for architecture search
    train_size = int(0.5 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(val_dataset)}")

    # Create model
    model = DARTSNetwork(input_size, hidden_sizes, output_size, n_layers).to(device)

    # Optimizers
    # Separate optimizers for weights and architecture
    optimizer_w = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    optimizer_arch = optim.Adam(model.arch_parameters(), lr=3e-4, betas=(0.5, 0.999))

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_w, epochs)

    criterion = nn.CrossEntropyLoss()

    # Search
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'architectures': []
    }

    print(f"\n{'='*60}")
    print("ARCHITECTURE SEARCH")
    print(f"{'='*60}")

    for epoch in range(epochs):
        # Training phase (update weights)
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)

            optimizer_w.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_w.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)

        # Architecture phase (update architecture parameters)
        model.train()
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)

            optimizer_arch.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_arch.step()

            # Only do a few steps per epoch
            if batch_idx >= 5:
                break

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data = data.view(data.size(0), -1).to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)

        # Update scheduler
        scheduler.step()

        # Record history
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['architectures'].append(model.get_architecture())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Get final architecture
    final_arch = model.get_architecture()

    print(f"\n{'='*60}")
    print("FINAL ARCHITECTURE")
    print(f"{'='*60}")
    print(f"  Layers: {final_arch['layers']}")
    print(f"  Activations: {final_arch['activations']}")

    return final_arch, history, model


def build_discrete_model(arch, input_size, hidden_sizes, output_size):
    """Build discrete model from DARTS architecture."""

    class DiscreteModel(nn.Module):
        def __init__(self):
            super().__init__()

            layers = []
            sizes = [input_size] + hidden_sizes

            for i, (layer_op, act_op) in enumerate(zip(arch['layers'], arch['activations'])):
                # Add layer
                if layer_op == 'linear':
                    layers.append(nn.Linear(sizes[i], sizes[i+1]))
                elif layer_op == 'skip':
                    if sizes[i] == sizes[i+1]:
                        layers.append(nn.Identity())
                    else:
                        layers.append(nn.Linear(sizes[i], sizes[i+1]))
                # 'zero' means skip this layer entirely

                # Add activation
                if act_op == 'relu':
                    layers.append(nn.ReLU())
                elif act_op == 'tanh':
                    layers.append(nn.Tanh())
                elif act_op == 'sigmoid':
                    layers.append(nn.Sigmoid())
                # 'none' means no activation

            layers.append(nn.Linear(hidden_sizes[-1], output_size))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    return DiscreteModel()


def retrain_discrete_architecture(arch, dataset='mnist', epochs=20):
    """Retrain the discrete architecture from scratch."""
    print(f"\n{'='*60}")
    print("RETRAINING DISCRETE ARCHITECTURE")
    print(f"{'='*60}")

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        input_size = 784
        output_size = 10
        hidden_sizes = [256, 256, 256]
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        input_size = 3072
        output_size = 10
        hidden_sizes = [256, 256, 256]

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Build model
    model = build_discrete_model(arch, input_size, hidden_sizes, output_size).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Train
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1).to(device)
                target = target.to(device)

                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)

        test_acc = test_correct / test_total
        best_test_acc = max(best_test_acc, test_acc)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test Acc={test_acc:.4f}")

    print(f"\n  Best Test Accuracy: {best_test_acc:.4f}")

    return best_test_acc, n_params


def visualize_architecture_evolution(history, dataset='mnist'):
    """Visualize how architecture evolves during search."""
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training curves
    ax = axes[0, 0]
    epochs = range(len(history['train_acc']))
    ax.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training Progress', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Architecture evolution - layers
    ax = axes[0, 1]
    layer_counts = {op: [] for op in LAYER_OPS}

    for arch in history['architectures']:
        counts = {op: arch['layers'].count(op) for op in LAYER_OPS}
        for op in LAYER_OPS:
            layer_counts[op].append(counts[op])

    for op, color in zip(LAYER_OPS, ['blue', 'green', 'red']):
        ax.plot(epochs, layer_counts[op], label=op.capitalize(),
               linewidth=2, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Layer Operation Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Architecture evolution - activations
    ax = axes[1, 0]
    act_counts = {op: [] for op in ACTIVATION_OPS}

    for arch in history['architectures']:
        counts = {op: arch['activations'].count(op) for op in ACTIVATION_OPS}
        for op in ACTIVATION_OPS:
            act_counts[op].append(counts[op])

    for op, color in zip(ACTIVATION_OPS, ['blue', 'orange', 'green', 'red']):
        ax.plot(epochs, act_counts[op], label=op.capitalize(),
               linewidth=2, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Activation Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Final architecture
    ax = axes[1, 1]
    final_arch = history['architectures'][-1]

    arch_text = "Final Architecture:\n\n"
    for i, (layer, act) in enumerate(zip(final_arch['layers'], final_arch['activations'])):
        arch_text += f"Layer {i+1}: {layer} → {act}\n"

    ax.text(0.1, 0.5, arch_text, fontsize=11, family='monospace',
           verticalalignment='center')
    ax.axis('off')
    ax.set_title('Discovered Architecture', fontsize=13, fontweight='bold')

    plt.tight_layout()

    output_file = f'{output_dir}/darts_{dataset}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_file}")
    plt.close()


def main():
    print("="*60)
    print("DARTS: Differentiable Architecture Search")
    print("="*60)

    dataset = 'mnist'

    # Run DARTS search
    start_time = time.time()
    final_arch, history, darts_model = darts_search(
        dataset=dataset,
        n_layers=3,
        hidden_sizes=[256, 256, 256],
        epochs=50,
        batch_size=256
    )
    search_time = time.time() - start_time

    print(f"\n  Search time: {search_time:.1f}s ({search_time/60:.1f} min)")

    # Retrain discrete architecture
    final_acc, n_params = retrain_discrete_architecture(final_arch, dataset=dataset, epochs=20)

    # Visualize
    visualize_architecture_evolution(history, dataset)

    # Save results
    output_file = 'results/darts_search.json'
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': dataset,
            'search_time': search_time,
            'final_architecture': final_arch,
            'final_accuracy': final_acc,
            'n_parameters': n_params,
            'history': {
                'train_acc': [float(x) for x in history['train_acc']],
                'val_acc': [float(x) for x in history['val_acc']]
            }
        }, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    print(f"\n{'='*60}")
    print("✓ DARTS Complete!")
    print(f"{'='*60}")
    print(f"\n  Final Architecture: {final_arch}")
    print(f"  Test Accuracy: {final_acc:.4f}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Search Time: {search_time/60:.1f} min")


if __name__ == '__main__':
    main()
