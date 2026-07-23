"""
ENAS: Efficient Neural Architecture Search via Parameter Sharing

Key Innovation:
- Train ONE supernet containing all possible architectures
- Search by sampling sub-networks from the supernet (reusing weights)
- 1000x faster than training each architecture from scratch

Comparison to other methods:
- Random/Evolutionary/RL: Train each architecture separately (~10-20 epochs each)
- DARTS: Differentiable but still trains the full search space
- ENAS: Train supernet once, search by sampling (no retraining)
"""

import json
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# Supernet: Contains all possible architectures
# ============================================================

class MixedOp(nn.Module):
    """A mixed operation that can be any of: linear, skip, zero"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ops = nn.ModuleDict({
            'linear': nn.Linear(in_features, out_features),
            'skip': nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features),
            'zero': Zero(out_features)
        })
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, op_name):
        """Execute the selected operation"""
        return self.ops[op_name](x)


class Zero(nn.Module):
    """Zero operation (outputs zeros)"""
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        return torch.zeros(x.size(0), self.out_features, device=x.device, dtype=x.dtype)


class Supernet(nn.Module):
    """
    Supernet containing all possible MLP architectures.

    Search space for each layer:
    - Operations: {linear, skip, zero}
    - Activations: {relu, tanh, sigmoid, none}
    - Hidden sizes: {64, 128, 256} (fixed per layer in supernet)
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Build layers
        self.layers = nn.ModuleList()
        in_features = input_size
        for i in range(n_layers):
            self.layers.append(MixedOp(in_features, hidden_size))
            in_features = hidden_size

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'none': nn.Identity()
        }

    def forward(self, x, architecture):
        """
        Forward pass using the specified architecture.

        architecture: dict with keys 'operations' and 'activations'
            operations: list of operation names for each layer
            activations: list of activation names for each layer
        """
        x = x.view(x.size(0), -1)

        for i, (layer, op_name, act_name) in enumerate(zip(
            self.layers,
            architecture['operations'],
            architecture['activations']
        )):
            x = layer(x, op_name)
            x = self.activations[act_name](x)

        x = self.output_layer(x)
        return x


# ============================================================
# Controller: Samples architectures
# ============================================================

class Controller(nn.Module):
    """
    RNN controller that samples architectures.

    For each layer, outputs:
    - Operation choice: {linear, skip, zero}
    - Activation choice: {relu, tanh, sigmoid, none}
    """
    def __init__(self, n_layers=3, hidden_size=64):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        # Operation choices
        self.op_choices = ['linear', 'skip', 'zero']
        self.n_ops = len(self.op_choices)

        # Activation choices
        self.act_choices = ['relu', 'tanh', 'sigmoid', 'none']
        self.n_acts = len(self.act_choices)

        # LSTM controller
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        # Embedding for start token and previous choices
        self.embedding = nn.Embedding(self.n_ops + self.n_acts + 1, self.hidden_size)

        # Output heads
        self.op_head = nn.Linear(self.hidden_size, self.n_ops)
        self.act_head = nn.Linear(self.hidden_size, self.n_acts)

    def sample(self, batch_size=1, temperature=1.0):
        """
        Sample architectures from the controller.

        Returns:
            architectures: list of dicts with 'operations' and 'activations'
            log_probs: log probabilities for REINFORCE
            entropies: entropy for exploration bonus
        """
        architectures = []
        log_probs_list = []
        entropies_list = []

        for _ in range(batch_size):
            ops = []
            acts = []
            log_probs = []
            entropies = []

            # Initialize LSTM state
            h = torch.zeros(1, self.hidden_size)
            c = torch.zeros(1, self.hidden_size)

            # Start token
            input_token = torch.zeros(1, dtype=torch.long)
            inputs = self.embedding(input_token)

            for layer_idx in range(self.n_layers):
                # LSTM step
                h, c = self.lstm(inputs, (h, c))

                # Sample operation
                op_logits = self.op_head(h) / temperature
                op_probs = F.softmax(op_logits, dim=-1)
                op_dist = torch.distributions.Categorical(op_probs)
                op_idx = op_dist.sample()

                log_probs.append(op_dist.log_prob(op_idx))
                entropies.append(op_dist.entropy())
                ops.append(self.op_choices[op_idx.item()])

                # Sample activation
                act_logits = self.act_head(h) / temperature
                act_probs = F.softmax(act_logits, dim=-1)
                act_dist = torch.distributions.Categorical(act_probs)
                act_idx = act_dist.sample()

                log_probs.append(act_dist.log_prob(act_idx))
                entropies.append(act_dist.entropy())
                acts.append(self.act_choices[act_idx.item()])

                # Next input: embedding of the choices we just made
                # Use op_idx as token (offset by 1 for start token)
                inputs = self.embedding(op_idx + 1)

            architectures.append({
                'operations': ops,
                'activations': acts
            })
            log_probs_list.append(torch.stack(log_probs).sum())
            entropies_list.append(torch.stack(entropies).sum())

        return architectures, torch.stack(log_probs_list), torch.stack(entropies_list)


# ============================================================
# ENAS Training
# ============================================================

def train_supernet(supernet, train_loader, val_loader, controller, n_epochs=10, device='cpu'):
    """
    Train the supernet by sampling architectures from the controller.
    """
    supernet.to(device)
    supernet.train()

    optimizer = optim.Adam(supernet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*60}")
    print("TRAINING SUPERNET")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            data, target = data.to(device), target.to(device)

            # Sample an architecture
            archs, _, _ = controller.sample(batch_size=1)
            arch = archs[0]

            # Forward pass with sampled architecture
            optimizer.zero_grad()
            output = supernet(data, arch)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        train_acc = correct / total

        # Validation
        val_acc = evaluate_supernet(supernet, val_loader, controller, device, n_samples=5)

        print(f"  Epoch {epoch+1}/{n_epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    return supernet


def evaluate_supernet(supernet, loader, controller, device, n_samples=10):
    """
    Evaluate supernet by averaging over multiple sampled architectures.
    """
    supernet.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # Average predictions over multiple architectures
            outputs = []
            for _ in range(n_samples):
                archs, _, _ = controller.sample(batch_size=1)
                output = supernet(data, archs[0])
                outputs.append(output)

            avg_output = torch.stack(outputs).mean(dim=0)
            pred = avg_output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return correct / total


def train_controller(controller, supernet, val_loader, n_episodes=100, baseline_decay=0.95, device='cpu'):
    """
    Train the controller using REINFORCE.
    """
    controller.to(device)
    supernet.eval()

    optimizer = optim.Adam(controller.parameters(), lr=0.001)
    baseline = None

    print(f"\n{'='*60}")
    print("TRAINING CONTROLLER (REINFORCE)")
    print(f"{'='*60}")

    history = []

    for episode in range(n_episodes):
        # Sample architectures
        archs, log_probs, entropies = controller.sample(batch_size=1)
        arch = archs[0]
        log_prob = log_probs[0]
        entropy = entropies[0]

        # Evaluate architecture on validation set
        reward = evaluate_architecture(supernet, arch, val_loader, device)

        # Update baseline (exponential moving average)
        if baseline is None:
            baseline = reward
        else:
            baseline = baseline_decay * baseline + (1 - baseline_decay) * reward

        # REINFORCE loss
        advantage = reward - baseline
        loss = -log_prob * advantage - 0.01 * entropy  # Entropy regularization

        # Update controller
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append({
            'episode': episode + 1,
            'reward': reward,
            'baseline': baseline,
            'architecture': arch
        })

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1}/{n_episodes}: Reward={reward:.4f}, Baseline={baseline:.4f}")

    return controller, history


def evaluate_architecture(supernet, architecture, loader, device):
    """
    Evaluate a single architecture on the validation set.
    """
    supernet.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = supernet(data, architecture)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return correct / total


# ============================================================
# Main Experiment
# ============================================================

def main():
    print(f"{'='*60}")
    print("ENAS: Efficient Neural Architecture Search")
    print(f"{'='*60}")

    # Config
    dataset = 'mnist'
    device = 'cpu'
    n_layers = 3
    hidden_size = 128

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    # Split train into train/val
    train_size = 50000
    val_size = 10000
    train_subset = Subset(train_dataset, range(train_size))
    val_subset = Subset(train_dataset, range(train_size, train_size + val_size))

    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"\n{'='*60}")
    print(f"ENAS on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Device: {device}")
    print(f"  Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")

    # Initialize supernet and controller
    input_size = 28 * 28
    output_size = 10

    supernet = Supernet(input_size, hidden_size, output_size, n_layers=n_layers)
    controller = Controller(n_layers=n_layers)

    # Phase 1: Train supernet
    start_time = time.time()
    supernet = train_supernet(supernet, train_loader, val_loader, controller, n_epochs=20, device=device)
    supernet_time = time.time() - start_time

    # Phase 2: Train controller to search
    start_time = time.time()
    controller, search_history = train_controller(controller, supernet, val_loader, n_episodes=100, device=device)
    search_time = time.time() - start_time

    # Find best architecture from search
    best_episode = max(search_history, key=lambda x: x['reward'])
    best_arch = best_episode['architecture']

    print(f"\n{'='*60}")
    print("BEST ARCHITECTURE FOUND")
    print(f"{'='*60}")
    print(f"  Operations: {best_arch['operations']}")
    print(f"  Activations: {best_arch['activations']}")
    print(f"  Val Accuracy: {best_episode['reward']:.4f}")

    # Evaluate on test set
    test_acc = evaluate_architecture(supernet, best_arch, test_loader, device)
    print(f"  Test Accuracy: {test_acc:.4f}")

    print(f"\n  Supernet training time: {supernet_time:.1f}s ({supernet_time/60:.1f} min)")
    print(f"  Controller search time: {search_time:.1f}s ({search_time/60:.1f} min)")
    print(f"  Total time: {(supernet_time + search_time)/60:.1f} min")

    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    results = {
        'best_architecture': best_arch,
        'val_accuracy': float(best_episode['reward']),
        'test_accuracy': float(test_acc),
        'supernet_time_sec': float(supernet_time),
        'search_time_sec': float(search_time),
        'total_time_sec': float(supernet_time + search_time),
        'search_history': [
            {
                'episode': h['episode'],
                'reward': float(h['reward']),
                'baseline': float(h['baseline']),
                'architecture': h['architecture']
            }
            for h in search_history
        ]
    }

    output_file = results_dir / 'enas_search.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    # Plot search progress
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Reward over episodes
    episodes = [h['episode'] for h in search_history]
    rewards = [h['reward'] for h in search_history]
    baselines = [h['baseline'] for h in search_history]

    ax1.plot(episodes, rewards, 'o-', alpha=0.6, label='Reward', markersize=3)
    ax1.plot(episodes, baselines, 'r-', label='Baseline (EMA)', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('ENAS Controller Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Best reward found so far
    best_so_far = []
    current_best = 0
    for r in rewards:
        current_best = max(current_best, r)
        best_so_far.append(current_best)

    ax2.plot(episodes, best_so_far, 'g-', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Best Validation Accuracy')
    ax2.set_title('Best Architecture Found')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = plots_dir / 'enas_search.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_file}")

    print(f"\n{'='*60}")
    print("✓ ENAS Complete!")
    print(f"{'='*60}")
    print(f"\n  Best Architecture: {best_arch}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Total Time: {(supernet_time + search_time)/60:.1f} min")
    print(f"\n  Key Insight: Weight sharing allows searching 100 architectures")
    print(f"  in ~{(supernet_time + search_time)/60:.0f} minutes vs. hours/days without sharing!")


if __name__ == '__main__':
    main()
