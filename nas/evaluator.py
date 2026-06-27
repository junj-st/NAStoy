"""
Architecture evaluator with training and caching.

Trains candidate architectures and caches results to avoid re-evaluation.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class Evaluator:
    """Evaluates architectures by training and validation."""

    def __init__(
        self,
        dataset: str = 'mnist',
        train_budget: int = 10,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        save_curves: bool = False
    ):
        """
        Args:
            dataset: Dataset name ('mnist' or 'cifar10')
            train_budget: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            cache_dir: Directory to cache evaluation results
            device: Device to train on ('cuda' or 'cpu')
            save_curves: Whether to save per-epoch learning curves
        """
        self.dataset = dataset
        self.train_budget = train_budget
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_curves = save_curves

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load datasets
        self.train_loader, self.val_loader = self._load_data()

    def _load_data(self):
        """Load train and validation datasets.

        Returns:
            train_loader, val_loader: PyTorch DataLoaders
        """
        if self.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            val_dataset = datasets.MNIST('./data', train=False, transform=transform)
        elif self.dataset == 'fashionmnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST stats
            ])
            train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
            val_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
        elif self.dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            val_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader

    def _get_cache_key(self, config: Dict) -> str:
        """Generate cache key from config and dataset.

        Args:
            config: Architecture configuration

        Returns:
            cache_key: SHA256 hash of dataset + sorted config
        """
        # Include dataset in cache key to prevent collisions
        cache_data = {
            'dataset': self.dataset,
            'config': config
        }
        config_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached evaluation result.

        Args:
            cache_key: Cache key for this architecture

        Returns:
            result: Cached evaluation result or None
        """
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key: str, result: Dict):
        """Save evaluation result to cache.

        Args:
            cache_key: Cache key for this architecture
            result: Evaluation result to save
        """
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)

    def evaluate(self, model: nn.Module, config: Dict) -> Dict:
        """Train and evaluate an architecture.

        Args:
            model: PyTorch model to evaluate
            config: Architecture configuration

        Returns:
            result: Dictionary with keys:
                - val_acc: Validation accuracy
                - train_time: Training time in seconds
                - n_params: Number of parameters
                - config: Architecture configuration
                - train_acc: Final training accuracy
                - best_epoch: Epoch with best validation accuracy
        """
        # Check cache first
        cache_key = self._get_cache_key(config)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Move model to device
        model = model.to(self.device)

        # Setup optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop with early stopping
        best_val_acc = 0.0
        best_epoch = 0
        patience = 3
        epochs_without_improvement = 0

        # Learning curves
        learning_curve = []

        start_time = time.time()

        for epoch in range(self.train_budget):
            # Train
            train_acc = self._train_epoch(model, optimizer, criterion)

            # Validate
            val_acc = self._validate(model, criterion)

            # Save curve data
            if self.save_curves:
                learning_curve.append({
                    'epoch': epoch,
                    'train_acc': train_acc,
                    'val_acc': val_acc
                })

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                break

        train_time = time.time() - start_time

        # Prepare result
        result = {
            'val_acc': best_val_acc,
            'train_acc': train_acc,
            'train_time': train_time,
            'n_params': self.count_parameters(model),
            'config': config,
            'best_epoch': best_epoch
        }

        if self.save_curves:
            result['learning_curve'] = learning_curve

        # Cache result
        self._save_to_cache(cache_key, result)

        return result

    def _train_epoch(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch.

        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            train_acc: Training accuracy
        """
        model.train()
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Flatten data for MLP
            data = data.view(data.size(0), -1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        return correct / total

    def _validate(self, model: nn.Module, criterion: nn.Module) -> float:
        """Validate model.

        Args:
            model: Model to validate
            criterion: Loss function

        Returns:
            val_acc: Validation accuracy
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Flatten data for MLP
                data = data.view(data.size(0), -1)

                output = model(data)

                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        return correct / total

    def count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters in model.

        Args:
            model: PyTorch model

        Returns:
            n_params: Number of trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
