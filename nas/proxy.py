"""
Zero-cost proxy estimators for architecture performance.

Implements fast proxy metrics that correlate with architecture accuracy
without full training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict


def grad_norm(model: nn.Module, data_loader: DataLoader, device: str = 'cpu') -> float:
    """Compute gradient norm proxy.

    Measures the L2 norm of gradients on a single minibatch.
    Hypothesis: Architectures with higher gradient norms learn faster.

    Args:
        model: PyTorch model
        data_loader: DataLoader for getting a batch
        device: Device to compute on

    Returns:
        score: Gradient norm (higher is better)
    """
    model = model.to(device)
    model.train()

    # Get a single batch
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)

    # Flatten data for MLP
    data = data.view(data.size(0), -1)

    # Forward pass
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Compute L2 norm of all gradients
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm


def synflow(model: nn.Module, input_shape: tuple, device: str = 'cpu') -> float:
    """Compute SynFlow proxy score.

    Uses synthetic gradients to measure parameter importance.
    Based on: "Pruning neural networks without any data" (Tanaka et al., 2020)

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, *dims) or just input_size as int
        device: Device to compute on

    Returns:
        score: SynFlow score (higher is better)
    """
    model = model.to(device)
    model.train()

    # Handle input_shape as int or tuple
    if isinstance(input_shape, int):
        input_size = input_shape
        synthetic_input = torch.ones(1, input_size).to(device)
    else:
        synthetic_input = torch.ones(input_shape).to(device)
        # Flatten for MLP
        synthetic_input = synthetic_input.view(synthetic_input.size(0), -1)

    # Forward pass
    output = model(synthetic_input)

    # Compute sum of outputs as the loss
    # (SynFlow uses this to propagate gradients uniformly)
    loss = output.sum()

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Compute SynFlow score: sum of |param| * |grad|
    synflow_score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score = (p.data.abs() * p.grad.data.abs()).sum().item()
            synflow_score += score

    return synflow_score


def compute_proxies(model: nn.Module, data_loader: DataLoader, input_shape: tuple, device: str = 'cpu') -> Dict[str, float]:
    """Compute all proxy scores for a model.

    Args:
        model: PyTorch model
        data_loader: DataLoader for gradient norm
        input_shape: Input shape for SynFlow (can be int or tuple)
        device: Device to compute on

    Returns:
        scores: Dictionary of proxy scores
    """
    gn = grad_norm(model, data_loader, device)
    sf = synflow(model, input_shape, device)
    return {
        'grad_norm': gn,
        'synflow': sf
    }
