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
    # TODO: Implement gradient norm proxy
    # 1. Get a single batch
    # 2. Forward pass + compute loss
    # 3. Backward pass
    # 4. Compute L2 norm of all gradients
    raise NotImplementedError


def synflow(model: nn.Module, input_shape: tuple, device: str = 'cpu') -> float:
    """Compute SynFlow proxy score.

    Uses synthetic gradients to measure parameter importance.
    Based on: "Pruning neural networks without any data" (Tanaka et al., 2020)

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, *dims)
        device: Device to compute on

    Returns:
        score: SynFlow score (higher is better)
    """
    # TODO: Implement SynFlow proxy
    # 1. Create synthetic input (all ones)
    # 2. Linearize network (convert all activations to linear)
    # 3. Forward pass with synthetic input
    # 4. Backward pass with output sum as loss
    # 5. Compute sum of |param * grad|
    raise NotImplementedError


def compute_proxies(model: nn.Module, data_loader: DataLoader, input_shape: tuple, device: str = 'cpu') -> Dict[str, float]:
    """Compute all proxy scores for a model.

    Args:
        model: PyTorch model
        data_loader: DataLoader for gradient norm
        input_shape: Input shape for SynFlow
        device: Device to compute on

    Returns:
        scores: Dictionary of proxy scores
    """
    return {
        'grad_norm': grad_norm(model, data_loader, device),
        'synflow': synflow(model, input_shape, device)
    }
