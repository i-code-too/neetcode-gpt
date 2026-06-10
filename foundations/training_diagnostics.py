import torch
import torch.nn as nn
from typing import List, Dict


class Solution:
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        activation_stats = []
        with torch.no_grad(): # no need to compute gradients
            for layer in model.children():
                x = layer(x) # output becomes new input after every layer
                if isinstance(layer, nn.Linear):
                    mean = x.mean().item() # .item() to access value, not tensor
                    std = x.std().item()
                    dead_fraction = (x <= 0).all(dim = 0).float().mean().item()
                    activation_stats.append({"mean": round(mean, 4), "std": round(std, 4), "dead_fraction": round(dead_fraction, 4)})
        return activation_stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        gradient_stats = []
        model.zero_grad()
        output = model(x) # run whole model on input
        loss = nn.MSELoss()(output, y) # loss of forward + backward pass
        loss.backward() # computing gradients
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad # storing gradient of weights (dL/dw)
                mean = grad.mean().item() # mean of gradients, not of outputs
                std = grad.std().item()
                norm = torch.norm(grad).item() # L2 norm of gradients
                gradient_stats.append({"mean": round(mean, 4), "std": round(std, 4), "norm": round(norm, 4)})
        return gradient_stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for stat in activation_stats:
            if stat["dead_fraction"] > 0.5:
                return 'dead_neurons'
            if stat["std"] < 0.1:
                return 'vanishing_gradients'
            if stat["std"] > 10.0:
                return 'exploding_gradients'

        for stat in gradient_stats:
            if stat["norm"] < 1e-5:
                return "vanishing_gradients"
            if stat["norm"] > 1000:
                return "exploding_gradients"

        else: return 'healthy'
