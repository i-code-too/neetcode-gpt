import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2/(fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std # creating random weight matrix and multiply with std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2/fan_in)
        weights = torch.randn(fan_out, fan_in) * std # creating random weight matrix and multiply with std
        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        weights = []
        fan_in = input_dim
        fan_out = hidden_dim
        activations_std = []

        for i in range(num_layers):
            if init_type == "xavier":
                std = math.sqrt(2/(fan_in + fan_out))
                weight = torch.randn(fan_out, fan_in) * std
            elif init_type == "kaiming":
                std = math.sqrt(2/fan_in)
                weight = torch.randn(fan_out, fan_in) * std
            elif init_type == "random":
                weight = torch.randn(fan_out, fan_in)
            weights.append(weight)

        x = torch.randn(1, input_dim)

        for w in weights: 
            z = x @ w.T # @ for matrix multiplication
            y_hat = torch.relu(z)
            std = y_hat.std().item() # .item() to get value; not tensor
            activations_std.append(round(std, 2))
            x = y_hat
            fan_in = hidden_dim

        return activations_std
