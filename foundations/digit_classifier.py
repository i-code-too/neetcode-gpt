import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.fc1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        torch.manual_seed(0)
        problem_input = self.fc1(images)
        relu_input = self.relu(problem_input)
        drop_relu = self.dropout(relu_input)
        lin_drop = self.fc2(drop_relu)
        output = self.sigmoid(lin_drop)

        return torch.round(output, decimals = 4)
