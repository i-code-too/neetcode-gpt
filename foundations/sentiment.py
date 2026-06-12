import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.fc1 = nn.Embedding(vocabulary_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        # Return a B, 1 tensor and round to 4 decimal places
        embeddings = self.fc1(x) # (B, T, 16)
        embeddings = embeddings.mean(dim = 1) # (B, 16) by averaging across the dimension T
        lin_embeddings = self.fc2(embeddings) 
        output = self.sigmoid(lin_embeddings)
        return torch.round(output, decimals = 4)
