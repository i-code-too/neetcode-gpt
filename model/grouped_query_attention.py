import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        # head_dim = size of one word's embedding divided by no. of heads
        # each head receives one head_dim-dimensional slice of an original token/word embedding
        # each query head computes its own attention using that slice
        # groups of query heads share same head_dim-dimensional key and value heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        B, T, D = x.shape
        # B = number of sentences, T = number of tokens, D = model_dim (embedding length of each token)
        # reshape Q from (B, T, D) to (B, num_heads, T, head_dim)
        # reshape K and V from (B, T, D) to (B, num_kv_heads, T, head_dim)
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # repeat_kv_heads = number of times each KV head will be repeated on a group of query_heads
        repeat_kv_heads = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat_kv_heads, dim=1)
        V = V.repeat_interleave(repeat_kv_heads, dim=1)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device = x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = torch.softmax(scores, dim=-1) @ V 
        # scores are of (B, num_heads, T, head_dim) dimensions
        # but we need scores to be of (B, T, model_dim) dimensions 
        # convert scores to (B, T, head_dim * num_heads)
        scores = scores.transpose(1, 2).contiguous().view(B, T, -1) 
        output = self.output_proj(scores)
        return torch.round(output, decimals=4)
