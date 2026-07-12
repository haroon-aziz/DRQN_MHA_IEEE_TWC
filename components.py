
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUEncoder(nn.Module):


    def __init__(self, input_dim: int = 10, hidden_dim: int = 128,
                 num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out, _ = self.gru(x)  # (B, T, hidden)
        return out


clas

    def __init__(self, d_model: int = 128, num_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=False,           # no in_proj_bias, no out_proj.bias
            kdim=d_model,
            vdim=d_model,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        # Residual + LayerNorm (post-norm variant)
        return self.norm(x + attn_out)


class DuelingStream(nn.Module):
   
    def __init__(self, in_dim: int = 64, hidden_dim: int = 32,
                 out_dim: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def dueling_combine(value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    
    return value + advantage - advantage.mean(dim=1, keepdim=True)
