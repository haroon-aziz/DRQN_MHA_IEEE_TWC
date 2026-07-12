
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    GRUEncoder,
    MultiHeadSelfAttention,
    DuelingStream,
    dueling_combine,
)


class DRQNMHA(nn.Module):

    def __init__(self,
                 state_dim: int = 10,
                 gru_hidden: int = 128,
                 num_heads: int = 4,
                 d_k: int = 32,
                 mha_bias: bool = False,
                 shared_hidden: int = 64,
                 dropout: float = 0.1,
                 dueling_hidden: int = 32,
                 n_actions: int = 6) -> None:
        super().__init__()
        d_model = num_heads * d_k
        assert d_model == gru_hidden, \
            f"d_model ({d_model}) must equal gru_hidden ({gru_hidden})"

        self.state_dim = state_dim
        self.n_actions = n_actions

       
        self.gru = GRUEncoder(
            input_dim=state_dim,
            hidden_dim=gru_hidden,
            num_layers=1,
        )
        self.mha = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
     
        self.shared = nn.Sequential(
            nn.Linear(gru_hidden, shared_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
      
        self.value_stream = DuelingStream(
            in_dim=shared_hidden, hidden_dim=dueling_hidden, out_dim=1,
        )
        self.advantage_stream = DuelingStream(
            in_dim=shared_hidden, hidden_dim=dueling_hidden, out_dim=n_actions,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
      
        h = self.gru(x)
        
        h = self.mha(h)
        last = h[:, -1, :]
        z = self.shared(last)  
        
        v = self.value_stream(z)         
        a = self.advantage_stream(z)      
    
        q = dueling_combine(v, a)        
        return q

 
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Greedy action prediction. Returns argmax_a Q(s, a) per batch element."""
        self.eval()
        q = self.forward(x)
        return q.argmax(dim=1)


def count_parameters(model: nn.Module) -> int:
 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_breakdown(model: DRQNMHA) -> dict[str, int]:
   
    def n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
    gru = n(model.gru)
    mha_attn = n(model.mha.attn)
    mha_norm = n(model.mha.norm)
    shared = n(model.shared)
    value = n(model.value_stream)
    adv = n(model.advantage_stream)
    return {
        "GRU encoder": gru,
        "Multi-Head Attention": mha_attn,
        "LayerNorm": mha_norm,
        "Shared dense": shared,
        "Value stream": value,
        "Advantage stream": adv,
        "Total": gru + mha_attn + mha_norm + shared + value + adv,
    }
