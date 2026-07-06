"""
Full DRQN-MHA network (paper Sec IV-D, Table III).

Assembles GRU encoder → multi-head self-attention → shared dense → dueling
streams into a single end-to-end model. Total trainable parameters: 132,199.
"""
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
    """Deep Recurrent Q-Network with Multi-Head Attention and Dueling streams.

    Forward pass:
        x : (B, T, state_dim)
        → GRU        (B, T, 128)
        → MHA + LN   (B, T, 128)
        → last step  (B, 128)
        → shared     (B, 64)
        → value      (B, 1)
        → advantage  (B, 6)
        → Q(s,a)     (B, 6)

    Parameters
    ----------
    state_dim : int
        State feature dimension (10 per Table II).
    gru_hidden : int
        GRU hidden size (128 per Table III).
    num_heads : int
        Number of attention heads (4).
    d_k : int
        Per-head dimension (32); d_model = num_heads * d_k = 128.
    mha_bias : bool
        If False, MHA matches paper's 65,536 params exactly.
    shared_hidden : int
        Shared dense hidden size (64).
    dropout : float
        Dropout rate on shared dense (0.1).
    dueling_hidden : int
        Hidden size of value/advantage streams (32).
    n_actions : int
        Number of actions (6 CW values).
    """

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

        # 1) GRU encoder
        self.gru = GRUEncoder(
            input_dim=state_dim,
            hidden_dim=gru_hidden,
            num_layers=1,
        )
        # 2) Multi-head self-attention + residual + LayerNorm
        self.mha = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        # 3) Shared dense (128 → 64, ReLU, Dropout)
        self.shared = nn.Sequential(
            nn.Linear(gru_hidden, shared_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 4) Dueling streams
        self.value_stream = DuelingStream(
            in_dim=shared_hidden, hidden_dim=dueling_hidden, out_dim=1,
        )
        self.advantage_stream = DuelingStream(
            in_dim=shared_hidden, hidden_dim=dueling_hidden, out_dim=n_actions,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of state sequences.

        Parameters
        ----------
        x : torch.Tensor
            State sequence of shape (B, T, state_dim), T = sequence_length = 10.

        Returns
        -------
        torch.Tensor
            Q-values of shape (B, n_actions).
        """
        # 1) GRU encoder: (B, T, state_dim) → (B, T, gru_hidden)
        h = self.gru(x)
        # 2) Self-attention + residual + LayerNorm: (B, T, gru_hidden)
        h = self.mha(h)
        # 3) Take last-timestep representation
        last = h[:, -1, :]  # (B, gru_hidden)
        # 4) Shared dense
        z = self.shared(last)  # (B, shared_hidden)
        # 5) Dueling streams
        v = self.value_stream(z)          # (B, 1)
        a = self.advantage_stream(z)      # (B, n_actions)
        # 6) Combine
        q = dueling_combine(v, a)         # (B, n_actions)
        return q

    # ------------------------------------------------------------------
    # Convenience for inference (single sequence)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Greedy action prediction. Returns argmax_a Q(s, a) per batch element."""
        self.eval()
        q = self.forward(x)
        return q.argmax(dim=1)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_breakdown(model: DRQNMHA) -> dict[str, int]:
    """Return a per-component parameter count matching Table III.

    Returns
    -------
    dict[str, int]
        Keys: "GRU encoder", "Multi-Head Attention", "LayerNorm",
              "Shared dense", "Value stream", "Advantage stream", "Total".
    """
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
