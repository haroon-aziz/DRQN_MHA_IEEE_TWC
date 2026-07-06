"""
DRQN-MHA network components (paper Sec IV-D, Table III).

Architecture overview (input = state sequence s_{t-9:t} of shape (B, 10, 10)):

    1. GRU Encoder (input_dim=10, hidden=128)         → (B, 10, 128)
    2. Multi-Head Self-Attention (4 heads, d_k=32)     → (B, 10, 128)
       + residual connection + LayerNorm
    3. Shared Dense (128 → 64, ReLU, Dropout 0.1)      → (B, 64)
       (applied to the last-timestep representation)
    4. Dueling streams:
         Value stream      (64 → 32 → 1)               → (B, 1)
         Advantage stream  (64 → 32 → 6)               → (B, 6)
       Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]         → (B, 6)

Parameter breakdown (matches paper Table III exactly when bias=False for MHA):
    GRU encoder           53,760
    Multi-Head Attention  65,536   (3*128*128 + 128*128, no bias)
    LayerNorm                256   (128 weight + 128 bias)
    Shared dense           8,256   (128*64 + 64)
    Value stream           2,113   (64*32+32 + 32*1+1)
    Advantage stream       2,278   (64*32+32 + 32*6+6)
    -------------------------------------
    Total                132,199
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    """Single-layer GRU encoder for temporal memory (paper Sec IV-D).

    Parameters
    ----------
    input_dim : int
        State feature dimension (10 per Table II).
    hidden_dim : int
        GRU hidden size (128 per Table III).
    num_layers : int
        Number of stacked GRU layers (1 in the paper).
    """

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
        """Encode a state sequence.

        Parameters
        ----------
        x : torch.Tensor
            State sequence of shape (B, T, input_dim).

        Returns
        -------
        torch.Tensor
            Hidden representation h of shape (B, T, hidden_dim).
        """
        out, _ = self.gru(x)  # (B, T, hidden)
        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with residual + LayerNorm (paper Sec IV-D).

    Implemented with `nn.MultiheadAttention` configured to match the paper's
    reported 65,536 parameters exactly. This requires `bias=False` on both
    the in-projection and out-projection:

        in_proj_weight  : 3 · d_model · d_model = 3 · 128 · 128 = 49,152
        out_proj.weight : d_model · d_model     =     128 · 128 = 16,384
        -------------------------------------------------------------
        Total           :                                          65,536

    Parameters
    ----------
    d_model : int
        Model dimension (128 = 4 heads × 32 d_k).
    num_heads : int
        Number of attention heads (4 per Table III).
    dropout : float
        Attention dropout (0.1 default).
    """

    def __init__(self, d_model: int = 128, num_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        # PyTorch's MHA: bias=False on both in_proj and out_proj gives exactly
        # 65,536 params (matches paper Table III).
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
        """Apply self-attention with residual connection and LayerNorm.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, T, d_model).

        Returns
        -------
        torch.Tensor
            Output of shape (B, T, d_model).
        """
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        # Residual + LayerNorm (post-norm variant)
        return self.norm(x + attn_out)


class DuelingStream(nn.Module):
    """Two-layer MLP for either the value or advantage stream.

    Both streams share the architecture 64 → 32 → out_dim with ReLU on the
    hidden layer. The output layer is linear.

    Parameters
    ----------
    in_dim : int
        Input dimension (64 from the shared dense).
    hidden_dim : int
        Hidden layer dimension (32 per Table III).
    out_dim : int
        Output dimension: 1 for value stream, 6 for advantage stream.
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 32,
                 out_dim: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the stream output.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, in_dim).

        Returns
        -------
        torch.Tensor
            Output of shape (B, out_dim).
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def dueling_combine(value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    """Combine value and advantage streams into Q-values (Wang et al. [10]).

        Q(s, a) = V(s) + A(s, a) − mean_a[A(s, a)]

    Parameters
    ----------
    value : torch.Tensor
        Value stream output of shape (B, 1).
    advantage : torch.Tensor
        Advantage stream output of shape (B, n_actions).

    Returns
    -------
    torch.Tensor
        Q-values of shape (B, n_actions).
    """
    return value + advantage - advantage.mean(dim=1, keepdim=True)
