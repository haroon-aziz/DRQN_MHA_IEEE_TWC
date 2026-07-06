"""
Double DQN agent with DRQN-MHA network (paper Sec IV-E).

Implements:
    * Double DQN target  (van Hasselt et al. [9])
        a* = argmax_a Q_online(s', a)
        y  = r + γ · (1 − done) · Q_target(s', a*)
    * ε-greedy exploration with linear decay 1.0 → 0.01 over 200,000 steps
    * Soft target-network update with τ = 0.005 every 1,000 steps
    * Huber loss (SmoothL1Loss with δ = 1.0)
    * Gradient clipping (max global norm = 10.0)
    * Adam optimizer, learning rate 3 × 10⁻⁴

The agent maintains a rolling state-sequence buffer of length L=10 to feed
the GRU encoder during both action selection and training.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.drqn_mha import DRQNMHA
from .replay_buffer import SequentialReplayBuffer


class DRQNAgent:
    """Double DQN agent with a DRQN-MHA function approximator.

    Parameters
    ----------
    state_dim : int
        State feature dimension (10).
    n_actions : int
        Number of discrete actions (6).
    seq_len : int
        GRU sequence length (10).
    device : torch.device
        Device for the online and target networks.
    lr : float
        Adam learning rate (3e-4 per Table IV).
    gamma : float
        Discount factor (0.99).
    epsilon_start, epsilon_end : float
        ε-greedy schedule endpoints (1.0 → 0.01).
    epsilon_decay_steps : int
        Linear decay duration (200,000 per Table IV).
    target_update_tau : float
        Polyak averaging coefficient for soft target updates (0.005).
    target_update_every : int
        Soft-update period in env steps (1,000).
    grad_clip_norm : float
        Maximum gradient global norm (10.0).
    buffer_capacity : int
        Replay buffer size (1,000,000).
    batch_size : int
        Training batch size (64).
    """

    def __init__(self,
                 state_dim: int = 10,
                 n_actions: int = 6,
                 seq_len: int = 10,
                 device: torch.device | str = "cpu",
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 200_000,
                 target_update_tau: float = 0.005,
                 target_update_every: int = 1_000,
                 grad_clip_norm: float = 10.0,
                 buffer_capacity: int = 1_000_000,
                 batch_size: int = 64,
                 huber_delta: float = 1.0) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_tau = target_update_tau
        self.target_update_every = target_update_every
        self.grad_clip_norm = grad_clip_norm
        self.batch_size = batch_size

        # Online + target networks
        self.online = DRQNMHA(
            state_dim=state_dim,
            n_actions=n_actions,
        ).to(self.device)
        self.target = DRQNMHA(
            state_dim=state_dim,
            n_actions=n_actions,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

        # Optimizer + loss
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(beta=huber_delta)

        # Replay buffer
        self.buffer = SequentialReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            seq_len=seq_len,
        )

        # Per-agent state: rolling state sequence for action selection
        self._seq: deque[np.ndarray] = deque(maxlen=seq_len)
        self._step_count = 0

    # ------------------------------------------------------------------
    # ε-greedy schedule
    # ------------------------------------------------------------------
    def epsilon(self, step: int | None = None) -> float:
        """Linear ε decay from `epsilon_start` to `epsilon_end`."""
        s = step if step is not None else self._step_count
        frac = min(1.0, max(0.0, s / max(1, self.epsilon_decay_steps)))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def reset_state_sequence(self, initial_state: np.ndarray) -> None:
        """Reset the rolling state sequence at the start of an episode.

        The sequence is initialized with `seq_len` copies of `initial_state`
        so that the first action decision has a full (L, D) input.
        """
        self._seq.clear()
        for _ in range(self.seq_len):
            self._seq.append(initial_state.copy())

    def observe_and_act(self, state: np.ndarray, eval_mode: bool = False,
                        step: int | None = None) -> int:
        """Append `state` to the rolling sequence and return an action.

        Parameters
        ----------
        state : np.ndarray
            New state s_t (shape (state_dim,)).
        eval_mode : bool
            If True, act greedily (no exploration).
        step : int, optional
            Global step count for ε scheduling. Defaults to internal counter.

        Returns
        -------
        int
            Action index in [0, n_actions).
        """
        self._seq.append(state.copy())
        seq_arr = np.stack(list(self._seq), axis=0)  # (L, D)
        seq_t = torch.from_numpy(seq_arr).float().unsqueeze(0).to(self.device)  # (1, L, D)

        if eval_mode:
            with torch.no_grad():
                q = self.online(seq_t)
            return int(q.argmax(dim=1).item())

        eps = self.epsilon(step)
        if np.random.random() < eps:
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            q = self.online(seq_t)
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    def store(self, state, action, reward, next_state, done) -> None:
        """Store one transition in the replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def learn(self) -> dict[str, float]:
        """Run one Double DQN gradient step.

        Returns
        -------
        dict[str, float]
            Diagnostics: {"loss", "q_mean", "q_max", "q_min", "td_mean"}.
        """
        if not self.buffer.can_sample(self.batch_size):
            return {}

        batch = self.buffer.sample(self.batch_size)
        states = torch.from_numpy(batch["states"]).to(self.device)      # (B, L, D)
        actions = torch.from_numpy(batch["actions"]).to(self.device)    # (B,)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)    # (B,)
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)        # (B,)

        # Q(s, a) for the action that was actually taken at the LAST step
        q_all = self.online(states)                # (B, n_actions)
        q_sa = q_all.gather(1, actions.view(-1, 1)).squeeze(1)  # (B,)

        # Double DQN: a* = argmax_a Q_online(s', a); target = Q_target(s', a*)
        with torch.no_grad():
            q_next_online = self.online(next_states)        # (B, n_actions)
            next_actions = q_next_online.argmax(dim=1)      # (B,)
            q_next_target = self.target(next_states)        # (B, n_actions)
            q_next = q_next_target.gather(1, next_actions.view(-1, 1)).squeeze(1)
            td_target = rewards + self.gamma * (1.0 - dones) * q_next

        loss = self.loss_fn(q_sa, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (global norm)
        torch.nn.utils.clip_grad_norm_(self.online.parameters(),
                                        self.grad_clip_norm)
        self.optimizer.step()

        with torch.no_grad():
            td = (q_sa - td_target).abs()
        return {
            "loss": float(loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "q_max": float(q_sa.max().item()),
            "q_min": float(q_sa.min().item()),
            "td_mean": float(td.mean().item()),
        }

    # ------------------------------------------------------------------
    # Target network soft update
    # ------------------------------------------------------------------
    def maybe_update_target(self, step: int) -> bool:
        """Soft-update target network params every `target_update_every` steps.

        τ = 0.005 (Table IV):  θ_target ← τ · θ_online + (1 − τ) · θ_target
        """
        if step > 0 and step % self.target_update_every == 0:
            with torch.no_grad():
                for tp, op in zip(self.target.parameters(),
                                  self.online.parameters()):
                    tp.mul_(1.0 - self.target_update_tau)
                    tp.add_(self.target_update_tau * op)
            return True
        return False

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return a serializable state dict (online + target + optimizer)."""
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load a previously saved state dict."""
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._step_count = state.get("step_count", 0)

    def increment_step(self) -> None:
        """Advance the internal step counter (called once per env step)."""
        self._step_count += 1
