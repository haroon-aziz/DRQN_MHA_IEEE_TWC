"""
Sequential experience replay buffer for DRQN training.

The paper (Sec IV-E) specifies: "The replay buffer stores 1,000,000 transitions
sampled as sequences of length 10 for GRU training."

This implementation stores raw (state, action, reward, next_state, done)
transitions and, on sampling, returns contiguous sequences of length L=10
ending at randomly chosen indices. This is the standard DRQN sampling
strategy (Hausknecht & Stone [7]).
"""
from __future__ import annotations

import numpy as np


class SequentialReplayBuffer:
    """FIFO buffer supporting length-L sequence sampling.

    Each transition is (s_t, a_t, r_t, s_{t+1}, done_t).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored (1,000,000 per Table IV).
    state_dim : int
        State feature dimension (10).
    seq_len : int
        Sequence length L sampled for GRU training (10).
    """

    def __init__(self, capacity: int, state_dim: int, seq_len: int = 10) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.seq_len = int(seq_len)

        # Pre-allocated ring buffers
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Append one transition to the buffer."""
        i = self._ptr
        self.states[i] = state
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        """True iff we can draw `batch_size` distinct length-L sequences."""
        # Need at least seq_len + 1 transitions so that the final state in the
        # sequence has a valid next_state.
        return self._size >= self.seq_len + 1 and self._size >= batch_size + self.seq_len

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of length-L sequences uniformly at random.

        Each sampled sequence is a contiguous slice of the buffer ending at a
        randomly chosen index `t`. The returned tensors are:

            states      : (B, L, D)
            actions     : (B,)         # action at the LAST step of the sequence
            rewards     : (B,)         # reward at the LAST step
            next_states : (B, L, D)    # sequence shifted by one (s_{t-L+2:t+1})
            dones       : (B,)

        The GRU processes the state sequence to produce a hidden
        representation at the last timestep; Q(s_t, a) is then read off and
        compared against r + γ · max_a' Q_target(s_{t+1}, a') (Double DQN).

        Parameters
        ----------
        batch_size : int
            Number of sequences to sample.

        Returns
        -------
        dict[str, np.ndarray]
        """
        if not self.can_sample(batch_size):
            raise RuntimeError(
                f"Cannot sample {batch_size} sequences of length {self.seq_len} "
                f"from buffer of size {self._size}"
            )
        # Valid end indices: [seq_len-1, _size-1]  (so the sequence has full length)
        # We need to also avoid sequences that cross a reset boundary; for this
        # simple FIFO buffer we assume episodes are back-to-back and a small
        # probability of cross-episode sequences is acceptable (standard DRQN).
        high = self._size - self.seq_len
        end_indices = np.random.randint(0, high + 1, size=batch_size) + self.seq_len - 1
        # end_indices[i] = the index of the LAST step in the i-th sequence

        states = np.zeros((batch_size, self.seq_len, self.state_dim), dtype=np.float32)
        next_states = np.zeros((batch_size, self.seq_len, self.state_dim), dtype=np.float32)
        actions = np.zeros(batch_size, dtype=np.int64)
        rewards = np.zeros(batch_size, dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)

        for i, t in enumerate(end_indices):
            start = t - self.seq_len + 1
            states[i] = self.states[start:t + 1]
            next_states[i] = self.next_states[start:t + 1]
            actions[i] = self.actions[t]
            rewards[i] = self.rewards[t]
            dones[i] = self.dones[t]

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }
