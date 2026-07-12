
from __future__ import annotations

import numpy as np

class SequentialReplayBuffer:

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
       
        if not self.can_sample(batch_size):
            raise RuntimeError(
                f"Cannot sample {batch_size} sequences of length {self.seq_len} "
                f"from buffer of size {self._size}"
            )
     
        high = self._size - self.seq_len
        end_indices = np.random.randint(0, high + 1, size=batch_size) + self.seq_len - 1
        

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
