"""
State vector construction (paper Table II).

Builds the 10-dimensional state s_t at each slot:
    0  N_est           — estimated STA count / N_max     ∈ [0,1]
    1  P_coll_smooth   — EWMA of collision outcomes      ∈ [0,1]
    2  η_prev          — previous-slot throughput        ∈ [0,1]
    3  H_coll_mean     — mean of last 10 collision flags ∈ [0,1]
    4  SINR_norm       — (SINR − 2) / 28                 ∈ [0,1]
    5  Q_len           — normalized queue length         ∈ [0,1]
    6  T_slot_mod      — (t mod 1000) / 1000             ∈ [0,1]
    7  CW_{t-1}_norm   — last CW / CW_max                ∈ [0,1]
    8  CW_{t-2}_norm
    9  CW_{t-3}_norm
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StateTracker:
    """Maintains the rolling statistics needed to build s_t.

    Attributes
    ----------
    n_max : int
        Maximum STA count (512 in the paper) used to normalize N_est.
    cw_max : int
        Maximum CW (1024) used to normalize the CW history.
    sinr_min, sinr_max : float
        SINR normalization range (Table II: (SINR − 2)/28).
    ewma_alpha : float
        EWMA smoothing factor for P_coll_smooth (index 1).
    """
    n_max: int = 512
    cw_max: int = 1024
    sinr_min: float = 2.0
    sinr_max: float = 30.0
    ewma_alpha: float = 0.1

    # Rolling state
    p_coll_smooth: float = 0.0
    eta_prev: float = 0.0
    coll_history: deque = field(default_factory=lambda: deque(maxlen=10))
    cw_history: deque = field(default_factory=lambda: deque([1024, 1024, 1024], maxlen=3))
    t: int = 0

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset all rolling state. Optionally seed SINR via rng."""
        self.p_coll_smooth = 0.0
        self.eta_prev = 0.0
        self.coll_history.clear()
        self.cw_history.clear()
        # Initialize CW history to CW_max so the first state is well-defined.
        for _ in range(3):
            self.cw_history.append(self.cw_max)
        self.t = 0

    def update(self, *,
               n_est: int,
               collision_flag: bool,
               eta: float,
               cw: int,
               sinr: float | None = None,
               q_len: float = 0.5) -> np.ndarray:
        """Advance the tracker by one slot and return the new s_t.

        Parameters
        ----------
        n_est : int
            Estimated STA count for this slot (the env's true N).
        collision_flag : bool
            Whether a collision occurred in the slot just observed.
        eta : float
            Throughput S observed in the slot just observed.
        cw : int
            CW chosen by the agent for the slot just observed.
        sinr : float | None
            Optional SINR sample (dB). If None, sampled uniformly in [2, 30].
        q_len : float
            Normalized queue length ∈ [0, 1].

        Returns
        -------
        np.ndarray[float32] of shape (10,).
        """
        coll_int = float(bool(collision_flag))
        # EWMA collision update
        self.p_coll_smooth = (
            (1.0 - self.ewma_alpha) * self.p_coll_smooth + self.ewma_alpha * coll_int
        )
        # Push collision flag to history (after update so H_coll_mean excludes
        # the just-observed flag, matching "mean of last 10 collision outcomes"
        # interpreted as the trailing 10-slot window ending at the previous slot)
        self.coll_history.append(coll_int)
        h_coll_mean = float(np.mean(self.coll_history)) if self.coll_history else 0.0

        # Throughput memory
        self.eta_prev = float(eta)

        # CW history (most-recent-last; index 7 = t-1, 8 = t-2, 9 = t-3)
        self.cw_history.append(int(cw))
        cw_arr = list(self.cw_history)
        # cw_arr[-1] = current, cw_arr[-2] = t-1, ...
        cw_t1 = cw_arr[-2] if len(cw_arr) >= 2 else self.cw_max
        cw_t2 = cw_arr[-3] if len(cw_arr) >= 3 else self.cw_max
        cw_t3 = cw_arr[-4] if len(cw_arr) >= 4 else self.cw_max

        # SINR normalization
        if sinr is None:
            sinr = 16.0  # mid-range default
        sinr_norm = (sinr - self.sinr_min) / (self.sinr_max - self.sinr_min)
        sinr_norm = float(np.clip(sinr_norm, 0.0, 1.0))

        # Slot-time phase
        t_slot_mod = (self.t % 1000) / 1000.0

        n_est_norm = float(np.clip(n_est / self.n_max, 0.0, 1.0))

        state = np.array([
            n_est_norm,
            self.p_coll_smooth,
            self.eta_prev,
            h_coll_mean,
            sinr_norm,
            float(np.clip(q_len, 0.0, 1.0)),
            t_slot_mod,
            cw_t1 / self.cw_max,
            cw_t2 / self.cw_max,
            cw_t3 / self.cw_max,
        ], dtype=np.float32)

        self.t += 1
        return state
