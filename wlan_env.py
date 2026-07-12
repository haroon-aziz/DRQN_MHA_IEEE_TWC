
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .markov_chain import PhyParams
from .precompute_cache import load_or_build_cache, build_lookup_table
from utils.state import StateTracker


# Action space (paper Sec IV-B)
DEFAULT_ACTION_SPACE = (32, 64, 128, 256, 512, 1024)


@dataclass
class RewardWeights:
    """Multi-objective reward weights (paper Table IX ★, paper-exact)."""
    alpha: float = 0.70   
    beta: float = 0.10    
    gamma: float = 0.15   
    delta: float = 0.10   
    epsilon: float = 0.05 


class WlanEnv:
  

    metadata = {"render_modes": []}

    def __init__(self,
                 phy: PhyParams,
                 cache_dir: str,
                 reward_weights: RewardWeights | None = None,
                 action_space: tuple[int, ...] = DEFAULT_ACTION_SPACE,
                 default_ru: int = 6,
                 n_drift: int = 2,
                 hol_normalize_slots: int = 500,
                 ewma_alpha: float = 0.1,
                 seed: int | None = None) -> None:

        self.phy = phy
        self.action_space = tuple(action_space)
        self.n_actions = len(self.action_space)
        self.default_ru = default_ru
        self.n_drift = n_drift
        self.hol_normalize_slots = hol_normalize_slots
        self.reward_weights = reward_weights or RewardWeights()
        self.rng = np.random.default_rng(seed)

        # Build / load cache and lookup table
        cache = load_or_build_cache(cache_dir, phy)
        self.lookup = build_lookup_table(cache)  # shape (512, 3, 6, 4)

        # Map r and CW values to indices
        self._r_idx = {3: 0, 6: 1, 9: 2}
        self._cw_idx = {cw: i for i, cw in enumerate(self.action_space)}

        # State tracker
        self.tracker = StateTracker(
            n_max=512,
            cw_max=phy.cw_max,
            sinr_min=2.0,
            sinr_max=30.0,
            ewma_alpha=ewma_alpha,
        )

        # Runtime state
        self.N: int = 0
        self.r: int = default_ru
        self.current_cw: int = self.action_space[0]
        self.prev_S: float = 0.0
        self.t: int = 0


    def _r_to_idx(self, r: int) -> int:
        if r not in self._r_idx:
            raise ValueError(f"r must be one of {list(self._r_idx)}, got {r}")
        return self._r_idx[r]

    def _cw_to_idx(self, cw: int) -> int:
        if cw not in self._cw_idx:
            raise ValueError(f"CW must be one of {self.action_space}, got {cw}")
        return self._cw_idx[cw]

    def _metrics_for(self, n: int, r: int, cw: int) -> tuple[float, float, float, float]:
        """O(1) cache lookup of (τ, p_coll, S, D_hol)."""
        n = int(np.clip(n, 1, 512))
        r_idx = self._r_to_idx(r)
        cw_idx = self._cw_to_idx(cw)
        row = self.lookup[n - 1, r_idx, cw_idx]  # (4,)
        return float(row[0]), float(row[1]), float(row[2]), float(row[3])

    
    def reset(self, n_init: int | None = None,
              r: int | None = None) -> np.ndarray:
        """Reset the environment for a new episode.

        Parameters
        ----------
        n_init : int, optional
            Initial STA count. If None, uniformly sampled in [1, 512].
        r : int, optional
            Number of RUs. Defaults to `self.default_ru`.

        Returns
        -------
        np.ndarray[float32] of shape (10,)
            Initial state s_0.
        """
        if n_init is None:
            n_init = int(self.rng.integers(1, 513))
        self.N = int(np.clip(n_init, 1, 512))
        self.r = int(r) if r is not None else self.default_ru
        self.current_cw = self.action_space[0]
        self.prev_S = 0.0
        self.t = 0
        self.tracker.reset(self.rng)

        # Build first state by observing the (initial-CW, initial-N) pair
        tau, p_coll, S, D_hol = self._metrics_for(self.N, self.r, self.current_cw)
        collision = bool(self.rng.random() < p_coll)
        sinr = float(self.rng.uniform(2.0, 30.0))
        state = self.tracker.update(
            n_est=self.N, collision_flag=collision, eta=S,
            cw=self.current_cw, sinr=sinr, q_len=float(self.rng.uniform(0.3, 0.9)),
        )
        self.prev_S = S
        return state

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply one environment step.

        Parameters
        ----------
        action : int
            Index into `self.action_space` (0..5).

        Returns
        -------
        state : np.ndarray[float32] (10,)
            Next state s_{t+1}.
        reward : float
            Multi-objective reward R_t (Sec IV-C).
        done : bool
            Always False (continuing task). The training loop caps episode length.
        info : dict
            Auxiliary diagnostics: {N, r, CW, tau, p_coll, S, D_hol, collision}.
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")

        cw_new = self.action_space[action]
        cw_old = self.current_cw

        # 1) Lookup metrics for current (N, r, CW_new)
        tau, p_coll, S, D_hol = self._metrics_for(self.N, self.r, cw_new)

        # 2) Stochastic collision outcome ~ Bernoulli(p_coll)
        collision = bool(self.rng.random() < p_coll)

        # 3) Multi-objective reward (Sec IV-C)
        delta_S = S - self.prev_S
        D_norm = min(D_hol / self.hol_normalize_slots, 1.0)
        fairness = 1.0 - p_coll
        delta_cw_norm = abs(cw_new - cw_old) / self.phy.cw_max

        w = self.reward_weights
        reward = (
            w.alpha * delta_S
            - w.beta * D_norm
            - w.gamma * float(collision)
            + w.delta * fairness
            - w.epsilon * delta_cw_norm
        )

        # 4) SINR sample
        sinr = float(self.rng.uniform(2.0, 30.0))
        q_len = float(self.rng.uniform(0.3, 0.9))

        # 5) Update state tracker (uses N before drift so the agent observes
        #    the slot it just acted on)
        state = self.tracker.update(
            n_est=self.N, collision_flag=collision, eta=S,
            cw=cw_new, sinr=sinr, q_len=q_len,
        )

        # 6) N drifts ±2 STAs (paper Sec V-A)
        drift = int(self.rng.integers(-self.n_drift, self.n_drift + 1))
        self.N = int(np.clip(self.N + drift, 1, 512))

        # Bookkeeping
        self.current_cw = cw_new
        self.prev_S = S
        self.t += 1

        info = {
            "N": self.N,
            "r": self.r,
            "CW": cw_new,
            "tau": tau,
            "p_coll": p_coll,
            "S": S,
            "D_hol": D_hol,
            "collision": collision,
            "delta_S": delta_S,
        }
        return state, float(reward), False, info


    def set_n(self, n: int) -> None:
            # ------------------------------------------------------------------
    # Convenience for evaluation
    # ------------------------------------------------------------------
        self.N = int(np.clip(n, 1, 512))

    def set_r(self, r: int) -> None:
        """Force the current number of RUs."""
        if r not in self._r_idx:
            raise ValueError(f"r must be one of {list(self._r_idx)}")
        self.r = int(r)
