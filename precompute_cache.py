"""
Pre-compute cache of all (N, r, CW) combinations.

The paper (Sec V-A) states: "computing S, P_coll, and D_hol via O(1) lookup
from a pre-computed cache of 9,216 unique (N, r, CW) combinations."

Grid:
    N ∈ [1, 512]    → 512 values
    r ∈ {3, 6, 9}   →   3 values
    CW ∈ {32, 64, 128, 256, 512, 1024} → 6 values
    Total = 512 × 3 × 6 = 9,216 entries ✓
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

from .markov_chain import PhyParams, compute_metrics


CACHE_VERSION = 1


def build_cache(phy: PhyParams) -> dict:
    """Build the 9,216-entry (N, r, CW) lookup cache.

    Parameters
    ----------
    phy : PhyParams
        PHY/MAC parameters from Table I.

    Returns
    -------
    dict with keys:
        - "N"     : np.ndarray[int32]   shape (9216,)
        - "r"     : np.ndarray[int32]   shape (9216,)
        - "CW"    : np.ndarray[int32]   shape (9216,)
        - "tau"   : np.ndarray[float32] shape (9216,)
        - "pcoll" : np.ndarray[float32] shape (9216,)
        - "S"     : np.ndarray[float32] shape (9216,)
        - "Dhol"  : np.ndarray[float32] shape (9216,)
        - "version": int
    """
    n_values = np.arange(1, 513, dtype=np.int32)        # 1..512
    r_values = np.array([3, 6, 9], dtype=np.int32)
    cw_values = np.array([32, 64, 128, 256, 512, 1024], dtype=np.int32)

    total = len(n_values) * len(r_values) * len(cw_values)
    assert total == 9216, f"expected 9216 combinations, got {total}"

    N_arr = np.empty(total, dtype=np.int32)
    R_arr = np.empty(total, dtype=np.int32)
    CW_arr = np.empty(total, dtype=np.int32)
    tau_arr = np.empty(total, dtype=np.float32)
    pcoll_arr = np.empty(total, dtype=np.float32)
    S_arr = np.empty(total, dtype=np.float32)
    Dhol_arr = np.empty(total, dtype=np.float32)

    idx = 0
    for n in n_values:
        for r in r_values:
            for cw in cw_values:
                tau, p_coll, S, D = compute_metrics(int(cw), int(n), int(r), phy)
                N_arr[idx] = n
                R_arr[idx] = r
                CW_arr[idx] = cw
                tau_arr[idx] = tau
                pcoll_arr[idx] = p_coll
                S_arr[idx] = S
                Dhol_arr[idx] = D
                idx += 1

    return {
        "N": N_arr, "r": R_arr, "CW": CW_arr,
        "tau": tau_arr, "pcoll": pcoll_arr,
        "S": S_arr, "Dhol": Dhol_arr,
        "version": CACHE_VERSION,
    }


def cache_path(cache_dir: str | os.PathLike) -> Path:
    """Return the on-disk cache path."""
    return Path(cache_dir) / "markov_cache.pkl"


def load_or_build_cache(cache_dir: str | os.PathLike,
                        phy: PhyParams,
                        force_rebuild: bool = False) -> dict:
    """Load the cache from disk, or build and persist it.

    Parameters
    ----------
    cache_dir : str | PathLike
        Directory in which to store `markov_cache.pkl`.
    phy : PhyParams
        PHY parameters; if the cached version differs, the cache is rebuilt.
    force_rebuild : bool
        If True, always rebuild.

    Returns
    -------
    dict
        The cache (see `build_cache` for schema).
    """
    p = cache_path(cache_dir)
    if not force_rebuild and p.exists():
        try:
            with open(p, "rb") as f:
                cache = pickle.load(f)
            if cache.get("version") == CACHE_VERSION:
                return cache
        except Exception:
            pass
    cache = build_cache(phy)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def build_lookup_table(cache: dict) -> np.ndarray:
    """Return a 3D lookup array indexed by [N-1, r_idx, cw_idx].

    Returns
    -------
    np.ndarray[float32] of shape (512, 3, 6, 4)
        Last axis = [tau, p_coll, S, D_hol].
        r_idx: 0→r=3, 1→r=6, 2→r=9.
        cw_idx: 0→CW=32, 1→64, 2→128, 3→256, 4→512, 5→1024.
    """
    table = np.zeros((512, 3, 6, 4), dtype=np.float32)
    r_map = {3: 0, 6: 1, 9: 2}
    cw_map = {32: 0, 64: 1, 128: 2, 256: 3, 512: 4, 1024: 5}
    N = cache["N"]; r = cache["r"]; CW = cache["CW"]
    tau = cache["tau"]; pcoll = cache["pcoll"]
    S = cache["S"]; Dhol = cache["Dhol"]
    for i in range(len(N)):
        n_idx = int(N[i]) - 1
        r_idx = r_map[int(r[i])]
        cw_idx = cw_map[int(CW[i])]
        table[n_idx, r_idx, cw_idx, 0] = tau[i]
        table[n_idx, r_idx, cw_idx, 1] = pcoll[i]
        table[n_idx, r_idx, cw_idx, 2] = S[i]
        table[n_idx, r_idx, cw_idx, 3] = Dhol[i]
    return table
